#!/usr/bin/env python3
"""
LLM Flashcard Generator - Enhanced Version
A modern flashcard application with improved UI, better error handling,
and enhanced user experience.
"""

import sys
import json
import os
import random
import datetime
import re
import hashlib
import asyncio
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import logging
import sqlite3
import difflib
from contextlib import closing, contextmanager



# GUI Imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, 
    QGroupBox, QMessageBox, QFileDialog, QProgressDialog, QFrame,
    QScrollArea, QSplitter, QStatusBar, QMenuBar, QAction, QDialog,
    QDialogButtonBox, QSpinBox, QCheckBox, QSlider, QGridLayout,
    QListWidget, QListWidgetItem, QTextBrowser, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QPainter

# Document processing
PDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF's historical import name
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pymupdf as fitz  # Newer alias
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

    
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# LLM Integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flashcard_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
APP_NAME = "FlashCard Studio"
APP_VERSION = "2.0"
USE_MOCK = not OLLAMA_AVAILABLE  # Automatically enable mock if Ollama not available

class AppConfig:
    """Centralized configuration management"""
    def __init__(self):
        self.app_dir = Path.home() / ".flashcard_studio"
        self.app_dir.mkdir(exist_ok=True)
        
        self.cache_dir = self.app_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.history_file = self.app_dir / "history.json"
        self.settings = QSettings("FlashCardStudio", "Settings")
        self.db_path = self.app_dir / "flashcards.db"
        self.db = None  # set after Database is defined


        
    def get_models(self) -> List[str]:
        """Get available Ollama models"""
        default_models = [
            "mistral:7b-instruct",
            "llama3.1:8b", 
            "qwen2.5:7b",
            "gemma2:9b",
            "codellama:7b",
            "neural-chat:7b"
        ]
        
        if not OLLAMA_AVAILABLE:
            return ["mock-model (Ollama not available)"]
            
        try:
            models = ollama.list()
            available = [model['name'] for model in models['models']]
            return available if available else default_models
        except:
            return default_models

config = AppConfig()


def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing"""
    if not text:
        return ""
    
    # Remove extra whitespace but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove table of contents, headers, footers
    text = re.sub(r'table of contents.*?\n\n', '', text, flags=re.I|re.S)
    text = re.sub(r'©.*?\n', '', text, flags=re.I)
    text = re.sub(r'page \d+.*?\n', '', text, flags=re.I)
    
    return text.strip()

def chunk_document(text: str, max_words: int = 400) -> List[str]:
    """Improved document chunking with better structure preservation"""
    if not text:
        return []
    
    # First split by major headers
    sections = re.split(r'\n\s*#{1,3}\s+.+\n', text)
    
    final_chunks = []
    for section in sections:
        if not section.strip():
            continue
            
        words = section.split()
        if len(words) <= max_words:
            if len(words) > 50:  # Only include substantial chunks
                final_chunks.append(section.strip())
        else:
            # Split large sections by paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            current_chunk = []
            current_words = 0
            
            for para in paragraphs:
                para_words = len(para.split())
                if current_words + para_words > max_words and current_chunk:
                    final_chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_words = para_words
                else:
                    current_chunk.append(para)
                    current_words += para_words
            
            if current_chunk:
                final_chunks.append('\n\n'.join(current_chunk))
    
    return final_chunks

def normalize_for_hash(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^a-z0-9 %.,;:?!()\[\]{}<>/\\+-=_\'\"|@#*&$]', ' ', s)
    return s.strip()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


class DocumentReader:
    """Enhanced document reading with better error handling"""
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """Read document content based on file type"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        try:
            if ext == '.txt':
                return DocumentReader._read_txt(path)
            elif ext == '.pdf' and PDF_AVAILABLE:
                return DocumentReader._read_pdf(path)
            elif ext in ['.docx', '.doc'] and DOCX_AVAILABLE:
                return DocumentReader._read_docx(path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise
    
    @staticmethod
    def _read_txt(path: Path) -> str:
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")
    
    @staticmethod
    def _read_pdf(path: Path) -> str:
        with fitz.open(str(path)) as doc:
            chunks = []
            for page in doc:
                # "text" gives plain text; good default across PyMuPDF versions
                chunks.append(page.get_text("text"))
            return "\n".join(chunks)

    
    @staticmethod
    def _read_docx(path: Path) -> str:
        doc = DocxDocument(str(path))
        return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init()

    @contextmanager
    def conn(self):
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init(self):
        with self.conn() as con:
            cur = con.cursor()
            # Core tables
            cur.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS projects(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents(
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                path TEXT,
                title TEXT,
                doc_hash TEXT NOT NULL,
                mtime REAL,
                size INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(doc_hash),
                FOREIGN KEY(project_id) REFERENCES projects(id)
            );

            CREATE TABLE IF NOT EXISTS cards(
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                question TEXT NOT NULL,
                options_json TEXT NOT NULL,
                correct_index INTEGER NOT NULL,
                explanation TEXT,
                card_hash TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                due_at TEXT,      -- optional spaced repetition
                ease REAL,        -- optional
                interval_days REAL, -- optional
                reps INTEGER DEFAULT 0,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS sessions(
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                document_id INTEGER,
                mode TEXT,
                started_at TEXT,
                duration_sec INTEGER,
                total_questions INTEGER,
                correct_answers INTEGER,
                time_limit_sec INTEGER,
                FOREIGN KEY(project_id) REFERENCES projects(id),
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS session_cards(
                session_id INTEGER,
                card_id INTEGER,
                is_correct INTEGER,
                elapsed_ms INTEGER,
                PRIMARY KEY(session_id, card_id),
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(card_id) REFERENCES cards(id)
            );
            """)
            con.commit()
            # Optional FTS for fast search
            try:
                cur.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts
                USING fts5(question, explanation, content='cards', content_rowid='id');
                INSERT INTO cards_fts(rowid, question, explanation)
                SELECT id, question, COALESCE(explanation, '')
                FROM cards
                WHERE id NOT IN (SELECT rowid FROM cards_fts);
                """)
                con.commit()
                self.fts_enabled = True
            except sqlite3.OperationalError:
                # FTS5 not available
                self.fts_enabled = False

    # Project/Document
    def get_or_create_project(self, name: str, root_path: str = None) -> int:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id FROM projects WHERE name=?", (name,))
            row = cur.fetchone()
            if row: return row["id"]
            cur.execute("INSERT INTO projects(name, root_path) VALUES(?,?)", (name, root_path))
            return cur.lastrowid

    def get_or_create_document(self, project_id: int, path: str, title: str, doc_hash: str, mtime: float, size: int) -> int:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id FROM documents WHERE doc_hash=?", (doc_hash,))
            row = cur.fetchone()
            if row: return row["id"]
            cur.execute("""
                INSERT INTO documents(project_id, path, title, doc_hash, mtime, size)
                VALUES (?,?,?,?,?,?)
            """, (project_id, path, title, doc_hash, mtime, size))
            return cur.lastrowid

    # Cards
    def card_exists(self, card_hash: str) -> bool:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT 1 FROM cards WHERE card_hash=?", (card_hash,))
            return cur.fetchone() is not None

    def insert_cards(self, document_id: int, cards: List[Dict]) -> List[int]:
        """Inserts non-duplicate cards, returns DB ids for inserted ones."""
        inserted_ids = []
        with self.conn() as con:
            cur = con.cursor()
            for c in cards:
                options_json = json.dumps(c["options"], ensure_ascii=False)
                base_for_hash = normalize_for_hash(c["question"] + " || " + options_json)
                chash = sha256_text(base_for_hash)
                if self.card_exists(chash):
                    continue
                cur.execute("""
                    INSERT INTO cards(document_id, question, options_json, correct_index, explanation, card_hash)
                    VALUES (?,?,?,?,?,?)
                """, (document_id, c["question"], options_json, int(c["correct_index"]), c.get("explanation",""), chash))
                inserted_ids.append(cur.lastrowid)
            # keep FTS in sync
            if self.fts_enabled and inserted_ids:
                cur.execute(f"INSERT INTO cards_fts(rowid, question, explanation) "
                            f"SELECT id, question, COALESCE(explanation,'') FROM cards WHERE id IN ({','.join('?'*len(inserted_ids))})",
                            inserted_ids)
        return inserted_ids

    def fetch_cards_for_document(self, document_id: int) -> List[Dict]:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id, question, options_json, correct_index, explanation FROM cards WHERE document_id=?", (document_id,))
            rows = cur.fetchall()
            result = []
            for r in rows:
                result.append({
                    "db_id": r["id"],
                    "question": r["question"],
                    "options": json.loads(r["options_json"]),
                    "correct_index": int(r["correct_index"]),
                    "explanation": r["explanation"]
                })
            return result

    def search_cards(self, query: str) -> List[Dict]:
        q = query.strip()
        if not q:
            return []
        with self.conn() as con:
            cur = con.cursor()
            if self.fts_enabled:
                cur.execute("""SELECT c.id, c.question, c.options_json, c.correct_index, c.explanation
                               FROM cards c JOIN cards_fts f ON c.id=f.rowid
                               WHERE cards_fts MATCH ? ORDER BY rank""", (q,))
            else:
                like = f"%{q}%"
                cur.execute("""SELECT id, question, options_json, correct_index, explanation
                               FROM cards WHERE question LIKE ? OR explanation LIKE ?""", (like, like))
            rows = cur.fetchall()
            return [{
                "db_id": r["id"],
                "question": r["question"],
                "options": json.loads(r["options_json"]),
                "correct_index": int(r["correct_index"]),
                "explanation": r["explanation"]
            } for r in rows]

    # Sessions
    def record_session(self, project_id: int, document_id: int, mode: str, started_at: str,
                       duration_sec: int, total: int, correct: int, time_limit_sec: int,
                       per_card: List[Tuple[int, bool, int]]) -> int:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("""INSERT INTO sessions(project_id, document_id, mode, started_at, duration_sec,
                         total_questions, correct_answers, time_limit_sec)
                         VALUES (?,?,?,?,?,?,?,?)""",
                        (project_id, document_id, mode, started_at, duration_sec, total, correct, time_limit_sec))
            sid = cur.lastrowid
            if per_card:
                cur.executemany("""INSERT INTO session_cards(session_id, card_id, is_correct, elapsed_ms)
                                   VALUES (?,?,?,?)""", [(sid, cid, 1 if ok else 0, ms) for cid, ok, ms in per_card])
            return sid

    def recent_sessions(self, limit=10) -> List[sqlite3.Row]:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("""SELECT s.*, d.title as doc_title, p.name as project_name
                           FROM sessions s
                           LEFT JOIN documents d ON s.document_id=d.id
                           LEFT JOIN projects p ON s.project_id=p.id
                           ORDER BY s.started_at DESC LIMIT ?""", (limit,))
            return cur.fetchall()

# Ensure DB is initialized once Database is defined
if config.db is None:
    config.db = Database(config.db_path)



class FlashcardGenerator(QThread):
    """Enhanced flashcard generation with better async handling"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    status_update = pyqtSignal(str)

    def __init__(self, model: str, prompt: str, document_text: str, num_cards: int = 5):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.document_text = document_text
        self.num_cards = num_cards
        self.cache_dir = config.cache_dir
        self._stop_requested = False

    def stop(self):
        """Request to stop generation"""
        self._stop_requested = True

    def get_cache_key(self) -> str:
        clean = preprocess_text(self.document_text)
        doc_hash = sha256_text(normalize_for_hash(clean))
        prompt_hash = sha256_text(normalize_for_hash(self.prompt + self.model + str(self.num_cards)))
        return sha256_text(doc_hash + "::" + prompt_hash)


    def load_from_cache(self) -> Optional[List[Dict]]:
        """Try to load cached flashcards"""
        cache_file = self.cache_dir / f"{self.get_cache_key()}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except:
                return None
        return None

    def save_to_cache(self, flashcards: List[Dict]):
        """Save flashcards to cache"""
        cache_file = self.cache_dir / f"{self.get_cache_key()}.json"
        try:
            cache_file.write_text(json.dumps(flashcards, indent=2))
        except Exception as e:
            logger.warning(f"Could not save to cache: {e}")

    def generate_chunk_flashcards(self, chunk: str) -> List[Dict]:
        """Generate flashcards for a single chunk"""
        if self._stop_requested:
            return []
            
        try:
            prompt = f"""
            Create {min(self.num_cards, 8)} multiple-choice flashcards based on the following text.
            
            Requirements:
            1. Questions should be clear, specific, and test understanding
            2. Each question must have exactly 4 options (A, B, C, D)
            3. Only one option should be correct
            4. Incorrect options should be plausible but clearly wrong
            5. Focus on key concepts, facts, and important details
            6. Vary question types (factual, conceptual, application)
            
            {self.prompt}

            Text to analyze:
            {chunk[:2000]}
            
            Respond with valid JSON only:
            [
                {{
                    "question": "Your question here?",
                    "options": ["Correct answer", "Wrong option 1", "Wrong option 2", "Wrong option 3"],
                    "correct_index": 0,
                    "explanation": "Brief explanation of why the correct answer is right"
                }}
            ]
            """

            if USE_MOCK:
                # Enhanced mock data for testing
                mock_cards = []
                for i in range(min(self.num_cards, 3)):
                    mock_cards.append({
                        "question": f"Mock question {i+1} about the content?",
                        "options": [
                            f"Correct answer {i+1}",
                            f"Incorrect option A{i+1}",
                            f"Incorrect option B{i+1}",
                            f"Incorrect option C{i+1}"
                        ],
                        "correct_index": 0,
                        "explanation": f"This is the correct answer for question {i+1}"
                    })
                return mock_cards
            
            # Real Ollama generation
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.7, "num_predict": 2000}
            )
            
            response_text = response['message']['content'].strip()
            
            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                chunk_cards = json.loads(json_text)
                
                # Validate and process cards
                valid_cards = []
                for card in chunk_cards:
                    if self._validate_card(card):
                        # Shuffle options while tracking correct answer
                        correct_answer = card["options"][card["correct_index"]]
                        random.shuffle(card["options"])
                        card["correct_index"] = card["options"].index(correct_answer)
                        valid_cards.append(card)
                
                return valid_cards
            else:
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            return []

    def _validate_card(self, card: Dict) -> bool:
        """Validate flashcard structure"""
        required_keys = ["question", "options", "correct_index"]
        
        if not all(key in card for key in required_keys):
            return False
        
        if not isinstance(card["options"], list) or len(card["options"]) != 4:
            return False
        
        if not (0 <= card["correct_index"] < 4):
            return False
        
        if not card["question"].strip():
            return False
            
        return True

    def run(self):
        """Main generation thread"""
        try:
            # Check cache first
            self.status_update.emit("Checking cache...")
            cached_cards = self.load_from_cache()
            if cached_cards:
                logger.info("Using cached flashcards")
                self.finished.emit(cached_cards)
                return

            self.status_update.emit("Preprocessing document...")
            clean_text = preprocess_text(self.document_text)
            
            if len(clean_text.split()) < 50:
                raise ValueError("Document too short to generate meaningful flashcards")
            
            chunks = chunk_document(clean_text, max_words=500)
            if not chunks:
                raise ValueError("Could not extract meaningful content from document")
            
            total_chunks = min(len(chunks), 10)  # Limit chunks for performance
            chunks = chunks[:total_chunks]
            
            self.status_update.emit(f"Processing {total_chunks} sections...")
            
            all_flashcards = []
            
            for i, chunk in enumerate(chunks):
                if self._stop_requested:
                    break
                    
                self.status_update.emit(f"Generating flashcards for section {i+1}...")
                chunk_cards = self.generate_chunk_flashcards(chunk)
                all_flashcards.extend(chunk_cards)
                self.progress.emit(i + 1, total_chunks)
                
                # Small delay to prevent overwhelming the LLM
                if not USE_MOCK:
                    self.msleep(500)
            
            # Remove duplicates based on question similarity
            unique_cards = self._remove_duplicates(all_flashcards)
            
            if unique_cards:
                self.save_to_cache(unique_cards)
                logger.info(f"Generated {len(unique_cards)} unique flashcards")
                self.finished.emit(unique_cards)
            else:
                self.error.emit("No valid flashcards could be generated from this document")

        except Exception as e:
            logger.error(f"Generation error: {e}")
            self.error.emit(f"Generation failed: {str(e)}")

    def _remove_duplicates(self, flashcards: List[Dict]) -> List[Dict]:
        if not flashcards:
            return []
        unique = []
        seen_norm = []
        for card in flashcards:
            qn = normalize_for_hash(card["question"])
            dup = False
            for s in seen_norm:
                # fuzzy ratio prevents trivial reworded duplicates
                if difflib.SequenceMatcher(None, qn, s).ratio() >= 0.87:
                    dup = True
                    break
            if not dup:
                unique.append(card)
                seen_norm.append(qn)
        return unique


class ModernButton(QPushButton):
    """Custom styled button"""
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.update_style()
    
    def update_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #1565C0;
                }
                QPushButton:disabled {
                    background-color: #CCCCCC;
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #F5F5F5;
                    color: #333333;
                    border: 1px solid #DDDDDD;
                    border-radius: 6px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #EEEEEE;
                    border-color: #BBBBBB;
                }
                QPushButton:pressed {
                    background-color: #E0E0E0;
                }
            """)

class StyledGroupBox(QGroupBox):
    """Custom styled group box"""
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin: 10px 0;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #2196F3;
            }
        """)

class SettingsDialog(QDialog):
    """Settings configuration dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Number of flashcards
        cards_group = StyledGroupBox("Generation Settings")
        cards_layout = QGridLayout(cards_group)
        
        cards_layout.addWidget(QLabel("Flashcards per section:"), 0, 0)
        self.cards_spinbox = QSpinBox()
        self.cards_spinbox.setRange(1, 10)
        self.cards_spinbox.setValue(5)
        cards_layout.addWidget(self.cards_spinbox, 0, 1)
        
        cards_layout.addWidget(QLabel("Max document sections:"), 1, 0)
        self.sections_spinbox = QSpinBox()
        self.sections_spinbox.setRange(1, 20)
        self.sections_spinbox.setValue(10)
        cards_layout.addWidget(self.sections_spinbox, 1, 1)
        
        layout.addWidget(cards_group)
        
        # UI Settings
        ui_group = StyledGroupBox("Interface")
        ui_layout = QGridLayout(ui_group)
        
        self.dark_mode_check = QCheckBox("Dark Mode (restart required)")
        ui_layout.addWidget(self.dark_mode_check, 0, 0)
        
        layout.addWidget(ui_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
        self.debug_check = QCheckBox("Enable debug logging")
        ui_layout.addWidget(self.debug_check, 1, 0)


    def load_settings(self):
        self.cards_spinbox.setValue(config.settings.value("cards_per_section", 5, type=int))
        self.sections_spinbox.setValue(config.settings.value("max_sections", 10, type=int))
        self.dark_mode_check.setChecked(config.settings.value("dark_mode", False, type=bool))
    
    def accept(self):
        config.settings.setValue("cards_per_section", self.cards_spinbox.value())
        config.settings.setValue("max_sections", self.sections_spinbox.value())
        config.settings.setValue("dark_mode", self.dark_mode_check.isChecked())
        config.settings.setValue("debug_logging", self.debug_check.isChecked())
        logging.getLogger().setLevel(logging.DEBUG if self.debug_check.isChecked() else logging.INFO)
        super().accept()

class FlashcardWidget(QWidget):
    """Enhanced flashcard display widget"""
    
    answer_selected = pyqtSignal(int, bool)  # index, is_correct
    
    def __init__(self):
        super().__init__()
        self.flashcard = None
        self.answered = False
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Question area
        question_frame = QFrame()
        question_frame.setFrameStyle(QFrame.Box)
        question_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        question_layout = QVBoxLayout(question_frame)
        
        self.question_label = QTextEdit()
        self.question_label.setReadOnly(True)
        self.question_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        self.question_label.setMaximumHeight(100)
        self.question_label.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
            }
        """)
        question_layout.addWidget(self.question_label)
        layout.addWidget(question_frame)
        
        # Options area
        self.option_buttons = []
        options_layout = QGridLayout()
        
        for i in range(4):
            btn = QPushButton()
            btn.setMinimumHeight(50)
            btn.setFont(QFont("Segoe UI", 10))
            btn.clicked.connect(lambda _, idx=i: self.select_answer(idx))
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 10px 15px;
                    border: 2px solid #E0E0E0;
                    border-radius: 6px;
                    background-color: white;
                }
                QPushButton:hover {
                    border-color: #2196F3;
                    background-color: #F5F9FF;
                }
            """)
            
            row, col = i // 2, i % 2
            options_layout.addWidget(btn, row, col)
            self.option_buttons.append(btn)
        
        layout.addLayout(options_layout)
    
    def set_flashcard(self, flashcard: Dict):
        """Display a new flashcard"""
        self.flashcard = flashcard
        self.answered = False
        
        self.question_label.setText(flashcard["question"])
        
        for i, btn in enumerate(self.option_buttons):
            if i < len(flashcard["options"]):
                option_letter = chr(ord('A') + i)
                btn.setText(f"{option_letter}. {flashcard['options'][i]}")
                btn.setVisible(True)
                btn.setEnabled(True)
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 10px 15px;
                        border: 2px solid #E0E0E0;
                        border-radius: 6px;
                        background-color: white;
                    }
                    QPushButton:hover {
                        border-color: #2196F3;
                        background-color: #F5F9FF;
                    }
                """)
            else:
                btn.setVisible(False)
    
    def select_answer(self, index: int):
        """Handle answer selection"""
        if self.answered or not self.flashcard:
            return
        
        self.answered = True
        correct_index = self.flashcard["correct_index"]
        is_correct = index == correct_index
        
        # Update button styles
        for i, btn in enumerate(self.option_buttons):
            btn.setEnabled(False)
            if i == correct_index:
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 10px 15px;
                        border: 2px solid #4CAF50;
                        border-radius: 6px;
                        background-color: #E8F5E8;
                        color: #2E7D32;
                        font-weight: bold;
                    }
                """)
            elif i == index and not is_correct:
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 10px 15px;
                        border: 2px solid #F44336;
                        border-radius: 6px;
                        background-color: #FFEBEE;
                        color: #C62828;
                        font-weight: bold;
                    }
                """)
        
        self.answer_selected.emit(index, is_correct)

class FlashcardApp(QMainWindow):
    """Main application window with enhanced UI and features"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, 1200, 900)
        self._per_card_results = []  # tuples: (db_card_id, is_correct, elapsed_ms)
        self._question_started_at = None

        
        # Application state
        self.current_flashcards = []
        self.current_index = 0
        self.score = {"correct": 0, "wrong": 0}
        self.study_start_time = None
        self.generator_thread = None
        self.progress_dialog = None
        
        # Load settings
        self.load_settings()
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Apply theme
        self.apply_theme()
        
        # Load history
        self.load_history()
        
        logger.info(f"{APP_NAME} started")
    
    def _current_project_id(self) -> int:
        # Simple heuristic: project = directory name of selected file, fallback to "Default Project"
        path = self.file_path_edit.text().strip()
        if path and os.path.exists(path):
            pname = Path(path).parent.name or "Default Project"
            return config.db.get_or_create_project(pname, str(Path(path).parent))
        return config.db.get_or_create_project("Default Project", None)

    def _register_document(self, file_path: str, raw_text: str) -> Tuple[int, int]:
        """Returns (project_id, document_id) and registers if new."""
        project_id = self._current_project_id()
        clean = preprocess_text(raw_text)
        doc_hash = sha256_text(normalize_for_hash(clean))
        p = Path(file_path)
        mtime = p.stat().st_mtime
        size = p.stat().st_size
        document_id = config.db.get_or_create_document(project_id, str(p), p.name, doc_hash, mtime, size)
        # Stash for session
        self._last_document_id = document_id
        self._last_project_id = project_id
        self._last_doc_hash = doc_hash
        return project_id, document_id

    def _merge_with_stored(self, new_cards: List[Dict], document_id: int, target_count: int, mode: str) -> List[Dict]:
        """Mix stored + new based on mode and remove duplicates."""
        stored = config.db.fetch_cards_for_document(document_id)
        # Map to plain dicts (same shape as generator)
        stored_plain = [{
            "db_id": c["db_id"], "question": c["question"], "options": c["options"],
            "correct_index": c["correct_index"], "explanation": c.get("explanation","")
        } for c in stored]

        def key(card): return normalize_for_hash(card["question"] + " || " + json.dumps(card["options"], ensure_ascii=False))
        seen = set()
        out = []

        def add_list(lst):
            for c in lst:
                k = key(c)
                if k in seen: continue
                seen.add(k)
                out.append(c)
                if len(out) >= target_count:
                    break

        random.shuffle(stored_plain)
        random.shuffle(new_cards)

        if mode == "New only":
            add_list(new_cards)
            if len(out) < target_count:
                add_list(stored_plain)
        elif mode == "Review only":
            add_list(stored_plain)
            if len(out) < target_count:
                add_list(new_cards)
        else:  # Mixed
            half = max(1, target_count // 2)
            add_list(stored_plain[:])  # try to add ~half from stored, then fill
            if len(out) < half:
                # ensure we have at least half from stored, if possible
                pass
            add_list(new_cards[:])
            if len(out) < target_count:
                add_list(stored_plain + new_cards)

        # Final shuffle so order feels fresh
        random.shuffle(out)
        return out[:target_count]


    def start_stored_session(self):
        """Start a session using stored cards for the currently selected document path."""
        if not self.file_path_edit.text().strip():
            QMessageBox.information(self, "Select a document", "Choose a document first (used to locate the project's card bank).")
            return
        # Read and register current doc so we can fetch by document_id
        try:
            document_text = DocumentReader.read_file(self.file_path_edit.text().strip())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read document:\n{e}")
            return
        _, document_id = self._register_document(self.file_path_edit.text().strip(), document_text)

        # Pull stored cards
        stored_cards = config.db.fetch_cards_for_document(document_id)
        if not stored_cards:
            QMessageBox.information(self, "No stored cards", "No cards stored yet for this document. Generate first, then try again.")
            return

        # Convert to generator-like format & start
        cards = [{
            "db_id": c["db_id"],
            "question": c["question"],
            "options": c["options"],
            "correct_index": c["correct_index"],
            "explanation": c.get("explanation", "")
        } for c in stored_cards]

        # honor controls
        target = self.session_cards_spin.value()
        mode = self.mode_combo.currentText()
        # In 'Review only' we simply subsample stored
        random.shuffle(cards)
        if mode == "Review only":
            chosen = cards[:target]
        else:
            # 'Mixed'/'New only' without new generation will just use stored
            chosen = cards[:target]

        self._session_mode = mode
        self._session_time_limit = self.time_limit_spin.value() * 60
        self._session_card_db_ids = [c.get("db_id") for c in chosen]

        self.current_flashcards = chosen
        self._start_session_common()

    def _start_session_common(self):
        self.current_index = 0
        self.score = {"correct": 0, "wrong": 0}
        self.study_start_time = datetime.datetime.now()
        random.shuffle(self.current_flashcards)
        self.next_btn.setEnabled(False)
        self.finish_btn.setEnabled(True)
        self.show_current_flashcard()
        self.update_study_display()
        self.study_timer.start(1000)  # elapsed up-counter

        if self._session_time_limit > 0:
            self._remaining_seconds = self._session_time_limit
            self.countdown_timer.start(1000)
        else:
            self._remaining_seconds = 0
            self.countdown_timer.stop()

    def _update_countdown(self):
        if self._remaining_seconds <= 0:
            self.countdown_timer.stop()
            self.finish_session()
            return
        self._remaining_seconds -= 1
        m, s = divmod(self._remaining_seconds, 60)
        self.timer_label.setText(f"Time left: {m:02d}:{s:02d}")



    def load_settings(self):
        """Load application settings"""
        self.restoreGeometry(config.settings.value("geometry", b""))
        self.restoreState(config.settings.value("windowState", b""))



    
    def save_settings(self):
        """Save application settings"""
        config.settings.setValue("geometry", self.saveGeometry())
        config.settings.setValue("windowState", self.saveState())
    
    def setup_ui(self):
        """Setup main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Main tab widget
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 10))
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                margin-top: -1px;
            }
            QTabBar::tab {
                background-color: #F5F5F5;
                padding: 8px 20px;
                margin-right: 2px;
                border: 1px solid #E0E0E0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #EEEEEE;
            }
        """)
        
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.setup_generation_tab()
        self.setup_study_tab()
        self.setup_history_tab()
        self.setup_statistics_tab()
    
    def setup_menu(self):
        """Setup application menu"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Document", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.show_settings)
        settings_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status labels
        self.status_label = QLabel("Ready")
        self.model_label = QLabel("")
        self.cards_label = QLabel("")
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.cards_label)
        self.status_bar.addPermanentWidget(self.model_label)
        
        self.update_status_bar()
    
    def update_status_bar(self):
        """Update status bar information"""
        model = getattr(self, 'current_model', 'None')
        self.model_label.setText(f"Model: {model}")
        
        if self.current_flashcards:
            total = len(self.current_flashcards)
            current = self.current_index + 1 if self.current_index < total else total
            self.cards_label.setText(f"Cards: {current}/{total}")
        else:
            self.cards_label.setText("Cards: 0/0")
    
    def setup_generation_tab(self):
        """Setup flashcard generation tab"""
        self.generation_tab = QWidget()
        self.tabs.addTab(self.generation_tab, "📚 Generate")
        
        layout = QVBoxLayout(self.generation_tab)
        layout.setSpacing(15)
        
        # Model selection
        model_group = StyledGroupBox("Model Configuration")
        model_layout = QHBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Ollama Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        models = config.get_models()
        self.model_combo.addItems(models)
        self.current_model = models[0] if models else "None"
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        refresh_btn = ModernButton("Refresh Models")
        refresh_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(refresh_btn)
        
        model_layout.addStretch()
        layout.addWidget(model_group)
        
        # Document selection
        doc_group = StyledGroupBox("Document Selection")
        doc_layout = QVBoxLayout(doc_group)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Document:"))
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a document file...")
        self.file_path_edit.textChanged.connect(self.on_file_changed)
        file_layout.addWidget(self.file_path_edit)
        
        browse_btn = ModernButton("Browse Files")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        doc_layout.addLayout(file_layout)
        
        # Supported formats info
        formats_label = QLabel("Supported formats: ")
        supported = ["TXT"]
        if PDF_AVAILABLE:
            supported.append("PDF")
        if DOCX_AVAILABLE:
            supported.append("DOCX")
        if not OLLAMA_AVAILABLE:
            supported.append(" (Mock mode - Ollama not available)")
        
        formats_label.setText(f"Supported formats: {', '.join(supported)}")
        formats_label.setStyleSheet("color: #666; font-size: 9pt;")
        doc_layout.addWidget(formats_label)
        
        layout.addWidget(doc_group)
        
        # Generation settings
        settings_group = StyledGroupBox("Generation Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Cards per section:"), 0, 0)
        self.cards_spinbox = QSpinBox()
        self.cards_spinbox.setRange(1, 10)
        self.cards_spinbox.setValue(config.settings.value("cards_per_section", 5, type=int))
        settings_layout.addWidget(self.cards_spinbox, 0, 1)
        
        settings_layout.addWidget(QLabel("Difficulty level:"), 0, 2)
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["Easy", "Medium", "Hard", "Mixed"])
        self.difficulty_combo.setCurrentText("Medium")
        settings_layout.addWidget(self.difficulty_combo, 0, 3)
        
        layout.addWidget(settings_group)
        
        # Custom prompt
        prompt_group = StyledGroupBox("Custom Instructions (Optional)")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_text = QTextEdit()
        self.prompt_text.setMaximumHeight(120)
        self.prompt_text.setPlaceholderText("Enter custom instructions for flashcard generation...")
        default_prompt = """Focus on key concepts, important facts, and practical applications.
Create questions that test understanding rather than just memorization.
Include a mix of factual and analytical questions."""
        self.prompt_text.setPlainText(default_prompt)
        prompt_layout.addWidget(self.prompt_text)
        
        layout.addWidget(prompt_group)
        
        # Generate button
        self.generate_btn = ModernButton("🚀 Generate Flashcards", primary=True)
        self.generate_btn.setMinimumHeight(50)
        self.generate_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.generate_btn.clicked.connect(self.generate_flashcards)
        layout.addWidget(self.generate_btn)
        
        layout.addStretch()
    
    def setup_study_tab(self):
        """Setup study tab with enhanced UI"""
        self.study_tab = QWidget()
        self.tabs.addTab(self.study_tab, "🎯 Study")
        
        layout = QVBoxLayout(self.study_tab)
        layout.setSpacing(20)
        
        # Progress and score header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)

                # Challenge controls
        self.session_cards_spin = QSpinBox()
        self.session_cards_spin.setRange(1, 200)
        self.session_cards_spin.setValue(20)
        self.session_cards_spin.setToolTip("Cards in this session")
        header_layout.addWidget(QLabel("Cards:"))
        header_layout.addWidget(self.session_cards_spin)

        self.time_limit_spin = QSpinBox()
        self.time_limit_spin.setRange(0, 240)
        self.time_limit_spin.setValue(0)
        self.time_limit_spin.setToolTip("Time limit (minutes). 0 = no limit")
        header_layout.addWidget(QLabel("Time limit:"))
        header_layout.addWidget(self.time_limit_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Mixed", "New only", "Review only"])
        header_layout.addWidget(QLabel("Mode:"))
        header_layout.addWidget(self.mode_combo)

        self.start_challenge_btn = ModernButton("Start Challenge", primary=True)
        self.start_challenge_btn.clicked.connect(self.start_stored_session)
        header_layout.addWidget(self.start_challenge_btn)


        self.progress_label = QLabel("Progress: 0/0")
        self.progress_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        header_layout.addWidget(self.progress_label)
        
        header_layout.addStretch()
        
        self.score_label = QLabel("Score: 0 correct, 0 wrong")
        self.score_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        header_layout.addWidget(self.score_label)
        
        self.timer_label = QLabel("")
        self.timer_label.setFont(QFont("Segoe UI", 11))
        header_layout.addWidget(self.timer_label)
        
        layout.addWidget(header_frame)
        
        # Flashcard widget
        self.flashcard_widget = FlashcardWidget()
        self.flashcard_widget.answer_selected.connect(self.on_answer_selected)
        layout.addWidget(self.flashcard_widget)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.prev_btn = ModernButton("← Previous")
        self.prev_btn.clicked.connect(self.previous_card)
        self.prev_btn.setEnabled(False)
        controls_layout.addWidget(self.prev_btn)
        
        controls_layout.addStretch()
        
        self.next_btn = ModernButton("Next →")
        self.next_btn.clicked.connect(self.next_card)
        self.next_btn.setEnabled(False)
        controls_layout.addWidget(self.next_btn)
        
        self.finish_btn = ModernButton("Finish Session", primary=True)
        self.finish_btn.clicked.connect(self.finish_session)
        self.finish_btn.setEnabled(False)
        controls_layout.addWidget(self.finish_btn)
        
        layout.addLayout(controls_layout)
        
        # Study timer
        self.study_timer = QTimer()
        self.study_timer.timeout.connect(self.update_study_timer)
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._update_countdown)
        self._remaining_seconds = 0
        self._last_answer_timestamps = {}  # card_index -> ms elapsed
        self._session_mode = "Mixed"
        self._session_time_limit = 0
        self._session_card_db_ids = []  # filled if cards come from DB

    def setup_history_tab(self):
        """Setup history tab"""
        self.history_tab = QWidget()
        self.tabs.addTab(self.history_tab, "📖 History")
        
        layout = QVBoxLayout(self.history_tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        search_label = QLabel("Search:")
        controls_layout.addWidget(search_label)
        
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search flashcards...")
        self.history_search.textChanged.connect(self.filter_history)
        controls_layout.addWidget(self.history_search)
        
        clear_btn = ModernButton("Clear History")
        clear_btn.clicked.connect(self.clear_history)
        controls_layout.addWidget(clear_btn)
        
        layout.addLayout(controls_layout)
        
        # History display
        self.history_browser = QTextBrowser()
        self.history_browser.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.history_browser)
    
    def setup_statistics_tab(self):
        """Setup statistics tab"""
        self.stats_tab = QWidget()
        self.tabs.addTab(self.stats_tab, "📊 Statistics")
        
        layout = QVBoxLayout(self.stats_tab)
        
        # Stats display
        stats_group = StyledGroupBox("Study Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.total_cards_label = QLabel("Total flashcards created: 0")
        stats_layout.addWidget(self.total_cards_label, 0, 0)
        
        self.total_sessions_label = QLabel("Study sessions completed: 0")
        stats_layout.addWidget(self.total_sessions_label, 0, 1)
        
        self.avg_score_label = QLabel("Average score: 0%")
        stats_layout.addWidget(self.avg_score_label, 1, 0)
        
        self.total_time_label = QLabel("Total study time: 0 minutes")
        stats_layout.addWidget(self.total_time_label, 1, 1)
        
        layout.addWidget(stats_group)
        
        # Recent activity
        activity_group = StyledGroupBox("Recent Activity")
        activity_layout = QVBoxLayout(activity_group)
        
        self.activity_list = QListWidget()
        activity_layout.addWidget(self.activity_list)
        
        layout.addWidget(activity_group)
        
        layout.addStretch()
    
    def apply_theme(self):
        """Apply application theme"""
        dark_mode = config.settings.value("dark_mode", False, type=bool)
        
        if dark_mode:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #3c3c3c;
                }
                QTextEdit, QLineEdit {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                    color: #ffffff;
                }
            """)
    
    # Event handlers
    def on_model_changed(self, model):
        """Handle model selection change"""
        self.current_model = model
        self.update_status_bar()
    
    def on_file_changed(self, path):
        """Handle file path change"""
        self.status_label.setText("Document loaded" if path else "Ready")
    
    def on_answer_selected(self, index, is_correct):
        """Handle flashcard answer selection"""
        if is_correct:
            self.score["correct"] += 1
        else:
            self.score["wrong"] += 1
        
        self.update_study_display()
        
        # Enable next button after a short delay
        QTimer.singleShot(1500, lambda: self.next_btn.setEnabled(True))

        if self._question_started_at:
            elapsed_ms = int((datetime.datetime.now() - self._question_started_at).total_seconds() * 1000)
        else:
            elapsed_ms = 0
        db_id = None
        if 0 <= self.current_index < len(self.current_flashcards):
            db_id = self.current_flashcards[self.current_index].get("db_id")
        if db_id:
            self._per_card_results.append((db_id, is_correct, elapsed_ms))

    
    def refresh_models(self):
        """Refresh available Ollama models"""
        self.model_combo.clear()
        models = config.get_models()
        self.model_combo.addItems(models)
        if models:
            self.current_model = models[0]
            self.update_status_bar()
    
    def browse_file(self):
        """Browse and select document file"""
        file_types = "All Supported (*.txt"
        if PDF_AVAILABLE:
            file_types += " *.pdf"
        if DOCX_AVAILABLE:
            file_types += " *.docx"
        file_types += ");;Text Files (*.txt)"
        
        if PDF_AVAILABLE:
            file_types += ";;PDF Files (*.pdf)"
        if DOCX_AVAILABLE:
            file_types += ";;Word Documents (*.docx)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Document", "", file_types
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
    
    def generate_flashcards(self):
        """Generate flashcards from document"""
        file_path = self.file_path_edit.text().strip()
        
        if not file_path:
            QMessageBox.warning(self, "No Document", "Please select a document first.")
            return
        
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", "The selected file does not exist.")
            return
        
        try:
            # Read document
            self.status_label.setText("Reading document...")
            document_text = DocumentReader.read_file(file_path)
            
            if len(document_text.strip()) < 100:
                QMessageBox.warning(self, "Document Too Short", 
                    "The document is too short to generate meaningful flashcards.")
                return
            
        except Exception as e:
            QMessageBox.critical(self, "Error Reading Document", 
                f"Failed to read the document:\n{str(e)}")
            return
        
        # Setup progress dialog
        self.progress_dialog = QProgressDialog("Generating flashcards...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.canceled.connect(self.cancel_generation)
        
        # Get generation parameters
        prompt = self.prompt_text.toPlainText().strip()
        num_cards = self.cards_spinbox.value()
        difficulty = self.difficulty_combo.currentText()
        
        # Add difficulty to prompt
        if difficulty != "Mixed":
            prompt += f"\nDifficulty level: {difficulty}"
        
        # Start generation thread
        self.generator_thread = FlashcardGenerator(
            model=self.current_model,
            prompt=prompt,
            document_text=document_text,
            num_cards=num_cards
        )
        
        self.generator_thread.finished.connect(self.on_generation_finished)
        self.generator_thread.error.connect(self.on_generation_error)
        self.generator_thread.progress.connect(self.on_generation_progress)
        self.generator_thread.status_update.connect(self.status_label.setText)
        
        self.generate_btn.setEnabled(False)
        self.generator_thread.start()
    
    def cancel_generation(self):
        """Cancel flashcard generation"""
        if self.generator_thread:
            self.generator_thread.stop()
            self.generator_thread.wait(3000)
        
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Generation cancelled")
    
    def on_generation_progress(self, current, total):
        """Update generation progress"""
        if self.progress_dialog:
            percent = int((current / total) * 100)
            self.progress_dialog.setValue(percent)
    
    def on_generation_finished(self, flashcards):
        """Handle generation completion"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        self.generate_btn.setEnabled(True)

        if not flashcards:
            QMessageBox.information(self, "No Flashcards",
                "No flashcards could be generated from this document.")
            self.status_label.setText("Generation failed")
            return

        # We have cards — persist to DB (best effort)
        try:
            file_path = self.file_path_edit.text().strip()
            if file_path:
                document_text = DocumentReader.read_file(file_path)
                project_id, document_id = self._register_document(file_path, document_text)
                inserted_ids = config.db.insert_cards(document_id, flashcards)
                logger.info(f"Inserted {len(inserted_ids)} new cards into the local bank (doc_id={document_id}).")
        except Exception as e:
            logger.error(f"Failed to persist cards: {e}")

        # Proceed to study
        self.current_flashcards = flashcards
        self.current_index = 0
        self.score = {"correct": 0, "wrong": 0}

        # Keep the legacy json history (optional)
        self.save_to_history(flashcards)

        # Switch to study tab and start session (this will mix with stored based on controls)
        self.tabs.setCurrentIndex(1)
        self.start_study_session()

        self.status_label.setText(f"Generated {len(flashcards)} flashcards")

    
    def on_generation_error(self, error):
        """Handle generation error"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.generate_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Generation Error", 
            f"Failed to generate flashcards:\n{error}")
        
        self.status_label.setText("Generation failed")
    
    def start_study_session(self):
        """Start after generation (mix new + stored according to controls)."""
        if not self.current_flashcards:
            return
        # When coming from generation, we already know project/doc
        target = self.session_cards_spin.value()
        mode = self.mode_combo.currentText()
        # If we have document context, merge with stored
        if hasattr(self, "_last_document_id"):
            merged = self._merge_with_stored(self.current_flashcards, self._last_document_id, target, mode)
            self.current_flashcards = merged
            self._session_card_db_ids = [c.get("db_id") for c in merged if c.get("db_id")]
        else:
            # Fallback: trim to target
            self.current_flashcards = self.current_flashcards[:target]
            self._session_card_db_ids = [c.get("db_id") for c in self.current_flashcards if c.get("db_id")]

        self._session_mode = mode
        self._session_time_limit = self.time_limit_spin.value() * 60
        self._start_session_common()
        
    def show_current_flashcard(self):
        """Display the current flashcard"""
        if 0 <= self.current_index < len(self.current_flashcards):
            flashcard = self.current_flashcards[self.current_index]
            self.flashcard_widget.set_flashcard(flashcard)
            
            # Update navigation buttons
            self.prev_btn.setEnabled(self.current_index > 0)
            self.next_btn.setEnabled(False)  # Will be enabled after answering
            self._question_started_at = datetime.datetime.now()

    
    def next_card(self):
        """Move to next flashcard"""
        if self.current_index < len(self.current_flashcards) - 1:
            self.current_index += 1
            self.show_current_flashcard()
            self.update_study_display()
        else:
            # Last card, finish session
            self.finish_session()
    
    def previous_card(self):
        """Move to previous flashcard"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_flashcard()
            self.update_study_display()
    
    def update_study_display(self):
        """Update study session display"""
        total = len(self.current_flashcards)
        current = self.current_index + 1
        
        self.progress_label.setText(f"Progress: {current}/{total}")
        self.score_label.setText(
            f"Score: {self.score['correct']} correct, {self.score['wrong']} wrong"
        )
        
        self.update_status_bar()
    
    def update_study_timer(self):
        """Update study session timer"""
        if self.study_start_time:
            elapsed = datetime.datetime.now() - self.study_start_time
            minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
            self.timer_label.setText(f"Time: {minutes:02d}:{seconds:02d}")
    
    def finish_session(self):
        """Finish current study session"""
        if not self.current_flashcards:
            return

        self.study_timer.stop()

        # Calculate session stats first
        total_questions = len(self.current_flashcards)
        correct = self.score["correct"]
        percentage = (correct / total_questions * 100) if total_questions > 0 else 0
        elapsed_time = datetime.datetime.now() - self.study_start_time if self.study_start_time else datetime.timedelta()
        minutes = int(elapsed_time.total_seconds() / 60)

        # Record to DB (best effort)
        try:
            project_id = getattr(self, "_last_project_id", self._current_project_id())
            document_id = getattr(self, "_last_document_id", None)
            started_at = self.study_start_time.isoformat() if self.study_start_time else datetime.datetime.now().isoformat()
            duration_sec = int(elapsed_time.total_seconds())
            tlimit = self._session_time_limit
            config.db.record_session(
                project_id, document_id, self._session_mode, started_at,
                duration_sec, total_questions, correct, tlimit,
                [(cid, ok, ms) for (cid, ok, ms) in self._per_card_results]
            )
        except Exception as e:
            logger.error(f"Failed to store session: {e}")
        finally:
            self._per_card_results = []

        # Show results
        result_text = f"""

Study Session Complete!

📊 Results:
• Questions answered: {total_questions}
• Correct answers: {correct}
• Accuracy: {percentage:.1f}%
• Time spent: {minutes} minutes

Great job studying! 🎉
        """
        QMessageBox.information(self, "Session Complete", result_text)

        # Save legacy stats json (optional)
        self.save_session_stats(total_questions, correct, minutes)

        # Reset UI
        self.finish_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)

        # Update statistics tab from our json file
        self.update_statistics()

        self.status_label.setText("Study session completed")

    
    def save_to_history(self, flashcards):
        """Save flashcards to history"""
        try:
            history = []
            if config.history_file.exists():
                history = json.loads(config.history_file.read_text())
            
            # Add metadata
            session_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_cards": len(flashcards),
                "model": self.current_model,
                "flashcards": flashcards
            }
            
            history.append(session_data)
            
            # Keep only last 50 sessions
            if len(history) > 50:
                history = history[-50:]
            
            config.history_file.write_text(json.dumps(history, indent=2))
            self.load_history()
            
        except Exception as e:
            logger.error(f"Error saving to history: {e}")
    
    def load_history(self):
        """Load sessions from DB and render compact history."""
        try:
            rows = config.db.recent_sessions(limit=10)
            if not rows:
                self.history_browser.setHtml("<p>No sessions yet.</p>")
                return

            html = """
            <style>
            body { font-family: 'Segoe UI', sans-serif; margin: 10px; }
            .session { background:#f8f9fa; margin:10px 0; padding:12px; border-radius:8px; }
            .hdr { font-weight:bold; color:#2196F3; margin-bottom:6px; }
            .meta { color:#666; font-size:10pt; }
            </style>
            <body>
            """
            for r in rows:
                started = r["started_at"] or ""
                title = r["doc_title"] or "(unknown doc)"
                proj  = r["project_name"] or "(project)"
                total = r["total_questions"] or 0
                correct = r["correct_answers"] or 0
                pct = f"{(correct/total*100):.0f}%" if total else "0%"
                tlim = r["time_limit_sec"] or 0
                html += f"""
                <div class='session'>
                <div class='hdr'>📅 {started} • {title} • {proj}</div>
                <div class='meta'>Mode: {r['mode']} • Score: {correct}/{total} ({pct}) • Time limit: {int(tlim/60)} min</div>
                </div>
                """
            html += "</body>"
            self.history_browser.setHtml(html)
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.history_browser.setHtml(f"<p>Error: {e}</p>")
         
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.history_browser.setHtml(f"<p>Error loading history: {e}</p>")
    
    def filter_history(self):
        q = self.history_search.text().strip()
        if not q:
            self.load_history()
            return
        try:
            matches = config.db.search_cards(q)
            if not matches:
                self.history_browser.setHtml("<p>No matching cards.</p>")
                return
            html = """
            <style>
            body { font-family:'Segoe UI', sans-serif; margin:10px;}
            .card { background:#fff; border-left:4px solid #4CAF50; margin:8px 0; padding:10px; border-radius:4px;}
            .q { font-weight:bold; margin-bottom:5px;}
            .a { color:#666;}
            </style><body>
            """
            for c in matches[:50]:
                correct = c["options"][c["correct_index"]] if c["options"] else ""
                html += f"<div class='card'><div class='q'>Q: {c['question']}</div><div class='a'>A: {correct}</div></div>"
            html += "</body>"
            self.history_browser.setHtml(html)
        except Exception as e:
            logger.error(f"search error: {e}")
            self.history_browser.setHtml(f"<p>Error: {e}</p>")

    
    def clear_history(self):
        """Clear flashcard history"""
        reply = QMessageBox.question(
            self, "Clear History", 
            "Are you sure you want to clear all flashcard history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if config.history_file.exists():
                    config.history_file.unlink()
                self.history_browser.setHtml("<p>History cleared.</p>")
                self.status_label.setText("History cleared")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear history: {e}")
    
    def save_session_stats(self, total, correct, minutes):
        """Save session statistics"""
        try:
            stats_file = config.app_dir / "statistics.json"
            stats = {"sessions": [], "total_cards": 0, "total_time": 0}
            
            if stats_file.exists():
                stats = json.loads(stats_file.read_text())
            
            # Add new session
            session_stat = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_questions": total,
                "correct_answers": correct,
                "accuracy": (correct / total * 100) if total > 0 else 0,
                "time_minutes": minutes
            }
            
            stats["sessions"].append(session_stat)
            stats["total_cards"] += total
            stats["total_time"] += minutes
            
            # Keep only last 100 sessions
            if len(stats["sessions"]) > 100:
                stats["sessions"] = stats["sessions"][-100:]
            
            stats_file.write_text(json.dumps(stats, indent=2))
            
        except Exception as e:
            logger.error(f"Error saving session stats: {e}")
    
    def update_statistics(self):
        """Update statistics display"""
        try:
            stats_file = config.app_dir / "statistics.json"
            if not stats_file.exists():
                return
            
            stats = json.loads(stats_file.read_text())
            sessions = stats.get("sessions", [])
            
            if not sessions:
                return
            
            # Calculate statistics
            total_sessions = len(sessions)
            total_cards = sum(s["total_questions"] for s in sessions)
            total_time = sum(s["time_minutes"] for s in sessions)
            avg_accuracy = sum(s["accuracy"] for s in sessions) / total_sessions
            
            # Update labels
            self.total_cards_label.setText(f"Total flashcards studied: {total_cards}")
            self.total_sessions_label.setText(f"Study sessions completed: {total_sessions}")
            self.avg_score_label.setText(f"Average accuracy: {avg_accuracy:.1f}%")
            self.total_time_label.setText(f"Total study time: {total_time} minutes")
            
            # Update recent activity
            self.activity_list.clear()
            for session in sessions[-10:]:  # Last 10 sessions
                timestamp = datetime.datetime.fromisoformat(session["timestamp"])
                activity_text = (
                    f"{timestamp.strftime('%m/%d %H:%M')} - "
                    f"{session['correct_answers']}/{session['total_questions']} "
                    f"({session['accuracy']:.0f}%) in {session['time_minutes']}min"
                )
                self.activity_list.addItem(activity_text)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Reload settings that might have changed
            self.apply_theme()
            self.update_status_bar()
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
        <h2>{APP_NAME} v{APP_VERSION}</h2>
        
        <p>An intelligent flashcard generator powered by local LLM models.</p>
        
        <h3>Features:</h3>
        <ul>
            <li>📚 Generate flashcards from PDF, DOCX, and TXT documents</li>
            <li>🎯 Interactive study sessions with progress tracking</li>
            <li>📊 Detailed statistics and performance analytics</li>
            <li>🔄 Smart caching for faster regeneration</li>
            <li>🎨 Modern, user-friendly interface</li>
        </ul>
        
        <h3>Requirements:</h3>
        <ul>
            <li>Ollama installed and running</li>
            <li>At least one language model downloaded</li>
        </ul>
        
        <p><b>Status:</b></p>
        <ul>
            <li>Ollama: {"✅ Available" if OLLAMA_AVAILABLE else "❌ Not available"}</li>
            <li>PDF Support: {"✅ Available" if PDF_AVAILABLE else "❌ Not available"}</li>
            <li>DOCX Support: {"✅ Available" if DOCX_AVAILABLE else "❌ Not available"}</li>
        </ul>
        
        <p>Created with ❤️ using PyQt5 and modern UI principles.</p>
        """
        
        QMessageBox.about(self, "About FlashCard Studio", about_text)
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop any running threads
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.stop()
            self.generator_thread.wait(3000)
        
        # Save settings
        self.save_settings()
        
        # Stop timers
        if hasattr(self, 'study_timer'):
            self.study_timer.stop()
        
        logger.info("Application closed")
        event.accept()

def setup_application():
    """Setup application with proper error handling"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("FlashCard Studio")
    
    # Set application icon (if available)
    try:
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except:
        pass
    
    # Apply application-wide stylesheet
    app.setStyleSheet("""
        QApplication {
            font-family: "Segoe UI", "Arial", sans-serif;
        }
        QToolTip {
            background-color: #333333;
            color: white;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 3px;
        }
    """)
    
    return app

def check_dependencies():
    """Check and report missing dependencies"""
    missing = []
    warnings = []
    
    if not OLLAMA_AVAILABLE:
        missing.append("ollama (pip install ollama)")
        warnings.append("⚠️ Ollama not available - will use mock mode for testing")
    
    if not PDF_AVAILABLE:
        warnings.append("⚠️ PDF support not available (pip install pymupdf)")
    
    if not DOCX_AVAILABLE:
        warnings.append("⚠️ DOCX support not available (pip install python-docx)")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print()
    
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    return len(missing) == 0

def main():
    """Main application entry point"""
    print(f"🚀 Starting {APP_NAME} v{APP_VERSION}")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies and try again.")
        return 1
    
    try:
        # Setup application
        app = setup_application()
        
        # Create and show main window
        window = FlashcardApp()
        window.show()
        
        # Center window on screen
        screen = app.primaryScreen().geometry()
        window_geo = window.geometry()
        x = (screen.width() - window_geo.width()) // 2
        y = (screen.height() - window_geo.height()) // 2
        window.move(x, y)
        
        logger.info("Application started successfully")
        print(f"✅ {APP_NAME} started successfully!")
        
        # Run application
        return app.exec_()
        
    except Exception as e:
        error_msg = f"Failed to start application: {e}"
        logger.critical(error_msg)
        print(f"❌ {error_msg}")
        
        # Show error dialog if possible
        try:
            app = QApplication(sys.argv) if 'app' not in locals() else app
            QMessageBox.critical(None, "Startup Error", error_msg)
        except:
            pass
        
        return 1

if __name__ == "__main__":
    sys.exit(main())