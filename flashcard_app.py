#!/usr/bin/env python3
"""
FlashCard Studio — v2.0
LLM Flashcard Generator + Study App

- Generate MCQ flashcards from TXT/PDF/DOCX via local LLM (Ollama) or mock.
- Study sessions with keyboard shortcuts and sound effects.
- Clickable History: view full session details (your answer vs correct).
- Local SQLite storage with FTS (if available).
"""

import sys
import json
import os
import random
import datetime
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import sqlite3
import difflib
from contextlib import contextmanager

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox,
    QGroupBox, QMessageBox, QFileDialog, QProgressDialog, QFrame,
    QStatusBar, QAction, QDialog, QDialogButtonBox, QSpinBox, QCheckBox,
    QGridLayout, QListWidget, QListWidgetItem, QTextBrowser, QSplitter
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSettings, QSize, QUrl, QObject, QEvent
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtMultimedia import QSoundEffect

# -------- Optional document libs --------
PDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF legacy import
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pymupdf as fitz  # new alias
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# -------- Optional LLM (Ollama) --------
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# -------- Logging --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flashcard_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

APP_NAME = "FlashCard Studio"
APP_VERSION = "2.0"
USE_MOCK = not OLLAMA_AVAILABLE  # use mock generation if Ollama missing


# =========================
# Utilities / Config
# =========================

def normalize_for_hash(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 %.,;:?!()\[\]{}<>/\\+\-=_'\"|@#*&$]", " ", s)
    return s.strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def preprocess_text(text: str) -> str:
    """Trim whitespace & remove boilerplate lines."""
    if not text:
        return ""
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"table of contents.*?\n\n", "", text, flags=re.I | re.S)
    text = re.sub(r"©.*?\n", "", text, flags=re.I)
    text = re.sub(r"page \d+.*?\n", "", text, flags=re.I)
    return text.strip()


def chunk_document(text: str, max_words: int = 500) -> List[str]:
    """Chunk text by headers/paragraphs within word limits."""
    if not text:
        return []
    sections = re.split(r"\n\s*#{1,3}\s+.+\n", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        words = sec.split()
        if len(words) <= max_words:
            if len(words) > 50:
                chunks.append(sec)
        else:
            paragraphs = [p.strip() for p in sec.split("\n\n") if p.strip()]
            cur, w = [], 0
            for p in paragraphs:
                pw = len(p.split())
                if cur and w + pw > max_words:
                    chunks.append("\n\n".join(cur))
                    cur, w = [p], pw
                else:
                    cur.append(p)
                    w += pw
            if cur:
                chunks.append("\n\n".join(cur))
    return chunks


class AppConfig:
    def __init__(self):
        self.app_dir = Path.home() / ".flashcard_studio"
        self.app_dir.mkdir(exist_ok=True)

        self.cache_dir = self.app_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.history_file = self.app_dir / "history.json"
        self.settings = QSettings("FlashCardStudio", "Settings")
        self.db_path = self.app_dir / "flashcards.db"
        self.db = None  # after Database init

    def get_models(self) -> List[str]:
        defaults = [
            "mistral:7b-instruct", "llama3.1:8b", "qwen2.5:7b",
            "gemma2:9b", "codellama:7b", "neural-chat:7b"
        ]
        if not OLLAMA_AVAILABLE:
            return ["mock-model (Ollama not available)"]
        try:
            res = ollama.list()
            names = [m["name"] for m in res.get("models", [])]
            return names or defaults
        except Exception:
            return defaults


config = AppConfig()


# =========================
# I/O Readers
# =========================

class DocumentReader:
    @staticmethod
    def read_file(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = p.suffix.lower()
        try:
            if ext == ".txt":
                return DocumentReader._read_txt(p)
            if ext == ".pdf" and PDF_AVAILABLE:
                return DocumentReader._read_pdf(p)
            if ext in (".docx", ".doc") and DOCX_AVAILABLE:
                return DocumentReader._read_docx(p)
            raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"read_file error on {file_path}: {e}")
            raise

    @staticmethod
    def _read_txt(path: Path) -> str:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        with fitz.open(str(path)) as doc:
            return "\n".join([pg.get_text("text") for pg in doc])

    @staticmethod
    def _read_docx(path: Path) -> str:
        doc = DocxDocument(str(path))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# =========================
# Database
# =========================

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
                due_at TEXT,
                ease REAL,
                interval_days REAL,
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
                chosen_index INTEGER,
                PRIMARY KEY(session_id, card_id),
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(card_id) REFERENCES cards(id)
            );
            """)
            # Optional FTS
            try:
                cur.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts
                USING fts5(question, explanation, content='cards', content_rowid='id');
                INSERT INTO cards_fts(rowid, question, explanation)
                SELECT id, question, COALESCE(explanation, '')
                FROM cards
                WHERE id NOT IN (SELECT rowid FROM cards_fts);
                """)
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False

    # --- Projects/Documents ---
    def get_or_create_project(self, name: str, root_path: Optional[str]) -> int:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id FROM projects WHERE name=?", (name,))
            row = cur.fetchone()
            if row:
                return row["id"]
            cur.execute("INSERT INTO projects(name, root_path) VALUES(?,?)", (name, root_path))
            return cur.lastrowid

    def get_or_create_document(self, project_id: int, path: str, title: str,
                               doc_hash: str, mtime: float, size: int) -> int:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id FROM documents WHERE doc_hash=?", (doc_hash,))
            row = cur.fetchone()
            if row:
                return row["id"]
            cur.execute("""
                INSERT INTO documents(project_id, path, title, doc_hash, mtime, size)
                VALUES (?,?,?,?,?,?)
            """, (project_id, path, title, doc_hash, mtime, size))
            return cur.lastrowid

    # --- Cards ---
    def card_exists(self, card_hash: str) -> bool:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT 1 FROM cards WHERE card_hash=?", (card_hash,))
            return cur.fetchone() is not None

    def insert_cards(self, document_id: int, cards: List[Dict]) -> List[int]:
        inserted = []
        with self.conn() as con:
            cur = con.cursor()
            for c in cards:
                options_json = json.dumps(c["options"], ensure_ascii=False)
                key = normalize_for_hash(c["question"] + " || " + options_json)
                chash = sha256_text(key)
                if self.card_exists(chash):
                    continue
                cur.execute("""
                    INSERT INTO cards(document_id, question, options_json, correct_index, explanation, card_hash)
                    VALUES (?,?,?,?,?,?)
                """, (document_id, c["question"], options_json, int(c["correct_index"]), c.get("explanation", ""), chash))
                inserted.append(cur.lastrowid)
            if self.fts_enabled and inserted:
                qmarks = ",".join("?" * len(inserted))
                cur.execute(
                    f"INSERT INTO cards_fts(rowid, question, explanation) "
                    f"SELECT id, question, COALESCE(explanation,'') FROM cards WHERE id IN ({qmarks})",
                    inserted
                )
        return inserted

    def fetch_cards_for_document(self, document_id: int) -> List[Dict]:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id, question, options_json, correct_index, explanation FROM cards WHERE document_id=?", (document_id,))
            rows = cur.fetchall()
            return [{
                "db_id": r["id"],
                "question": r["question"],
                "options": json.loads(r["options_json"]),
                "correct_index": int(r["correct_index"]),
                "explanation": r["explanation"] or ""
            } for r in rows]

    def search_cards(self, query: str) -> List[Dict]:
        q = query.strip()
        if not q:
            return []
        with self.conn() as con:
            cur = con.cursor()
            if self.fts_enabled:
                cur.execute("""
                    SELECT c.id, c.question, c.options_json, c.correct_index, c.explanation
                    FROM cards c JOIN cards_fts f ON c.id = f.rowid
                    WHERE cards_fts MATCH ? ORDER BY rank
                """, (q,))
            else:
                like = f"%{q}%"
                cur.execute("""
                    SELECT id, question, options_json, correct_index, explanation
                    FROM cards
                    WHERE question LIKE ? OR explanation LIKE ?
                """, (like, like))
            rows = cur.fetchall()
            return [{
                "db_id": r["id"],
                "question": r["question"],
                "options": json.loads(r["options_json"]),
                "correct_index": int(r["correct_index"]),
                "explanation": r["explanation"] or ""
            } for r in rows]

    # --- Sessions ---
    def record_session(self, project_id: int, document_id: Optional[int], mode: str, started_at: str,
                       duration_sec: int, total: int, correct: int, time_limit_sec: int,
                       per_card: List[Tuple[int, bool, int, Optional[int]]]) -> int:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO sessions(project_id, document_id, mode, started_at, duration_sec,
                                     total_questions, correct_answers, time_limit_sec)
                VALUES (?,?,?,?,?,?,?,?)
            """, (project_id, document_id, mode, started_at, duration_sec, total, correct, time_limit_sec))
            sid = cur.lastrowid
            if per_card:
                rows = []
                for t in per_card:
                    if len(t) == 4:
                        cid, ok, ms, chosen = t
                    else:
                        cid, ok, ms = t
                        chosen = None
                    rows.append((sid, cid, 1 if ok else 0, ms, chosen))
                cur.executemany("""
                    INSERT INTO session_cards(session_id, card_id, is_correct, elapsed_ms, chosen_index)
                    VALUES (?,?,?,?,?)
                """, rows)
            return sid

    def get_session_overview(self, limit: int = 50) -> List[sqlite3.Row]:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("""
                SELECT s.id, s.started_at, s.mode, s.total_questions, s.correct_answers,
                       s.time_limit_sec, d.title as doc_title, p.name as project_name
                FROM sessions s
                LEFT JOIN documents d ON s.document_id = d.id
                LEFT JOIN projects p ON s.project_id = p.id
                ORDER BY s.started_at DESC
                LIMIT ?
            """, (limit,))
            return cur.fetchall()

    def get_session_cards(self, session_id: int) -> List[Dict]:
        with self.conn() as con:
            cur = con.cursor()
            cur.execute("""
                SELECT sc.card_id, sc.is_correct, sc.elapsed_ms, sc.chosen_index,
                       c.question, c.options_json, c.correct_index, c.explanation
                FROM session_cards sc
                JOIN cards c ON c.id = sc.card_id
                WHERE sc.session_id = ?
                ORDER BY sc.rowid
            """, (session_id,))
            out = []
            for r in cur.fetchall():
                out.append({
                    "card_id": r["card_id"],
                    "is_correct": bool(r["is_correct"]),
                    "elapsed_ms": r["elapsed_ms"] or 0,
                    "chosen_index": r["chosen_index"] if r["chosen_index"] is not None else -1,
                    "question": r["question"],
                    "options": json.loads(r["options_json"]),
                    "correct_index": int(r["correct_index"]),
                    "explanation": r["explanation"] or ""
                })
            return out


# init DB
if config.db is None:
    config.db = Database(config.db_path)


# =========================
# Flashcard Generator Thread
# =========================

class FlashcardGenerator(QThread):
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
        self._stop_requested = True

    def _cache_key(self) -> str:
        clean = preprocess_text(self.document_text)
        doc_hash = sha256_text(normalize_for_hash(clean))
        prompt_hash = sha256_text(normalize_for_hash(self.prompt + self.model + str(self.num_cards)))
        return sha256_text(doc_hash + "::" + prompt_hash)

    def _cache_load(self) -> Optional[List[Dict]]:
        f = self.cache_dir / f"{self._cache_key()}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _cache_save(self, cards: List[Dict]):
        try:
            (self.cache_dir / f"{self._cache_key()}.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"cache save failed: {e}")

    def _validate_card(self, card: Dict) -> bool:
        if not all(k in card for k in ("question", "options", "correct_index")):
            return False
        if not isinstance(card["options"], list) or len(card["options"]) != 4:
            return False
        if not (0 <= int(card["correct_index"]) < 4):
            return False
        if not str(card["question"]).strip():
            return False
        return True

    def _remove_dups(self, cards: List[Dict]) -> List[Dict]:
        out, seen = [], []
        for c in cards:
            qn = normalize_for_hash(c["question"])
            if any(difflib.SequenceMatcher(None, qn, s).ratio() >= 0.87 for s in seen):
                continue
            out.append(c)
            seen.append(qn)
        return out

    def _gen_for_chunk(self, chunk: str) -> List[Dict]:
        if self._stop_requested:
            return []
        # build prompt
        prompt = f"""
Create {min(self.num_cards, 8)} multiple-choice flashcards from the text.

Requirements:
1) 4 options exactly (A-D), one correct.
2) Vary factual/conceptual/application.
3) Plausible distractors.
4) JSON array ONLY.

{self.prompt}

Text:
{chunk[:2000]}

JSON format:
[
  {{
    "question": "…",
    "options": ["optA","optB","optC","optD"],
    "correct_index": 0,
    "explanation": "why correct"
  }}
]
""".strip()

        if USE_MOCK:
            mock = []
            for i in range(min(self.num_cards, 3)):
                mock.append({
                    "question": f"Mock question {i+1} from this section?",
                    "options": [f"Correct {i+1}", f"Wrong A{i+1}", f"Wrong B{i+1}", f"Wrong C{i+1}"],
                    "correct_index": 0,
                    "explanation": f"Because it's the correct {i+1}."
                })
            return mock

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator. Return VALID JSON ONLY."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.7, "num_predict": 2000}
            )
            text = resp["message"]["content"].strip()
            l, r = text.find("["), text.rfind("]") + 1
            if l == -1 or r <= l:
                raise ValueError("No JSON array found")
            data = json.loads(text[l:r])
            out = []
            for card in data:
                if self._validate_card(card):
                    correct_answer = card["options"][card["correct_index"]]
                    random.shuffle(card["options"])
                    card["correct_index"] = card["options"].index(correct_answer)
                    out.append(card)
            return out
        except Exception as e:
            logger.error(f"LLM gen error: {e}")
            return []

    def run(self):
        try:
            self.status_update.emit("Checking cache…")
            cached = self._cache_load()
            if cached:
                self.finished.emit(cached)
                return

            self.status_update.emit("Preprocessing document…")
            clean = preprocess_text(self.document_text)
            if len(clean.split()) < 50:
                raise ValueError("Document too short")

            chunks = chunk_document(clean, max_words=500)
            if not chunks:
                raise ValueError("No meaningful content found")

            total = min(len(chunks), 10)
            chunks = chunks[:total]
            self.status_update.emit(f"Processing {total} sections…")

            all_cards: List[Dict] = []
            for i, ch in enumerate(chunks, 1):
                if self._stop_requested:
                    break
                self.status_update.emit(f"Generating cards for section {i}…")
                all_cards.extend(self._gen_for_chunk(ch))
                self.progress.emit(i, total)
                if not USE_MOCK:
                    self.msleep(400)

            unique = self._remove_dups(all_cards)
            if not unique:
                self.error.emit("No valid flashcards were generated.")
                return
            self._cache_save(unique)
            self.finished.emit(unique)
        except Exception as e:
            self.error.emit(f"Generation failed: {e}")


# =========================
# UI helpers (widgets/styles/sound)
# =========================

class ModernButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self._restyle()

    def _restyle(self):
        if self.primary:
            self.setStyleSheet("""
            QPushButton{background:#2196F3;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-weight:bold;}
            QPushButton:hover{background:#1976D2;}
            QPushButton:pressed{background:#1565C0;}
            QPushButton:disabled{background:#CCC;color:#666;}
            """)
        else:
            self.setStyleSheet("""
            QPushButton{background:#F5F5F5;color:#333;border:1px solid #DDD;border-radius:6px;padding:8px 16px;}
            QPushButton:hover{background:#EEE;border-color:#BBB;}
            QPushButton:pressed{background:#E0E0E0;}
            """)


class StyledGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setStyleSheet("""
        QGroupBox{font-weight:bold;border:2px solid #E0E0E0;border-radius:8px;margin:10px 0;padding-top:10px;}
        QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 10px;color:#2196F3;}
        """)


class SoundManager:
    def __init__(self):
        self.enabled = bool(config.settings.value("sound_enabled", True, type=bool))
        self.correct_path = config.settings.value("sound_correct_path", "", type=str)
        self.wrong_path = config.settings.value("sound_wrong_path", "", type=str)
        self.ui_path = config.settings.value("sound_ui_path", "", type=str)
        self._eff_correct = self._make(self.correct_path)
        self._eff_wrong = self._make(self.wrong_path)
        self._eff_ui = self._make(self.ui_path)

    def _make(self, path: str) -> Optional[QSoundEffect]:
        if not path:
            return None
        try:
            eff = QSoundEffect()
            eff.setSource(QUrl.fromLocalFile(path))
            eff.setVolume(0.7)
            return eff
        except Exception:
            return None

    def _play(self, eff: Optional[QSoundEffect]):
        if not self.enabled:
            return
        if eff:
            eff.play()
        else:
            QApplication.beep()

    def play_correct(self): self._play(self._eff_correct)
    def play_wrong(self):   self._play(self._eff_wrong)
    def play_ui(self):      self._play(self._eff_ui)


class UiSoundFilter(QObject):
    def __init__(self, sounder: SoundManager, parent=None):
        super().__init__(parent)
        self.s = sounder

    def eventFilter(self, obj, event):
        if not self.s.enabled:
            return False
        et = event.type()
        if et == QEvent.MouseButtonPress and isinstance(obj, QPushButton):
            self.s.play_ui()
        elif et == QEvent.KeyPress:
            self.s.play_ui()
        return False


class FlashcardWidget(QWidget):
    answer_selected = pyqtSignal(int, bool)  # (index, is_correct)

    def __init__(self):
        super().__init__()
        self.flashcard: Optional[Dict] = None
        self.answered = False
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        qframe = QFrame()
        qframe.setFrameStyle(QFrame.Box)
        qframe.setStyleSheet("""
        QFrame{background:#F8F9FA;border:1px solid #E0E0E0;border-radius:8px;padding:15px;}
        """)
        ql = QVBoxLayout(qframe)
        self.question_label = QTextEdit()
        self.question_label.setReadOnly(True)
        self.question_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        self.question_label.setMaximumHeight(120)
        self.question_label.setStyleSheet("QTextEdit{border:none;background:transparent;}")
        ql.addWidget(self.question_label)
        layout.addWidget(qframe)

        self.option_buttons: List[QPushButton] = []
        grid = QGridLayout()
        for i in range(4):
            btn = QPushButton()
            btn.setMinimumHeight(50)
            btn.setFont(QFont("Segoe UI", 10))
            btn.clicked.connect(lambda _, idx=i: self.select_answer(idx))
            btn.setStyleSheet("""
            QPushButton{text-align:left;padding:10px 15px;border:2px solid #E0E0E0;border-radius:6px;background:#fff;}
            QPushButton:hover{border-color:#2196F3;background:#F5F9FF;}
            """)
            r, c = i // 2, i % 2
            grid.addWidget(btn, r, c)
            self.option_buttons.append(btn)
        layout.addLayout(grid)

    def set_flashcard(self, card: Dict):
        self.flashcard = card
        self.answered = False
        self.question_label.setText(card["question"])
        for i, btn in enumerate(self.option_buttons):
            if i < len(card["options"]):
                btn.setText(f"{chr(ord('A')+i)}. {card['options'][i]}")
                btn.setEnabled(True)
                btn.setVisible(True)
                btn.setStyleSheet("""
                QPushButton{text-align:left;padding:10px 15px;border:2px solid #E0E0E0;border-radius:6px;background:#fff;}
                QPushButton:hover{border-color:#2196F3;background:#F5F9FF;}
                """)
            else:
                btn.setVisible(False)

    def select_answer(self, index: int):
        if self.answered or not self.flashcard:
            return
        self.answered = True
        correct = int(self.flashcard["correct_index"])
        is_correct = (index == correct)
        # Styles
        for i, btn in enumerate(self.option_buttons):
            btn.setEnabled(False)
            if i == correct:
                btn.setStyleSheet("""
                QPushButton{text-align:left;padding:10px 15px;border:2px solid #4CAF50;border-radius:6px;background:#E8F5E8;color:#2E7D32;font-weight:bold;}
                """)
            elif i == index and not is_correct:
                btn.setStyleSheet("""
                QPushButton{text-align:left;padding:10px 15px;border:2px solid #F44336;border-radius:6px;background:#FFEBEE;color:#C62828;font-weight:bold;}
                """)
        self.answer_selected.emit(index, is_correct)


# =========================
# Settings dialog
# =========================

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(420, 320)
        self._build()
        self._load()

    def _build(self):
        root = QVBoxLayout(self)
        grp_gen = StyledGroupBox("Generation")
        grid = QGridLayout(grp_gen)

        grid.addWidget(QLabel("Flashcards per section:"), 0, 0)
        self.cards_spinbox = QSpinBox()
        self.cards_spinbox.setRange(1, 10)
        grid.addWidget(self.cards_spinbox, 0, 1)

        grid.addWidget(QLabel("Max document sections:"), 1, 0)
        self.sections_spinbox = QSpinBox()
        self.sections_spinbox.setRange(1, 20)
        grid.addWidget(self.sections_spinbox, 1, 1)

        root.addWidget(grp_gen)

        grp_ui = StyledGroupBox("Interface")
        ui = QGridLayout(grp_ui)
        self.dark_mode_check = QCheckBox("Dark Mode (restart may be required)")
        ui.addWidget(self.dark_mode_check, 0, 0)
        self.debug_check = QCheckBox("Enable debug logging")
        ui.addWidget(self.debug_check, 1, 0)
        self.sound_enable_check = QCheckBox("Enable sounds")
        ui.addWidget(self.sound_enable_check, 2, 0)
        root.addWidget(grp_ui)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _load(self):
        self.cards_spinbox.setValue(config.settings.value("cards_per_section", 5, type=int))
        self.sections_spinbox.setValue(config.settings.value("max_sections", 10, type=int))
        self.dark_mode_check.setChecked(config.settings.value("dark_mode", False, type=bool))
        self.debug_check.setChecked(config.settings.value("debug_logging", False, type=bool))
        self.sound_enable_check.setChecked(config.settings.value("sound_enabled", True, type=bool))

    def accept(self):
        config.settings.setValue("cards_per_section", self.cards_spinbox.value())
        config.settings.setValue("max_sections", self.sections_spinbox.value())
        config.settings.setValue("dark_mode", self.dark_mode_check.isChecked())
        config.settings.setValue("debug_logging", self.debug_check.isChecked())
        config.settings.setValue("sound_enabled", self.sound_enable_check.isChecked())
        logging.getLogger().setLevel(logging.DEBUG if self.debug_check.isChecked() else logging.INFO)
        super().accept()


# =========================
# Main Window
# =========================

class FlashcardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(80, 60, 1200, 900)

        # state
        self.current_flashcards: List[Dict] = []
        self.current_index = 0
        self.score = {"correct": 0, "wrong": 0}
        self.study_start_time: Optional[datetime.datetime] = None
        self.generator_thread: Optional[FlashcardGenerator] = None
        self.progress_dialog: Optional[QProgressDialog] = None
        self._per_card_results: List[Tuple[int, bool, int, Optional[int]]] = []
        self._question_started_at: Optional[datetime.datetime] = None
        self._remaining_seconds = 0
        self._session_mode = "Mixed"
        self._session_time_limit = 0
        self._session_card_db_ids: List[Optional[int]] = []

        self._load_win_settings()
        self._build_ui()
        self._build_menu()
        self._build_status()
        self.apply_theme()

        # Sounds
        self.sounds = SoundManager()
        self._ui_filter = UiSoundFilter(self.sounds, self)
        QApplication.instance().installEventFilter(self._ui_filter)

        self.load_history()
        logger.info(f"{APP_NAME} started")

    # ---- app/util ----
    def _load_win_settings(self):
        self.restoreGeometry(config.settings.value("geometry", b""))
        self.restoreState(config.settings.value("windowState", b""))

    def _save_win_settings(self):
        config.settings.setValue("geometry", self.saveGeometry())
        config.settings.setValue("windowState", self.saveState())

    def _current_project_id(self) -> int:
        path = self.file_path_edit.text().strip()
        if path and os.path.exists(path):
            pname = Path(path).parent.name or "Default Project"
            return config.db.get_or_create_project(pname, str(Path(path).parent))
        return config.db.get_or_create_project("Default Project", None)

    def _register_document(self, file_path: str, raw_text: str) -> Tuple[int, int]:
        project_id = self._current_project_id()
        clean = preprocess_text(raw_text)
        doc_hash = sha256_text(normalize_for_hash(clean))
        p = Path(file_path)
        document_id = config.db.get_or_create_document(
            project_id, str(p), p.name, doc_hash, p.stat().st_mtime, p.stat().st_size
        )
        self._last_document_id = document_id
        self._last_project_id = project_id
        self._last_doc_hash = doc_hash
        return project_id, document_id

    # ---- UI build ----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 10))
        self.tabs.setStyleSheet("""
        QTabWidget::pane{border:1px solid #E0E0E0;border-radius:4px;margin-top:-1px;}
        QTabBar::tab{background:#F5F5F5;padding:8px 20px;margin-right:2px;border:1px solid #E0E0E0;border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;}
        QTabBar::tab:selected{background:#fff;border-bottom:1px solid #fff;}
        QTabBar::tab:hover{background:#EEE;}
        """)
        root.addWidget(self.tabs)

        self._build_generate_tab()
        self._build_study_tab()
        self._build_history_tab()
        self._build_stats_tab()

        # timers
        self.study_timer = QTimer()
        self.study_timer.timeout.connect(self._update_study_timer)
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._update_countdown)

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        act_open = QAction("Open Document", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.browse_file)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_quit = QAction("Exit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        settings_menu = menubar.addMenu("Settings")
        act_pref = QAction("Preferences", self)
        act_pref.triggered.connect(self.show_settings)
        settings_menu.addAction(act_pref)

        help_menu = menubar.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

    def _build_status(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.model_label = QLabel("")
        self.cards_label = QLabel("")
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.cards_label)
        self.status_bar.addPermanentWidget(self.model_label)
        self._update_status_bar()

    def _update_status_bar(self):
        model = getattr(self, "current_model", "None")
        self.model_label.setText(f"Model: {model}")
        if self.current_flashcards:
            total = len(self.current_flashcards)
            current = min(self.current_index + 1, total)
            self.cards_label.setText(f"Cards: {current}/{total}")
        else:
            self.cards_label.setText("Cards: 0/0")

    # ---- Generate tab ----
    def _build_generate_tab(self):
        w = QWidget()
        self.tabs.addTab(w, "📚 Generate")
        layout = QVBoxLayout(w)
        layout.setSpacing(15)

        # Model row
        model_group = StyledGroupBox("Model Configuration")
        hl = QHBoxLayout(model_group)
        hl.addWidget(QLabel("Ollama Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        models = config.get_models()
        self.model_combo.addItems(models)
        self.current_model = models[0] if models else "None"
        self.model_combo.currentTextChanged.connect(self._on_model_change)
        hl.addWidget(self.model_combo)
        btn_refresh = ModernButton("Refresh Models")
        btn_refresh.clicked.connect(self._refresh_models)
        hl.addWidget(btn_refresh)
        hl.addStretch()
        layout.addWidget(model_group)

        # Document row
        doc_group = StyledGroupBox("Document Selection")
        vl = QVBoxLayout(doc_group)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Document:"))
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a document file…")
        self.file_path_edit.textChanged.connect(self._on_file_changed)
        fl.addWidget(self.file_path_edit)
        b = ModernButton("Browse Files")
        b.clicked.connect(self.browse_file)
        fl.addWidget(b)
        vl.addLayout(fl)

        supported = ["TXT"]
        if PDF_AVAILABLE: supported.append("PDF")
        if DOCX_AVAILABLE: supported.append("DOCX")
        if not OLLAMA_AVAILABLE: supported.append("(Mock mode - Ollama not available)")
        lbl = QLabel(f"Supported formats: {', '.join(supported)}")
        lbl.setStyleSheet("color:#666;font-size:9pt;")
        vl.addWidget(lbl)
        layout.addWidget(doc_group)

        # Generation settings
        settings_group = StyledGroupBox("Generation Settings")
        g = QGridLayout(settings_group)
        g.addWidget(QLabel("Cards per section:"), 0, 0)
        self.cards_spinbox = QSpinBox()
        self.cards_spinbox.setRange(1, 10)
        self.cards_spinbox.setValue(config.settings.value("cards_per_section", 5, type=int))
        g.addWidget(self.cards_spinbox, 0, 1)

        g.addWidget(QLabel("Difficulty level:"), 0, 2)
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["Easy", "Medium", "Hard", "Mixed"])
        self.difficulty_combo.setCurrentText("Medium")
        g.addWidget(self.difficulty_combo, 0, 3)
        layout.addWidget(settings_group)

        # Prompt
        prompt_group = StyledGroupBox("Custom Instructions (Optional)")
        pv = QVBoxLayout(prompt_group)
        self.prompt_text = QTextEdit()
        self.prompt_text.setMaximumHeight(120)
        self.prompt_text.setPlaceholderText("Enter custom instructions…")
        self.prompt_text.setPlainText(
            "Focus on key concepts, important facts, and practical applications.\n"
            "Create questions that test understanding rather than memorization.\n"
            "Include a mix of factual and analytical questions."
        )
        pv.addWidget(self.prompt_text)
        layout.addWidget(prompt_group)

        # Generate button
        self.generate_btn = ModernButton("🚀 Generate Flashcards", primary=True)
        self.generate_btn.setMinimumHeight(48)
        self.generate_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.generate_btn.clicked.connect(self.generate_flashcards)
        layout.addWidget(self.generate_btn)

        layout.addStretch()

    # ---- Study tab ----
    def _build_study_tab(self):
        w = QWidget()
        self.tabs.addTab(w, "🎯 Study")
        layout = QVBoxLayout(w)
        layout.setSpacing(20)

        header = QFrame()
        header.setStyleSheet("QFrame{background:#F8F9FA;border-radius:8px;padding:10px;}")
        hl = QHBoxLayout(header)

        self.session_cards_spin = QSpinBox()
        self.session_cards_spin.setRange(1, 200)
        self.session_cards_spin.setValue(20)
        self.session_cards_spin.setToolTip("Cards in this session")
        hl.addWidget(QLabel("Cards:"))
        hl.addWidget(self.session_cards_spin)

        self.time_limit_spin = QSpinBox()
        self.time_limit_spin.setRange(0, 240)
        self.time_limit_spin.setValue(0)
        self.time_limit_spin.setToolTip("Time limit (minutes). 0 = no limit")
        hl.addWidget(QLabel("Time limit:"))
        hl.addWidget(self.time_limit_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Mixed", "New only", "Review only"])
        hl.addWidget(QLabel("Mode:"))
        hl.addWidget(self.mode_combo)

        start_btn = ModernButton("Start Challenge", primary=True)
        start_btn.clicked.connect(self.start_stored_session)
        hl.addWidget(start_btn)

        self.progress_label = QLabel("Progress: 0/0")
        self.progress_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        hl.addWidget(self.progress_label)
        hl.addStretch()

        self.score_label = QLabel("Score: 0 correct, 0 wrong")
        self.score_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        hl.addWidget(self.score_label)

        self.timer_label = QLabel("")
        self.timer_label.setFont(QFont("Segoe UI", 11))
        hl.addWidget(self.timer_label)

        layout.addWidget(header)

        self.flashcard_widget = FlashcardWidget()
        self.flashcard_widget.answer_selected.connect(self._on_answer_selected)
        layout.addWidget(self.flashcard_widget)

        controls = QHBoxLayout()
        self.prev_btn = ModernButton("← Previous")
        self.prev_btn.clicked.connect(self.previous_card)
        self.prev_btn.setEnabled(False)
        controls.addWidget(self.prev_btn)

        controls.addStretch()

        self.next_btn = ModernButton("Next →")
        self.next_btn.clicked.connect(self.next_card)
        self.next_btn.setEnabled(False)
        controls.addWidget(self.next_btn)

        self.finish_btn = ModernButton("Finish Session", primary=True)
        self.finish_btn.clicked.connect(self.finish_session)
        self.finish_btn.setEnabled(False)
        controls.addWidget(self.finish_btn)
        layout.addLayout(controls)

    # ---- History tab ----
    def _build_history_tab(self):
        w = QWidget()
        self.tabs.addTab(w, "📖 History")
        root = QVBoxLayout(w)

        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search cards…")
        self.history_search.textChanged.connect(self.filter_history)
        top.addWidget(self.history_search)
        self.refresh_history_btn = ModernButton("Refresh")
        self.refresh_history_btn.clicked.connect(self.load_history)
        top.addWidget(self.refresh_history_btn)
        clear_btn = ModernButton("Clear History")
        clear_btn.clicked.connect(self.clear_history)
        top.addWidget(clear_btn)
        root.addLayout(top)

        split = QSplitter()
        split.setOrientation(Qt.Horizontal)

        self.history_list = QListWidget()
        self.history_list.itemSelectionChanged.connect(self._on_history_select)
        split.addWidget(self.history_list)

        self.history_details = QTextBrowser()
        self.history_details.setOpenExternalLinks(False)
        split.addWidget(self.history_details)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        root.addWidget(split)

    # ---- Stats tab ----
    def _build_stats_tab(self):
        w = QWidget()
        self.tabs.addTab(w, "📊 Statistics")
        layout = QVBoxLayout(w)

        grp = StyledGroupBox("Study Statistics")
        grid = QGridLayout(grp)
        self.total_cards_label = QLabel("Total flashcards studied: 0")
        grid.addWidget(self.total_cards_label, 0, 0)
        self.total_sessions_label = QLabel("Study sessions completed: 0")
        grid.addWidget(self.total_sessions_label, 0, 1)
        self.avg_score_label = QLabel("Average accuracy: 0%")
        grid.addWidget(self.avg_score_label, 1, 0)
        self.total_time_label = QLabel("Total study time: 0 minutes")
        grid.addWidget(self.total_time_label, 1, 1)
        layout.addWidget(grp)

        activity = StyledGroupBox("Recent Activity")
        self.activity_list = QListWidget()
        v = QVBoxLayout(activity)
        v.addWidget(self.activity_list)
        layout.addWidget(activity)
        layout.addStretch()

    # ---- Theme ----
    def apply_theme(self):
        if config.settings.value("dark_mode", False, type=bool):
            self.setStyleSheet("""
            QMainWindow, QWidget{background:#2b2b2b;color:#fff;}
            QTabWidget::pane{border:1px solid #555;background:#3c3c3c;}
            QTextEdit, QLineEdit{background:#3c3c3c;border:1px solid #555;color:#fff;}
            """)

    # ---- Handlers ----
    def _on_model_change(self, m: str):
        self.current_model = m
        self._update_status_bar()

    def _on_file_changed(self, _path: str):
        self.status_label.setText("Document loaded" if _path else "Ready")

    def _refresh_models(self):
        self.model_combo.clear()
        models = config.get_models()
        self.model_combo.addItems(models)
        if models:
            self.current_model = models[0]
            self._update_status_bar()

    # ---- File pick ----
    def browse_file(self):
        types = "All Supported (*.txt"
        if PDF_AVAILABLE: types += " *.pdf"
        if DOCX_AVAILABLE: types += " *.docx"
        types += ");;Text Files (*.txt)"
        if PDF_AVAILABLE: types += ";;PDF Files (*.pdf)"
        if DOCX_AVAILABLE: types += ";;Word Documents (*.docx)"
        fp, _ = QFileDialog.getOpenFileName(self, "Select Document", "", types)
        if fp:
            self.file_path_edit.setText(fp)

    # ---- Generate ----
    def generate_flashcards(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "No Document", "Please select a document first.")
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", "The selected file does not exist.")
            return
        try:
            self.status_label.setText("Reading document…")
            doc_text = DocumentReader.read_file(file_path)
            if len(doc_text.strip()) < 100:
                QMessageBox.warning(self, "Document Too Short", "Too short to generate meaningful flashcards.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Error Reading Document", f"Failed to read the document:\n{e}")
            return

        self.progress_dialog = QProgressDialog("Generating flashcards…", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.canceled.connect(self.cancel_generation)

        prompt = self.prompt_text.toPlainText().strip()
        num_cards = self.cards_spinbox.value()
        difficulty = self.difficulty_combo.currentText()
        if difficulty != "Mixed":
            prompt += f"\nDifficulty level: {difficulty}"

        self.generator_thread = FlashcardGenerator(
            model=self.current_model, prompt=prompt, document_text=doc_text, num_cards=num_cards
        )
        self.generator_thread.finished.connect(self._on_generation_finished)
        self.generator_thread.error.connect(self._on_generation_error)
        self.generator_thread.progress.connect(self._on_generation_progress)
        self.generator_thread.status_update.connect(self.status_label.setText)

        self.generate_btn.setEnabled(False)
        self.generator_thread.start()

    def cancel_generation(self):
        if self.generator_thread:
            self.generator_thread.stop()
            self.generator_thread.wait(3000)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Generation cancelled")

    def _on_generation_progress(self, current, total):
        if self.progress_dialog:
            self.progress_dialog.setValue(int(current / total * 100))

    def _on_generation_finished(self, cards: List[Dict]):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        self.generate_btn.setEnabled(True)

        if not cards:
            QMessageBox.information(self, "No Flashcards", "Could not generate flashcards for this document.")
            self.status_label.setText("Generation failed")
            return

        # persist to DB
        try:
            fp = self.file_path_edit.text().strip()
            if fp:
                text = DocumentReader.read_file(fp)
                project_id, document_id = self._register_document(fp, text)
                config.db.insert_cards(document_id, cards)
                logger.info(f"Cards saved to DB (doc_id={document_id}).")
        except Exception as e:
            logger.error(f"Persist cards failed: {e}")

        self.current_flashcards = cards
        self.current_index = 0
        self.score = {"correct": 0, "wrong": 0}
        self._save_legacy_history(cards)

        self.tabs.setCurrentIndex(1)
        self.start_study_session()
        self.status_label.setText(f"Generated {len(cards)} flashcards")

    def _on_generation_error(self, err: str):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Generation Error", err)
        self.status_label.setText("Generation failed")

    # ---- Study session ----
    def start_study_session(self):
        if not self.current_flashcards:
            return
        target = self.session_cards_spin.value()
        mode = self.mode_combo.currentText()
        if hasattr(self, "_last_document_id"):
            merged = self._merge_with_stored(self.current_flashcards, self._last_document_id, target, mode)
            self.current_flashcards = merged
            self._session_card_db_ids = [c.get("db_id") for c in merged if c.get("db_id")]
        else:
            self.current_flashcards = self.current_flashcards[:target]
            self._session_card_db_ids = [c.get("db_id") for c in self.current_flashcards if c.get("db_id")]
        self._session_mode = mode
        self._session_time_limit = self.time_limit_spin.value() * 60
        self._start_session_common()

    def start_stored_session(self):
        if not self.file_path_edit.text().strip():
            QMessageBox.information(self, "Select a document", "Choose a document first (to locate the project's card bank).")
            return
        try:
            text = DocumentReader.read_file(self.file_path_edit.text().strip())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read document:\n{e}")
            return
        _, doc_id = self._register_document(self.file_path_edit.text().strip(), text)
        stored = config.db.fetch_cards_for_document(doc_id)
        if not stored:
            QMessageBox.information(self, "No stored cards", "No cards stored yet for this document. Generate first.")
            return
        cards = [{
            "db_id": c["db_id"], "question": c["question"], "options": c["options"],
            "correct_index": c["correct_index"], "explanation": c.get("explanation", "")
        } for c in stored]
        random.shuffle(cards)
        target = self.session_cards_spin.value()
        mode = self.mode_combo.currentText()
        chosen = cards[:target]  # in Review only or otherwise, just sample
        self._session_mode = mode
        self._session_time_limit = self.time_limit_spin.value() * 60
        self._session_card_db_ids = [c.get("db_id") for c in chosen]
        self.current_flashcards = chosen
        self._start_session_common()

    def _merge_with_stored(self, new_cards: List[Dict], document_id: int, target: int, mode: str) -> List[Dict]:
        stored = config.db.fetch_cards_for_document(document_id)
        stored_plain = [{
            "db_id": c["db_id"], "question": c["question"], "options": c["options"],
            "correct_index": c["correct_index"], "explanation": c.get("explanation", "")
        } for c in stored]

        def key(c): return normalize_for_hash(c["question"] + " || " + json.dumps(c["options"], ensure_ascii=False))
        seen = set()
        out: List[Dict] = []

        def add_list(lst):
            for c in lst:
                k = key(c)
                if k in seen:
                    continue
                seen.add(k)
                out.append(c)
                if len(out) >= target:
                    break

        random.shuffle(stored_plain)
        random.shuffle(new_cards)

        if mode == "New only":
            add_list(new_cards)
            if len(out) < target: add_list(stored_plain)
        elif mode == "Review only":
            add_list(stored_plain)
            if len(out) < target: add_list(new_cards)
        else:
            half = max(1, target // 2)
            add_list(stored_plain[:])
            add_list(new_cards[:])
            if len(out) < target:
                add_list(stored_plain + new_cards)

        random.shuffle(out)
        return out[:target]

    def _start_session_common(self):
        self.current_index = 0
        self.score = {"correct": 0, "wrong": 0}
        self.study_start_time = datetime.datetime.now()
        random.shuffle(self.current_flashcards)
        self.next_btn.setEnabled(False)
        self.finish_btn.setEnabled(True)
        self._show_current_flashcard()
        self._update_study_display()
        self.study_timer.start(1000)
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

    def _show_current_flashcard(self):
        if 0 <= self.current_index < len(self.current_flashcards):
            card = self.current_flashcards[self.current_index]
            self.flashcard_widget.set_flashcard(card)
            self.prev_btn.setEnabled(self.current_index > 0)
            self.next_btn.setEnabled(False)  # enabled after answering
            self._question_started_at = datetime.datetime.now()

    def next_card(self):
        if self.current_index < len(self.current_flashcards) - 1:
            self.current_index += 1
            self._show_current_flashcard()
            self._update_study_display()
        else:
            self.finish_session()

    def previous_card(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_flashcard()
            self._update_study_display()

    def _update_study_display(self):
        total = len(self.current_flashcards)
        cur = self.current_index + 1 if total else 0
        self.progress_label.setText(f"Progress: {cur}/{total}")
        self.score_label.setText(f"Score: {self.score['correct']} correct, {self.score['wrong']} wrong")
        self._update_status_bar()

    def _update_study_timer(self):
        if self.study_start_time:
            elapsed = datetime.datetime.now() - self.study_start_time
            m, s = divmod(int(elapsed.total_seconds()), 60)
            self.timer_label.setText(f"Time: {m:02d}:{s:02d}")

    def _on_answer_selected(self, index: int, is_correct: bool):
        if is_correct:
            self.score["correct"] += 1
            self.sounds.play_correct()
        else:
            self.score["wrong"] += 1
            self.sounds.play_wrong()
        self._update_study_display()
        self.next_btn.setEnabled(True)  # enable immediately

        elapsed_ms = 0
        if self._question_started_at:
            elapsed_ms = int((datetime.datetime.now() - self._question_started_at).total_seconds() * 1000)
        db_id = None
        if 0 <= self.current_index < len(self.current_flashcards):
            db_id = self.current_flashcards[self.current_index].get("db_id")
        if db_id:
            self._per_card_results.append((db_id, is_correct, elapsed_ms, index))

    def finish_session(self):
        if not self.current_flashcards:
            return
        self.study_timer.stop()
        self.countdown_timer.stop()

        total_q = len(self.current_flashcards)
        correct = self.score["correct"]
        pct = (correct / total_q * 100) if total_q else 0.0
        elapsed = datetime.datetime.now() - (self.study_start_time or datetime.datetime.now())
        minutes = int(elapsed.total_seconds() / 60)

        # store DB
        try:
            project_id = getattr(self, "_last_project_id", self._current_project_id())
            document_id = getattr(self, "_last_document_id", None)
            started_at = (self.study_start_time or datetime.datetime.now()).isoformat()
            config.db.record_session(
                project_id, document_id, self._session_mode, started_at,
                int(elapsed.total_seconds()), total_q, correct, self._session_time_limit,
                self._per_card_results
            )
        except Exception as e:
            logger.error(f"Store session failed: {e}")
        finally:
            self._per_card_results = []

        QMessageBox.information(
            self, "Session Complete",
            f"Study Session Complete!\n\n"
            f"📊 Results:\n"
            f"• Questions answered: {total_q}\n"
            f"• Correct answers: {correct}\n"
            f"• Accuracy: {pct:.1f}%\n"
            f"• Time spent: {minutes} minutes\n\n"
            f"Great job studying! 🎉"
        )

        self._save_session_stats(total_q, correct, minutes)
        # close session UI, redirect to Generate
        self.finish_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.current_flashcards = []
        self.current_index = 0
        self.study_start_time = None
        self.timer_label.setText("")
        self.progress_label.setText("Progress: 0/0")
        self.update_statistics()
        self.status_label.setText("Study session completed")
        self.tabs.setCurrentIndex(0)

    # ---- History persistence (legacy json) ----
    def _save_legacy_history(self, cards: List[Dict]):
        try:
            hist = []
            if config.history_file.exists():
                hist = json.loads(config.history_file.read_text(encoding="utf-8"))
            hist.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "total_cards": len(cards),
                "model": self.current_model,
                "flashcards": cards
            })
            if len(hist) > 50:
                hist = hist[-50:]
            config.history_file.write_text(json.dumps(hist, indent=2), encoding="utf-8")
            self.load_history()
        except Exception as e:
            logger.error(f"save legacy history failed: {e}")

    # ---- History tab functions ----
    def load_history(self):
        try:
            rows = config.db.get_session_overview(limit=50)
            self.history_list.clear()
            if not rows:
                self.history_details.setHtml("<p>No sessions yet.</p>")
                return
            for r in rows:
                started = r["started_at"] or ""
                title = r["doc_title"] or "(unknown doc)"
                proj = r["project_name"] or "(project)"
                total = r["total_questions"] or 0
                correct = r["correct_answers"] or 0
                pct = f"{(correct/total*100):.0f}%" if total else "0%"
                text = f"📅 {started} • {title} • {proj} — {correct}/{total} ({pct})"
                it = QListWidgetItem(text)
                it.setData(Qt.UserRole, r["id"])
                self.history_list.addItem(it)
            if self.history_list.count() > 0:
                self.history_list.setCurrentRow(0)
        except Exception as e:
            logger.error(f"load_history error: {e}")
            self.history_details.setHtml(f"<p>Error: {e}</p>")

    def _on_history_select(self):
        items = self.history_list.selectedItems()
        if not items:
            self.history_details.setHtml("<p>Select a session to view details.</p>")
            return
        sid = items[0].data(Qt.UserRole)
        try:
            cards = config.db.get_session_cards(sid)
            if not cards:
                self.history_details.setHtml("<p>No cards recorded for this session.</p>")
                return
            html = [
                "<style>body{font-family:'Segoe UI',sans-serif}"
                ".q{font-weight:bold;margin-top:8px}"
                ".ok{color:#2e7d32}.bad{color:#c62828}.opt{margin-left:16px}</style>",
                "<div>"
            ]
            for idx, c in enumerate(cards, 1):
                your = c["chosen_index"]
                corr = c["correct_index"]
                badge = "✅" if c["is_correct"] else "❌"
                html.append(f"<div class='card'><div class='q'>{idx}. {badge} {c['question']}</div>")
                for i, opt in enumerate(c["options"]):
                    cls = []
                    suffix = ""
                    if i == corr:
                        cls.append("ok"); suffix = " (correct)"
                    if i == your and your != corr:
                        cls.append("bad"); suffix = " (your choice)"
                    if i == your and your == corr:
                        suffix = " (you)"
                    mark = chr(ord('A') + i)
                    html.append(f"<div class='opt {' '.join(cls)}'>{mark}. {opt}{suffix}</div>")
                if c["explanation"]:
                    html.append(f"<div style='margin:6px 0 12px 0;color:#555'>💡 {c['explanation']}</div>")
                html.append("</div>")
            html.append("</div>")
            self.history_details.setHtml("".join(html))
        except Exception as e:
            logger.error(f"history detail error: {e}")
            self.history_details.setHtml(f"<p>Error: {e}</p>")

    def filter_history(self):
        q = self.history_search.text().strip()
        if not q:
            self.load_history()
            return
        try:
            matches = config.db.search_cards(q)
            if not matches:
                self.history_details.setHtml("<p>No matching cards.</p>")
                return
            html = [
                "<style>body{font-family:'Segoe UI',sans-serif;margin:10px}"
                ".card{background:#fff;border-left:4px solid #4CAF50;margin:8px 0;padding:10px;border-radius:4px}"
                ".q{font-weight:bold;margin-bottom:5px}.a{color:#666}</style><body>"
            ]
            for c in matches[:50]:
                answer = c["options"][c["correct_index"]] if c["options"] else ""
                html.append(f"<div class='card'><div class='q'>Q: {c['question']}</div><div class='a'>A: {answer}</div></div>")
            html.append("</body>")
            self.history_details.setHtml("".join(html))
        except Exception as e:
            logger.error(f"search error: {e}")
            self.history_details.setHtml(f"<p>Error: {e}</p>")

    def clear_history(self):
        if QMessageBox.question(self, "Clear History",
                                "This clears only the legacy JSON history (sessions remain in DB). Continue?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                if config.history_file.exists():
                    config.history_file.unlink()
                self.history_details.setHtml("<p>History cleared.</p>")
                self.status_label.setText("History cleared")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear history: {e}")

    # ---- Stats (simple JSON file aggregation) ----
    def _save_session_stats(self, total: int, correct: int, minutes: int):
        try:
            stats_file = config.app_dir / "statistics.json"
            stats = {"sessions": [], "total_cards": 0, "total_time": 0}
            if stats_file.exists():
                stats = json.loads(stats_file.read_text(encoding="utf-8"))
            sess = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_questions": total,
                "correct_answers": correct,
                "accuracy": (correct / total * 100) if total > 0 else 0,
                "time_minutes": minutes
            }
            stats["sessions"].append(sess)
            stats["total_cards"] += total
            stats["total_time"] += minutes
            if len(stats["sessions"]) > 100:
                stats["sessions"] = stats["sessions"][-100:]
            stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"save_session_stats error: {e}")

    def update_statistics(self):
        try:
            stats_file = config.app_dir / "statistics.json"
            if not stats_file.exists():
                return
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            sessions = stats.get("sessions", [])
            if not sessions:
                return
            total_sessions = len(sessions)
            total_cards = sum(s["total_questions"] for s in sessions)
            total_time = sum(s["time_minutes"] for s in sessions)
            avg_acc = sum(s["accuracy"] for s in sessions) / total_sessions
            self.total_cards_label.setText(f"Total flashcards studied: {total_cards}")
            self.total_sessions_label.setText(f"Study sessions completed: {total_sessions}")
            self.avg_score_label.setText(f"Average accuracy: {avg_acc:.1f}%")
            self.total_time_label.setText(f"Total study time: {total_time} minutes")

            self.activity_list.clear()
            for s in sessions[-10:]:
                ts = datetime.datetime.fromisoformat(s["timestamp"])
                txt = f"{ts.strftime('%m/%d %H:%M')} - {s['correct_answers']}/{s['total_questions']} ({s['accuracy']:.0f}%) in {s['time_minutes']}min"
                self.activity_list.addItem(txt)
        except Exception as e:
            logger.error(f"update_statistics error: {e}")

    # ---- Settings/About/Close ----
    def show_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.apply_theme()
            # refresh sound enable flag immediately
            self.sounds.enabled = bool(config.settings.value("sound_enabled", True, type=bool))
            self._update_status_bar()

    def show_about(self):
        about = f"""
        <h2>{APP_NAME} v{APP_VERSION}</h2>
        <p>LLM-powered flashcard generator and trainer.</p>
        <h3>Features</h3>
        <ul>
            <li>📚 Generate flashcards from PDF, DOCX, and TXT</li>
            <li>🎯 Study sessions with keyboard shortcuts</li>
            <li>🔊 Sounds for correct/wrong answers and UI</li>
            <li>📖 Clickable history with per-question detail</li>
            <li>📊 Simple statistics</li>
        </ul>
        <p><b>Status:</b></p>
        <ul>
            <li>Ollama: {"✅ Available" if OLLAMA_AVAILABLE else "❌ Not available (mock mode)"}</li>
            <li>PDF Support: {"✅" if PDF_AVAILABLE else "❌"}</li>
            <li>DOCX Support: {"✅" if DOCX_AVAILABLE else "❌"}</li>
        </ul>
        """
        QMessageBox.about(self, "About FlashCard Studio", about)

    def keyPressEvent(self, event):
        if self.tabs.currentWidget() is not getattr(self, "study_tab", self.tabs.widget(1)):  # best-effort
            return super().keyPressEvent(event)
        key = event.key()
        # Answer shortcuts
        if key in (Qt.Key_1, Qt.Key_A):
            self.flashcard_widget.select_answer(0)
        elif key in (Qt.Key_2, Qt.Key_B):
            self.flashcard_widget.select_answer(1)
        elif key in (Qt.Key_3, Qt.Key_C):
            self.flashcard_widget.select_answer(2)
        elif key in (Qt.Key_4, Qt.Key_D):
            self.flashcard_widget.select_answer(3)
        # Nav
        elif key in (Qt.Key_Right, Qt.Key_N, Qt.Key_Return, Qt.Key_Enter):
            if self.next_btn.isEnabled(): self.next_card()
        elif key in (Qt.Key_Left, Qt.Key_P):
            if self.prev_btn.isEnabled(): self.previous_card()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.stop()
            self.generator_thread.wait(3000)
        self._save_win_settings()
        if hasattr(self, "study_timer"):
            self.study_timer.stop()
        logger.info("Application closed")
        event.accept()


# =========================
# App wiring
# =========================

def setup_application() -> QApplication:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("FlashCard Studio")
    try:
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass
    app.setStyleSheet("""
    QApplication{font-family:"Segoe UI","Arial",sans-serif;}
    QToolTip{background:#333;color:#fff;border:1px solid #555;padding:4px;border-radius:3px;}
    """)
    return app


def check_dependencies() -> bool:
    # All optional; app can start regardless
    warnings = []
    if not OLLAMA_AVAILABLE:
        warnings.append("⚠️ Ollama not available — running in mock mode.")
    if not PDF_AVAILABLE:
        warnings.append("⚠️ PDF support not available (pip install pymupdf).")
    if not DOCX_AVAILABLE:
        warnings.append("⚠️ DOCX support not available (pip install python-docx).")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"   {w}")
        print()
    return True


def main():
    print(f"🚀 Starting {APP_NAME} v{APP_VERSION}")
    print("=" * 50)

    if not check_dependencies():
        print("❌ Please install missing dependencies and try again.")
        return 1
    try:
        app = setup_application()
        win = FlashcardApp()
        win.show()
        # center
        screen = app.primaryScreen().geometry()
        geo = win.geometry()
        win.move((screen.width() - geo.width()) // 2, (screen.height() - geo.height()) // 2)
        logger.info("Application started successfully")
        print(f"✅ {APP_NAME} started successfully!")
        return app.exec_()
    except Exception as e:
        msg = f"Failed to start application: {e}"
        logger.critical(msg)
        print(f"❌ {msg}")
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Startup Error", msg)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
