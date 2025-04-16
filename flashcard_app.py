import sys
import json
import os
import random
import datetime
import re
import hashlib
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QGroupBox,
                             QMessageBox, QFileDialog, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import fitz  # PyMuPDF
from docx import Document

USE_MOCK = False  # Set to True to use mock flashcards for testing

if not USE_MOCK:
    import ollama

def preprocess_text(text: str) -> str:
    """Clean and normalize text for better LLM processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove common irrelevant sections
    text = re.sub(r'table of contents.*?\n\n', '', text, flags=re.I|re.S)
    return text

def chunk_document(text: str, max_words: int = 300) -> List[str]:
    """Split document into logical chunks based on structure"""
    # Split by headers first
    chunks = re.split(r'\n\s*#{1,6}\s+', text)
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > max_words:
            # Further split by paragraphs
            paragraphs = chunk.split('\n\n')
            current_chunk = []
            current_words = 0
            for para in paragraphs:
                para_words = len(para.split())
                if current_words + para_words > max_words:
                    if current_chunk:
                        final_chunks.append(' '.join(current_chunk))
                    current_chunk = [para]
                    current_words = para_words
                else:
                    current_chunk.append(para)
                    current_words += para_words
            if current_chunk:
                final_chunks.append(' '.join(current_chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks

class FlashcardGenerator(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, model, prompt, document_text):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.document_text = document_text
        self.cache_dir = "flashcard_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self) -> str:
        """Generate unique cache key based on inputs"""
        content = f"{self.document_text}{self.prompt}{self.model}"
        return hashlib.md5(content.encode()).hexdigest()

    def load_from_cache(self) -> List[Dict]:
        """Try to load cached flashcards"""
        cache_file = os.path.join(self.cache_dir, f"{self.get_cache_key()}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def save_to_cache(self, flashcards: List[Dict]):
        """Save generated flashcards to cache"""
        cache_file = os.path.join(self.cache_dir, f"{self.get_cache_key()}.json")
        with open(cache_file, 'w') as f:
            json.dump(flashcards, f)

    async def async_generate_chunk_flashcards(self, chunk: str) -> List[Dict]:
        try:
            prompt = f"""
            {self.prompt}

            Document section to create flashcards from:
            {chunk}
            
            Return the flashcards as a JSON array with this format:
            [
                {{
                    "question": "Question text here",
                    "options": ["Correct answer", "Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
                    "correct_index": 0
                }}
            ]
            """

            if USE_MOCK:
                response_text = """[{"question": "Mock question?", "options": ["Right", "Wrong", "Wrong", "Wrong"], "correct_index": 0}]"""
            else:
                response = await ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert educator creating diverse and challenging flashcards."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response['message']['content']

            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                chunk_cards = json.loads(json_text)
                
                # Validate and shuffle options
                for card in chunk_cards:
                    if not all(key in card for key in ["question", "options", "correct_index"]):
                        raise ValueError("Invalid flashcard format")
                    correct_answer = card["options"][card["correct_index"]]
                    random.shuffle(card["options"])
                    card["correct_index"] = card["options"].index(correct_answer)
                    
                return chunk_cards
            else:
                raise ValueError("No valid JSON found")
        except Exception as e:
            print(f"Error generating flashcards for chunk: {e}")
            return []

    def run(self):
        try:
            # Try loading from cache first
            cached_cards = self.load_from_cache()
            if cached_cards:
                print("[Generator] Using cached flashcards")
                self.finished.emit(cached_cards)
                return

            print("[Generator] Preprocessing document...")
            clean_text = preprocess_text(self.document_text)
            chunks = chunk_document(clean_text)
            total_chunks = len(chunks)

            print(f"[Generator] Processing {total_chunks} document chunks...")
            
            # Create event loop and run async tasks
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            all_flashcards = []
            tasks = []
            
            for chunk in chunks:
                tasks.append(self.async_generate_chunk_flashcards(chunk))
            
            chunk_results = loop.run_until_complete(asyncio.gather(*tasks))
            loop.close()

            for i, chunk_cards in enumerate(chunk_results):
                all_flashcards.extend(chunk_cards)
                self.progress.emit(i + 1, total_chunks)

            # Deduplicate flashcards
            seen_questions = set()
            unique_cards = []
            for card in all_flashcards:
                if card["question"] not in seen_questions:
                    seen_questions.add(card["question"])
                    unique_cards.append(card)

            # Save to cache
            self.save_to_cache(unique_cards)

            print(f"[Generator] Generated {len(unique_cards)} unique flashcards")
            self.finished.emit(unique_cards)

        except Exception as e:
            print(f"[Generator] Error: {e}")
            self.error.emit(str(e))

class FlashcardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Flashcard Generator")
        self.setGeometry(100, 100, 1000, 800)

        self.ollama_model = "mistral"
        self.file_path = ""
        self.history_file = "flashcard_history.json"
        self.current_flashcards = []
        self.current_index = 0
        self.correct_count = 0
        self.wrong_count = 0

        self.default_prompt = """Create diverse multiple-choice flashcards for studying the following document.
For each flashcard:
1. Ask a unique and specific question about an important concept
2. Provide one correct answer
3. Generate three plausible but clearly incorrect options
4. Ensure questions cover different topics and avoid repetition
5. Mix factual recall with conceptual understanding

Generate 5 flashcards for this section."""
        self.custom_prompt = self.default_prompt
        self.generator_thread = None
        self.progress = None

        self.setup_ui()
        self.load_history()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_tab = QWidget()
        self.study_tab = QWidget()
        self.history_tab = QWidget()

        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.study_tab, "Study")
        self.tabs.addTab(self.history_tab, "History")

        self.setup_setup_tab()
        self.setup_study_tab()
        self.setup_history_tab()

    def setup_setup_tab(self):
        layout = QVBoxLayout(self.setup_tab)

        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        model_layout = QHBoxLayout()
        config_layout.addLayout(model_layout)
        model_label = QLabel("Ollama Model:")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["mistral", "llama2", "codellama", "neural-chat", "deepseek-r1:14b","mistral:7b-instruct"])
        self.model_combo.setCurrentText("mistral")
        self.model_combo.currentTextChanged.connect(self.set_ollama_model)
        model_layout.addWidget(self.model_combo)

        file_layout = QHBoxLayout()
        config_layout.addLayout(file_layout)
        file_label = QLabel("Document:")
        file_layout.addWidget(file_label)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.textChanged.connect(self.set_file_path)
        file_layout.addWidget(self.file_path_edit)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)

        prompt_group = QGroupBox("Custom Prompt")
        prompt_layout = QVBoxLayout()
        prompt_group.setLayout(prompt_layout)
        config_layout.addWidget(prompt_group)

        self.prompt_text = QTextEdit()
        self.prompt_text.setPlainText(self.default_prompt)
        self.prompt_text.textChanged.connect(self.update_prompt)
        prompt_layout.addWidget(self.prompt_text)

        generate_button = QPushButton("Generate Flashcards")
        generate_button.clicked.connect(self.generate_flashcards)
        generate_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        generate_button.setFixedHeight(40)
        layout.addWidget(generate_button)

    def setup_study_tab(self):
        layout = QVBoxLayout(self.study_tab)

        question_group = QGroupBox("Question")
        question_layout = QVBoxLayout()
        question_group.setLayout(question_layout)
        layout.addWidget(question_group)

        self.question_text = QTextEdit()
        self.question_text.setReadOnly(True)
        self.question_text.setFont(QFont("Arial", 12))
        question_layout.addWidget(self.question_text)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        self.option_buttons = []
        for i in range(4):
            btn = QPushButton()
            btn.setStyleSheet("text-align: left; padding: 10px;")
            btn.setFont(QFont("Arial", 11))
            btn.clicked.connect(lambda _, idx=i: self.check_answer(idx))
            options_layout.addWidget(btn)
            self.option_buttons.append(btn)

        progress_layout = QHBoxLayout()
        layout.addLayout(progress_layout)

        self.progress_label = QLabel("Progress: 0/0")
        self.score_label = QLabel("Score: 0 correct, 0 wrong")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.score_label)

    def setup_history_tab(self):
        layout = QVBoxLayout(self.history_tab)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        layout.addWidget(self.history_text)

    def set_ollama_model(self, model):
        self.ollama_model = model

    def set_file_path(self, path):
        self.file_path = path

    def update_prompt(self):
        self.custom_prompt = self.prompt_text.toPlainText()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document", "", "Text Files (*.txt);;PDF Files (*.pdf);;Word Documents (*.docx)")
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def read_document(self, file_path):
        """Read document content based on file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        elif ext == '.docx':
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def generate_flashcards(self):
        if not self.file_path:
            QMessageBox.warning(self, "Error", "Please select a document first.")
            return

        try:
            document_text = self.read_document(self.file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read document: {str(e)}")
            return

        self.progress = QProgressDialog("Generating flashcards...", None, 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(True)
        self.progress.setValue(0)

        self.generator_thread = FlashcardGenerator(self.ollama_model, self.custom_prompt, document_text)
        self.generator_thread.finished.connect(self.on_flashcards_generated)
        self.generator_thread.error.connect(self.on_flashcards_generation_error)
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.start()

    def on_flashcards_generation_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Failed to generate flashcards: {error_message}")

    def update_progress(self, current, total):
        if self.progress:
            percent = int((current / total) * 100)
            self.progress.setValue(percent)
        
        self.progress_label.setText(f"Progress: {current}/{total}")
        self.score_label.setText(f"Score: {self.correct_count} correct, {self.wrong_count} wrong")

    def show_flashcard(self):
        if self.current_index < len(self.current_flashcards):
            flashcard = self.current_flashcards[self.current_index]
            self.question_text.setPlainText(flashcard["question"])
            for i, btn in enumerate(self.option_buttons):
                btn.setText(flashcard["options"][i])
                btn.setStyleSheet("text-align: left; padding: 10px;")  # Reset style

    def check_answer(self, selected_index):
        flashcard = self.current_flashcards[self.current_index]
        correct_answer = flashcard["correct_index"]
        
        # Highlight correct/incorrect answers
        for i, btn in enumerate(self.option_buttons):
            if i == correct_answer:
                btn.setStyleSheet("text-align: left; padding: 10px; background-color: #90EE90;")  # Light green
            elif i == selected_index and selected_index != correct_answer:
                btn.setStyleSheet("text-align: left; padding: 10px; background-color: #FFB6C1;")  # Light red

        if selected_index == correct_answer:
            self.correct_count += 1
        else:
            self.wrong_count += 1

        # Wait a moment before moving to next card
        QThread.msleep(1000)
        
        self.current_index += 1
        if self.current_index < len(self.current_flashcards):
            self.show_flashcard()
        else:
            self.save_history()
            QMessageBox.information(self, "Complete", 
                f"Study session complete!\nFinal score: {self.correct_count} correct, {self.wrong_count} wrong")

    def on_flashcards_generated(self, flashcards):
        if self.progress:
            self.progress.setValue(100)
            self.progress.close()
            self.progress = None

        self.current_flashcards = flashcards
        self.current_index = 0
        self.correct_count = 0
        self.wrong_count = 0
        
        if flashcards:
            self.show_flashcard()
            self.tabs.setCurrentIndex(1)  # Switch to Study tab
        else:
            QMessageBox.warning(self, "Warning", "No flashcards were generated. Please try again.")

    def save_history(self):
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
            except:
                pass
        
        # Add new flashcards to history
        history.extend(self.current_flashcards)
        
        with open(self.history_file, "w") as f:
            json.dump(history, f)

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
                    formatted_history = "\n\n".join(
                        f"Q: {entry['question']}\nA: {entry['options'][entry['correct_index']]}"
                        for entry in history
                    )
                    self.history_text.setPlainText(formatted_history)
            except Exception as e:
                print(f"Error loading history: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlashcardApp()
    window.show()
    sys.exit(app.exec_())
