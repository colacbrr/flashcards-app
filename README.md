# flashcards-app

Pași pentru Windows cu bash:
1. Clonează repository-ul:
git clone https://github.com/colacbrr/flashcards-app.git
cd flashcards-app

3. Verifică Python și pip:
python --version
pip --version

# Dacă nu ai Python, descarcă de pe python.org
3. Installează dependențele Python:
pip install PyQt5 pymupdf python-docx aiohttp ollama
4. Installează Ollama pe Windows:

Mergi pe ollama.com
Descarcă "Ollama for Windows"
Rulează installer-ul
După instalare, deschide Command Prompt sau PowerShell și verifică:

ollama --version

5. Descarcă un model AI:
ollama pull mistral

(Poate dura câteva minute, modelul are ~4GB)

7. Pornește Ollama (dacă nu pornește automat):
ollama serve

Lasă terminalul ăsta deschis - Ollama trebuie să ruleze în background.

9. În alt terminal, pornește aplicația:
cd flashcards-app
python main.py

(sau cum se numește fișierul principal - probabil flashcard_app.py sau similar)

Structura proiectului ar trebui să arate așa:
flashcards-app/
├── main.py (sau flashcard_app.py)
├── requirements.txt (poate)
├── README.md
└── alte fișiere...
Tips pentru Windows:

Dacă folosești Git Bash și ai probleme cu PyQt5, încearcă din Command Prompt normal
Pentru WSL: Ai nevoie de X server (ca Xming) pentru interfața grafică
Dacă Ollama nu pornește: Verifică să nu fie blocat de antivirus/firewall

Testare rapidă fără Ollama:
Dacă vrei să testezi interfața fără să aștepți să se instaleze Ollama:

Deschide fișierul Python
Schimbă USE_MOCK = False în USE_MOCK = True
Rulează aplicația - va genera flashcards mock pentru testare

