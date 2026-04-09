import os

# ============== CONFIG FLAGS (toggle datasets) ==============
ENABLE_SYNTH = True     # enable Synthea CSV ingestion
ENABLE_BOOKS = True      # enable Books (PDF) ingestion
ENABLE_PUB= True 
# ============== PATHS & SAVE LOCATIONS ==============
# Put these in Google Drive for persistence in Colab:
# ---------------- LOCAL PATHS (VS CODE MODE) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

BASE = PROJECT_ROOT  # main ragbot folder

DATA_DIR = os.path.join(BASE, "data")
INDEX_DIR = os.path.join(BASE, "index_store", "indexes")
LOG_DIR = os.path.join(BASE, "index_store", "logs")

for p in (DATA_DIR, INDEX_DIR, LOG_DIR):
    os.makedirs(p, exist_ok=True)


# ---------- MedQuAD ----------
MED_DIR = os.path.join(DATA_DIR, "medquad")
os.makedirs(MED_DIR, exist_ok=True)
MEDQUAD_CSV = os.path.join(MED_DIR, "medquad.csv")
FAISS_MED = os.path.join(INDEX_DIR, "faiss_medquAD.bin")
DOCS_MED = os.path.join(INDEX_DIR, "doc_store_medquAD.pkl")

# ---------- Synthea CSV ----------
SYN_DIR = os.path.join(DATA_DIR, "synthea/csv")
os.makedirs(SYN_DIR, exist_ok=True)
FAISS_SYN = os.path.join(INDEX_DIR, "faiss_synthea.bin")
DOCS_SYN = os.path.join(INDEX_DIR, "doc_store_synthea.pkl")

# ---------- Books PDFs ----------
BOOKS_DIR = os.path.join(DATA_DIR, "books")
os.makedirs(BOOKS_DIR, exist_ok=True)
FAISS_BOOKS = os.path.join(INDEX_DIR, "faiss_books.bin")
DOCS_BOOKS = os.path.join(INDEX_DIR, "doc_store_books.pkl")

# ---------- PubMedQA (61k QA) ----------
PUBMED_DIR = os.path.join(DATA_DIR, "pubmed")
os.makedirs(PUBMED_DIR, exist_ok=True)
PUBMED_JSON = os.path.join(PUBMED_DIR, "ori_pqau.json")
FAISS_PUBMED = os.path.join(INDEX_DIR, "faiss_pubmed.bin")
DOCS_PUBMED = os.path.join(INDEX_DIR, "doc_store_pubmed.pkl")


CHATLOG_PATH = os.path.join(LOG_DIR, "chat_history.json")



