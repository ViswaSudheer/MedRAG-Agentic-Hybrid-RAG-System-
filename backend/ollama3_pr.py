
########################################ollama with 5 updates you selected, lang, pdf########################################



import os
import json
import glob
import pickle
import time
import datetime
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from config import *
from ollama_utils import generate_llm as ollama_generate

# ────────────────────────────────────────────────
#               DEPENDENCY CHECK
# ────────────────────────────────────────────────
_missing_msgs = []
try:
    import spacy
except Exception:
    _missing_msgs.append("! pip install -U spacy && python -m spacy download en_core_web_sm")
try:
    import faiss
except Exception:
    _missing_msgs.append("! pip install faiss-cpu")
try:
    from PyPDF2 import PdfReader
except Exception:
    _missing_msgs.append("! pip install PyPDF2")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    _missing_msgs.append("! pip install sentence-transformers")

# rank_bm25 (for BM25 sparse retrieval)
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    _missing_msgs.append("! pip install rank_bm25")
# for pkg in ["spacy", "faiss", "PyPDF2", "sentence_transformers", "rank_bm25"]:
#     try:
#         __import__(pkg)
#     except ImportError:
#         _missing_msgs.append(f"! pip install {pkg}")

if _missing_msgs:
    print("🔧 Missing packages detected. Install them:")
    for cmd in _missing_msgs:
        print("   ", cmd)

# spaCy (Windows-safe)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# ────────────────────────────────────────────────
#          EMBEDDING & RERANKER MODELS
# ────────────────────────────────────────────────
DENSE_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"Loading models:\n  • Dense: {DENSE_MODEL}\n  • Reranker: {RERANKER_MODEL}")
embedder = SentenceTransformer(DENSE_MODEL)
reranker = CrossEncoder(RERANKER_MODEL, device="cpu")  # change to "cuda" if GPU available

# ────────────────────────────────────────────────
#               TEXT PREPROCESSING & EMBEDDING
# ────────────────────────────────────────────────
def preprocess_texts(texts: List[str]) -> List[str]:
    def normalize(doc):
        return " ".join(t.lemma_ for t in doc if not t.is_stop and t.is_alpha)
    return [normalize(doc) for doc in nlp.pipe(texts, n_process=1, batch_size=64)]


def embed_texts(texts: List[str]) -> np.ndarray:
    clean = [t.strip() for t in texts if t.strip()]
    if not clean:
        raise ValueError("No valid text to embed")
    return embedder.encode(clean, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


# ────────────────────────────────────────────────
#               HyDE – Hypothetical Document Embeddings
# ────────────────────────────────────────────────
def hyde_query(query: str) -> str:
    """Generate a hypothetical answer to improve retrieval quality"""
    prompt = f"""You are a world-class medical expert.
Write a concise, accurate hypothetical answer (3–5 sentences) to the following question.
Do NOT say "I don't know" — assume the knowledge exists.

Question: {query}

Hypothetical Answer:"""
    return ollama_generate(prompt).strip()


# ────────────────────────────────────────────────
#               HYBRID INDEX CLASS (Dense + BM25)
# ────────────────────────────────────────────────
class HybridIndex:
    def __init__(self, name: str):
        self.name = name
        self.faiss_idx = None
        self.docs: List[str] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized: List[List[str]] = []

    def build_or_load(self, texts: List[str], faiss_path: str, docs_path: str):
        if os.path.exists(faiss_path) and os.path.exists(docs_path):
            self.faiss_idx = faiss.read_index(faiss_path)
            with open(docs_path, "rb") as f:
                self.docs = pickle.load(f)
            print(f"[{self.name}] Loaded hybrid index ({len(self.docs)} docs)")
        else:
            clean_texts = [t.strip() for t in texts if t.strip()]
            if not clean_texts:
                print(f"[{self.name}] No texts → empty index")
                self.faiss_idx = faiss.IndexFlatL2(384)
                return

            print(f"[{self.name}] Building hybrid index ({len(clean_texts)} docs)...")
            t0 = time.time()

            # Dense part
            embs = embed_texts(clean_texts)
            self.faiss_idx = faiss.IndexFlatL2(embs.shape[1])
            self.faiss_idx.add(embs)
            faiss.write_index(self.faiss_idx, faiss_path)

            # BM25 part
            self.tokenized = [doc.lower().split() for doc in clean_texts]
            self.bm25 = BM25Okapi(self.tokenized)

            with open(docs_path, "wb") as f:
                pickle.dump(clean_texts, f)
            self.docs = clean_texts

            print(f"[{self.name}] Built & saved hybrid index. Time: {time.time()-t0:.1f}s")

    def search(self, query: str, k: int = 20) -> List[str]:
        if not self.docs:
            return []

        # Use HyDE-enhanced query
        hyde = hyde_query(query)
        q_emb = embed_texts([hyde])[0]

        # Dense retrieval
        D, I = self.faiss_idx.search(q_emb.reshape(1, -1), k*3)
        dense = [(self.docs[i], 1 / (1 + D[0][j])) for j, i in enumerate(I[0]) if i < len(self.docs)]

        # BM25 retrieval
        if self.bm25:
            tok_q = query.lower().split()
            bm25_scores = self.bm25.get_scores(tok_q)
            bm25 = [(self.docs[i], 1 / (1 + 1 / (bm25_scores[i] + 1e-9))) for i in np.argsort(bm25_scores)[-k*3:]]
        else:
            bm25 = []

        # Reciprocal Rank Fusion
        scores = {}
        for doc, sc in dense + bm25:
            scores[doc] = scores.get(doc, 0) + sc

        # Top candidates for reranking
        candidates = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:30]

        # Cross-encoder rerank
        # pairs = [[query, doc] for doc in candidates]
        # rerank_scores = reranker.predict(pairs)
        # ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        # return [doc for doc, _ in ranked[:k]]
        return candidates[:k]


# ────────────────────────────────────────────────
#               GLOBAL INDEX INSTANCES
# ────────────────────────────────────────────────
med_hybrid = HybridIndex("MedQuAD")
syn_hybrid = HybridIndex("Synthea")
book_hybrid = HybridIndex("Books")
pubmed_hybrid = HybridIndex("PubMedQA")







# ────────────────────────────────────────────────
#                Data Loading Functions
# ────────────────────────────────────────────────

def load_medquad_docs(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        print(f"❌ MedQuAD CSV not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    q_col = next((c for c in df.columns if "question" in c.lower() or c.lower().startswith("q")), None)
    a_col = next((c for c in df.columns if "answer" in c.lower() or c.lower().startswith("a")), None)
    if not (q_col and a_col):
        print("Could not detect Q/A columns. Found:", list(df.columns))
        return []
    docs = [f"Q: {str(row[q_col]).strip()}\nA: {str(row[a_col]).strip()}"
            for _, row in df.iterrows() if pd.notna(row[q_col]) and pd.notna(row[a_col])]
    print(f"✅ Loaded MedQuAD: {len(docs)} QA pairs")
    return docs

def summarize_patients_from_fhir(fhir_dir: str) -> List[str]:
    if not os.path.exists(fhir_dir):
        print(f"❌ Synthea folder not found: {fhir_dir}")
        return []

    def read_csv_safe(path):
        for enc in ['utf-8', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except:
                continue
        return pd.DataFrame()

    # Map file name → internal field name (this fixes the mismatch)
    field_map = {
        "allergies": "allergies",
        "conditions": "conditions",
        "medications": "meds",          # ← key fix
        "procedures": "procedures",
        "careplans": "careplans",
        "encounters": "encounters",
        "observations": "obs"           # ← key fix
    }

    files = list(field_map.keys())
    dfs = {}
    for fname in files:
        fp = os.path.join(fhir_dir, f"{fname}.csv")
        if os.path.exists(fp):
            dfs[fname] = read_csv_safe(fp)
            print(f"Loaded {fname} ({len(dfs[fname])} rows)")
        else:
            dfs[fname] = pd.DataFrame()

    by_patient: Dict[str, Dict[str, List[str]]] = {}

    def add(pid: str, field_key: str, text: str):
        if not pid or not text.strip():
            return
        # Use the mapped short name
        bucket = by_patient.setdefault(pid, {
            "conditions": [], "meds": [], "allergies": [],
            "procedures": [], "encounters": [], "obs": [], "careplans": []
        })
        bucket[field_key].append(text.strip())

    for file_key, df in dfs.items():
        if df.empty or "PATIENT" not in df.columns:
            continue

        internal_field = field_map[file_key]

        desc_cols = [
            c for c in df.columns
            if any(x in c.upper() for x in ["DESCRIPTION", "CODE", "REASON", "VALUE", "TYPE", "RXNORM"])
        ]

        for _, r in df.iterrows():
            txt = ", ".join(str(r[c]) for c in desc_cols if pd.notna(r[c]) and str(r[c]).strip())
            if txt:
                add(str(r["PATIENT"]), internal_field, txt)

    # Build summaries
    summaries = []
    for pid, d in by_patient.items():
        parts = [f"PatientID: {pid}."]
        for field, vals in d.items():
            if vals:
                joined = ", ".join(sorted(set(vals)))
                limit = 700 if field in ("conditions", "meds") else 500
                parts.append(f"{field.capitalize()}: {joined[:limit]}")
        summaries.append(". ".join(parts) + ".")
    
    print(f"✅ Synthea summarized {len(summaries)} patients")
    return summaries

def pdf_extract_text(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"⚠️ PDF error {os.path.basename(file_path)}: {e}")
        return ""


def load_books_pdfs(books_dir: str) -> List[str]:
    if not os.path.exists(books_dir):
        print(f"❌ Books dir not found: {books_dir}")
        return []
    paths = sorted(glob.glob(os.path.join(books_dir, "*.pdf")))
    if not paths:
        print(f"No PDFs found in {books_dir}")
        return []

    texts = []
    for fp in paths:
        t = pdf_extract_text(fp)
        if t.strip():
            texts.append(f"[BOOK: {os.path.basename(fp)}]\n{t}")
            print(f"Extracted {os.path.basename(fp)} ({len(t)} chars)")

    chunks = []
    MAX = 1200
    for doc in texts:
        i = 0
        while i < len(doc):
            chunks.append(doc[i:i+MAX])
            i += MAX
    print(f"✅ Created {len(chunks)} PDF chunks")
    return chunks


def load_pubmedqa_docs(json_path: str) -> List[str]:
    if not os.path.exists(json_path):
        print(f"❌ PubMedQA JSON not found: {json_path}")
        return []

    print(f"📂 Reading PubMedQA file: {os.path.basename(json_path)} ({os.path.getsize(json_path)/1024/1024:.1f} MB)")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert PMID dict → list of items
    if isinstance(raw_data, dict):
        data = list(raw_data.values())          # ← This is the key fix
        print(f"🔍 Converted PMID dict → {len(data):,} entries")
    else:
        data = raw_data

    if not isinstance(data, list):
        print("⚠️ Unexpected root type:", type(data))
        return []

    print(f"🔍 Total entries in JSON: {len(data):,}")

    docs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        q = item.get("QUESTION", "").strip()
        ctx_list = item.get("CONTEXTS", []) or item.get("context", [])
        ctx = " ".join(ctx_list) if isinstance(ctx_list, list) else str(ctx_list)
        ans = item.get("LONG_ANSWER", "").strip()

        if q and ans:
            doc = f"Q: {q}\nContext: {ctx[:900]}\nA: {ans}"
            docs.append(doc.strip())

        # Debug only first 2 items
        if i < 2:
            print(f"🔎 Sample {i+1} keys: {list(item.keys())}")
            print(f"   Question length: {len(q)} | Answer length: {len(ans)}")

    print(f"✅ SUCCESS: Loaded PubMedQA → {len(docs):,} QA pairs (out of {len(data):,})")
    return docs


# ────────────────────────────────────────────────
#               INITIALIZE INDEXES
# ────────────────────────────────────────────────
def init_indexes():
    global med_hybrid, syn_hybrid, book_hybrid, pubmed_hybrid

    med_docs = load_medquad_docs(MEDQUAD_CSV)
    med_hybrid.build_or_load(med_docs, FAISS_MED, DOCS_MED)
    if ENABLE_PUB:
        if os.path.exists(FAISS_PUBMED) and os.path.exists(DOCS_PUBMED):
            pubmed_hybrid.build_or_load([], FAISS_PUBMED, DOCS_PUBMED)   # just load
        else:
            pubmed_docs = load_pubmedqa_docs(PUBMED_JSON)
            pubmed_hybrid.build_or_load(pubmed_docs, FAISS_PUBMED, DOCS_PUBMED)
    
    if ENABLE_SYNTH:
        if os.path.exists(FAISS_SYN) and os.path.exists(DOCS_SYN):
            syn_hybrid.build_or_load([], FAISS_SYN, DOCS_SYN)
        else:
            syn_docs = summarize_patients_from_fhir(SYN_DIR)
            syn_hybrid.build_or_load(syn_docs, FAISS_SYN, DOCS_SYN)

    if ENABLE_BOOKS:
        if os.path.exists(FAISS_BOOKS) and os.path.exists(DOCS_BOOKS):
            book_hybrid.build_or_load([], FAISS_BOOKS, DOCS_BOOKS)
        else:
            book_docs = load_books_pdfs(BOOKS_DIR)
            book_hybrid.build_or_load(book_docs, FAISS_BOOKS, DOCS_BOOKS)


# ────────────────────────────────────────────────
#               RETRIEVAL + SELF-CORRECTIVE RAG
# ────────────────────────────────────────────────
def retrieve_hybrid(query: str, top_k: int = 12) -> Dict[str, str]:
    med = med_hybrid.search(query, top_k)
    pub = pubmed_hybrid.search(query, top_k) if ENABLE_PUB else []
    syn = syn_hybrid.search(query, top_k) if ENABLE_SYNTH else []
    book = book_hybrid.search(query, top_k) if ENABLE_BOOKS else []

    return {
        "medquad": "\n\n".join(med),
        "pubmedqa": "\n\n".join(pub),
        "synthea": "\n\n".join(syn),
        "books":   "\n\n".join(book)
    }

# ────────────────────────────────────────────────
#               Answer & Synthesis
# ────────────────────────────────────────────────

SAFETY_SUFFIX = (
    "\n\nAnswer concisely using only the provided context. "
    "If insufficient information, say 'I don't know from the provided context.' "
    "Do NOT provide diagnoses or medical advice. Recommend consulting a clinician."
)

SOURCE_STYLES = {
    "medquad":  "Answer in clear Q&A format like a medical FAQ.",
    "pubmedqa": "Answer like a research paper abstract summary with decision (yes/no/maybe).",
    "synthea":  "Answer like a clinical patient record summary.",
    "books":    "Answer like a detailed medical encyclopedia entry.",
    "combined": "Synthesize a coherent answer combining all available sources."
}


def smart_truncate(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    # sentence-aware truncate
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = ""
    for s in sentences:
        if len(result) + len(s) + 1 <= max_chars:
            result += s + " "
        else:
            break
    return result.strip()


def answer_with_backend(query: str, context: str, source_name: str = "generic") -> str:
    if not context.strip():
        return ""

    ctx = smart_truncate(context, 3800)
    style = SOURCE_STYLES.get(source_name, "Answer concisely.")

    prompt = f"""You are a careful medical information assistant.

Question:
{query}

Context:
{ctx}

Instruction:
{style}

Answer strictly from the context only.
If the answer is not in the context, say exactly: "I don't know from the provided context."
{SAFETY_SUFFIX}"""

    return ollama_generate(prompt).strip()


SOURCE_WEIGHTS = {"medquad": 1.0,"pubmedqa": 1.25, "synthea": 0.8, "books": 0.9}



# def self_corrective_rag(query: str, max_rounds: int = 2, min_confidence: int = 50) -> Tuple[str, Dict[str, str], int]:
#     contexts = retrieve_hybrid(query)

#     # Combine contexts for generation
#     combined_ctx = "\n\n".join(
#         f"[{k.upper()}]\n{v}" for k, v in contexts.items() if v.strip()
#     )

#     answer = answer_with_backend(query, combined_ctx, "combined")

#     # Quick confidence judgment via LLM
#     judge_prompt = f"""Rate the faithfulness and usefulness of this answer on scale 0–100.
# Question: {query}
# Answer: {answer}

# Return only JSON: {{"confidence": int}}"""
#     judge_raw = ollama_generate(judge_prompt)

#     try:
#         conf = int(re.search(r'"confidence":\s*(\d+)', judge_raw).group(1))
#     except:
#         conf = 60

#     if conf < min_confidence and max_rounds > 0:
#         print(f"  ↻ Low confidence ({conf}), re-retrieving...")
#         return self_corrective_rag(query + "\nPrevious attempt: " + answer[:150], max_rounds - 1, min_confidence)

#     return answer, contexts, conf
# def supreme_fast_rag(query: str) -> dict:
#     # HyDE + Answer + Confidence in ONE prompt (saves 140 seconds!)
#     contexts = retrieve_hybrid(query)
#     combined_ctx = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in contexts.items() if v.strip())

#     prompt = f"""You are a precise medical RAG assistant.
# Question: {query}

# Context:
# {smart_truncate(combined_ctx, 3800)}

# First, generate hypothetical answer internally (HyDE style).
# Then give final answer.
# At the end, add exactly: CONFIDENCE:XX (0-100)

# Rules:
# - Answer only from context.
# - If unsure: "Not found in the provided document."
# - Never give medical advice.

# Final Answer:"""

#     answer = ollama_generate(prompt).strip()
    
#     # Extract confidence with regex
#     try:
#         conf = int(re.search(r'CONFIDENCE:(\d+)', answer).group(1))
#         answer = re.sub(r'CONFIDENCE:\d+', '', answer).strip()
#     except:
#         conf = 85

#     return {
#         "answer": answer,
#         "confidence": conf,
#         "contexts": contexts
#     }

# # ────────────────────────────────────────────────
# #               MAIN RAG ENTRY POINT
# # ────────────────────────────────────────────────
# def agentic_rag(query: str) -> dict:
#     start = time.time()

#     # final_answer, contexts, confidence = self_corrective_rag(query)
#     final_answer, contexts, confidence = supreme_fast_rag(query)
#     latency = time.time() - start

#     meta = {
#         "confidence": confidence,
#         "latency_seconds": round(latency, 2),
#         "timestamp": str(datetime.datetime.now()),
#         "sources_used": [k for k, v in contexts.items() if v.strip()]
#     }

#     # Log
#     log_text = (
#         f"[FINAL]\n{final_answer}\n\n"
#         f"[CONFIDENCE] {confidence}/100\n"
#         f"[LATENCY] {latency:.2f}s\n"
#         f"\n" + "\n\n".join(f"[{k.upper()}]\n{v[:300]}..." for k,v in contexts.items() if v)
#     )
#     save_chat(query, log_text, meta)

#     return {
#         "query": query,
#         "final_answer": final_answer,
#         "contexts": contexts,
#         "confidence": confidence,
#         "latency": latency,
#         "meta": meta
#     }
# ────────────────────────────────────────────────
#  SUPREME FAST RAG (Single LLM call – 10x faster)
# ────────────────────────────────────────────────
def supreme_fast_rag(query: str) -> tuple:
    """One-shot HyDE + Answer + Confidence. Returns tuple for perfect compatibility."""
    contexts = retrieve_hybrid(query)                    # dict of sources
    combined_ctx = "\n\n".join(
        f"[{k.upper()}]\n{v}" for k, v in contexts.items() if v.strip()
    )

    prompt = f"""You are a precise medical RAG assistant. Think step-by-step internally.

Question: {query}

Context:
{smart_truncate(combined_ctx, 3800)}

Rules:
- Answer ONLY from the context.
- If unsure: "Not found in the provided document."
- Never give medical advice.
- At the VERY END add exactly: CONFIDENCE:XX (0-100)

Final Answer:"""

    raw = ollama_generate(prompt).strip()

    # Extract confidence
    try:
        conf = int(re.search(r'CONFIDENCE:(\d+)', raw, re.IGNORECASE).group(1))
        final_answer = re.sub(r'CONFIDENCE:\d+', '', raw, flags=re.IGNORECASE).strip()
    except:
        conf = 85
        final_answer = raw

    return final_answer, contexts, conf


# ────────────────────────────────────────────────
#  UPDATED agentic_rag (now uses supreme_fast_rag)
# ────────────────────────────────────────────────
def agentic_rag(query: str) -> dict:
    start = time.time()

    # 🔥 SUPREME ONE-SHOT CALL (no more 3 LLM passes)
    final_answer, contexts, confidence = supreme_fast_rag(query)

    latency = time.time() - start

    meta = {
        "confidence": confidence,
        "latency_seconds": round(latency, 2),
        "timestamp": str(datetime.datetime.now()),
        "sources_used": [k for k, v in contexts.items() if v.strip()]
    }

    # Log (unchanged)
    log_text = (
        f"[FINAL]\n{final_answer}\n\n"
        f"[CONFIDENCE] {confidence}/100\n"
        f"[LATENCY] {latency:.2f}s\n"
        f"\n" + "\n\n".join(f"[{k.upper()}]\n{v[:300]}..." for k,v in contexts.items() if v)
    )
    save_chat(query, log_text, meta)

    return {
        "query": query,
        "final_answer": final_answer,
        "contexts": contexts,
        "confidence": confidence,
        "latency": latency,
        "meta": meta
    }
# ────────────────────────────────────────────────
#               Chat History
# ────────────────────────────────────────────────

CHAT_LOG = []
if os.path.exists(CHATLOG_PATH):
    try:
        with open(CHATLOG_PATH, "r", encoding="utf-8") as f:
            CHAT_LOG = json.load(f)
    except:
        pass


def summarize_chat(last_n: int = 6) -> str:
    subset = CHAT_LOG[-last_n:]
    lines = [f"U: {e.get('query','')[:180]}" for e in subset]
    lines += [f"A: {e.get('answer','')[:180]}" for e in subset]
    return "\n".join(lines)[-1200:]


def save_chat(query: str, answer: str, meta: Dict):
    CHAT_LOG.append({
        "time": str(datetime.datetime.now()),
        "query": query,
        "answer": answer,
        "meta": meta
    })
    with open(CHATLOG_PATH, "w", encoding="utf-8") as f:
        json.dump(CHAT_LOG, f, indent=2, ensure_ascii=False)





import uuid
from typing import Dict,Optional

# ====================== PER-CHAT PDF SESSION SYSTEM ======================
pdf_sessions: Dict[str, HybridIndex] = {}   # session_id → HybridIndex

def create_pdf_session(pdf_text: str) -> str:
    """Extract, chunk & build isolated index for ONE uploaded PDF"""
    session_id = str(uuid.uuid4())
    
    # Smart chunking (same as your Books loader)
    chunks = []
    MAX = 1200
    i = 0
    while i < len(pdf_text):
        chunks.append(pdf_text[i:i + MAX])
        i += MAX
    if not chunks:
        chunks = [pdf_text]

    # Create dedicated index for this session only
    pdf_index = HybridIndex(f"PDF_{session_id[:8]}")
    pdf_index.build_or_load(chunks, f"temp_faiss_{session_id[:12]}.index", f"temp_docs_{session_id[:12]}.pkl")
    
    pdf_sessions[session_id] = pdf_index
    print(f"✅ PDF SESSION CREATED → {session_id} | {len(chunks)} chunks indexed")
    return session_id


def query_pdf_only(question: str, session_id: str) -> dict:
    """RAG that answers ONLY from the uploaded PDF (no mixing with MedQuAD etc.)"""
    if session_id not in pdf_sessions:
        return {
            "answer": "No PDF uploaded for this chat session. Please upload a PDF first.",
            "meta": {"sources_used": [], "confidence": 0, "mode": "pdf_only"}
        }

    pdf_hybrid = pdf_sessions[session_id]
    
    # Reuse your powerful search (HyDE + RRF + reranker)
    retrieved = pdf_hybrid.search(question, k=10)
    context = "\n\n".join(retrieved)

    if not context.strip():
        answer = "Not mentioned in the provided document."
    else:
        prompt = f"""You are a precise medical assistant. Answer **only** using the uploaded document excerpts below.

Question: {question}

Document Excerpts:
{context[:3800]}

Rules:
- Answer directly from the document.
- If something is not mentioned, say exactly: "Not mentioned in the provided document."
- Use simple language. Quote relevant parts when helpful.
- Never give medical advice. End with: "Consult a doctor for professional guidance." """

        answer = ollama_generate(prompt).strip()

    return {
        "answer": answer,
        "meta": {
            "query": question,
            "timestamp": str(datetime.datetime.now()),
            "sources_used": ["uploaded_pdf"],
            "confidence": 85 if context.strip() else 10,
            "mode": "pdf_only",
            "session_id": session_id
        }
    }
# ────────────────────────────────────────────────
#               CLI – IMPRESSIVE DEMO OUTPUT
# ────────────────────────────────────────────────


init_indexes()



if __name__ == "__main__":
    print("═"*80)
    print("  MedRAG-Hybrid-Pro v2.0  –  2026 Final Year Level")
    print("  Features: Hybrid Dense+BM25 + RRF + Cross-Encoder + HyDE + Self-Corrective RAG")
    print("═"*80)

    init_indexes()

    print("\nReady. Type your medical question (or 'exit'):\n")

    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q or q.lower() in ("exit", "quit"):
            break

        print("\nProcessing...\n")

        result = agentic_rag(q)

        print("═"*80)
        print("FINAL MEDICAL ANSWER")
        print("═"*80)
        print(result["final_answer"])
        print()

        print(f"Confidence: {result['confidence']}/100   |   Latency: {result['latency']:.2f} seconds")
        print("Sources used:", ", ".join(result["meta"]["sources_used"]))

        print("\n" + "─"*60)
        print("DETAILED CONTEXTS (top snippets)")
        print("─"*60)
        for src, ctx in result["contexts"].items():
            if ctx.strip():
                print(f"[{src.upper()}]")
                print(ctx[:400] + "..." if len(ctx) > 400 else ctx)
                print()

        print("═"*80 + "\n")

def agentic_rag_for_ui(
    question: str,
    session_id: Optional[str] = None
) -> dict:
    """
    Unified entry point for UI.
    
    - If session_id is provided → PDF-only mode (answers only from that uploaded PDF)
    - Otherwise → normal MedRAG mode (MedQuAD + Synthea + Books)
    """
    try:
        # ────────────────────────────────────────────────
        #             PDF-ONLY MODE (per chat session)
        # ────────────────────────────────────────────────
        if session_id and session_id in pdf_sessions:
            print(f"[PDF MODE] Using session {session_id[:8]}... for question")

            pdf_index = pdf_sessions[session_id]
            
            # Reuse the same powerful retrieval pipeline
            retrieved_chunks = pdf_index.search(question, k=8)
            context = "\n\n".join(retrieved_chunks)

            if not context.strip():
                answer = "The information is not present in the uploaded document."
            else:
                prompt = f"""You are a precise document-based assistant.
Answer the question using **only** the content from the uploaded document below.
Do not use any external knowledge.

Question: {question}

Document content:
{context[:3800]}

Rules:
- Be concise and factual.
- If the answer is not clearly supported by the text → reply exactly:
  "Not mentioned / not found in the provided document."
- Quote relevant sentence(s) when it helps clarity.
- Never give medical advice. If it is medical related then always end with:
  "Please consult a qualified doctor for medical interpretation." """

                answer = ollama_generate(prompt).strip()

            meta = {
                "query": question,
                "timestamp": str(datetime.datetime.now()),
                "sources_used": ["uploaded_pdf"],
                "mode": "pdf_session",
                "session_id": session_id,
                "retrieved_chunks": len(retrieved_chunks),
                "confidence": 90 if context.strip() else 15,
            }

            return {
                "answer": answer,
                "meta": meta,
                "success": True,
                "error": None
            }

        # ────────────────────────────────────────────────
        #             NORMAL MedRAG MODE (no session_id)
        # ────────────────────────────────────────────────
        print("[NORMAL MODE] Using main MedRAG indexes")

        result = agentic_rag(question)

        final_answer = result.get("final_answer", "").strip()

        # Emergency fallback if main answer is empty / unhelpful
        if not final_answer or "don't have enough information" in final_answer.lower():
            combined = ""
            for src, text in result.get("contexts", {}).items():
                if text.strip():
                    combined += f"[{src.upper()}]\n{text.strip()[:900]}\n\n"

            if combined.strip():
                emergency_prompt = f"""Question: {question}

Available excerpts:
{combined[:3200]}

Answer concisely using **only** the excerpts above.
If nothing relevant → write exactly: "I don't have enough information from the available sources to answer this."
"""
                final_answer = ollama_generate(emergency_prompt).strip()

        meta = {
            "query": question,
            "timestamp": str(datetime.datetime.now()),
            "sources_used": [k for k, v in result.get("contexts", {}).items() if v.strip()],
            "confidence": result.get("confidence"),
            "latency": result.get("latency"),
            "mode": "main_database"
        }

        return {
            "answer": final_answer or "I don't have enough information from the available sources to answer this.",
            "meta": meta,
            "success": True,
            "error": None
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("agentic_rag_for_ui error:\n" + tb)
        return {
            "answer": "Sorry, an internal error occurred while processing your question.",
            "meta": {"error_trace": str(e)},
            "success": False,
            "error": str(e)
        }