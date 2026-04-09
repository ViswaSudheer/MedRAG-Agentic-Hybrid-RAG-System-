########################################ollama with 5 updates you selected, lang########################################



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
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:k]]


# ────────────────────────────────────────────────
#               GLOBAL INDEX INSTANCES
# ────────────────────────────────────────────────
med_hybrid = HybridIndex("MedQuAD")
syn_hybrid = HybridIndex("Synthea")
book_hybrid = HybridIndex("Books")








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





# ────────────────────────────────────────────────
#               INITIALIZE INDEXES
# ────────────────────────────────────────────────
def init_indexes():
    global med_hybrid, syn_hybrid, book_hybrid

    med_docs = load_medquad_docs(MEDQUAD_CSV)
    med_hybrid.build_or_load(med_docs, FAISS_MED, DOCS_MED)

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
    syn = syn_hybrid.search(query, top_k) if ENABLE_SYNTH else []
    book = book_hybrid.search(query, top_k) if ENABLE_BOOKS else []

    return {
        "medquad": "\n\n".join(med),
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


SOURCE_WEIGHTS = {"medquad": 1.0, "synthea": 0.8, "books": 0.9}



def self_corrective_rag(query: str, max_rounds: int = 2, min_confidence: int = 50) -> Tuple[str, Dict[str, str], int]:
    contexts = retrieve_hybrid(query)

    # Combine contexts for generation
    combined_ctx = "\n\n".join(
        f"[{k.upper()}]\n{v}" for k, v in contexts.items() if v.strip()
    )

    answer = answer_with_backend(query, combined_ctx, "combined")

    # Quick confidence judgment via LLM
    judge_prompt = f"""Rate the faithfulness and usefulness of this answer on scale 0–100.
Question: {query}
Answer: {answer}

Return only JSON: {{"confidence": int}}"""
    judge_raw = ollama_generate(judge_prompt)

    try:
        conf = int(re.search(r'"confidence":\s*(\d+)', judge_raw).group(1))
    except:
        conf = 60

    if conf < min_confidence and max_rounds > 0:
        print(f"  ↻ Low confidence ({conf}), re-retrieving...")
        return self_corrective_rag(query + "\nPrevious attempt: " + answer[:150], max_rounds - 1, min_confidence)

    return answer, contexts, conf


# ────────────────────────────────────────────────
#               MAIN RAG ENTRY POINT
# ────────────────────────────────────────────────
def agentic_rag(query: str) -> dict:
    start = time.time()

    final_answer, contexts, confidence = self_corrective_rag(query)

    latency = time.time() - start

    meta = {
        "confidence": confidence,
        "latency_seconds": round(latency, 2),
        "timestamp": str(datetime.datetime.now()),
        "sources_used": [k for k, v in contexts.items() if v.strip()]
    }

    # Log
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


def agentic_rag_for_ui(question: str) -> dict:
    try:
        # init_indexes()
        result = agentic_rag(question)

        final_answer = result.get("final_answer", "").strip()

        # If main pipeline gave nothing useful → try emergency fallback
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

        # Build metadata
        meta = {
            "query": question,
            "timestamp": str(datetime.datetime.now()),
            "sources_used": [k for k, v in result.get("contexts", {}).items() if v.strip()],
            "confidence": result.get("confidence"),
            "latency": result.get("latency"),
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
        print(tb)
        return {
            "answer": "Sorry, internal processing error occurred.",
            "meta": {"error_trace": str(e)},
            "success": False,
            "error": str(e)
        }
    
