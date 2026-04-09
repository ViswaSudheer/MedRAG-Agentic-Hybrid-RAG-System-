#####################################################1st ollama usage###############################################
import os
# from dotenv import load_dotenv
# load_dotenv()

import json
import glob
import pickle
import time
import datetime
import re
import multiprocessing
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


from config import *
from ollama_utils import generate_llm as ollama_generate
# from ollama_utils import generate_embedding as ollama_embedding
# from ollama_utils import safe_embedding
# from ollama_utils import generate_embeddings_batch
# ---------- Optional deps guard (spaCy + PDF) ----------



_missing_msgs = []
try:
    import spacy
except Exception:
    _missing_msgs.append("! pip install -U spacy && python -m spacy download en_core_web_sm")
try:
    import faiss
except Exception:
    _missing_msgs.append("! pip install faiss-cpu")


if _missing_msgs:
    print("🔧 Missing packages detected. If you see errors later, run:")
    for cmd in _missing_msgs:
        print("   ", cmd)

use_iterative = True
# ============== Metrics Logging ==============
metrics_log: list = []


# ============== spaCy (preprocessing) ==============
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception:
    raise RuntimeError("spaCy 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")



EMBED_MODEL = "all-MiniLM-L6-v2"
print("Loading embedding model:", EMBED_MODEL)

embedder = SentenceTransformer(EMBED_MODEL)


def preprocess_texts(texts: List[str]) -> List[str]:
    """Lowercase, lemmatize, remove stopwords/non-alpha using spaCy pipe."""
    
    def normalize(doc):
        return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])
    
    # Windows-safe
    n_cores = 1
    return [normalize(doc) for doc in nlp.pipe(texts, n_process=n_cores, batch_size=64)]

def embed_texts(texts: List[str]) -> np.ndarray:
    """Fast local embeddings using SentenceTransformer"""

    clean_texts = [t.strip() for t in texts if t.strip()]

    if not clean_texts:
        raise RuntimeError("No valid texts for embedding")

    embeddings = embedder.encode(
        clean_texts,
        batch_size=64,
        show_progress_bar=True
    )

    return np.array(embeddings).astype(np.float32)


def check_ollama_models():
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]

        if "llama3:latest" not in models:
            print("❌ llama3 not found → run: ollama pull llama3")

        if "nomic-embed-text:latest" not in models:
            print("❌ nomic-embed-text not found → run: ollama pull nomic-embed-text")

    except:
        print("❌ Ollama not running on localhost:11434")

check_ollama_models()






# ============== Load / Ingest Functions ==============
def load_medquad_docs(csv_path: str) -> List[str]:
    """Robust loader for MedQuAD CSV -> list of 'Q: ... \\nA: ...' strings."""
    docs = []
    if not os.path.exists(csv_path):
        print(f"❌ MedQuAD CSV not found at {csv_path}. Place medquad.csv there.")
        return docs
    df = pd.read_csv(csv_path)
    q_col, a_col = None, None
    for c in df.columns:
        lc = c.lower()
        if ("question" in lc) or lc.startswith("q"):
            q_col = q_col or c
        if ("answer" in lc) or lc.startswith("a"):
            a_col = a_col or c
    if not (q_col and a_col):
        print("Could not auto-detect Q/A columns in MedQuAD CSV. Columns:", list(df.columns))
        return docs
    for _, row in df.iterrows():
        q = str(row[q_col]).strip() if pd.notna(row[q_col]) else ""
        a = str(row[a_col]).strip() if pd.notna(row[a_col]) else ""
        if q and a:
            docs.append(f"Q: {q}\nA: {a}")
    print(f"✅ Loaded MedQuAD docs: {len(docs)}")
    return docs

# ---------- Synthea summarizer from CSV (keeps original function name) ----------
def summarize_patients_from_fhir(fhir_dir: str) -> List[str]:
    """
    (Retained name for compatibility)
    Build one summary per patient from Synthea CSVs in fhir_dir.
    Expected files (any subset): allergies.csv, conditions.csv, medications.csv,
    procedures.csv, careplans.csv, encounters.csv, observations.csv.
    Each CSV should contain 'PATIENT' plus description-like fields.
    """
    if not os.path.exists(fhir_dir):
        print(f"❌ Synthea CSV folder not found at {fhir_dir}")
        return []

    # Helper to safely read CSV
    def read_csv_safe(path):
        try:
            return pd.read_csv(path)
        except Exception:
            try:
                return pd.read_csv(path, encoding="latin-1")
            except Exception:
                return pd.DataFrame()

    # Collect frames
    files = {
        "allergies": "allergies.csv",
        "conditions": "conditions.csv",
        "medications": "medications.csv",
        "procedures": "procedures.csv",
        "careplans": "careplans.csv",
        "encounters": "encounters.csv",
        "observations": "observations.csv",
    }
    dfs = {}
    for key, fname in files.items():
        fp = os.path.join(fhir_dir, fname)
        if os.path.exists(fp):
            dfs[key] = read_csv_safe(fp)
            print(f"Loaded {key} ({len(dfs[key])} rows)")
        else:
            dfs[key] = pd.DataFrame()

    # Normalize into {patient: {field: [texts...]}}
    by_patient: Dict[str, Dict[str, List[str]]] = {}
    def add(patient, field, text):
        if not patient or not isinstance(text, str) or not text.strip():
            return
        bucket = by_patient.setdefault(patient, {
            "conditions": [], "meds": [], "procedures": [],
            "encounters": [], "obs": [], "allergies": [], "careplans": []
        })
        bucket[field].append(text.strip())

    # Conditions
    if not dfs["conditions"].empty:
        cdf = dfs["conditions"]
        pcol = "PATIENT" if "PATIENT" in cdf.columns else None
        desc_cols = [c for c in cdf.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in cdf.columns if c.lower() in ("code", "type", "reason")]
        if pcol:
            for _, r in cdf.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in cdf.columns and pd.notna(r[c])])
                add(str(r[pcol]), "conditions", txt or "")

    # Medications
    if not dfs["medications"].empty:
        mdf = dfs["medications"]
        pcol = "PATIENT" if "PATIENT" in mdf.columns else None
        desc_cols = [c for c in mdf.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in mdf.columns if c.lower() in ("code", "reason", "rxnorm", "medication")]
        if pcol:
            for _, r in mdf.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in mdf.columns and pd.notna(r[c])])
                add(str(r[pcol]), "meds", txt or "")

    # Allergies
    if not dfs["allergies"].empty:
        adf = dfs["allergies"]
        pcol = "PATIENT" if "PATIENT" in adf.columns else None
        desc_cols = [c for c in adf.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in adf.columns if c.lower() in ("allergen", "type", "code")]
        if pcol:
            for _, r in adf.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in adf.columns and pd.notna(r[c])])
                add(str(r[pcol]), "allergies", txt or "")

    # Procedures
    if not dfs["procedures"].empty:
        pdf_ = dfs["procedures"]
        pcol = "PATIENT" if "PATIENT" in pdf_.columns else None
        desc_cols = [c for c in pdf_.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in pdf_.columns if c.lower() in ("code", "reason")]
        if pcol:
            for _, r in pdf_.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in pdf_.columns and pd.notna(r[c])])
                add(str(r[pcol]), "procedures", txt or "")

    # Encounters
    if not dfs["encounters"].empty:
        edf = dfs["encounters"]
        pcol = "PATIENT" if "PATIENT" in edf.columns else None
        desc_cols = [c for c in edf.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in edf.columns if c.lower() in ("reason", "code", "encounter")]
        if pcol:
            for _, r in edf.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in edf.columns and pd.notna(r[c])])
                add(str(r[pcol]), "encounters", txt or "")

    # Observations
    if not dfs["observations"].empty:
        odf = dfs["observations"]
        pcol = "PATIENT" if "PATIENT" in odf.columns else None
        desc_cols = [c for c in odf.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in odf.columns if c.lower() in ("code", "value", "type")]
        if pcol:
            for _, r in odf.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in odf.columns and pd.notna(r[c])])
                add(str(r[pcol]), "obs", txt or "")

    # Careplans
    if not dfs["careplans"].empty:
        cpdf = dfs["careplans"]
        pcol = "PATIENT" if "PATIENT" in cpdf.columns else None
        desc_cols = [c for c in cpdf.columns if "DESCRIPTION" in c.upper()] or \
                    [c for c in cpdf.columns if c.lower() in ("reason", "code", "type")]
        if pcol:
            for _, r in cpdf.iterrows():
                txt = ", ".join([str(r[c]) for c in desc_cols if c in cpdf.columns and pd.notna(r[c])])
                add(str(r[pcol]), "careplans", txt or "")

    # Build summaries
    summaries = []
    for pid, b in by_patient.items():
        s = f"PatientID: {pid}. "
        if b["conditions"]:
            s += "Conditions: " + ", ".join(sorted(set(b["conditions"])))[:700] + ". "
        if b["meds"]:
            s += "Medications: " + ", ".join(sorted(set(b["meds"])))[:700] + ". "
        if b["allergies"]:
            s += "Allergies: " + ", ".join(sorted(set(b["allergies"])))[:500] + ". "
        if b["procedures"]:
            s += "Procedures: " + ", ".join(sorted(set(b["procedures"])))[:500] + ". "
        if b["encounters"]:
            s += "Encounters: " + ", ".join(sorted(set(b["encounters"])))[:500] + ". "
        if b["careplans"]:
            s += "Careplans: " + ", ".join(sorted(set(b["careplans"])))[:500] + ". "
        if b["obs"]:
            s += "Observations: " + ", ".join(sorted(set(b["obs"])))[:500] + "."
        summaries.append(s.strip())
    print(f"✅ Synthea summarized {len(summaries)} patients")
    return summaries


# ---------- Books (PDF) loader ----------
from PyPDF2 import PdfReader

def pdf_extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print(f"⚠️ PDF read error for {file_path}: {e}")
    return text


def load_books_pdfs(books_dir: str) -> List[str]:
    """Extract text from PDFs in books_dir. Returns list of chunks."""
    if not os.path.exists(books_dir):
        print(f"❌ Books folder not found at {books_dir}")
        return []
    paths = sorted(glob.glob(os.path.join(books_dir, "*.pdf")))
    if not paths:
        print(f"ℹ️ No PDFs found in {books_dir}")
        return []

    texts = []
    for fp in paths:
        t = pdf_extract_text(fp)
        if t and t.strip():
            texts.append(f"[BOOK: {os.path.basename(fp)}]\n{t}")
            print(f"Extracted: {os.path.basename(fp)} ({len(t)} chars)")
        else:
            print(f"⚠️ No text extracted from {os.path.basename(fp)}")

    # Chunking (simple, by chars)
    chunks = []
    MAX_CHARS = 1200
    for doc in texts:
        s = doc.strip()
        i = 0
        while i < len(s):
            chunks.append(s[i:i+MAX_CHARS])
            i += MAX_CHARS
    print(f"✅ Created {len(chunks)} PDF chunks")
    return chunks


# ============== Index build / load helpers ==============
def build_or_load_faiss(texts: List[str], faiss_path: str, docs_path: str) -> Tuple["faiss.IndexFlatL2", List[str]]:
    """Load saved index/docs if present; otherwise build, save, and return."""

    if os.path.exists(faiss_path) and os.path.exists(docs_path):
        idx = faiss.read_index(faiss_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        print(f"📦 Loaded persisted index: {faiss_path} ({len(docs)} docs)")
        return idx, docs

    if not texts:
        print("No texts provided to build index:", faiss_path)
        return faiss.IndexFlatL2(384), []   # ✅ fixed

    t0 = time.time()
    print(f"🧱 Building FAISS index ({len(texts)} docs) -> {faiss_path} ...")

    # ✅ CLEAN TEXTS FIRST (IMPORTANT)
    clean_texts = [t.strip() for t in texts if t.strip()]

    embs = embed_texts(clean_texts)

    if embs.shape[0] == 0:
        raise ValueError("No embeddings generated — cannot build FAISS index")

    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)

    faiss.write_index(idx, faiss_path)

    # ✅ SAVE CLEAN TEXTS (NOT ORIGINAL)
    with open(docs_path, "wb") as f:
        pickle.dump(clean_texts, f)

    print(f"✅ Saved index & docs ({len(clean_texts)} items). Build time: {time.time()-t0:.1f}s")

    return idx, clean_texts
# ============== Build/load all requested indexes ==============
start_total = time.time()

med_docs = load_medquad_docs(MEDQUAD_CSV)
med_index, med_store = build_or_load_faiss(med_docs, FAISS_MED, DOCS_MED)

if ENABLE_SYNTH:
    if os.path.exists(FAISS_SYN) and os.path.exists(DOCS_SYN):
        # Load directly from disk
        syn_index, syn_store = build_or_load_faiss([], FAISS_SYN, DOCS_SYN)
    else:
        # First time → summarize patients and build index
        syn_docs = summarize_patients_from_fhir(SYN_DIR)
        syn_index, syn_store = build_or_load_faiss(syn_docs, FAISS_SYN, DOCS_SYN)
else:
    syn_index = syn_store = None


if ENABLE_BOOKS:
    if os.path.exists(FAISS_BOOKS) and os.path.exists(DOCS_BOOKS):
        # Load directly from disk
        books_index, books_store = build_or_load_faiss([], FAISS_BOOKS, DOCS_BOOKS)
    else:
        # First time → extract PDFs and build index
        book_docs = load_books_pdfs(BOOKS_DIR)
        books_index, books_store = build_or_load_faiss(book_docs, FAISS_BOOKS, DOCS_BOOKS)
else:
    books_index = books_store = None

print("⏱️ Total setup time:", time.time() - start_total, "s")

































def expand_queries(q: str) -> List[str]:

    return [
        q,
        q + "?",
        "Explain " + q,
        "Details about " + q
    ]
def faiss_search(index: "faiss.IndexFlatL2", store: List[str], q_emb: np.ndarray, k:int=5) -> List[Tuple[str,float,int]]:
    D, I = index.search(q_emb, k)
    hits = []
    for dist, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: continue
        hits.append((store[idx], float(dist), int(idx)))
    return hits

# # =========================
# # RETRIEVAL
# # =========================
# def retrieve_agentic_multi(query: str, top_k_initial: int = 8, top_k_final: int = 5):
#     expanded = expand_queries(query)
#     q_embs = embed_texts(expanded)

#     def search_pool(index, store):
#         if index is None or store is None:
#             return []
#         all_hits = []
#         for e in q_embs:
#             hs = faiss_search(index, store, e.reshape(1, -1), k=top_k_initial)
#             all_hits.extend(hs)

#         seen = set()
#         uniq = []
#         for doc, dist, idx in all_hits:
#             if doc not in seen:
#                 seen.add(doc)
#                 uniq.append(doc)

#         return uniq[:top_k_final]

#     return {
#         "medquad": search_pool(med_index, med_store),
#         "synthea": search_pool(syn_index, syn_store),
#         "books": search_pool(books_index, books_store)
#     }


# # =========================
# # SMART CONTEXT BUILDER
# # =========================
# def build_balanced_context(contexts, max_per_source=1000):
#     final = ""

#     for src in ("medquad", "books", "synthea"):
#         docs = contexts.get(src, [])
#         if not docs:
#             continue

#         chunk = ""
#         for d in docs:
#             if len(chunk) + len(d) < max_per_source:
#                 chunk += d.strip() + "\n\n"
#             else:
#                 break

#         if chunk:
#             final += f"\n\n[{src.upper()}]\n{chunk}"

#     return final.strip()


# # =========================
# # ANSWER GENERATION
# # =========================
# def answer_with_backend(query, context):
#     prompt = f"""
# You are a medical expert assistant.

# Answer the question clearly and directly using the context.

# IMPORTANT:
# - Always provide an answer
# - If partial info is available, still answer
# - Keep it simple and structured

# QUESTION:
# {query}

# CONTEXT:
# {context}

# FINAL ANSWER:
# """

#     print("🧠 Prompt size:", len(prompt))

#     result = ollama_generate(prompt)

#     if not result or not result.strip():
#         print("⚠️ Empty LLM response → fallback triggered")
#         return context[:500]

#     return result.strip()


# # =========================
# # MAIN PIPELINE
# # =========================
# def agentic_rag(query: str):
#     raw_contexts = retrieve_agentic_multi(query)

#     print("📊 Retrieved docs:",
#           {k: len(v) for k, v in raw_contexts.items()})

#     context = build_balanced_context(raw_contexts)

#     print("📊 Final context size:", len(context))

#     answer = answer_with_backend(query, context)

#     return {
#         "query": query,
#         "answer": answer,
#         "context_preview": context[:500]
#     }







# # ============== Retrieval & Agentic components ==============
_PATIENT_HINTS = re.compile(r"\b(patient|patientid|id[:\s]|medication|allerg|allergy|lab|encounter|condition|diagnos|ehr|record)\b", re.I)

def route_query(q: str) -> str:
    """Decide backend: 'medquad', 'synthea', 'books', or 'both'"""
    ql = q.lower()
    if ENABLE_SYNTH and _PATIENT_HINTS.search(ql):
        if any(w in ql for w in ["what is", "symptom", "treatment", "cause", "risk"]):
            return "both"   # mix medical knowledge + patient summaries
        return "synthea"
    # Books preferred for longer clinical background queries
    if ENABLE_BOOKS and any(w in ql for w in ["guideline", "who", "encyclopedia", "overview", "chapter", "definition", "management"]):
        return "books"
    return "medquad"

















def rerank_with_llm(query, candidates, top_k=5):
    return candidates[:top_k]


def compact_context(snippets: List[str], max_chars: int = 4000) -> str:
    seen = set(); uniq = []
    for s in snippets:
        s = s.strip()
        if s and s not in seen:
            seen.add(s); uniq.append(s)
    ctx = ""
    for s in uniq:
        if len(ctx) + len(s) + 2 > max_chars: break
        ctx += s + "\n\n"
    return ctx.strip()

# -----------------------------
# Agent wrappers & iterative refinement (Additive)
# -----------------------------

from dataclasses import dataclass, field
import statistics
import time
import re, json

# Small agent wrappers that reuse existing functions (non-invasive)
@dataclass
class RetrieverAgent:
    name: str = "RetrieverAgent"
    # uses retrieve_agentic_multi under the hood
    def retrieve(self, query: str):
        t0 = time.time()
        contexts, meta = retrieve_agentic_multi(query)
        t1 = time.time()
        return {"contexts": contexts, "meta": meta, "latency_s": t1 - t0}

@dataclass
class RerankerAgent:
    name: str = "RerankerAgent"
    # re-rank per-source using existing rerank_with_llm
    def rerank(self, query: str, candidates: List[str], top_k: int = 5):
        t0 = time.time()
        ranked = rerank_with_llm(query, candidates, top_k=top_k)
        t1 = time.time()
        return {"ranked": ranked, "latency_s": t1 - t0}

@dataclass
class SynthesizerAgent:
    name: str = "SynthesizerAgent"
    def synth(self, query: str, context: str, source_name:str, prefer:str="auto"):
        t0 = time.time()
        out = answer_with_backend(query, context, source_name=source_name, prefer=prefer)
        t1 = time.time()
        return {"answer": out, "latency_s": t1 - t0}

@dataclass
class JudgeAgent:
    name: str = "JudgeAgent"
    model: str = "deepseek"  # logical mapping to your available backends

    def judge(self, question: str, answer: str, contexts: Dict[str, str]) -> Dict:
        prompt = (
            f"Evaluate the following answer for the question. Return JSON: "
            f'{{"faithful": bool, "relevance": float, "explanation": str}}\n\n'
            f"QUESTION: {question}\n\nANSWER: {answer}\n\nCONTEXTS:\n"
        )
        for k, v in contexts.items():
            prompt += f"\n---{k}---\n{v[:4000]}\n"

        # Try with LLMs first
        raw = ollama_generate(prompt)

        print(raw)


        try:
            if isinstance(raw, dict):
                j = raw
            elif isinstance(raw, str):
                # remove markdown fences if present
                raw_clean = raw.strip()
                if raw_clean.startswith("```"):
                    raw_clean = re.sub(r"^```[a-zA-Z0-9]*\n", "", raw_clean)
                    raw_clean = raw_clean.rstrip("`").strip()
                # extract JSON object
                match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
                if not match:
                    raise ValueError("No JSON found in raw output")
                j = json.loads(match.group(0))
            else:
                raise TypeError(f"Unexpected raw type: {type(raw)}")

            return {
                "faithful": bool(j.get("faithful")),
                "relevance": float(j.get("relevance", 0.0)),
                "explanation": j.get("explanation", "")
            }
        except Exception as e:
            print("JSON parse failed:", e)
            # --- Heuristic fallback ---
            rel = 0.0
            faithful = False
            if answer and any(len(c) > 200 for c in contexts.values()):
                rel = 0.6
                ctx_text = " ".join(contexts.values()).lower()
                overlap = sum(1 for w in answer.lower().split() if w in ctx_text)
                faithful = overlap / max(1, len(answer.split())) > 0.15
            return {
                "faithful": faithful,
                "relevance": rel,
                "explanation": "Heuristic judge (LLM unavailable or JSON parse failed)."
            }


# Confidence combiner: combine retrieval stats + judge outputs into single score [0,1]
def compute_confidence(fetch_meta: Dict, per_source_meta: Dict, judge_scores: Dict) -> float:
    """
    Combine retrieval stats, judge outputs, and avg_distance into a confidence score [0,1].
    """
    try:
        src_counts = fetch_meta.get("sources", {})
        total_hits = sum(int(v) for v in src_counts.values() if isinstance(v, (int, float, str)))
    except Exception:
        total_hits = 0

    # Retrieval factor: more hits -> more confidence
    retrieval_factor = min(1.0, total_hits / 20.0)  # 0..1

    # Avg distance factor: lower distance = better
    dist_meta = fetch_meta.get("avg_distance", {})
    dist_scores = []
    for src, d in dist_meta.items():
        if d is None:
            continue
        # normalize: 0..3 mapped to 1..0
        q = max(0.0, min(1.0, 1.0 - (d / 3.0)))
        dist_scores.append(q)
    distance_factor = float(sum(dist_scores) / len(dist_scores)) if dist_scores else 0.5

    # Rerank factor: based on context lengths
    rerank_factor = 0.5
    for s, meta in per_source_meta.items():
        if meta.get("available"):
            rerank_factor += min(0.25, meta.get("char_count", 0) / 4000.0 * 0.25)
    rerank_factor = min(1.0, rerank_factor)

    # Judge factors
    judge_relevance = float(judge_scores.get("relevance", 0.0))
    judge_faithful = 1.0 if judge_scores.get("faithful") else 0.0

    # Weighted blend
    score = (
        0.25 * retrieval_factor +
        0.25 * distance_factor +
        0.25 * rerank_factor +
        0.15 * judge_relevance +
        0.10 * judge_faithful
    )

    return max(0.0, min(1.0, score))


# # Add an iterative refinement loop that preserves your agentic_rag output structure
# def iterative_refine(query: str, prefer_backend: str = "auto", confidence_threshold: float = 0.6, max_rounds: int = 2) -> Dict:
#     """
#     1) Retrieve contexts (retriever agent)
#     2) Synthesize per-source answers (synth agent)
#     3) Judge combined answer (judge agent)
#     4) If confidence < threshold, do one re-retrieval with broadened query and re-run.
#     Returns same structure as agentic_rag plus added 'confidence' and 'rounds'
#     """
#     retriever = RetrieverAgent()
#     synthesizer = SynthesizerAgent()
#     judge = JudgeAgent()

#     rounds = 0
#     last_meta = None
#     combined_answers = None
#     while rounds < max_rounds:
#         rounds += 1
#         ret = retriever.retrieve(query)
#         contexts = ret["contexts"]
#         meta = ret["meta"]
#         # Per-source synth
#         answers = {}
#         per_src_meta = {}
#         for source_key in ("medquad", "synthea", "books"):
#             ctx = contexts.get(source_key, "")
#             if not ctx:
#                 answers[source_key] = ""
#                 per_src_meta[source_key] = {"available": False, "char_count": 0}
#                 continue
#             out = synthesizer.synth(query, ctx, source_name=source_key, prefer=prefer_backend)
#             answers[source_key] = out["answer"]
#             per_src_meta[source_key] = {"available": True, "char_count": len(ctx), "latency_s": out["latency_s"]}

#         # Combine answers for judge
#         combined_text = "\n\n---\n\n".join([f"[{k}]\n{answers[k]}" for k in answers if answers[k]])
#         judge_scores = judge.judge(query, combined_text, contexts)
#         confidence = compute_confidence(meta, per_src_meta, judge_scores)

#         # Log metrics into your metrics_log for later cost/confidence analysis
#         metrics_log.append({
#             "time": str(datetime.datetime.now()),
#             "round": rounds,
#             "query": query,
#             "confidence": confidence,
#             "judge": judge_scores,
#             "retrieve_meta": meta,
#             "per_source_meta": per_src_meta,
#             "latency": ret.get("latency_s", None)
#         })

#         # If confident enough, stop; else modify query & retry once
#         if confidence >= confidence_threshold:
#             combined_answers = {"answers": answers, "contexts": contexts, "meta": meta}
#             break
#         else:
#             # Simple, safe re-query: append judge explanation as hint
#             # Keeps everything deterministic and non-destructive.
#             hint = judge_scores.get("explanation", "")
#             query = query + " " + ("Follow-up: " + hint if hint else " Please broaden sources.")
#             last_meta = {"prev_meta": meta, "judge": judge_scores}
#             # Continue loop to attempt improvement

#     # final packaging: same structure as agentic_rag plus confidence/rounds
#     result = {
#         "query": query,
#         "answers": combined_answers["answers"] if combined_answers else answers,
#         "contexts": combined_answers["contexts"] if combined_answers else contexts,
#         "meta": {
#             "confidence": confidence,
#             "rounds": rounds,
#             "retrieve_meta": (combined_answers["meta"] if combined_answers else meta),
#             "judge": judge_scores
#         }
#     }
#     # Save chat similar to existing save_chat for continuity
#     combined_answer_text = "\n\n---\n\n".join([f"[{k}]\n{result['answers'][k]}" for k in result['answers'] if result['answers'][k]])
#     save_chat(result["query"], combined_answer_text, result["meta"])
#     return result


# route = route_query(query)


def retrieve_agentic_multi(query: str, top_k_initial: int = 8, top_k_final: int = 5) -> Tuple[Dict[str, str], Dict]:
    """
    Returns:
      - contexts: dict { "medquad": ctx_str, "synthea": ctx_str, "books": ctx_str }
      - detail/meta: dict with counts, expanded queries, and avg distances
    """
    route = route_query(query)
    # route = llm_route_query(query) if (USE_GEMINI or USE_OPENAI or USE_GROK or USE_DEEPSEEK) else route_query(query)

    expanded = expand_queries(query)
    q_embs = embed_texts(expanded)

    contexts = {}
    detail = {"route": route, "expanded_queries": expanded, "sources": {}}

    def search_pool(idx_obj, store, label):
        """Search one pool, return list of (doc, distance, idx)."""
        if idx_obj is None or store is None:
            detail["sources"][label] = 0
            return []
        all_hits = []
        for e in q_embs:
            e = e.reshape(1, -1)
            hs = faiss_search(idx_obj, store, e, k=top_k_initial)
            all_hits.extend(hs)
        # De-dupe by doc text, but keep the *first* distance
        seen, uniq = {}, []
        for doc, dist, idx in all_hits:
            if doc not in seen:
                seen[doc] = dist
                uniq.append((doc, dist, idx))
        detail["sources"][label] = len(uniq)
        return uniq

    # Search each pool independently (if enabled)
    med_hits = search_pool(med_index, med_store, "medquad")
    syn_hits = search_pool(syn_index, syn_store, "synthea") if ENABLE_SYNTH else []
    book_hits = search_pool(books_index, books_store, "books") if ENABLE_BOOKS else []

    # --- Distance metrics block (new) ---
    def avg_dist(hits):
        if not hits:
            return None
        try:
            ds = [h[1] for h in hits if isinstance(h[1], (int, float))]
            return float(sum(ds) / len(ds)) if ds else None
        except Exception:
            return None

    detail["avg_distance"] = {
        "medquad": avg_dist(med_hits),
        "synthea": avg_dist(syn_hits),
        "books": avg_dist(book_hits),
    }

    # Re-rank per-source (optional) and compact
    med_ranked = rerank_with_llm(query, [h[0] for h in med_hits], top_k=top_k_final)
    syn_ranked = rerank_with_llm(query, [h[0] for h in syn_hits], top_k=top_k_final) if syn_hits else []
    book_ranked = rerank_with_llm(query, [h[0] for h in book_hits], top_k=top_k_final) if book_hits else []

    # contexts["medquad"] = compact_context(med_ranked, max_chars=4000)
    contexts["medquad"] = compact_context(
        [c for c in med_ranked if c.strip()],
        max_chars=4000
    )

    print("📊 medquad context length:", len(contexts["medquad"]))
    contexts["synthea"] = compact_context(syn_ranked, max_chars=4000) if syn_ranked else ""
    contexts["books"] = compact_context(book_ranked, max_chars=4000) if book_ranked else ""

    return contexts, detail


# ---------- Safety suffix and per-source answer generation ----------
SAFETY_SUFFIX = (
    "\n\nAnswer concisely and only using the provided context. "
    "If context is insufficient say 'I don't know from the provided context.' "
    "Do not provide a diagnosis. Recommend consulting a clinician."
)


# Define answer styles per dataset
SOURCE_STYLES = {
    "medquad": "Answer in Q&A format, as if from a medical FAQ.",
    "synthea": "Answer as a patient record summary, using clinical language.",
    "books": "Answer like a medical encyclopedia entry, detailed and structured."
}


def smart_truncate(text, max_chars=4000):
    sentences = text.split(". ")
    out = ""
    for s in sentences:
        if len(out) + len(s) < max_chars:
            out += s + ". "
        else:
            break
    return out.strip()
def answer_with_backend(query, context, source_name="generic", prefer="auto"):


    context = smart_truncate(context, max_chars=4000)
    style = SOURCE_STYLES.get(source_name, "Answer concisely.")

    prompt = f"""
You are a medical assistant.

Answer the question clearly using the context below.

If multiple contexts are given, combine them.

If the answer is partially available, still answer.

Question:
{query}

Context:
{context}

Instruction:
{style}

Answer only from the context.
If not present say "I don't know from the provided context."
"""
    print(f"🧠 Prompt size: {len(prompt)} chars")
    return ollama_generate(prompt)
# ---------- Added / Edited helper functions for final-answer synthesis ----------
# Insert these into your existing file after `answer_with_backend` (or anywhere appropriate).
# They implement LLM-based synthesis with a safe deterministic fallback.

SOURCE_WEIGHTS = {
    "medquad": 1.0,
    "synthea": 0.8,
    "books": 0.9
}

def _build_synthesis_prompt(query: str, answers: dict, contexts: dict) -> str:
    """
    Build a prompt that asks the LLM to synthesise a final answer from the per-source answers.
    The LLM is instructed to only use the provided source answers & contexts and to report conflicts.
    """
    parts = []
    parts.append("You are a careful medical synthesis assistant.")
    parts.append("User question: " + query)
    parts.append("\n--- PER-SOURCE ANSWERS (do NOT invent new facts) ---\n")
    for src in ("medquad", "synthea", "books"):
        ans = answers.get(src, "")
        ctx = contexts.get(src, "")
        parts.append(f"[{src}]\nAnswer:\n{ans or '(no answer)'}\nContext preview:\n{(ctx[:800] + '...') if ctx else '(no context)'}\n")
    parts.append("\nInstruction:")
    parts.append(
        "Produce a single, concise synthesis answer that:\n"
        "  1) Is strictly grounded in the per-source answers and contexts above.\n"
        "  2) If sources agree, state the consensus succinctly and list which sources contributed.\n"
        "  3) If sources conflict, explicitly list the conflicting claims and which source said each.\n"
        "  4) Provide a short provenance section listing which sources and context snippets support the final statements.\n"
        "  5) Do NOT provide any new clinical recommendations, diagnoses, or advice beyond what the contexts say.\n"
    )
    parts.append(SAFETY_SUFFIX)
    return "\n".join(parts)

def _fallback_merge(answers: dict, contexts: dict) -> dict:
    """
    Deterministic fallback when no LLM is available.
    Strategy:
      - If one source answer is non-empty and others empty -> choose it.
      - If multiple non-empty: prefer highest weighted source; if tie, choose longest answer.
      - Produce short provenance stating which source was chosen and char counts.
    Returns { "final": str, "provenance": str }
    """
    non_empty = {k: v for k, v in answers.items() if v and v.strip()}
    if not non_empty:
        return {"final": "I don't know from the provided context.", "provenance": "No source answers available."}

    # Score each source by weight (presence only) and length as tie-breaker
    scored = []
    for k, v in non_empty.items():
        w = SOURCE_WEIGHTS.get(k, 0.5)
        scored.append((k, w, len(v), v))
    # sort by weight desc, length desc
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    chosen = scored[0]
    src, weight, length, text = chosen
    other_sources = [s for s in non_empty.keys() if s != src]
    prov = f"Selected answer from source '{src}' (weight={weight}, chars={length})."
    if other_sources:
        prov += " Other non-empty sources: " + ", ".join(other_sources) + "."
    final = text.strip()
    # Ensure safety line
    if final:
        final += "\n\nNote: This synthesis is strictly derived from provided sources. Consult a clinician for diagnosis."
    return {"final": final, "provenance": prov}

def combine_answers(query: str, answers: dict, contexts: dict, prefer_backend: str = "auto") -> dict:
    """
    Produce a single final answer plus provenance from per-source answers.
    Returns:
      {
        "final_answer": str,
        "provenance": str,
        "method": "llm_synthesis" or "fallback_merge",
        "per_source": { ... }  # copy of input answers
      }
    """
    prompt = _build_synthesis_prompt(query, answers, contexts)

    out = ollama_generate(prompt)
    # If an LLM responded with something useful, try to split final/provenance heuristically
    if out and out.strip() and out.strip() != "No LLM backend available (set GEMINI_API_KEY, OPENAI_API_KEY, XAI_API_KEY, or DEEPSEEK_API_KEY).":
        text = out.strip()
        prov = ""
        for marker in ("\n\nProvenance:", "\n\nSources:", "\n\nEvidence:", "\n\nSource:", "\n\nPROVENANCE:"):
            if marker in text:
                parts = text.split(marker, 1)
                final = parts[0].strip()
                prov = marker.strip() + " " + parts[1].strip()
                return {"final_answer": final, "provenance": prov, "method": "llm_synthesis", "per_source": answers}
        # No explicit marker found — return entire output as final answer and minimal provenance
        return {"final_answer": text, "provenance": "Synthesis produced by LLM from per-source answers.", "method": "llm_synthesis", "per_source": answers}

    # No usable LLM output -> deterministic fallback
    fallback = _fallback_merge(answers, contexts)
    return {"final_answer": fallback["final"], "provenance": fallback["provenance"], "method": "fallback_merge", "per_source": answers}

# ---------- Replace / edit the existing agentic_rag with this updated orchestrator ----------
# This version calls per-source generators, then synthesizes into a single final answer.





# ---------- Chat logging ----------
if os.path.exists(CHATLOG_PATH):
    try:
        with open(CHATLOG_PATH,"r") as f:
            CHAT_LOG = json.load(f)
    except Exception:
        CHAT_LOG = []
else:
    CHAT_LOG = []

def summarize_chat(last_n: int = 6) -> str:
    subset = CHAT_LOG[-last_n:]
    lines = []
    for e in subset:
        lines.append(f"U: {e.get('query','')[:200]}")
        lines.append(f"A: {e.get('answer','')[:200]}")
    s = "\n".join(lines)
    return s[-1200:]

def save_chat(query: str, answer: str, meta: Dict):
    CHAT_LOG.append({
        "time": str(datetime.datetime.now()),
        "query": query,
        "answer": answer,
        "meta": meta
    })
    with open(CHATLOG_PATH, "w") as f:
        json.dump(CHAT_LOG, f, indent=2)


def agentic_rag(query: str, prefer_backend: str = "auto") -> dict:
    """
    Updated orchestrator:
      - retrieve per-source contexts
      - generate one answer per available source
      - synthesize a final answer (LLM if available, else deterministic fallback)
    """
    contexts, meta = retrieve_agentic_multi(query)
    chat_sum = summarize_chat(6)
    if chat_sum:
        # Attach recent convo summary to each non-empty context (helps LLM)
        for k in contexts:
            if contexts[k]:
                combined = contexts[k] + "\n\n[Recent conversation summary]\n" + chat_sum
                contexts[k] = smart_truncate(combined, max_chars=4000)

    answers = {}
    per_source_meta = {}
    combined_context = ""

    for src in ("medquad", "synthea", "books"):
        if contexts.get(src):
            combined_context += f"\n\n[{src.upper()}]\n{contexts[src][:800]}"

    print("📊 Combined context size:", len(combined_context))

    final_answer = answer_with_backend(query, combined_context, source_name="combined")
    print("🧠 Generated Answer:", final_answer)
    answers = {"combined": final_answer}
    # For each source, ask the backend to answer (only if context present)
    # for source_key in ("medquad", "synthea", "books"):
    #     ctx = contexts.get(source_key, "")
    #     if not ctx:
    #         answers[source_key] = ""  # no context found
    #         per_source_meta[source_key] = {"available": False, "char_count": 0}
    #         continue
    #     # Generate per-source answer (use the same prefer_backend routing)
    #     ans = answer_with_backend(query, ctx, source_name=source_key, prefer=prefer_backend)
    #     answers[source_key] = ans
    #     per_source_meta[source_key] = {"available": True, "char_count": len(ctx)}

    # Synthesize final answer from per-source answers
    synthesis = combine_answers(query, answers, contexts, prefer_backend=prefer_backend)

    combined_meta = {
        "retrieve_meta": meta,
        "per_source": per_source_meta,
        "synthesis_meta": {"method": synthesis["method"]},
        "timestamp": str(datetime.datetime.now())
    }

    # Save a combined chat entry with the final answer and per-source parts
    combined_answer_text = (
        "[FINAL]\n" + synthesis["final_answer"] + "\n\n[PROVENANCE]\n" + synthesis["provenance"] +
        "\n\n---\n\n" + "\n\n".join([f"[{k}]\n{answers[k]}" for k in answers if answers[k]])
    )
    save_chat(query, combined_answer_text, combined_meta)

    return {
        "query": query,
        "final_answer": synthesis["final_answer"],
        "provenance": synthesis["provenance"],
        "answers": answers,
        "contexts": contexts,
        "meta": combined_meta
    }


def print_backend_status():
    print("Backend: Ollama (local)")

# ---------- CLI ----------
if __name__ == "__main__":
    print("✅ Agentic RAG ready.")
    print_backend_status()
    print("Toggles: ENABLE_SYNTH =", ENABLE_SYNTH, ", ENABLE_BOOKS =", ENABLE_BOOKS)
    print("Data roots:")
    print("  MedQuAD CSV:", MEDQUAD_CSV)
    print("  Synthea CSV dir:", SYN_DIR)
    print("  Books dir:", BOOKS_DIR)
    print("\nType 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        print("…retrieving contexts from medquad / synthea / books…")


        # out = agentic_rag(q)
        out = agentic_rag(q, prefer_backend="auto")
        # if use_iterative:
        #     out = iterative_refine(q, prefer_backend="auto")
        # else:
        #     out = agentic_rag(q, prefer_backend="auto")


        # Nicely print per-source outputs:
        for src in ("medquad", "synthea", "books"):
            ctx = out["contexts"].get(src, "")
            ans = out["answers"].get(src, "")
            print("\n" + ("="*10) + f" SOURCE: {src} " + ("="*10))
            print("Context present:", bool(ctx))
            if ctx:
                # show short preview of context and answer
                print("Context (preview):", ctx[:500].replace("\n", " ") + ("..." if len(ctx) > 500 else ""))
            print("\nAnswer:")
            print(ans or "(no answer from this source)")
        print("\nmeta:", json.dumps(out["meta"], indent=2))







def agentic_rag_for_ui(question: str) -> dict:
    """
    Lightweight wrapper to expose the RAG pipeline to the Flask UI.
    Returns JSON-serializable dict: {"answer": str, "meta": dict}
    """
    try:
        # Calls your existing orchestrator. It should return final_answer & meta.
        out = agentic_rag(question, prefer_backend="auto")
        return {
            "answer": out.get("final_answer") or out.get("answers", {}).get("medquad") or "",
            "meta": out.get("meta", {})
        }
    except Exception as e:
        # Controlled fallback so UI doesn't crash.
        return {"answer": f"Agent error: {str(e)}", "meta": {"error": str(e)}}

