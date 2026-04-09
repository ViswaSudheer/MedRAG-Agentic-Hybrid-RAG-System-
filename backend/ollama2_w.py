############################################ollama redundant code####################################################

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
from sentence_transformers import SentenceTransformer

from config import *
from ollama_utils import generate_llm as ollama_generate

# Optional deps guard
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

if _missing_msgs:
    print("🔧 Missing packages detected. If you see errors later, run:")
    for cmd in _missing_msgs:
        print("   ", cmd)

# spaCy (Windows-safe: single process)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception:
    raise RuntimeError("spaCy 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")


# ────────────────────────────────────────────────
#            Embedding (local SentenceTransformer)
# ────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"
print(f"Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)


def preprocess_texts(texts: List[str]) -> List[str]:
    """Lowercase, lemmatize, remove stopwords/non-alpha (single process)"""
    def normalize(doc):
        return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])
    return [normalize(doc) for doc in nlp.pipe(texts, n_process=1, batch_size=64)]


def embed_texts(texts: List[str]) -> np.ndarray:
    """Fast local embeddings with SentenceTransformer"""
    clean_texts = [t.strip() for t in texts if t.strip()]
    if not clean_texts:
        raise ValueError("No valid texts to embed")
    
    embeddings = embedder.encode(
        clean_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


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

    files = {
        "allergies": "allergies.csv", "conditions": "conditions.csv",
        "medications": "medications.csv", "procedures": "procedures.csv",
        "careplans": "careplans.csv", "encounters": "encounters.csv",
        "observations": "observations.csv"
    }
    dfs = {k: read_csv_safe(os.path.join(fhir_dir, fname)) for k, fname in files.items()}

    by_patient: Dict[str, Dict[str, List[str]]] = {}
    def add(pid, field, text):
        if not pid or not text.strip():
            return
        by_patient.setdefault(pid, {f: [] for f in ["conditions","meds","allergies","procedures","encounters","obs","careplans"]})
        by_patient[pid][field].append(text.strip())

    for key, df in dfs.items():
        if df.empty or "PATIENT" not in df.columns:
            continue
        desc_cols = [c for c in df.columns if any(x in c.upper() for x in ["DESCRIPTION","CODE","REASON","VALUE","TYPE","RXNORM"])]
        for _, r in df.iterrows():
            txt = ", ".join(str(r[c]) for c in desc_cols if pd.notna(r[c]))
            if txt:
                add(str(r["PATIENT"]), key, txt)

    summaries = []
    for pid, d in by_patient.items():
        parts = [f"PatientID: {pid}."]
        for k, vals in d.items():
            if vals:
                joined = ", ".join(sorted(set(vals)))
                limit = 700 if k in ("conditions", "meds") else 500
                parts.append(f"{k.capitalize()}: {joined[:limit]}")
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
#               FAISS Index Helpers
# ────────────────────────────────────────────────

def build_or_load_faiss(
    texts: List[str],
    faiss_path: str,
    docs_path: str
) -> Tuple["faiss.IndexFlatL2", List[str]]:
    if os.path.exists(faiss_path) and os.path.exists(docs_path):
        idx = faiss.read_index(faiss_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        print(f"📦 Loaded FAISS: {faiss_path} ({len(docs)} docs)")
        return idx, docs

    if not texts:
        print(f"No texts to index: {faiss_path}")
        return faiss.IndexFlatL2(384), []  # all-MiniLM-L6-v2 dim

    print(f"🧱 Building FAISS → {faiss_path} ({len(texts)} docs)")
    t0 = time.time()

    clean_texts = [t.strip() for t in texts if t.strip()]
    if not clean_texts:
        raise ValueError("No valid texts after cleaning")

    embs = embed_texts(clean_texts)
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)

    faiss.write_index(idx, faiss_path)
    with open(docs_path, "wb") as f:
        pickle.dump(clean_texts, f)

    print(f"✅ Saved index & docs ({len(clean_texts)} items). Time: {time.time()-t0:.1f}s")
    return idx, clean_texts


# ────────────────────────────────────────────────
#               Load / Build Indexes
# ────────────────────────────────────────────────

start_total = time.time()

med_docs = load_medquad_docs(MEDQUAD_CSV)
med_index, med_store = build_or_load_faiss(med_docs, FAISS_MED, DOCS_MED)

if ENABLE_SYNTH:
    if os.path.exists(FAISS_SYN) and os.path.exists(DOCS_SYN):
        syn_index, syn_store = build_or_load_faiss([], FAISS_SYN, DOCS_SYN)
    else:
        syn_docs = summarize_patients_from_fhir(SYN_DIR)
        syn_index, syn_store = build_or_load_faiss(syn_docs, FAISS_SYN, DOCS_SYN)
else:
    syn_index = syn_store = None

if ENABLE_BOOKS:
    if os.path.exists(FAISS_BOOKS) and os.path.exists(DOCS_BOOKS):
        books_index, books_store = build_or_load_faiss([], FAISS_BOOKS, DOCS_BOOKS)
    else:
        book_docs = load_books_pdfs(BOOKS_DIR)
        books_index, books_store = build_or_load_faiss(book_docs, FAISS_BOOKS, DOCS_BOOKS)
else:
    books_index = books_store = None

print(f"⏱️ Total setup time: {time.time() - start_total:.2f} s")


# ────────────────────────────────────────────────
#               Retrieval Logic
# ────────────────────────────────────────────────

_PATIENT_HINTS = re.compile(r"\b(patient|patientid|id[:\s]|medication|allerg|allergy|lab|encounter|condition|diagnos|ehr|record)\b", re.I)

def route_query(q: str) -> str:
    ql = q.lower()
    if ENABLE_SYNTH and _PATIENT_HINTS.search(ql):
        if any(w in ql for w in ["what is", "symptom", "treatment", "cause", "risk"]):
            return "both"
        return "synthea"
    if ENABLE_BOOKS and any(w in ql for w in ["guideline", "who", "encyclopedia", "overview", "chapter", "definition", "management"]):
        return "books"
    return "medquad"


def expand_queries(q: str) -> List[str]:
    return [q, q + "?", f"Explain {q}", f"Details about {q}"]


def faiss_search(index: "faiss.IndexFlatL2", store: List[str], q_emb: np.ndarray, k: int = 8):
    D, I = index.search(q_emb.reshape(1, -1), k)
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        hits.append((store[idx], float(dist)))
    return hits


def compact_context(snippets: List[str], max_chars: int = 4000) -> str:
    seen = set()
    uniq = []
    for s in snippets:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    ctx = ""
    for s in uniq:
        if len(ctx) + len(s) + 2 > max_chars:
            break
        ctx += s + "\n\n"
    return ctx.strip()


def retrieve_agentic_multi(query: str, top_k: int = 8) -> Tuple[Dict[str, str], Dict]:
    route = route_query(query)
    expanded = expand_queries(query)
    q_embs = embed_texts(expanded)

    detail = {"route": route, "sources": {}, "avg_distance": {}}
    contexts = {}

    def search_pool(idx, store, label):
        if idx is None or store is None:
            return []
        all_hits = []
        for emb in q_embs:
            hits = faiss_search(idx, store, emb, k=top_k)
            all_hits.extend(hits)
        # dedup + keep best distance
        seen = {}
        for doc, dist in all_hits:
            if doc not in seen or dist < seen[doc]:
                seen[doc] = dist
        detail["sources"][label] = len(seen)
        ds = list(seen.values())
        detail["avg_distance"][label] = sum(ds)/len(ds) if ds else None
        return list(seen.keys())

    med_docs = search_pool(med_index, med_store, "medquad")
    syn_docs = search_pool(syn_index, syn_store, "synthea") if ENABLE_SYNTH else []
    book_docs = search_pool(books_index, books_store, "books") if ENABLE_BOOKS else []

    contexts["medquad"] = compact_context(med_docs)
    contexts["synthea"] = compact_context(syn_docs) if syn_docs else ""
    contexts["books"]   = compact_context(book_docs) if book_docs else ""

    print("Retrieved document counts:", {k: len(v.splitlines()) for k,v in contexts.items() if v})

    return contexts, detail


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


def _build_synthesis_prompt(query: str, answers: Dict, contexts: Dict) -> str:
    parts = [
        "You are a careful medical synthesis assistant.",
        f"User question: {query}",
        "\n--- SOURCE ANSWERS (use only this information) ---"
    ]
    for src in ("medquad", "synthea", "books"):
        ans = answers.get(src, "")
        prev = (contexts.get(src, "")[:700] + "...") if contexts.get(src) else "(empty)"
        parts.append(f"[{src.upper()}]\nAnswer: {ans or '(no answer)'}\nContext preview: {prev}")
    parts.extend([
        "\nInstructions:",
        "- Create one concise final answer grounded **only** in the above answers/contexts",
        "- If sources agree → state consensus + which sources support it",
        "- If conflict → clearly list what each source says",
        "- Add short provenance section showing supporting sources",
        "- NEVER add new medical advice, diagnoses or recommendations",
        SAFETY_SUFFIX
    ])
    return "\n".join(parts)


def _fallback_merge(answers: Dict) -> Dict:
    non_empty = {k:v for k,v in answers.items() if v and v.strip() != "I don't know from the provided context."}
    if not non_empty:
        return {"final": "I don't know from the provided context.", "provenance": "No usable source answers."}

    scored = [(k, SOURCE_WEIGHTS.get(k, 0.5), len(v), v) for k,v in non_empty.items()]
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    src, _, length, text = scored[0]
    others = ", ".join(k for k in non_empty if k != src)
    prov = f"Selected highest-weighted source '{src}' (weight={SOURCE_WEIGHTS.get(src)}, length={length})"
    if others:
        prov += f". Other sources: {others}"
    final = text
    if final:
        final += "\n\nNote: Strictly based on retrieved sources. Consult a doctor for medical decisions."
    return {"final": final, "provenance": prov}


def combine_answers(query: str, answers: Dict, contexts: Dict) -> Dict:
    prompt = _build_synthesis_prompt(query, answers, contexts)
    out = ollama_generate(prompt).strip()

    if not out:
        fb = _fallback_merge(answers)
        return {"final_answer": fb["final"], "provenance": fb["provenance"], "method": "fallback"}

    # heuristic split for provenance
    for marker in ("\n\nProvenance:", "\n\nSources:", "\n\nEvidence:", "\n\nPROVENANCE:"):
        if marker in out:
            final, prov = out.split(marker, 1)
            return {
                "final_answer": final.strip(),
                "provenance": marker.strip() + " " + prov.strip(),
                "method": "llm_synthesis"
            }
    return {
        "final_answer": out,
        "provenance": "LLM-generated synthesis",
        "method": "llm_synthesis"
    }


# ────────────────────────────────────────────────
#               Main RAG Function
# ────────────────────────────────────────────────

def agentic_rag(query: str) -> dict:
    contexts, retrieve_meta = retrieve_agentic_multi(query)

    answers = {}
    for src in ("medquad", "synthea", "books"):
        ctx = contexts.get(src, "")
        if ctx.strip():
            answers[src] = answer_with_backend(query, ctx, src)
        else:
            answers[src] = ""

    synthesis = combine_answers(query, answers, contexts)

    meta = {
        "retrieve": retrieve_meta,
        "synthesis_method": synthesis["method"],
        "timestamp": str(datetime.datetime.now())
    }

    # Log to chat history
    combined_text = (
        "[FINAL ANSWER]\n" + synthesis["final_answer"] +
        "\n\n[PROVENANCE]\n" + synthesis["provenance"] +
        "\n\n" + "\n\n".join(f"[{k.upper()}]\n{v}" for k,v in answers.items() if v)
    )
    save_chat(query, combined_text, meta)

    return {
        "query": query,
        "final_answer": synthesis["final_answer"],
        "provenance": synthesis["provenance"],
        "per_source_answers": answers,
        "contexts": contexts,
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
#               CLI Interface
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("✅ Medical RAG system ready (local embeddings + Ollama LLM)")
    print(f"Embedding model: {EMBED_MODEL} (384 dim)")
    print(f"Toggles: ENABLE_SYNTH = {ENABLE_SYNTH}, ENABLE_BOOKS = {ENABLE_BOOKS}")
    print("Data locations:")
    print(f"  • MedQuAD:  {MEDQUAD_CSV}")
    print(f"  • Synthea:  {SYN_DIR}")
    print(f"  • Books:    {BOOKS_DIR}")
    print("\nType 'exit' or 'quit' to stop.\n")

    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q or q.lower() in ("exit", "quit"):
            break

        print("\n… processing …")
        result = agentic_rag(q)

        print("\n" + "═"*60)
        print("FINAL ANSWER")
        print("═"*60)
        print(result["final_answer"] or "(no answer generated)")

        print("\nPROVENANCE:")
        print(result["provenance"])

        print("\n" + "─"*50)
        print("SOURCE ANSWERS & CONTEXTS")
        print("─"*50)
        for src in ("medquad", "synthea", "books"):
            ans = result["per_source_answers"].get(src, "")
            ctx = result["contexts"].get(src, "")
            print(f"[{src.upper()}]")
            print("Answer:", ans or "(no relevant context)")
            if ctx:
                print("Context preview:", ctx[:300].replace("\n", " ") + "..." if len(ctx)>300 else ctx)
            print()

        print("\nMetadata:")
        print(json.dumps(result["meta"], indent=2))
        print()