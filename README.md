# 🩺 MedRAG: Agentic Hybrid RAG Medical Assistant

**An Intelligent Medical Question-Answering System powered by Agentic Retrieval-Augmented Generation (RAG)**

A full-stack **Agentic RAG** application that answers medical queries using multiple trusted knowledge sources with high accuracy, context awareness, and safety.

---

## ✨ Features

- **Hybrid Retrieval** — Dense (FAISS) + Sparse (BM25) + Reciprocal Rank Fusion (RRF) + Cross-Encoder reranking
- **HyDE (Hypothetical Document Embeddings)** for improved retrieval quality
- **Multi-Source Knowledge Base**:
  - MedQuAD (Medical QA)
  - PubMedQA (61k+ research QA)
  - Synthea Patient Records
  - Medical Books (PDFs)
- **Isolated PDF Session RAG** — Upload any medical report and ask questions **only** about that document
- **Hindi Translation Support** — Auto-detects Hindi and translates answers naturally
- **Multilingual Voice Input & Output** (STT + TTS) — Works in English, Hindi, Marathi, Tamil
- **Self-Corrective & Safe Answering** — Never gives medical advice, always suggests consulting a doctor
- **Production-grade Flask Backend** with Ollama (llama3.2:3b)

---

## 🛠️ Tech Stack

| Layer          | Technology                          |
|----------------|-------------------------------------|
| Backend        | Python + Flask                      |
| LLM            | Ollama (llama3.2:3b)                |
| Embeddings     | all-MiniLM-L6-v2                    |
| Reranker       | cross-encoder/ms-marco-MiniLM-L-6-v2|
| Vector DB      | FAISS                               |
| Sparse Search  | BM25 (rank_bm25)                    |
| Frontend       | HTML5 + CSS3 + Vanilla JavaScript   |
| Others         | PyPDF2, langdetect, SpeechRecognition |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ViswaSudheer/MedRAG-Agentic-Hybrid-RAG-System.git
cd MedRAG-Agentic-Hybrid-RAG-System
