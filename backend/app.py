
# app.py - COMPLETE BACKEND WITH HINDI DETECTION + TRANSLATION
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import os
from langdetect import detect, DetectorFactory

# Very important: import here so init_indexes() runs during import
from ollama3_pr import agentic_rag_for_ui, med_hybrid, syn_hybrid, book_hybrid, create_pdf_session

# Fix langdetect randomness
DetectorFactory.seed = 0

app = Flask(__name__,
            static_folder='../static',
            static_url_path='')

CORS(app)

# ────────────────────────────────────────────────
#         DEBUG: SHOW INDEX STATUS AT STARTUP
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("FLASK STARTUP - INDEX STATUS CHECK")
print(f"MedQuAD documents: {len(med_hybrid.docs) if med_hybrid.docs else 0}")
print(f"Synthea documents: {len(syn_hybrid.docs) if syn_hybrid.docs else 0}")
print(f"Books   documents: {len(book_hybrid.docs)   if book_hybrid.docs else 0}")
print(f"Current working directory: {os.getcwd()}")
print("="*60 + "\n")

# ====================== LANGUAGE DETECTION & TRANSLATION ======================
def detect_language(text: str) -> str:
    try:
        lang = detect(text.lower())
        return 'hi' if lang in ['hi', 'mr', 'ne', 'bh'] else 'en'
    except:
        return 'en'


def translate_to_hindi(english_text: str) -> str:
    if not english_text or english_text.strip() == "":
        return english_text

    prompt = f"""You are an expert medical translator.
Translate the following medical answer into natural, simple Hindi.
Keep medical terms accurate (use common Indian Hindi medical words where appropriate).
Do not add extra explanations or greetings.

English:
{english_text}

Hindi:"""

    try:
        from ollama_utils import generate_llm
        hindi = generate_llm(prompt).strip()
        return hindi
    except Exception as e:
        print("Translation failed:", e)
        return english_text


# ====================== ROUTES ======================

@app.route('/')
def serve_frontend():
    return app.send_static_file('ragbot.html')


# @app.route('/api/chat', methods=['POST'])
# def chat():
#     try:
#         data = request.get_json()
#         question = data.get('question', '').strip()

#         if not question:
#             return jsonify({"success": False, "error": "No question provided"}), 400

#         detected_lang = detect_language(question)
#         print(f"🔍 Detected: {detected_lang} | Q: {question[:70]}...")

#         # ──── MAIN RAG CALL ────
#         rag_result = agentic_rag_for_ui(question)

#         english_answer = rag_result.get("answer", "").strip()

#         # If answer looks like fallback → log warning
#         if "don't have enough information" in english_answer.lower():
#             print("⚠️  RAG returned no useful context")

#         final_answer = english_answer
#         if detected_lang == 'hi':
#             print("→ Translating to Hindi...")
#             final_answer = translate_to_hindi(english_answer)

#         response = {
#             "answer": final_answer,
#             "original_answer": english_answer,
#             "detected_language": detected_lang,
#             "translated": detected_lang == 'hi',
#             "meta": rag_result.get("meta", {}),
#             "success": True
#         }

#         return jsonify(response)

#     except Exception as e:
#         import traceback
#         print("Backend Error:\n" + traceback.format_exc())
#         return jsonify({
#             "success": False,
#             "error": str(e),
#             "answer": "क्षमा करें, कुछ तकनीकी समस्या हुई। कृपया दोबारा प्रयास करें।"
#         }), 500


# @app.route('/api/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file"}), 400
    
#     file = request.files['file']
#     return jsonify({
#         "success": True,
#         "message": f"File received: {file.filename} (PDF indexing not yet implemented)"
#     })
# ====================== PDF UPLOAD (NEW FULLY WORKING) ======================
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file sent"}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400

    try:
        # Extract text in memory
        from PyPDF2 import PdfReader
        import io
        pdf_bytes = file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pdf_text = "".join(page.extract_text() or "" for page in reader.pages)

        if len(pdf_text.strip()) < 100:
            return jsonify({"success": False, "error": "PDF appears empty or unreadable"}), 400

        # Create isolated session index
        from ollama3_pr import create_pdf_session
        session_id = create_pdf_session(pdf_text)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": f"✅ PDF uploaded successfully ({len(reader.pages)} pages). Now ask questions about this document only.",
            "pages": len(reader.pages),
            "chars": len(pdf_text)
        })

    except Exception as e:
        print("Upload Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ====================== UPDATED CHAT ENDPOINT ======================
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        session_id = data.get('session_id')          # ← Frontend must send this

        if not question:
            return jsonify({"success": False, "error": "No question"}), 400

        detected_lang = detect_language(question)
        print(f"🔍 Lang: {detected_lang} | Session: {session_id or 'MAIN'} | Q: {question[:70]}...")

        # Pass session_id → switches to PDF-only mode automatically
        rag_result = agentic_rag_for_ui(question, session_id=session_id)

        english_answer = rag_result.get("answer", "")

        final_answer = english_answer
        if detected_lang == 'hi':
            final_answer = translate_to_hindi(english_answer)

        response = {
            "answer": final_answer,
            "original_answer": english_answer,
            "detected_language": detected_lang,
            "translated": detected_lang == 'hi',
            "session_id": session_id,
            "meta": rag_result.get("meta", {}),
            "success": True
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        print("Chat Error:\n", traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "answer": "Something went wrong. Please try again."
        }), 500

if __name__ == '__main__':
    print("\n🚀 Starting MedRAG Flask backend with Hindi support")
    print("   Visit →  http://127.0.0.1:5000")
    print("   API    →  POST /api/chat")
    app.run(host='0.0.0.0', port=5000, debug=True)