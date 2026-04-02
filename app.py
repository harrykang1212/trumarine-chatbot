import os
import hashlib
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

conversation_history = []
chunks_store = []
index_ready = False
index_error = None

DOCX_PATH = os.path.join(os.path.dirname(__file__), "307_Service_Manual_Ed2_Iss2.docx")


def clean_docs(docs):
    cleaned, seen = [], set()
    for d in docs or []:
        text = (d.page_content or "").strip()
        if not text:
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        cleaned.append(d)
    return cleaned


def build_index():
    global chunks_store, index_ready, index_error
    try:
        print("⏳ Loading document...")
        print(f"   File path: {DOCX_PATH}")
        print(f"   File exists: {os.path.exists(DOCX_PATH)}")

        from docx import Document
        doc = Document(DOCX_PATH)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        # Split into chunks manually — no heavy libraries needed
        chunk_size = 1000
        overlap = 150
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap

        chunks_store = chunks
        print(f"✅ Loaded {len(chunks)} chunks — ready!")
        index_ready = True

    except Exception as e:
        index_error = str(e)
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


threading.Thread(target=build_index, daemon=True).start()


def search_chunks(query, k=8):
    """Simple keyword search — no embeddings needed"""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks_store:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True)
    return [c for _, c in scored[:k]]


def build_context(results, max_chars=3500):
    parts, total = [], 0
    for t in results:
        t = t.strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            t = t[:max_chars - total]
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(parts) if parts else "No relevant information found."


def run_chatbot(query: str) -> str:
    global conversation_history

    results = search_chunks(query, k=8)
    context = build_context(results, max_chars=3500)

    system_prompt = f"""You are TurboAssist, an expert assistant for turbocharger service manuals.
Answer the user's question using ONLY the reference text below.
If the answer is not in the text, say you cannot find it in the manual.
Be concise and practical. Remember the conversation history for context.

Reference text:
{context}"""

    conversation_history.append({"role": "user", "content": query})
    trimmed_history = conversation_history[-10:]

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=trimmed_history,
        )
        answer = msg.content[0].text.strip()
        while "\n\n\n" in answer:
            answer = answer.replace("\n\n\n", "\n\n")
        answer = answer or "I couldn't find a clear answer in the manual."

        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return answer

    except Exception as e:
        return f"Sorry, there was a problem reaching the AI. ({e})"


@app.route("/health", methods=["GET"])
def health():
    if index_error:
        return jsonify({"status": "error", "message": index_error}), 500
    if not index_ready:
        return jsonify({"status": "loading", "message": "Still starting up, please wait..."}), 200
    return jsonify({"status": "ok", "message": "TurboAssist is running"})


@app.route("/chat", methods=["POST"])
def chat():
    if not index_ready:
        return jsonify({"answer": "The server is still starting up, please wait about a minute and try again!"}), 200
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    answer = run_chatbot(query)
    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok", "message": "Conversation reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
