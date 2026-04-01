import os
import hashlib
from typing import List
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

# ── Langchain imports ──────────────────────────────────
from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Flask setup ────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Anthropic client ───────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Conversation history ───────────────────────────────
conversation_history = []

# ── Document path ──────────────────────────────────────
# Place your .docx file in the same folder as this app.py
DOCX_PATH = os.path.join(os.path.dirname(__file__), "manual.docx")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ── Helper: clean documents ───────────────────────────
def clean_docs(docs: List[LCDocument]) -> List[LCDocument]:
    cleaned, seen = [], set()
    for d in docs or []:
        text = (d.page_content or "").strip()
        if not text:
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        cleaned.append(LCDocument(
            page_content=text,
            metadata=getattr(d, "metadata", {}) or {}
        ))
    return cleaned

# ── Build or load index ────────────────────────────────
print("⏳ Loading document and building index...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader = UnstructuredWordDocumentLoader(DOCX_PATH)
documents = loader.load()
docs = clean_docs(documents)

if not docs:
    raise ValueError("No usable text found in the DOCX file.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = clean_docs(splitter.split_documents(docs))
print(f"   Loaded {len(chunks)} chunks")

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 12

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("⚡ Loading existing Chroma index...")
    vector_store = Chroma(
        collection_name="turbo_chunks",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
else:
    print("⏳ Building Chroma index (first time only)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="turbo_chunks",
        persist_directory=CHROMA_DIR,
    )

print("✅ Index ready.")

# ── RAG helpers ────────────────────────────────────────
def build_context(docs_list, max_chars=3500):
    parts, total = [], 0
    for d in docs_list:
        t = (d.page_content or "").strip()
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

    # BM25 retrieval
    try:
        bm25_retriever.k = 12
        bm25_results = bm25_retriever.invoke(query) or []
    except:
        bm25_results = []

    # Vector retrieval
    try:
        vector_results = vector_store.similarity_search(query, k=12) or []
    except:
        vector_results = []

    # Merge + dedupe
    unique = {}
    for d in bm25_results + vector_results:
        text = (getattr(d, "page_content", "") or "").strip()
        if text and text not in unique:
            unique[text] = d

    context = build_context(list(unique.values()), max_chars=3500)

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


# ── Flask routes ───────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "TurboAssist is running"})


@app.route("/chat", methods=["POST"])
def chat():
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


# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
