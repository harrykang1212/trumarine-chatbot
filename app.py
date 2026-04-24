import os
import json
import math
import threading
import datetime
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

conversation_history = []
chunks_store = []
tfs_store = []
df_store = Counter()
index_ready = False
index_error = None

TEXT_PATH      = os.path.join(os.path.dirname(__file__), "manual_text.txt")
ANALYTICS_PATH = os.path.join(os.path.dirname(__file__), "analytics_log.json")

# ── Topic categories Claude will classify questions into ──
TOPICS = [
    "Cleaning & maintenance",
    "Torque specifications",
    "Fault diagnosis",
    "Tools & equipment",
    "Safety procedures",
    "Parts & components",
    "Installation & assembly",
    "Company information",
    "Pricing & availability",
    "General enquiry",
]


# ── Analytics helpers ──────────────────────────────────────
def load_analytics():
    """Load existing analytics log or return empty structure."""
    if os.path.exists(ANALYTICS_PATH):
        try:
            with open(ANALYTICS_PATH, "r") as f:
                return json.load(f)
        except:
            pass
    return {"topics": {}, "total": 0, "logs": []}


def save_analytics(data):
    """Save analytics log to file."""
    try:
        with open(ANALYTICS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Analytics save error: {e}")


def classify_topic(question: str) -> str:
    """Ask Claude to classify the question into a topic."""
    try:
        topics_list = "\n".join(f"- {t}" for t in TOPICS)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"""Classify this customer question into exactly one of these topics:
{topics_list}

Question: "{question}"

Reply with ONLY the topic name, nothing else."""
            }]
        )
        topic = msg.content[0].text.strip()
        # Make sure it matches one of our topics
        for t in TOPICS:
            if t.lower() in topic.lower() or topic.lower() in t.lower():
                return t
        return "General enquiry"
    except:
        return "General enquiry"


def log_question(question: str, topic: str):
    """Log question and topic to analytics file."""
    try:
        data = load_analytics()
        data["total"] = data.get("total", 0) + 1
        data["topics"][topic] = data["topics"].get(topic, 0) + 1
        data["logs"].append({
            "question": question[:200],
            "topic": topic,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        # Keep only last 500 logs to avoid file getting too big
        if len(data["logs"]) > 500:
            data["logs"] = data["logs"][-500:]
        save_analytics(data)
    except Exception as e:
        print(f"Logging error: {e}")


# ── TF-IDF search ──────────────────────────────────────────
def tokenize(text):
    return text.lower().split()


def build_tfidf_index(chunks):
    df = Counter()
    tfs = []
    for chunk in chunks:
        tokens = tokenize(chunk)
        tf = Counter(tokens)
        tfs.append(tf)
        for word in set(tokens):
            df[word] += 1
    return tfs, df


def tfidf_search(query, k=20):
    n = len(chunks_store)
    query_tokens = tokenize(query)
    scores = []
    for i, (chunk, tf) in enumerate(zip(chunks_store, tfs_store)):
        score = 0
        for token in query_tokens:
            if token in tf:
                idf = math.log((n + 1) / (df_store.get(token, 0) + 1)) + 1
                score += tf[token] * idf
        scores.append((score, i))
    scores.sort(reverse=True)
    return [chunks_store[i] for _, i in scores[:k]]


def build_index():
    global chunks_store, tfs_store, df_store, index_ready, index_error
    try:
        print("⏳ Loading text file...")
        with open(TEXT_PATH, "r", encoding="utf-8") as f:
            full_text = f.read()
        print(f"   Loaded {len(full_text)} characters")

        chunk_size, overlap = 1000, 150
        chunks, start = [], 0
        while start < len(full_text):
            chunk = full_text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap

        print(f"   Split into {len(chunks)} chunks")
        tfs, df = build_tfidf_index(chunks)
        chunks_store = chunks
        tfs_store = tfs
        df_store = df
        index_ready = True
        print(f"✅ Ready! {len(chunks)} chunks indexed.")

    except Exception as e:
        index_error = str(e)
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


threading.Thread(target=build_index, daemon=True).start()


def build_context(chunks, max_chars=6000):
    parts, total = [], 0
    for t in chunks:
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

    candidates = tfidf_search(query, k=20)
    context = build_context(candidates, max_chars=6000)

    system_prompt = f"""You are TruAssist, a representative of Tru-Marine — a turbocharger service and supply company headquartered in Singapore.
 
IDENTITY & TONE:
- You ARE Tru-Marine. Always speak in first-person plural: "we", "our", "us".
- Never say "they", "them", or refer to Tru-Marine as a third party.
- Speak like a confident, professional Tru-Marine employee — not an external assistant.
- Be warm, direct and solution-focused.
 
ANSWERING QUESTIONS:
- Use the reference text below to answer questions. Use semantic understanding — match meaning, not just keywords.
- If a question is about pricing, scopes of work, or assessments, explain that we tailor our responses based on actual inspection findings, and offer to assist further.
- Never say "I don't know", "I don't have that information", "I cannot find it", or "it's not in my reference material".
- Instead, confidently acknowledge the question and guide the customer to next steps.
 
WHEN YOU CANNOT FULLY ANSWER:
- Respond confidently. For example: "We can certainly assist with that. To provide a precise answer, we'll need a few more details..."
- Always offer to help gather the right information — vessel details, location, issue description, urgency.
- Always end with contact options:
  📞 +65 9827 3471
  📧 turbo@trumarine.com
  [WhatsApp us] or [Send us an enquiry] (remind the customer these options are available in the chat)
 
GLOBAL SUPPORT QUESTIONS:
- If asked about support in a specific country or region, confirm that we provide global service coverage.
- Offer to check: nearest service support, estimated response time, spare parts availability.
- Ask for vessel details or current port location to advise accordingly.
 
UNCERTAIN OR COMPLEX QUESTIONS:
- Never say we cannot help. Instead, say we will assess and advise.
- Use confident language: "We can certainly assist", "Our team will evaluate", "Once we have these details, we can provide a clear quotation."
- Replace any "I don't have" or "I cannot confirm" with action-oriented responses.
 
PRICING QUESTIONS:
- Never give a flat price. Explain that pricing is based on actual condition and scope of work after inspection.
- Structure the response around: inspection findings, scope of work, spare parts needed — then offer to provide a quotation once details are shared.
 
RESPONSE FORMAT:
- Be concise and structured. Use bullet points where helpful.
- End responses that need follow-up with contact details and a reminder that WhatsApp and enquiry form options are available in the chat.
- Do not add unnecessary filler like "That's a great question" more than once per conversation.
- Remember the full conversation history for follow-up questions.
 
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
        answer = answer or "I couldn't find a clear answer. Please contact our sales team."

        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        # Log topic in background so it doesn't slow down the response
        threading.Thread(
            target=lambda: log_question(query, classify_topic(query)),
            daemon=True
        ).start()

        return answer

    except Exception as e:
        return f"Sorry, there was a problem reaching the AI. ({e})"


# ── Flask routes ───────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    if index_error:
        return jsonify({"status": "error", "message": index_error}), 500
    if not index_ready:
        return jsonify({"status": "loading", "message": "Still starting up, please wait..."}), 200
    return jsonify({"status": "ok", "message": "TurboAssist is fully ready!"})


@app.route("/chat", methods=["POST"])
def chat():
    if not index_ready:
        return jsonify({"answer": "The server is still starting up, please wait a moment!"}), 200
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


@app.route("/analytics/reset", methods=["POST"])
def reset_analytics():
    empty = {"topics": {}, "total": 0, "logs": []}
    save_analytics(empty)
    return jsonify({"status": "ok", "message": "Analytics reset!"})


@app.route("/analytics", methods=["GET"])
def analytics():
    """Returns analytics data for the dashboard."""
    data = load_analytics()
    return jsonify(data)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
