"""
========================================================================================
STARBUCKS AI BARISTA ASSISTANT 🍵☕🤖
========================================================================================
RAG-Based AI Chat Assistant for Starbucks Store

Features:
1. RAG Application: Load Starbucks knowledge base (TXT), chunk text, and retrieve
   relevant context using lightweight TF-IDF similarity search.
2. Web Chat Interface: Flask-based modern web UI.
3. Starbucks Theme: Green-themed UI matching Starbucks brand identity.

Knowledge Base Source : starbucks_knowledge_base.txt
Retrieval Engine      : TF-IDF Cosine Similarity (pure Python — no heavy ML deps)
LLM Engine            : Groq LLM
Web Framework         : Flask
========================================================================================
"""

import os
import sys
import json
import math
import re
from collections import Counter
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# Configure UTF-8 output for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for RAG components (initialized at startup)
chunks_store = []
tfidf_vectors = []
idf_dict = {}
vocabulary = set()
groq_client = None


# ======================================================================================
# TF-IDF RETRIEVAL ENGINE (Pure Python — no ML dependencies)
# ======================================================================================
def tokenize(text):
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r'[a-z0-9]+', text.lower())


def compute_tf(tokens):
    """Compute term frequency for a list of tokens."""
    counter = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counter.items()} if total > 0 else {}


def compute_idf(documents_tokens):
    """Compute inverse document frequency across all documents."""
    n = len(documents_tokens)
    idf = {}
    all_words = set()
    for tokens in documents_tokens:
        all_words.update(tokens)
    for word in all_words:
        doc_count = sum(1 for tokens in documents_tokens if word in set(tokens))
        idf[word] = math.log((n + 1) / (doc_count + 1)) + 1  # smoothed IDF
    return idf


def compute_tfidf_vector(tokens, idf):
    """Compute TF-IDF vector for a document."""
    tf = compute_tf(tokens)
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two sparse vectors (dicts)."""
    common_words = set(vec1.keys()) & set(vec2.keys())
    if not common_words:
        return 0.0
    dot_product = sum(vec1[w] * vec2[w] for w in common_words)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


# ======================================================================================
# STEP 1: LOAD & PARSE THE STARBUCKS KNOWLEDGE BASE (.txt)
# ======================================================================================
def load_starbucks_knowledge_base(file_path="starbucks_knowledge_base.txt"):
    """
    Loads the Starbucks knowledge base from a plain text file.
    Returns the full text as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: Knowledge Base '{file_path}' not found! "
            "Please ensure starbucks_knowledge_base.txt is in the project directory."
        )

    print(f"STEP 1: Loading Starbucks knowledge base from '{file_path}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    paragraphs = [line.strip() for line in full_text.split('\n') if line.strip()]
    print(f"Loaded! {len(full_text)} characters | {len(paragraphs)} non-empty lines.")
    return full_text


# ======================================================================================
# STEP 2: CHUNK TEXT FOR RETRIEVAL
# ======================================================================================
def chunk_knowledge_base(text):
    """
    Splits the full text into overlapping chunks using RecursiveCharacterTextSplitter.
    """
    print("STEP 2: Splitting knowledge base into text chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks.")
    return chunks


# ======================================================================================
# STEP 3: BUILD TF-IDF INDEX
# ======================================================================================
def build_tfidf_index(chunks):
    """
    Builds a TF-IDF index from the text chunks.
    Pure Python implementation — no external ML libraries needed.
    """
    print("STEP 3: Building TF-IDF search index...")
    documents_tokens = [tokenize(chunk) for chunk in chunks]
    idf = compute_idf(documents_tokens)
    vectors = [compute_tfidf_vector(tokens, idf) for tokens in documents_tokens]
    print(f"TF-IDF index built for {len(chunks)} chunks ({len(idf)} unique terms).")
    return vectors, idf


# ======================================================================================
# STEP 4: RAG SEARCH — RETRIEVE RELEVANT CONTEXT
# ======================================================================================
def retrieve_relevant_context(user_query, top_k=4):
    """
    Converts user query into TF-IDF vector, computes cosine similarity against all
    chunks, and returns top K matching chunks combined as context for LLM.
    """
    query_tokens = tokenize(user_query)
    query_vector = compute_tfidf_vector(query_tokens, idf_dict)

    # Compute similarities
    similarities = []
    for i, doc_vec in enumerate(tfidf_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        similarities.append((sim, i))

    # Sort by similarity (descending) and take top_k
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_indices = [idx for _, idx in similarities[:top_k]]

    retrieved_docs = [chunks_store[i] for i in top_indices]
    combined_context = "\n---\n".join(retrieved_docs)
    return combined_context


# ======================================================================================
# STEP 5: GENERATE AI BARISTA RESPONSE USING GROQ LLM
# ======================================================================================
def generate_barista_response(user_query, context):
    """
    Sends retrieved Starbucks knowledge base context and user query to Groq LLM.
    Returns the AI Barista's response as a string.
    """
    system_prompt = """You are Alex, the friendly and knowledgeable Starbucks AI Barista Assistant.

YOUR PERSONA:
- Warm, enthusiastic, and passionate about coffee and Starbucks drinks
- Use coffee emojis naturally but don't overdo it
- Sound like a real friendly Starbucks barista, not a robot
- Address the customer warmly

YOUR GUIDELINES:
1. KNOWLEDGE: Answer ONLY based on the retrieved Starbucks knowledge base context provided. Do NOT make up prices, items, or policies.
2. RECOMMENDATIONS: When a customer seems unsure, proactively suggest popular options matching their preference (sweet, strong, cold, etc.)
3. CUSTOMIZATIONS: Mention customization options relevant to the customer's question (milk alternatives, syrups, etc.)
4. FORMAT: Use clear formatting with bullet points or numbered lists when listing multiple items. Keep responses friendly but informative.
5. UNAVAILABLE INFO: If a specific detail isn't in the context, say: "I don't have that exact detail handy, but you can check the Starbucks app or ask your local barista for the most current info!"
6. UPSELLING: Naturally suggest add-ons or related items when appropriate (e.g., pair a drink with food).
7. SEASONAL: Mention when items are seasonal or limited-time.

Remember: You represent the Starbucks brand — be positive, helpful, and make the customer feel valued!
"""

    user_prompt = f"""STARBUCKS KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER QUESTION:
{user_query}

YOUR RESPONSE AS STARBUCKS AI BARISTA:"""

    # Try models in order of preference
    candidate_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]

    for model_name in candidate_models:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception:
            continue

    return "I'm sorry, I'm having a little trouble right now. Please try again shortly!"


# ======================================================================================
# FLASK ROUTES — WEB APPLICATION ENDPOINTS
# ======================================================================================

@app.route("/")
def index():
    """Serve the Starbucks AI Barista chat interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    API endpoint for chat messages.
    Accepts JSON: {"message": "user question"}
    Returns JSON: {"response": "AI barista answer"}
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data["message"].strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # Step 1: Retrieve context via TF-IDF search
        context = retrieve_relevant_context(user_message)

        # Step 2: Generate AI response
        ai_response = generate_barista_response(user_message, context)

        return jsonify({
            "response": ai_response,
            "status": "success"
        })
    except Exception as e:
        print(f"Error handling chat request: {e}")
        return jsonify({
            "response": "Hello! I had a slight hiccup looking that up. Please ask again or try another question!",
            "status": "error"
        }), 200


@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    """Returns preset quick-question suggestions for the chat UI."""
    suggestions = [
        "What are your most popular drinks?",
        "What cold drinks do you recommend for summer?",
        "Do you have any dairy-free options?",
        "What is the Pumpkin Spice Latte?",
        "Tell me about Frappuccinos",
        "What food items do you have?",
        "How does the Starbucks Rewards program work?",
        "What is the Pink Drink?",
        "What are your strongest coffee options?",
        "Show me breakfast sandwich options",
    ]
    return jsonify({"suggestions": suggestions})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "knowledge_base": "starbucks_knowledge_base.txt",
        "retrieval": "TF-IDF Cosine Similarity",
        "llm": "Groq LLM",
        "chunks_indexed": len(chunks_store)
    })


# ======================================================================================
# APPLICATION STARTUP
# ======================================================================================
def initialize_app():
    """Initialize all RAG components at application startup."""
    global chunks_store, tfidf_vectors, idf_dict, groq_client

    print("=" * 60)
    print("STARBUCKS AI BARISTA ASSISTANT — INITIALIZING")
    print("=" * 60)

    # Verify Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found!")
        print("Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    print("Groq API client initialized.")

    # Load knowledge base
    kb_text = load_starbucks_knowledge_base("starbucks_knowledge_base.txt")

    # Chunk it
    chunks_store = chunk_knowledge_base(kb_text)

    # Build TF-IDF index (pure Python — no ML models!)
    tfidf_vectors, idf_dict = build_tfidf_index(chunks_store)

    print("=" * 60)
    print("STARBUCKS AI BARISTA IS READY!")
    print("=" * 60)


if __name__ == "__main__":
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)
else:
    # When launched via WSGI servers like Gunicorn (e.g., on Render)
    initialize_app()
