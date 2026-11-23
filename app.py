"""
========================================================================================
STARBUCKS AI BARISTA ASSISTANT 🍵☕🤖
========================================================================================
RAG-Based AI Chat Assistant for Starbucks Store

Features:
1. RAG Application: Load Starbucks knowledge base (TXT), chunk text, generate vector
   embeddings, store in ChromaDB, and retrieve relevant context.
2. Web Chat Interface: Flask-based modern web UI with real-time streaming responses.
3. Starbucks Theme: Green-themed UI matching Starbucks brand identity.

Knowledge Base Source : starbucks_knowledge_base.txt
Embedding Model       : SentenceTransformer ('all-MiniLM-L6-v2')
Vector Database       : ChromaDB (Local Persistent Storage)
LLM Engine            : Groq Llama-3.3-70b-versatile
Web Framework         : Flask with Server-Sent Events (SSE) for streaming
========================================================================================
"""

import os
import sys
import json
import time
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
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
embedding_model = None
collection = None
groq_client = None

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
            f"❌ Error: Knowledge Base '{file_path}' not found! "
            "Please ensure starbucks_knowledge_base.txt is in the project directory."
        )

    print(f"📖 STEP 1: Loading Starbucks knowledge base from '{file_path}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split into non-empty lines for paragraph counting
    paragraphs = [line.strip() for line in full_text.split('\n') if line.strip()]
    print(f"✅ Loaded! {len(full_text)} characters | {len(paragraphs)} non-empty lines.")
    return full_text


# ======================================================================================
# STEP 2: CHUNK TEXT FOR VECTOR RETRIEVAL
# ======================================================================================
def chunk_knowledge_base(text):
    """
    Splits the full text into overlapping chunks using RecursiveCharacterTextSplitter.
    Chunk size of 500 chars with 80 overlap balances context richness and retrieval speed.
    """
    print("\n✂️  STEP 2: Splitting knowledge base into manageable text chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks


# ======================================================================================
# STEP 3: INITIALIZE EMBEDDINGS & CHROMADB VECTOR DATABASE
# ======================================================================================
def initialize_vector_store(chunks, db_path="./chroma_starbucks_db"):
    """
    Generates 384-dimensional vector embeddings for each chunk using ChromaDB's
    built-in ONNX embedding function (all-MiniLM-L6-v2 via ONNX Runtime).
    Much lighter on memory than PyTorch-based SentenceTransformers.
    Re-uses existing index if already populated.
    """
    print("\n🧠 STEP 3: Loading ONNX embedding model ('all-MiniLM-L6-v2')...")
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

    print(f"💾 Initializing ChromaDB vector store at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    # Get or create collection with built-in embedding function
    col = client.get_or_create_collection(
        name="starbucks_barista_kb",
        embedding_function=emb_fn
    )

    if col.count() == 0:
        print(f"⏳ Embedding and indexing {len(chunks)} chunks into ChromaDB...")
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "starbucks_knowledge_base.txt", "chunk_id": i} for i in range(len(chunks))]

        # ChromaDB will auto-embed the documents using the ONNX embedding function
        col.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"✅ Indexed {col.count()} chunks into ChromaDB successfully!")
    else:
        print(f"✅ Found existing vector collection with {col.count()} indexed chunks.")

    return col, emb_fn


# ======================================================================================
# STEP 4: RAG SEMANTIC SEARCH — RETRIEVE RELEVANT CONTEXT
# ======================================================================================
def retrieve_relevant_context(user_query, top_k=4):
    """
    ChromaDB auto-embeds the query text using its built-in embedding function,
    performs cosine similarity search, and returns top K matching chunks.
    """
    results = collection.query(
        query_texts=[user_query],
        n_results=top_k
    )
    retrieved_docs = results.get("documents", [[]])[0]
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
    system_prompt = """You are Alex, the friendly and knowledgeable Starbucks AI Barista Assistant. 🍵☕

YOUR PERSONA:
- Warm, enthusiastic, and passionate about coffee and Starbucks drinks
- Use coffee emojis naturally (☕, 🍵, 🎃, 💚, ⭐) but don't overdo it
- Sound like a real friendly Starbucks barista, not a robot
- Address the customer warmly

YOUR GUIDELINES:
1. KNOWLEDGE: Answer ONLY based on the retrieved Starbucks knowledge base context provided. Do NOT make up prices, items, or policies.
2. RECOMMENDATIONS: When a customer seems unsure, proactively suggest popular options matching their preference (sweet, strong, cold, etc.)
3. CUSTOMIZATIONS: Mention customization options relevant to the customer's question (milk alternatives, syrups, etc.)
4. FORMAT: Use clear formatting with bullet points or numbered lists when listing multiple items. Keep responses friendly but informative.
5. UNAVAILABLE INFO: If a specific detail isn't in the context, say: "I don't have that exact detail handy, but you can check the Starbucks app or ask your local barista for the most current info! ☕"
6. UPSELLING: Naturally suggest add-ons or related items when appropriate (e.g., pair a drink with food).
7. SEASONAL: Mention when items are seasonal or limited-time.

Remember: You represent the Starbucks brand — be positive, helpful, and make the customer feel valued! 💚
"""

    user_prompt = f"""STARBUCKS KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER QUESTION:
{user_query}

YOUR RESPONSE AS STARBUCKS AI BARISTA:"""

    # Try models in order of preference (updated to currently available models)
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

    return "I'm sorry, I'm having a little trouble right now. Please try again shortly! ☕"


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
    Returns JSON: {"response": "AI barista answer", "context_preview": "..."}
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data["message"].strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # Step 1: Retrieve context from ChromaDB
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
            "response": "Hello! I had a slight hiccup looking that up. Please ask again or try another question! ☕",
            "status": "error"
        }), 200


@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    """Returns preset quick-question suggestions for the chat UI."""
    suggestions = [
        "What are your most popular drinks? ☕",
        "What cold drinks do you recommend for summer? 🧊",
        "Do you have any dairy-free options? 🌱",
        "What is the Pumpkin Spice Latte? 🎃",
        "Tell me about Frappuccinos 🥤",
        "What food items do you have? 🥐",
        "How does the Starbucks Rewards program work? ⭐",
        "What is the Pink Drink? 🌸",
        "What are your strongest coffee options? 💪",
        "Show me breakfast sandwich options 🥪",
    ]
    return jsonify({"suggestions": suggestions})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "knowledge_base": "starbucks_knowledge_base.txt",
        "model": "all-MiniLM-L6-v2",
        "llm": "Groq Llama-3.3-70b-versatile",
        "chunks_indexed": collection.count() if collection else 0
    })


# ======================================================================================
# APPLICATION STARTUP
# ======================================================================================
def initialize_app():
    """Initialize all RAG components at application startup."""
    global embedding_model, collection, groq_client

    print("\n" + "=" * 70)
    print("☕  STARBUCKS AI BARISTA ASSISTANT — INITIALIZING")
    print("=" * 70)

    # Verify Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file!")
        print("   Please create a .env file with: GROQ_API_KEY=your_api_key_here")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    print("✅ Groq API client initialized.")

    # Load knowledge base
    kb_text = load_starbucks_knowledge_base("starbucks_knowledge_base.txt")

    # Chunk it
    chunks = chunk_knowledge_base(kb_text)

    # Initialize vector store
    collection, embedding_model = initialize_vector_store(chunks)

    print("\n" + "=" * 70)
    print("✅ STARBUCKS AI BARISTA IS READY! 🍵☕")
    print("   Open your browser at: http://localhost:5000")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)
else:
    # When launched via WSGI servers like Gunicorn (e.g., on Render)
    initialize_app()
