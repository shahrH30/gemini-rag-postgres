import os
import argparse
import numpy as np
from dotenv import load_dotenv
from psycopg2 import connect, sql
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise EnvironmentError("Missing GEMINI_API_KEY or POSTGRES_URL")

genai.configure(api_key=GEMINI_API_KEY)

# --- Helper: Parse embedding string to numpy array ---
def parse_embedding(embedding_str: str) -> np.ndarray:
    return np.fromstring(embedding_str.strip("[]"), sep=",")

# --- Fetch all document embeddings ---
def fetch_embeddings(conn) -> tuple[list[str], np.ndarray]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_text, embedding FROM documents;")
            rows = cur.fetchall()
        texts = [row[0] for row in rows]
        vectors = np.stack([parse_embedding(row[1]) for row in rows])
        print(f"✅ Fetched {len(texts)} documents from database.")
        return texts, vectors
    except Exception as e:
        print(f"❌ Error fetching embeddings: {e}")
        return [], np.array([])

# --- Connect to database ---
def connect_db():
    try:
        conn = connect(POSTGRES_URL)
        print("🟢 Connected to PostgreSQL.")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return None

# --- Generate embedding for the query ---
def get_query_embedding(text: str) -> np.ndarray | None:
    if not text.strip():
        return None
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return np.array(response["embedding"])
    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        return None

# --- Search logic ---
def search_documents(query: str, top_k: int = 5):
    print(f"\n🔍 Searching for: \"{query}\"")
    query_emb = get_query_embedding(query)
    if query_emb is None:
        print("⚠️ Cannot generate embedding for query.")
        return

    conn = connect_db()
    if not conn:
        return

    try:
        texts, vectors = fetch_embeddings(conn)
        if not texts or vectors.size == 0:
            print("⚠️ No embeddings found in database.")
            return

        scores = cosine_similarity([query_emb], vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        print("\n📚 Top Results:\n")
        for i, idx in enumerate(top_indices, 1):
            print(f"{i}. Score: {scores[idx]:.4f}")
            print(f"{texts[idx]}\n")
    finally:
        conn.close()
        print("🔒 Database connection closed.")

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Semantic search on indexed documents.")
    parser.add_argument("query", type=str, nargs="+", help="Your search query")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to return (default: 5)")
    args = parser.parse_args()

    query_text = " ".join(args.query)
    search_documents(query_text, top_k=args.top)

if __name__ == "__main__":
    main()
