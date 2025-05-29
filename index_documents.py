import os
import re
import argparse
import numpy as np
import nltk
from dotenv import load_dotenv
from psycopg2 import connect, sql
from google.generativeai import configure, embed_content
from pypdf import PdfReader
from docx import Document

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise EnvironmentError("Missing GEMINI_API_KEY or POSTGRES_URL in .env file.")

# Configure Gemini
configure(api_key=GEMINI_API_KEY)

# Ensure punkt for sentence splitting
nltk.download('punkt', quiet=True)

# ========== Text Extraction ==========

def extract_text(filepath: str) -> str | None:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            return " ".join((page.extract_text() or "") for page in reader.pages).strip()
        elif ext == ".docx":
            document = Document(filepath)
            return " ".join(p.text for p in document.paragraphs).strip()
        else:
            print(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        print(f"Failed to extract text from {filepath}: {e}")
        return None

# ========== Splitting Strategies ==========

def split_fixed(text: str, size=1000, overlap=200) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

def split_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split('\n\n') if p.strip()]

SPLIT_STRATEGIES = {
    "fixed_size": split_fixed,
    "sentence": split_sentences,
    "paragraph": split_paragraphs
}

# ========== Embedding ==========

def get_embedding(text: str) -> np.ndarray | None:
    if not text.strip():
        return None
    try:
        response = embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_DOCUMENT")
        return np.array(response['embedding'])
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# ========== Database ==========

def connect_db():
    try:
        return connect(POSTGRES_URL)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None

def insert_chunk(conn, chunk: str, embedding: np.ndarray, filename: str, strategy: str) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    INSERT INTO documents (chunk_text, embedding, filename, split_strategy)
                    VALUES (%s, %s::vector, %s, %s);
                """),
                (chunk, f"[{','.join(map(str, embedding.tolist()))}]", filename, strategy)
            )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Insertion failed: {e}")
        return False

# ========== Main Indexer ==========

def index_file(filepath: str, strategy_name: str):
    filename = os.path.basename(filepath)
    text = extract_text(filepath)

    if not text:
        print(f"No text found in file: {filename}")
        return

    if strategy_name not in SPLIT_STRATEGIES:
        print(f"Invalid strategy '{strategy_name}'. Choose from {list(SPLIT_STRATEGIES)}")
        return

    chunks = SPLIT_STRATEGIES[strategy_name](text)
    print(f"{len(chunks)} chunks created using '{strategy_name}' strategy.")

    conn = connect_db()
    if not conn:
        return

    indexed = 0
    try:
        for chunk in chunks:
            if not chunk.strip():
                continue
            emb = get_embedding(chunk)
            if emb is not None and insert_chunk(conn, chunk, emb, filename, strategy_name):
                indexed += 1
    finally:
        conn.close()

    print(f"Indexed {indexed} chunks from '{filename}'")

# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="Index documents with Gemini embeddings into PostgreSQL.")
    parser.add_argument("filepath", help="Path to the PDF or DOCX file.")
    parser.add_argument("--strategy", choices=SPLIT_STRATEGIES.keys(), default="paragraph")
    args = parser.parse_args()
    index_file(args.filepath, args.strategy)

if __name__ == "__main__":
    main()
