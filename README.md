# Document RAG System with Google Gemini and PostgreSQL

This project implements a basic Retrieval-Augmented Generation (RAG) system using Google Gemini for embeddings and PostgreSQL as the vector database. It includes two main Python scripts: `index_documents.py` for processing and storing documents, and `search_documents.py` for querying the indexed data.

## Features

* **Document Ingestion:** Supports PDF and DOCX file formats.
* **Text Chunking Strategies:**
    * Fixed-size with overlap
    * Sentence-based splitting
    * Paragraph-based splitting (default)
* **Gemini Embeddings:** Generates high-quality embeddings for both document chunks and user queries using Google's `embedding-001` model.
* **PostgreSQL Storage:** Stores document chunks, their embeddings, and metadata in a PostgreSQL database with a `vector` extension.
* **Semantic Search:** Finds the most relevant document chunks based on cosine similarity to a user's query.
* **Secure Configuration:** Uses `.env` for API keys and database credentials.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.9+** installed.
2.  **Google Cloud Project & Gemini API Key:**
    * Enable the Gemini API (formerly Generative Language API) in your Google Cloud project.
    * Generate an API key for your project.
3.  **PostgreSQL Database:**
    * A running PostgreSQL instance (e.g., local installation, ElephantSQL, Supabase).
    * Ensure the `vector` extension is enabled in your database. You can do this by running the following SQL command in your database's SQL editor (e.g., in Supabase's SQL Editor):
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        ```
    * Create the `documents` table with the following schema:
        ```sql
        CREATE TABLE documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_text TEXT NOT NULL,
            embedding VECTOR(768), -- IMPORTANT: Gemini's embedding-001 model produces 768-dimensional vectors.
            filename TEXT NOT NULL,
            split_strategy TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        ```
        **Important Note on Embedding Dimension:**
        The `embedding-001` model actually produces 768-dimensional embeddings. **Please ensure your `documents` table's `embedding` column is defined as `VECTOR(768)`.** If you previously used `VECTOR(1536)` (or any other dimension), you will need to **drop and re-create the table with `VECTOR(768)`** and then re-index your documents.

## Setup

1.  **Clone the repository (after you create it on GitHub) or download the files:**
    ```bash
    git clone <your-repo-url>
    cd document-rag-system # Or whatever your folder name is
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install python-dotenv psycopg2-binary google-generativeai numpy scikit-learn pypdf python-docx nltk
    ```
    (Note: `psycopg2-binary` is generally easier to install than `psycopg2` directly. `nltk` is required for sentence splitting.)

5.  **Create a `.env` file:**
    In the root directory of your project (where `index_documents.py` and `search_documents.py` are), create a file named `.env` and add your credentials. **Do NOT upload this file to GitHub.**
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    POSTGRES_URL="postgresql://user:password@host:port/database"
    ```
    * Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API Key.
    * Replace `"postgresql://user:password@host:port/database"` with your PostgreSQL connection string (e.g., from your Supabase project settings -> Database -> Connection string -> URI).

## Usage

### 1. Indexing Documents

Use `index_documents.py` to extract text, split it into chunks, generate embeddings, and store them in your PostgreSQL database.

```bash
python index_documents.py <path_to_your_document> --strategy <splitting_strategy>