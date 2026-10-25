# Hybrid RAG Assistant

A document question-answering system built with **Retrieval-Augmented Generation (RAG)** that combines semantic search, keyword-based retrieval, reciprocal rank fusion, and cross-encoder reranking to retrieve relevant information from uploaded documents and generate grounded answers using an LLM.

The project uses FastAPI for the backend, React and TypeScript for the frontend, Supabase for persistent document and vector storage, and Groq for LLM inference.

> **Project status:** Local development version. Cloud deployment is not currently configured.

## Features

- **Document ingestion:** Upload and process documents into searchable text chunks.
- **Dense retrieval:** Use BGE embeddings to find semantically relevant document chunks.
- **Sparse retrieval:** Use BM25 to retrieve chunks based on keyword relevance.
- **Hybrid search:** Combine dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** Reorder retrieved candidates based on their relevance to the user's query.
- **Grounded question answering:** Generate responses using retrieved document context.
- **Persistent storage:** Store original documents and processed chunks using Supabase Storage and PostgreSQL with pgvector.
- **REST API:** Expose backend functionality through FastAPI.
- **Interactive frontend:** Upload documents and ask questions through a React-based interface.

*Note: Ensure cross-encoder reranking is enabled in the current code before publishing this feature as implemented.*

## Architecture

```text
                 User
                  |
                  v
         React + TypeScript
                  |
                  v
              FastAPI
                  |
          Document Upload
                  |
                  v
          Document Processing
                  |
                  v
         Chunking and Metadata
                  |
                  v
       Embedding and Persistence
                  |
                  v
       Supabase Storage + pgvector
                  |
        User Question Submitted
                  |
          +-------+-------+
          |               |
          v               v
     Dense Search      BM25 Search
          |               |
          +-------+-------+
                  |
                  v
       Reciprocal Rank Fusion
                  |
                  v
        Cross-Encoder Reranking
                  |
                  v
       Top Relevant Document Chunks
                  |
                  v
          Groq LLM Generation
                  |
                  v
          Grounded Answer
```

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | React, TypeScript, Vite |
| Embeddings | Sentence Transformers |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Hybrid fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| LLM inference | Groq API |
| Database | PostgreSQL with Supabase |
| Vector search | Supabase pgvector |
| Document storage | Supabase Storage |
| API documentation | FastAPI / Swagger UI |

## Project Structure

```text
RAG-Pipeline/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── config.py
│   ├── ingestion.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── supabase_client.py
│   └── main.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

### 1. Document ingestion

Uploaded documents are processed into text chunks with associated metadata. Original documents are stored in Supabase Storage, while processed chunks and their embeddings are persisted in the database.

### 2. Dense retrieval

The user's question is converted into an embedding using `BAAI/bge-small-en-v1.5`. Supabase pgvector retrieves chunks with similar semantic representations.

### 3. Sparse retrieval

BM25 searches document text using keyword relevance. This helps retrieve content containing important terms, names, identifiers, and phrases.

### 4. Hybrid retrieval with RRF

Dense and sparse results are combined using Reciprocal Rank Fusion, balancing semantic similarity with keyword matching.

### 5. Cross-encoder reranking

A cross-encoder evaluates the question and candidate chunks together to improve the ordering of retrieved results.

### 6. Answer generation

The selected chunks are provided as context to the LLM through Groq. The application uses retrieved information to answer the user's question.

## Prerequisites

- Python 3.11 recommended
- Node.js and npm
- A Supabase project with PostgreSQL, pgvector, and Storage configured
- A Groq API key
- Git

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/krishnasai81/hybrid-rag-assistant
cd hybrid-rag-assistant
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Set up the Python backend

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_server_side_supabase_key
SUPABASE_BUCKET=rag-documents
```

Use the exact Groq model identifier supported by your account and the configuration in your application. Never commit `.env` or expose server-side Supabase credentials in frontend code.

### 4. Configure Supabase

Before running the application, configure the required database tables, vector column, `match_document_chunks` RPC function, and private `rag-documents` Storage bucket.

The vector column must match the embedding dimension produced by the configured model. The current BGE small English model produces 384-dimensional embeddings.

Configure appropriate database access policies and keep privileged credentials on the backend only.

### 5. Start the backend

From the repository root:

```powershell
uvicorn app.main:app --reload
```

The backend should be available at:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/api/v1/health`

### 6. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, typically `http://localhost:5173`.

If your frontend uses `VITE_API_BASE_URL`, configure it with the backend API base URL expected by your frontend code.

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Authenticates requests to Groq |
| `GROQ_MODEL` | Selects the generation model |
| `SUPABASE_URL` | Identifies the Supabase project |
| `SUPABASE_KEY` | Backend database and storage access |
| `SUPABASE_BUCKET` | Name of the document Storage bucket |

The embedding model and reranker are configured in the application code. Update the configuration if you change models.

## Current Limitations

- Cloud deployment is not currently configured.
- The backend requires valid Groq and Supabase credentials.
- Retrieval quality depends on document extraction, chunking, embeddings, and ranking.
- Large document collections can increase memory usage, particularly when building the BM25 index or loading machine-learning models.
- The application should be evaluated with representative questions and documents before production use.

## Future Improvements

- Automated retrieval and answer-quality evaluation.
- Better document parsing and metadata filtering.
- Retrieval latency and memory monitoring.
- Authentication and per-user document access controls.
- Improved error handling and structured logging.
- Deployment with an appropriate memory budget.
- Automated testing for ingestion, retrieval, and answer generation.

## Security

- Never commit API keys, `.env` files, or database credentials.
- Do not upload confidential or private documents to a public repository.
- Keep Supabase service-role credentials on the server.
- Apply access controls to document storage and database operations.
- Review repository history for accidentally committed secrets before making the repository public.

## License

Choose a license before allowing others to reuse your code. For example, the MIT License permits broad reuse with attribution. Add a `LICENSE` file if you choose MIT.
