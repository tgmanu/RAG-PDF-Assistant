# RAG-PDF-Assistant 

A local RAG (Retrieval-Augmented Generation) chatbot that lets you chat with any PDF using AI. Upload any document and ask questions in plain English — it finds the relevant sections and answers accurately, entirely from your documents.

## Features

- **Multi-PDF support** — index multiple PDFs and search across all at once
- **Smart chunking** — automatically detects document type and picks the best chunking strategy (recursive or semantic)
- **Balanced retrieval** — prevents large documents from drowning out smaller ones
- **Conversation memory** — follow-up questions work naturally
- **Source transparency** — every answer shows exactly which chunks were used
- **No hallucination** — answers only from your documents, never made up
- **Clean session on startup** — no stale PDFs from previous sessions ever appear
- **Working reset** — Reset Database fully wipes all indexed PDFs, not just the chat

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (BAAI/bge-small-en-v1.5) |
| Vector DB | ChromaDB 0.4.24 |
| Chunking | LangChain RecursiveCharacterTextSplitter + SemanticChunker |
| PDF parsing | pypdf |

## How It Works

```
PDF → Extract text → Detect strategy → Chunk → Embed → ChromaDB
                                                              ↓
User question → Embed → Similarity search → Top-k chunks → LLM → Answer
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-pdf-assistant.git
cd rag-pdf-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

## Project Structure

```
rag-pdf-assistant/
├── app.py                  # Streamlit UI
├── ingest.py               # PDF processing and indexing
├── rag.py                  # Query engine and retrieval
├── requirements.txt        # Dependencies
├── packages.txt            # System-level dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml         # Streamlit server config for cloud deployment
├── .python-version         # Pins Python 3.11 for compatibility
├── pdficon.png             # App icon
└── .gitignore
```

## Key Concepts Implemented

- **RAG pipeline** — retrieval-augmented generation from scratch
- **Hybrid chunking** — auto-detects document type, picks recursive or semantic splitting
- **Balanced multi-doc retrieval** — equal representation across documents of different sizes
- **Conversation history** — multi-turn chat with context window management
- **Groq inference** — fast LLM responses via Groq's free API
- **Session-scoped ChromaDB** — each browser session gets its own isolated DB path in `/tmp`
- **Startup cleanup** — on every new session, leftover `chroma_db_*` folders from previous sessions are wiped from `/tmp` automatically
- **True database reset** — reset button deletes the ChromaDB folder on disk and rotates to a new session ID, forcing ChromaDB to reinitialize from scratch instead of serving from cache

## Bugs Fixed

### Stale PDFs on startup
**Problem:** Previously indexed PDFs would reappear when the app was restarted because `/tmp/chroma_db_*` folders persisted across sessions (especially on Streamlit Cloud).

**Fix:** On the first run of every new session, `app.py` scans `/tmp` and deletes all leftover `chroma_db_*` folders before doing anything else.

### Reset Database not working
**Problem:** The reset button called `shutil.rmtree` to delete the folder but ChromaDB's in-memory client cache still served old data, so PDFs kept showing up.

**Fix:** After deleting the folder, a new `session_id` (UUID) is generated so `DB_PATH` changes to a fresh path. ChromaDB is forced to create a brand-new client on the next call — no stale cache.

### DB_PATH cached at import time
**Problem:** Both `ingest.py` and `rag.py` read `CHROMA_DB_PATH` from the environment once at module load time. After a reset changed the env var, they still pointed to the old path.

**Fix:** Replaced the module-level constant with a `_db_path()` function that reads from `os.getenv()` on every `get_collection()` call, so path changes are always picked up immediately.

## Deployment (Streamlit Cloud)

1. Push your repo to GitHub (make sure `chroma_db/` and `.env` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. Select your repo, branch `main`, main file `app.py`
4. In **Advanced settings** → set Python version to **3.11**
5. In **Advanced settings → Secrets**, add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
6. Click **Deploy**

> **Note:** First deploy takes 3–5 minutes as it downloads the sentence-transformers model. Subsequent cold starts are faster.

## Live Demo

[RAG-PDF-Assistant on Streamlit Cloud](https://rag-pdfassistant.streamlit.app/)
