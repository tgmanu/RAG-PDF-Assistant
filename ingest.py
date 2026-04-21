import chromadb
import json
import re
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


# ── Always read DB_PATH fresh from env (never cache at module level) ──
def _db_path():
    return os.getenv("CHROMA_DB_PATH", "./chroma_db")


def get_collection():
    chroma = chromadb.PersistentClient(path=_db_path())
    return chroma.get_or_create_collection("pdf_docs")


# Load embedding model once (runs locally, no API key needed)
EMBED_MODEL = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cpu"
)


# ── helpers ──────────────────────────────────────────────

def embed_text(text: str):
    return EMBED_MODEL.encode(text).tolist()


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        pages.append(text)
    return "\n\n".join(pages), len(reader.pages)


def detect_strategy(text, num_pages):
    chars_per_page = len(text) / max(num_pages, 1)
    if num_pages <= 5 or chars_per_page < 800:
        return "recursive"
    start_words = set(text[:2000].lower().split())
    end_words   = set(text[-2000:].lower().split())
    overlap = len(start_words & end_words) / len(start_words | end_words)
    if num_pages > 80:
        return "recursive"   # semantic too slow for large docs
    elif num_pages > 15 or overlap < 0.3:
        return "semantic"
    return "recursive"


def make_chunk_id(display_name, index):
    base = display_name.replace(".", "_").replace(" ", "_").replace("-", "_")
    return f"{base}_{index}"


# ── chunking strategies ───────────────────────────────────

def chunk_recursive(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100, chunk_overlap=150
    )
    return [{"content": c, "title": ""} for c in splitter.split_text(text)]


def chunk_semantic(text):
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    splitter = SemanticChunker(
        hf_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    return [{"content": c, "title": ""} for c in splitter.split_text(text)]


# ── indexed files helper ──────────────────────────────────

def get_indexed_files():
    try:
        all_meta = get_collection().get(include=["metadatas"])["metadatas"]
        files = set(m.get("display_name", "") for m in all_meta if m)
        files.discard("")
        return files
    except Exception:
        return set()


# ── main ingest ───────────────────────────────────────────

def ingest(pdf_path: str, force: bool = False, display_name: str = None):
    filename = display_name if display_name else os.path.basename(pdf_path)
    print(f"\nReading: {filename}")

    already_indexed = get_indexed_files()
    if filename in already_indexed and not force:
        print(f"Already indexed: {filename} — skipping.")
        return

    if filename in already_indexed and force:
        existing = get_collection().get(
            where={"display_name": filename},
            include=["metadatas"]
        )
        if existing["ids"]:
            get_collection().delete(ids=existing["ids"])
            print(f"Cleared old chunks for: {filename}")

    text, num_pages = extract_text(pdf_path)
    print(f"Extracted {len(text)} characters from {num_pages} pages")

    strategy = detect_strategy(text, num_pages)
    print(f"Detected strategy: {strategy}")

    chunks = chunk_semantic(text) if strategy == "semantic" else chunk_recursive(text)
    print(f"Split into {len(chunks)} chunks — embedding now...")

    for i, chunk in enumerate(chunks):
        embed_input = (
            f"{chunk['title']}: {chunk['content']}"
            if chunk["title"] else chunk["content"]
        )
        embedding = embed_text(embed_input)
        chunk_id  = make_chunk_id(filename, i)

        get_collection().add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[{
                "title":        chunk["title"],
                "strategy":     strategy,
                "display_name": filename,
                "source":       pdf_path
            }]
        )
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i+1}/{len(chunks)} chunks...")

    print(f"\nDone! Stored {len(chunks)} chunks using [{strategy}] strategy")
    print(f"Total chunks in DB: {get_collection().count()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py your_file.pdf")
        sys.exit(1)
    ingest(sys.argv[1])