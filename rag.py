import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


# ── Always read DB_PATH fresh from env (never cache at module level) ──
def _db_path():
    return os.getenv("CHROMA_DB_PATH", "./chroma_db")


def get_collection():
    chroma = chromadb.PersistentClient(path=_db_path())
    return chroma.get_or_create_collection("pdf_docs")


# Embedding model — runs locally, no API needed
EMBED_MODEL = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cpu"
)

# Groq client — reads GROQ_API_KEY from environment
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── helpers ──────────────────────────────────────────────

def embed(text: str):
    keywords = "information details explanation dataset attributes table results"
    expanded_query = f"{text}. Related terms: {keywords}"
    return EMBED_MODEL.encode(expanded_query).tolist()


def get_indexed_files():
    try:
        all_meta = get_collection().get(include=["metadatas"])["metadatas"]
        files = set(m.get("display_name", "") for m in all_meta if m)
        files.discard("")
        return files
    except Exception:
        return set()


def retrieve_balanced(question: str, top_k_per_doc: int = 4):
    q_vec   = embed(question)
    indexed = get_indexed_files()

    all_chunks    = []
    all_metadatas = []
    all_distances = []

    for doc_name in indexed:
        try:
            results = get_collection().query(
                query_embeddings=[q_vec],
                n_results=top_k_per_doc,
                include=["documents", "metadatas", "distances"],
                where={"display_name": doc_name}
            )
            all_chunks.extend(results["documents"][0])
            all_metadatas.extend(results["metadatas"][0])
            all_distances.extend(results["distances"][0])
        except Exception:
            pass

    combined = sorted(
        zip(all_distances, all_chunks, all_metadatas),
        key=lambda x: x[0]
    )
    combined = combined[:10]
    return (
        [x[1] for x in combined],
        [x[2] for x in combined],
        [x[0] for x in combined]
    )


def retrieve_single(question: str, top_k: int, filter_name: str):
    q_vec   = embed(question)
    results = get_collection().query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where={"display_name": filter_name}
    )
    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )


def build_context(chunks, metadatas):
    parts = []
    for chunk, meta in zip(chunks, metadatas):
        title    = meta.get("title", "")
        filename = meta.get("display_name", "")
        prefix   = f"[{title} — {filename}]" if title else f"[{filename}]"
        parts.append(f"{prefix}\nRelevant document excerpt:\n{chunk}")
    return "\n\n---\n\n".join(parts)


def build_sources(chunks, metadatas, distances):
    sources = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        sources.append({
            "text":     chunk,
            "title":    meta.get("title", ""),
            "strategy": meta.get("strategy", ""),
            "filename": meta.get("display_name", ""),
            "score":    round(1 - dist, 3)
        })
    return sources


# ── main query functions ──────────────────────────────────

def query(question: str, top_k: int = 4, filter_source: str = None):
    if filter_source:
        chunks, metadatas, distances = retrieve_single(question, top_k, filter_source)
    else:
        chunks, metadatas, distances = retrieve_balanced(question, top_k_per_doc=2)

    context = build_context(chunks, metadatas)

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Answer ONLY using the provided context.
- Quote or summarize the relevant sentences from the document.
- Do NOT infer information not present in the context.
- If the question asks about datasets, attributes, or tables, look carefully for lists or structured information in the context.
- If the answer is not found, say: "I don't have enough information in the document to answer this."
- If answering from multiple documents, mention which document each point comes from.


Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer, build_sources(chunks, metadatas, distances)


def query_with_history(question: str, history: list,
                       top_k: int = 4, filter_source: str = None):
    if filter_source:
        chunks, metadatas, distances = retrieve_single(question, top_k, filter_source)
    else:
        chunks, metadatas, distances = retrieve_balanced(question, top_k_per_doc=2)

    context = build_context(chunks, metadatas)

    system_prompt = f"""You are a helpful assistant that answers questions based on the provided document context.
Answer ONLY using the context below. If the answer is not in the context, say "I don't have enough information in the document to answer this".
If answering from multiple documents, mention which document each point comes from.

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant",  "content": turn["assistant"]})
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    answer = response.choices[0].message.content
    return answer, build_sources(chunks, metadatas, distances)