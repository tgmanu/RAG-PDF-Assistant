from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import tempfile
import os
from ingest import ingest, get_indexed_files, collection
from rag import query_with_history

st.set_page_config(page_title="DocuMind", page_icon="pdficon.png", layout="wide")

if "messages" not in st.session_state: st.session_state.messages = []
if "history"  not in st.session_state: st.session_state.history  = []

# ── sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("DocuMind")
    st.caption("Local RAG — runs fully on your machine")
    st.divider()

    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded:
        col1, col2 = st.columns(2)
        index_btn   = col1.button("Index PDF", type="primary", use_container_width=True)
        reindex_btn = col2.button("Re-index", use_container_width=True)

        if index_btn or reindex_btn:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                tmp_path = f.name

            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    # pass original filename as display_name
                    ingest(tmp_path, force=reindex_btn, display_name=uploaded.name)
                    st.session_state.messages = []
                    st.session_state.history  = []
                    st.success("Done! Ask questions below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_path)

    # Show indexed PDFs — use display_name which is the clean original name
    st.divider()
    st.subheader("Indexed PDFs")
    indexed = get_indexed_files()
    if indexed:
        for fname in sorted(indexed):
            st.markdown(f"✅ `{fname}`")
        st.caption(f"{collection.count()} total chunks across {len(indexed)} PDF(s)")
    else:
        st.info("No PDFs indexed yet")

    # Search scope — filter by display_name
    st.divider()
    st.subheader("Search scope")
    search_all    = st.toggle("Search ALL PDFs", value=True)
    selected_file = None
    if not search_all and indexed:
        labels        = sorted(list(indexed))
        chosen        = st.selectbox("Pick a PDF to search", labels)
        selected_file = chosen   # this is already the display_name

    st.divider()
    st.subheader("Settings")
    top_k        = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    show_sources = st.toggle("Show sources", value=True)

    # Clear only the conversation
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# Reset database (remove indexed PDFs)
    if st.button("Reset database", use_container_width=True):
        import shutil
        shutil.rmtree("/tmp/chroma_db", ignore_errors=True)
        st.success("All indexed PDFs cleared")
        st.rerun()

# ── main area ─────────────────────────────────────────────
st.title("I read it so you don't have to")

if not get_indexed_files():
    st.info("Drop a PDF in the sidebar. I'll read it. You just ask.")
    with st.expander("How does this work?"):
        st.markdown("""
1. **Drop any PDF** in the sidebar — resume, research paper, textbook, anything
2. **I'll read it** — chunk it, understand it, index it locally
3. **You just ask** — I'll find the answer and tell you exactly where it came from
4. **Multiple PDFs?** No problem — I'll search across all of them at once

**Everything stays on your machine. I don't share your docs with anyone.**
        """)
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_sources and "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
                for i, s in enumerate(msg["sources"]):
                    raw   = s["score"]
                    score = max(0.0, round(1 / (1 + abs(raw)), 3)) if raw < 0 else raw
                    title = f"**{s['title']}** — " if s["title"] else ""
                    st.markdown(f"{i+1}. {title}`{s['filename']}` · score `{score}`")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))
                    if i < len(msg["sources"]) - 1:
                        st.divider()

# Chat input
if question := st.chat_input("What do you want to know?"):
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = query_with_history(
                    question,
                    st.session_state.history,
                    top_k=top_k,
                    filter_source=selected_file if not search_all else None
                )
                st.write(answer)

                if show_sources:
                    with st.expander(f"Sources ({len(sources)} chunks)"):
                        for i, s in enumerate(sources):
                            raw   = s["score"]
                            score = max(0.0, round(1 / (1 + abs(raw)), 3)) if raw < 0 else raw
                            title = f"**{s['title']}** — " if s["title"] else ""
                            st.markdown(f"{i+1}. {title}`{s['filename']}` · score `{score}`")
                            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))
                            if i < len(sources) - 1:
                                st.divider()

                st.session_state.history.append({"user": question, "assistant": answer})
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")