"""
app.py — OmegaTK RAG Coding Assistant (Streamlit UI)

Run with:
    streamlit run app.py

Requires GOOGLE_API_KEY in environment or a .env file.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OmegaTK Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API key guard — show a form if key is missing
# ---------------------------------------------------------------------------
def _ensure_api_key() -> bool:
    """Return True if a valid API key is available, otherwise render input form."""
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    if "api_key_entered" in st.session_state and st.session_state.api_key_entered:
        return True

    st.title("🧬 OmegaTK Coding Assistant")
    st.warning(
        "No `GOOGLE_API_KEY` found. Enter it below, or add it to a `.env` file "
        "in the same directory and restart the app."
    )
    with st.form("api_key_form"):
        key = st.text_input("Google API Key", type="password")
        submitted = st.form_submit_button("Continue")
        if submitted and key:
            os.environ["GOOGLE_API_KEY"] = key
            st.session_state.api_key_entered = True
            st.rerun()
    return False


# ---------------------------------------------------------------------------
# RAG chain — cached so it only builds once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_rag_chain():
    """Build and return the RAG chain. FAISS index is cached to disk."""
    from rag.indexer import get_or_build_index
    from rag.chain import build_llm, build_rag_chain as _build

    vectorstore = get_or_build_index()
    llm = build_llm()
    return _build(vectorstore, llm)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []        # list of {"role", "content"}
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []      # LangChain HumanMessage/AIMessage
    if "sources" not in st.session_state:
        st.session_state.sources = {}         # turn_index -> list[source_url]


def _reset_conversation():
    st.session_state.messages = []
    st.session_state.lc_history = []
    st.session_state.sources = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("🧬 OmegaTK Assistant")
        st.caption("Powered by LangChain · FAISS · Gemini 2.5 Flash")
        st.divider()

        st.subheader("Controls")
        if st.button("🗑️ Clear conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()
        st.subheader("Example prompts")
        examples = [
            "Generate conformers for a SMILES string and save to SDF",
            "How do I set max conformers to 50 and energy window to 10?",
            "Generate conformers for all molecules in an SDF file",
            "How do I generate macrocycle conformers?",
            "Enumerate stereoisomers and generate conformers for each",
        ]
        for i, ex in enumerate(examples):
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state.pending_prompt = ex
                st.rerun()

        st.divider()
        st.subheader("Index status")
        index_exists = os.path.exists("faiss_index")
        if index_exists:
            st.success("✅ FAISS index cached on disk")
            if st.button("🔄 Rebuild index", use_container_width=True):
                import shutil
                shutil.rmtree("faiss_index", ignore_errors=True)
                st.cache_resource.clear()
                st.rerun()
        else:
            st.info("⏳ Index will be built on first load (~60s)")


# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
def render_chat(rag_chain):
    from langchain_core.messages import HumanMessage, AIMessage

    st.title("🧬 OmegaTK Coding Assistant")
    st.caption(
        "Ask me anything about the OmegaTK Python toolkit. "
        "I'll generate executable code using the official API."
    )

    # Render existing conversation
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i in st.session_state.sources:
                srcs = st.session_state.sources[i]
                if srcs:
                    with st.expander("📄 Sources", expanded=False):
                        for src in srcs:
                            st.markdown(f"- {src}")

    # Handle example button clicks
    prompt = st.session_state.pop("pending_prompt", None)

    # Chat input
    if user_input := (st.chat_input("Ask about OmegaTK...") or prompt):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            sources = []
            t0 = time.time()

            try:
                chain_input = {
                    "input": user_input,
                    "chat_history": st.session_state.lc_history,
                }

                with st.spinner("Retrieving docs..."):
                    result = rag_chain.invoke(chain_input)

                full_response = result.get("answer", "")
                context_docs = result.get("context", [])

                # Extract unique source URLs
                seen: set[str] = set()
                for doc in context_docs:
                    src = doc.metadata.get("source", "")
                    if src and src not in seen:
                        sources.append(src)
                        seen.add(src)

                # Simulate streaming — word by word reveal
                words = full_response.split(" ")
                streamed = ""
                for word in words:
                    streamed += word + " "
                    placeholder.markdown(streamed + "▌")

            except Exception as e:
                full_response = f"⚠️ Error: {e}"
                placeholder.markdown(full_response)

            elapsed = time.time() - t0
            placeholder.markdown(full_response)

            if sources:
                with st.expander("📄 Sources", expanded=False):
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")

            st.caption(f"⏱ {elapsed:.1f}s")

        # Save to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        turn_index = len(st.session_state.messages) - 1
        if sources:
            st.session_state.sources[turn_index] = sources

        # Update LangChain history
        st.session_state.lc_history.append(HumanMessage(content=user_input))
        st.session_state.lc_history.append(AIMessage(content=full_response))

        # Cap at 20 messages (10 turns) to control context size
        if len(st.session_state.lc_history) > 20:
            st.session_state.lc_history = st.session_state.lc_history[-20:]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not _ensure_api_key():
        return

    _init_session()
    render_sidebar()

    with st.spinner("🔧 Loading RAG chain (first run builds the index — ~60s)..."):
        try:
            rag_chain = build_rag_chain()
        except Exception as e:
            st.error(f"Failed to build RAG chain: {e}")
            st.stop()

    render_chat(rag_chain)


if __name__ == "__main__":
    main()
