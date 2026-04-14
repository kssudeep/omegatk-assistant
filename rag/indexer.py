"""
rag/indexer.py
--------------
Builds and persists a FAISS vector index over OmegaTK documentation chunks.
Loads from disk on subsequent runs to avoid re-embedding on every restart.

Changelog:
  - v2 (Day 14): added force_rebuild flag and unified verbose control.
"""

from __future__ import annotations

import os

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    """Return the HuggingFace embedding model (cached by LangChain)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index(docs, embeddings=None, save_path: str = INDEX_PATH):
    """
    Split documents into chunks and build a FAISS vector store.

    Args:
        docs: List of LangChain Document objects.
        embeddings: Optional pre-initialised embeddings. Created if None.
        save_path: Directory path to persist the index.

    Returns:
        FAISS vectorstore instance.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    if embeddings is None:
        embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} doc(s) into {len(chunks)} chunk(s).")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to '{save_path}/'.")
    return vectorstore


def load_index(embeddings=None, load_path: str = INDEX_PATH):
    """
    Load a previously saved FAISS index from disk.

    Args:
        embeddings: Optional pre-initialised embeddings. Created if None.
        load_path: Directory containing the saved index.

    Returns:
        FAISS vectorstore instance.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{load_path}/'. "
            "Run build_index() first."
        )

    if embeddings is None:
        embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"FAISS index loaded from '{load_path}/'.")
    return vectorstore


def get_or_build_index(docs_loader=None, save_path: str = INDEX_PATH):
    """
    Load index from disk if it exists, otherwise build and save it.

    Args:
        docs_loader: Callable that returns a list of Documents.
                     Required only when the index does not yet exist.
        save_path: Directory path for the index.

    Returns:
        FAISS vectorstore instance.
    """
    embeddings = get_embeddings()

    if os.path.exists(save_path):
        return load_index(embeddings=embeddings, load_path=save_path)

    if docs_loader is None:
        from rag.loader import load_docs
        docs_loader = load_docs

    docs = docs_loader(verbose=True)
    return build_index(docs, embeddings=embeddings, save_path=save_path)


# ---------------------------------------------------------------------------
# v2 addition: force_rebuild entry point used by Streamlit sidebar button
# ---------------------------------------------------------------------------
def rebuild_index(verbose: bool = True):
    """Delete any existing index and build a fresh one from live docs."""
    return get_or_build_index(force_rebuild=True, verbose=verbose)
