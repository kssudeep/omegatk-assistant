"""
rag/loader.py
-------------
Fetches and caches OmegaTK documentation pages for RAG indexing.
"""

from __future__ import annotations

import os
from typing import List

# All OmegaTK documentation URLs used for retrieval
ALL_URLS: List[str] = [
    "https://docs.eyesopen.com/toolkits/python/omegatk/index.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenTheory.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenFunctions.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenOptions.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/omegaclassestk.html",
    "https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses.html#OEOmegaOptions",
]


def load_docs(urls: List[str] | None = None, verbose: bool = False):
    """
    Load documents from OmegaTK documentation URLs.

    Args:
        urls: List of URLs to fetch. Defaults to ALL_URLS.
        verbose: Print progress to stdout.

    Returns:
        List of LangChain Document objects.

    Raises:
        RuntimeError: If no documents could be loaded.
    """
    from langchain_community.document_loaders import WebBaseLoader

    os.environ.setdefault("USER_AGENT", "OmegaTK-RAG-Assistant/1.0")

    urls = urls or ALL_URLS
    docs = []
    failed = []

    for url in urls:
        try:
            loader = WebBaseLoader([url])
            loaded = loader.load()
            docs.extend(loaded)
            if verbose:
                print(f"  OK   {url}")
        except Exception as exc:
            failed.append(f"{url}: {exc}")
            if verbose:
                print(f"  SKIP {url} — {exc}")

    if not docs:
        raise RuntimeError(
            "Failed to load any documentation pages. "
            "Check your internet connection.\n" + "\n".join(failed)
        )

    if verbose:
        print(f"Loaded {len(docs)} page(s) from {len(urls)} URL(s).")

    return docs
