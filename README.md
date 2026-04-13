# OmegaTK RAG Coding Assistant

A retrieval-augmented generation (RAG) coding assistant for the
[OmegaTK Python toolkit](https://docs.eyesopen.com/toolkits/python/omegatk/)
by OpenEye Scientific Software (Cadence Molecular Sciences).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI — the main way to use the assistant |
| `test_prompts.py` | Automated test suite covering all OmegaTK prompt categories |
| `requirements.txt` | Python dependencies |
| `Makefile` | Convenience targets: `install`, `run`, `test`, `clean` |
| `rag/loader.py` | Fetches OmegaTK documentation pages |
| `rag/indexer.py` | Builds and caches the FAISS vector index |
| `rag/prompts.py` | All LLM prompt templates in one place |
| `rag/chain.py` | Assembles the two-step RAG chain |

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd omegatk-assistant
```

### 2. Create a virtual environment and install dependencies

```bash
make install
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set your API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# then edit .env and paste your key
```

Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com).

### 4. Run

**Streamlit UI (recommended):**
```bash
make run
# or: streamlit run app.py
```

**Automated tests:**
```bash
make test
# or: python test_prompts.py
```
Results are written to `test_results.md`.

## First run note

On first run the assistant fetches 8 OmegaTK documentation pages and builds
a FAISS vector index. This takes about 60 seconds. The index is saved to
`faiss_index/` on disk and loaded instantly on all subsequent runs.

To force a rebuild:
```bash
make clean-index
```
Or use the **Rebuild index** button in the Streamlit sidebar.

## Architecture

```
User question
     |
     v
History-aware retriever
  (condenses history + question -> search query)
     |
     v
FAISS MMR search (k=8, fetch_k=20)
  over OmegaTK docs chunks
     |
     v
Gemini 2.5 Flash
  (system prompt + chat history + retrieved context)
     |
     v
Answer + source URLs
```

## What it handles

| Prompt type | Behaviour |
|-------------|-----------|
| Clear OmegaTK request | Generates executable Python code using official API |
| Vague request | Asks clarifying questions (input format, output format, options) |
| Out-of-scope | Politely declines |
| Follow-up (e.g. "add error handling") | Uses conversation history to understand context |

## VS Code setup

Open the folder in VS Code. Recommended extensions are listed in
`.vscode/extensions.json` — VS Code will prompt you to install them.

To debug with breakpoints, use the **Streamlit: app.py** launch config
(`F5` with the Run & Debug panel open).
