# VS Code Setup Guide — OmegaTK RAG Assistant

Step-by-step instructions for getting the project running locally in VS Code.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | any | [git-scm.com](https://git-scm.com/) |
| VS Code | latest | [code.visualstudio.com](https://code.visualstudio.com/) |

---

## Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd omegatk-assistant
```

Open the folder in VS Code:

```bash
code .
```

---

## Step 2 — Install recommended extensions

VS Code will show a notification: **"This workspace has extension recommendations."**
Click **Install All**, or go to Extensions (`Ctrl+Shift+X`) and install manually:

- **Python** (ms-python.python)
- **Pylance** (ms-python.pylance)
- **Python Debugger** (ms-python.debugpy)

---

## Step 3 — Create a virtual environment

Open the integrated terminal (`Ctrl+`` `):

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

VS Code will prompt: **"We noticed a new virtual environment..."** → click **Yes**
to select it as the interpreter. Or press `Ctrl+Shift+P` →
**Python: Select Interpreter** → choose `.venv`.

---

## Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This downloads ~500 MB (models + libraries). Get a coffee. ☕

Common issues:

| Error | Fix |
|-------|-----|
| `faiss-cpu` build fails on Windows | `pip install faiss-cpu --no-build-isolation` |
| `sentence-transformers` version conflict | `pip install sentence-transformers==2.7.0` |
| SSL error on doc fetch | Check corporate proxy / VPN settings |

---

## Step 5 — Create your .env file

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
GOOGLE_API_KEY=AIza...your_actual_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

> ⚠️ Never commit `.env` — it is already in `.gitignore`.

---

## Step 6 — Run the Streamlit app

```bash
streamlit run app.py
# or: make run
```

A browser tab opens at `http://localhost:8501`.

**First run only:** the app fetches 8 OmegaTK documentation pages and
builds the FAISS vector index (~60 seconds). The index is saved to
`faiss_index/` and reloaded instantly on every subsequent run.

---

## Step 7 — Run the test suite (optional)

```bash
python test_prompts.py
# or: make test
```

Results are written to `test_results.md`. All 15 tests should pass.

---

## Debugging in VS Code

The project includes a pre-configured launch profile.

1. Open **Run and Debug** (`Ctrl+Shift+D`)
2. Select **"Streamlit: app.py"** from the dropdown
3. Press **F5**

Breakpoints in `app.py`, `rag/chain.py`, etc. will now pause execution.

To debug the test suite instead, select **"Python: test_prompts.py"**.

---

## Rebuilding the FAISS index

If the OmegaTK docs change and you want fresh embeddings:

```bash
make clean-index   # deletes faiss_index/
make run           # rebuilds on next startup
```

Or click **Rebuild index** in the Streamlit sidebar.

---

## Project structure

```
omegatk-assistant/
├── app.py                  # Streamlit UI
├── test_prompts.py         # Automated tests
├── requirements.txt        # Python dependencies
├── Makefile                # Dev shortcuts
├── .env.example            # API key template
├── .gitignore
├── README.md
├── SETUP.md                # This file
├── rag/
│   ├── __init__.py
│   ├── loader.py           # Fetch OmegaTK docs
│   ├── indexer.py          # Build/load FAISS index
│   ├── prompts.py          # LLM prompt templates
│   └── chain.py            # RAG chain assembly
└── .vscode/
    ├── extensions.json     # Recommended extensions
    ├── launch.json         # Debug configurations
    └── settings.json       # Workspace settings
```
