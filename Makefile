# Makefile — convenience targets for the OmegaTK RAG Assistant
# Usage: make <target>

.PHONY: help install run test clean-index clean

help:
	@echo ""
	@echo "OmegaTK RAG Assistant — available targets:"
	@echo ""
	@echo "  make install      Install Python dependencies into .venv"
	@echo "  make run          Start the Streamlit app"
	@echo "  make test         Run the automated test suite"
	@echo "  make clean-index  Delete the cached FAISS index (forces rebuild)"
	@echo "  make clean        Remove .venv, __pycache__, and index"
	@echo ""

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate with: source .venv/bin/activate"

run:
	streamlit run app.py

test:
	python test_prompts.py

clean-index:
	rm -rf faiss_index/
	@echo "FAISS index deleted. It will be rebuilt on next run."

clean:
	rm -rf .venv faiss_index/ __pycache__/ rag/__pycache__/
	find . -name "*.pyc" -delete
	@echo "Clean complete."
