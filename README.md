# Cybersecurity Policy & Compliance Assistant

Capstone 1 — RAG-based cybersecurity policy/compliance assistant.

## Current phase

Phase 1: PDF ingestion and chunking.

Corpus:
- CIS Critical Security Controls v8
- NIST Cybersecurity Framework (CSF) 2.0
- NIST SP 800-53 Revision 5

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the three PDFs in `data/raw/`, then:

```bash
python main.py
```

The first milestone is deliberately limited to extraction + chunk inspection. We will add embeddings and Chroma only after validating the chunks.
