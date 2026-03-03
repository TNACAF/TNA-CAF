# TNA-CAF (Code Release)

This repository contains a minimal implementation of **TNA-CAF** (Token–Node Aligned Cross-Attention Fusion) and the associated preprocessing pipeline.

**Files included**
- `TNA-CAF.py` — model training + evaluation script (TNA-CAF)
- `preprocessing.py` — preprocessing: global deduplication, AST graph construction, Laplacian positional encodings
- `requirements.txt` — Python package versions used in our experiments

> **Dataset note:** This repository does **not** redistribute Draper VDISC (or any raw source code). You must obtain the dataset from the official source and generate the required CSV files locally.

---

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
