# RAG Evaluation Studio

**A clean, professional RAG benchmark studio** — ingest, evaluate, compare, and iterate on Retrieval‑Augmented Generation pipelines with repeatable metrics and a polished dashboard.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-2E7D32)](#)
[![LangChain](https://img.shields.io/badge/RAG-LangChain-000000)](#)

> **Disclaimer (Ownership & Use)**  
> Owned by **Lalit Nayyar** (`lalitnayyar@gmail.com`) .  
> Provided for **internal evaluation, research, and benchmarking**. Redistribution or production deployment without owner consent is discouraged.  
> You are responsible for compliance with applicable policies/laws and third‑party terms (OpenAI/HuggingFace/etc.).

---

## Why this exists

RAG systems fail silently: retrieval can look “fine” while answers hallucinate or omit critical constraints. This app gives you a **single, repeatable, end‑to‑end benchmark loop**:

- **Ingest** → chunking + embeddings → Chroma vector DB
- **Retrieve** → compute retrieval metrics per category
- **Answer** → score answer quality signals
- **Compare** → iterate quickly with consistent settings + logs

---

## Screen preview

![RAG Evaluation Studio UI](ScreenShot/XTUYYvKclx.png)

---

## What you get (high‑signal features)

- **Professional evaluation dashboards**
  - Retrieval: **MRR**, **nDCG**, **Coverage**
  - Answer: **Accuracy**, **Completeness**, **Relevance** (all out of `/5`)
  - Category-wise breakdowns (direct_fact, temporal, comparative, numerical, relationship, spanning, holistic)

- **Configurable ingestion & retrieval**
  - Chunk size / overlap
  - Retrieval top‑K
  - Collection naming and storage paths

- **Provider‑aware embeddings**
  - Choose **OpenAI** or **HuggingFace**
  - Resilient execution with fallbacks (keeps app usable under strict corporate TLS / runtime constraints)

- **Operational controls**
  - Live ingestion progress meter
  - **Force New DB Run** (fresh run directory on demand)
  - DB lock auto-fix + **Reset Session**

- **Built-in logging**
  - Live log panel
  - File output: `applog_YYYYMMDD_HHMMSS.log`
  - Select log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`)

---

## Quick start (Windows / UV)

### 1) Install deps

```bash
uv sync --system-certs
```

### 2) Configure environment

```bash
copy .env.example .env
```

Set `OPENAI_API_KEY=` in `.env` if you will use OpenAI embeddings or LLM answering.

### 3) Run the app

```bash
uv run --system-certs --no-sync python -m streamlit run app.py --server.port 8504
```

---

## User guide (end-to-end)

### Step 1 — Prepare sample data
- Click **Create/Load 1000 Sample JSON**  
  Creates `knowledge-base/sample_knowledge.json` (exactly **1000** records).

### Step 2 — Ingest
- Optionally click **Force New DB Run** (fresh DB folder)
- Click **Ingest into Chroma Vector DB**
- Watch ingestion progress (split → chunk → embed → batch index)

### Step 3 — Evaluate
- Click **Run Evaluation**
- Review:
  - **Retrieval Evaluation** tab
  - **Answer Evaluation** tab

### Step 4 — Troubleshooting
- **Corporate TLS / SSL issues (HuggingFace)**
  - Set environment variable `REQUESTS_CA_BUNDLE` or `SSL_CERT_FILE` to your corporate CA bundle.
  - Use `uv ... --system-certs`.
- **Windows DB file lock**
  - Use **Fix DB Lock and Retry Cleanup**
  - If still stuck, click **Reset Session** or use **Force New DB Run**

---

## Tech stack

- Streamlit (UI + progress + live logs)
- LangChain + Chroma
- Embeddings: OpenAI / HuggingFace (+ resilient fallbacks)
- Plotly for charts

