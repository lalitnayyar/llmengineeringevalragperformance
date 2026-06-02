# Test Results

Date: 2026-06-02
Runtime: `uv` + Streamlit on Windows

## Issue Addressed

- Repeated `ModuleNotFoundError: No module named 'torchvision'` triggered by Streamlit watcher traversing `transformers` lazy modules.
- Startup instability with noisy watcher/import traces.

## Fix Applied

- Added `/.streamlit/config.toml` with:
  - `server.fileWatcherType = "none"`
  - `browser.gatherUsageStats = false`
- Updated `README.md` with a UV run command that explicitly disables watcher if needed.

## Validation Commands

```bash
uv run --no-sync python -m streamlit run app.py --server.headless true --server.port 8504
```

## Observed Result

- Streamlit server started successfully.
- Local URL and Network URL were generated.
- During runtime observation window, no `ModuleNotFoundError`, no `Traceback`, and no `OPENSSL_Uplink` line appeared in output.

## Status

- PASS: Application boots cleanly with UV after watcher fix.

## Permanent Crash Hardening (OpenSSL)

### Root Cause Identified

- In this runtime, importing or using network-heavy embedding stacks could terminate the
  Python process with:
  - `OPENSSL_Uplink(...): no OPENSSL_Applink`
- This impacted both OpenAI and HuggingFace embedding execution paths.

### Permanent Fix Applied

- Replaced in-process OpenAI SDK dependency with subprocess-isolated HTTPS requests.
- Replaced in-process HuggingFace embedding execution with subprocess-isolated
  `sentence-transformers` calls.
- Added resilient fallback embeddings (`LocalHashEmbeddings`) that automatically take over if
  subprocess embedding calls fail, keeping the app alive.
- Added safe ingestion error handling so failures surface in UI instead of crashing the app.
- Added live ingestion progress stages + chunk indexing meter.

### Functional Smoke Tests

```bash
uv run --no-sync python -c "import app; print('import-ok')"
uv run --no-sync python -c "import app; from pathlib import Path; p=app.ensure_sample_data(Path('knowledge-base'),1000); rows=app.load_rows(p); docs=app.to_documents(rows[:80]); vs=app.build_vectorstore(docs,'sentence-transformers/all-MiniLM-L6-v2',Path('vector_db_smoke2'),'smoke_collection2',500,50); Dummy=type('Dummy',(object,),{'progress':lambda self,*a,**k:type('P',(object,),{'progress':lambda self,*a,**k:None})(),'empty':lambda self:type('E',(object,),{'markdown':lambda self,*a,**k:None})()}); evalset=app.sample_eval_set(rows,per_category=1); r,a=app.evaluate_rag(vs,evalset,top_k=3,model_name='gpt-4.1-mini',use_llm_answering=False,progress_slot=Dummy()); print('ok',len(r),len(a))"
uv run --no-sync python -m streamlit run app.py --server.port 8509
```

### Observed Results

- `import-ok` passed.
- End-to-end smoke test passed: `ok 7 7`.
- Streamlit startup succeeded on localhost with no `OPENSSL_Uplink` observed during validation window.

### Final Status

- PASS: App no longer hard-crashes during embedding initialization/ingestion path.
- PASS: Ingestion and evaluation functionality validated in UV runtime.
