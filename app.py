from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import gc
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


APP_ROOT = Path(__file__).parent
DEFAULT_DB_NAME = str(APP_ROOT / "vector_db")
DEFAULT_KNOWLEDGE_BASE = str(APP_ROOT / "knowledge-base")

DEFAULT_MODEL = "gpt-4.1-mini"
OPENAI_EMBEDDING_OPTIONS = {
    "text-embedding-3-small": "Lowest OpenAI cost, strong baseline",
    "text-embedding-3-large": "Higher quality, higher cost",
}
HUGGINGFACE_EMBEDDING_OPTIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": "Most cost-effective (local/free), fast",
    "BAAI/bge-small-en-v1.5": "Low-cost local option, good retrieval",
    "BAAI/bge-base-en-v1.5": "Balanced quality/cost local option",
}
COLLECTION_OPTIONS = ["langchain", "rag_eval_prod", "rag_eval_experiment"]

CATEGORIES = [
    "direct_fact",
    "temporal",
    "comparative",
    "numerical",
    "relationship",
    "spanning",
    "holistic",
]


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

DISCLAIMER_TEXT = (
    "Disclaimer: Owned by Lalit Nayyar (lalitnayyar@gmail.com), "
    "Phone: +971508320336 / +919595353336, Company: Symbiotic India."
)


def _concise_error_text(message: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(message)).strip()
    openssl_markers = [
        "OPENSSL_Uplink",
        "OPENSSL_Applink",
    ]
    if any(m in text for m in openssl_markers):
        return (
            "Windows OpenSSL runtime issue detected in embedding backend. "
            "App switched to resilient fallback to keep ingestion running."
        )
    ssl_markers = [
        "CERTIFICATE_VERIFY_FAILED",
        "unable to get local issuer certificate",
        "SSLCertVerificationError",
    ]
    if any(m in text for m in ssl_markers):
        return (
            "SSL certificate verification failed while connecting to HuggingFace. "
            "Configure corporate/system CA (REQUESTS_CA_BUNDLE / SSL_CERT_FILE) "
            "or use OpenAI embedding."
        )
    return text[:limit]


def _get_subprocess_python() -> str:
    preferred = Path("C:/Python313/python.exe")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def _reset_vector_db_path(db_path: Path, logger: logging.Logger, status_box) -> Path:
    # Release references that can lock sqlite/chroma files on Windows.
    stale_vs = st.session_state.pop("vectorstore", None)
    st.session_state.pop("retrieval_df", None)
    st.session_state.pop("answer_df", None)
    if stale_vs is not None:
        try:
            client = getattr(stale_vs, "_client", None)
            if client is not None and hasattr(client, "reset"):
                client.reset()
        except Exception:
            pass
        del stale_vs
        gc.collect()

    if db_path.exists():
        logger.info("Clearing previous vector DB at %s", db_path)
        status_box.info("Clearing previous vector data...")
        last_err = None
        for attempt in range(1, 8):
            try:
                shutil.rmtree(db_path)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Vector DB cleanup retry %s/7 failed: %s",
                    attempt,
                    _concise_error_text(exc),
                )
                time.sleep(0.35 * attempt)
        if last_err is not None:
            # Fallback: avoid hard-failing on Windows file locks by switching
            # to a fresh run directory under the same base path.
            fresh_root = db_path / "_fresh_runs"
            fresh_root.mkdir(parents=True, exist_ok=True)
            fresh_dir = fresh_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            fresh_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "DB path locked; using fresh run directory instead: %s",
                fresh_dir,
            )
            status_box.warning(
                "Previous DB is locked; switched to a fresh vector DB directory for this run."
            )
            st.session_state["active_vector_db_path"] = str(fresh_dir)
            return fresh_dir

    db_path.mkdir(parents=True, exist_ok=True)
    st.session_state["active_vector_db_path"] = str(db_path)
    return db_path


class UILogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            st.session_state.setdefault("log_lines", []).append(line)
            # Keep most recent log lines in memory
            st.session_state["log_lines"] = st.session_state["log_lines"][-400:]
        except Exception:
            pass


def ensure_logger(level_name: str) -> tuple[logging.Logger, Path]:
    st.session_state.setdefault("log_file_path", None)
    st.session_state.setdefault("logger_level_name", "INFO")
    st.session_state.setdefault("log_lines", [])

    if not st.session_state["log_file_path"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["log_file_path"] = str(APP_ROOT / f"applog_{ts}.log")

    log_file = Path(st.session_state["log_file_path"])
    logger = logging.getLogger("rag_eval_app")
    logger.propagate = False

    target_level = LOG_LEVELS.get(level_name, logging.INFO)
    if st.session_state.get("logger_level_name") != level_name:
        st.session_state["logger_level_name"] = level_name
        st.session_state["log_lines"].append(
            f"{datetime.now().isoformat()} | INFO | Logger level switched to {level_name}"
        )

    logger.setLevel(target_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Streamlit reruns can duplicate handlers if we don't reset explicitly.
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    uh = UILogHandler()
    uh.setFormatter(formatter)
    uh.setLevel(logging.DEBUG)
    logger.addHandler(uh)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) or isinstance(handler, UILogHandler):
            handler.setLevel(target_level)

    return logger, log_file


class OpenAIHTTPEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/embeddings"

    @staticmethod
    def _openai_subprocess_request(payload: dict, timeout_seconds: int = 120) -> dict:
        # Isolate OpenAI HTTPS call in child process to avoid crashing parent app
        # on certain Windows OpenSSL runtime combinations.
        script = """
import json
import requests
import sys

raw = sys.stdin.read()
payload = json.loads(raw)
headers = {
    "Authorization": f"Bearer {payload['api_key']}",
    "Content-Type": "application/json",
}
resp = requests.post(
    payload["url"],
    headers=headers,
    json=payload["body"],
    timeout=payload["timeout"],
)
resp.raise_for_status()
print(resp.text)
"""
        proc = subprocess.run(
            [_get_subprocess_python(), "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "Unknown subprocess error").strip()
            raise RuntimeError(f"OpenAI subprocess failed: {err}")
        return json.loads(proc.stdout)

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        payload = {
            "api_key": self.api_key,
            "url": self.url,
            "timeout": 60,
            "body": {"model": self.model, "input": inputs},
        }
        response_json = self._openai_subprocess_request(payload)
        data = response_json.get("data", [])
        data = sorted(data, key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class HuggingFaceSubprocessEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        payload = {
            "model_name": self.model_name,
            "inputs": inputs,
            "ca_bundle": os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or "",
            "local_files_only": False,
        }
        script = """
import json
import os
import sys
from sentence_transformers import SentenceTransformer

raw = sys.stdin.read()
payload = json.loads(raw)
ca_bundle = payload.get("ca_bundle")
if ca_bundle:
    os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
    os.environ["SSL_CERT_FILE"] = ca_bundle
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

model = SentenceTransformer(
    payload["model_name"],
    local_files_only=payload.get("local_files_only", False),
)
vectors = model.encode(payload["inputs"], convert_to_numpy=True).tolist()
print(json.dumps({"vectors": vectors}))
"""
        proc = subprocess.run(
            [_get_subprocess_python(), "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=180,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "Unknown subprocess error").strip()
            # If online download fails (SSL/cert/proxy), try local cache-only load.
            if "CERTIFICATE_VERIFY_FAILED" in err or "SSLError" in err:
                cached_payload = dict(payload)
                cached_payload["local_files_only"] = True
                cached_proc = subprocess.run(
                    [_get_subprocess_python(), "-c", script],
                    input=json.dumps(cached_payload),
                    text=True,
                    capture_output=True,
                    timeout=180,
                )
                if cached_proc.returncode == 0:
                    return json.loads(cached_proc.stdout)["vectors"]
            raise RuntimeError(
                f"HuggingFace subprocess failed: {_concise_error_text(err)}"
            )
        return json.loads(proc.stdout)["vectors"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return [0.0] * self.dim
        vec = np.zeros(self.dim, dtype=float)
        for token in tokens:
            h = abs(hash(token)) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


class ResilientEmbeddings(Embeddings):
    def __init__(self, primary: Embeddings, fallback: Embeddings, provider_name: str):
        self.primary = primary
        self.fallback = fallback
        self.provider_name = provider_name
        self._use_fallback = False
        self.last_warning: str | None = None

    def _run(self, fn_primary, fn_fallback):
        if self._use_fallback:
            return fn_fallback()
        try:
            return fn_primary()
        except Exception as exc:
            self._use_fallback = True
            self.last_warning = (
                f"{self.provider_name} embedding backend failed; switched to local hash embeddings. "
                f"Cause: {_concise_error_text(exc)}"
            )
            return fn_fallback()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._run(
            lambda: self.primary.embed_documents(texts),
            lambda: self.fallback.embed_documents(texts),
        )

    def embed_query(self, text: str) -> List[float]:
        return self._run(
            lambda: self.primary.embed_query(text),
            lambda: self.fallback.embed_query(text),
        )


@dataclass
class EvalRow:
    query: str
    category: str
    expected_doc_id: str
    expected_answer: str
    expected_keywords: List[str]


def apply_professional_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6f9fc 0%, #ffffff 100%);
        }
        .hero {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: white;
            border: 1px solid #e8edf3;
            box-shadow: 0 12px 30px rgba(31, 71, 120, 0.08);
            animation: riseIn 500ms ease-out;
        }
        .metric-card {
            border: 1px solid #ecf1f6;
            border-left: 6px solid #ff9f1a;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            background: white;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.04);
            animation: riseIn 450ms ease-out;
        }
        .green-ok {
            background: #ebf8f1;
            color: #196c47;
            border: 1px solid #c7ecd9;
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            font-weight: 600;
        }
        @keyframes riseIn {
          from {opacity: 0; transform: translateY(8px);}
          to {opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(expected: str, predicted: str) -> float:
    e = normalize_tokens(expected)
    p = normalize_tokens(predicted)
    if not e or not p:
        return 0.0
    common = len(set(e) & set(p))
    if common == 0:
        return 0.0
    precision = common / len(set(p))
    recall = common / len(set(e))
    return (2 * precision * recall) / (precision + recall)


def ndcg_from_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1 / math.log2(rank + 1)


def ensure_sample_data(kb_path: Path, sample_count: int = 1000) -> Path:
    kb_path.mkdir(parents=True, exist_ok=True)
    dataset_path = kb_path / "sample_knowledge.json"
    if dataset_path.exists():
        return dataset_path

    random.seed(42)
    topics = [
        "supply chain resilience",
        "renewable energy storage",
        "clinical trial optimization",
        "financial risk modelling",
        "education analytics",
        "smart city planning",
        "cybersecurity incident response",
        "retail demand forecasting",
        "agri-tech irrigation",
        "space mission telemetry",
    ]
    regions = ["NA", "EMEA", "APAC", "LATAM", "MEA"]
    years = list(range(2018, 2026))

    rows = []
    for i in range(sample_count):
        topic = random.choice(topics)
        region = random.choice(regions)
        year = random.choice(years)
        category = CATEGORIES[i % len(CATEGORIES)]
        score = round(random.uniform(48.0, 97.5), 2)
        delta = round(random.uniform(-7.0, 9.0), 2)
        entity = f"{topic.title()} Unit {i % 55 + 1}"
        benchmark = f"Benchmark-{i % 12 + 1}"
        answer = (
            f"{entity} in {region} during {year} achieved {score}% performance with "
            f"{delta:+}% change versus baseline under {benchmark}."
        )
        content = (
            f"Record {i+1}: {entity}. Topic: {topic}. Region: {region}. "
            f"Year: {year}. Category: {category}. Performance score: {score}. "
            f"Change from baseline: {delta}. Benchmark group: {benchmark}. "
            f"Primary recommendation is to improve retrieval grounding and "
            f"answer factual consistency for stakeholder reporting."
        )
        keywords = [topic.split()[0], region.lower(), str(year), category, "performance"]
        query = f"What was the performance and trend of {entity} in {year} for {region}?"
        rows.append(
            {
                "id": f"doc-{i+1}",
                "title": f"{entity} performance snapshot",
                "category": category,
                "topic": topic,
                "region": region,
                "year": year,
                "content": content,
                "answer": answer,
                "keywords": keywords,
                "query": query,
            }
        )
    dataset_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return dataset_path


def load_rows(dataset_path: Path) -> List[dict]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def to_documents(rows: List[dict]) -> List[Document]:
    docs = []
    for row in rows:
        docs.append(
            Document(
                page_content=row["content"],
                metadata={
                    "id": row["id"],
                    "title": row["title"],
                    "category": row["category"],
                    "topic": row["topic"],
                    "region": row["region"],
                    "year": row["year"],
                    "answer": row["answer"],
                    "keywords": ",".join(row["keywords"]),
                    "query": row["query"],
                },
            )
        )
    return docs


@st.cache_resource(show_spinner=False)
def get_embedding_model(name: str):
    fallback = LocalHashEmbeddings(dim=384)
    if name.startswith("text-embedding-3"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it in .env to use OpenAI embeddings."
            )
        primary = OpenAIHTTPEmbeddings(model=name, api_key=api_key)
        return ResilientEmbeddings(primary, fallback, provider_name="OpenAI")
    primary = HuggingFaceSubprocessEmbeddings(model_name=name)
    return ResilientEmbeddings(primary, fallback, provider_name="HuggingFace")


def build_vectorstore(
    docs: List[Document],
    embedding_name: str,
    db_dir: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback: Callable[[float, str], None] | None = None,
    batch_size: int = 120,
    logger: logging.Logger | None = None,
) -> Chroma:
    if logger:
        logger.info(
            "Starting vectorstore build | collection=%s | chunk_size=%s | overlap=%s",
            collection_name,
            chunk_size,
            chunk_overlap,
        )
    if progress_callback:
        progress_callback(0.05, "Initializing text splitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    if progress_callback:
        progress_callback(0.20, "Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    if progress_callback:
        progress_callback(0.40, f"Generated {len(chunks)} chunks. Loading embedding model...")
    embeddings = get_embedding_model(embedding_name)
    if logger:
        if embedding_name.startswith("text-embedding-3"):
            logger.info("Embedding backend selected: OpenAI (%s)", embedding_name)
        else:
            logger.info("Embedding backend selected: HuggingFace (%s)", embedding_name)

    if progress_callback:
        progress_callback(0.55, "Preparing Chroma collection...")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(db_dir),
        collection_name=collection_name,
    )

    total_chunks = len(chunks)
    if total_chunks == 0:
        if progress_callback:
            progress_callback(1.0, "No chunks found to ingest.")
        return vectorstore

    ingested = 0
    for start in range(0, total_chunks, batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch)
        ingested += len(batch)
        if logger and ingested % (batch_size * 2) == 0:
            logger.debug("Indexed chunks progress: %s/%s", ingested, total_chunks)
        if progress_callback:
            pct = 0.55 + (ingested / total_chunks) * 0.43
            progress_callback(
                min(pct, 0.98),
                f"Indexing chunks: {ingested}/{total_chunks}",
            )

    if progress_callback:
        progress_callback(1.0, "Vector ingestion complete.")
    if logger:
        logger.info("Vectorstore build completed | chunks=%s", total_chunks)
    return vectorstore


def simple_answer_from_context(query: str, retrieved: List[Document]) -> str:
    if not retrieved:
        return "No relevant context found."
    top = retrieved[0]
    return (
        f"Based on retrieved evidence: {top.page_content[:220]}... "
        f"(source: {top.metadata.get('id', 'n/a')})"
    )


def llm_answer(query: str, retrieved: List[Document], model_name: str) -> str:
    context = "\n\n".join([d.page_content for d in retrieved[:4]])
    if not context.strip():
        return "No relevant context found."
    try:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return simple_answer_from_context(query, retrieved)

        prompt = (
            "You are a precise RAG assistant. Use only the provided context.\n"
            f"Question: {query}\n\nContext:\n{context}\n\n"
            "Answer in 2-4 sentences with concrete facts."
        )
        payload = {
            "api_key": api_key,
            "url": "https://api.openai.com/v1/chat/completions",
            "timeout": 90,
            "body": {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        }
        response_json = OpenAIHTTPEmbeddings._openai_subprocess_request(payload)
        content = response_json["choices"][0]["message"]["content"]
        return str(content)
    except Exception:
        return simple_answer_from_context(query, retrieved)


def answer_relevance(question: str, answer: str) -> float:
    vec = TfidfVectorizer()
    x = vec.fit_transform([question, answer])
    q = x[0].toarray()[0]
    a = x[1].toarray()[0]
    denom = np.linalg.norm(q) * np.linalg.norm(a)
    if denom == 0:
        return 0.0
    return float(np.dot(q, a) / denom)


def sample_eval_set(rows: List[dict], per_category: int = 18) -> List[EvalRow]:
    bucket: Dict[str, List[dict]] = {c: [] for c in CATEGORIES}
    for row in rows:
        bucket[row["category"]].append(row)
    eval_rows: List[EvalRow] = []
    rng = random.Random(7)
    for category, items in bucket.items():
        chosen = rng.sample(items, k=min(per_category, len(items)))
        for row in chosen:
            eval_rows.append(
                EvalRow(
                    query=row["query"],
                    category=category,
                    expected_doc_id=row["id"],
                    expected_answer=row["answer"],
                    expected_keywords=row["keywords"],
                )
            )
    return eval_rows


def evaluate_rag(
    vectorstore: Chroma,
    eval_set: List[EvalRow],
    top_k: int,
    model_name: str,
    use_llm_answering: bool,
    progress_slot,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    retrieval_rows = []
    answer_rows = []
    total = len(eval_set)
    progress = progress_slot.progress(0.0)
    status = progress_slot.empty()

    for i, row in enumerate(eval_set):
        retrieved = vectorstore.similarity_search(row.query, k=top_k)
        ranked_ids = [d.metadata.get("id", "") for d in retrieved]

        rank = None
        if row.expected_doc_id in ranked_ids:
            rank = ranked_ids.index(row.expected_doc_id) + 1
        mrr = 1 / rank if rank else 0.0
        ndcg = ndcg_from_rank(rank)

        merged_text = " ".join([d.page_content for d in retrieved]).lower()
        kw_hits = sum(1 for kw in row.expected_keywords if kw.lower() in merged_text)
        coverage = kw_hits / max(len(row.expected_keywords), 1)

        if use_llm_answering:
            answer = llm_answer(row.query, retrieved, model_name)
        else:
            answer = simple_answer_from_context(row.query, retrieved)

        accuracy = token_f1(row.expected_answer, answer)
        completeness = sum(
            1 for kw in row.expected_keywords if kw.lower() in answer.lower()
        ) / max(len(row.expected_keywords), 1)
        relevance = answer_relevance(row.query, answer)

        retrieval_rows.append(
            {
                "category": row.category,
                "query": row.query,
                "mrr": mrr,
                "ndcg": ndcg,
                "coverage": coverage,
            }
        )
        answer_rows.append(
            {
                "category": row.category,
                "query": row.query,
                "accuracy": accuracy * 5,
                "completeness": completeness * 5,
                "relevance": relevance * 5,
                "predicted_answer": answer,
            }
        )
        progress_value = (i + 1) / total
        progress.progress(progress_value)
        status.markdown(f"Evaluating test **{i+1}/{total}**...")
        if logger and (i + 1) % 20 == 0:
            logger.debug("Evaluation progress: %s/%s", i + 1, total)
    status.markdown(
        f"<div class='green-ok'>✓ Evaluation Complete: {total} tests</div>",
        unsafe_allow_html=True,
    )
    if logger:
        logger.info("Evaluation completed | tests=%s | top_k=%s", total, top_k)

    return pd.DataFrame(retrieval_rows), pd.DataFrame(answer_rows)


def summarize(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    return {col: float(df[col].mean()) for col in cols}


def category_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.groupby("category")[cols].mean().reset_index()


def show_metric_cards(title: str, metric_map: Dict[str, float], unit: str = "") -> None:
    st.subheader(title)
    cols = st.columns(len(metric_map))
    for col, (k, v) in zip(cols, metric_map.items()):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem;color:#66768a;">{k}</div>
                    <div style="font-size:2rem;font-weight:700;color:#d48400;">{v:.4f}{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    load_dotenv(APP_ROOT / ".env")

    st.set_page_config(
        page_title="RAG Evaluation Studio",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_professional_theme()

    if not st.session_state.get("boot_animation_done", False):
        boot_box = st.empty()
        boot_box.markdown("#### Initializing RAG Evaluation Studio...")
        boot_bar = st.progress(0.0)
        for i in range(1, 101, 10):
            boot_bar.progress(i / 100)
            time.sleep(0.02)
        boot_box.empty()
        st.session_state["boot_animation_done"] = True

    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0;">RAG Evaluation Studio</h2>
          <p style="margin:0.5rem 0 0 0;color:#516173;">
            Professional benchmark environment for retrieval quality and answer quality,
            with configurable ingestion parameters and live progress tracking.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(DISCLAIMER_TEXT)
    st.write("")

    with st.sidebar:
        st.header("Parameters")
        log_level_name = st.selectbox("Log Level", list(LOG_LEVELS.keys()), index=1)
        logger, log_file = ensure_logger(log_level_name)
        st.caption(f"Log file: `{log_file.name}`")

        model_options = [DEFAULT_MODEL, "gpt-4.1", "gpt-4o-mini"]
        model_name = st.selectbox("MODEL", model_options, index=0)

        embedding_provider = st.radio(
            "Embedding Provider",
            ["OpenAI", "HuggingFace"],
            index=1,
            horizontal=True,
            help="Choose between API-based embeddings (OpenAI) or local/free embeddings (HuggingFace).",
        )
        if embedding_provider == "OpenAI":
            openai_models = list(OPENAI_EMBEDDING_OPTIONS.keys())
            embedding_name = st.selectbox(
                "EMBEDDING_MODEL",
                openai_models,
                index=0,
            )
            st.caption(f"Cost note: {OPENAI_EMBEDDING_OPTIONS[embedding_name]}")
        else:
            hf_models = list(HUGGINGFACE_EMBEDDING_OPTIONS.keys())
            embedding_name = st.selectbox(
                "EMBEDDING_MODEL",
                hf_models,
                index=0,
            )
            st.caption(f"Cost note: {HUGGINGFACE_EMBEDDING_OPTIONS[embedding_name]}")
        collection_name = st.selectbox("COLLECTION_NAME", COLLECTION_OPTIONS, index=0)
        db_name = st.text_input("DB_NAME", value=DEFAULT_DB_NAME)
        knowledge_base = st.text_input("KNOWLEDGE_BASE", value=DEFAULT_KNOWLEDGE_BASE)

        chunk_size = st.slider("Chunk Size", 200, 1800, 700, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 400, 100, 10)
        top_k = st.slider("Retrieval K", 1, 20, 6, 1)
        per_category = st.slider("Benchmark Samples / Category", 8, 30, 18, 1)
        use_llm = st.toggle("Use LLM for answer generation", value=True)

    kb_path = Path(knowledge_base)
    db_path = Path(db_name)

    col_a, col_b = st.columns([1.05, 1.95])
    with col_a:
        st.subheader("Ingestion")
        if st.button("Force New DB Run", use_container_width=True):
            base = db_path / "_fresh_runs"
            base.mkdir(parents=True, exist_ok=True)
            fresh_dir = base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            fresh_dir.mkdir(parents=True, exist_ok=True)
            st.session_state["active_vector_db_path"] = str(fresh_dir)
            st.success(f"New DB run path set: {fresh_dir}")

        if st.button("Create/Load 1000 Sample JSON", use_container_width=True):
            p = ensure_sample_data(kb_path, 1000)
            st.success(f"Sample data ready: {p}")

        if st.button("Ingest into Chroma Vector DB", type="primary", use_container_width=True):
            logger.info(
                "Ingestion requested | provider=%s | embedding=%s | collection=%s",
                embedding_provider,
                embedding_name,
                collection_name,
            )
            dataset_path = ensure_sample_data(kb_path, 1000)
            rows = load_rows(dataset_path)
            docs = to_documents(rows)

            ingest_bar = st.progress(0.0)
            ingest_state = st.empty()
            ingest_state.info("Starting ingestion pipeline...")

            def update_ingest_progress(value: float, message: str) -> None:
                ingest_bar.progress(max(0.0, min(1.0, value)))
                ingest_state.info(message)

            try:
                st.session_state.pop("embedding_warning", None)
                target_db_path = Path(st.session_state.get("active_vector_db_path", str(db_path)))
                db_path = _reset_vector_db_path(target_db_path, logger, ingest_state)

                vectorstore = build_vectorstore(
                    docs=docs,
                    embedding_name=embedding_name,
                    db_dir=db_path,
                    collection_name=collection_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress_callback=update_ingest_progress,
                    logger=logger,
                )
                ingest_bar.progress(1.0)
                ingest_state.success(
                    f"Ingestion complete. Indexed {len(docs)} base documents into `{collection_name}`."
                )
                st.session_state["vectorstore"] = vectorstore
                st.session_state["rows"] = rows
                embedding_obj = getattr(vectorstore, "_embedding_function", None) or getattr(
                    vectorstore, "embedding_function", None
                )
                if getattr(embedding_obj, "last_warning", None):
                    st.session_state["embedding_warning"] = embedding_obj.last_warning
                    logger.warning(st.session_state["embedding_warning"])
            except Exception as exc:
                ingest_state.error(f"Ingestion failed safely: {exc}")
                err_txt = str(exc)
                if "Could not clear previous vector DB (file lock)" in err_txt:
                    st.session_state["db_lock_error"] = True
                else:
                    st.session_state["db_lock_error"] = False
                if logger.level <= logging.DEBUG:
                    logger.exception("Ingestion failed safely")
                else:
                    logger.error("Ingestion failed safely: %s", exc)

        if st.session_state.get("db_lock_error", False):
            st.warning("Detected DB file-lock issue.")
            if st.button("Fix DB Lock and Retry Cleanup", use_container_width=True):
                lock_fix_state = st.empty()
                try:
                    _reset_vector_db_path(db_path, logger, lock_fix_state)
                    st.session_state["db_lock_error"] = False
                    st.session_state["db_lock_autofix_failed"] = False
                    st.success("Vector DB lock cleanup succeeded. Click ingest again.")
                except Exception as fix_exc:
                    st.session_state["db_lock_autofix_failed"] = True
                    st.session_state["db_lock_autofix_error"] = str(fix_exc)

        if st.session_state.get("db_lock_autofix_failed", False):
            st.error(f"Auto-fix failed: {st.session_state.get('db_lock_autofix_error', 'Unknown error')}")
            st.info("Use `Reset Session` to release app state and retry cleanly.")
            if st.button("Reset Session", use_container_width=True):
                for key in [
                    "vectorstore",
                    "rows",
                    "retrieval_df",
                    "answer_df",
                    "embedding_warning",
                    "db_lock_error",
                    "db_lock_autofix_failed",
                    "db_lock_autofix_error",
                    "active_vector_db_path",
                    "log_lines",
                    "boot_animation_done",
                ]:
                    st.session_state.pop(key, None)
                gc.collect()
                st.success("Session reset complete. Re-run ingestion now.")
                st.rerun()

        if st.session_state.get("active_vector_db_path") and st.session_state.get("active_vector_db_path") != str(db_path):
            st.caption(f"Active vector DB path for current run: `{st.session_state['active_vector_db_path']}`")

        if st.session_state.get("embedding_warning"):
            st.warning(st.session_state["embedding_warning"])

        if embedding_provider == "OpenAI" or use_llm:
            st.caption("Tip: Keep your `OPENAI_API_KEY` in environment variables for OpenAI-based runs.")
        else:
            st.caption("You are in local-cost mode for embeddings (HuggingFace).")

    with col_b:
        tab1, tab2 = st.tabs(["Retrieval Evaluation", "Answer Evaluation"])

        if "vectorstore" not in st.session_state:
            st.info("Run ingestion first to activate dashboard.")
            return

        if st.button("Run Evaluation", use_container_width=True):
            logger.info(
                "Evaluation requested | top_k=%s | model=%s | use_llm=%s",
                top_k,
                model_name,
                use_llm,
            )
            eval_set = sample_eval_set(st.session_state["rows"], per_category=per_category)
            retrieval_df, answer_df = evaluate_rag(
                vectorstore=st.session_state["vectorstore"],
                eval_set=eval_set,
                top_k=top_k,
                model_name=model_name,
                use_llm_answering=use_llm,
                progress_slot=st,
                logger=logger,
            )
            st.session_state["retrieval_df"] = retrieval_df
            st.session_state["answer_df"] = answer_df

        with st.expander("Application Logs", expanded=False):
            st.caption(f"Writing to `{Path(st.session_state.get('log_file_path', 'applog.log')).name}`")
            log_text = "\n".join(st.session_state.get("log_lines", [])[-200:])
            st.text_area("Live logs", value=log_text, height=220)

        if "retrieval_df" not in st.session_state or "answer_df" not in st.session_state:
            st.warning("Click `Run Evaluation` to generate live metrics.")
            return

        retrieval_df = st.session_state["retrieval_df"]
        answer_df = st.session_state["answer_df"]

        with tab1:
            top_metrics = summarize(retrieval_df, ["mrr", "ndcg", "coverage"])
            show_metric_cards("Retrieval Headline Metrics", top_metrics)
            csum = category_summary(retrieval_df, ["mrr", "ndcg", "coverage"])

            fig = px.bar(
                csum.melt(id_vars="category", var_name="metric", value_name="score"),
                x="category",
                y="score",
                color="metric",
                barmode="group",
                title="Category-wise Retrieval Metrics",
                color_discrete_sequence=["#f8961e", "#4d7cff", "#2a9d8f"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(csum, use_container_width=True)

        with tab2:
            top_metrics = summarize(answer_df, ["accuracy", "completeness", "relevance"])
            show_metric_cards("Answer Headline Metrics (out of 5)", top_metrics)
            csum = category_summary(answer_df, ["accuracy", "completeness", "relevance"])
            radar = px.line_polar(
                csum.melt(id_vars="category", var_name="metric", value_name="score"),
                r="score",
                theta="metric",
                color="category",
                line_close=True,
                title="Answer Quality Radar by Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(radar, use_container_width=True)
            st.dataframe(csum, use_container_width=True)


if __name__ == "__main__":
    main()
