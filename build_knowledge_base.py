#!/usr/bin/env python3
"""Build a normalized FAISS knowledge base from local medical QA CSV files."""
from __future__ import annotations

import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

LOG = logging.getLogger("medical_rag.build_knowledge_base")
ROOT = Path(__file__).resolve().parent
DATASETS = (ROOT / "data" / "raw" / "medquad.csv", ROOT / "data" / "raw" / "disease_diagnosis.csv")
OUTPUT_DIR = ROOT / "vector_store"
INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
METADATA_PATH = OUTPUT_DIR / "metadata.pkl"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

def _clean(value: Any) -> str:
    """Normalize whitespace, lowercase text, and turn missing values into empty strings."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()

def _find_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    columns = {str(c).strip().lower(): str(c) for c in df.columns}
    return next((columns[n] for n in names if n in columns), None)

def _row_to_document(row: pd.Series, question_col: str | None, answer_col: str | None) -> tuple[str, dict[str, Any]] | None:
    """Map common MedQuAD and disease-diagnosis schemas into a QA document."""
    q = _clean(row[question_col]) if question_col else ""
    a = _clean(row[answer_col]) if answer_col else ""
    if not q:
        symptom_columns = [c for c in row.index if "symptom" in str(c).lower()]
        symptoms = [_clean(row[c]) for c in symptom_columns if _clean(row[c])]
        q = "symptoms: " + ", ".join(symptoms) if symptoms else ""
    if not a:
        value_cols = [c for c in row.index if any(t in str(c).lower() for t in ("diagnos", "treatment", "severity", "answer"))]
        values = [f"{str(c).strip().lower()}: {_clean(row[c])}" for c in value_cols if _clean(row[c])]
        a = "; ".join(values)
    elif answer_col:
        supplemental = [f"{str(c).strip().lower()}: {_clean(row[c])}" for c in row.index
                        if c != answer_col and any(t in str(c).lower() for t in ("treatment", "severity")) and _clean(row[c])]
        if supplemental:
            a += "; " + "; ".join(supplemental)
    if not q or not a:
        return None
    doc = f"Question: {q} | Answer: {a}"
    metadata = {str(c): _clean(row[c]) for c in row.index if not pd.isna(row[c]) and not any(s in str(c).lower() for s in ("patient_id", "mrn", "medical_record", "patient_name"))}
    return doc, metadata

def load_medical_documents(paths: tuple[Path, ...] = DATASETS) -> tuple[list[str], list[dict[str, Any]]]:
    """Load, normalize, deduplicate, and merge supported local CSV datasets."""
    docs: list[str] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset is missing: {path}")
        LOG.info("Loading %s", path)
        frame = pd.read_csv(path).drop_duplicates().fillna("")
        q_col = _find_column(frame, ("question", "query", "medical_question"))
        a_col = _find_column(frame, ("answer", "response", "diagnosis", "treatment_plan"))
        if not q_col and not any("symptom" in str(c).lower() for c in frame.columns):
            raise ValueError(f"No question or symptom columns found in {path.name}")
        for _, row in frame.iterrows():
            item = _row_to_document(row, q_col, a_col)
            if item is None:
                continue
            doc, meta = item
            if doc in seen:
                continue
            seen.add(doc)
            docs.append(doc)
            metadata.append({"document_id": len(docs) - 1, "source_file": path.name, "text": doc, "source_metadata": meta})
        LOG.info("Loaded %d usable records from %s", len(docs), path.name)
    if not docs:
        raise ValueError("No usable question/answer records were found in the configured datasets")
    return docs, metadata

def build_knowledge_base() -> dict[str, Any]:
    """Encode all documents in batches and persist a FAISS index plus aligned metadata."""
    started = time.perf_counter()
    docs, metadata = load_medical_documents()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("Loading embedding model %s on %s", MODEL_NAME, device)
    model_started = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=device)
    LOG.info("Embedding model initialized in %.2fs", time.perf_counter() - model_started)
    # Length bucketing reduces padding waste; vectors are restored to original row order afterward.
    order = np.asarray(sorted(range(len(docs)), key=lambda index: len(docs[index])), dtype=np.int64)
    ordered_docs = [docs[int(index)] for index in order]
    encoded = model.encode(ordered_docs, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    encoded = np.ascontiguousarray(encoded, dtype=np.float32)
    vectors = np.empty_like(encoded)
    vectors[order] = encoded
    if vectors.ndim != 2 or vectors.shape[0] != len(metadata):
        raise RuntimeError("Embedding output is not aligned with source document metadata")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index.ntotal != len(docs):
        raise RuntimeError(f"FAISS indexed {index.ntotal} vectors for {len(docs)} documents")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    payload = {"documents": metadata, "embedding_model": MODEL_NAME, "embedding_dimension": int(vectors.shape[1]), "index_type": "IndexFlatIP", "normalized_embeddings": True}
    with METADATA_PATH.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    LOG.info("Saved %d vectors in %.2fs to %s", index.ntotal, elapsed, OUTPUT_DIR)
    return {"document_count": len(docs), "dimension": int(vectors.shape[1]), "elapsed_seconds": elapsed, "index_path": str(INDEX_PATH), "metadata_path": str(METADATA_PATH)}

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        print(build_knowledge_base())
        return 0
    except Exception:
        LOG.exception("Knowledge-base build failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
