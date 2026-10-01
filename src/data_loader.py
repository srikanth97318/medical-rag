from pathlib import Path
from typing import Any
import pandas as pd
from .utils import is_empty, new_id, normalize_text

ALIASES = {
    "question": ["question", "query", "questions", "medical_question"],
    "answer": ["answer", "response", "answers", "diagnosis", "treatment", "treatment_plan"],
    "symptoms": ["symptoms", "symptom", "clinical_features"],
    "source": ["source", "url", "document", "source_name"],
    "focus": ["focus", "focus_area", "topic", "disease", "condition"],
    "severity": ["severity", "risk_level", "severity_level"],
    "treatment_plan": ["treatment_plan", "treatment", "management", "care_plan"],
}

def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"Dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".csv", ".tsv"}: return pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",")
    if suffix in {".json", ".jsonl"}: return pd.read_json(p, lines=suffix == ".jsonl")
    if suffix in {".xlsx", ".xls"}: return pd.read_excel(p)
    raise ValueError(f"Unsupported dataset format: {suffix}")

def _find_column(columns: list[str], target: str, mapping: dict[str, str]) -> str | None:
    if target in mapping and mapping[target] in columns: return mapping[target]
    low = {c.lower().strip(): c for c in columns}
    return next((low[a] for a in ALIASES[target] if a in low), None)

def dataframe_to_documents(df: pd.DataFrame, mapping: dict[str, str] | None = None, source_name: str = "dataset") -> list[dict[str, Any]]:
    mapping = mapping or {}
    columns = [str(c) for c in df.columns]
    resolved = {name: _find_column(columns, name, mapping) for name in ALIASES}
    docs, seen = [], set()
    for _, row in df.iterrows():
        fields = {k: normalize_text(row[col]) if col and col in row else "" for k, col in resolved.items()}
        # Preserve all useful source columns, including dataset-specific clinical fields.
        content = "\n".join(f"{k.title()}: {v}" for k, v in fields.items() if v and k != "source")
        known_columns = {c for c in resolved.values() if c}
        extras = [(str(c), normalize_text(row[c])) for c in columns if c not in known_columns and not is_empty(row[c])]
        if extras: content += "\n" + "\n".join(f"{k}: {v}" for k,v in extras)
        content = normalize_text(content)
        if not content or content.casefold() in seen: continue
        seen.add(content.casefold())
        docs.append({"document_id": new_id(), "text": content, "source": fields["source"] or source_name,
                     "metadata": {"question": fields["question"], "focus_area": fields["focus"], "severity": fields["severity"], "treatment_plan": fields["treatment_plan"], "schema_columns": columns}})
    return docs

def load_documents(paths: list[str | Path], mappings: dict[str, str] | None = None) -> list[dict[str, Any]]:
    docs = []
    for path in paths:
        p = Path(path)
        docs.extend(dataframe_to_documents(load_table(p), mappings, p.name))
    return docs
