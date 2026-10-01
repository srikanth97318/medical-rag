from typing import Any
from .utils import normalize_text

def chunk_documents(documents: list[dict[str, Any]], chunk_size: int = 1200, overlap: int = 150) -> list[dict[str, Any]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size: raise ValueError("Require chunk_size > overlap >= 0")
    chunks = []
    for doc in documents:
        text = normalize_text(doc.get("text", ""))
        start = 0
        while start < len(text):
            part = text[start:start + chunk_size].strip()
            if part:
                chunks.append({"document_id": f"{doc['document_id']}:{len(chunks)}", "parent_id": doc["document_id"],
                               "text": part, "source": doc.get("source", "unknown"), "metadata": doc.get("metadata", {})})
            if start + chunk_size >= len(text): break
            start += chunk_size - overlap
    return chunks
