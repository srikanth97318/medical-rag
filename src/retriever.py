import json
from pathlib import Path
from typing import Any
import numpy as np

class Retriever:
    def __init__(self, embedder: Any, index: Any, documents: list[dict[str, Any]], threshold: float = 0.0):
        self.embedder, self.index, self.documents, self.threshold = embedder, index, documents, threshold
    @classmethod
    def build(cls, documents: list[dict[str, Any]], embedder: Any, threshold: float = 0.0) -> "Retriever":
        import faiss
        if not documents: raise ValueError("Cannot build an index without documents")
        vectors = embedder.encode([d["text"] for d in documents])
        index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
        return cls(embedder, index, documents, threshold)
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or not self.documents: return []
        vector = self.embedder.encode_queries([query]) if hasattr(self.embedder,"encode_queries") else self.embedder.encode([query])
        scores, ids = self.index.search(vector, top_k)
        out = []
        for rank, (score, ix) in enumerate(zip(scores[0], ids[0]), 1):
            if ix < 0 or float(score) < self.threshold: continue
            doc = self.documents[int(ix)]
            out.append({"rank": rank, "score": float(score), "text": doc["text"], "source": doc.get("source", "unknown"),
                        "document_id": doc["document_id"], "metadata": doc.get("metadata", {})})
        return out
    def save(self, directory: str | Path, embedding_model: str) -> None:
        import faiss
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        (p / "documents.json").write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")
        (p / "index_metadata.json").write_text(json.dumps({"embedding_model": embedding_model, "dimension": self.index.d, "count": len(self.documents)}), encoding="utf-8")
    @classmethod
    def load(cls, directory: str | Path, embedder: Any, threshold: float = 0.0) -> "Retriever":
        import faiss
        p = Path(directory)
        if not (p / "index.faiss").exists(): raise FileNotFoundError(f"FAISS index missing in {p}; run scripts/build_index.py")
        return cls(embedder, faiss.read_index(str(p / "index.faiss")), json.loads((p / "documents.json").read_text(encoding="utf-8")), threshold)
