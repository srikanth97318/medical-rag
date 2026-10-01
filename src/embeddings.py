from typing import Any
import numpy as np

class SentenceEmbeddingModel:
    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer
        import torch
        actual = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = SentenceTransformer(model_name, device=actual)
        self.model_name, self.device = model_name, actual
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True), dtype="float32")
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(["Represent this sentence for searching relevant passages: " + t for t in texts])
