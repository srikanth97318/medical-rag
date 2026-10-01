from typing import Any
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class EvidenceSufficiencyChecker:
    def __init__(self, supported_threshold: float = 0.55, partial_threshold: float = 0.3):
        self.supported_threshold, self.partial_threshold = supported_threshold, partial_threshold
    def check(self, query: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
        if not passages: return {"status":"NOT_COVERED", "score":0.0, "reason":"No relevant passages were retrieved."}
        qterms = {w.lower() for w in query.split() if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 2}
        if not qterms: return {"status":"NOT_COVERED", "score":0.0, "reason":"The question contains no informative terms."}
        allterms = set()
        for p in passages: allterms.update(w.lower() for w in p.get("text", "").split())
        coverage = len(qterms & allterms) / len(qterms)
        relevance = max(0.0, min(1.0, sum(max(0.0, float(p.get("score", 0))) for p in passages[:3]) / min(3, len(passages))))
        # Joint signal: lexical query coverage and semantic retrieval relevance.
        score = 0.65 * coverage + 0.35 * relevance
        if score >= self.supported_threshold: status = "SUPPORTED"
        elif score >= self.partial_threshold: status = "PARTIALLY_SUPPORTED"
        else: status = "NOT_COVERED"
        reason = f"Combined query-term coverage ({coverage:.2f}) and retrieval relevance ({relevance:.2f})."
        return {"status":status, "score":float(score), "reason":reason}
