from typing import Any
from .utils import normalize_text

class QAGVerifier:
    """Lightweight extractive QAG proxy: tests whether claim terms occur in evidence; not an LLM QAG judge."""
    def verify(self, claim: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
        terms = {w.lower().strip(".,:;()") for w in claim.split() if len(w) > 3}
        scored = []
        for p in passages:
            text = normalize_text(p.get("text", "")); low = text.lower()
            overlap = len([w for w in terms if w in low]) / max(1, len(terms))
            scored.append((overlap, p))
        if not scored or max(x[0] for x in scored) < .2:
            return {"verification_question":f"What evidence supports: {claim}","target_claim":claim,"context_answer":None,"support_result":"NOT_COVERED","supporting_chunks":[]}
        score, best = max(scored, key=lambda x:x[0])
        negations=(" not "," no "," never "," without ","n't")
        claim_negative=any(n in f" {claim.lower()} " for n in negations)
        context_negative=any(n in f" {best['text'].lower()} " for n in negations)
        status = "CONTRADICTED" if score >= .55 and claim_negative != context_negative else ("SUPPORTED" if score >= .55 else "UNSUPPORTED")
        return {"verification_question":f"What evidence supports: {claim}","target_claim":claim,"context_answer":best["text"],"support_result":status,"supporting_chunks":[best["document_id"]] if status == "SUPPORTED" else []}
