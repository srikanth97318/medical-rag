"""Claim-level NLI, answer consistency, and NOT_COVERED rule engine."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_THRESHOLD = 0.60
CONTRADICTION_THRESHOLD = 0.60
RELEVANCE_THRESHOLD = 0.25
MEDICAL_TERMS = re.compile(r"\b(medical|health|symptom|disease|diagnos|treatment|pain|fever|medication|drug|doctor|patient|blood|infection|cancer|heart|lung|therapy|pregnan|vaccine|headache|rash|injury|condition|syndrome|illness)\w*\b", re.I)
STOPWORDS = {"what", "when", "where", "which", "who", "why", "how", "does", "this", "that", "with", "from", "have", "about", "are", "the", "and", "for", "can", "you", "your", "into", "after", "before"}

def _lexical_relevance(query: str, contexts: list[str]) -> float:
    """Fallback coverage estimate for direct verifier calls without retriever scores."""
    terms = {word for word in re.findall(r"\b\w+\b", query.lower()) if len(word) > 2 and word not in STOPWORDS}
    if not terms:
        return 0.0
    return max((len(terms & set(re.findall(r"\b\w+\b", passage.lower()))) / len(terms) for passage in contexts), default=0.0)

def split_claims(text: str) -> list[str]:
    """Split a response into sentence-like factual claims without extra corpora."""
    text = re.sub(r"(?im)^\s*(disclaimer|note|warning):.*$", "", text)
    return [part.strip(" \t\n•-*;") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.strip()) > 12]

@lru_cache(maxsize=2)
def _load_nli(model_name: str, device: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=device)

def _label_indices(model: CrossEncoder) -> dict[str, int]:
    config = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(config, "id2label", {}) if config else {}
    labels = {str(label).lower(): int(index) for index, label in id2label.items()}
    # The named DeBERTa NLI cross-encoder uses the conventional contradiction/entailment/neutral order.
    if not any("entail" in label for label in labels):
        return {"contradiction": 0, "entailment": 1, "neutral": 2}
    return {kind: next(i for label, i in labels.items() if kind in label) for kind in ("entailment", "neutral", "contradiction") if any(kind in label for label in labels)}

def _consistency_score(primary_claims: list[str], sample_answers: list[str]) -> dict[str, Any]:
    """SelfCheck-inspired claim agreement across stochastic sample answers."""
    if not primary_claims or not sample_answers:
        return {"score": None, "claim_agreement": [], "samples": len(sample_answers)}
    samples = [split_claims(answer) for answer in sample_answers]
    agreements = []
    for claim in primary_claims:
        tokens = {w.lower() for w in re.findall(r"\b\w+\b", claim) if len(w) > 3}
        per_sample = []
        for sample_claims in samples:
            best = 0.0
            for candidate in sample_claims:
                other = {w.lower() for w in re.findall(r"\b\w+\b", candidate) if len(w) > 3}
                best = max(best, len(tokens & other) / max(1, len(tokens | other)))
            per_sample.append(best >= 0.35)
        agreements.append({"claim": claim, "agreement_ratio": sum(per_sample) / len(per_sample)})
    return {"score": sum(x["agreement_ratio"] for x in agreements) / len(agreements), "claim_agreement": agreements, "samples": len(sample_answers), "method": "SelfCheck-inspired sampled-answer claim consistency; not official SelfCheckGPT"}

def verify_response(query: str, context: list[str], response: str, context_scores: list[float] | None = None, sample_answers: list[str] | None = None, model_name: str = NLI_MODEL, relevance_threshold: float = RELEVANCE_THRESHOLD, device: str | None = None, generator: Any | None = None, selfcheck_samples: int = 3) -> dict[str, Any]:
    """Evaluate answer claims against retrieved passages, preserving insufficient evidence."""
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    query = (query or "").strip()
    if not MEDICAL_TERMS.search(query):
        return {"is_grounded": False, "hallucination_detected": False, "is_off_topic": True, "status": "INSUFFICIENT_INFORMATION", "nli_support_ratio": 0.0, "unsupported_claims": [], "claim_results": [], "selfcheck": _consistency_score([], sample_answers or []), "reason": "Query is outside the lexical medical topic guard."}
    if "INSUFFICIENT_CONTEXT" in (response or "").upper():
        return {"is_grounded": False, "hallucination_detected": False, "is_off_topic": False, "status": "INSUFFICIENT_INFORMATION", "nli_support_ratio": 0.0, "unsupported_claims": [], "claim_results": [], "selfcheck": _consistency_score([], sample_answers or []), "reason": "Generator explicitly reported insufficient context."}
    relevance = max(context_scores) if context_scores else _lexical_relevance(query, context)
    if not context or relevance < relevance_threshold:
        return {"is_grounded": False, "hallucination_detected": False, "is_off_topic": False, "status": "INSUFFICIENT_INFORMATION", "nli_support_ratio": 0.0, "unsupported_claims": [], "claim_results": [], "context_relevance": relevance, "selfcheck": _consistency_score([], sample_answers or []), "reason": "No retrieved passage reached the context relevance threshold."}
    claims = split_claims(response)
    if not claims:
        return {"is_grounded": True, "hallucination_detected": False, "is_off_topic": False, "status": "GROUNDED", "nli_support_ratio": 1.0, "unsupported_claims": [], "claim_results": [], "context_relevance": relevance, "selfcheck": _consistency_score([], sample_answers or []), "reason": "No verifiable factual claims were extracted."}
    if sample_answers is None:
        if generator is None:
            from rag_core import get_medical_generator
            generator = get_medical_generator(0.7)
        sample_answers = [generator.generate(query, context, temperature=0.7) for _ in range(selfcheck_samples)]
    model = _load_nli(model_name, actual_device)
    pairs = [(passage, claim) for claim in claims for passage in context]
    predictions = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
    label_indices = _label_indices(model)
    claim_results, unsupported = [], []
    for index, claim in enumerate(claims):
        passage_scores = predictions[index * len(context):(index + 1) * len(context)]
        combined = {label: max(float(row[label_indices[label]]) for row in passage_scores) if label in label_indices else 0.0 for label in ("entailment", "neutral", "contradiction")}
        if combined["entailment"] >= ENTAILMENT_THRESHOLD:
            status = "entailment"
        elif combined["contradiction"] >= CONTRADICTION_THRESHOLD:
            status = "contradiction"
            unsupported.append(claim)
        else:
            status = "neutral"
            unsupported.append(claim)
        claim_results.append({"claim": claim, **combined, "status": status})
    support_ratio = sum(item["status"] == "entailment" for item in claim_results) / len(claim_results)
    hallucinated = bool(unsupported)
    return {"is_grounded": not hallucinated, "hallucination_detected": hallucinated, "is_off_topic": False, "status": "HALLUCINATED" if hallucinated else "GROUNDED", "nli_support_ratio": support_ratio, "unsupported_claims": unsupported, "claim_results": claim_results, "context_relevance": relevance, "selfcheck": _consistency_score(claims, sample_answers or []), "reason": "Claim-level NLI against retrieved context."}
