"""State-machine RAG workflow with claim verification, correction, and node timings."""
from __future__ import annotations

import time
from typing import Any

from rag_core import get_medical_generator, get_medical_retriever
from verification_engine import MEDICAL_TERMS, verify_response

SAFE_FALLBACK = "I could not verify a sufficiently grounded answer from the available medical context. Please consult a qualified healthcare professional or an authoritative medical source."
DEFAULT_RELEVANCE_THRESHOLD = 0.25

def _record(state: dict[str, Any], node: str, started: float, detail: str = "") -> None:
    elapsed = time.perf_counter() - started
    state["phase_timings"][node] = state["phase_timings"].get(node, 0.0) + elapsed
    state["state_history"].append({"node": node, "elapsed_seconds": elapsed, "attempt": state["attempt_count"], "detail": detail})

def retrieve_node(state: dict[str, Any], retriever: Any | None = None) -> dict[str, Any]:
    """Fetch ranked Top-K passages and remember the evidence scores."""
    started = time.perf_counter()
    retriever = retriever or get_medical_retriever()
    state["retrieved_docs"] = retriever.retrieve(state["query"], state["top_k"])
    state["context_scores"] = [float(item["score"]) for item in state["retrieved_docs"]]
    _record(state, "retrieval", started, f"Retrieved {len(state['retrieved_docs'])} passages")
    return state

def generate_node(state: dict[str, Any], generator: Any | None = None, correction_feedback: str | None = None) -> dict[str, Any]:
    """Generate an answer, or mark a below-threshold retrieval insufficient."""
    started = time.perf_counter()
    if not state["retrieved_docs"] or max(state["context_scores"], default=-1.0) < state["relevance_threshold"]:
        state["current_answer"] = "INSUFFICIENT_CONTEXT"
        _record(state, "generation", started, "Skipped: no passage met the relevance threshold; this also filters off-topic queries")
        return state
    generator = generator or get_medical_generator(state["temperature"])
    contexts = [item["text"] for item in state["retrieved_docs"]]
    state["current_answer"] = generator.generate(state["query"], contexts, temperature=state["temperature"], correction_feedback=correction_feedback)
    _record(state, "generation" if state["attempt_count"] == 0 else "correction_generation", started)
    return state

def verify_node(state: dict[str, Any], generator: Any | None = None, verifier=verify_response, nli_model: str | None = None) -> dict[str, Any]:
    """Run three sampled response checks and NLI against retrieved evidence."""
    started = time.perf_counter()
    contexts = [item["text"] for item in state["retrieved_docs"]]
    samples: list[str] = []
    if state["current_answer"] != "INSUFFICIENT_CONTEXT" and contexts:
        generator = generator or get_medical_generator(state["temperature"])
        for _ in range(state["selfcheck_samples"]):
            samples.append(generator.generate(state["query"], contexts, temperature=max(state["temperature"], 0.65)))
    options = {"context_scores": state["context_scores"], "relevance_threshold": state["relevance_threshold"], "device": state["device"]}
    if nli_model:
        options["model_name"] = nli_model
    state["verification_result"] = verifier(state["query"], contexts, state["current_answer"], sample_answers=samples, **options)
    _record(state, "verification", started, state["verification_result"].get("status", "unknown"))
    return state

def correct_node(state: dict[str, Any], generator: Any | None = None) -> dict[str, Any]:
    """Regenerate after a hallucination with claim-specific correction feedback."""
    started = time.perf_counter()
    result = state["verification_result"]
    evidence = "\n".join(item["text"] for item in state["retrieved_docs"])
    claims = "\n".join(f"- {claim}" for claim in result.get("unsupported_claims", [])) or "- Claims identified by NLI as neutral or contradictory"
    feedback = f"The previous answer contained unsupported or contradicted claims. Remove these claims unless directly supported by evidence:\n{claims}\n\nAvailable supporting passages:\n{evidence}"
    state["attempt_count"] += 1
    generate_node(state, generator, feedback)
    _record(state, "correct", started, f"Correction attempt {state['attempt_count']}")
    return state

def initial_state(query: str, top_k: int = 5, max_retries: int = 2, temperature: float = 0.1, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD, selfcheck_samples: int = 3, device: str | None = None) -> dict[str, Any]:
    """Construct the documented workflow state dictionary."""
    return {"query": query, "retrieved_docs": [], "current_answer": "", "attempt_count": 0, "verification_result": {}, "is_final": False,
            "top_k": int(top_k), "max_retries": int(max_retries), "temperature": float(temperature), "relevance_threshold": float(relevance_threshold),
            "selfcheck_samples": int(selfcheck_samples), "device": device, "context_scores": [], "state_history": [], "phase_timings": {},
            "correction_history": [], "fallback_applied": False, "total_seconds": 0.0}

def run_dag(query: str, top_k: int = 5, max_retries: int = 2, temperature: float = 0.1, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD, selfcheck_samples: int = 3, device: str | None = None, retriever: Any | None = None, generator: Any | None = None, verifier=verify_response, nli_model: str | None = None) -> dict[str, Any]:
    """Execute Retrieve → Generate → Verify → optional Correct until final."""
    total_started = time.perf_counter()
    state = initial_state(query, top_k, max_retries, temperature, relevance_threshold, selfcheck_samples, device)
    retrieve_node(state, retriever)
    generate_node(state, generator)
    verify_node(state, generator, verifier, nli_model)
    while state["verification_result"].get("status") == "HALLUCINATED":
        if state["attempt_count"] >= state["max_retries"]:
            state["current_answer"] = SAFE_FALLBACK
            state["fallback_applied"] = True
            state["state_history"].append({"node": "safe_fallback", "elapsed_seconds": 0.0, "attempt": state["attempt_count"], "detail": "Retry limit reached"})
            state["is_final"] = True
            break
        previous = state["current_answer"]
        correction_started = time.perf_counter()
        correct_node(state, generator)
        state["correction_history"].append({"attempt": state["attempt_count"], "previous_answer": previous, "corrected_answer": state["current_answer"]})
        verify_node(state, generator, verifier, nli_model)
        state["phase_timings"]["self_correction_loop"] = state["phase_timings"].get("self_correction_loop", 0.0) + time.perf_counter() - correction_started
    if state["verification_result"].get("status") in {"GROUNDED", "INSUFFICIENT_INFORMATION"}:
        state["is_final"] = True
    state["total_seconds"] = time.perf_counter() - total_started
    state["phase_timings"]["total"] = state["total_seconds"]
    state["final_answer"] = state["current_answer"]
    verification_status = state["verification_result"].get("status", "INSUFFICIENT_INFORMATION")
    state["status"] = "CORRECTED" if verification_status == "GROUNDED" and state["attempt_count"] else verification_status
    state["confidence"] = state["verification_result"].get("nli_support_ratio", 0.0)
    return state

def run_self_correction_dag(query: str, top_k: int = 5, max_retries: int = 2, temperature: float = 0.1, **kwargs: Any) -> dict[str, Any]:
    """Compatibility-friendly public entry point for the state-machine workflow."""
    return run_dag(query, top_k=top_k, max_retries=max_retries, temperature=temperature, **kwargs)
