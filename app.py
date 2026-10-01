"""Interactive Streamlit interface for the medical RAG correction DAG."""
from __future__ import annotations

import streamlit as st

from rag_core import MedicalRetriever
from self_correction_dag import run_dag
from verification_engine import MEDICAL_TERMS

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺", layout="wide")
st.title("Medical RAG Assistant")
st.caption("Educational research software grounded in the configured local knowledge base; not a substitute for professional medical advice.")

with st.sidebar:
    st.header("Pipeline settings")
    top_k = st.slider("Top-K retrieval", min_value=1, max_value=15, value=5)
    max_retries = st.slider("Maximum self-correction retries", min_value=0, max_value=4, value=2)
    temperature = st.slider("Generation temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    relevance_threshold = st.slider("Minimum context relevance", min_value=-0.1, max_value=1.0, value=0.25, step=0.01)
    debug = st.checkbox("Show raw diagnostics", value=False)

query = st.text_area("Medical question", placeholder="Enter a medical question supported by the local knowledge base", height=100)
left, right = st.columns([1, 6])
ask = left.button("Ask", type="primary", use_container_width=True)
reset = right.button("Reset", use_container_width=False)
if reset:
    st.session_state.pop("rag_result", None)
    st.rerun()

if ask:
    if not query.strip():
        st.session_state["rag_result"] = {"status": "INSUFFICIENT_INFORMATION", "final_answer": "Please enter a medical question.", "verification_result": {}, "retrieved_docs": [], "state_history": [], "phase_timings": {}}
    elif not MEDICAL_TERMS.search(query):
        st.session_state["rag_result"] = {"status": "INSUFFICIENT_INFORMATION", "final_answer": "This assistant is limited to medical information questions. Please enter an in-scope query.", "verification_result": {"is_off_topic": True, "hallucination_detected": False}, "retrieved_docs": [], "state_history": [{"node": "topic_guard", "detail": "Query classified as off topic", "elapsed_seconds": 0.0}], "phase_timings": {}}
    else:
        try:
            st.session_state["rag_result"] = run_dag(query.strip(), top_k=top_k, max_retries=max_retries, temperature=temperature, relevance_threshold=relevance_threshold)
        except FileNotFoundError as exc:
            st.session_state["rag_result"] = {"status": "INSUFFICIENT_INFORMATION", "final_answer": "The local knowledge base is not ready. Build it with `python build_knowledge_base.py` after placing MedQuAD.csv and Disease_Diagnosis.csv beside the script.", "error": str(exc), "verification_result": {}, "retrieved_docs": [], "state_history": [], "phase_timings": {}}
        except Exception as exc:
            st.session_state["rag_result"] = {"status": "INSUFFICIENT_INFORMATION", "final_answer": "The request could not be completed safely. Please try again after checking the local model and index setup.", "error": f"{type(exc).__name__}: {exc}", "verification_result": {}, "retrieved_docs": [], "state_history": [], "phase_timings": {}}

result = st.session_state.get("rag_result")
if result:
    status = result.get("status", "INSUFFICIENT_INFORMATION")
    if result.get("verification_result", {}).get("is_off_topic"):
        status = "OFF_TOPIC"
    badges = {"GROUNDED": "✅ Grounded", "CORRECTED": "🛠️ Self-corrected", "HALLUCINATED": "🚩 Verification failed", "INSUFFICIENT_INFORMATION": "ℹ️ Insufficient information", "OFF_TOPIC": "↪️ Off topic"}
    st.subheader(badges.get(status, status))
    st.markdown(result.get("final_answer", ""))
    verification = result.get("verification_result", {})
    metrics = st.columns(4)
    metrics[0].metric("NLI support", f"{verification['nli_support_ratio']:.1%}" if verification.get("nli_support_ratio") is not None else "N/A")
    metrics[1].metric("Confidence", f"{result.get('confidence', 0):.1%}" if verification.get("nli_support_ratio") is not None else "N/A")
    metrics[2].metric("Hallucination detected", "Yes" if verification.get("hallucination_detected") else "No")
    metrics[3].metric("Correction attempts", int(result.get("attempt_count", 0)))
    if result.get("fallback_applied"):
        st.warning("The retry limit was reached. A safe fallback answer is shown.")
    if result.get("error"):
        with st.expander("Setup or execution detail"):
            st.code(result["error"])
    with st.expander("Retrieved Context Passages"):
        docs = result.get("retrieved_docs", [])
        if not docs:
            st.info("No passages were retrieved.")
        for doc in docs:
            st.markdown(f"**#{doc.get('rank')} · Similarity {doc.get('score', 0):.4f} · {doc.get('source', 'local')}**")
            st.write(doc.get("text", ""))
    with st.expander("NLI Claim Breakdown"):
        claim_results = verification.get("claim_results", [])
        if not claim_results:
            st.info("No factual claims were evaluated.")
        for claim in claim_results:
            good = claim.get("status") == "entailment"
            st.markdown(f":{'green' if good else 'red'}[{claim.get('status', 'unknown').upper()}] {claim.get('claim', '')}")
            st.caption(f"Entailment {claim.get('entailment', 0):.3f} · Neutral {claim.get('neutral', 0):.3f} · Contradiction {claim.get('contradiction', 0):.3f}")
        sc = verification.get("selfcheck", {})
        if sc.get("score") is not None:
            st.metric("SelfCheck sampled-answer agreement", f"{sc['score']:.1%}")
            st.caption(sc.get("method", ""))
    with st.expander("Execution DAG & Performance Profiling"):
        st.write("**State history**")
        for step in result.get("state_history", []):
            st.write(f"{step.get('node')} · attempt {step.get('attempt', 0)} · {step.get('elapsed_seconds', 0):.4f}s · {step.get('detail', '')}")
        st.write("**Phase timings (seconds)**")
        st.json(result.get("phase_timings", {}))
        st.metric("End-to-end", f"{result.get('total_seconds', 0):.4f}s")
    if result.get("correction_history"):
        with st.expander("Self-correction history"):
            for correction in result["correction_history"]:
                st.markdown(f"**Attempt {correction['attempt']}**")
                st.write(correction["corrected_answer"])
    if debug:
        st.json(result)
