#!/usr/bin/env python3
"""Benchmark RAG latency and reference-based quality metrics, with local LLM judges."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rag_core import get_medical_generator
from self_correction_dag import run_dag

LOG = logging.getLogger("medical_rag.evaluate")
ROOT = Path(__file__).resolve().parent

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def _lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for word_a in a:
        current = [0]
        for j, word_b in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if word_a == word_b else max(previous[j], current[-1]))
        previous = current
    return previous[-1]

def reference_metrics(reference: str, prediction: str) -> dict[str, float]:
    """Return smoothed sentence BLEU, ROUGE-1 F1, and ROUGE-L F1."""
    ref, pred = _tokens(reference), _tokens(prediction)
    if not ref or not pred:
        return {"bleu": 0.0, "rouge_1": 0.0, "rouge_l": 0.0}
    from collections import Counter
    ref_counts, pred_counts = Counter(ref), Counter(pred)
    overlap = sum(min(n, pred_counts[t]) for t, n in ref_counts.items())
    precision = overlap / max(1, len(pred))
    recall = overlap / max(1, len(ref))
    rouge1 = 2 * precision * recall / max(1e-12, precision + recall)
    lcs = _lcs_length(ref, pred)
    lp, lr = lcs / len(pred), lcs / len(ref)
    rouge_l = 2 * lp * lr / max(1e-12, lp + lr)
    precisions = []
    for n in (1, 2, 3, 4):
        ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref) - n + 1)))
        pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(max(0, len(pred) - n + 1)))
        hit = sum(min(v, pred_ngrams[k]) for k, v in ref_ngrams.items())
        precisions.append((hit + 1) / (sum(pred_ngrams.values()) + 1))
    brevity = min(1.0, math.exp(1 - len(ref) / len(pred)))
    bleu = brevity * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return {"bleu": bleu, "rouge_1": rouge1, "rouge_l": rouge_l}

def _semantic_relevancy(query: str, answer: str, model: SentenceTransformer) -> float:
    vectors = model.encode([query, answer], normalize_embeddings=True, convert_to_numpy=True)
    return float(np.clip(np.dot(vectors[0], vectors[1]), 0.0, 1.0))

def _parse_score(text: str) -> float | None:
    match = re.search(r"(?:score|rating)\s*[:=\-]?\s*([1-5](?:\.\d+)?)", text, re.I)
    if not match:
        match = re.search(r"\b([1-5])\s*/\s*5\b", text)
    if not match:
        return None
    return max(1.0, min(5.0, float(match.group(1))))

def llm_judge_scores(query: str, answer: str, contexts: list[str]) -> dict[str, Any]:
    """Run local Qwen G-Eval-style and Prometheus-rubric assessments; return parsed scores only."""
    generator = get_medical_generator(0.0)
    context = "\n\n".join(contexts)
    geval_prompt = ("Evaluate the answer's factual consistency and faithfulness to the evidence. "
                    "Reason carefully internally, then provide a final score from 1 to 5 and a concise rationale. "
                    "Return exactly 'Score: N/5' followed by 'Rationale: ...'.\n"
                    f"Question: {query}\nEvidence: {context}\nAnswer: {answer}")
    prometheus_prompt = ("You are a strict factual-consistency judge. Rubric: 1=mostly unsupported or contradicted; "
                         "2=major unsupported claims; 3=mixed support; 4=mostly supported with minor issues; 5=all factual claims supported. "
                         "Score only from the evidence. Return 'Score: N/5' and a brief rationale.\n"
                         f"Question: {query}\nEvidence: {context}\nResponse: {answer}")
    geval_raw = generator.generate(geval_prompt, [context], temperature=0.0)
    prometheus_raw = generator.generate(prometheus_prompt, [context], temperature=0.0)
    return {"geval_score": _parse_score(geval_raw), "geval_rationale": geval_raw,
            "prometheus_score": _parse_score(prometheus_raw), "prometheus_rationale": prometheus_raw,
            "judge_models": {"geval": "Qwen/Qwen2.5-3B-Instruct", "prometheus_rubric": "Qwen/Qwen2.5-3B-Instruct"},
            "prometheus_method": "Prometheus-style rubric prompt; score produced by local Qwen, not a Prometheus checkpoint"}

def load_evaluation_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON/JSONL or CSV queries with optional reference_answer/answer fields."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {p}")
    if p.suffix.lower() == ".csv":
        frame = pd.read_csv(p).fillna("")
        return frame.to_dict(orient="records")
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix.lower() != ".jsonl" else [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def evaluate_dataset(rows: list[dict[str, Any]], top_k: int = 5, max_retries: int = 2, temperature: float = 0.1, use_llm_judges: bool = True) -> dict[str, Any]:
    """Run each query through the DAG and aggregate observed latency and quality."""
    semantic_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    results = []
    for index, row in enumerate(rows):
        query = str(row.get("query", row.get("question", ""))).strip()
        reference = str(row.get("reference_answer", row.get("ground_truth", row.get("answer", "")))).strip()
        if not query:
            results.append({"query_id": row.get("query_id", index), "status": "FAILED", "error": "Missing query text", "failed": True})
            continue
        started = time.perf_counter()
        try:
            state = run_dag(query, top_k, max_retries, temperature)
            elapsed = time.perf_counter() - started
            answer = state["final_answer"]
            context = [d["text"] for d in state["retrieved_docs"]]
            row_result: dict[str, Any] = {"query_id": row.get("query_id", index), "query": query, "reference_answer": reference, "answer": answer, "status": state["status"], "correction_count": state["attempt_count"], "latency": {**state["phase_timings"], "initial_generation": state["phase_timings"].get("generation", 0.0), "verification": state["phase_timings"].get("verification", 0.0), "self_correction_loop": state["phase_timings"].get("self_correction_loop", 0.0), "total": elapsed}, "nli_support_ratio": state["verification_result"].get("nli_support_ratio"), "faithfulness": state["verification_result"].get("nli_support_ratio"), "answer_relevancy": _semantic_relevancy(query, answer, semantic_model), "failed": False}
            if reference:
                row_result.update(reference_metrics(reference, answer))
            else:
                row_result.update({"bleu": None, "rouge_1": None, "rouge_l": None})
            if use_llm_judges:
                row_result.update(llm_judge_scores(query, answer, context))
            results.append(row_result)
        except Exception as exc:
            LOG.exception("Evaluation failed for query_id=%s", row.get("query_id", index))
            results.append({"query_id": row.get("query_id", index), "query": query, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "latency": {"total": time.perf_counter() - started}, "failed": True})
    successful = [item for item in results if not item.get("failed")]
    metric_names = ("bleu", "rouge_1", "rouge_l", "faithfulness", "answer_relevancy", "geval_score", "prometheus_score")
    summary = {}
    for name in metric_names:
        values = [float(r[name]) for r in successful if r.get(name) is not None]
        summary[name] = {"mean": statistics.mean(values) if values else None, "count": len(values)}
    latency_names = ("retrieval", "generation", "verification", "self_correction_loop", "total")
    for name in latency_names:
        values = [float(r.get("latency", {}).get(name, 0.0)) for r in successful]
        summary[f"{name}_seconds"] = {"mean": statistics.mean(values) if values else None, "count": len(values)}
    return {"results": results, "summary": summary, "query_count": len(rows), "successful_count": len(successful), "failed_count": len(rows) - len(successful), "llm_judges_enabled": use_llm_judges}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/benchmark_queries.json", help="JSON/JSONL/CSV query set; may include reference_answer")
    parser.add_argument("--output", default="evaluation_report.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--skip-llm-judges", action="store_true", help="Skip the two additional local Qwen judge generations")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        report = evaluate_dataset(load_evaluation_rows(args.queries), args.top_k, args.max_retries, args.temperature, not args.skip_llm_judges)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"summary": report["summary"], "successful_count": report["successful_count"], "failed_count": report["failed_count"], "output": args.output}, indent=2))
        return 1 if report["failed_count"] else 0
    except Exception:
        LOG.exception("Evaluation run failed")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
