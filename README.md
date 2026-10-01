# Medical RAG: Hallucination and Off-Topic Reduction

Research project: **Minimizing Hallucinations and Off-Topic Responses in a Medical Question Answering System Using Retrieval-Augmented Generation (RAG)**.

## Standalone scripts and supplied datasets

The standalone modules at the repository root are the recommended runnable path. The supplied `medquad.csv.zip` was extracted to `data/raw/medquad.csv` and `disease_diagnosis.csv` was copied to `data/raw/disease_diagnosis.csv`. The index builder reads those two files by default, normalizes/merges their question and answer fields (using symptoms, diagnosis, treatment, and severity for the disease dataset), and writes `vector_store/faiss_index.bin` with aligned `vector_store/metadata.pkl`. Raw data and generated indexes are git-ignored.

```bash
pip install -r requirements.txt
python build_knowledge_base.py
streamlit run app.py
python evaluate_system.py --queries data/benchmark_queries.json
```

`rag_core.py`, `verification_engine.py`, and `self_correction_dag.py` provide the importable retrieval, NLI/SelfCheck-style evaluation, and retry workflow. `evaluate_system.py` requires reference answers for BLEU/ROUGE; without references those fields remain null. Faithfulness is the claim-level NLI support ratio. Its Prometheus evaluation is a local Qwen rubric judge, not a separately hosted Prometheus model. SelfCheck is an explicitly labeled sampled-answer consistency approximation.

The project retrieves evidence from datasets you supply locally, generates a grounded response, and checks claims. It is educational/research software. **This system provides information grounded in its supplied knowledge base and is not a substitute for professional medical advice.** Faithfulness means support by retrieved text; it does not establish clinical correctness. No medication dose or treatment should be inferred where supplied evidence is silent.

## Architecture

Local medical exports → flexible schema mapping and normalization → chunking and BGE embeddings → persistent FAISS inner-product index → query guard → retrieval → evidence sufficiency → Qwen generation → atomic claim extraction → NLI / extractive QAG / faithfulness / optional SelfCheck-style consistency → decision → optional correction → answer and measured timings.

The benchmark contrasts two paths:

```text
Baseline RAG:  Retrieval → Generation

Proposed:      Retrieval → Generation → NLI + QAG + Faithfulness + SelfCheck
                                                   ↓
                                                Decision
                                                   ↓
                                      Optional Self-Correction
```

The retrieval and evidence-sufficiency stages are shared safeguards in this implementation; the baseline ablation omits answer verification and correction. SelfCheck sampling is configurable and enabled by default for the proposed modes because it adds multiple generations and measurable overhead.

## Request lifecycle

1. Validate non-empty input and conservatively classify its topic using a lexical heuristic.
2. Retrieve top-K local passages with cosine-equivalent scores (normalized vectors and inner product).
3. Combine retrieval relevance with query-term coverage for evidence sufficiency.
4. Stop with `INSUFFICIENT_INFORMATION` for `NOT_COVERED`; no generation or correction occurs.
5. Generate only from retrieved text, extract claims, and run configured judges. Unsupported claims can be flagged only after evidence exists.
6. In full self-correction mode, provide claim-specific feedback and reverify up to the configured attempt cap.

### Evidence and verification states

| State | Meaning |
|---|---|
| `SUPPORTED` | Passages have sufficient joint relevance and query-term coverage. |
| `PARTIALLY_SUPPORTED` | Some evidence exists, but coverage is below the supported threshold. |
| `NOT_COVERED` | Medical question, insufficient local evidence; answer is insufficient information, not hallucination. |
| `OFF_TOPIC` | Topic heuristic does not recognize a medical question. |
| `INVALID_QUERY` | Empty input. |

`HALLUCINATION` means factual generated content is unsupported or contradicted despite adequate evidence. NLI neutral/low entailment is `UNSUPPORTED`, not automatically a contradiction. QAG reports `NOT_COVERED` if it cannot find enough text evidence. These judges are research signals, not clinical validators.

## Structure

```text
app.py, README.md, requirements.txt, config/config.yaml
data/raw/ data/processed/ data/benchmark_queries.json
indexes/ results/ scripts/{build_index,benchmark}.py
src/ (schemas, ingestion, preprocessing, embedding/retrieval, guards,
      generation, claim verification, confidence, correction, timings, pipeline)
tests/
```

## Dependencies and execution

Python 3.10+. Transformers and sentence-transformers download configured open models on first use; core inference does not use a paid API. The application needs no data source to be downloaded automatically.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the root-level standalone path, the supplied CSVs are already staged in `data/raw/` and the builder uses them by default. It requires both named datasets and saves its artifacts in `vector_store/`.

```bash
python build_knowledge_base.py
streamlit run app.py
python evaluate_system.py --queries data/benchmark_queries.json
```

The Streamlit app loads the persistent index and never rebuilds it at startup. Missing data/indexes yield explicit errors. The earlier `src/` package workflow remains available separately through `scripts/build_index.py` and `scripts/benchmark.py`, and stores its index under `indexes/default/`.

## Configuration

`config/config.yaml` controls embedding/generation/NLI IDs, top-K and score threshold, evidence and verification thresholds, correction cap, QAG/SelfCheck toggles, chunk size, benchmark warmups/runs, CUDA sync and directories. Device `auto` uses CUDA if available, otherwise CPU. Optional 4-bit model loading requires a compatible quantization runtime. FAISS CPU is in requirements; GPU FAISS can be installed separately if compatible.

NLI label probabilities are taken from the configured model's actual logits and label map. The model loader refuses a label configuration with no entailment class. QAG here is an extractive claim-term evidence proxy, not an LLM judge. `SelfCheckVerifier` is SelfCheck-inspired lexical consistency, not the official SelfCheckGPT implementation; it uses sampled model responses when enabled. G-Eval and Prometheus return no fabricated metrics and are not enabled.

## Performance Benchmarking

`scripts/benchmark.py` writes per-query runs, errors, category labels, correction counts, timings, and observed summaries to JSON and CSV. Modes are baseline RAG, RAG+NLI, multi-judge and multi-judge+correction. Warmup requests are excluded. Timings use `time.perf_counter`; optional CUDA synchronization brackets measured model work. Cold startup durations for embedding model load, generation model load, NLI model load and FAISS index load are recorded separately in the benchmark JSON. A first-ever model download may be included in its model-load duration; record that fact and do not compare that cold-start value to warm inference latency.

Hardware, accelerator, precision, package versions, model IDs and benchmark settings vary by host; record them with published results. Throughput is successful completed queries divided by total observed per-query end-to-end durations (not concurrent service throughput). Add category labels to query objects for category slices. The bundled query JSON demonstrates input structure only and is not gold-labeled research data.

Overhead uses observed mean latencies: `(verified latency - baseline latency) / baseline latency × 100`. Raw per-query timings include retrieval, generation, verification and correction. No synthetic results are included. Result data supports report tables for latency/ablation; reliability confusion matrices and quality metrics require labeled references, which this implementation does not invent.

## Limitations

The lexical topic guard can miss paraphrases or accept unrelated text with medical keywords. Evidence sufficiency combines transparent lexical query coverage and embedding score, not trained answerability. Extractive QAG is a proxy. The included benchmark JSON is example structure, not scientific gold data. Use clinically reviewed, properly licensed corpora and never use this application for diagnosis or treatment decisions.

## Report confidentiality wording

Do not claim authority approval for open-source release unless formal authorization exists. If authorization has not been established, suitable wording is:

> Repository and Confidentiality Statement
>
> The repository contains only project material intended for academic demonstration. Any organizationally confidential, proprietary, or restricted information has been excluded. Public dissemination remains subject to applicable institutional and project policies.

Use an approval statement only if genuinely authorized. Remove generic sample-report NDA instructions from the final report unless they genuinely apply.

## Reproducibility

Index metadata records embedding model, vector dimension and count. Benchmark rows preserve IDs, categories, modes, run count, timings and failures. Config holds model IDs and seed. Exact deterministic behavior is not guaranteed across devices, kernels or library versions. Record host hardware, accelerator, precision, package versions, dataset provenance and licensing when reporting experiments.
