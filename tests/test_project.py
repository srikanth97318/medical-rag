import json
import pytest
from src.claim_extractor import extract_claims
from src.data_loader import dataframe_to_documents
from src.evidence_checker import EvidenceSufficiencyChecker
from src.evaluator import faithfulness
from src.qag_verifier import QAGVerifier
from src.topic_guard import TopicGuard
from src.pipeline import MedicalRAGPipeline
from src.benchmark import summarize, run_benchmarks
import pandas as pd

DOCS=[{"document_id":"d1","text":"Condition X commonly causes fever and headache.","source":"local","score":.9,"metadata":{}}]

class FakeRetriever:
    def __init__(self, docs=DOCS): self.docs=docs
    def search(self, query, top_k): return self.docs[:top_k]
class FakeGenerator:
    def __init__(self, answers=None): self.answers=list(answers or ["Condition X causes fever and headache."])
    def generate(self, query, docs, feedback=None): return self.answers.pop(0) if len(self.answers)>1 else self.answers[0]
class FakeVerifier:
    def __init__(self, statuses=None): self.statuses=list(statuses or ["GROUNDED"])
    def verify(self, claims, docs, evidence_status, run_nli=True, run_qag=True, run_faithfulness=True):
        status=self.statuses.pop(0) if len(self.statuses)>1 else self.statuses[0]
        cs=[{"claim":c,"status":"UNSUPPORTED" if status=="HALLUCINATION" else "ENTAILED"} for c in claims]
        return {"decision":status,"claims":cs,"nli":[{"status":"ENTAILED"} for _ in cs] if run_nli else {"status":"NOT_EVALUATED"},"qag":[{"support_result":"SUPPORTED"} for _ in cs] if run_qag else {"status":"NOT_EVALUATED"},"faithfulness":faithfulness(cs),"timings":{"nli_seconds":0.0,"qag_seconds":0.0}}
class FakeNLI:
    def verify(self, claim, docs): return {"claim":claim,"status":"ENTAILED","entailment":.9,"neutral":.05,"contradiction":.05,"supporting_chunk_ids":["d1"]}

def pipeline(docs=DOCS, answers=None, outcomes=None, correction=2):
    cfg={"retrieval":{"top_k":5},"evidence":{"supported_threshold":.35,"partial_threshold":.15},"correction":{"max_attempts":correction},"qag":{"enabled":True},"selfcheck":{"enabled":False}}
    return MedicalRAGPipeline(FakeRetriever(docs),FakeGenerator(answers),FakeVerifier(outcomes),cfg,nli=FakeNLI())

def test_empty_question_invalid(): assert pipeline().run(" ").status=="INVALID_QUERY"
def test_medical_question_not_offtopic(): assert TopicGuard().classify("What causes fever in patients?")=="MEDICAL"
def test_unrelated_question_offtopic(): assert TopicGuard().classify("How does football scoring work?")=="OFF_TOPIC"
def test_no_docs_is_not_covered(): assert EvidenceSufficiencyChecker().check("medical fever question",[])["status"]=="NOT_COVERED"
def test_low_score_is_not_covered():
    d=[dict(DOCS[0],score=.01,text="unrelated words only")]
    assert EvidenceSufficiencyChecker(.8,.6).check("medical question about fever",d)["status"]=="NOT_COVERED"
def test_partial_evidence_status():
    d=[dict(DOCS[0],score=.4,text="fever")]
    assert EvidenceSufficiencyChecker(.9,.2).check("medical question about fever headache",d)["status"]=="PARTIALLY_SUPPORTED"
def test_supported_evidence_status(): assert EvidenceSufficiencyChecker(.2,.1).check("fever headache",DOCS)["status"]=="SUPPORTED"
def test_neutral_nli_is_unsupported_not_contradicted():
    class Neutral:
        def verify(self,claim,docs): return {"claim":claim,"status":"UNSUPPORTED","entailment":.04,"neutral":.91,"contradiction":.05}
    from src.multi_judge import MultiJudgeVerifier
    out=MultiJudgeVerifier(Neutral()).verify(["claim statement"],DOCS,"SUPPORTED",run_qag=False)
    assert out["decision"]=="HALLUCINATION" and out["claims"][0]["status"]=="UNSUPPORTED"
def test_contradicted_nli_is_distinct():
    class Contradicted:
        def verify(self,claim,docs): return {"claim":claim,"status":"CONTRADICTED","entailment":.02,"neutral":.03,"contradiction":.95}
    from src.multi_judge import MultiJudgeVerifier
    out=MultiJudgeVerifier(Contradicted()).verify(["claim statement"],DOCS,"SUPPORTED",run_qag=False)
    assert out["claims"][0]["status"]=="CONTRADICTED"
def test_qag_returns_not_covered(): assert QAGVerifier().verify("Unrelated molecular phrase",DOCS)["support_result"]=="NOT_COVERED"
def test_qag_detects_explicit_negation_conflict():
    d=[dict(DOCS[0],text="Condition X causes fever and headache.")]
    assert QAGVerifier().verify("Condition X does not cause fever or headache.",d)["support_result"]=="CONTRADICTED"
def test_claim_extraction_filters_disclaimer():
    assert all("Disclaimer" not in c for c in extract_claims("Disclaimer: educational only. Fever is a symptom of this condition."))
def test_faithfulness_excludes_not_covered_claims():
    r=faithfulness([{"status":"ENTAILED"},{"status":"NOT_COVERED"}]); assert r["score"]==1 and r["not_covered_claims"]==1
def test_unsupported_generated_claim_is_reported():
    r=FakeVerifier(["HALLUCINATION"]).verify(["Claim"],DOCS,"SUPPORTED"); assert r["decision"]=="HALLUCINATION"
def test_successful_self_correction():
    p=pipeline(answers=["Condition X causes fever and headache.","Condition X causes fever and headache."],outcomes=["HALLUCINATION","GROUNDED"])
    r=p.run("What symptoms occur with condition X?","full_selfcorrect"); assert r.status=="CORRECTED" and r.correction_count==1
def test_failed_self_correction_caps_attempts():
    p=pipeline(answers=["Condition X causes fever and headache."],outcomes=["HALLUCINATION"],correction=1)
    r=p.run("What symptoms occur with condition X?","full_selfcorrect"); assert r.status=="VERIFICATION_FAILED" and r.correction_count==1
def test_not_covered_avoids_generation_and_correction():
    p=pipeline(docs=[])
    r=p.run("What treatment is used for a medical condition?","full_selfcorrect"); assert r.status=="INSUFFICIENT_INFORMATION" and r.correction_count==0
def test_maximum_correction_attempts_respected():
    p=pipeline(answers=["Condition X causes fever and headache."],outcomes=["HALLUCINATION"],correction=2)
    r=p.run("What symptoms occur with condition X?","full_selfcorrect"); assert r.correction_count==2
def test_missing_index_is_explicit(tmp_path):
    from src.retriever import Retriever
    with pytest.raises(FileNotFoundError): Retriever.load(tmp_path,object())
def test_missing_dataset_is_explicit(tmp_path):
    from src.data_loader import load_table
    with pytest.raises(FileNotFoundError): load_table(tmp_path/"missing.csv")
def test_cpu_fake_pipeline_executes(): assert pipeline().run("What symptoms occur with condition X?").status in {"GROUNDED","VERIFICATION_FAILED"}
def test_timings_are_recorded():
    r=pipeline().run("What symptoms occur with condition X?"); assert r.timing.total_seconds>0 and r.timing.retrieval_seconds>=0
def test_baseline_stops_after_retrieval_and_generation():
    r=pipeline().run("What symptoms occur with condition X?","baseline")
    assert r.status=="BASELINE_UNVERIFIED" and r.claims==[] and r.verification["decision"]=="NOT_EVALUATED"
def test_proposed_mode_runs_configured_selfcheck_samples():
    class CountingGenerator(FakeGenerator):
        def __init__(self): super().__init__(); self.calls=0
        def generate(self,*args,**kwargs): self.calls+=1; return super().generate(*args,**kwargs)
    p=pipeline(); p.generator=CountingGenerator(); p.config["selfcheck"]={"enabled":True,"samples":2}
    r=p.run("What symptoms occur with condition X?","full")
    assert p.generator.calls==3 and r.verification["selfcheck"]["status"]=="EVALUATED" and r.timing.selfcheck_seconds>0
def test_benchmark_summary_observed_statistics():
    s=summarize([1.0,3.0]); assert s["mean"]==2 and s["p50"] in (1.0,3.0) and s["p95"] in (1.0,3.0)
def test_failed_benchmark_query_is_recorded_as_failed():
    class Broken:
        def run(self,*args,**kwargs): raise RuntimeError("model unavailable")
    result=run_benchmarks(Broken(),[{"query_id":"q1","query":"a medical question","category":"SUPPORTED"}],0,1,False)
    assert result["failed_count"]==4 and all(x["failed"] for x in result["rows"])
def test_dataset_adapter_flexible_columns_and_metadata():
    docs=dataframe_to_documents(pd.DataFrame({"Q":["What?"],"Diagnosis":["Example"],"severity":["mild"]}),{"question":"Q","answer":"Diagnosis"})
    assert "Question: What?" in docs[0]["text"] and "Severity: mild" in docs[0]["text"]
