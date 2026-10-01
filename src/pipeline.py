import time
from typing import Any
from .schemas import PipelineResult, PipelineTiming
from .topic_guard import TopicGuard
from .evidence_checker import EvidenceSufficiencyChecker
from .claim_extractor import extract_claims
from .confidence import grounding_confidence
from .corrector import correction_feedback
from .logging_utils import log_result
from .config import resolve_path

DISCLAIMER = "This system provides information grounded in its supplied knowledge base and is not a substitute for professional medical advice."

class MedicalRAGPipeline:
    def __init__(self, retriever: Any, generator: Any, verifier: Any, config: dict, nli: Any=None, topic_guard=None):
        self.retriever, self.generator, self.verifier, self.nli = retriever, generator, verifier, nli
        self.config=config; self.topic_guard=topic_guard or TopicGuard()
        ec=config.get("evidence",{}); self.evidence=EvidenceSufficiencyChecker(ec.get("supported_threshold",.55),ec.get("partial_threshold",.3))
    def _record(self, result: PipelineResult) -> PipelineResult:
        log_result(result.to_dict(),resolve_path("results/pipeline_events.jsonl"))
        return result
    def run(self, query: str, mode: str="full", query_id: str="", synchronize: bool=False) -> PipelineResult:
        t0=time.perf_counter(); timing=PipelineTiming(); query=(query or "").strip()
        def sync():
            if synchronize:
                try:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                except ImportError: pass
        qstart=time.perf_counter()
        if not query:
            timing.preprocessing_seconds=time.perf_counter()-qstart; timing.total_seconds=time.perf_counter()-t0
            return self._record(PipelineResult(query_id,query,"INVALID_QUERY","Please enter a question.","NOT_COVERED",0,timing=timing))
        timing.preprocessing_seconds=time.perf_counter()-qstart
        s=time.perf_counter(); topic=self.topic_guard.classify(query); timing.topic_classification_seconds=time.perf_counter()-s
        if topic == "OFF_TOPIC":
            timing.total_seconds=time.perf_counter()-t0
            return self._record(PipelineResult(query_id,query,"OFF_TOPIC","Please ask a medical information question.","NOT_COVERED",0,timing=timing))
        s=time.perf_counter(); docs=self.retriever.search(query,self.config.get("retrieval",{}).get("top_k",5)); timing.retrieval_seconds=time.perf_counter()-s
        s=time.perf_counter(); evidence=self.evidence.check(query,docs); timing.evidence_check_seconds=time.perf_counter()-s
        if evidence["status"] == "NOT_COVERED":
            timing.total_seconds=time.perf_counter()-t0
            return self._record(PipelineResult(query_id,query,"INSUFFICIENT_INFORMATION","Insufficient information in the retrieved medical knowledge base to answer this question reliably.",evidence["status"],evidence["score"],retrieved_documents=docs,timing=timing))
        def generate(feedback=None):
            st=time.perf_counter(); answer=self.generator.generate(query,docs,feedback); return answer,time.perf_counter()-st
        sync(); answer, timing.generation_seconds=generate(); sync()
        corrections=[]; max_attempts=self.config.get("correction",{}).get("max_attempts",2)
        do_nli=mode in {"nli","full","full_selfcorrect"}; do_qag=mode in {"full","full_selfcorrect"} and self.config.get("qag",{}).get("enabled",True)
        def verify(ans):
            c0=time.perf_counter(); claims=extract_claims(ans); claim_t=time.perf_counter()-c0
            v0=time.perf_counter(); vv=self.verifier.verify(claims,docs,evidence["status"],run_nli=do_nli,run_qag=do_qag,run_faithfulness=mode in {"full","full_selfcorrect"})
            selfcheck_seconds=0.0
            if mode in {"full","full_selfcorrect"} and self.config.get("selfcheck",{}).get("enabled",False):
                sh=time.perf_counter(); samples=[]; count=int(self.config.get("selfcheck",{}).get("samples",3))
                for _ in range(count):
                    sampled,_=generate(); samples.append(sampled)
                from .selfcheck import SelfCheckVerifier
                vv["selfcheck"]=SelfCheckVerifier(count).verify(ans,samples)
                selfcheck_seconds=time.perf_counter()-sh
            else:
                vv["selfcheck"]={"status":"NOT_EVALUATED","consistency":None,"method":"Disabled or excluded by benchmark mode"}
            vtotal=time.perf_counter()-v0
            stages=vv.get("timings",{})
            return claims,vv,claim_t,stages.get("nli_seconds",0.0),stages.get("qag_seconds",0.0),stages.get("faithfulness_seconds",0.0),selfcheck_seconds,vtotal
        if mode == "baseline":
            claims=[]
            ver={"decision":"NOT_EVALUATED","claims":[],"nli":{"status":"NOT_EVALUATED"},"qag":{"status":"NOT_EVALUATED"},"faithfulness":{"score":None,"supported_claims":0,"unsupported_claims":0,"contradicted_claims":0,"not_covered_claims":0,"evaluated_claims":0}}
        else:
            claims,ver,timing.claim_extraction_seconds,timing.nli_seconds,timing.qag_seconds,timing.faithfulness_seconds,timing.selfcheck_seconds,timing.verification_total_seconds=verify(answer)
        if mode == "baseline": ver["selfcheck"]={"status":"NOT_EVALUATED","consistency":None,"method":"Baseline excludes verification"}
        if mode == "nli": ver["decision"]="HALLUCINATION" if any(c.get("status") in {"UNSUPPORTED","CONTRADICTED"} for c in ver.get("claims",[])) else ("GROUNDED" if claims else "INCONCLUSIVE")
        if mode == "full_selfcorrect" and ver["decision"] == "HALLUCINATION" and evidence["status"] != "NOT_COVERED":
            loop=time.perf_counter()
            for attempt in range(max_attempts):
                feedback=correction_feedback(ver.get("claims",[]),docs)
                new_answer, gen_t=generate(feedback); timing.correction_generation_seconds += gen_t
                claims,ver,claim_t,nli_t,qag_t,faith_t,selfcheck_t,verify_t=verify(new_answer); timing.correction_verification_seconds += verify_t
                timing.claim_extraction_seconds += claim_t; timing.nli_seconds += nli_t; timing.qag_seconds += qag_t; timing.faithfulness_seconds += faith_t; timing.selfcheck_seconds += selfcheck_t
                corrections.append({"attempt":attempt+1,"feedback":feedback,"answer":new_answer,"decision":ver["decision"]})
                answer=new_answer
                if ver["decision"] != "HALLUCINATION": break
            timing.self_correction_total_seconds=time.perf_counter()-loop
        # Reuse computed claims: avoid loading a separate judge when a mode disables it.
        claim_results=ver.get("claims",[]); f=ver.get("faithfulness",{})
        nli_items=ver.get("nli",[]); nli_items=nli_items if isinstance(nli_items,list) else []
        nli_ratio=(sum(x.get("status")=="ENTAILED" for x in nli_items)/len(nli_items)) if nli_items else None
        qag_items=ver.get("qag",[]); qag_items=qag_items if isinstance(qag_items,list) else []
        qag_ratio=(sum(x.get("support_result")=="SUPPORTED" for x in qag_items)/len(qag_items)) if qag_items else None
        confidence=grounding_confidence(evidence["score"],f.get("score"),nli_ratio,qag_ratio)
        ver["grounding_confidence"]=confidence
        bad=ver.get("decision")=="HALLUCINATION"
        status="BASELINE_UNVERIFIED" if mode=="baseline" else ("CORRECTED" if corrections and not bad and ver.get("decision")=="GROUNDED" else ("VERIFICATION_FAILED" if ver.get("decision") != "GROUNDED" else "GROUNDED"))
        answer=answer.rstrip()+"\n\n"+DISCLAIMER
        sync()
        timing.total_seconds=time.perf_counter()-t0
        return self._record(PipelineResult(query_id,query,status,answer,evidence["status"],evidence["score"],claims=claim_results,verification=ver,retrieved_documents=docs,correction_count=len(corrections),correction_history=corrections,timing=timing))
