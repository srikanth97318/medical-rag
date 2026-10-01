import time
from .evaluator import faithfulness

class MultiJudgeVerifier:
    def __init__(self, nli=None, qag=None): self.nli, self.qag = nli, qag
    def verify(self, claims: list[str], passages: list[dict], evidence_status: str, run_nli: bool=True, run_qag: bool=True, run_faithfulness: bool=True) -> dict:
        if evidence_status == "NOT_COVERED": return {"nli":[],"qag":[],"faithfulness":faithfulness([]),"decision":"NOT_COVERED","confidence":None}
        n0=time.perf_counter()
        nli_results = [self.nli.verify(c, passages) for c in claims] if run_nli and self.nli else []
        nli_seconds=time.perf_counter()-n0
        q0=time.perf_counter()
        qag_results = [self.qag.verify(c, passages) for c in claims] if run_qag and self.qag else []
        qag_seconds=time.perf_counter()-q0
        for i, c in enumerate(claims):
            nr = next((x for x in nli_results if x["claim"] == c), None)
            qr = qag_results[i] if i < len(qag_results) else None
            if nr: c_result = {"claim":c,"status":nr["status"]}
            elif qr: c_result = {"claim":c,"status":qr["support_result"]}
            else: c_result = {"claim":c,"status":"NOT_EVALUATED"}
            if qr and nr:
                c_result["status"] = "CONTRADICTED" if "CONTRADICTED" in {nr["status"],qr["support_result"]} else ("ENTAILED" if nr["status"] == "ENTAILED" and qr["support_result"] in {"SUPPORTED","NOT_COVERED"} else nr["status"])
            if i < len(nli_results): c_result.update({k:nli_results[i].get(k) for k in ("entailment","neutral","contradiction","supporting_chunk_ids")})
            if qr: c_result["qag"] = qr
            if i == 0: combined=[]
            combined.append(c_result)
        f0=time.perf_counter(); stats = faithfulness(combined if claims else []) if run_faithfulness else {"score":None,"supported_claims":0,"unsupported_claims":0,"contradicted_claims":0,"not_covered_claims":0,"evaluated_claims":0}; faithfulness_seconds=time.perf_counter()-f0 if run_faithfulness else 0.0
        bad = [c for c in combined if c["status"] in {"UNSUPPORTED","CONTRADICTED"}]
        inconclusive = [c for c in combined if c["status"] in {"NOT_COVERED","NOT_EVALUATED"}]
        decision = "HALLUCINATION" if bad else ("INCONCLUSIVE" if inconclusive or not (nli_results or qag_results) else "GROUNDED")
        return {"nli":nli_results or {"status":"NOT_EVALUATED"},"qag":qag_results or {"status":"NOT_EVALUATED"},"claims":combined if claims else [],"faithfulness":stats,"decision":decision,"timings":{"nli_seconds":nli_seconds,"qag_seconds":qag_seconds,"faithfulness_seconds":faithfulness_seconds}}
