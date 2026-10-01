def faithfulness(claim_results: list[dict]) -> dict:
    supported = [c for c in claim_results if c.get("status") in {"ENTAILED", "SUPPORTED"}]
    evaluable = [c for c in claim_results if c.get("status") in {"ENTAILED", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}]
    return {"score":len(supported)/len(evaluable) if evaluable else None,
            "supported_claims":len(supported), "unsupported_claims":sum(c.get("status")=="UNSUPPORTED" for c in claim_results),
            "contradicted_claims":sum(c.get("status")=="CONTRADICTED" for c in claim_results),
            "not_covered_claims":sum(c.get("status")=="NOT_COVERED" for c in claim_results), "evaluated_claims":len(evaluable)}
