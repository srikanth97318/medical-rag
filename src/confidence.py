def grounding_confidence(evidence_score: float, faithfulness_score: float | None, nli_ratio: float | None, qag_ratio: float | None) -> dict:
    values = [("evidence", evidence_score, .25)]
    if faithfulness_score is not None: values.append(("faithfulness",faithfulness_score,.35))
    if nli_ratio is not None: values.append(("nli_support",nli_ratio,.25))
    if qag_ratio is not None: values.append(("qag_support",qag_ratio,.15))
    denom = sum(w for _,_,w in values)
    return {"score":sum(v*w for _,v,w in values)/denom if denom else None,"components":{k:v for k,v,_ in values},"formula":"weighted mean of available normalized signals; not clinical correctness probability"}
