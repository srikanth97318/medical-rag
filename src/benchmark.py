import statistics
import time
from collections import defaultdict
from typing import Any

def summarize(values: list[float]) -> dict[str,float | None]:
    if not values: return {k:None for k in ("mean","median","min","max","std","p50","p95","p99")}
    s=sorted(values); interp=lambda p: s[min(len(s)-1, max(0, int(round((len(s)-1)*p))))]
    return {"mean":statistics.mean(s),"median":statistics.median(s),"min":min(s),"max":max(s),"std":statistics.stdev(s) if len(s)>1 else 0.0,"p50":interp(.5),"p95":interp(.95),"p99":interp(.99)}

def run_benchmarks(pipeline: Any, queries: list[dict], warmup: int, runs: int, sync_cuda: bool=True) -> dict:
    modes={"A_baseline":"baseline","B_nli":"nli","C_multi_judge":"full","D_full_self_correct":"full_selfcorrect"}
    rows=[]
    for q in queries:
        for label,mode in modes.items():
            for _ in range(warmup):
                try: pipeline.run(q["query"],mode,query_id=q.get("query_id","warmup"),synchronize=sync_cuda)
                except Exception: pass  # Warmup failures are excluded; measured failures are recorded below.
            for run in range(runs):
                start=time.perf_counter()
                try:
                    r=pipeline.run(q["query"],mode,query_id=q.get("query_id",str(len(rows))),synchronize=sync_cuda)
                    item=r.to_dict(); item.update({"pipeline_mode":label,"category":q.get("category","UNKNOWN"),"run":run+1,"failed":False})
                except Exception as ex:
                    item={"query_id":q.get("query_id"),"status":"FAILED","pipeline_mode":label,"category":q.get("category","UNKNOWN"),"run":run+1,"error":f"{type(ex).__name__}: {ex}","timing":{"total_seconds":time.perf_counter()-start},"failed":True}
                rows.append(item)
    stages=defaultdict(list)
    for row in rows:
        if row.get("failed"): continue
        t=row.get("timing",{})
        for phase,key in (("Retrieval","retrieval_seconds"),("Generation","generation_seconds"),("Multi-Judge Verification","verification_total_seconds"),("Self-Correction","self_correction_total_seconds"),("End-to-End","total_seconds")):
            stages[(row["pipeline_mode"],phase)].append(t.get(key,0.0))
    summary={f"{mode}/{phase}":summarize(vals) for (mode,phase),vals in stages.items()}
    categories=defaultdict(list)
    for row in rows:
        if not row.get("failed"):
            categories[(row.get("category","UNKNOWN"),row["pipeline_mode"])].append(row.get("timing",{}).get("total_seconds",0.0))
    category_summary={f"{category}/{mode}":summarize(vals) for (category,mode),vals in categories.items()}
    elapsed=sum(r.get("timing",{}).get("total_seconds",0) for r in rows if not r.get("failed"))
    count=sum(not r.get("failed") for r in rows)
    return {"rows":rows,"summary":summary,"summary_by_category":category_summary,"throughput_queries_per_second":count/elapsed if elapsed else 0.0,"elapsed_seconds":elapsed,"query_count":count,"failed_count":sum(r.get("failed",False) for r in rows),"verification_overhead_percent":_overhead(summary)}

def _overhead(summary: dict) -> dict:
    def val(k): return summary.get(k,{}).get("mean")
    base=val("A_baseline/End-to-End"); out={}
    for label in ("B_nli","C_multi_judge","D_full_self_correct"):
        x=val(label+"/End-to-End")
        out[label]=((x-base)/base*100) if x is not None and base else None
    return out
