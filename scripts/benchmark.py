#!/usr/bin/env python3
import argparse
import csv
import json
import importlib.metadata
from pathlib import Path
import sys
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_path
from src.factory import create_pipeline
from src.benchmark import run_benchmarks

def main():
    p=argparse.ArgumentParser(description="Benchmark Medical RAG pipeline latency")
    p.add_argument("--queries",default="data/benchmark_queries.json"); p.add_argument("--runs",type=int,default=None); p.add_argument("--warmup",type=int,default=None); p.add_argument("--config",default=None)
    args=p.parse_args(); cfg=load_config(args.config); queries=json.loads(Path(args.queries).read_text(encoding="utf-8"))
    if not isinstance(queries,list) or any("query" not in q for q in queries): p.error("Query JSON must be a list of objects with a query field")
    b=cfg.get("benchmark",{}); warmup=args.warmup if args.warmup is not None else b.get("warmup_queries",3); runs=args.runs if args.runs is not None else b.get("measured_runs",10)
    import random
    import numpy as np
    import torch
    seed=int(b.get("seed",42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    pipeline=create_pipeline(cfg)
    # Time model/index initialization before warmups; keep it separate from steady-state inference.
    pipeline.generator.load(); pipeline.nli.load()
    result=run_benchmarks(pipeline,queries,warmup,runs,b.get("synchronize_cuda",True))
    device=cfg.get("device",{}).get("preferred_device","auto")
    actual_device="cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device)
    versions={}
    for package in ("torch","transformers","sentence-transformers","faiss-cpu","numpy","pandas","scikit-learn"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]="not-installed"
    result["reproducibility"]={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":seed,"device":actual_device,"gpu_name":torch.cuda.get_device_name(0) if actual_device=="cuda" and torch.cuda.is_available() else None,"model_ids":cfg.get("models",{}),"benchmark_config":{"warmup_queries":warmup,"measured_runs":runs,"synchronize_cuda":b.get("synchronize_cuda",True)},"software_versions":versions,"query_categories":{q.get("category","UNKNOWN"):sum(x.get("category")==q.get("category","UNKNOWN") for x in queries) for q in queries}}
    out=resolve_path(b.get("output_directory","results")); out.mkdir(parents=True,exist_ok=True)
    result["cold_start_timings_seconds"]=getattr(pipeline,"initialization_timings",{})
    (out/"benchmark_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    flat=[]
    for row in result["rows"]:
        t=row.get("timing",{}); flat.append({"query_id":row.get("query_id"),"category":row.get("category"),"pipeline_mode":row.get("pipeline_mode"),"status":row.get("status"),"failed":row.get("failed"),"correction_count":row.get("correction_count",0),"error":row.get("error"),**t})
    with (out/"benchmark_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys() if flat else ["query_id"]); w.writeheader(); w.writerows(flat)
    print(json.dumps({"summary":result["summary"],"throughput_queries_per_second":result["throughput_queries_per_second"],"failed_count":result["failed_count"]},indent=2))
    if result["failed_count"]: raise SystemExit("Benchmark completed with failed measured queries; see failed rows in the result files.")
if __name__=="__main__": main()
