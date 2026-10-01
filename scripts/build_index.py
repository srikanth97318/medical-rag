#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_path
from src.data_loader import load_documents
from src.embeddings import SentenceEmbeddingModel
from src.preprocessing import chunk_documents
from src.retriever import Retriever

def main():
    parser=argparse.ArgumentParser(description="Build persistent FAISS index from local medical datasets")
    parser.add_argument("datasets",nargs="*",help="CSV, TSV, JSON/JSONL, or Excel paths; default scans data/raw")
    parser.add_argument("--config",default=None)
    parser.add_argument("--output",default=None)
    args=parser.parse_args(); cfg=load_config(args.config); dc=cfg.get("retrieval_data",{})
    paths=[Path(x) for x in args.datasets] if args.datasets else [p for p in resolve_path(dc.get("raw_dir","data/raw")).glob("*") if p.suffix.lower() in {".csv",".tsv",".json",".jsonl",".xlsx",".xls"}]
    if not paths: parser.error("No local datasets found. Place files in data/raw or pass dataset paths; no data is downloaded.")
    docs=load_documents(paths,dc.get("mappings")); chunks=chunk_documents(docs,dc.get("chunk_size",1200),dc.get("chunk_overlap",150))
    emb=SentenceEmbeddingModel(cfg["models"]["embedding_model"],cfg.get("device",{}).get("preferred_device","auto"))
    retriever=Retriever.build(chunks,emb,cfg.get("retrieval",{}).get("relevance_threshold",.25))
    out=Path(args.output) if args.output else resolve_path(dc.get("index_dir","indexes/default"))
    retriever.save(out,cfg["models"]["embedding_model"])
    (out/"build_metadata.json").write_text(json.dumps({"datasets":[str(p) for p in paths],"records":len(docs),"chunks":len(chunks),"embedding_model":cfg["models"]["embedding_model"]},indent=2),encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len(docs)} records at {out}")
if __name__=="__main__": main()
