from functools import lru_cache
import time
from .config import load_config, resolve_path
from .embeddings import SentenceEmbeddingModel
from .retriever import Retriever
from .generator import MedicalAnswerGenerator
from .nli_verifier import NLIVerifier
from .qag_verifier import QAGVerifier
from .multi_judge import MultiJudgeVerifier
from .pipeline import MedicalRAGPipeline

class LazyResource:
    def __init__(self, loader, name, timings): self._loader=loader; self._name=name; self._timings=timings; self._resource=None
    def load(self):
        if self._resource is None:
            started=time.perf_counter(); self._resource=self._loader(); self._timings[self._name]=time.perf_counter()-started
        return self._resource
    def __getattr__(self, name): return getattr(self.load(),name)

@lru_cache(maxsize=4)
def load_embedder(model_name: str, device: str): return SentenceEmbeddingModel(model_name,device)
@lru_cache(maxsize=4)
def load_generator(model_name: str, device: str, temperature: float, max_tokens: int, quantized: bool):
    return MedicalAnswerGenerator(model_name,device,temperature,max_tokens,quantized)
@lru_cache(maxsize=4)
def load_nli(model_name: str, device: str, entailment: float, contradiction: float):
    return NLIVerifier(model_name,device,entailment,contradiction)

def create_pipeline(config=None, overrides=None):
    cfg=config or load_config()
    for section, vals in (overrides or {}).items(): cfg.setdefault(section,{}).update(vals)
    models=cfg["models"]; device=cfg.get("device",{}).get("preferred_device","auto")
    startup={}
    started=time.perf_counter(); emb=load_embedder(models["embedding_model"],device); startup["embedding_model_load_seconds"]=time.perf_counter()-started
    retcfg=cfg.get("retrieval",{})
    started=time.perf_counter(); retriever=Retriever.load(resolve_path(cfg.get("retrieval_data",{}).get("index_dir","indexes/default")),emb,retcfg.get("relevance_threshold",.25)); startup["faiss_index_load_seconds"]=time.perf_counter()-started
    gen_cfg=cfg.get("generation",{})
    generator=LazyResource(lambda: load_generator(models["generation_model"],device,gen_cfg.get("temperature",.1),gen_cfg.get("max_new_tokens",256),gen_cfg.get("load_in_4bit",False)),"generation_model_load_seconds",startup)
    vc=cfg.get("verification",{})
    nli=LazyResource(lambda: load_nli(models["nli_model"],device,vc.get("entailment_threshold",.65),vc.get("contradiction_threshold",.65)),"nli_model_load_seconds",startup)
    judges=MultiJudgeVerifier(nli=nli,qag=QAGVerifier())
    pipeline=MedicalRAGPipeline(retriever,generator,judges,cfg,nli=nli)
    pipeline.initialization_timings=startup
    return pipeline
