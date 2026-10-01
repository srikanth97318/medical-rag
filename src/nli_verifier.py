from typing import Any

class NLIVerifier:
    def __init__(self, model_name: str, device: str = "auto", entailment_threshold: float = .65, contradiction_threshold: float = .65):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval(); self.entailment_threshold = entailment_threshold; self.contradiction_threshold = contradiction_threshold
        config = self.model.config
        self.labels = {int(k): str(v).lower() for k,v in getattr(config, "id2label", {}).items()}
        if self.labels and not any(any(term in v for term in ("entail","neutral","contrad")) for v in self.labels.values()):
            self.labels = {int(v):str(k).lower() for k,v in getattr(config,"label2id",{}).items()}
        if not any("entail" in v for v in self.labels.values()): raise ValueError(f"NLI model has no entailment label mapping: {self.labels}")
    def verify(self, claim: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        if not passages: return {"claim":claim,"entailment":None,"neutral":None,"contradiction":None,"status":"NOT_COVERED","supporting_chunk_ids":[]}
        pairs = [(p["text"], claim) for p in passages]
        encoded = self.tokenizer([a for a,_ in pairs], [b for _,b in pairs], return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.inference_mode(): probs = torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
        def pick(term: str) -> float:
            ids = [i for i,label in self.labels.items() if term in label]
            return float(max((probs[:,i].max() for i in ids), default=0.0))
        ent, con = pick("entail"), pick("contrad")
        neu = pick("neutral")
        status = "ENTAILED" if ent >= self.entailment_threshold else ("CONTRADICTED" if con >= self.contradiction_threshold else "UNSUPPORTED")
        support = [p["document_id"] for p, row in zip(passages, probs) if row[max((i for i,v in self.labels.items() if "entail" in v), key=int)] >= self.entailment_threshold]
        return {"claim":claim,"entailment":ent,"neutral":neu,"contradiction":con,"status":status,"supporting_chunk_ids":support}
