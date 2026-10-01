from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class PipelineTiming:
    preprocessing_seconds: float = 0.0
    topic_classification_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    evidence_check_seconds: float = 0.0
    generation_seconds: float = 0.0
    claim_extraction_seconds: float = 0.0
    nli_seconds: float = 0.0
    qag_seconds: float = 0.0
    selfcheck_seconds: float = 0.0
    faithfulness_seconds: float = 0.0
    verification_total_seconds: float = 0.0
    correction_generation_seconds: float = 0.0
    correction_verification_seconds: float = 0.0
    self_correction_total_seconds: float = 0.0
    total_seconds: float = 0.0
    def to_dict(self) -> dict[str, float]: return asdict(self)

@dataclass
class ClaimResult:
    claim: str
    entailment: float | None = None
    neutral: float | None = None
    contradiction: float | None = None
    status: str = "NOT_EVALUATED"
    supporting_chunk_ids: list[str] = field(default_factory=list)
    qag_question: str | None = None
    context_answer: str | None = None

@dataclass
class PipelineResult:
    query_id: str
    query: str
    status: str
    answer: str
    evidence_status: str
    evidence_score: float
    claims: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    correction_count: int = 0
    correction_history: list[dict[str, Any]] = field(default_factory=list)
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    error: str | None = None
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timing"] = self.timing.to_dict()
        return d
