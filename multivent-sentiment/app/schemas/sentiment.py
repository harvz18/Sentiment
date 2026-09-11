"""Public API and provider contracts."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.preprocessing.text_cleaner import normalize_text

Sentiment = Literal["negative", "positive"]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feedback: str = Field(min_length=1, max_length=5000, strict=True)

    @field_validator("feedback")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if not normalize_text(value) or not any(c.isalnum() for c in value):
            raise ValueError("Feedback must contain letters or numbers")
        return value


class RetrievalRequest(FeedbackRequest):
    top_k: int | None = Field(None, ge=1, le=100, strict=True)


class TopicInfo(BaseModel):
    id: int | None
    name: str
    probability: float = Field(ge=0, le=1)
    keywords: list[str]
    in_vocabulary: bool = True


class ReferenceExample(BaseModel):
    id: str
    text: str
    sentiment: Sentiment
    similarity: float = Field(ge=-1, le=1)
    split: Literal["train", "validation"]
    source_dataset: str
    source_name: str
    source_record_id: str
    source_domain: str
    source_url: str
    license: str
    language: str
    aspect: str
    pool: Literal["language", "domain"]
    domain_relevance_score: float = Field(ge=0, le=1)
    source_weight: float = Field(default=1, ge=0)
    rerank_score: float = Field(default=0, ge=0)
    label_basis: str = "source label"


class RetrievalResult(BaseModel):
    retrieved_examples: list[ReferenceExample]


class BinaryScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def sum_to_one(self):
        if abs(self.positive + self.negative - 1) > 1e-6:
            raise ValueError("Binary evidence shares must sum to one")
        return self


class EvidenceContribution(BaseModel):
    id: str
    sentiment: Sentiment
    similarity: float
    domain_relevance_score: float
    source_weight: float
    weighted_contribution: float


class FallbackDebug(BaseModel):
    formula: str = "max(0, similarity) * domain_relevance_score * source_weight"
    positive_score: float
    negative_score: float
    evidence: list[EvidenceContribution]
    tie_break: str = "negative when positive evidence does not exceed negative evidence"


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
    scores: BinaryScores
    language: str
    reason: str
    provider: str


class AnalysisResponse(ProviderResult):
    feedback: str
    topic: TopicInfo
    rag: RetrievalResult
    debug: FallbackDebug | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    lda_loaded: bool
    rag_loaded: bool
    ai_provider: str
    openai_configured: bool
    errors: dict[str, str] = Field(default_factory=dict)
