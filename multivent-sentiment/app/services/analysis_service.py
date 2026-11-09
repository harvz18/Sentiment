"""Pipeline orchestration and optional local research logs."""
import json
import logging
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter

from app.config.settings import Settings
from app.preprocessing.text_cleaner import prepare_text
from app.preprocessing.text_normalizer import normalize_informal_text
from app.rag.retrieval_service import RetrievalService
from app.schemas.sentiment import AnalysisResponse, FeedbackRequest, RetrievalResult, Sentiment
from app.sentiment.base_provider import SentimentProvider
from app.sentiment.heuristic_provider import evidence_breakdown
from app.topics.lda_service import LDAService

log = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, settings: Settings, lda: LDAService, retrieval: RetrievalService,
                 provider: SentimentProvider):
        self.settings, self.lda, self.retrieval, self.provider = settings, lda, retrieval, provider
        self._log_lock = Lock()

    def analyze(self, feedback: str, ground_truth: Sentiment | None = None, debug: bool = False) -> AnalysisResponse:
        start = perf_counter()
        original = FeedbackRequest(feedback=feedback).feedback
        normalization = normalize_informal_text(original)
        cleaned = prepare_text(normalization.normalized, self.settings.enable_pii_sanitization)
        topic = self.lda.infer(cleaned)
        references = self.retrieval.retrieve(cleaned, self.settings.rag_top_k)
        # Original language preserved. Optional PII mode masks before provider transmission.
        provider_text = cleaned if self.settings.enable_pii_sanitization else original
        verdict = self.provider.analyze(provider_text, topic, references)
        response = AnalysisResponse(feedback=original, text_original=original,
                                    text_normalized=normalization.normalized,
                                    normalizations=[change.__dict__ for change in normalization.changes],
                                    **verdict.model_dump(), topic=topic,
                                    rag=RetrievalResult(retrieved_examples=references),
                                    debug=evidence_breakdown(references) if debug and verdict.provider == "rag_weighted_fallback" else None)
        if self.settings.enable_analysis_logging:
            event = {"timestamp": datetime.now(timezone.utc).isoformat(),
                     "feedback": provider_text, "text_normalized": cleaned,
                     "normalizations": [change.__dict__ for change in normalization.changes],
                     "topic": topic.model_dump(),
                     "retrieved_example_ids": [e.id for e in references], "provider": verdict.provider,
                     "predicted_sentiment": verdict.sentiment, "confidence": verdict.confidence,
                     "scores": verdict.scores.model_dump(),
                     "ground_truth": ground_truth, "latency_ms": (perf_counter() - start) * 1000}
            try:
                path = self.settings.root / "logs/analysis.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_lock, path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError:
                log.exception("Analysis succeeded but local experiment logging failed")
        return response
