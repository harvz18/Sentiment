from abc import ABC, abstractmethod

from app.schemas.sentiment import (FeedbackSummaryRequest, FeedbackSummaryResponse,
                                   ProviderResult, ReferenceExample, TopicInfo)


class ProviderError(RuntimeError):
    """Safe public provider failure without upstream secrets or request bodies."""


class SentimentProvider(ABC):
    name: str

    @abstractmethod
    def analyze(self, feedback: str, topic_context: TopicInfo,
                retrieved_examples: list[ReferenceExample]) -> ProviderResult:
        """Classify original-language feedback independently of topic discovery."""

    def summarize(self, request: FeedbackSummaryRequest) -> FeedbackSummaryResponse:
        """Create a grounded aggregate summary when supported by the provider."""
        raise ProviderError("AI feedback summaries require AI_PROVIDER=openai")
