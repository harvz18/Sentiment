from abc import ABC, abstractmethod

from app.schemas.sentiment import ProviderResult, ReferenceExample, TopicInfo


class ProviderError(RuntimeError):
    """Safe public provider failure without upstream secrets or request bodies."""


class SentimentProvider(ABC):
    name: str

    @abstractmethod
    def analyze(self, feedback: str, topic_context: TopicInfo,
                retrieved_examples: list[ReferenceExample]) -> ProviderResult:
        """Classify original-language feedback independently of topic discovery."""
