from app.schemas.sentiment import BinaryScores, ProviderResult, ReferenceExample, TopicInfo
from app.sentiment.base_provider import SentimentProvider


class MockSentimentProvider(SentimentProvider):
    """Deterministic simulated response for wiring checks only."""
    name = "mock"

    def analyze(self, feedback: str, topic_context: TopicInfo,
                retrieved_examples: list[ReferenceExample]) -> ProviderResult:
        return ProviderResult(sentiment="negative", confidence=0, language="unknown",
                              scores=BinaryScores(positive=0.5, negative=0.5),
                              reason="SIMULATED result for interface testing; no sentiment inference was performed.",
                              provider=self.name)
