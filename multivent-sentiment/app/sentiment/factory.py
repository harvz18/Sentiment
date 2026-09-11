import logging

from app.config.settings import Settings
from app.sentiment.base_provider import SentimentProvider
from app.sentiment.heuristic_provider import HeuristicSentimentProvider
from app.sentiment.mock_provider import MockSentimentProvider


def create_provider(settings: Settings) -> SentimentProvider:
    if settings.ai_provider == "mock":
        return MockSentimentProvider()
    if settings.ai_provider == "openai":
        if not settings.openai_configured:
            logging.getLogger(__name__).warning("OPENAI_API_KEY is blank; selecting rag_weighted_fallback")
            return HeuristicSentimentProvider()
        from app.sentiment.openai_provider import OpenAISentimentProvider
        return OpenAISentimentProvider(settings)
    return HeuristicSentimentProvider()
