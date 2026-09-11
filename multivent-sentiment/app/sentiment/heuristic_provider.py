from app.schemas.sentiment import (BinaryScores, EvidenceContribution, FallbackDebug,
                                    ProviderResult, ReferenceExample, TopicInfo)
from app.sentiment.base_provider import SentimentProvider


def evidence_breakdown(examples: list[ReferenceExample]) -> FallbackDebug:
    evidence = [EvidenceContribution(id=e.id, sentiment=e.sentiment, similarity=e.similarity,
                                    domain_relevance_score=e.domain_relevance_score, source_weight=e.source_weight,
                                    weighted_contribution=max(0.0, e.similarity) * e.domain_relevance_score * e.source_weight)
                for e in examples]
    return FallbackDebug(positive_score=sum(e.weighted_contribution for e in evidence if e.sentiment == "positive"),
                         negative_score=sum(e.weighted_contribution for e in evidence if e.sentiment == "negative"),
                         evidence=evidence)


class HeuristicSentimentProvider(SentimentProvider):
    """Temporary binary evidence vote; shares are not calibrated probabilities."""
    name = "rag_weighted_fallback"

    def analyze(self, feedback: str, topic_context: TopicInfo,
                retrieved_examples: list[ReferenceExample]) -> ProviderResult:
        debug = evidence_breakdown(retrieved_examples)
        total = debug.positive_score + debug.negative_score
        positive = debug.positive_score / total if total else 0.5
        negative = debug.negative_score / total if total else 0.5
        sentiment = "positive" if positive > negative else "negative"
        confidence = max(positive, negative) if total else 0.0
        detail = ("No usable evidence: equal debug shares and zero confidence; required binary tie-break is negative."
                  if total == 0 else "Equal evidence; required binary tie-break is negative."
                  if positive == negative else f"{sentiment.capitalize()} has the larger weighted evidence total.")
        return ProviderResult(sentiment=sentiment, confidence=confidence,
                              scores=BinaryScores(positive=positive, negative=negative), language="unknown",
                              provider=self.name,
                              reason=f"Temporary local neighbor vote weighted by similarity, domain relevance, and source weight. {detail} Evidence shares are not calibrated probabilities.")
