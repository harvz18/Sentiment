from app.config.settings import Settings
from app.preprocessing.text_cleaner import prepare_text
from app.preprocessing.text_normalizer import normalize_informal_text
from app.rag.embedding_service import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.schemas.sentiment import ReferenceExample


class RetrievalService:
    """Retrieve from both pools, rerank by relevance, then retain the requested mix."""
    def __init__(self, embeddings: EmbeddingProvider, store: VectorStore, settings: Settings):
        self.embeddings, self.store, self.settings = embeddings, store, settings

    def retrieve(self, feedback: str, top_k: int = 6) -> list[ReferenceExample]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        normalized = normalize_informal_text(feedback).normalized
        text = prepare_text(normalized, self.settings.enable_pii_sanitization)
        vector = self.embeddings.encode([text])[0]
        candidates = {}
        for pool in ("language", "domain"):
            examples = self.store.search(vector, min(100, max(12, top_k * 3)), pool=pool)
            for example in examples:
                example.source_weight = (self.settings.rag_hilisenti_weight if example.source_dataset == "hilisenti"
                                         else self.settings.rag_domain_weight)
                example.rerank_score = max(0.0, example.similarity) * example.domain_relevance_score * example.source_weight
            candidates[pool] = sorted(examples, key=lambda e: (-e.rerank_score, -e.similarity, e.id))
        total_quota = self.settings.rag_language_k + self.settings.rag_domain_k
        if total_quota == 0:
            raise ValueError("RAG_LANGUAGE_K and RAG_DOMAIN_K cannot both be zero")
        local_k = int(top_k * self.settings.rag_language_k / total_quota + 0.5)
        if top_k >= 2 and self.settings.rag_language_k and self.settings.rag_domain_k:
            local_k = max(1, min(top_k - 1, local_k))
        selected = candidates["language"][:local_k] + candidates["domain"][:top_k - local_k]
        ids = {row.id for row in selected}
        remaining = sorted([e for pool in candidates.values() for e in pool if e.id not in ids],
                           key=lambda e: (-e.rerank_score, -e.similarity, e.id))
        selected.extend(remaining[:top_k - len(selected)])
        return sorted(selected, key=lambda e: (-e.rerank_score, -e.similarity, e.id))
