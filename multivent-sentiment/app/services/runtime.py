"""Load components independently so missing artifacts yield actionable API errors."""
import logging

from app.artifacts import file_hash
from app.corpus.artifacts import verify_prepared

from app.config.settings import Settings
from app.rag.embedding_service import LocalEmbeddingProvider
from app.rag.retrieval_service import RetrievalService
from app.rag.vector_store import VectorStore
from app.schemas.sentiment import HealthResponse
from app.sentiment.factory import create_provider
from app.services.analysis_service import AnalysisService
from app.topics.lda_service import LDAService


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lda = self.retrieval = self.provider = self.analysis = None
        self.errors: dict[str, str] = {}

    def load(self) -> None:
        log = logging.getLogger(__name__)
        for component, loader, instruction in (
            ("lda", self._load_lda,
             "Run python scripts/train_lda.py and restart the server."),
            ("rag", self._load_rag, "Run python scripts/build_rag_index.py and restart the server."),
            ("provider", lambda: create_provider(self.settings), "Check AI_PROVIDER and OPENAI_MODEL in .env."),
        ):
            try:
                value = loader()
                setattr(self, "retrieval" if component == "rag" else component, value)
            except Exception as exc:
                # This boundary keeps diagnostics available, while dependent routes fail closed.
                self.errors[component] = f"{component} unavailable ({type(exc).__name__}). {instruction}"
                log.error(self.errors[component])
        if self.lda and self.retrieval and self.provider:
            self.analysis = AnalysisService(self.settings, self.lda, self.retrieval, self.provider)

    def _load_lda(self) -> LDAService:
        verify_prepared(self.settings)
        service = LDAService.load(self.settings.lda_dir, self.settings.enable_pii_sanitization)
        if service.metadata.get("corpus_manifest_sha256") != file_hash(self.settings.corpus_dir / "manifest.json"):
            raise ValueError("LDA was fitted on a different corpus snapshot; retrain")
        return service

    def _load_rag(self) -> RetrievalService:
        if not (self.settings.vector_dir / "manifest.json").exists():
            raise FileNotFoundError("RAG manifest missing")
        verify_prepared(self.settings)
        encoder = LocalEmbeddingProvider(self.settings)
        store = VectorStore.load(self.settings.vector_dir, encoder.identity,
                                 self.settings.enable_pii_sanitization)
        if store.manifest.get("corpus_manifest_sha256") != file_hash(self.settings.corpus_dir / "manifest.json"):
            raise ValueError("RAG was built from a different corpus snapshot; rebuild")
        return RetrievalService(encoder, store, self.settings)

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok" if self.analysis else "degraded",
                              lda_loaded=self.lda is not None, rag_loaded=self.retrieval is not None,
                              ai_provider=self.provider.name if self.provider else "unavailable",
                              openai_configured=self.settings.openai_configured, errors=self.errors)
