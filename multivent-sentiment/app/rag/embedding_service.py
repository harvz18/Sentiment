"""Replaceable embedding interface; the local encoder is never fine-tuned."""
from abc import ABC, abstractmethod
from threading import Lock

import numpy as np

from app.config.settings import Settings


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def identity(self) -> dict:
        """Serializable identity used to reject incompatible query encoders."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one float32 vector per text."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """Cached multilingual sentence transformer, inference only, CPU by default.

    Hiligaynon quality is an experimental assumption, not a documented guarantee.
    Long text is chunked at the encoder's token window and mean pooled.
    """
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.settings = settings
        self._lock = Lock()
        local = settings.root / "models/embeddings/encoder"
        identity_path = local / "multivent_identity.json"
        from app.artifacts import read_json, write_json
        expected = {"model": settings.embedding_model, "revision": settings.embedding_revision}
        if identity_path.exists():
            saved = read_json(identity_path)
            if any(saved[key] != value for key, value in expected.items()):
                raise ValueError("Embedding model/revision changed; use a fresh encoder directory and rebuild RAG")
            self.model = SentenceTransformer(str(local), device=settings.embedding_device,
                                             local_files_only=True)
            self._identity = saved
        else:
            self.model = SentenceTransformer(settings.embedding_model,
                                             revision=settings.embedding_revision,
                                             cache_folder=str(settings.root / "models/embeddings/cache"),
                                             device=settings.embedding_device,
                                             local_files_only=settings.embedding_local_files_only,
                                             trust_remote_code=False)
            self.model.save(str(local))
            resolved = getattr(self.model[0].auto_model.config, "_commit_hash", None)
            self._identity = {**expected, "resolved_revision": resolved,
                              "dimension": self.model.get_sentence_embedding_dimension(),
                              "pooling": "token_chunks_mean_v1"}
            write_json(identity_path, self._identity)

    @property
    def identity(self) -> dict:
        return self._identity

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.identity["dimension"]), dtype="float32")
        with self._lock:
            tokenizer = self.model.tokenizer
            window = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
            chunks, owners = [], []
            for owner, text in enumerate(texts):
                # Full tokenization is intentional: chunks below enforce the model window.
                tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
                if len(tokens) <= window:
                    chunks.append(text)
                    owners.append(owner)
                else:
                    for start in range(0, len(tokens), window):
                        chunks.append(tokenizer.decode(tokens[start:start + window], skip_special_tokens=True))
                        owners.append(owner)
            vectors = self.model.encode(chunks, batch_size=self.settings.embedding_batch_size,
                                        normalize_embeddings=True, convert_to_numpy=True,
                                        show_progress_bar=len(chunks) > 100)
            pooled = np.zeros((len(texts), vectors.shape[1]), dtype="float32")
            np.add.at(pooled, owners, vectors)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            return pooled / np.maximum(norms, 1e-12)
