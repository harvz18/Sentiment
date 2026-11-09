"""Validated configuration; all artifact paths are relative to the project."""
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    root: Path = ROOT
    app_env: str = "development"
    ai_provider: Literal["rag_fallback", "mock", "openai"] = "rag_fallback"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    analysis_api_key: SecretStr = SecretStr("")
    binary_sentiment: bool = True
    rag_top_k: int = Field(6, ge=1, le=100)
    rag_language_k: int = Field(2, ge=0, le=100)
    rag_domain_k: int = Field(4, ge=0, le=100)
    rag_balance_classes: bool = True
    rag_hilisenti_weight: float = Field(1.0, ge=0, le=10)
    rag_domain_weight: float = Field(1.2, ge=0, le=10)
    domain_relevance_threshold: float = Field(0.40, ge=0, le=1)
    domain_test_fraction: float = Field(0.10, gt=0, lt=0.4)
    domain_validation_fraction: float = Field(0.10, gt=0, lt=0.4)
    rag_include_validation: bool = False
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_revision: str = "main"
    embedding_batch_size: int = Field(32, ge=1)
    embedding_device: str = "cpu"
    embedding_local_files_only: bool = False
    lda_num_topics: int = Field(10, ge=2)
    lda_max_features: int = Field(10000, ge=2)
    lda_min_df: int = Field(2, ge=1)
    lda_max_df: float = Field(0.95, gt=0, le=1)
    lda_random_state: int = 42
    lda_max_iter: int = Field(20, ge=1)
    lda_ngram_max: int = Field(1, ge=1, le=2)
    enable_pii_sanitization: bool = False
    enable_analysis_logging: bool = True

    @field_validator("binary_sentiment")
    @classmethod
    def require_binary(cls, value: bool) -> bool:
        if not value:
            raise ValueError("This experiment supports binary sentiment only; BINARY_SENTIMENT must be true")
        return value

    @property
    def processed(self) -> Path:
        return self.root / "data/processed"

    @property
    def lda_dir(self) -> Path:
        return self.root / "models/lda"

    @property
    def corpus_dir(self) -> Path:
        return self.processed / "multivent_corpus"

    @property
    def vector_dir(self) -> Path:
        return self.root / "vector_store"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key.get_secret_value().strip())


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    values = {name: os.environ[name.upper()] for name in Settings.model_fields
              if name != "root" and name.upper() in os.environ}
    return Settings(**values)
