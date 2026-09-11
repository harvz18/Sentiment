import numpy as np
import pandas as pd
import pytest

from app.config.settings import Settings
from app.dataset import clean_frames
from app.corpus.adapters import normalized_record
from app.corpus.registry import load_registry
from app.config.settings import ROOT
from app.rag.embedding_service import EmbeddingProvider
from app.rag.retrieval_service import RetrievalService
from app.rag.vector_store import VectorStore
from app.sentiment.factory import create_provider
from app.services.analysis_service import AnalysisService
from app.services.runtime import Runtime
from app.topics.lda_service import LDAService


class TinyEncoder(EmbeddingProvider):
    """Deterministic test double, never used for application/evaluation embeddings."""
    @property
    def identity(self):
        return {"model": "test-only", "dimension": 3}

    def encode(self, texts):
        return np.array([[1 + text.lower().count("nami"), 1 + text.lower().count("dugay"), 1]
                         for text in texts], dtype="float32")


@pytest.fixture
def settings(tmp_path):
    return Settings(root=tmp_path, lda_num_topics=2, lda_min_df=1, lda_max_df=1.0,
                    lda_max_iter=3, enable_analysis_logging=False)


@pytest.fixture
def clean_data():
    rows = [("Nami gid ang food", "2", "train"), ("Dugay gid ang service", "0", "train"),
            ("May schedule ang event", "1", "train"), ("Nami ang staff", "2", "train"),
            ("Dugay ang order", "0", "train"), ("May event sa plaza", "1", "train"),
            ("Test reserved feedback", "1", "test"), ("Validation feedback", "2", "validation")]
    return clean_frames([pd.DataFrame(rows, columns=["sentence", "label", "split"])])[0]


@pytest.fixture
def corpus_data():
    registry = {source.id: source for source in load_registry(ROOT / "config/datasets.yaml")}
    rows = []
    for index, (text, label, source) in enumerate([
        ("Nami gid ang food", "positive", "hilisenti"), ("Dugay gid ang service", "negative", "hilisenti"),
        ("Nami ang staff", "positive", "hilisenti"), ("Dugay ang order", "negative", "hilisenti"),
        ("Wonderful restaurant staff", "positive", "uci_restaurants"), ("Terrible food service", "negative", "uci_restaurants"),
        ("Clean hotel venue", "positive", "booking_hotel"), ("Dirty hotel room", "negative", "booking_hotel"),
        ("Delicious meal", "positive", "mams"), ("Very late dinner", "negative", "mams"),
    ]):
        rows.append({**normalized_record(registry[source], str(index), text, label, "train", "general", False),
                     "domain_relevance_score": .8})
    rows.append({**normalized_record(registry["hilisenti"], "held-out", "Reserved test text", "positive", "test", "general", False), "domain_relevance_score": .7})
    return pd.DataFrame(rows)


@pytest.fixture
def runtime(settings, corpus_data):
    training = corpus_data[corpus_data.split == "train"]
    lda = LDAService.train(training, settings)
    encoder = TinyEncoder()
    records = [{**r, "text": r["text_clean"]} for r in training.to_dict("records")]
    store = VectorStore.build(encoder.encode(training.text_clean.tolist()), records, encoder.identity,
                              set(corpus_data[corpus_data.split == "test"].text_hash))
    instance = Runtime(settings)
    instance.lda = lda
    instance.retrieval = RetrievalService(encoder, store, settings)
    instance.provider = create_provider(settings)
    instance.analysis = AnalysisService(settings, lda, instance.retrieval, instance.provider)
    return instance
