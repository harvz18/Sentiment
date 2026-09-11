"""Unsupervised topic discovery from training text only, independent of labels."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from app.artifacts import file_hash, read_json, write_json
from app.config.settings import Settings
from app.schemas.sentiment import TopicInfo
from app.topics.topic_names import topic_name


class LDAService:
    def __init__(self, vectorizer: CountVectorizer, model: LatentDirichletAllocation,
                 metadata: dict):
        self.vectorizer = vectorizer
        self.model = model
        self.metadata = metadata

    @classmethod
    def train(cls, frame: pd.DataFrame, settings: Settings) -> "LDAService":
        if frame.empty or set(frame["split"]) != {"train"}:
            raise ValueError("LDA fitting accepts nonempty training records only")
        if set(frame["sentiment"]) - {"positive", "negative"}:
            raise ValueError("LDA corpus must exclude neutral records for the binary experiment")
        stopwords = set()
        for language in ("english", "filipino", "hiligaynon"):
            path = settings.root / "data/stopwords" / f"{language}.txt"
            if path.exists():
                stopwords.update(line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
                                 if line.strip() and not line.lstrip().startswith("#"))
        vectorizer = CountVectorizer(max_features=settings.lda_max_features,
                                     min_df=settings.lda_min_df, max_df=settings.lda_max_df,
                                     stop_words=sorted(stopwords) or None,
                                     ngram_range=(1, settings.lda_ngram_max),
                                     token_pattern=r"(?u)\b\w+\b")
        try:
            counts = vectorizer.fit_transform(frame["text_clean"])
        except ValueError as exc:
            raise ValueError("LDA vocabulary is empty/invalid; inspect training text and LDA_MIN_DF/MAX_DF") from exc
        model = LatentDirichletAllocation(n_components=settings.lda_num_topics,
                                         max_iter=settings.lda_max_iter, learning_method="batch",
                                         random_state=settings.lda_random_state, n_jobs=1)
        model.fit(counts)
        terms = vectorizer.get_feature_names_out()
        topics = [{"id": i, "keywords": terms[weights.argsort()[-20:][::-1]].tolist()}
                  for i, weights in enumerate(model.components_)]
        metadata = {"version": 2, "binary_sentiment": True, "topics": topics, "training_rows": len(frame), "training_split": "train",
                    "training_text_hashes": frame["text_hash"].tolist(),
                    "pii_sanitization": settings.enable_pii_sanitization,
                    "config": {k: v for k, v in settings.model_dump().items() if k.startswith("lda_")},
                    "stopwords": sorted(stopwords)}
        return cls(vectorizer, model, metadata)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "lda_model.joblib")
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        self.metadata["files"] = {name: file_hash(directory / name)
                                  for name in ("lda_model.joblib", "vectorizer.joblib")}
        write_json(directory / "topics.json", self.metadata)

    @classmethod
    def load(cls, directory: Path, pii: bool = False) -> "LDAService":
        metadata = read_json(directory / "topics.json")
        if metadata.get("version") != 2 or not metadata.get("binary_sentiment"):
            raise ValueError("Legacy LDA model detected; fit the binary Multivent corpus")
        if metadata["pii_sanitization"] != pii:
            raise ValueError("PII configuration changed; clean data and retrain LDA")
        for name, digest in metadata["files"].items():
            if file_hash(directory / name) != digest:
                raise ValueError("LDA artifact mismatch; rerun train_lda.py")
        return cls(joblib.load(directory / "vectorizer.joblib"),
                   joblib.load(directory / "lda_model.joblib"), metadata)

    def infer(self, text: str) -> TopicInfo:
        counts = self.vectorizer.transform([text])
        if counts.nnz == 0:
            return TopicInfo(id=None, name="No known topic terms", probability=0,
                             keywords=[], in_vocabulary=False)
        distribution = self.model.transform(counts)[0]
        topic_id = int(np.argmax(distribution))
        return TopicInfo(id=topic_id, name=topic_name(topic_id),
                         probability=float(distribution[topic_id]),
                         keywords=self.metadata["topics"][topic_id]["keywords"])

    def all_topics(self) -> list[dict]:
        return [{**topic, "name": topic_name(topic["id"])} for topic in self.metadata["topics"]]
