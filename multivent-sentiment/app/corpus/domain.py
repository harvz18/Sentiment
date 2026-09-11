"""Explainable keyword/category + frozen-encoder domain relevance scoring."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.rag.embedding_service import EmbeddingProvider


class DomainScorer:
    def __init__(self, config_path: Path, encoder: EmbeddingProvider):
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.encoder = encoder
        self.patterns = {word: re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.I)
                         for word in self.config["keywords"] + self.config["irrelevant_keywords"]}

    def score(self, frame: pd.DataFrame, cache: Path) -> pd.DataFrame:
        """No fitting, labels, or test-set tuning; cache only frozen embeddings."""
        payload = {"texts": frame.text_hash.tolist(), "encoder": self.encoder.identity,
                   "anchors": self.config["descriptions"]}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                similarities = saved["similarities"]
        else:
            anchors = self.encoder.encode(self.config["descriptions"])
            anchors = anchors / np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-12)
            vectors = self.encoder.encode(frame.text_clean.tolist())
            vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            similarities = np.clip((vectors @ anchors.T).max(axis=1), 0, 1)
            np.savez_compressed(path, similarities=similarities)
        if len(similarities) != len(frame):
            raise ValueError("Domain embedding cache does not match corpus")
        result = frame.copy()
        scores, keyword_scores, penalties, keyword_lists = [], [], [], []
        for row, semantic in zip(frame.itertuples(), similarities):
            hits = [w for w in self.config["keywords"] if self.patterns[w].search(row.text_clean)]
            unrelated = [w for w in self.config["irrelevant_keywords"] if self.patterns[w].search(row.text_clean)]
            keyword = min(1.0, len(hits) / 2)
            category = self.config["category_priors"].get(row.source_domain, 0.0)
            penalty = self.config["irrelevant_penalty"] if unrelated else 0.0
            relevance = self.config["semantic_weight"] * float(semantic) + self.config["keyword_weight"] * keyword + self.config["category_weight"] * category - penalty
            scores.append(float(np.clip(relevance, 0, 1)))
            keyword_scores.append(keyword)
            penalties.append(penalty)
            keyword_lists.append("|".join(hits))
        result["domain_relevance_score"] = scores
        result["domain_semantic_similarity"] = similarities
        result["domain_keyword_score"] = keyword_scores
        result["domain_irrelevant_penalty"] = penalties
        result["domain_keyword_hits"] = keyword_lists
        return result
