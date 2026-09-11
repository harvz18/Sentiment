"""Fail-closed split and fingerprint checks shared by fitting, indexing and evaluation."""
import pandas as pd

from app.preprocessing.text_cleaner import text_fingerprint


def fingerprint_set(frame: pd.DataFrame) -> set[str]:
    hashes = set(frame.text_clean.map(text_fingerprint))
    if "text_original" in frame:
        hashes.update(frame.text_original.map(text_fingerprint))
    return hashes


def assert_no_leakage(training: pd.DataFrame, evaluation: pd.DataFrame,
                      protected_hashes: set[str] | None = None) -> None:
    if set(training.sentiment) - {"positive", "negative"}:
        raise ValueError("Non-binary record in training/RAG corpus")
    if set(training.split) - {"train", "validation"}:
        raise ValueError("Test or unknown split cannot enter training/RAG")
    # Recompute from text; do not trust a caller-supplied text_hash column.
    forbidden = fingerprint_set(evaluation) | (protected_hashes or set())
    if fingerprint_set(training).intersection(forbidden):
        raise ValueError("LEAKAGE: training/RAG fingerprint collides with held-out evaluation text")
    if "group_id" in training and "group_id" in evaluation:
        if set(training.group_id).intersection(evaluation.group_id):
            raise ValueError("LEAKAGE: fields from the same source review cross evaluation/training")


def load_corpus(directory, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError("Run python scripts/prepare_multivent_dataset.py first")
    return pd.read_csv(path, keep_default_na=False)
