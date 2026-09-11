import csv
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.config.settings import ROOT, Settings
from app.corpus.adapters import adapt_source
from app.corpus.guards import assert_no_leakage
from app.corpus.prepare import assign_splits, balance_rag, deduplicate
from app.corpus.registry import DatasetSource, load_registry
from app.preprocessing.text_cleaner import text_fingerprint
from app.schemas.sentiment import ProviderResult
from app.sentiment.heuristic_provider import evidence_breakdown


@pytest.fixture
def registry():
    return {s.id: s for s in load_registry(ROOT / "config/datasets.yaml")}


def test_registry_blocks_unknown_license(registry):
    row = registry["mams"].model_dump()
    row["license"] = "unknown"
    with pytest.raises(ValidationError, match="Unreviewed license"):
        DatasetSource.model_validate(row)
    row["enabled"] = False
    assert not DatasetSource.model_validate(row).enabled


def test_neutral_is_excluded_never_reassigned(registry, tmp_path):
    path = tmp_path / "test.csv"
    path.write_text('sentence,label\nNeutral event,"""1"""\nNami food,2\nDugay service,0\nBad label,8\n', encoding="utf-8")
    frame, bad, stats, protected = adapt_source(registry["hilisenti"], [path])
    assert set(frame.sentiment) == {"negative", "positive"}
    assert stats["neutral_excluded"] == 1 and stats["malformed_excluded"] == 1
    assert len(bad) == 2
    assert text_fingerprint("Neutral event") in protected
    assert set(frame.split) == {"test"}


def test_mams_mixed_polarities_are_not_majority_voted(registry, tmp_path):
    path = tmp_path / "mams.parquet"
    pd.DataFrame([
        {"text": "Different aspect opinions", "category": "food,service,price", "polarity": "positive,negative,positive"},
        {"text": "Clear food praise", "category": "food,service", "polarity": "positive,neutral"},
        {"text": "Just information", "category": "place", "polarity": "neutral"},
        {"text": "Broken alignment", "category": "food,staff", "polarity": "negative"},
    ]).to_parquet(path)
    frame, bad, stats, _ = adapt_source(registry["mams"], [path])
    assert len(frame) == 1 and frame.iloc[0].sentiment == "positive" and frame.iloc[0].aspect == "food"
    assert {r["reason"] for r in bad} == {"conflicting_aspects", "neutral", "malformed"}
    assert stats["neutral_excluded"] == 1


def test_booking_fields_and_placeholders_keep_review_groups(registry, tmp_path, settings):
    path = tmp_path / "hotel.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uniq_id", "positive_review_text", "negative_review_text", "language"])
        writer.writeheader()
        writer.writerows([
            {"uniq_id": "one", "positive_review_text": "Great venue", "negative_review_text": "Slow staff", "language": "en"},
            {"uniq_id": "two", "positive_review_text": "\\N", "negative_review_text": "Nothing to dislike", "language": "en"},
        ])
    frame, bad, stats, _ = adapt_source(registry["booking_hotel"], [path])
    assert len(frame) == 2 and set(frame.sentiment) == {"positive", "negative"}
    assert frame.group_id.nunique() == 1
    assert assign_splits(frame, settings).split.nunique() == 1
    assert {r["reason"] for r in bad} == {"empty_or_placeholder", "no_complaint_in_negative_field"}
    assert set(frame.aspect) == {"general"}
    assert stats["original_rows"] == 2 and stats["candidate_units"] == 4


def test_uci_uses_own_label_mapping_and_only_restaurants(registry, tmp_path):
    source = registry["uci_restaurants"]
    path = tmp_path / "uci.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(source.archive_member, "Wonderful meal\t1\nBad meal\t0\nMalformed row")
        archive.writestr("sentiment labelled sentences/imdb_labelled.txt", "Unrelated movie\t1")
    frame, bad, _, _ = adapt_source(source, [path])
    assert frame.iloc[0].sentiment == "positive"
    assert len(frame) == 2 and len(bad) == 1
    assert "Unrelated movie" not in set(frame.text_original)


def test_split_assignment_is_order_independent(corpus_data, settings):
    a = assign_splits(corpus_data, settings).set_index("id").split.sort_index()
    b = assign_splits(corpus_data.sample(frac=1, random_state=9), settings).set_index("id").split.sort_index()
    pd.testing.assert_series_equal(a, b)
    local = corpus_data[corpus_data.source_dataset == "hilisenti"]
    assert assign_splits(local, settings).split.tolist() == local.split.tolist()


def test_balancing_is_deterministic_and_preserves_excess(corpus_data):
    frame = corpus_data[corpus_data.split == "train"]
    extra = frame.iloc[[0]].copy()
    extra["id"] = "extra"
    combined = pd.concat([frame, extra], ignore_index=True)
    balanced, excess = balance_rag(combined, True, 42)
    again, _ = balance_rag(combined.sample(frac=1, random_state=1), True, 42)
    assert balanced.sentiment.value_counts().nunique() == 1
    assert set(balanced.id) == set(again.id)
    assert len(balanced) + len(excess) == len(combined)
    assert set(balanced.pool) == {"language", "domain"}


def test_leakage_fails_for_test_split_and_forged_hash(corpus_data):
    evaluation = corpus_data[corpus_data.split == "test"]
    with pytest.raises(ValueError, match="Test or unknown"):
        assert_no_leakage(corpus_data, evaluation)
    forged = evaluation.copy()
    forged["split"] = "train"
    forged["text_hash"] = "pretend-this-is-new"
    with pytest.raises(ValueError, match="fingerprint"):
        assert_no_leakage(forged, evaluation)


def test_same_review_different_fields_cannot_cross_eval(corpus_data):
    train = corpus_data.iloc[[0]].copy()
    test = corpus_data.iloc[[1]].copy()
    test["split"] = "test"
    test["group_id"] = train.iloc[0].group_id
    with pytest.raises(ValueError, match="same source review"):
        assert_no_leakage(train, test)


def test_deduplication_preserves_test_copy_and_audits(corpus_data):
    test = corpus_data[corpus_data.split == "test"]
    duplicate = test.copy()
    duplicate["split"] = "train"
    duplicate["id"] = "new-train-copy"
    clean, bad = deduplicate(pd.concat([test, duplicate]), {test.iloc[0].text_hash})
    assert len(clean) == 1 and clean.iloc[0].split == "test" and len(bad) == 1


def test_binary_schema_and_zero_weight_behavior(runtime):
    with pytest.raises(ValidationError):
        Settings(binary_sentiment=False)
    with pytest.raises(ValidationError):
        ProviderResult(sentiment="neutral", confidence=.5, scores={"positive": .5, "negative": .5},
                       language="unknown", reason="invalid", provider="mock")
    examples = runtime.retrieval.retrieve("food", 6)
    for example in examples:
        example.source_weight = 0
    verdict = runtime.provider.analyze("food", runtime.lda.infer("food"), examples)
    assert verdict.sentiment == "negative" and verdict.confidence == 0


def test_debug_contributions_match_normalized_scores(runtime):
    result = runtime.analysis.analyze("food service", debug=True)
    positive = negative = 0.0
    for row in result.debug.evidence:
        expected = max(0, row.similarity) * row.domain_relevance_score * row.source_weight
        assert row.weighted_contribution == pytest.approx(expected)
        if row.sentiment == "positive":
            positive += expected
        else:
            negative += expected
    assert result.scores.positive == pytest.approx(positive / (positive + negative))
    assert result.scores.negative == pytest.approx(negative / (positive + negative))


def test_domain_score_uses_keywords_and_no_sentiment_labels(corpus_data, tmp_path):
    from app.corpus.domain import DomainScorer
    from conftest import TinyEncoder
    scorer = DomainScorer(ROOT / "config/domain.yaml", TinyEncoder())
    frame = corpus_data.iloc[:2].copy()
    before = scorer.score(frame, tmp_path)
    frame["sentiment"] = ["negative", "positive"]
    after = scorer.score(frame, tmp_path)
    assert before.domain_relevance_score.tolist() == after.domain_relevance_score.tolist()
    assert before.domain_relevance_score.between(0, 1).all()
    assert before.domain_keyword_score.gt(0).all()
