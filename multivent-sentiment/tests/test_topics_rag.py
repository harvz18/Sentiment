import numpy as np
import pytest

from app.artifacts import read_json, write_json
from app.rag.vector_store import VectorStore
from app.topics.lda_service import LDAService


def test_lda_inference_persistence(runtime, tmp_path):
    topic = runtime.lda.infer("Nami gid ang food")
    assert topic.id in {0, 1} and 0 < topic.probability <= 1
    assert "Unassigned Topic" in topic.name
    assert topic.keywords
    runtime.lda.save(tmp_path)
    loaded = LDAService.load(tmp_path)
    assert loaded.infer("Nami gid ang food") == topic
    assert loaded.infer("zzzzunknownword").id is None
    with pytest.raises(ValueError, match="PII"):
        LDAService.load(tmp_path, pii=True)


def test_lda_rejects_test_data(clean_data, settings):
    with pytest.raises(ValueError, match="training records only"):
        LDAService.train(clean_data, settings)


def test_no_test_record_can_enter_index(runtime):
    store = runtime.retrieval.store
    for split in ("test", "unknown", "TEST"):
        record = {**store.records[0], "split": split}
        with pytest.raises(ValueError, match="refuses test"):
            VectorStore.build(np.ones((1, 3)), [record], {}, set())


def test_test_duplicate_under_train_label_is_rejected(runtime):
    from app.preprocessing.text_cleaner import text_fingerprint
    record = runtime.retrieval.store.records[0]
    with pytest.raises(ValueError, match="overlaps held-out"):
        VectorStore.build(np.ones((1, 3)), [record], {}, {text_fingerprint(record["text"])})


def test_source_aware_retrieval_retains_two_pools(runtime):
    examples = runtime.retrieval.retrieve("Nami food service", 6)
    assert len(examples) == 6
    assert sum(e.pool == "language" for e in examples) == 2
    assert sum(e.pool == "domain" for e in examples) == 4
    assert len({e.id for e in examples}) == 6
    assert examples[0].rerank_score >= examples[-1].rerank_score


def test_missing_pool_fills_from_available_pool(runtime):
    original = runtime.retrieval.store
    records = [r for r in original.records if r["pool"] == "domain"]
    runtime.retrieval.store = VectorStore.build(np.ones((len(records), 3)), records, {}, set())
    assert len(runtime.retrieval.retrieve("food", 6)) == 6


def test_neutral_cannot_enter_binary_index(runtime):
    record = {**runtime.retrieval.store.records[0], "sentiment": "neutral"}
    with pytest.raises(ValueError, match="neutral"):
        VectorStore.build(np.ones((1, 3)), [record], {}, set())


def test_cosine_retrieval_and_reload(runtime, tmp_path):
    store = runtime.retrieval.store
    expected = store.search(np.array([3, 1, 1]), 100)
    assert len(expected) == len(store.records)
    assert expected[0].similarity >= expected[-1].similarity
    store.save(tmp_path)
    restored = VectorStore.load(tmp_path, runtime.retrieval.embeddings.identity)
    assert restored.search(np.array([3, 1, 1]), 100) == expected
    with pytest.raises(ValueError, match="differs"):
        VectorStore.load(tmp_path, {"model": "other"})
    with pytest.raises(ValueError):
        restored.search(np.zeros(3), 5)


def test_tampered_test_record_is_rejected_on_load(runtime, tmp_path):
    from app.artifacts import file_hash
    store = runtime.retrieval.store
    store.save(tmp_path)
    records = read_json(tmp_path / "metadata.json")
    records[0]["split"] = "test"
    write_json(tmp_path / "metadata.json", records)
    manifest = read_json(tmp_path / "manifest.json")
    manifest["files"]["metadata.json"] = file_hash(tmp_path / "metadata.json")
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="refuses test"):
        VectorStore.load(tmp_path, runtime.retrieval.embeddings.identity)
