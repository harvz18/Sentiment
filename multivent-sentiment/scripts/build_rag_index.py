"""Rebuild binary FAISS from the verified balanced corpus; any leakage fails."""
import _bootstrap  # noqa: F401
import pandas as pd

from app.artifacts import file_hash
from app.config.settings import get_settings
from app.corpus.artifacts import archive_existing, verify_prepared
from app.corpus.guards import assert_no_leakage, fingerprint_set, load_corpus
from app.rag.embedding_service import LocalEmbeddingProvider
from app.rag.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    manifest = verify_prepared(settings)
    data = load_corpus(settings.corpus_dir, "rag.csv")
    evaluation = pd.concat([load_corpus(settings.corpus_dir, name) for name in
                            ("evaluation_hilisenti.csv", "evaluation_domain.csv")], ignore_index=True)
    forbidden = set(manifest["protected_hashes"]) | fingerprint_set(evaluation)
    # Recompute hashes and compare before encoding, even if someone tampered with CSV hash fields.
    assert_no_leakage(data, evaluation, forbidden)
    records = [{**row, "text": row["text_clean"]} for row in data.to_dict("records")]
    VectorStore.validate_records(records, forbidden)
    encoder = LocalEmbeddingProvider(settings)
    if encoder.identity != manifest["embedding"]:
        raise ValueError("Encoder changed since corpus scoring; prepare the corpus again")
    vectors = encoder.encode(data.text_clean.tolist())
    store = VectorStore.build(vectors, records, encoder.identity, forbidden, settings.enable_pii_sanitization)
    store.manifest["corpus_manifest_sha256"] = file_hash(settings.corpus_dir / "manifest.json")
    store.manifest["dataset_sha256"] = file_hash(settings.corpus_dir / "rag.csv")
    archive_existing(settings.root, ["vector_store"])
    store.save(settings.vector_dir)
    print(f"Indexed {len(records)} binary references: {data.pool.value_counts().to_dict()}; leakage checks passed")


if __name__ == "__main__":
    main()
