"""Fit LDA on the full domain-relevant training corpus, independent of labels."""
import _bootstrap  # noqa: F401
import pandas as pd

from app.artifacts import file_hash
from app.config.settings import get_settings
from app.corpus.artifacts import archive_existing, verify_prepared
from app.corpus.guards import assert_no_leakage, load_corpus
from app.topics.lda_service import LDAService


def main() -> None:
    settings = get_settings()
    manifest = verify_prepared(settings)
    data = load_corpus(settings.corpus_dir, "lda_train.csv")
    evaluation = pd.concat([load_corpus(settings.corpus_dir, name) for name in
                            ("evaluation_hilisenti.csv", "evaluation_domain.csv")], ignore_index=True)
    assert_no_leakage(data, evaluation, set(manifest["protected_hashes"]))
    service = LDAService.train(data, settings)
    service.metadata["corpus_manifest_sha256"] = file_hash(settings.corpus_dir / "manifest.json")
    archive_existing(settings.root, ["models/lda", "reports/lda_topics.md"])
    service.save(settings.lda_dir)
    lines = ["# Discovered Multivent LDA topics", "", f"Fitted on {len(data)} relevant training rows; no test text or sentiment labels used as features.", ""]
    for topic in service.all_topics():
        lines += [f"## Topic {topic['id']} - {topic['name']}", "", ", ".join(topic["keywords"]), ""]
    report = settings.root / "reports/lda_topics.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Fitted {settings.lda_num_topics} topics on {len(data)} relevant train rows: {settings.lda_dir}")


if __name__ == "__main__":
    main()
