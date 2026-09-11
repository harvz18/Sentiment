"""Prepare a versioned binary domain corpus while preserving original source data."""
import hashlib
import logging
from pathlib import Path

import pandas as pd

from app.artifacts import file_hash, write_json
from app.config.settings import Settings
from app.corpus.adapters import adapt_source
from app.corpus.domain import DomainScorer
from app.corpus.download import download_source
from app.corpus.guards import assert_no_leakage, fingerprint_set
from app.corpus.registry import load_registry
from app.corpus.reports import write_reports
from app.rag.embedding_service import LocalEmbeddingProvider

log = logging.getLogger(__name__)


def assign_splits(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Preserve HiliSenti splits; hash whole domain review groups before filtering."""
    result = frame.copy()
    for index, row in result.iterrows():
        if row.source_dataset == "hilisenti":
            continue
        unit = int(hashlib.sha256(f"{settings.lda_random_state}:{row.group_id}".encode()).hexdigest()[:16], 16) / 2**64
        result.at[index, "split"] = ("test" if unit < settings.domain_test_fraction else
                                      "validation" if unit < settings.domain_test_fraction + settings.domain_validation_fraction else "train")
    return result


def deduplicate(frame: pd.DataFrame, protected: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Held-out priority, no row relocation, and quarantine contradictory labels."""
    priority = {"test": 0, "validation": 1, "train": 2}
    kept, excluded = [], []
    for _, group in frame.groupby("text_hash", sort=False):
        rows = group.sort_values(["split", "id"], key=lambda values: values.map(priority) if values.name == "split" else values)
        if group.sentiment.nunique() > 1:
            excluded.extend({**r, "reason": "conflicting_duplicate_labels"} for r in rows.to_dict("records"))
            continue
        eligible = rows[~((rows.split != "test") & (rows.text_hash.isin(protected) | rows.original_text_hash.isin(protected)))]
        rejected_ids = set(rows.id) - set(eligible.id)
        excluded.extend({**r, "reason": "protected_hilisenti_test_text"} for r in rows[rows.id.isin(rejected_ids)].to_dict("records"))
        if not eligible.empty:
            kept.append(eligible.iloc[0].to_dict())
            excluded.extend({**r, "reason": "duplicate_text_held_out_priority"} for r in eligible.iloc[1:].to_dict("records"))
    return pd.DataFrame(kept, columns=frame.columns), excluded


def balance_rag(frame: pd.DataFrame, enabled: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal classes within each source, retaining both linguistic and domain pools."""
    if not enabled:
        return frame.copy(), frame.iloc[:0].copy()
    pieces = []
    for _, source in frame.groupby("source_dataset", sort=True):
        count = min(int((source.sentiment == label).sum()) for label in ("negative", "positive"))
        for label in ("negative", "positive"):
            pieces.append(source[source.sentiment == label].sort_values("id").sample(n=count, random_state=seed))
    balanced = pd.concat(pieces).sort_values("id").reset_index(drop=True) if pieces else frame.iloc[:0].copy()
    excess = frame[~frame.id.isin(balanced.id)].copy()
    return balanced, excess


def prepare_multivent(settings: Settings) -> dict:
    sources = [s for s in load_registry(settings.root / "config/datasets.yaml") if s.enabled]
    if not any(s.id == "hilisenti" for s in sources) or not any(s.pool == "domain" for s in sources):
        raise ValueError("The experiment requires HiliSenti and at least one reviewed domain source")
    all_dir = settings.processed / "all_sources"
    all_dir.mkdir(parents=True, exist_ok=True)
    stats, frames, exclusions, protected = {}, [], [], set()
    for source in sources:
        log.info("Preparing %s under %s", source.name, source.license)
        paths = download_source(source, settings.root)
        frame, bad, summary, reserved = adapt_source(source, paths, settings.enable_pii_sanitization)
        frame.to_csv(all_dir / f"{source.id}.csv", index=False, encoding="utf-8")
        frames.append(frame)
        exclusions.extend(bad)
        protected.update(reserved)
        stats[source.id] = summary
        log.info("%s: %d source records, %d accepted binary units before deduplication", source.id, summary["original_rows"], len(frame))
    pd.DataFrame(exclusions).to_csv(all_dir / "exclusions.csv", index=False, encoding="utf-8")
    combined = assign_splits(pd.concat(frames, ignore_index=True), settings)
    combined, duplicates = deduplicate(combined, protected)
    pd.DataFrame(duplicates).to_csv(all_dir / "duplicates.csv", index=False, encoding="utf-8")
    if combined.empty:
        raise ValueError("No binary records survived source validation")
    encoder = LocalEmbeddingProvider(settings)
    scorer = DomainScorer(settings.root / "config/domain.yaml", encoder)
    combined = scorer.score(combined, settings.root / "models/embeddings/domain_score_cache")
    relevant = combined[combined.domain_relevance_score >= settings.domain_relevance_threshold].copy()
    # Test selection is fixed before relevance filtering/balancing; do not cherry-pick easy tests.
    evaluation = combined[combined.split == "test"].copy()
    validation = combined[combined.split == "validation"].copy()
    forbidden = protected | fingerprint_set(evaluation)
    if not settings.rag_include_validation:
        forbidden |= fingerprint_set(validation)
    eligible = relevant[relevant.split.isin(["train", "validation"] if settings.rag_include_validation else ["train"])].copy()
    lda = relevant[relevant.split == "train"].copy()
    rag, excess = balance_rag(eligible, settings.rag_balance_classes, settings.lda_random_state)
    assert_no_leakage(rag, evaluation, forbidden)
    assert_no_leakage(lda, evaluation, forbidden)
    if rag.empty or set(rag.pool) != {"language", "domain"}:
        raise ValueError("Filtering/balancing left an empty language or domain pool; inspect threshold/source settings")
    directory = settings.corpus_dir
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {"all_binary.csv": combined, "domain_filtered.csv": relevant, "rag.csv": rag,
               "balance_excluded.csv": excess, "lda_train.csv": lda, "validation.csv": validation,
               "evaluation_hilisenti.csv": evaluation[evaluation.source_dataset == "hilisenti"],
               "evaluation_domain.csv": evaluation[evaluation.source_dataset != "hilisenti"],
               "domain_excluded.csv": combined[combined.domain_relevance_score < settings.domain_relevance_threshold]}
    for name, frame in outputs.items():
        frame.to_csv(directory / name, index=False, encoding="utf-8")
    for source in sources:
        sid = source.id
        stats[sid].update({"duplicates_removed": sum(row["source_dataset"] == sid for row in duplicates),
                           "domain_irrelevant_excluded": int(((combined.source_dataset == sid) & (combined.domain_relevance_score < settings.domain_relevance_threshold)).sum()),
                           "final_accepted": int((relevant.source_dataset == sid).sum()),
                           "rag_rows": int((rag.source_dataset == sid).sum()),
                           "lda_rows": int((lda.source_dataset == sid).sum()),
                           "evaluation_rows": int((evaluation.source_dataset == sid).sum())})
    thresholds = []
    for threshold in sorted({0.2, 0.3, 0.4, 0.5, 0.6, settings.domain_relevance_threshold}):
        subset = combined[(combined.split == "train") & (combined.domain_relevance_score >= threshold)]
        thresholds.append({"threshold": threshold, "train_rows": len(subset),
                           "positive": int((subset.sentiment == "positive").sum()),
                           "negative": int((subset.sentiment == "negative").sum()),
                           "sources": subset.source_dataset.value_counts().to_dict()})
    manifest = {"version": 2, "binary_sentiment": True, "sources": [s.model_dump() for s in sources],
                "source_stats": stats, "total_source_records_examined": sum(s["original_rows"] for s in stats.values()),
                "neutral_excluded": sum(s["neutral_excluded"] for s in stats.values()),
                "embedding": encoder.identity, "pii_sanitization": settings.enable_pii_sanitization,
                "domain_relevance_threshold": settings.domain_relevance_threshold,
                "rag_balance_classes": settings.rag_balance_classes, "rag_include_validation": settings.rag_include_validation,
                "seed": settings.lda_random_state, "domain_test_fraction": settings.domain_test_fraction,
                "domain_validation_fraction": settings.domain_validation_fraction,
                "protected_hashes": sorted(forbidden), "threshold_statistics": thresholds,
                "registry_sha256": file_hash(settings.root / "config/datasets.yaml"),
                "domain_config_sha256": file_hash(settings.root / "config/domain.yaml"),
                "files": {name: file_hash(directory / name) for name in outputs},
                "sizes": {name: len(frame) for name, frame in outputs.items()}, "leakage_checks_passed": True}
    write_json(directory / "manifest.json", manifest)
    write_reports(settings.root / "reports", manifest, outputs)
    log.info("Prepared %d RAG rows and %d LDA rows; held out %d binary test rows", len(rag), len(lda), len(evaluation))
    return manifest
