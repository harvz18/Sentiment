"""Report binary HiliSenti and held-out hospitality/restaurant metrics separately."""
import argparse
from datetime import datetime, timezone
from time import perf_counter

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.artifacts import file_hash, read_json, write_json
from app.config.settings import get_settings
from app.corpus.artifacts import archive_existing, verify_prepared
from app.corpus.guards import assert_no_leakage, fingerprint_set, load_corpus
from app.services.runtime import Runtime


def binary_metrics(truth: list[str], predicted: list[str]) -> dict:
    if not truth or len(truth) != len(predicted) or (set(truth) | set(predicted)) - {"negative", "positive"}:
        raise ValueError("Evaluation requires nonempty, aligned binary labels")
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary",
                                                            pos_label="positive", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(truth, predicted,
                                                                 labels=["negative", "positive"], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predicted)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "macro_f1": float(macro_f1),
            "precision_macro": float(macro_p), "recall_macro": float(macro_r),
            "positive_class": "positive", "label_order": ["negative", "positive"],
            "confusion_matrix": confusion_matrix(truth, predicted, labels=["negative", "positive"]).tolist()}


def check_leakage(runtime: Runtime, evaluation: pd.DataFrame) -> None:
    manifest = verify_prepared(runtime.settings)
    records = pd.DataFrame(runtime.retrieval.store.records)
    assert_no_leakage(records, evaluation, set(manifest["protected_hashes"]))
    if set(runtime.lda.metadata["training_text_hashes"]).intersection(fingerprint_set(evaluation)):
        raise ValueError("Evaluation aborted: test text entered LDA fitting")
    if runtime.retrieval.store.manifest["corpus_manifest_sha256"] != file_hash(runtime.settings.corpus_dir / "manifest.json"):
        raise ValueError("Evaluation corpus differs from built index")


def evaluate_set(runtime: Runtime, test: pd.DataFrame, name: str, limit: int | None) -> dict:
    if set(test.split) != {"test"}:
        raise ValueError("Only held-out test records may be evaluated")
    available = len(test)
    if limit:
        test = test.sample(n=min(limit, available), random_state=runtime.settings.lda_random_state)
    start = perf_counter()
    predictions = []
    for number, row in enumerate(test.itertuples(), 1):
        result = runtime.analysis.analyze(row.text_original, ground_truth=row.sentiment)
        predictions.append({"id": row.id, "source_dataset": row.source_dataset, "ground_truth": row.sentiment,
                            "predicted": result.sentiment, "confidence": result.confidence,
                            "positive_probability": result.scores.positive, "negative_probability": result.scores.negative})
        if number % 100 == 0:
            print(f"{name}: {number}/{len(test)}", flush=True)
    report = {"evaluation": name, "sample_count": len(test), "available_test_rows": available,
              "source_distribution": test.source_dataset.value_counts().to_dict(),
              "class_distribution": test.sentiment.value_counts().to_dict(),
              "elapsed_seconds": perf_counter() - start,
              **binary_metrics([r["ground_truth"] for r in predictions], [r["predicted"] for r in predictions])}
    by_source = {}
    prediction_frame = pd.DataFrame(predictions)
    for source, rows in prediction_frame.groupby("source_dataset"):
        by_source[source] = {"sample_count": len(rows), **binary_metrics(rows.ground_truth.tolist(), rows.predicted.tolist())}
    report["metrics_by_source"] = by_source
    directory = runtime.settings.root / "reports"
    prediction_frame.to_csv(directory / f"predictions_{name}.csv", index=False)
    pd.DataFrame(report["confusion_matrix"], index=pd.Index(["negative", "positive"], name="actual / predicted"),
                 columns=["negative", "positive"]).to_csv(directory / f"confusion_matrix_{name}.csv")
    write_json(directory / f"evaluation_{name}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["rag_fallback", "mock", "openai"])
    parser.add_argument("--limit", type=int, default=None, help="Seeded maximum per evaluation set")
    parser.add_argument("--dataset", choices=["both", "hilisenti", "domain"], default="both")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    settings = get_settings()
    if args.provider:
        settings.ai_provider = args.provider
    if settings.ai_provider == "openai" and (not settings.openai_configured or not settings.openai_model.strip()):
        parser.error("OpenAI evaluation requires key and model; refusing to mislabel fallback metrics")
    runtime = Runtime(settings)
    runtime.load()
    if runtime.analysis is None:
        raise RuntimeError(runtime.errors)
    tests = {name: load_corpus(settings.corpus_dir, f"evaluation_{name}.csv") for name in ("hilisenti", "domain")}
    check_leakage(runtime, pd.concat(tests.values(), ignore_index=True))
    # Archive previous reports, including the earlier three-class experiment.
    archive_existing(settings.root, ["reports/evaluation.json", "reports/confusion_matrix.csv", "reports/predictions.csv"] +
                      [f"reports/{prefix}_{name}.{ext}" for name in tests
                       for prefix, ext in (("evaluation", "json"), ("predictions", "csv"), ("confusion_matrix", "csv"))])
    reports = {name: evaluate_set(runtime, test, name, args.limit) for name, test in tests.items()
               if args.dataset in {"both", name}}
    combined = {"version": 2, "binary_sentiment": True, "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": runtime.provider.name, "results": reports, "seed": settings.lda_random_state,
                "limit_per_set": args.limit, "rag_top_k": settings.rag_top_k,
                "rag_language_k": settings.rag_language_k, "rag_domain_k": settings.rag_domain_k,
                "rag_hilisenti_weight": settings.rag_hilisenti_weight, "rag_domain_weight": settings.rag_domain_weight,
                "corpus_manifest_sha256": file_hash(settings.corpus_dir / "manifest.json"),
                "lda_sha256": file_hash(settings.lda_dir / "lda_model.joblib"),
                "index_sha256": file_hash(settings.vector_dir / "index.faiss"),
                "embedding": runtime.retrieval.embeddings.identity,
                "openai_model": settings.openai_model if runtime.provider.name == "openai" else None,
                "leakage_checks_passed": True,
                "limitations": "Binary metrics are not directly comparable with earlier three-class metrics. Domain set is a published hospitality/restaurant proxy with some weak field labels; not a Multivent event benchmark. No test-driven threshold tuning performed."}
    write_json(settings.root / "reports/evaluation.json", combined)
    lines = ["# Binary fallback evaluation", "", f"Provider: {runtime.provider.name}. No test rows entered RAG/LDA.", "",
             "Precision, recall and F1 use positive as the positive class. Macro F1 averages both classes.", "",
             "| Evaluation set | N | Accuracy | Precision | Recall | F1 | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, report in reports.items():
        lines.append(f"| {name} | {report['sample_count']} | " + " | ".join(f"{report[m]:.4f}" for m in ("accuracy", "precision", "recall", "f1", "macro_f1")) + " |")
        print(f"{name}: n={report['sample_count']} accuracy={report['accuracy']:.4f} F1={report['f1']:.4f} macro F1={report['macro_f1']:.4f}")
    lines += ["", combined["limitations"], "", "Inspect per-source metrics in the JSON reports; record all configuration changes before making model-comparison claims."]
    (settings.root / "reports/binary_evaluation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
