"""Run the approved 100-review MULTIVENT OpenAI-only evaluation.

The script intentionally calls ``AnalysisService.analyze`` for every record.  It
adds only a transparent client proxy to preserve Responses API usage metadata;
there is no fallback prediction path.  Results are written after every attempt.
"""
import argparse
import csv
import html
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import _bootstrap  # noqa: F401
from app.artifacts import file_hash, write_json
from app.config.settings import get_settings
from app.corpus.artifacts import verify_prepared
from app.corpus.guards import assert_no_leakage, fingerprint_set
from app.services.runtime import Runtime


RESULT_COLUMNS = [
    "Test ID", "Dataset Source", "Original Review", "Ground Truth Label", "Model Prediction",
    "Correct / Incorrect", "Confusion Matrix Category", "OpenAI Model Used", "Input Tokens",
    "Output Tokens", "Evaluation Status",
]
INPUT_RATE = 2.00 / 1_000_000
OUTPUT_RATE = 12.00 / 1_000_000


class UsageRecordingResponses:
    """Delegate the SDK call while recording the most recent non-secret metadata."""

    def __init__(self, delegate):
        self._delegate = delegate
        self.last_usage = {}
        self.last_model = None

    def parse(self, *args, **kwargs):
        result = self._delegate.parse(*args, **kwargs)
        usage = getattr(result, "usage", None)
        if usage is not None:
            self.last_usage = usage.model_dump() if hasattr(usage, "model_dump") else {
                key: getattr(usage, key) for key in ("input_tokens", "output_tokens", "total_tokens")
                if hasattr(usage, key)
            }
        else:
            self.last_usage = {}
        self.last_model = getattr(result, "model", None)
        return result


class UsageRecordingClient:
    def __init__(self, delegate):
        self._delegate = delegate
        self.responses = UsageRecordingResponses(delegate.responses)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def binary_metrics(truth: list[str], predicted: list[str]) -> dict:
    """Same positive-class definition and calculations as scripts/evaluate.py."""
    if not truth or len(truth) != len(predicted) or (set(truth) | set(predicted)) - {"negative", "positive"}:
        raise ValueError("Evaluation requires nonempty, aligned binary labels")
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", pos_label="positive", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=["negative", "positive"], average="macro", zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=["negative", "positive"]).tolist()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1_score": float(f1),
        "macro_f1_score": float(macro_f1),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "positive_class": "positive",
        "label_order": ["negative", "positive"],
        "confusion_matrix": matrix,
        "correct_predictions": int(matrix[0][0] + matrix[1][1]),
        "incorrect_predictions": int(matrix[0][1] + matrix[1][0]),
        "total_evaluated_reviews": len(truth),
    }


def usage_value(usage: dict, name: str) -> int | None:
    value = usage.get(name)
    return int(value) if isinstance(value, int | float) else None


def category(truth: str, predicted: str) -> str:
    return {("positive", "positive"): "TP", ("negative", "negative"): "TN",
            ("negative", "positive"): "FP", ("positive", "negative"): "FN"}[(truth, predicted)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()


def xlsx_escape(value) -> str:
    value = "" if value is None else str(value)
    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", value)[:32767]
    return html.escape(value, quote=False)


def write_xlsx(path: Path, rows: list[dict]) -> None:
    """Create a dependency-free .xlsx export using inline strings for Excel."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "_rels").mkdir()
        (root / "xl" / "_rels").mkdir(parents=True)
        (root / "xl" / "worksheets").mkdir(parents=True)
        (root / "[Content_Types].xml").write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>', encoding="utf-8"
        )
        (root / "_rels" / ".rels").write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>', encoding="utf-8"
        )
        (root / "xl" / "workbook.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Prediction Results" sheetId="1" r:id="rId1"/></sheets></workbook>', encoding="utf-8"
        )
        (root / "xl" / "_rels" / "workbook.xml.rels").write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>', encoding="utf-8"
        )
        all_rows = [dict(zip(RESULT_COLUMNS, RESULT_COLUMNS)), *rows]
        row_xml = []
        for row_number, row in enumerate(all_rows, 1):
            cells = "".join(
                f'<c r="{chr(65 + index)}{row_number}" t="inlineStr"><is><t xml:space="preserve">{xlsx_escape(row.get(column))}</t></is></c>'
                for index, column in enumerate(RESULT_COLUMNS)
            )
            row_xml.append(f'<row r="{row_number}">{cells}</row>')
        (root / "xl" / "worksheets" / "sheet1.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>'
            '<sheetData>' + "".join(row_xml) + '</sheetData></worksheet>', encoding="utf-8"
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in root.rglob("*"):
                if item.is_file():
                    archive.write(item, item.relative_to(root).as_posix())


def source_metrics(result_rows: list[dict], source: str) -> dict:
    rows = [row for row in result_rows if row["Dataset Source"] == source and row["Evaluation Status"] == "completed"]
    failed = [row for row in result_rows if row["Dataset Source"] == source and row["Evaluation Status"] != "completed"]
    report = {"selected_reviews": len(rows) + len(failed), "failed_requests": len(failed)}
    if rows:
        report.update(binary_metrics([row["Ground Truth Label"] for row in rows], [row["Model Prediction"] for row in rows]))
    else:
        report.update({"total_evaluated_reviews": 0, "metrics_unavailable": "No completed OpenAI responses"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue a previously interrupted, same-manifest run")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Stop cleanly after this many pending records; useful for resumable batches")
    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be positive")
    settings = get_settings()
    settings.enable_analysis_logging = False
    if settings.ai_provider != "openai" or not settings.openai_configured or not settings.openai_model.strip():
        raise SystemExit("OpenAI provider, API key, and model are required; refusing any fallback evaluation")
    output = args.output or settings.root / "reports" / "evaluation_100_openai"
    sample_path = output / "selected_test_samples.csv"
    manifest_path = output / "evaluation_manifest.json"
    if not sample_path.exists() or not manifest_path.exists():
        raise SystemExit("Run prepare_openai_100_evaluation.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_pending_explicit_user_approval_no_openai_requests_made":
        raise SystemExit("Manifest does not describe an unrun, approval-gated evaluation")
    if file_hash(sample_path) != manifest["artifact_hashes"]["selected_samples_sha256"]:
        raise SystemExit("Selected sample file changed after approval; refusing to submit altered records")
    sample = pd.read_csv(sample_path, keep_default_na=False)
    if len(sample) != 100 or sample["Test ID"].nunique() != 100 or set(sample["split"]) != {"test"}:
        raise SystemExit("Selected sample invariant failed")
    held_out = pd.concat([
        pd.read_csv(settings.corpus_dir / "evaluation_hilisenti.csv"),
        pd.read_csv(settings.corpus_dir / "evaluation_domain.csv"),
    ], ignore_index=True)
    held_out = held_out.loc[held_out.id.isin(sample["id"])].copy()
    if len(held_out) != 100 or held_out.id.nunique() != 100:
        raise SystemExit("Selected records cannot be reconstructed from frozen held-out corpus files")

    runtime = Runtime(settings)
    runtime.load()
    if runtime.analysis is None or runtime.provider is None or runtime.provider.name != "openai":
        raise SystemExit(f"OpenAI runtime unavailable; no fallback permitted: {runtime.errors}")
    if runtime.settings.openai_model != manifest["api_cost_estimate"]["model"]:
        raise SystemExit("Configured model changed after approval; refusing to run")
    verify_prepared(settings)
    assert_no_leakage(pd.DataFrame(runtime.retrieval.store.records), held_out)
    if set(runtime.lda.metadata["training_text_hashes"]).intersection(fingerprint_set(held_out)):
        raise SystemExit("Selected test text entered LDA fitting")
    if runtime.retrieval.store.manifest["corpus_manifest_sha256"] != file_hash(settings.corpus_dir / "manifest.json"):
        raise SystemExit("FAISS index does not match corpus manifest")

    recorder = UsageRecordingClient(runtime.provider.client)
    runtime.provider.client = recorder
    raw_path = output / "raw_openai_responses.jsonl"
    results_path = output / "prediction_results.csv"
    xlsx_path = output / "prediction_results.xlsx"
    if args.resume:
        if not results_path.exists():
            raise SystemExit("Cannot resume because prediction_results.csv is missing")
        with results_path.open(encoding="utf-8-sig", newline="") as stream:
            result_rows = list(csv.DictReader(stream))
        existing_ids = {item["Test ID"] for item in result_rows}
        if len(existing_ids) != len(result_rows) or not existing_ids.issubset(set(sample["Test ID"])):
            raise SystemExit("Existing result file is not a valid prefix of this manifest")
        raw_path.touch(exist_ok=True)
        if raw_path.stat().st_size == 0 and result_rows:
            # The tabular export is flushed after every response. Rebuild a minimal
            # audit trail rather than reissuing calls after external interruption.
            with raw_path.open("w", encoding="utf-8") as recovered:
                for item in result_rows:
                    recovered.write(json.dumps({
                        "test_id": item["Test ID"], "dataset_source": item["Dataset Source"],
                        "ground_truth": item["Ground Truth Label"], "status": item["Evaluation Status"],
                        "prediction": item["Model Prediction"], "usage": {
                            "input_tokens": item["Input Tokens"], "output_tokens": item["Output Tokens"],
                        }, "response_model": item["OpenAI Model Used"],
                        "audit_note": "Reconstructed from the per-request CSV after external interruption; no request was repeated.",
                    }, ensure_ascii=False) + "\n")
    else:
        if raw_path.exists() or results_path.exists() or xlsx_path.exists():
            raise SystemExit("Evaluation output already exists; use --resume rather than issue duplicate API requests")
        raw_path.touch(exist_ok=False)
        result_rows, existing_ids = [], set()
    started = datetime.now(timezone.utc)
    attempts_this_run = 0
    with raw_path.open("a", encoding="utf-8") as raw_stream:
        for number, row in enumerate(sample.itertuples(index=False), 1):
            original = sample.iloc[number - 1]
            if original["Test ID"] in existing_ids:
                continue
            if args.max_requests is not None and attempts_this_run >= args.max_requests:
                break
            recorder.responses.last_usage, recorder.responses.last_model = {}, None
            start = perf_counter()
            # itertuples renames headers containing spaces, so use the original positional mapping.
            result_row = {
                "Test ID": original["Test ID"], "Dataset Source": original["Dataset Source"],
                "Original Review": original["Original Review"], "Ground Truth Label": original["Ground Truth Label"],
                "Model Prediction": "", "Correct / Incorrect": "", "Confusion Matrix Category": "",
                "OpenAI Model Used": settings.openai_model, "Input Tokens": "", "Output Tokens": "",
                "Evaluation Status": "",
            }
            raw_event = {
                "test_id": original["Test ID"], "dataset_source": original["Dataset Source"],
                "record_id": original["id"], "ground_truth": original["Ground Truth Label"],
                "configured_model": settings.openai_model,
            }
            try:
                response = runtime.analysis.analyze(original["Original Review"], ground_truth=original["Ground Truth Label"])
                if response.provider != "openai":
                    raise RuntimeError(f"Unexpected non-OpenAI provider: {response.provider}")
                prediction = response.sentiment
                result_row.update({
                    "Model Prediction": prediction,
                    "Correct / Incorrect": "Correct" if prediction == original["Ground Truth Label"] else "Incorrect",
                    "Confusion Matrix Category": category(original["Ground Truth Label"], prediction),
                    "OpenAI Model Used": recorder.responses.last_model or settings.openai_model,
                    "Input Tokens": usage_value(recorder.responses.last_usage, "input_tokens") or "",
                    "Output Tokens": usage_value(recorder.responses.last_usage, "output_tokens") or "",
                    "Evaluation Status": "completed",
                })
                raw_event.update({
                    "status": "completed", "prediction": prediction, "provider": response.provider,
                    "confidence": response.confidence, "scores": response.scores.model_dump(), "language": response.language,
                    "reason": response.reason, "topic": response.topic.model_dump(),
                    "retrieved_example_ids": [item.id for item in response.rag.retrieved_examples],
                    "usage": recorder.responses.last_usage, "response_model": recorder.responses.last_model,
                })
            except Exception as exc:  # preserve failures; never convert to a local classification
                result_row["Evaluation Status"] = f"failed: {type(exc).__name__}: {str(exc)}"
                result_row["Input Tokens"] = usage_value(recorder.responses.last_usage, "input_tokens") or ""
                result_row["Output Tokens"] = usage_value(recorder.responses.last_usage, "output_tokens") or ""
                raw_event.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                  "usage": recorder.responses.last_usage, "response_model": recorder.responses.last_model})
            raw_event["elapsed_seconds"] = round(perf_counter() - start, 4)
            raw_stream.write(json.dumps(raw_event, ensure_ascii=False) + "\n")
            raw_stream.flush()
            result_rows.append(result_row)
            existing_ids.add(original["Test ID"])
            attempts_this_run += 1
            write_csv(results_path, result_rows)
            print(f"{len(result_rows)}/100 {original['Test ID']}: {result_row['Evaluation Status']}", flush=True)

    if len(result_rows) < 100:
        print(json.dumps({"status": "partial", "saved_attempts": len(result_rows),
                          "remaining": 100 - len(result_rows), "attempts_this_run": attempts_this_run}, indent=2))
        return

    write_xlsx(xlsx_path, result_rows)
    complete = [row for row in result_rows if row["Evaluation Status"] == "completed"]
    input_tokens = sum(int(row["Input Tokens"] or 0) for row in result_rows)
    output_tokens = sum(int(row["Output Tokens"] or 0) for row in result_rows)
    source_names = sample["Dataset Source"].drop_duplicates().tolist()
    overall = binary_metrics([row["Ground Truth Label"] for row in complete], [row["Model Prediction"] for row in complete]) if complete else {
        "total_evaluated_reviews": 0, "metrics_unavailable": "No completed OpenAI responses"
    }
    report = {
        "evaluation_id": manifest["evaluation_id"], "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "configured_model": settings.openai_model,
        "provider": "openai", "selected_reviews": len(result_rows), "completed_reviews": len(complete),
        "failed_reviews": len(result_rows) - len(complete), "token_usage": {
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
        },
        "actual_cost_usd": round(input_tokens * INPUT_RATE + output_tokens * OUTPUT_RATE, 6),
        "pricing": {"input_usd_per_million": 2.00, "output_usd_per_million": 12.00},
        "metrics_by_dataset": {source: source_metrics(result_rows, source) for source in source_names},
        "overall_metrics": overall,
        "limitations": [
            "Metrics exclude failed OpenAI requests; failures are preserved in the raw and tabular exports.",
            "All predictions are fresh requests from the configured OpenAI model; no prior saved predictions were reused.",
            "This is not independently human-validated.",
        ],
    }
    write_json(output / "evaluation_results.json", report)
    lines = ["# MULTIVENT — 100-Review OpenAI Evaluation", "", f"Model: `{settings.openai_model}`", "",
             f"Completed: {len(complete)}/100; failed: {len(result_rows) - len(complete)}.",
             f"Token usage: {input_tokens:,} input; {output_tokens:,} output; {input_tokens + output_tokens:,} total.",
             f"Estimated actual cost: ${report['actual_cost_usd']:.6f} USD.", "",
             "| Dataset | Evaluated | Accuracy | Positive Precision | Positive Recall | Positive F1 | Macro F1 | Correct | Incorrect |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source, metrics in report["metrics_by_dataset"].items():
        if metrics["total_evaluated_reviews"]:
            lines.append("| {source} | {n} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {macro:.4f} | {correct} | {incorrect} |".format(
                source=source, n=metrics["total_evaluated_reviews"], accuracy=metrics["accuracy"],
                precision=metrics["positive_precision"], recall=metrics["positive_recall"],
                f1=metrics["positive_f1_score"], macro=metrics["macro_f1_score"],
                correct=metrics["correct_predictions"], incorrect=metrics["incorrect_predictions"],
            ))
        else:
            lines.append(f"| {source} | 0 | — | — | — | — | — | — | — |")
    if overall["total_evaluated_reviews"]:
        lines += ["", "## Overall", "", f"Accuracy {overall['accuracy']:.4f}; positive F1 {overall['positive_f1_score']:.4f}; macro F1 {overall['macro_f1_score']:.4f}.",
                  f"Confusion matrix (actual negative/positive × predicted negative/positive): `{overall['confusion_matrix']}`."]
    (output / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["status"] = "completed"
    manifest["execution_report"] = "evaluation_results.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"completed": len(complete), "failed": len(result_rows) - len(complete),
                      "input_tokens": input_tokens, "output_tokens": output_tokens,
                      "actual_cost_usd": report["actual_cost_usd"]}, indent=2))


if __name__ == "__main__":
    main()
