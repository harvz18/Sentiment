"""Auditable CSV ingestion, label normalization, and held-out split protection."""
import csv
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.artifacts import read_json, write_json
from app.preprocessing.text_cleaner import normalize_text, prepare_text, text_fingerprint

LABELS = {0: "negative", 1: "neutral", 2: "positive"}
SOURCE = "HiliSenti v1"
SPLITS = {"train", "validation", "test"}
COLUMNS = ["id", "sentence_original", "sentence_clean", "label", "sentiment",
           "split", "source", "text_hash", "source_file", "source_row"]


def normalize_label(value: Any) -> int:
    """Accept integers and quote/whitespace-wrapped digits, never truncate floats."""
    if value is None or isinstance(value, bool):
        raise ValueError("Missing or invalid label")
    label = str(value).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in "\"'":
        label = label[1:-1].strip()
    if label not in {"0", "1", "2"}:
        raise ValueError("Label must be 0, 1, or 2")
    return int(label)


def read_source_csv(path: Path, split: str) -> tuple[pd.DataFrame, list[dict]]:
    """Frame valid CSV records with pandas; quarantine structural errors with raw text.

    Strict csv parsing handles commas and multiline quoted sentences. An unclosed
    quote can consume the remainder of a file; that complete text is retained.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    reader = csv.reader(io.StringIO("".join(lines)), strict=True)
    header = next(reader, None)
    if header != ["sentence", "label"]:
        raise ValueError(f"Unexpected header in {path}: {header!r}; expected sentence,label")
    rows, rejected = [], []
    while True:
        start = reader.line_num
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            rejected.append({"split": split, "source_file": path.name,
                             "source_row": start + 1, "reason": f"CSV syntax: {exc}",
                             "raw_record": "".join(lines[start:reader.line_num])})
            continue
        record = {"split": split, "source_file": path.name, "source_row": start + 1}
        if len(row) != 2:
            rejected.append({**record, "reason": f"Expected 2 fields, got {len(row)}",
                             "raw_record": "".join(lines[start:reader.line_num])})
        else:
            rows.append({**record, "sentence": row[0], "label": row[1]})
    return pd.DataFrame(rows, columns=["split", "source_file", "source_row", "sentence", "label"]), rejected


def clean_frames(frames: list[pd.DataFrame], pii: bool = False,
                 structural_rejects: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Retain held-out duplicates first; quarantine within-split label conflicts.

    Split priority is test > validation > train, without relocating any record.
    Test fingerprints are also blocked even if their labels are malformed.
    """
    raw = pd.concat(frames, ignore_index=True)
    rejected = list(structural_rejects or [])
    records: list[dict] = []
    missing = {"sentence": 0, "label": 0}
    reserved: dict[str, int] = {}
    priority = {"test": 0, "validation": 1, "train": 2}
    for row in raw.to_dict("records"):
        text, split = row.get("sentence"), row.get("split")
        if isinstance(text, str) and normalize_text(text) and split in priority:
            key = text_fingerprint(prepare_text(text, pii))
            reserved[key] = min(reserved.get(key, 3), priority[split])
    for index, row in enumerate(raw.to_dict("records")):
        sentence, label = row.get("sentence"), row.get("label")
        for key, value in (("sentence", sentence), ("label", label)):
            if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
                missing[key] += 1
        try:
            if not isinstance(sentence, str) or not normalize_text(sentence):
                raise ValueError("Missing/empty sentence")
            if row.get("split") not in SPLITS:
                raise ValueError("Unknown dataset split")
            numeric = normalize_label(label)
            cleaned = prepare_text(sentence, pii)
            records.append({"id": f"hilisenti:{row['split']}:{row.get('source_file', 'input')}:{row.get('source_row', index)}",
                            "sentence_original": sentence, "sentence_clean": cleaned,
                            "label": numeric, "sentiment": LABELS[numeric], "split": row["split"],
                            "source": SOURCE, "text_hash": text_fingerprint(cleaned),
                            "source_file": row.get("source_file", "input"),
                            "source_row": row.get("source_row", index)})
        except ValueError as exc:
            rejected.append({**row, "reason": str(exc), "raw_record": json.dumps(row, default=str, ensure_ascii=False)})
    valid = pd.DataFrame(records, columns=COLUMNS)
    duplicate_count = int(valid.duplicated("text_hash").sum())
    duplicate_rows: list[dict] = []
    kept: list[dict] = []
    for key, group in valid.groupby("text_hash", sort=False):
        eligible = group[group["split"].map(priority) == reserved[key]]
        lower = group[group["split"].map(priority) != reserved[key]]
        duplicate_rows.extend({**r, "reason": "Text reserved in held-out split"} for r in lower.to_dict("records"))
        if eligible["label"].nunique() > 1:
            rejected.extend({**r, "reason": "Conflicting labels within retained split"} for r in eligible.to_dict("records"))
            continue
        if not eligible.empty:
            kept.append(eligible.iloc[0].to_dict())
            duplicate_rows.extend({**r, "reason": "Duplicate normalized text"} for r in eligible.iloc[1:].to_dict("records"))
    clean = pd.DataFrame(kept, columns=COLUMNS)
    report = {"source": SOURCE, "total_rows": len(raw) + len(structural_rejects or []),
              "clean_rows": len(clean), "raw_rows_per_split": raw["split"].value_counts().to_dict(),
              "rows_per_split": clean["split"].value_counts().to_dict(),
              "sentiment_counts": {name: int((clean["sentiment"] == name).sum()) for name in LABELS.values()},
              "duplicates": duplicate_count, "duplicate_rows_removed": len(duplicate_rows),
              "malformed_rows": len(rejected), "missing_values": missing,
              "pii_sanitization": pii, "normalization": "NFC + whitespace; optional email/PH phone masking",
              "deduplication": "casefolded normalized text; test > validation > train; within-split conflicts quarantined"}
    return clean, pd.DataFrame(rejected, columns=sorted(set().union(*(r.keys() for r in rejected))) or ["reason"]), pd.DataFrame(duplicate_rows, columns=COLUMNS + ["reason"]), report


def load_clean(processed: Path) -> pd.DataFrame:
    path = processed / "hilisenti_clean.csv"
    if not path.exists():
        raise FileNotFoundError("Run python scripts/clean_dataset.py first")
    return pd.read_csv(path, keep_default_na=False)


def clean_download(raw_dir: Path, processed: Path, pii: bool) -> dict:
    manifest = read_json(raw_dir / "manifest.json")
    frames, rejects = [], []
    for item in manifest["files"]:
        frame, bad = read_source_csv(raw_dir / item["local_path"], item["split"])
        frames.append(frame)
        rejects.extend(bad)
    clean, malformed, duplicates, report = clean_frames(frames, pii, rejects)
    processed.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed / "hilisenti_clean.csv", index=False, encoding="utf-8")
    malformed.to_csv(processed / "rejected_rows.csv", index=False, encoding="utf-8")
    duplicates.to_csv(processed / "duplicate_rows.csv", index=False, encoding="utf-8")
    report["dataset_revision"] = manifest["revision"]
    write_json(processed / "dataset_report.json", report)
    if clean.empty or set(clean["split"]) != SPLITS:
        raise ValueError("Cleaning did not produce all three nonempty splits; inspect reports")
    return report
