"""Source-specific label interpretation; never infer labels from review keywords."""
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from app.corpus.registry import DatasetSource
from app.dataset import normalize_label, read_source_csv
from app.preprocessing.text_cleaner import normalize_text, prepare_text, text_fingerprint

CORPUS_COLUMNS = ["id", "text_original", "text_clean", "sentiment", "source_dataset",
                  "source_record_id", "source_domain", "language", "split", "aspect", "license",
                  "source_url", "text_hash", "original_text_hash", "source_name", "language_basis",
                  "source_split", "pool", "label_basis", "group_id"]
PLACEHOLDERS = {"", "\\n", "n/a", "na", "none", "nil", "null", "nan", "-", "--", ".", "..."}
NO_COMPLAINT = re.compile(r"^(?:nothing\b|no negatives?\b|no complaints?\b|nothing to dislike\b|"
                          r"not applicable\b|all (?:was |is )?(?:good|great|perfect)\b|"
                          r"everything (?:was |is )?(?:good|great|perfect)\b|niente\b)", re.I)
ASPECT_MAP = {"place": "venue", "menu": "food", "miscellaneous": "other"}


def read_review_csv(path: Path) -> tuple[list[dict], list[dict]]:
    """Preserve malformed logical CSV records instead of skipping bad lines silently."""
    lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    reader = csv.reader(io.StringIO("".join(lines)), strict=True)
    header = next(reader)
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV headers")
    records, rejected = [], []
    while True:
        start = reader.line_num
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            rejected.append({"source_record_id": str(start + 1), "reason": "malformed_csv",
                             "detail": str(exc), "raw_record": "".join(lines[start:reader.line_num])})
            continue
        if len(row) != len(header):
            rejected.append({"source_record_id": str(start + 1), "reason": "malformed_csv",
                             "detail": f"Expected {len(header)} fields; got {len(row)}",
                             "raw_record": "".join(lines[start:reader.line_num])})
        else:
            records.append({**dict(zip(header, row)), "_source_line": start + 1})
    return records, rejected


def normalized_record(source: DatasetSource, record_id: str, text: str, sentiment: str,
                      source_split: str, aspect: str, pii: bool, suffix: str = "",
                      label_basis: str = "source label", language: str | None = None) -> dict:
    cleaned = prepare_text(text, pii)
    if not cleaned or not any(c.isalnum() for c in cleaned):
        raise ValueError("Empty/meaningless text")
    return dict(zip(CORPUS_COLUMNS, [
        f"{source.id}:{record_id}{suffix}", text, cleaned, sentiment, source.id, record_id,
        source.domain, language or source.language_metadata, source_split, aspect, source.license,
        source.url, text_fingerprint(cleaned), text_fingerprint(text), source.name, source.language_basis,
        source_split, source.pool, label_basis, f"{source.id}:{record_id}",
    ]))


def adapt_source(source: DatasetSource, paths: list[Path], pii: bool = False) -> tuple[pd.DataFrame, list[dict], dict, set[str]]:
    records, exclusions = [], []
    protected: set[str] = set()
    examined = 0
    candidates = 0
    neutral_aspects = 0

    def exclude(record_id, reason, raw, detail=""):
        exclusions.append({"source_dataset": source.id, "source_record_id": str(record_id),
                           "reason": reason, "detail": detail,
                           "raw_record": json.dumps(raw, ensure_ascii=False, default=str)})

    if source.adapter == "hilisenti":
        for path in paths:
            split = {"train": "train", "val": "validation", "test": "test"}[path.stem]
            frame, rejected = read_source_csv(path, split)
            examined += len(frame) + len(rejected)
            candidates += len(frame) + len(rejected)
            for row in rejected:
                exclude(row["source_row"], "malformed_csv", row)
            for row in frame.to_dict("records"):
                record_id = f"{split}:{row['source_row']}"
                if split == "test" and normalize_text(row["sentence"]):
                    protected.add(text_fingerprint(prepare_text(row["sentence"], pii)))
                    protected.add(text_fingerprint(row["sentence"]))
                try:
                    label = str(normalize_label(row["label"]))
                    mapped = source.label_mapping[label]
                    if mapped == "exclude_neutral":
                        exclude(record_id, "neutral", row)
                        continue
                    records.append(normalized_record(source, record_id, row["sentence"], mapped,
                                                      split, "general", pii))
                except (ValueError, KeyError) as exc:
                    exclude(record_id, "malformed", row, str(exc))
    elif source.adapter == "mams":
        frame = pd.read_parquet(paths[0])
        if not {"text", "category", "polarity"}.issubset(frame.columns):
            raise ValueError("MAMS schema changed")
        examined = candidates = len(frame)
        for number, row in enumerate(frame.to_dict("records")):
            try:
                aspects = [s.strip().lower() for s in row["category"].split(",")]
                polarities = [s.strip().lower() for s in row["polarity"].split(",")]
                if len(aspects) != len(polarities) or not aspects or set(polarities) - set(source.label_mapping):
                    raise ValueError("Unaligned/unknown MAMS aspects and polarities")
                neutral_aspects += polarities.count("neutral")
                evaluative = set(polarities) - {"neutral"}
                if not evaluative:
                    exclude(number, "neutral", row)
                elif len(evaluative) > 1:
                    exclude(number, "conflicting_aspects", row, "No annotated overall dominant polarity; majority counting prohibited")
                else:
                    selected_aspects = sorted({ASPECT_MAP.get(a, a) for a, p in zip(aspects, polarities) if p != "neutral"})
                    records.append(normalized_record(source, str(number), row["text"], evaluative.pop(),
                                                      "train", "|".join(selected_aspects), pii,
                                                      label_basis="single non-neutral aspect polarity; neutral aspects ignored"))
            except (ValueError, TypeError, AttributeError) as exc:
                exclude(number, "malformed", row, str(exc))
    elif source.adapter == "booking":
        rows, bad = read_review_csv(paths[0])
        examined = len(rows) + len(bad)
        candidates = len(rows) * 2 + len(bad)
        for row in bad:
            exclude(row["source_record_id"], "malformed_csv", row)
        for row in rows:
            if not set(source.label_mapping).issubset(row) or "uniq_id" not in row:
                raise ValueError("Booking review schema changed")
            record_id = row["uniq_id"] or str(row["_source_line"])
            for field, sentiment in source.label_mapping.items():
                text = row[field]
                candidate_id = f"{record_id}:{field}"
                clean = normalize_text(text)
                if clean.casefold() in PLACEHOLDERS or not any(c.isalnum() for c in clean):
                    exclude(candidate_id, "empty_or_placeholder", {"field": field, "text": text})
                elif sentiment == "negative" and NO_COMPLAINT.search(clean):
                    # Do not relabel 'nothing to dislike' as negative or turn it positive.
                    exclude(candidate_id, "no_complaint_in_negative_field", {"field": field, "text": text})
                elif row.get("language", "en") not in {"en", "english"}:
                    exclude(candidate_id, "unreviewed_language", {"field": field, "text": text, "language": row.get("language")})
                else:
                    records.append(normalized_record(source, record_id, text, sentiment, "train", "general", pii,
                                                      suffix=f":{field}", label_basis=f"source field: {field}", language="english"))
    elif source.adapter == "uci":
        with zipfile.ZipFile(paths[0]) as archive:
            lines = archive.read(source.archive_member).decode("utf-8-sig").splitlines()
        examined = candidates = len(lines)
        for number, line in enumerate(lines, 1):
            try:
                text, label = line.rsplit("\t", 1)
                sentiment = source.label_mapping[label.strip()]
                records.append(normalized_record(source, str(number), text, sentiment, "train", "general", pii))
            except (ValueError, KeyError) as exc:
                exclude(number, "malformed", line, str(exc))
    frame = pd.DataFrame(records, columns=CORPUS_COLUMNS)
    if not frame.empty and set(frame.sentiment) - {"positive", "negative"}:
        raise ValueError("Adapter produced a non-binary label")
    stats = {"original_rows": examined, "candidate_units": candidates,
             "positive": int((frame.sentiment == "positive").sum()),
             "negative": int((frame.sentiment == "negative").sum()),
             "neutral_excluded": sum(r["reason"] == "neutral" for r in exclusions),
             "neutral_aspects_ignored": neutral_aspects,
             "malformed_excluded": sum(r["reason"].startswith("malformed") for r in exclusions),
             "exclusion_reasons": pd.Series([r["reason"] for r in exclusions], dtype=str).value_counts().to_dict()}
    return frame, exclusions, stats, protected
