"""Generated tables distinguish source reviews from expanded reference units."""
from pathlib import Path

import pandas as pd


def distribution(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].str.split("|").explode() if column == "aspect" else frame[column]
    return values.value_counts().to_dict()


def write_reports(directory: Path, manifest: dict, frames: dict[str, pd.DataFrame]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# Multivent binary dataset report", "",
             f"Total source records examined: **{manifest['total_source_records_examined']:,}**.",
             f"Neutral rows excluded: **{manifest['neutral_excluded']:,}** (never relabeled).", "",
             "Original rows count source reviews/sentences. Positive/negative count accepted binary candidate units before deduplication. A hotel review supplies up to two units. Other exclusion counts are candidate units, not necessarily whole reviews.", "",
             "| Dataset | Original rows | Positive candidates | Negative candidates | Neutral excluded | Malformed excluded | Domain excluded | Duplicates removed | Relevant accepted | RAG |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source in manifest["sources"]:
        s = manifest["source_stats"][source["id"]]
        values = [s[k] for k in ("original_rows", "positive", "negative", "neutral_excluded", "malformed_excluded", "domain_irrelevant_excluded", "duplicates_removed", "final_accepted", "rag_rows")]
        lines.append(f"| {source['name']} | " + " | ".join(str(v) for v in values) + " |")
    lines += ["", "Relevant accepted includes all splits; test and validation records are kept separately. Domain exclusions are not applied to evaluation: evaluations include all deduplicated valid binary test units.", ""]
    for source in manifest["sources"]:
        s = manifest["source_stats"][source["id"]]
        lines += [f"### {source['name']}: label/schema exclusions", ""]
        lines += [f"- {reason}: {count}" for reason, count in s["exclusion_reasons"].items()]
        lines += [f"- Neutral aspect annotations observed: {s['neutral_aspects_ignored']} (separate from neutral-only rows).", ""]
    for name in ("domain_filtered.csv", "rag.csv", "lda_train.csv", "evaluation_hilisenti.csv", "evaluation_domain.csv"):
        frame = frames[name]
        lines += [f"## {name}: {len(frame):,} rows", ""]
        for column in ("sentiment", "language", "source_domain", "aspect", "source_dataset"):
            lines += [f"**{column}**: " + "; ".join(f"{key}={value}" for key, value in distribution(frame, column).items()), ""]
    rag_lang = distribution(frames["rag.csv"], "language")
    lines += ["Language counts use source metadata, not language detection. HiliSenti is reported as hiligaynon_code_switched; separate Hiligaynon-only vs Filipino/Tagalog vs English counts within HiliSenti are unavailable and must not be fabricated.",
              f"Known English RAG rows: {rag_lang.get('english', 0)}; Hiligaynon/code-switched rows: {rag_lang.get('hiligaynon_code_switched', 0)}; separately tagged Tagalog rows: {rag_lang.get('tagalog', 0)} (does not mean no Tagalog code-switching).", "",
              "## Training-only threshold comparison", "",
              "These untuned heuristic thresholds do not use test metrics for selection.", "",
              "| Threshold | Training rows | Positive | Negative | Sources |", "|---|---:|---:|---:|---|"]
    for row in manifest["threshold_statistics"]:
        lines.append(f"| {row['threshold']:.2f} | {row['train_rows']} | {row['positive']} | {row['negative']} | {row['sources']} |")
    lines += ["", "## Isolation and balancing", "",
              "Domain review groups are assigned deterministically to test/validation/train before domain filtering or class balancing. Hotel positive and negative fields remain in the same group. HiliSenti keeps its predefined splits. Exact duplicates use test > validation > train priority; contradictory labels are quarantined. All HiliSenti test fingerprints, including neutral text, are protected. Index construction fails on any remaining evaluation collision.", "",
              "RAG balances positive/negative within each source with seed 42 by default. Excess rows remain in all_binary.csv and balance_excluded.csv. LDA uses the full relevant training corpus, independent of sentiment labels. Aspect labels come only from supplied MAMS categories; otherwise they are general.", ""]
    (directory / "multivent_dataset_report.md").write_text("\n".join(lines), encoding="utf-8")
    sources_lines = ["# Dataset sources and attribution", "", "Academic, non-commercial experiment. Each record retains its own source license; the combined corpus is not relicensed under a blanket permissive license.", ""]
    for source in manifest["sources"]:
        stats = manifest["source_stats"][source["id"]]
        sources_lines += [f"## {source['name']}", "", f"- Author/organization: {source['author']}",
                          f"- Published dataset: {source['url']}", f"- Original project: {source.get('original_url') or source['url']}",
                          f"- DOI: {source.get('doi') or 'Not supplied'}", f"- License: {source['license']} ({source['license_url']})",
                          f"- Citation: {source['citation']}", f"- Purpose: {source['domain']}; {source['pool']} retrieval pool",
                          f"- Records used: RAG {stats['rag_rows']}; LDA {stats['lda_rows']}; held-out evaluation {stats['evaluation_rows']}",
                          "- Modifications: NFC/whitespace normalization; source-defined binary mapping; neutral/invalid/conflicting rows excluded with audit; exact deduplication; frozen-embedding domain scoring; split reservation; deterministic RAG balancing.",
                          f"- Specific processing/limitations: {source['notes']}", ""]
    sources_lines += ["Hotel fields provide weak sentiment labels and may still contain mixed or incorrectly placed statements. MAMS retained rows have only one non-neutral polarity; no invented majority label is assigned. UCI sentences lack review-level IDs, so exact text is protected but paraphrase/shared-review leakage cannot be excluded. Hotel-derived domain evaluation is a transferable hospitality proxy, not an independently annotated Multivent event benchmark. Names/locations may remain; review PII and source rights before redistribution."]
    (directory / "dataset_sources.md").write_text("\n".join(sources_lines), encoding="utf-8")
