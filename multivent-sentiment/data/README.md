# HiliSenti attribution and data handling

- Dataset: **HiliSenti v1**
- Author: **Jessie James T. Jarder**
- Dataset URL: https://huggingface.co/datasets/jjjardev/hilisenti-v1
- Permanent DOI: https://doi.org/10.57967/hf/8737
- License: **CC BY-NC-SA 4.0**
- Purpose here: non-commercial academic BSIT capstone research for Multivent.

The dataset and original sentiment labels belong to their credited creators. This
project prepares the supplied labels; it does not claim authorship of them. Preserve
attribution, non-commercial restrictions, and ShareAlike conditions when sharing
adaptations. Review the source dataset card and license before redistribution or
future commercial deployment. The repository snapshot revision and source SHA-256
checksums are in `raw/hilisenti/manifest.json`.

Original Hub CSVs are copied unchanged into `raw/hilisenti/source/`. Never edit them.
The downloader first tries `datasets.load_dataset`; the upstream quoted label issue
can cause `DatasetGenerationError`. The fallback downloads the original Hub files.
Both paths use the same auditable CSV cleaning, including quote-normalized labels.
Running the downloader again verifies the existing snapshot without overwriting it.
For a new upstream snapshot, archive this entire experiment and use a fresh project
data directory so the old artifacts remain reproducible.

The cleaner frames CSV records with Python's strict CSV parser and constructs pandas
frames. It preserves commas and multiline quoted text, and writes structural errors
and invalid labels to `processed/rejected_rows.csv` with reasons and source locations.
An unclosed quote can cause the remainder of the file to be quarantined together;
inspect the raw record before attempting manual recovery. No malformed text is guessed.

`processed/hilisenti_clean.csv` contains stable source-row IDs, `sentence_original`,
`sentence_clean`, `label`, `sentiment`, `split`, `source`, `text_hash`, `source_file`,
and `source_row`. Labels are `0=negative`, `1=neutral`, `2=positive`.

Normalization is NFC plus whitespace collapse/trim. No translation, stemming,
spelling correction, punctuation removal, abbreviation expansion, or particle removal
is applied. Case is retained in the dataset and for embeddings/providers. Duplicate
identity uses casefolded normalized text. Original rows remain in the raw snapshot.

Duplicate handling reserves text in test first, validation second, train last. No row
changes split. Within each retained split, identical text with the same label is kept
once; conflicting labels are quarantined. A malformed held-out label still reserves
its text from lower-priority splits. `duplicate_rows.csv` lists every removed duplicate.
The report counts missing sentence/label cells and raw/clean rows and labels. Counts
refer to logical CSV records; a structurally broken multiline record counts once.
This cleaned evaluation set can differ from the author's original benchmark.

PII sanitization defaults to **false** for controlled evaluation. When enabled, it
masks recognizable email addresses and Philippine mobile numbers in normalized text,
references, provider input, and feedback logs. It is deliberately limited: it does
not automatically identify names or locations and is **not full anonymization**.
Original text remains in raw/processed audit files and API response `feedback`.
Review PII before redistributing any raw or derived datasets, indexes, or logs.
Changing this setting requires cleaning, LDA fitting, and index building again.

Stopword files start empty. Each accepts one native-speaker-reviewed word per line
and `#` comments. Stopwords affect only LDA, never embeddings or sentiment input.
LDA lowercases tokens and uses Unicode word tokens (including single characters);
punctuation is omitted from its count features only. Preserve negators such as
`indi`, `wala`, and `not`, and sentiment-bearing particles and words.
