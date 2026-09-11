# Recorded validation — 2026-09-07

Environment: Windows, CPython 3.13.5, CPU inference. Exact runtime dependencies are
recorded in `requirements-lock.txt`; optional browser tooling is in `requirements-dev.txt`.

## Data and artifacts

- Downloaded `jjjardev/hilisenti-v1` revision
  `27ae2471f1b3c6806b5da7ba532b42e6e3dcb84e`.
- Normal `datasets` loading reproduced the upstream quoted-label
  `DatasetGenerationError`. Hub CSV fallback completed successfully.
- Original rows: **23,337**. Clean rows: **22,233**.
- Clean train / validation / test: **17,805 / 2,201 / 2,227**.
- Removed duplicates: **1,104**, each retained in the duplicate audit CSV.
- Unrecoverable malformed rows: **0**. Missing sentence/label values: **0**.
- Fitted **10 LDA topics** on training text only; saved vectorizer, model, and metadata.
- Built **17,805 training-only FAISS references**, 384 dimensions each.
- Local encoder revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.

## Checks executed

- `python -m pytest`: **37 passed**. Includes quoted labels, malformed/multiline CSV,
  deduplication, held-out conflicts, LDA inference/persistence, cosine retrieval,
  index split rejection at build/load, mislabeled training-text leakage rejection,
  API validation/missing artifacts, provider selection, logs, and mocked Responses API.
- `python -m pip check`: no broken requirements.
- `node --check app/static/app.js`: passed.
- Live `/health`: status `ok`, LDA and RAG loaded, provider `rag_weighted_fallback`,
  `openai_configured=false`.
- Fresh runtime loading and analysis succeeded with `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, and socket connections explicitly blocked; five actual
  local references were returned without any network access.
- `python scripts/smoke_browser.py`: real headless Chrome submission and rendered
  references passed; mobile overflow and text-injection rendering checks passed.
  Screenshots: `reports/ui-desktop.png` and `reports/ui-mobile.png`.

Two upstream test-library deprecation warnings were reported (Starlette's httpx
adapter and AnyIO's portal alias); neither caused a failure. FAISS used its available
standard CPU build when the optional AVX2 Python module was absent. Windows Hub
symlink warnings did not prevent downloads. A tokenizer length warning during the
initial run referred to intentional full tokenization before chunking; the code now
suppresses that warning at tokenization without changing chunking or vectors.

## Actual held-out evaluation

Executed `python scripts/evaluate.py` on **all 2,227 cleaned test rows** with seed 42,
K=5, training-only RAG, PII sanitization disabled, and the actual local encoder.
No OpenAI requests were made. All leakage checks passed.

| Metric | Result |
|---|---:|
| Accuracy | 0.677593 |
| Macro precision | 0.675194 |
| Macro recall | 0.670653 |
| Macro F1 | 0.671196 |

Confusion matrix, rows=actual and columns=predicted:

| | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 547 | 124 | 145 |
| Neutral | 177 | 357 | 99 |
| Positive | 112 | 61 | 605 |

Complete evaluation took about 69 seconds after model loading on this machine.
Machine-readable results, artifact hashes, and predictions are in `reports/`.

## Practical limitations

This is a provisional neighbor-voting baseline. Hiligaynon/Tagalog semantic coverage
of the selected encoder is unverified; inspect retrieved references. The supplied
caterer-delay/food-praise example returned **positive** with about **59.6% vote share**
in the browser check, illustrating that mixed sentiment and semantic similarity can
mislead this fallback. No rule was added to force the example's desired label.

LDA's conservative empty stopword lists leave frequent particles prominent in topic
terms. Topics remain honestly unassigned. Native-speaker review and validation-only
experiments should precede human naming or model-selection claims. No automatic
language identification is implemented. Vote shares are not calibrated probabilities.

The source test split was deduplicated, so these metrics are not directly comparable
to the author's published classifier benchmark. Event-booking performance needs a
separate, representative evaluation. OpenAI's adapter was tested with a mocked SDK
response only; live API authentication, billing, and model behavior remain untested
because no key was supplied. PII review is still required before redistribution.
