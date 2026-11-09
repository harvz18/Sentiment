# Informal text normalization sources

This project normalizes informal feedback only to improve LDA topic inference and
embedding/RAG retrieval. Normalization does not translate text, assign sentiment, or
replace the existing LDA + RAG architecture. Customer input remains available as
`text_original`; the derived form is returned as `text_normalized`.

## FiliSenti

- Author: Jessie James T. Jarder
- Repository: https://github.com/jjjardev/filisenti
- License: Apache License 2.0 (repository `LICENSE`)
- Files reviewed: `app.py`, `scripts/FiliSenti.py`, and
  `filisenti_app/lib/services/text_normalizer.dart`
- Use here: selected documented Tagalog/Hiligaynon shortcut mappings plus the concepts
  of NFKC normalization, laughter normalization, repeated-character limiting, numeric
  repetition, and hyphenated reduplication.
- Integration method: adapted and independently structured for this project. FiliSenti's
  classifier, tokenizer, model weights, and sentiment logic were not copied or used.

Loaded FiliSenti mappings:

| Shortcut | Normalized | Dictionary |
|---|---|---|
| `waay` | `wala` | Hiligaynon |
| `way` | `wala` | Hiligaynon |
| `ndi` | `indi` | Hiligaynon |
| `gd` | `gid` | Hiligaynon |
| `gud` | `gid` | Hiligaynon |
| `mn` | `man` | Hiligaynon |
| `tni` | `tani` | Hiligaynon |
| `tne` | `tani` | Hiligaynon |
| `sbng` | `subong` | Hiligaynon |
| `krn` | `karon` | Hiligaynon |
| `pro` | `pero` | Hiligaynon |
| `kg` | `kag` | Hiligaynon |
| `kng` | `kon` | Hiligaynon |
| `kun` | `kon` | Hiligaynon |
| `wla` | `wala` | Filipino |
| `lng` | `lang` | Filipino |
| `nlng` | `nalang` | Filipino |
| `na lng` | `nalang` | Filipino |
| `dko` | `di ko` | Filipino |
| `gnito` | `ganito` | Filipino |
| `gnyan` | `ganyan` | Filipino |
| `gnun` | `ganun` | Filipino |
| `sakin` | `sa akin` | Filipino |
| `pru` | `pero` | Filipino |
| `kc` | `kasi` | Filipino |
| `ksi` | `kasi` | Filipino |
| `nman` | `naman` | Filipino |
| `nmn` | `naman` | Filipino |
| `sya` | `siya` | Filipino |
| `nya` | `niya` | Filipino |
| `ikw` | `ikaw` | Filipino |
| `dpt` | `dapat` | Filipino |
| `dba` | `di ba` | Filipino |
| `diba` | `di ba` | Filipino |

Some shorter entries present in FiliSenti were intentionally not loaded because they
are especially ambiguous without contextual disambiguation. Dictionary replacement is
case-insensitive but token-boundary aware; casing style is reconstructed in the output.

## Filipino Slang Norm

- SEACrowd organization: https://huggingface.co/datasets/SEACrowd/filipino_slang_norm
- Original authors: Lorenzo Jaime Flores and Dragomir Radev
- Original repository:
  https://github.com/ljyflores/efficient-spelling-normalization-filipino
- Paper: https://aclanthology.org/2022.sustainlp-1.5
- Declared license: **Unknown** in both the SEACrowd loader and Hugging Face dataset
  card. The original repository has no root license covering the normalization data.
- Published source schema: `id`, `src_sent`, `norm_sent`; SEACrowd also exposes a
  text-to-text projection (`text_1`, `text_2`).
- Splits observed in the original source: 303 train lines and 100 test lines (403 total).
  This conflicts with the dataset card description of 398 items and should be resolved
  with the maintainers before relying on a record count.
- Use here: documentation/reference only. **No records or mappings were copied,
  redistributed, or loaded** because permission is unclear.

## Multivent custom dictionary

The manually reviewed project dictionary is
`app/preprocessing/dictionaries/multivent_custom.json`. It currently contains:

| Shortcut | Normalized |
|---|---|
| `nmi` | `nami` |
| `lg` | `lang` |

Additions to this file require manual review. Unknown or ambiguous abbreviations remain
unchanged rather than being guessed.
