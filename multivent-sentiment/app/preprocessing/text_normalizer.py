"""Auditable normalization for informal Filipino and Hiligaynon feedback.

Dictionary replacements are token-boundary aware and never infer sentiment. The
structural rules are conservative adaptations of FiliSenti's Apache-2.0 normalizer.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata


DICTIONARY_DIR = Path(__file__).with_name("dictionaries")
DICTIONARY_SOURCES = (
    ("multivent_custom.json", "multivent_custom"),
    ("hiligaynon.json", "filisenti"),
    ("filipino.json", "filisenti"),
)


@dataclass(frozen=True)
class NormalizationChange:
    original: str
    normalized: str
    source: str


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    normalized: str
    changes: tuple[NormalizationChange, ...]


@dataclass(frozen=True)
class _Mapping:
    shortcut: str
    normalized: str
    source: str


def _read_dictionary(filename: str) -> dict[str, str]:
    path = DICTIONARY_DIR / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Normalization dictionary must be a nonempty object: {path}")
    result: dict[str, str] = {}
    for shortcut, normalized in value.items():
        if not isinstance(shortcut, str) or not isinstance(normalized, str):
            raise ValueError(f"Normalization mappings must contain strings: {path}")
        shortcut, normalized = shortcut.strip().casefold(), normalized.strip()
        if not shortcut or not normalized:
            raise ValueError(f"Normalization mappings cannot be blank: {path}")
        result[shortcut] = normalized
    return result


@lru_cache(maxsize=1)
def loaded_mappings() -> tuple[_Mapping, ...]:
    """Load reviewed dictionaries once, rejecting conflicting shortcut meanings."""
    combined: dict[str, tuple[str, str]] = {}
    for filename, source in DICTIONARY_SOURCES:
        for shortcut, normalized in _read_dictionary(filename).items():
            previous = combined.get(shortcut)
            if previous and previous[0].casefold() != normalized.casefold():
                raise ValueError(f"Conflicting normalization for {shortcut!r}")
            combined.setdefault(shortcut, (normalized, source))
    # Long and multi-token entries run first so a shorter entry cannot consume them.
    ordered = sorted(combined.items(), key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]))
    return tuple(_Mapping(key, value, source)
                 for key, (value, source) in ordered)


def mappings_by_source() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {"filisenti": {}, "filipino_slang_norm": {},
                                         "multivent_custom": {}}
    for mapping in loaded_mappings():
        result[mapping.source][mapping.shortcut] = mapping.normalized
    return result


@lru_cache(maxsize=1)
def _combined_mapping_pattern() -> re.Pattern[str]:
    alternatives = [re.escape(mapping.shortcut).replace(r"\ ", r"\s+")
                    for mapping in loaded_mappings()]
    return re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", re.IGNORECASE)


def _match_case(replacement: str, original: str) -> str:
    letters = "".join(character for character in original if character.isalpha())
    if letters and letters.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_with_audit(text: str, pattern: re.Pattern[str], replacement,
                        source: str, changes: list[NormalizationChange]) -> str:
    def apply(match: re.Match[str]) -> str:
        value = replacement(match) if callable(replacement) else replacement
        if value != match.group(0):
            changes.append(NormalizationChange(match.group(0), value, source))
        return value

    return pattern.sub(apply, text)


def normalize_informal_text(text: str) -> NormalizationResult:
    """Normalize informal text without translation, sentiment rules, or substring edits."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    original = text
    changes: list[NormalizationChange] = []

    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        changes.append(NormalizationChange(text, normalized, "unicode_nfkc"))

    whitespace = " ".join(normalized.split())
    if whitespace != normalized:
        changes.append(NormalizationChange(normalized, whitespace, "whitespace_cleanup"))
    normalized = whitespace

    mapping_lookup = {mapping.shortcut: mapping for mapping in loaded_mappings()}

    def expand_mapping(match: re.Match[str]) -> str:
        key = " ".join(match.group(0).casefold().split())
        mapping = mapping_lookup[key]
        value = _match_case(mapping.normalized, match.group(0))
        if value != match.group(0):
            changes.append(NormalizationChange(match.group(0), value, mapping.source))
        return value

    normalized = _combined_mapping_pattern().sub(expand_mapping, normalized)

    # FiliSenti-compatible, but limited to letter runs so punctuation, digits and emoji survive.
    repeated_letters = re.compile(r"([^\W\d_])\1{2,}")
    normalized = _replace_with_audit(
        normalized, repeated_letters, lambda match: match.group(1) * 2,
        "filisenti_rule:repeated_character", changes,
    )

    two_repeat = re.compile(r"(?<!\w)([^\W\d_]{3,})2(?!\w)")
    normalized = _replace_with_audit(
        normalized, two_repeat, lambda match: f"{match.group(1)} {match.group(1)}",
        "filisenti_rule:numeric_reduplication", changes,
    )

    reduplication = re.compile(r"(?<!\w)([^\W\d_]+)-\1(?!\w)")
    normalized = _replace_with_audit(
        normalized, reduplication, lambda match: f"{match.group(1)} {match.group(1)}",
        "filisenti_rule:hyphen_reduplication", changes,
    )

    for syllable, canonical in (("ha", "hahaha"), ("he", "hehehe")):
        laughter = re.compile(rf"(?<!\w)(?:{syllable}){{2,}}{syllable[0]}*(?!\w)", re.IGNORECASE)
        normalized = _replace_with_audit(
            normalized, laughter,
            lambda match, value=canonical: _match_case(value, match.group(0)),
            "filisenti_rule:laughter", changes,
        )

    return NormalizationResult(original, normalized, tuple(changes))
