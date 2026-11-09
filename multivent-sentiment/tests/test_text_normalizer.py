import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.preprocessing.text_normalizer import (mappings_by_source,
                                                normalize_informal_text)


@pytest.mark.parametrize(("original", "expected"), [
    ("Nmi gd ila service", "Nami gid ila service"),
    ("Wla gd ko nalipay", "Wala gid ko nalipay"),
    ("Okay mn pero dugay lg", "Okay man pero dugay lang"),
    ("Ndi nami ila service", "Indi nami ila service"),
    ("sbng lg", "subong lang"),
    ("nami gid", "nami gid"),
    ("wala gid", "wala gid"),
])
def test_approved_normalizations(original, expected):
    assert normalize_informal_text(original).normalized == expected


def test_punctuation_uppercase_code_switching_and_emoji_are_preserved():
    result = normalize_informal_text("NMI GD ang FOOD!!! Great venue 😍😍😍")
    assert result.normalized == "NAMI GID ang FOOD!!! Great venue 😍😍😍"


def test_structural_rules_are_conservative_and_audited():
    result = normalize_informal_text("sobraaaang saya hahaha ganda-ganda sulit2")
    assert result.normalized == "sobraang saya hahaha ganda ganda sulit sulit"
    assert {change.source for change in result.changes} == {
        "filisenti_rule:repeated_character",
        "filisenti_rule:hyphen_reduplication",
        "filisenti_rule:numeric_reduplication",
    }


def test_unknown_slang_and_token_substrings_remain_unchanged():
    original = "xyz gdg badge sbngx unknown_sl4ng"
    assert normalize_informal_text(original).normalized == original
    assert normalize_informal_text("gd, (lg)!").normalized == "gid, (lang)!"


def test_nfkc_and_whitespace_cleanup():
    result = normalize_informal_text("  Ｎｍｉ\tｇｄ  ")
    assert result.normalized == "Nami gid"
    assert result.changes[0].source == "unicode_nfkc"
    assert any(change.source == "whitespace_cleanup" for change in result.changes)


def test_mapping_provenance_excludes_unlicensed_dataset():
    mappings = mappings_by_source()
    assert mappings["multivent_custom"] == {"lg": "lang", "nmi": "nami"}
    assert mappings["filisenti"]["gd"] == "gid"
    assert mappings["filisenti"]["wla"] == "wala"
    assert mappings["filipino_slang_norm"] == {}


def test_normalize_endpoint_and_analysis_audit(runtime):
    with TestClient(create_app(runtime=runtime)) as client:
        normalized = client.post("/normalize", json={"text": "Nmi gd ila service pero dugay lg"})
        assert normalized.status_code == 200
        assert normalized.json()["normalized"] == "Nami gid ila service pero dugay lang"
        assert normalized.json()["changes"] == ["Nmi → Nami", "gd → gid", "lg → lang"]

        analyzed = client.post("/analyze", json={"feedback": "Nmi gd ila service"})
        assert analyzed.status_code == 200
        payload = analyzed.json()
        assert payload["feedback"] == payload["text_original"] == "Nmi gd ila service"
        assert payload["text_normalized"] == "Nami gid ila service"
        assert payload["normalizations"][0] == {
            "original": "Nmi", "normalized": "Nami", "source": "multivent_custom"
        }
