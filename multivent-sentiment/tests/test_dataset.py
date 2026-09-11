import pandas as pd
import pytest

from app.dataset import clean_frames, normalize_label, read_source_csv
from app.preprocessing.text_cleaner import normalize_text, prepare_text


@pytest.mark.parametrize("raw, expected", [(0, 0), (" 1 ", 1), ('"0"', 0), (' \"\'2\'\" ', 2)])
def test_labels(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", [None, True, "", "3", "negative", "1.0", 1.5, '"1', float("nan")])
def test_reject_invalid_labels(raw):
    with pytest.raises(ValueError):
        normalize_label(raw)


def test_conservative_normalization():
    assert normalize_text("  Indi\t gid!  Nami ang food.\n") == "Indi gid! Nami ang food."
    assert normalize_text("Cafe\u0301") == "Café"
    assert prepare_text("Juan sa Iloilo, a@sample.com 0917-123-4567", True) == "Juan sa Iloilo, [EMAIL] [PHONE]"


def test_clean_and_audit_duplicates():
    raw = pd.DataFrame([
        (" Nami gid! ", '"2"', "train"), ("nami gid!", "2", "test"),
        ("Nami gid!", "2", "validation"), ("Another", "0", "train"),
        ("Another", "0", "train"), ("", "1", "test"), ("Invalid", "7", "train"),
        ("Conflict", "0", "train"), ("Conflict", "2", "train"),
    ], columns=["sentence", "label", "split"])
    clean, bad, duplicates, report = clean_frames([raw])
    assert len(clean) == 2
    assert clean.loc[clean.sentence_clean == "nami gid!", "split"].item() == "test"
    assert len(duplicates) == 3
    assert len(bad) == 4
    assert report["total_rows"] == len(clean) + len(bad) + len(duplicates)
    assert report["missing_values"]["sentence"] == 1


def test_invalid_test_label_still_reserves_text():
    frame = pd.DataFrame([("Same text", "0", "train"), ("Same text", "oops", "test")],
                         columns=["sentence", "label", "split"])
    clean, bad, duplicates, _ = clean_frames([frame])
    assert clean.empty and len(bad) == len(duplicates) == 1


def test_csv_quotes_multiline_and_malformed(tmp_path):
    path = tmp_path / "test.csv"
    original = 'sentence,label\n"Nami, gid!",2\n"Line one\nline two",1\nextra,fields,0\nplain,"""0"""\n'
    path.write_text(original, encoding="utf-8")
    frame, rejected = read_source_csv(path, "test")
    assert len(frame) == 3 and len(rejected) == 1
    assert "extra,fields,0" in rejected[0]["raw_record"]
    assert normalize_label(frame.iloc[-1].label) == 0
    assert path.read_text(encoding="utf-8") == original


def test_unclosed_csv_is_quarantined(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text('sentence,label\n"unclosed,0\nremaining text', encoding="utf-8")
    frame, rejected = read_source_csv(path, "train")
    assert frame.empty and "remaining text" in rejected[0]["raw_record"]
