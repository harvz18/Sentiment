"""Conservative normalization shared by dataset preparation and inference."""
import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """NFC and whitespace only; preserve case, punctuation, particles and language."""
    if not isinstance(text, str):
        raise ValueError("Sentence must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def sanitize_pii(text: str) -> str:
    """Optional conservative email/phone masking; does not identify names/locations."""
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    return re.sub(r"(?<!\w)(?:\+63[ -]?9\d{2}|09\d{2})[ -]?\d{3}[ -]?\d{4}(?!\w)",
                  "[PHONE]", text)


def prepare_text(text: str, pii: bool = False) -> str:
    normalized = normalize_text(text)
    return sanitize_pii(normalized) if pii else normalized


def text_fingerprint(text: str) -> str:
    """Case-insensitive exact-text identity for duplicate and leakage checks."""
    return hashlib.sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()
