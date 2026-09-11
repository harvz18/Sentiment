"""Verify prepared inputs and preserve prior development models before rebuilding."""
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.artifacts import file_hash, read_json
from app.config.settings import Settings


def archive_existing(root: Path, relative_paths: list[str]) -> Path | None:
    existing = [(relative, root / relative) for relative in relative_paths if (root / relative).exists()]
    if not existing:
        return None
    destination = root / "reports/archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    for relative, source in existing:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return destination


def verify_prepared(settings: Settings) -> dict:
    manifest = read_json(settings.corpus_dir / "manifest.json")
    if manifest.get("version") != 2 or not manifest.get("binary_sentiment"):
        raise ValueError("Prepare the binary Multivent corpus first")
    for key, value in (("pii_sanitization", settings.enable_pii_sanitization),
                       ("domain_relevance_threshold", settings.domain_relevance_threshold),
                       ("rag_balance_classes", settings.rag_balance_classes),
                       ("rag_include_validation", settings.rag_include_validation),
                       ("seed", settings.lda_random_state),
                       ("domain_test_fraction", settings.domain_test_fraction),
                       ("domain_validation_fraction", settings.domain_validation_fraction)):
        if manifest[key] != value:
            raise ValueError(f"Corpus setting {key} changed; rerun preparation and rebuild")
    for relative, key in (("config/datasets.yaml", "registry_sha256"), ("config/domain.yaml", "domain_config_sha256")):
        if file_hash(settings.root / relative) != manifest[key]:
            raise ValueError(f"{relative} changed; rerun preparation and rebuild")
    for name, digest in manifest["files"].items():
        if file_hash(settings.corpus_dir / name) != digest:
            raise ValueError(f"Prepared file {name} changed; rerun preparation and rebuild")
    return manifest
