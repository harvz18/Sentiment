"""Try datasets loading, then preserve immutable Hub CSV sources for safe cleaning."""
import argparse
import logging
import re
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from app.artifacts import file_hash, read_json, write_json
from app.config.settings import get_settings

REPO = "jjjardev/hilisenti-v1"
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hub commit or branch; resolved commit recorded")
    args = parser.parse_args()
    root = get_settings().root
    raw = root / "data/raw/hilisenti"
    manifest_path = raw / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for item in manifest["files"]:
            if file_hash(raw / item["local_path"]) != item["sha256"]:
                raise RuntimeError("Raw file changed; restore it before proceeding")
        log.info("Verified existing raw snapshot %s; originals not overwritten", manifest["revision"])
        return
    api = HfApi()
    revision = api.dataset_info(REPO, revision=args.revision).sha
    try:
        loaded = load_dataset(REPO, revision=revision, cache_dir=str(raw / "hf_cache"))
        normal = {split: len(data) for split, data in loaded.items()}
        log.info("Normal datasets loading succeeded: %s", normal)
    except Exception as exc:
        # Parser/library/network failures all lead to the explicitly requested Hub fallback.
        log.warning("datasets loading failed (%s); preserving and reading raw Hub CSVs", type(exc).__name__)
        normal = {"error_type": type(exc).__name__}
    files = []
    for filename in api.list_repo_files(REPO, repo_type="dataset", revision=revision):
        if not filename.lower().endswith(".csv"):
            continue
        match = re.search(r"(?:^|[/_.-])(train|validation|valid|val|test)(?:[/_.-]|$)", filename.lower())
        if not match:
            raise ValueError(f"Cannot determine predefined split for {filename}")
        split = {"val": "validation", "valid": "validation"}.get(match[1], match[1])
        cached = hf_hub_download(REPO, filename, repo_type="dataset", revision=revision,
                                 cache_dir=str(raw / "hub_cache"))
        destination = raw / "source" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_hash(destination) != file_hash(Path(cached)):
                raise RuntimeError(f"Refusing to overwrite original {destination}")
        else:
            shutil.copyfile(cached, destination)
        files.append({"repo_path": filename, "local_path": destination.relative_to(raw).as_posix(),
                      "split": split, "sha256": file_hash(destination)})
    if {item["split"] for item in files} != {"train", "validation", "test"}:
        raise RuntimeError("Expected train/validation/test CSVs were not all found")
    write_json(manifest_path, {"repo": REPO, "revision": revision, "normal_load": normal, "files": files})
    log.info("Saved %s original CSVs under %s", len(files), raw)


if __name__ == "__main__":
    main()
