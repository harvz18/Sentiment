"""Download reviewed public snapshots only; never overwrite original source files."""
import shutil
from pathlib import Path

import requests
import yaml
from huggingface_hub import hf_hub_download

from app.artifacts import file_hash, read_json, write_json
from app.corpus.registry import DatasetSource


def download_source(source: DatasetSource, root: Path) -> list[Path]:
    if not source.enabled:
        raise ValueError("Refusing to download a disabled source")
    # Revalidate even when a caller constructed/mutated a model without validation.
    source = DatasetSource.model_validate(source.model_dump())
    directory = root / "data/raw" / source.id
    if source.id == "hilisenti":
        directory = root / "data/raw/hilisenti"
    manifest_path = directory / "binary_source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["registry"] != source.model_dump():
            raise ValueError(f"Source registry changed for {source.id}; archive its snapshot before changing provenance")
        for item in manifest["files"]:
            if file_hash(directory / item["path"]) != item["sha256"]:
                raise ValueError(f"Original source checksum mismatch: {source.id}")
        return [directory / item["path"] for item in manifest["files"]]
    directory.mkdir(parents=True, exist_ok=True)
    card_path = directory / "source_card.md"
    if source.repository:
        card = hf_hub_download(source.repository, "README.md", repo_type="dataset",
                               revision=source.revision, cache_dir=str(directory / "hub_cache"))
        card_text = Path(card).read_text(encoding="utf-8")
        metadata = yaml.safe_load(card_text.split("---", 2)[1])
        if str(metadata.get("license", "")).lower() != source.license.lower():
            raise ValueError(f"Published license disagrees with registry for {source.id}; no review data downloaded")
        if not card_path.exists():
            shutil.copyfile(card, card_path)
    else:
        # UCI's license is on its primary dataset page; store that exact evidence.
        response = requests.get(source.license_url, timeout=60)
        response.raise_for_status()
        if "CC BY 4.0" not in response.text and "creativecommons.org/licenses/by/4.0" not in response.text:
            raise ValueError("Cannot verify UCI's published license; no archive downloaded")
        if not card_path.exists():
            card_path.write_text(response.text, encoding="utf-8")
    files = []
    for filename in source.files:
        path = directory / "source" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if source.repository:
            cached = Path(hf_hub_download(source.repository, filename, repo_type="dataset",
                                          revision=source.revision, cache_dir=str(directory / "hub_cache")))
            if path.exists() and file_hash(path) != file_hash(cached):
                raise ValueError(f"Refusing to overwrite original {path}")
            if not path.exists():
                shutil.copyfile(cached, path)
        elif not path.exists():
            response = requests.get(source.download_url, timeout=90)
            response.raise_for_status()
            temp = path.with_suffix(".download")
            temp.write_bytes(response.content)
            temp.replace(path)
        files.append({"path": path.relative_to(directory).as_posix(), "sha256": file_hash(path)})
    write_json(manifest_path, {"registry": source.model_dump(), "files": files,
                              "license_evidence_sha256": file_hash(card_path)})
    return [directory / item["path"] for item in files]
