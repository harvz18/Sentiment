"""Validate explicit provenance and licensing before downloading any review data."""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

APPROVED_LICENSES = {"CC-BY-NC-SA-4.0", "CC-BY-NC-4.0", "CC-BY-4.0", "Apache-2.0"}


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    author: str
    url: str
    original_url: str | None = None
    doi: str | None = None
    license: str
    license_url: str
    enabled: bool
    domain: str
    language: list[str]
    language_metadata: str
    language_basis: str
    label_mapping: dict[str, str]
    citation: str
    notes: str
    adapter: str
    repository: str | None = None
    revision: str | None = None
    download_url: str | None = None
    archive_member: str | None = None
    files: list[str]
    pool: str

    @model_validator(mode="after")
    def validate_license(self):
        if not self.enabled:
            return self
        if self.license not in APPROVED_LICENSES or not self.license_url.startswith("https://"):
            raise ValueError(f"Unreviewed license for {self.id}; disable and list in dataset_candidates.md")
        if not self.citation.strip() or not self.author.strip() or not self.url.startswith("https://"):
            raise ValueError("Source, author and citation are required")
        if self.adapter not in {"hilisenti", "mams", "booking", "uci"} or self.pool not in {"language", "domain"}:
            raise ValueError("Unimplemented source adapter or invalid retrieval pool")
        if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name for name in self.files):
            raise ValueError("Unsafe source filename")
        if self.repository and (not self.revision or len(self.revision) != 40):
            raise ValueError("Hub sources require a pinned 40-character commit")
        if self.adapter == "uci" and (not self.download_url or not self.archive_member):
            raise ValueError("UCI source requires archive URL and member")
        return self


def load_registry(path: Path) -> list[DatasetSource]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = [DatasetSource.model_validate(row) for row in value["datasets"]]
    if len({s.id for s in sources}) != len(sources):
        raise ValueError("Duplicate dataset IDs")
    return sources
