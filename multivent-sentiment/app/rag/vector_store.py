"""Binary FAISS index with separate language/domain searches and leakage guards."""
from pathlib import Path

import faiss
import numpy as np

from app.artifacts import file_hash, read_json, write_json
from app.preprocessing.text_cleaner import text_fingerprint
from app.schemas.sentiment import ReferenceExample


class VectorStore:
    """IndexFlatIP stores cosine vectors; derived pool indexes share its metadata."""
    def __init__(self, index, records: list[dict], manifest: dict):
        self.index, self.records, self.manifest = index, records, manifest
        self.pools = {}
        for pool in ("language", "domain"):
            positions = [i for i, row in enumerate(records) if row["pool"] == pool]
            if positions:
                vectors = np.array([index.reconstruct(i) for i in positions], dtype="float32")
                subindex = faiss.IndexFlatIP(index.d)
                subindex.add(vectors)
                self.pools[pool] = (subindex, positions)

    @staticmethod
    def validate_records(records: list[dict], forbidden_hashes: set[str]) -> None:
        if not records:
            raise ValueError("Cannot build an empty RAG index")
        ids, hashes = set(), set()
        for record in records:
            if record["split"] not in {"train", "validation"}:
                raise ValueError("RAG refuses test or unknown split records")
            if record["sentiment"] not in {"positive", "negative"}:
                raise ValueError("RAG refuses neutral or unknown sentiment")
            digest = text_fingerprint(record["text"])
            original = text_fingerprint(record.get("text_original", record["text"]))
            if {digest, original}.intersection(forbidden_hashes):
                raise ValueError("RAG record overlaps held-out evaluation text")
            if record["id"] in ids or digest in hashes:
                raise ValueError("Duplicate RAG record id or text")
            ReferenceExample(**record, similarity=0)
            ids.add(record["id"])
            hashes.add(digest)

    @classmethod
    def build(cls, embeddings: np.ndarray, records: list[dict], identity: dict,
              forbidden_hashes: set[str], pii: bool = False) -> "VectorStore":
        cls.validate_records(records, forbidden_hashes)
        vectors = np.array(embeddings, dtype="float32", order="C", copy=True)
        if vectors.ndim != 2 or vectors.shape[0] != len(records) or vectors.shape[1] == 0:
            raise ValueError("Embedding shape does not match records")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embeddings must be finite and nonzero")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, records, {"version": 2, "binary_sentiment": True, "embedding": identity,
                                   "count": len(records), "dimension": vectors.shape[1],
                                   "pii_sanitization": pii, "forbidden_hashes": sorted(forbidden_hashes),
                                   "splits": sorted({r["split"] for r in records})})

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())
        write_json(directory / "metadata.json", self.records)
        self.manifest["files"] = {name: file_hash(directory / name) for name in ("index.faiss", "metadata.json")}
        write_json(directory / "manifest.json", self.manifest)

    @classmethod
    def load(cls, directory: Path, identity: dict, pii: bool = False) -> "VectorStore":
        manifest = read_json(directory / "manifest.json")
        if manifest.get("version") != 2 or not manifest.get("binary_sentiment"):
            raise ValueError("Legacy index detected; rebuild the binary Multivent corpus/index")
        if manifest["embedding"] != identity or manifest["pii_sanitization"] != pii:
            raise ValueError("Embedding/PII configuration differs from index; rebuild RAG")
        for name, digest in manifest["files"].items():
            if file_hash(directory / name) != digest:
                raise ValueError("RAG artifacts do not match; rebuild index")
        records = read_json(directory / "metadata.json")
        cls.validate_records(records, set(manifest["forbidden_hashes"]))
        index = faiss.deserialize_index(np.frombuffer((directory / "index.faiss").read_bytes(), dtype="uint8"))
        if index.ntotal != len(records) or index.d != manifest["dimension"]:
            raise ValueError("FAISS shape does not match metadata")
        return cls(index, records, manifest)

    def search(self, embedding: np.ndarray, top_k: int, pool: str | None = None) -> list[ReferenceExample]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        query = np.array(embedding, dtype="float32", order="C", copy=True).reshape(1, -1)
        if query.shape[1] != self.index.d or not np.isfinite(query).all() or not np.linalg.norm(query):
            raise ValueError("Invalid query embedding")
        faiss.normalize_L2(query)
        if pool is not None:
            if pool not in {"language", "domain"}:
                raise ValueError("Unknown retrieval pool")
            if pool not in self.pools:
                return []
            index, positions = self.pools[pool]
        else:
            index, positions = self.index, list(range(len(self.records)))
        scores, indices = index.search(query, min(top_k, len(positions)))
        return [ReferenceExample(**self.records[positions[int(pos)]], similarity=float(np.clip(score, -1, 1)))
                for score, pos in zip(scores[0], indices[0]) if pos >= 0]
