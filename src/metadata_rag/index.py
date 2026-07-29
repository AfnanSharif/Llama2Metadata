from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from .models import Asset

TOKENS = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}")


@dataclass(slots=True)
class Neighbor:
    asset: Asset
    score: float


class Index(Protocol):
    backend: str
    def add(self, assets: list[Asset]) -> None: ...
    def search(self, query: str, limit: int = 4, exclude_id: str | None = None) -> list[Neighbor]: ...


class HashVectorIndex:
    """Stable feature-hashing fallback that needs no model downloads."""

    backend = "hash-vector"

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.assets: list[Asset] = []
        self.vectors: list[list[float]] = []

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in TOKENS.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1 if digest[4] & 1 else -1
            values[position] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def add(self, assets: list[Asset]) -> None:
        for asset in assets:
            self.assets.append(asset)
            self.vectors.append(self._embed(f"{asset.name} {asset.text}"))

    def search(self, query: str, limit: int = 4, exclude_id: str | None = None) -> list[Neighbor]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = self._embed(query)
        matches = [Neighbor(asset, sum(a * b for a, b in zip(query_vector, vector))) for asset, vector in zip(self.assets, self.vectors) if asset.id != exclude_id]
        return sorted(matches, key=lambda result: result.score, reverse=True)[:limit]


class FaissIndex:
    """Semantic FAISS index backed by a SentenceTransformer embedding model."""

    backend = "faiss"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        self.encoder = SentenceTransformer(model_name)
        self.assets: list[Asset] = []
        self.index = None

    def add(self, assets: list[Asset]) -> None:
        import numpy as np

        if not assets:
            return
        vectors = self.encoder.encode([f"{item.name}\n{item.text}" for item in assets], normalize_embeddings=True).astype("float32")
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.asarray(vectors))
        self.assets.extend(assets)

    def search(self, query: str, limit: int = 4, exclude_id: str | None = None) -> list[Neighbor]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if self.index is None:
            return []
        vector = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, positions = self.index.search(vector, min(limit + 1, len(self.assets)))
        results = [Neighbor(self.assets[int(position)], float(score)) for score, position in zip(scores[0], positions[0]) if position >= 0 and self.assets[int(position)].id != exclude_id]
        return results[:limit]


def create_index(backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Index:
    if backend not in {"auto", "faiss", "hash"}:
        raise ValueError("backend must be auto, faiss, or hash")
    if backend in {"auto", "faiss"}:
        try:
            return FaissIndex(model_name)
        except (ImportError, OSError):
            if backend == "faiss":
                raise
    return HashVectorIndex()
