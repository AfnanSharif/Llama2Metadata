from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Asset:
    id: str
    name: str
    text: str
    media_type: str
    size_bytes: int
    checksum: str
    source: str = "upload"
    extractor: str = "native"


@dataclass(slots=True)
class GeneratedMetadata:
    asset_id: str
    title: str
    summary: str
    keywords: list[str]
    content_type: str
    language: str
    entities: list[str] = field(default_factory=list)
    related_assets: list[str] = field(default_factory=list)
    confidence: float = 0.0
    generator: str = "heuristic"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QAResult:
    answer: str
    source_ids: list[str]
    provider: str
    grounded: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
