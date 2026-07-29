from __future__ import annotations

from pathlib import Path

from .extractors import extract_asset
from .index import Index, create_index
from .models import Asset, GeneratedMetadata, QAResult
from .providers import HeuristicProvider, MetadataProvider


class MetadataStudio:
    def __init__(self, provider: MetadataProvider | None = None, index: Index | None = None, asset_extractor=extract_asset) -> None:
        self.provider = provider or HeuristicProvider()
        self.index = index or create_index("hash")
        self.assets: list[Asset] = []
        self.metadata: dict[str, GeneratedMetadata] = {}
        self.asset_extractor = asset_extractor

    def ingest(
        self,
        paths: list[str | Path],
        *,
        extractor: str = "native",
        max_bytes: int = 20 * 1024 * 1024,
        textract_client=None,
        tika_parser=None,
    ) -> list[Asset]:
        assets = [self.extract(path, extractor=extractor, max_bytes=max_bytes, textract_client=textract_client, tika_parser=tika_parser) for path in paths]
        self.add(assets)
        return assets

    def extract(
        self,
        path: str | Path,
        *,
        extractor: str = "native",
        max_bytes: int = 20 * 1024 * 1024,
        textract_client=None,
        tika_parser=None,
    ) -> Asset:
        return self.asset_extractor(
            path,
            max_bytes=max_bytes,
            extractor=extractor,
            textract_client=textract_client,
            tika_parser=tika_parser,
        )

    def add(self, assets: list[Asset]) -> None:
        known = {asset.id for asset in self.assets}
        fresh = [asset for asset in assets if asset.id not in known]
        self.assets.extend(fresh)
        self.index.add(fresh)

    def generate(self, asset: Asset) -> GeneratedMetadata:
        related = self.index.search(asset.text[:1000], limit=4, exclude_id=asset.id)
        context = "\n---\n".join(item.asset.text[:1000] for item in related if item.score > 0.05)
        result = self.provider.generate(asset, context)
        result.related_assets = [item.asset.id for item in related if item.score > 0.05]
        self.metadata[asset.id] = result
        return result

    def ask(self, question: str, limit: int = 4) -> QAResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")
        if len(question) > 2_000:
            raise ValueError("question cannot exceed 2,000 characters")
        related = [item for item in self.index.search(question, limit) if item.score > 0]
        if not related:
            return QAResult("No indexed asset contains enough evidence to answer that question.", [], self.provider.name, False)
        return self.provider.answer(question.strip(), [item.asset for item in related])
