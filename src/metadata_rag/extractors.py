from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path

from .models import Asset

TEXT_SUFFIXES = {".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".yaml", ".yml", ".toml", ".json", ".csv", ".sql", ".html", ".css"}
TEXTRACT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEXTRACT_SYNC_MAX_BYTES = 10 * 1024 * 1024
EXTRACTORS = {"native", "tika", "textract"}


def _native_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to extract PDFs") from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Install python-docx to extract DOCX files") from exc
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError(f"Unsupported extension: {suffix or '(none)'}")


def _tika_text(path: Path, parser_adapter=None) -> str:
    if parser_adapter is None:
        try:
            from tika import parser as parser_adapter
        except ImportError as exc:
            raise RuntimeError("Install tika and a Java runtime to use Apache Tika extraction") from exc
    parsed = parser_adapter.from_file(str(path))
    if not isinstance(parsed, dict):
        raise RuntimeError("Apache Tika returned an invalid response")
    return str(parsed.get("content") or "")


def _textract_text(path: Path, raw: bytes, client=None) -> str:
    if path.suffix.lower() not in TEXTRACT_SUFFIXES:
        raise ValueError("Amazon Textract supports PDF, PNG, JPEG, and TIFF inputs in this workflow")
    if len(raw) > TEXTRACT_SYNC_MAX_BYTES:
        raise ValueError("Amazon Textract synchronous extraction is capped at 10 MiB in this workflow")
    if client is None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install boto3 and configure AWS credentials to use Amazon Textract") from exc
        client = boto3.client("textract", region_name=os.getenv("AWS_REGION") or None)
    response = client.detect_document_text(Document={"Bytes": raw})
    blocks = response.get("Blocks", []) if isinstance(response, dict) else []
    return "\n".join(
        str(block.get("Text", "")).strip()
        for block in blocks
        if isinstance(block, dict) and block.get("BlockType") == "LINE" and str(block.get("Text", "")).strip()
    )


def extract_asset(
    path: str | Path,
    *,
    max_bytes: int = 20 * 1024 * 1024,
    extractor: str = "native",
    textract_client=None,
    tika_parser=None,
) -> Asset:
    path = Path(path)
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if not path.is_file():
        raise ValueError(f"Not a readable file: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{path.name} exceeds the {max_bytes // (1024 * 1024)} MB limit")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"{path.name} exceeds the {max_bytes // (1024 * 1024)} MB limit")
    extractor = extractor.lower().strip()
    if extractor not in EXTRACTORS:
        raise ValueError(f"extractor must be one of: {', '.join(sorted(EXTRACTORS))}")
    if extractor == "native":
        text = _native_text(path)
    elif extractor == "tika":
        text = _tika_text(path, tika_parser)
    else:
        text = _textract_text(path, raw, textract_client)
    checksum = hashlib.sha256(raw).hexdigest()
    return Asset(
        id=checksum[:16],
        name=path.name,
        text=text.strip(),
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        size_bytes=size,
        checksum=checksum,
        source=str(path),
        extractor=extractor,
    )
