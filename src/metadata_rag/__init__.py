"""Metadata generation and retrieval toolkit."""

from .models import Asset, GeneratedMetadata, QAResult
from .service import MetadataStudio

__all__ = ["Asset", "GeneratedMetadata", "MetadataStudio", "QAResult"]
__version__ = "1.0.0"
