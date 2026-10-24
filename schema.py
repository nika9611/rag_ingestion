"""Shared data models for parsed document blocks and retrieval chunks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Kind(str, Enum):
    """Supported content categories emitted by document parsers."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    DATASET = "dataset"

@dataclass
class Block:
    """A source document element before text or table chunking.

    ``rows`` excludes the header row when ``headers`` is supplied. ``metadata``
    holds source coordinates such as page, sheet, table number, and file path.
    """

    document_id: str
    kind: Kind
    text: str = ""
    rows: Optional[list[list[str]]] = None
    headers: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """An indexed text segment with a stable identifier and source metadata."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

