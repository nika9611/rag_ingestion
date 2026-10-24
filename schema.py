from calendar import firstweekday
from dataclasses import dataclass, field
from enum import Enum
from tkinter import NO
from typing import Any, Optional

from sympy import li


class Kind(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    DATASET = "dataset"

@dataclass
class Block:
    document_id: str
    kind: Kind
    text: str = "" 
    rows: Optional[list[list[str]]]  = None
    headers: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str 
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

