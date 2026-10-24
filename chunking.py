"""Text, semantic, and parent-child chunking utilities for RAG ingestion."""

from dataclasses import replace
from enum import Enum
from typing import Any, Optional

from schema import Block, Chunk
from tables import table_chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


prose_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
    keep_separator=True,
    length_function=len,
    is_separator_regex=False
)

def fixed_chunks(
    text: str,
    size: int = 800,
    overlap: int = 100
) -> list[str]:
    """Split text into overlapping character windows.

    Args:
        text: Source text to split.
        size: Maximum number of characters per window.
        overlap: Characters repeated between adjacent full windows.

    Returns:
        Non-whitespace chunks in source order.

    Raises:
        ValueError: For a nonpositive size or an invalid overlap.

    This method does not preserve words or sentence boundaries.
    """

    if size <= 0:
        raise ValueError("Chunk size must be positive")

    if not 0 <= overlap < size:
        raise ValueError("Expected 0 <= overlap < size")

    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        fragment = text[start:start + size]
        if fragment.strip():
            chunks.append(fragment)
        if start + size >= len(text):
            break

    return chunks

def split_prose(text: str) -> list[str]:
    """Recursively split prose by paragraphs, lines, sentences, then words.

    Chunks target 700 characters with up to 100 characters of overlap;
    lengths are based on characters, not model tokens.
    """

    if not text or not text.strip():
        return []

    return [
        chunk
        for chunk in prose_splitter.split_text(text)
        if chunk.strip()
    ]

def semantic_chunks(
    sentences: list[str],
    model,
    threshold: float = 0.55,
    min_chars: int = 120,
    max_chars: int = 1200
) -> list[str]:
    """Group adjacent sentences according to embedding cosine similarity.

    Args:
        sentences: Text already segmented into individual sentences.
        model: Encoder offering ``encode(..., normalize_embeddings=True)``.
        threshold: Start a new chunk when adjacent cosine similarity is lower.
        min_chars: Suppress semantic boundaries before this size.
        max_chars: Soft maximum; an individual sentence may exceed this.

    Returns:
        Consecutive, nonempty semantic text groups.

    The similarity threshold must be calibrated for the chosen model and corpus.
    """

    if not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be between -1 and 1")

    if not 0 <= min_chars <= max_chars or max_chars == 0:
        raise ValueError("Invalid chunk length limits")

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence and sentence.strip()
    ]

    if not sentences:
        return []

    vectors = np.asarray(
        model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    )

    groups = []
    current = [sentences[0]]
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):

        similarity = float(
            np.dot(vectors[i - 1], vectors[i])
        )

        sentence = sentences[i]

        semantic_break = (
            similarity < threshold
            and current_length >= min_chars
        )

        size_break = (
            current_length + len(sentence) + 1 > max_chars
        )

        if semantic_break or size_break:

            groups.append(" ".join(current))

            current = [sentence]
            current_length = len(sentence)

        else:
            current.append(sentence)
            current_length += len(sentence) + 1

    if current:
        groups.append(" ".join(current))

    return groups

def make_parent_children(
    doc_id: str,
    section_id: str,
    section_text: str,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, str], list[Chunk]]:
    """Create retrievable child chunks and an external parent-text record.

    Returns:
        A mapping from parent ID to full section text and the child chunks.
        Store the mapping separately, then use ``parent_id`` in each child's
        metadata to retrieve larger context following vector search.
    """

    section_text = section_text.strip()

    if not section_text:
        return {}, []

    parent_id = f"{doc_id}|section_{section_id}"

    parent_store = {
        parent_id: section_text
    }

    children = []

    for i, text in enumerate(split_prose(section_text)):

        child_metadata = dict(metadata or {})

        child_metadata.update({
            "parent_id": parent_id,
            "section_id": section_id,
            "child_index": i,
            "content_type": "text"
        })

        children.append(
            Chunk(
                chunk_id=f"{parent_id}|child_{i}",
                document_id=doc_id,
                text=text,
                metadata=child_metadata
            )
        )

    return parent_store, children


def _kind_value(kind: Any) -> str:
    """Return the serialized content kind for an enum or plain string."""
    return str(kind.value if isinstance(kind, Enum) else kind).lower()


def blocks_to_chunks(blocks: list[Block]) -> list[Chunk]:
    """Convert parsed blocks into ordered, source-traceable retrieval chunks.

    Tables use row-aware splitting. Headings update section context and are
    included with subsequent prose rather than embedded in isolation. All
    other nonempty text, including dataset summaries, uses recursive splitting.

    Args:
        blocks: Parsed blocks in the order they appear in a document.

    Returns:
        Child-ready chunks with document, block, and source metadata.
    """
    result = []
    heading_stack = []

    for block_no, block in enumerate(blocks):
        kind = _kind_value(block.kind)
        text = block.text.strip()

        if kind == "heading":
            if not text:
                continue

            try:
                level = max(1, int(block.metadata.get("heading_level", 1)))
            except (TypeError, ValueError):
                level = 1

            heading_stack = [
                (previous_level, heading)
                for previous_level, heading in heading_stack
                if previous_level < level
            ]
            heading_stack.append((level, text))
            continue

        metadata = dict(block.metadata)
        metadata["block_index"] = block_no
        heading_path = [heading for _, heading in heading_stack]
        if heading_path:
            metadata["heading_path"] = heading_path

        if kind == "table":
            metadata.setdefault("table_no", block_no + 1)
            table_block = replace(block, metadata=metadata)
            result.extend(table_chunks(table_block, rows_per_chunk=8))
            continue

        if not text:
            continue

        prefix = (
            f"Section: {' > '.join(heading_path)}\n" if heading_path else ""
        )
        for index, part in enumerate(split_prose(text)):
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                "content_type": kind,
                "chunk_index": index,
            })
            result.append(
                Chunk(
                    chunk_id=f"{block.document_id}|block_{block_no}|chunk_{index}",
                    document_id=block.document_id,
                    text=prefix + part,
                    metadata=chunk_metadata,
                )
            )

    return result
