from typing import Any, Optional

from schema import Chunk

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
    metadata: Optional[dict[str, Any]] = None
):

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



