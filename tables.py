import uuid

from parsers import document_id
from schema import Block, Chunk


def table_row_text(block: Block, row: list[str]) -> str:
    assert block.headers is not None
    if len(row) != len(block.headers):
        raise ValueError("Column count mismatch: inspect source table")
    pairs = "; ".join(
        f"{name}={value}" for name, value
        in zip(block.headers, row)
    )
    location = block.metadata.get("sheet", block.metadata.get("page"))
    return f"Table in {block.document_id}, location={location}; \nContent: {pairs}"


import hashlib
import json


def table_row_text(block: Block, row: list[str]) -> str:
    headers = block.headers or []

    fields = []

    for i in range(max(len(headers), len(row))):
        column_name = (
            headers[i] if i < len(headers) and headers[i]
            else f"Column_{i + 1}"
        )

        value = row[i] if i < len(row) else ""

        fields.append(f"{column_name}: {value}")

    return " | ".join(fields)


def table_chunks(
    block: Block,
    rows_per_chunk: int = 10
) -> list[Chunk]:

    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be greater than 0")

    if not block.headers or not block.rows:
        return []

    chunks = []

    # Stable fallback when table_no is unavailable
    fingerprint = hashlib.sha256(
        json.dumps(
            [block.headers, block.rows],
            ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()[:12]

    table_id = block.metadata.get("table_no", fingerprint)

    source_location = block.metadata.get(
        "sheet",
        f"page_{block.metadata.get('page', 'unknown')}"
    )

    table_ref = (
        f"{block.document_id}|"
        f"{source_location}|table_{table_id}"
    )

    table_header = (
        f"Columns: {', '.join(block.headers)}.\n"
    )

    for start in range(0, len(block.rows), rows_per_chunk):

        subset = block.rows[start:start + rows_per_chunk]

        body = "\n".join(
            table_row_text(block, row)
            for row in subset
        )

        metadata = dict(block.metadata)

        metadata.update({
            "content_type": "table_rows",
            "table_ref": table_ref,
            "row_start": start,
            "row_end": start + len(subset),
            "row_count": len(subset)
        })

        chunks.append(
            Chunk(
                chunk_id=f"{table_ref}|rows_{start}",
                document_id=block.document_id,
                text=table_header + body,
                metadata=metadata
            )
        )

    return chunks