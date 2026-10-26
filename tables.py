"""Convert structured table blocks into embedding-friendly row chunks."""

import hashlib
import json

from schema import Block, Chunk


def table_row_text(block: Block, row: list[str]) -> str:
    """Render a row as labeled values without dropping unmatched columns.

    Missing cells become empty strings. Extra cells receive generated column
    names, preserving source data even when the table has ragged rows.
    """
    headers = block.headers or []
    fields = []

    for index in range(max(len(headers), len(row))):
        column = (
            str(headers[index]).strip()
            if index < len(headers) and headers[index]
            else f"Column_{index + 1}"
        )
        value = row[index] if index < len(row) else ""
        fields.append(f"{column}: {value}")

    return " | ".join(fields)


def table_chunks(block: Block, rows_per_chunk: int = 10) -> list[Chunk]:
    """Split a table into labeled row chunks with titles and row coordinates.

    Args:
        block: Table block with column headers, data rows, and source metadata.
        rows_per_chunk: Maximum number of data rows per output chunk.

    Returns:
        Chunks with zero-based, half-open row ranges relative to data rows.

    Raises:
        ValueError: If ``rows_per_chunk`` is not positive.

    Assign each source table a distinct ``table_no`` within its page or sheet;
    the content-hash fallback cannot distinguish identical tables.
    """
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be greater than 0")

    if not block.headers or not block.rows:
        return []

    table_id = block.metadata.get("table_no")
    if table_id is None:
        payload = json.dumps(
            [block.headers, block.rows], ensure_ascii=False
        ).encode("utf-8")
        table_id = hashlib.sha256(payload).hexdigest()[:12]

    source_location = block.metadata.get(
        "sheet", f"page_{block.metadata.get('page', 'unknown')}"
    )
    table_ref = f"{block.document_id}|{source_location}|table_{table_id}"

    title = block.metadata.get("table_title")
    # Include the title in embedded text for context. Each row already labels
    # its values, so a separate Columns line would repeat the same information.
    title_line = f"Table: {title}\n" if title else ""

    chunks = []
    for start in range(0, len(block.rows), rows_per_chunk):
        subset = block.rows[start:start + rows_per_chunk]
        body = "\n".join(table_row_text(block, row) for row in subset)

        metadata = dict(block.metadata)
        metadata.update({
            "content_type": "table_rows",
            "table_ref": table_ref,
            "row_start": start,
            "row_end": start + len(subset),
            "row_count": len(subset),
        })

        chunks.append(
            Chunk(
                chunk_id=f"{table_ref}|rows_{start}",
                document_id=block.document_id,
                text=title_line + body,
                metadata=metadata,
            )
        )

    return chunks
