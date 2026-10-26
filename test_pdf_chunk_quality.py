"""Regression checks for PDF layout artifacts and continued table extraction."""

from pathlib import Path

import pytest

from chunking import blocks_to_chunks
from parsers import _clean_pdf_table, parse_pdf
from schema import Kind


def test_clean_table_preserves_alignment_and_partial_empty_cells():
    assert _clean_pdf_table([
        [None, None, None, None],
        ["", "Name", "", "Value"],
        ["", "A", "", ""],
        ["", "B", "", "2"],
    ]) == [["Name", "Value"], ["A", ""], ["B", "2"]]
    assert _clean_pdf_table([[None, ""], []]) == []


def test_medicare_tables_headers_continuation_and_order():
    path = Path(__file__).parent / "data" / "MediCare_UseCase.pdf"
    if not path.exists():
        pytest.skip("MediCare sample PDF is unavailable")
    blocks = parse_pdf(path, "data/MediCare_UseCase.pdf")
    tables = [block for block in blocks if block.kind == Kind.TABLE]
    assert tables[0].headers == ["Domain", "Role", "Format", "Time Limit"]
    assert len(tables[0].rows) == 1
    assert tables[1].headers == ["Column", "Type", "Description"]
    assert len(tables[1].rows) == 10
    assert tables[2].headers == tables[1].headers
    assert len(tables[2].rows) == 8
    assert tables[2].rows[0][0] == "next_scheduled_visit"
    assert tables[2].metadata["headers_inherited"] is True
    assert blocks.index(tables[2]) < next(i for i,b in enumerate(blocks) if b.text == "3. Tasks")
    assert all(any(row) for table in tables for row in table.rows)

    chunks = blocks_to_chunks(blocks)
    assert all("Column_" not in chunk.text for chunk in chunks)
    assert all("Use Case | Page" not in chunk.text for chunk in chunks)
    schema_chunks = [chunk for chunk in chunks if "Column: " in chunk.text]
    assert all("Table: 2. Dataset Description" in chunk.text for chunk in schema_chunks)
    assert "Column: next_scheduled_visit | Type: Date" in "\n".join(chunk.text for chunk in schema_chunks)
    assert any("Your role:" in chunk.text and "You are a Data Scientist" in chunk.text for chunk in chunks)


def test_single_page_footer_is_not_blindly_removed(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / "single.pdf"
    document = canvas.Canvas(str(path))
    document.drawString(70, 780, "Document body")
    document.drawString(70, 25, "Unique important margin text")
    document.save()
    blocks = parse_pdf(path, "single.pdf")
    assert "Unique important margin text" in " ".join(block.text for block in blocks)
