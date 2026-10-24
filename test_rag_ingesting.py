"""Pytest examples for schema.py, tables.py, parsers.py, and chunking.py.

Usage: ``python -m pytest -v test_rag_pytest.py``

Put this alongside your four modules. Existing DOCX/XLSX sample files are
optional: tests for them skip if the files are absent. Other tests build their
inputs in memory or with pytest's ``tmp_path`` fixture.
"""

import hashlib
import io
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
# Supports the four modules either alongside this file or in an upload/ folder.
MODULE_DIR = HERE if (HERE / "schema.py").exists() else HERE / "upload"
sys.path.insert(0, str(MODULE_DIR))

from schema import Block, Chunk, Kind
from tables import table_chunks, table_row_text


@pytest.fixture
def sales_table():
    """One sample structured table shared by multiple tests."""
    return Block(
        document_id="retail.xlsx",
        kind=Kind.TABLE,
        headers=["SKU", "Revenue"],
        rows=[
            ["Laptop", "1200"],
            ["Mouse", "30", "Extra"],
            ["Keyboard", "70"],
        ],
        metadata={"sheet": "Sales", "table_no": 1, "table_title": "October"},
    )


def test_schema_metadata_independence():
    """Dataclass metadata dictionaries must not be shared across instances."""
    a = Block("one", Kind.PARAGRAPH)
    b = Block("two", Kind.PARAGRAPH)
    a.metadata["page"] = 1
    assert b.metadata == {}
    assert Chunk("id", "one", "sample").text == "sample"


def test_table_row_text_handles_extra_columns(sales_table):
    assert table_row_text(sales_table, sales_table.rows[1]) == (
        "SKU: Mouse | Revenue: 30 | Column_3: Extra"
    )


@pytest.mark.parametrize("rows_per_chunk, expected_ranges", [
    (1, [(0, 1), (1, 2), (2, 3)]),
    (2, [(0, 2), (2, 3)]),
    (10, [(0, 3)]),
])
def test_table_chunks(sales_table, rows_per_chunk, expected_ranges):
    chunks = table_chunks(sales_table, rows_per_chunk=rows_per_chunk)
    assert [(c.metadata["row_start"], c.metadata["row_end"]) for c in chunks] == expected_ranges
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all("Columns: SKU, Revenue" in c.text for c in chunks)
    assert all("Table: October" in c.text for c in chunks)


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_table_chunks_invalid_size(sales_table, invalid_size):
    with pytest.raises(ValueError):
        table_chunks(sales_table, rows_per_chunk=invalid_size)


def test_sha256_and_document_id(tmp_path):
    from parsers import document_id, sha256_file

    file = tmp_path / "document.txt"
    file.write_bytes(b"RAG document")
    assert sha256_file(file) == hashlib.sha256(b"RAG document").hexdigest()
    assert document_id(file, tmp_path) == "document.txt"
    with pytest.raises(ValueError):
        document_id(file, tmp_path / "different_root")


def test_docx_parsing(tmp_path):
    docx = pytest.importorskip("docx")
    from parsers import parse_docx

    doc = docx.Document()
    doc.add_heading("Retail Policies", level=1)
    doc.add_paragraph("Customers can return eligible items within 30 days.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Product"
    table.cell(0, 1).text = "Quantity"
    table.cell(1, 0).text = "Laptop"
    table.cell(1, 1).text = "3"
    file = tmp_path / "policy.docx"
    doc.save(file)

    blocks = parse_docx(file, "policy.docx")
    assert [b.kind for b in blocks] == [Kind.HEADING, Kind.PARAGRAPH, Kind.TABLE]
    assert blocks[0].metadata["heading_level"] == 1
    assert blocks[2].headers == ["Product", "Quantity"]
    assert blocks[2].rows == [["Laptop", "3"]]


def test_xlsx_parsing(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from parsers import parse_xlsx

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Inventory"
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Inventory report"
    sheet.append(["Item", "Stock"])
    sheet.append(["Laptop", 12])
    file = tmp_path / "inventory.xlsx"
    workbook.save(file)

    blocks = parse_xlsx(file, "inventory.xlsx")
    assert len(blocks) == 1
    assert blocks[0].headers == ["Item", "Stock"]
    assert blocks[0].rows == [["Laptop", "12"]]
    assert blocks[0].metadata["table_title"] == "Inventory report"
    assert blocks[0].metadata["header_row"] == 2


def test_pdf_parsing(monkeypatch):
    pytest.importorskip("pdfplumber")
    pytest.importorskip("reportlab")
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
    import parsers

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    table = Table([["Product", "Units"], ["Laptop", "8"]], colWidths=[180, 120])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    document.build([
        Paragraph("Warehouse report", getSampleStyleSheet()["Normal"]), table
    ])
    buffer.seek(0)

    original_open = parsers.pdfplumber.open
    monkeypatch.setattr(parsers.pdfplumber, "open", lambda _: original_open(buffer))
    blocks = parsers.parse_pdf(Path("example.pdf"), "example.pdf")

    assert any(b.kind == Kind.PARAGRAPH and "Warehouse" in b.text for b in blocks)
    extracted = [b for b in blocks if b.kind == Kind.TABLE]
    assert len(extracted) == 1
    assert extracted[0].headers == ["Product", "Units"]
    assert extracted[0].rows == [["Laptop", "8"]]


def test_html_parsing(tmp_path):
    pytest.importorskip("bs4")
    from parsers import parse_html

    file = tmp_path / "example.html"
    file.write_text(
        "<html><body><nav>Menu</nav><h1>Pricing</h1>"
        "<p>Discounts are available.</p><script>alert('ignore')</script>"
        "</body></html>", encoding="utf-8"
    )
    blocks = parse_html(file, "example.html")
    assert len(blocks) == 1
    assert "Pricing" in blocks[0].text
    assert "Discounts are available" in blocks[0].text
    assert "Menu" not in blocks[0].text
    assert "alert" not in blocks[0].text


@pytest.fixture
def chunking():
    """Skip LangChain-dependent tests if text splitters are not installed."""
    pytest.importorskip("langchain_text_splitters")
    import chunking as module
    return module


@pytest.mark.parametrize("size,overlap,expected", [
    (6, 2, ["abcdef", "efghij"]),
    (4, 0, ["abcd", "efgh", "ij"]),
])
def test_fixed_chunks(chunking, size, overlap, expected):
    assert chunking.fixed_chunks("abcdefghij", size=size, overlap=overlap) == expected


@pytest.mark.parametrize("size,overlap", [(0, 0), (3, 3), (4, -1)])
def test_fixed_chunks_invalid(chunking, size, overlap):
    with pytest.raises(ValueError):
        chunking.fixed_chunks("text", size=size, overlap=overlap)


def test_recursive_chunks(chunking):
    parts = chunking.split_prose("Sentence about RAG. " * 120)
    assert len(parts) > 1
    assert all(0 < len(p) <= 700 for p in parts)


def test_semantic_chunks(chunking):
    np = pytest.importorskip("numpy")

    class FakeEncoder:
        def encode(self, sentences, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["convert_to_numpy"] is True
            return np.array([[1., 0.], [0.99, 0.14], [0., 1.]])

    groups = chunking.semantic_chunks(
        ["Dogs enjoy walks.", "Puppies like parks.", "Invoices are overdue."],
        FakeEncoder(), threshold=0.5, min_chars=0
    )
    assert groups == [
        "Dogs enjoy walks. Puppies like parks.", "Invoices are overdue."
    ]


def test_parent_child_chunks(chunking):
    parent_store, children = chunking.make_parent_children(
        "policies.docx", "returns", "Items can be returned within 30 days. " * 100,
        metadata={"page": 2},
    )
    assert len(parent_store) == 1
    assert len(children) > 1
    parent_id = next(iter(parent_store))
    assert all(c.metadata["parent_id"] == parent_id for c in children)
    assert all(c.metadata["page"] == 2 for c in children)


def test_blocks_to_chunks(chunking):
    blocks = [
        Block("guide.docx", Kind.HEADING, text="Sales", metadata={"heading_level": 1}),
        Block("guide.docx", Kind.PARAGRAPH, text="Monthly revenue improved."),
        Block("guide.docx", Kind.TABLE, headers=["Month", "Revenue"],
              rows=[["October", "100"]], metadata={"table_no": 1}),
    ]
    chunks = chunking.blocks_to_chunks(blocks)
    assert len(chunks) == 2
    assert "Section: Sales" in chunks[0].text
    assert chunks[0].metadata["heading_path"] == ["Sales"]
    assert chunks[1].metadata["content_type"] == "table_rows"
