"""Extract text and tables from PDF, Word, Excel, and HTML sources."""

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
import pdfplumber
from schema import Block, Kind


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file's bytes, reading in 1 MiB parts."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def document_id(path: Path, root: Path) -> str:
    """Return a document's resolved path relative to the ingestion root.

    Raises:
        ValueError: If the resolved path is outside ``root``.
    """
    return str(path.resolve().relative_to(root.resolve()))


def parse_pdf(path: Path, doc_id: str) -> list[Block]:
    """Extract page prose and detected tables into separate PDF blocks.

    Table characters are excluded from prose using detected bounding boxes.
    Text extraction does not perform OCR on scanned pages, and the first row
    of each detected table is assumed to contain headers.
    """
    output = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            found = page.find_tables()
            boxes = [t.bbox for t in found]

            def inside_table(obj):
                """Check whether a PDF character's center lies inside a table."""
                if obj.get("object_type") != "char":
                    return False

                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2

                return any(
                    x0 <= x < x1 and y0 <= y <= y1
                    for x0, y0, x1, y1 in boxes
                )

            outside = page.filter(lambda o: not inside_table(o))
            prose = outside.extract_text() or ""

            if prose.strip():
                output.append(
                    Block(
                        document_id=doc_id,
                        kind=Kind.PARAGRAPH,
                        text=prose,
                        metadata={
                            "page": page_no,
                            "source": str(path),
                        },
                    )
                )

            for number, table in enumerate(found, start=1):
                rows = [
                    [
                        (c or "").strip() for c in row
                    ]
                    for row in (table.extract() or [])
                    if row is not None
                ]

                if not rows:
                    continue

                output.append(
                    Block(
                        document_id=doc_id,
                        kind=Kind.TABLE,
                        headers=rows[0],
                        rows=rows[1:],
                        metadata={
                            "page": page_no,
                            "table_no": number,
                            "bbox": list(table.bbox),
                            "source": str(path),
                        },
                    )
                )
    return output


def parse_docx(path: Path, doc_id: str) -> list[Block]:
    """Extract top-level Word paragraphs, headings, and tables in source order.

    Heading levels and sequential table numbers are saved in metadata. Nested
    tables inside cells, headers, and footers are not extracted by this parser.
    The first table row is assumed to be the column header.
    """
    result = []
    doc = Document(path)
    table_no = 0

    for element in doc.iter_inner_content():

        # Process paragraphs and headings
        if isinstance(element, Paragraph):
            text = element.text.strip()

            if text:
                style_name = element.style.name if element.style else "Normal"

                kind = (
                    Kind.HEADING
                    if style_name.lower().startswith("heading")
                    else Kind.PARAGRAPH
                )

                metadata = {
                    "style": style_name,
                    "source": str(path),
                }
                if kind == Kind.HEADING:
                    match = re.match(r"heading\s+(\d+)", style_name, re.I)
                    metadata["heading_level"] = int(match.group(1)) if match else 1

                result.append(
                    Block(
                        document_id=doc_id,
                        kind=kind,
                        text=text,
                        metadata=metadata,
                    )
                )

        # Process tables
        elif isinstance(element, Table):
            table_no += 1
            rows = [
                [cell.text.strip() for cell in row.cells]
                for row in element.rows
            ]

            if rows:
                result.append(
                    Block(
                        document_id=doc_id,
                        kind=Kind.TABLE,
                        headers=rows[0],
                        rows=rows[1:],
                        metadata={
                            "table_no": table_no,
                            "source": str(path),
                        },
                    )
                )

    return result


def parse_xlsx(
    path: Path,
    doc_id: str,
    data_only: bool = True
) -> list[Block]:
    """Extract contiguous, nonempty worksheet regions as table blocks.

    Blank rows separate regions, and column offsets and merged ranges are
    retained in metadata. ``data_only=True`` reads cached formula results,
    which can be missing when a workbook has not been recalculated in Excel.
    The first row of a region is assumed to contain headers unless it is a
    merged title row spanning the region's columns.

    This heuristic may split a table containing intentional blank rows.
    Normal worksheet loading preserves merged-range metadata but uses memory
    proportional to the size of the workbook.
    """

    blocks = []

    wb = load_workbook(
        path,
        data_only=data_only,
        read_only=False
    )

    try:
        for ws in wb.worksheets:

            sections = []
            current_section = []

            # Identify contiguous nonempty row groups
            for row_no, row in enumerate(
                ws.iter_rows(),
                start=1
            ):
                values = [
                    "" if cell.value is None
                    else str(cell.value).strip()
                    for cell in row
                ]

                if any(values):
                    current_section.append((row_no, values))

                elif current_section:
                    sections.append(current_section)
                    current_section = []

            if current_section:
                sections.append(current_section)

            # Convert each region into a table block
            for table_no, section in enumerate(
                sections,
                start=1
            ):

                # Determine occupied column boundaries
                occupied_cols = [
                    col_no
                    for _, row in section
                    for col_no, value in enumerate(row)
                    if value
                ]

                first_col = min(occupied_cols)
                last_col = max(occupied_cols) + 1

                matrix = [
                    row[first_col:last_col]
                    for _, row in section
                ]

                # A common report pattern uses a merged title above headers.
                table_title = None
                header_row = section[0][0]
                if len(matrix) >= 2:
                    first_row = section[0][0]
                    title_is_merged = any(
                        merged.min_row == merged.max_row == first_row
                        and merged.min_col <= first_col + 1
                        and merged.max_col >= last_col
                        for merged in ws.merged_cells.ranges
                    )
                    if (
                        title_is_merged
                        and sum(bool(value) for value in matrix[0]) == 1
                        and sum(bool(value) for value in matrix[1]) >= 2
                    ):
                        table_title = next(value for value in matrix[0] if value)
                        matrix = matrix[1:]
                        header_row += 1

                blocks.append(
                    Block(
                        document_id=doc_id,
                        kind=Kind.TABLE,
                        headers=matrix[0],
                        rows=matrix[1:],
                        metadata={
                            "sheet": ws.title,
                            "table_no": table_no,
                            "start_row": section[0][0],
                            "header_row": header_row,
                            "end_row": section[-1][0],
                            "start_col": first_col + 1,
                            "end_col": last_col,
                            "table_title": table_title,
                            "merged_ranges": [
                                str(r)
                                for r in ws.merged_cells.ranges
                            ],
                            "source": str(path),
                            "formula_mode": (
                                "cached_values"
                                if data_only else "formulas"
                            )
                        }
                    )
                )

    finally:
        wb.close()

    return blocks


def parse_html(path: Path, doc_id: str) -> list[Block]:
    """Extract visible HTML text after removing common boilerplate tags.

    Tables are flattened into prose; use a dedicated HTML table parser when
    retaining row and column structure is important.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    for tag in soup.select("script, style, nav, footer"):
        tag.decompose()
    text = soup.get_text(" ", strip=True)

    if not text:
        return []

    return [
        Block(
            document_id=doc_id,
            kind=Kind.PARAGRAPH,
            text=text,
            metadata={"source": str(path)},
        )
    ]


def ocr_pdf_page(path: Path, page_no: int) -> str:
    """OCR one 1-based PDF page at 250 DPI using Poppler and Tesseract.

    Requires the optional ``pdf2image`` and ``pytesseract`` packages plus their
    system executables. This helper does not run automatically in parse_pdf.
    """
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(
        str(path),
        dpi=250,
        first_page=page_no,
        last_page=page_no
    )
    return pytesseract.image_to_string(images[0]) if images else ""
