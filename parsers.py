import hashlib
from pathlib import Path

from schema import Block, Kind

from bs4 import BeautifulSoup
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from openpyxl import load_workbook
import pdfplumber
from pdf2image import convert_from_path
import pytesseract




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def document_id(path: Path, root: Path):
    return str(path.resolve().relative_to(root.resolve()))


def parse_pdf(path: Path, doc_id: str) -> list[Block]:
    output = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            found = page.find_tables()
            boxes = [t.bbox for t in found]

            def inside_table(obj):
                if obj.get("object_type") != "char":
                    return False

                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2

                return any(x0 <= x < x1 and y0 <= y <= y1 
                           for x0, y0, x1, y1 in boxes)
            
            outside = page.filter(lambda o: not inside_table(o))
            prose = outside.extract_text() or ""
            
            if prose.strip():
            
                output.append(
                    Block(
                        doc_id, 
                        Kind.PARAGRAPH,
                        text=prose,
                        metadata= {
                            "page": page_no,
                            "source": str(path)
                        }
                    )
                )

            for number, table in enumerate(found, start=1):
                rows = [
                    [
                        (c or "").strip() for c in row
                    ]
                    for row in table.extract()
                ]

                if not rows:
                    continue

                output.append(
                    Block(
                        doc_id,
                        Kind.TABLE,
                        headers=rows[0],
                        rows=rows[1:],
                        metadata={
                            "page": page_no,
                            "table_no": number,
                            "bbox": list(table.bbox),
                            "source": str(path)
                        }
                    )
                )
    return output

from pathlib import Path
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

from schema import Block, Kind


def parse_docx(path: Path, doc_id: str) -> list[Block]:
    result = []
    doc = Document(path)

    for element in doc.iter_inner_content():

        # Process paragraphs and headings
        if isinstance(element, Paragraph):
            text = element.text.strip()

            if text:
                style_name = element.style.name

                kind = (
                    Kind.HEADING
                    if style_name.lower().startswith("heading")
                    else Kind.PARAGRAPH
                )

                result.append(
                    Block(
                        document_id=doc_id,
                        kind=kind,
                        text=text,
                        metadata={
                            "style": style_name,
                            "source": str(path)
                        }
                    )
                )

        # Process tables
        elif isinstance(element, Table):
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
                            "source": str(path)
                        }
                    )
                )

    return result

def parse_xlsx(
    path: Path,
    doc_id: str,
    data_only: bool = True
) -> list[Block]:

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
                            "end_row": section[-1][0],
                            "start_col": first_col + 1,
                            "end_col": last_col,
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
    soup = BeautifulSoup(
        path.read_text(
            encoding="utf-8"
            ),
        "html.parser"
        )
    
    for tag in soup.select("script, style, nav, footer"):
        tag.decompose()
        text = soup.get_text(" ", strip=True)
    return [
        Block(
            doc_id, Kind.PARAGRAPH, 
            text=text,
            metadata={
                "source": str(path)
                }
            )
        ] if text else []

def ocr_pdf_page(path: Path, page_no: int) -> str:
    images = convert_from_path(
        str(path), 
        dpi=250, 
        first_page=page_no,
        last_page=page_no
    )
    return pytesseract.image_to_string(images[0]) if images else ""