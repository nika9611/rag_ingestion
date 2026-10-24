from pathlib import Path

from parsers import sha256_file, document_id, parse_pdf, parse_xlsx, parse_docx
from tables import table_row_text, table_chunks
from schema import Block, Kind

root = Path("/Users/puneetjain/Documents/Project/RAG")

files = [root / "data/MediCare_UseCase copy.pdf",
         root / "data/MediCare_UseCase.pdf",
         root / "data/rag_docx_parser_test.docx",
         root / "data/rag_xlsx_parser_test.xlsx"]


# for file in files:
#     print(f"Working on file: {file}")
#     print(f"SHA256 of the file is: {sha256_file(file)}")
#     print(f"Document ID of the file is: {document_id(file, root)}")


# blocks = parse_pdf(files[0], document_id(files[0], root))
# blocks = parse_docx(files[2], document_id(files[2], root))
# blocks = parse_xlsx(files[3], document_id(files[3], root))

# for block in blocks:
#     print(f"\nKind: {block.kind}")
#     print(f"Metadata: {block.metadata}")

#     if block.kind.value == "table":
#         print("Headers:", block.headers)
#         print("Rows:", block.rows)
#     else:
#         print("Text:", block.text[:500])


# example = Block(
#         "report_2026", Kind.TABLE,
#         headers=["region", "q2_revenue_usd_m"],
#         rows=[["Europe", "7.8"], ["North America", "13.4"]],
#         metadata={"page": 7, "table_no": 2}
#     )
# print(table_row_text(example, example.rows[0]))

block = Block(
    document_id="sales.xlsx",
    kind=Kind.TABLE,
    headers=["Product", "Price", "Quantity"],
    rows=[
        ["Laptop", "50000", "3"],
        ["Mouse", "1000", "10"],
        ["Keyboard", "2500", "5"],
        ["Monitor", "15000", "2"],
        ["Headphones", "3000", "8"]
    ],
    metadata={
        "sheet": "Products",
        "table_no": 1,
        "source": "documents/sales.xlsx"
    }
)

chunks = table_chunks(
    block,
    rows_per_chunk=2
)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("\nChunk ID:", chunk.chunk_id)
    print("Text:\n", chunk.text)
    print("Metadata:", chunk.metadata)