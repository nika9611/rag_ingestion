"""Index sample data and search the local Qdrant collection.

Run from the project directory:
    venv/bin/python sample_vectordb.py

The first run may download the embedding model. Data is saved in
./local_qdrant, using the collection configured in vectordb.py.
Stable sample IDs allow repeated runs to update the same points.
"""

from pathlib import Path

from chunking import blocks_to_chunks
from parsers import document_id, parse_pdf
from schema import Block, Chunk, Kind
from tables import table_chunks
from vectordb import COLLECTION, search, upsert_chunks, show_table, delete_data


def sample_chunks(revenue: Block) -> list[Chunk]:
    """Build revenue table chunks and policy prose with filterable metadata."""
    
    chunks = table_chunks(revenue, rows_per_chunk=1)
    chunks.append(
        Chunk(
            chunk_id="sample_policy.txt|returns",
            document_id="sample_policy.txt",
            text=(
                "Customers can return unused products within 30 days of "
                "purchase. A receipt is required for a full refund."
            ),
            metadata={"content_type": "paragraph", "source": "sample_policy.txt"},
        )
    )
    return chunks


def ingest_medicare_pdf() -> list[Chunk]:
    """Parse data/MediCare_UseCase.pdf and upsert its chunks into Qdrant.

    Resolves the PDF relative to this script so callers can run from another
    directory. Preserves page/source metadata and uses a project-relative
    document ID for repeatable chunk IDs. Returns the indexed chunks.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If no indexable content is extracted. Scanned pages need
            OCR, which ``parse_pdf`` does not perform automatically.

    Repeated runs update matching chunk IDs. As with ``upsert_chunks``, this
    does not delete obsolete chunks if the PDF is later shortened or changed.
    """
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "data" / "MediCare_UseCase.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Reuse the existing parser and chunker to retain page coordinates and
    # handle prose and tables with their respective splitting strategies.
    blocks = parse_pdf(pdf_path, document_id(pdf_path, project_root))
    chunks = blocks_to_chunks(blocks)
    if not chunks:
        raise ValueError(f"No indexable content extracted from {pdf_path}")

    upsert_chunks(chunks)
    print(
        f"Indexed {len(chunks)} chunks from {len(blocks)} blocks in "
        f"'{pdf_path.name}' into '{COLLECTION}'."
    )
    return chunks


def main() -> None:
    revenue = Block(
            document_id="sample_revenue.xlsx",
            kind=Kind.TABLE,
            headers=["Region", "Quarter", "Revenue (USD million)"],
            rows=[
                ["Europe", "Q2", "7.8"],
                ["North America", "Q2", "13.4"],
                ["Asia", "Q2", "9.2"],
            ],
            metadata={
                "sheet": "Revenue",
                "table_no": 1,
                "table_title": "Regional quarterly revenue",
                "source": "sample_revenue.xlsx",
            },
        )
    chunks = sample_chunks(revenue)
    upsert_chunks(chunks)
    print(f"Upserted {len(chunks)} sample chunks into '{COLLECTION}'.")

    examples = [
        ("Europe Q2 revenue", "table_rows"),
        ("How long do customers have to return products?", "paragraph"),
        ("quarterly revenue", None),
    ]
    for query, content_type in examples:
        print(f"\nQuery: {query}")
        print(f"Content type filter: {content_type or 'all'}")
        results = search(query, content_type=content_type, limit=3)
        print(f"============ Showing results for {query}")
        if not results:
            print("No matches found.")
        for point in results:
            payload = point.payload or {}
            print(f"  Score: {point.score:.3f}")
            print(f"  Document: {payload.get('document_id', '')}")
            print(f"  Text: {payload.get('text', '')}")


if __name__ == "__main__":
    # delete_data()
    # main()
    # ingest_medicare_pdf()

                
    data = show_table(limit=None)
    # print(data)
    data.to_csv(
        "data/sample_qdrant_db.csv",
        index=False,
    )
    # # Example after retrieving dense points from Qdrant:
    # dense_ids = [p.payload["chunk_id"] for p in search(question)]
    # lexical_ids = bm25_rank(question, all_chunks)
    # fused_ids = rrf(dense_ids, lexical_ids)
