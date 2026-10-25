"""Index sample data and search the local Qdrant collection.

Run from the project directory:
    venv/bin/python sample_vectordb.py

The first run may download the embedding model. Data is saved in
./local_qdrant, using the collection configured in vectordb.py.
Stable sample IDs allow repeated runs to update the same points.
"""

from schema import Block, Chunk, Kind
from tables import table_chunks
from vectordb import COLLECTION, search, upsert_chunks, show_table


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


def main() -> None:
    revenue = Block(
            document_id="sample_revenue.xlsx",
            kind=Kind.TABLE,
            headers=["Region", "Quarter", "Revenue (USD million)"],
            rows=[
                ["Europe", "Q2", "7.8"],
                # ["North America", "Q2", "13.4"],
                # ["Asia", "Q2", "9.2"],
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
    # main()
    show_table(limit=50).to_csv("data/sample_qdrant_db_with_vector.csv")
    # # Example after retrieving dense points from Qdrant:
    # dense_ids = [p.payload["chunk_id"] for p in search(question)]
    # lexical_ids = bm25_rank(question, all_chunks)
    # fused_ids = rrf(dense_ids, lexical_ids)
