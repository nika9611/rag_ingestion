"""Store embedded chunks and run similarity searches in local Qdrant.

Importing this module loads the embedding model and opens ``./local_qdrant``
relative to the current working directory. The shared client uses the
``enterprise_rag`` collection and closes automatically at normal process exit.
"""

from qdrant_client import QdrantClient, models
import pandas as pd
from embeddings import model, dimension, MODEL_NAME, embed_batches, embed_texts
from rank_bm25 import BM25Okapi


from schema import Chunk

from collections import defaultdict
import atexit
import re
import uuid


COLLECTION = "enterprise_rag"
# A disk path keeps indexed data between runs without a separate Qdrant server.
client = QdrantClient(path="./local_qdrant")
# Register cleanup when imported as well as when run directly. Exit handlers run
# before Python tears down its import system, so Qdrant can release its file lock.
atexit.register(client.close)

def ensure_collection():
    """Create the configured collection if it does not already exist.

    New collections use the embedding model's vector dimension and cosine
    distance. Existing collection settings are neither checked nor changed.
    """
    # Reuse the collection so indexing another document preserves existing data.
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                # Stored vectors and query vectors must have the same length.
                size=dimension, 
                # Cosine compares vector direction, matching our normalized embeddings.
                distance=models.Distance.COSINE
            ),
        )


def stable_point_id(chunk_id: str) -> str:
    """Return a deterministic UUID string for a chunk ID using UUID5.

    Reusing a chunk ID produces the same point ID, allowing later upserts to
    replace that point. Chunk IDs must be unique across the collection.
    """
    # Qdrant accepts UUIDs or unsigned integers as IDs, rather than arbitrary text.
    # UUID5 maps our readable chunk ID to a repeatable UUID: reruns update the point.
    # NAMESPACE_URL is a fixed UUID seed here; chunk_id does not need to be a URL.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
    
def upsert_chunks(chunks):
    """Embed chunks and insert or replace their points in the collection.

    Args:
        chunks: An iterable of ``Chunk`` objects, processed in batches of up
            to 64. The collection is created even when the iterable is empty.

    Each point stores the chunk's metadata plus its document ID, chunk ID,
    text, and embedding model name. These explicit fields override matching
    metadata keys. Writes wait for completion before the next batch starts;
    earlier batches remain stored if a later batch fails.

    Returns:
        None.
    """
    ensure_collection()
    # Process a batch at a time to avoid holding every embedding in memory.
    for group, vectors in embed_batches(chunks):
        points = []
        # embed_batches preserves order, so each chunk is paired with its own vector.
        for item, vector in zip(group, vectors):
            # Vectors support similarity search; payloads retain readable text and
            # source details for filtering, displaying results, and building context.
            payload = {
                **item.metadata,
                # Put these fields last so metadata cannot overwrite their values.
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "text": item.text,
                "embedding_model": MODEL_NAME,
            }
            print("--------------")
            print({**item.metadata})
            print("--------------")
            points.append(
                models.PointStruct(
                    id=stable_point_id(item.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )
        # One write per batch reduces call overhead. Waiting lets a subsequent
        # search see completed writes; an existing point ID replaces that point.
        client.upsert(COLLECTION, points=points, wait=True)


def search(
        query: str, 
        content_type: str | None = None,
        limit: int = 5
        ):
    """Embed a query and retrieve the most similar stored chunks.

    Args:
        query: Text to encode with the same model used for stored chunks.
        content_type: Optional exact match against the payload's
            ``content_type`` field. None or an empty string disables filtering.
        limit: Maximum number of results to request; defaults to 5.

    Returns:
        Qdrant scored points containing IDs, similarity scores, and payloads,
        ordered by relevance. No matches returns an empty list.

    The collection must already exist; call ``upsert_chunks`` or
    ``ensure_collection`` before searching a new database.
    """
    query_filter = None
    if content_type:
        # Restrict eligible points before ranking, so a table-only search cannot
        # return prose even when that prose has a high similarity score.
        query_filter = models.Filter(must=
                [
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value=content_type)
                )
            ]
        )
    # Use the ingestion model so the query and stored vectors share the same
    # embedding space. The encoder takes a list; [0] selects our single vector.
    vector = embed_texts([query])[0]
    return client.query_points(
                collection_name=COLLECTION,
                query=vector,
                query_filter=query_filter,
                # Return text and source metadata alongside scores for the caller.
                with_payload=True,
                # Bound the number of matches passed on as retrieval context.
                limit=limit,
            ).points

def show_table(
    limit: int | None = 20,
    content_type: str | None = None,
) -> pd.DataFrame:
    """Print stored point IDs and all payload fields and return a DataFrame.

    Args:
        limit: Maximum rows to retrieve; use None for all matching points.
            A numeric limit must be positive. Defaults to 20.
        content_type: Optional exact payload filter, e.g. ``"table_rows"``.

    Returns:
        One row per point, with ``point_id`` and a column for
        every payload key found in the retrieved records. Payload columns
        use a ``payload.`` prefix to avoid collisions with point fields.
        Nested values remain dictionaries or lists; absent fields are missing
        values. An empty result still has a ``point_id`` column.

    Text is retained in full; embedding vectors are not fetched or included.
    Records use Qdrant scroll order,
    not similarity ranking. Reading does not create a collection or compute
    embeddings. Retrieving all points loads the results into memory.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than 0 or None")
    if not client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' does not exist. Index chunks first.")
        return pd.DataFrame(columns=["point_id"])

    scroll_filter = None
    if content_type:
        scroll_filter = models.Filter(must=[
            models.FieldCondition(
                key="content_type", match=models.MatchValue(value=content_type)
            )
        ])

    rows = []
    offset = None
    while True:
        # Fetch payloads in bounded pages without loading large embedding vectors.
        # Continue until the requested limit or the end is reached.
        page_size = 256 if limit is None else min(256, limit - len(rows))
        records, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            rows.append({
                "point_id": record.id,
                **{f"payload.{key}": value for key, value in (record.payload or {}).items()},
            })
        if offset is None or (limit is not None and len(rows) >= limit):
            break

    dataframe = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["point_id"])
    if dataframe.empty:
        print("No stored chunks match the requested filter.")
    else:
        # Local display options reveal every column and full cell values without
        # changing pandas settings for the rest of the application.
        with pd.option_context(
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.max_seq_items", None,
        ):
            print(dataframe.to_string(index=False))
        print(f"Displayed {len(dataframe)} stored chunks from '{COLLECTION}'.")
        if offset is not None:
            print("More matches are available; increase limit or use limit=None.")
    return dataframe


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def bm25_rank(query: str, chunks: list[Chunk]) -> list[str]:
    tokens = [tokenize(c.text) for c in chunks]
    ranker = BM25Okapi(tokens)
    scores = ranker.get_scores(tokenize(query))
    order = sorted(
        range(len(chunks)),
        key=lambda i: scores[i], reverse=True
        )
    return [chunks[i].chunk_id for i in order]

def rrf(*ranked_id_lists: list[str], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for ids in ranked_id_lists:
        for rank, key in enumerate(ids, start=1):
            scores[key] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
