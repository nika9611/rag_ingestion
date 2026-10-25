"""Encode text and retrieval chunks using a shared SentenceTransformer model.

The model loads when this module is imported and may download its files on
first use. Embeddings are normalized for cosine similarity comparisons.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


from collections.abc import Iterable, Iterator
from itertools import islice
from schema import Chunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
dimension = model.get_embedding_dimension()
assert dimension is not None

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized vectors in the same order as the input.

    Args:
        texts: Texts to encode, processed in model batches of up to 32.

    Returns:
        One list of float32-converted values per text, each with ``dimension``
        components. An empty input returns an empty list without encoding.

    Raises:
        AssertionError: If the output shape is unexpected or values are not
            finite, when Python assertions are enabled.
    """
    if not texts:
        return []
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    assert vectors.shape == (len(texts), dimension)
    assert np.isfinite(vectors).all()
    return vectors.astype("float32").tolist()


def batched(
        items: Iterable[Chunk], 
        size: int = 64
    ) -> Iterator[list[Chunk]]:
    """Consume chunks lazily and yield lists in their original order.

    Args:
        items: An iterable of chunks, including a generator.
        size: Maximum chunks per batch. Use a positive integer; zero yields
            no batches, and a negative value raises ``ValueError``.

    Yields:
        Nonempty batches of up to ``size`` chunks. The final batch may be
        smaller; an empty iterable yields nothing.
    """
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch

        
def embed_batches(chunks: Iterable[Chunk]):
    """Yield groups of up to 64 chunks alongside their embedding vectors.

    Args:
        chunks: Chunks whose ``text`` fields will be encoded.

    Yields:
        Pairs of ``(group, vectors)`` with matching positions: each vector
        represents the text of the chunk at the same index in ``group``.
        Input order is preserved, and an empty iterable yields nothing.
    """
    for group in batched(chunks):
        vectors = embed_texts([chunk.text for chunk in group])
        yield group, vectors

