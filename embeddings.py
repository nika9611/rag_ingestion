from sentence_transformers import SentenceTransformer
import numpy as np


from collections.abc import Iterable, Iterator
from itertools import islice
from schema import Chunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
dimension = model.get_sentence_embedding_dimension()
assert dimension is not None

def embed_texts(texts: list[str]) -> list[list[float]]:
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
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch

        
def embed_batches(chunks: Iterable[Chunk]):
    for group in batched(chunks, size=64):
        vectors = embed_texts([chunk.text for chunk in group])
        yield group, vectors

