
from sentence_transformers import SentenceTransformer

text = "A" * 2000

chunks = fixed_chunks(
    text,
    size=800,
    overlap=100
)

print([len(chunk) for chunk in chunks])




model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "LightGBM is a gradient boosting algorithm.",
    "It grows trees using a leaf-wise strategy.",
    "Hyperparameters help control overfitting.",
    "The delivery truck arrived at the warehouse.",
    "Route optimization minimizes transportation costs."
]

chunks = semantic_chunks(
    sentences,
    model,
    threshold=0.55,
    min_chars=50,
    max_chars=500
)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)



parents, children = make_parent_children(
    doc_id="report_001",
    section_id="pricing",
    section_text="A long pricing section...",
    metadata={
        "page": 3,
        "source": "pricing.pdf"
    }
)