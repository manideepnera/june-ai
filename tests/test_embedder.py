# tests/test_embedder.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embeddings.embedder import Embedder
import math


def cosine_similarity(a: list, b: list) -> float:
    """Compute similarity between two vectors. 1.0 = identical, 0.0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (mag_a * mag_b)


def test_embed_single():
    embedder = Embedder()
    vector = embedder.embed("Hello world")

    assert isinstance(vector, list)
    assert len(vector) == 384  # all-MiniLM-L6-v2 produces 384 dims
    assert all(isinstance(v, float) for v in vector)
    print(f"✓ Single embed works. Dimensions: {len(vector)}")


def test_semantic_similarity():
    """Similar texts should have higher cosine similarity."""
    embedder = Embedder()

    vec_a = embedder.embed("neural networks and deep learning")
    vec_b = embedder.embed("artificial intelligence and machine learning")
    vec_c = embedder.embed("banana bread recipe with chocolate chips")

    sim_ab = cosine_similarity(vec_a, vec_b)  # should be high
    sim_ac = cosine_similarity(vec_a, vec_c)  # should be low

    print(f"✓ Similarity test:")
    print(f"  AI topics similarity   : {sim_ab:.3f}  (expect > 0.5)")
    print(f"  AI vs food similarity  : {sim_ac:.3f}  (expect < 0.3)")

    assert sim_ab > sim_ac, (
        "Similar texts should score higher than unrelated ones"
    )


def test_batch_embed():
    embedder = Embedder()
    texts = [
        "machine learning fundamentals",
        "python programming basics",
        "cooking pasta at home",
        "deep learning architectures"
    ]
    vectors = embedder.embed_batch(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == 384 for v in vectors)
    print(f"✓ Batch embed works. {len(vectors)} vectors produced.")


if __name__ == "__main__":
    print("\n--- Testing Embedder ---\n")
    test_embed_single()
    test_semantic_similarity()
    test_batch_embed()
    print("\n--- Embedder tests passed ---\n")