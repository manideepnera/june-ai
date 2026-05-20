import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectordb.chroma import ChromaStore
from rag.embeddings.embedder import Embedder


def test_store_and_retrieve():
    embedder = Embedder()
    store = ChromaStore(
        collection_name="test_collection",
        persist_path="storage/vectors/test"
    )

    # Clear any previous test data
    store.delete_collection()

    # Sample chunks to store
    texts = [
        "Neural networks learn from data using backpropagation.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large datasets to train.",
        "FastAPI is a modern web framework for building APIs.",
        "Transformers architecture revolutionized natural language processing."
    ]

    # Embed and store
    embeddings = embedder.embed_batch(texts)
    metadatas = [{"source": f"test_doc_{i}.txt"} for i in range(len(texts))]
    ids = [f"test_chunk_{i}" for i in range(len(texts))]

    store.add(texts, embeddings, metadatas, ids)

    assert store.count() == len(texts)
    print(f"✓ Stored {store.count()} chunks.")

    # Query for something related to neural networks
    query = "how do neural networks work?"
    query_vec = embedder.embed(query)
    results = store.query(query_vec, top_k=2)

    assert len(results) > 0
    print(f"✓ Query returned {len(results)} results:")
    for r in results:
        print(f"  Score {r['score']:.3f} | {r['text'][:60]}...")

    # Most relevant should be about neural networks or deep learning
    top_result = results[0]["text"].lower()
    assert any(word in top_result for word in
               ["neural", "learning", "transformer", "deep"])
    print("✓ Top result is semantically relevant.")

    # Cleanup test collection
    store.delete_collection()


if __name__ == "__main__":
    print("\n--- Testing ChromaDB ---\n")
    test_store_and_retrieve()
    print("\n--- ChromaDB tests passed ---\n")