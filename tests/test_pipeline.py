import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline


def test_index_file():
    pipeline = RAGPipeline()

    result = pipeline.index_file("storage/files/notes/ai_basics.txt")

    assert result["chunks_added"] > 0 or result["chunks_skipped"] > 0
    print(f"✓ File indexed: {result}")


def test_index_folder():
    pipeline = RAGPipeline()

    results = pipeline.index_folder("storage/files/notes")

    assert len(results) > 0
    print(f"✓ Folder indexed. Files processed: {len(results)}")
    for r in results:
        print(f"  {r['file']} → {r.get('chunks_added', 0)} chunks")


if __name__ == "__main__":
    print("\n--- Testing RAG Pipeline ---\n")
    test_index_file()
    test_index_folder()
    print("\n--- Pipeline tests passed ---\n")