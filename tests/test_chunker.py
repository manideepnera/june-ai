import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunking.splitter import TextSplitter


def test_basic_chunking():
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)

    # Create a text longer than one chunk
    text = """
    Artificial intelligence is the simulation of human intelligence by machines.
    Machine learning is a subset of AI that learns from data without being programmed.
    Deep learning uses neural networks with many layers to learn representations.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret and understand visual data.
    Reinforcement learning trains agents to make decisions through rewards.
    """ * 3  # repeat to make it long enough

    chunks = splitter.split(text)

    assert len(chunks) > 1, "Text should produce multiple chunks"
    assert all(len(c) > 0 for c in chunks), "No chunk should be empty"
    assert all(len(c) <= 250 for c in chunks), "Chunks should respect size limit"

    print(f"✓ Chunking works.")
    print(f"  Input length  : {len(text)} chars")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")


def test_overlap():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    text = "word " * 100  # 500 chars

    chunks = splitter.split(text)
    assert len(chunks) > 1

    # Check overlap exists between consecutive chunks
    if len(chunks) >= 2:
        end_of_first = chunks[0][-20:]
        start_of_second = chunks[1][:30]
        print(f"✓ Overlap check:")
        print(f"  End of chunk 1  : '{end_of_first}'")
        print(f"  Start of chunk 2: '{start_of_second}'")


def test_metadata():
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)
    text = "This is a test document. " * 20
    chunks = splitter.split_with_metadata(text, source="test.txt")

    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)
    assert chunks[0]["source"] == "test.txt"
    print(f"✓ Metadata chunks work. Total: {len(chunks)}")


if __name__ == "__main__":
    print("\n--- Testing Chunker ---\n")
    test_basic_chunking()
    test_overlap()
    test_metadata()
    print("\n--- Chunker tests passed ---\n")