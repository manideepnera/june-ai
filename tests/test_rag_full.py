import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline
from rag.retriever import RAGRetriever
from llm.manager import LLMManager


def test_full_rag_flow():
    """
    Full end-to-end test:
    Index documents → retrieve chunks → inject into prompt → LLM answers.
    """

    print("\n=== Full RAG Flow Test ===\n")

    # Step 1: Make sure documents are indexed
    print("Step 1: Indexing documents...")
    pipeline = RAGPipeline()
    pipeline.index_folder("storage/files/notes")

    # Step 2: Set up retriever
    retriever = RAGRetriever()
    assert retriever.has_knowledge(), \
        "Nothing in knowledge base. Index some documents first."
    print(f"✓ Knowledge base has {retriever.store.count()} chunks\n")

    # Step 3: Ask a question
    question = "What is RAG and how does it work?"
    print(f"Step 2: Question: '{question}'")

    # Step 4: Retrieve relevant chunks
    print("Step 3: Retrieving relevant chunks...")
    results = retriever.search(question, top_k=3)

    print(f"✓ Retrieved {len(results)} chunks:")
    for r in results:
        print(f"  Score {r['score']:.3f} | Source: {r['source']}")
        print(f"  Preview: {r['text'][:80]}...")
    print()

    # Step 5: Build prompt with context
    context = retriever.format_context(results)

    prompt = f"""{context}

---

Using the context above, answer this question clearly:
{question}"""

    # Step 6: Send to LLM
    print("Step 4: Sending to LLM...")
    llm = LLMManager()

    system = (
        "You are June, a personal AI assistant. "
        "Answer based on the provided context. "
        "If the context contains the answer, use it. "
        "Be concise and clear."
    )

    print("\nJune's answer:\n")
    print("-" * 40)

    response = ""
    for chunk in llm.stream(prompt, system=system):
        print(chunk, end="", flush=True)
        response += chunk

    print("\n" + "-" * 40)

    assert len(response) > 50, "Response too short — something may be wrong"
    print("\n✓ Full RAG flow working.")


if __name__ == "__main__":
    test_full_rag_flow()