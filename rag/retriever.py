from rag.embeddings.embedder import Embedder
from rag.vectordb.chroma import ChromaStore


class RAGRetriever:
    """
    Combines embedder + vector store into one interface.
    This is what the orchestrator calls.
    It never calls embedder or ChromaStore directly.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.store = ChromaStore()

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Search for chunks most relevant to the query.
        Returns list of dicts with text, source, score.
        """
        query_vector = self.embedder.embed(query)
        results = self.store.query(query_vector, top_k=top_k)
        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a clean context block
        ready to be injected into the LLM prompt.
        """
        if not results:
            return ""

        sections = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            text = r.get("text", "")
            sections.append(f"[Source {i}: {source}]\n{text}")

        return "Relevant context from your knowledge base:\n\n" + \
               "\n\n---\n\n".join(sections)

    def has_knowledge(self) -> bool:
        """Check if anything has been indexed yet."""
        return self.store.count() > 0