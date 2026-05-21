from typing import Optional
from sentence_transformers import SentenceTransformer
from rag.embeddings.embedder import Embedder
from rag.vectordb.chroma import ChromaStore


class RAGRetriever:
    """
    Combines embedder + vector store into one interface.
    This is what the orchestrator calls.
    It never calls embedder or ChromaStore directly.
    
    Can accept pre-initialized embed_model and collection for
    proper lifespan management, or will create them internally
    for backward compatibility (testing only).
    
    RAG Configuration:
    - top_k: 3 results (optimized from 4 - balance accuracy vs context size)
    - Injected dependencies are preferred to avoid duplicate initialization
    """

    # RAG Configuration - tune these for your needs
    RAG_TOP_K = 3  # Number of chunks to retrieve (reduced from 4)
    RAG_MAX_CONTEXT_LENGTH = 3000  # Max total characters of context

    def __init__(
        self,
        embed_model: Optional[SentenceTransformer] = None,
        collection = None
    ):
        self.embed_model = embed_model
        self.collection = collection
        
        # Only create fallback instances if NOT injected
        # This prevents duplicate initialization during startup
        self.embedder = None
        self.store = None
        
        if embed_model is None:
            self.embedder = Embedder()
        
        if collection is None:
            self.store = ChromaStore()

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Search for chunks most relevant to the query.
        Returns list of dicts with text, source, score.
        
        top_k: Number of results (default: RAG_TOP_K)
        """
        import time
        start_time = time.time()
        
        if top_k is None:
            top_k = self.RAG_TOP_K
        
        # Use injected embed_model if available, otherwise use embedder
        embed_start = time.time()
        if self.embed_model is not None:
            query_vector = self.embed_model.encode(query).tolist()
        else:
            query_vector = self.embedder.embed(query)
        embed_time = round((time.time() - embed_start) * 1000, 1)
        
        # Use injected collection if available, otherwise use store
        retrieval_start = time.time()
        if self.collection is not None:
            results_data = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not results_data["documents"][0]:
                retrieval_time = round((time.time() - retrieval_start) * 1000, 1)
                total_time = round((time.time() - start_time) * 1000, 1)
                return []

            output = []
            for text, meta, dist in zip(
                results_data["documents"][0],
                results_data["metadatas"][0],
                results_data["distances"][0]
            ):
                output.append({
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "score": round(1 - dist, 4)
                })
            
            retrieval_time = round((time.time() - retrieval_start) * 1000, 1)
            total_time = round((time.time() - start_time) * 1000, 1)
            
            # Store timing info for logging
            for item in output:
                item["_embed_ms"] = embed_time
                item["_retrieval_ms"] = retrieval_time
                
            return output
        else:
            results = self.store.query(query_vector, top_k=top_k)
            retrieval_time = round((time.time() - retrieval_start) * 1000, 1)
            for item in results:
                item["_embed_ms"] = embed_time
                item["_retrieval_ms"] = retrieval_time
            return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a clean context block
        ready to be injected into the LLM prompt.
        
        Limits total context size to RAG_MAX_CONTEXT_LENGTH chars.
        """
        if not results:
            return ""

        sections = []
        total_chars = 0
        
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            text = r.get("text", "")
            section = f"[Source {i}: {source}]\n{text}"
            
            # Stop adding if we exceed max context length
            if total_chars + len(section) > self.RAG_MAX_CONTEXT_LENGTH:
                break
            
            sections.append(section)
            total_chars += len(section)

        return "Relevant context from your knowledge base:\n\n" + \
               "\n\n---\n\n".join(sections)

    def has_knowledge(self) -> bool:
        """Check if anything has been indexed yet."""
        if self.collection is not None:
            return self.collection.count() > 0
        return self.store.count() > 0