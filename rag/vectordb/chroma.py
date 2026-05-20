from pathlib import Path


class ChromaStore:
    """
    Wrapper around ChromaDB.
    Stores text chunks + their embedding vectors.
    Retrieves most similar chunks for any query vector.
    All data saved locally in storage/vectors/.
    """

    def __init__(
        self,
        collection_name: str = "june_knowledge",
        persist_path: str = "storage/vectors"
    ):
        self.collection_name = collection_name
        self.persist_path = persist_path
        self._client = None
        self._collection = None
        self._setup()

    def _setup(self):
        """Initialize ChromaDB client and collection."""
        import chromadb

        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_path)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # use cosine similarity
        )

        print(
            f"✓ ChromaDB ready. "
            f"Collection: '{self.collection_name}' | "
            f"Documents: {self._collection.count()}"
        )

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str]
    ):
        """
        Store chunks with their vectors and metadata.
        ids must be unique strings per chunk.
        """
        if not texts:
            return

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 4
    ) -> list[dict]:
        """
        Find top_k most similar chunks to the query vector.
        Returns list of dicts with text, source, and score.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return []

        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "score": round(1 - dist, 4)  # convert distance to similarity
            })

        return output

    def count(self) -> int:
        """Return number of stored chunks."""
        return self._collection.count()

    def delete_collection(self):
        """Wipe everything. Useful for re-indexing."""
        self._client.delete_collection(self.collection_name)
        self._setup()
        print("✓ Collection wiped and recreated.")

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document chunk already exists."""
        try:
            result = self._collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except Exception:
            return False