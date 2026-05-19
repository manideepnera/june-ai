class Embedder:
    """
    Converts text into embedding vectors using sentence-transformer.
    Runs entirely on CPU - no GPU Needed.
    Model downloads automatically on first use (~80MB)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None # lasy load -  don't load until first use

    def _load_model(self):
        """Load model on firt use. takes ~3 seconds first time."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            print(f"(First time takes ~30 seconds to download ~80MB)")

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            print(f"✓ Embedding model loaded")


    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text. returns vector."""
        self._load_model()
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts at once.
        Much faster than calling embed() in a loop.
        This is usefull when indexing documents.
        """

        self._load_model()
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            batch_size = 32
        )
        return vectors.tolist()
    

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        self._load_model()
        return self._model.get_embedding_dimension()