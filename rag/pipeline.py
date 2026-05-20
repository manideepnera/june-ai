import hashlib
from pathlib import Path
from rag.loaders.router import LoaderRouter
from rag.chunking.splitter import TextSplitter
from rag.embeddings.embedder import Embedder
from rag.vectordb.chroma import ChromaStore

class RAGPipeline:
    """
    Full indexing pipeline.
    Takes a file path -> loads -> chunks -> embeds -> stores.
    Safe to run multiple times - skips already indexed files.
    """

    def __init__(self):
        self.loader = LoaderRouter()
        self.splitter = TextSplitter(chunk_size=1500, chunk_overlap=200)
        self.embedder = Embedder()
        self.store = ChromaStore()

    def index_file(self, file_path: str) -> dict:
        """
        Index a single file into the knowledge base.
        Returns a summery of what was indexed.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"\nIndexing: {path.name}")

        # 1. Load
        print(f" [1/4] Loading...")
        text = self.loader.load(file_path)
        print(f"        {len(text)} characters loaded")

        # 2. chunk
        print(f" [2/4] Chunking...")
        chunks_with_meta = self.splitter.split_with_metadata(
            text, source=path.name
        )
        print(f"        {len(chunks_with_meta)} chunks created")

        # 3. Generate IDs and skip already-indexed chunks
        new_chunks = []
        skipped = 0
        for chunk in chunks_with_meta:
            chunk_id = self._make_id(path.name, chunk["chunk_index"])
            if not self.store.document_exists(chunk_id):
                chunk["id"] = chunk_id
                new_chunks.append(chunk)
            else:
                skipped += 1

        if not new_chunks:
            print(f"       All chunks already indexed. Skipping.")
            return {"file": path.name, "chunks": 0, "skipped": skipped}
        
        # 4. Embed
        print(f"       All chunks already indexed. Skipping.")
        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 5. Store
        print(f"    [4/4] Storing in ChromaDB...")
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"]
            }
            for c in new_chunks
        ]
        ids = [c["id"] for c in new_chunks]
        self.store.add(texts, embeddings, metadatas, ids)

        total = self.store.count()
        print(f"✓ Done. New chunks: {len(new_chunks)} | "
              f"Skipped: {skipped} | Total in DB: {total}")
        
        return {
            "file": path.name,
            "chunks_added": len(new_chunks),
            "total_in_db": total
        }
    
    def index_folder(self, folder_path: str) -> list[dict]:
        """
        Index all supported files in a folder.
        Recursively finds pdf, txt, md files.
        """

        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        supported = [".txt", ".md", ".markdown", ".pdf"]
        files = [
            f for f in folder.rglob("*")
            if f.suffix.lower() in supported
        ]

        if not files:
            print(f"No supported files found in: {folder_path}")
            return []
        
        print(f"Found {len(files)} files to index in {folder_path}")
        results = []
        for file in files:
            try:
                result = self.index_file(str(file))
                results.append(result)
            except Exception as e:
                print(f"  ✗ Failed to index {file.name}: {e}")

        return results
    
    def _make_id(self, filename: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk."""
        raw = f"{filename}_chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    


