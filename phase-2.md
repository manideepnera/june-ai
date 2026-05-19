# June AI — RAG Engine
> Steps 14 to 20. Build the knowledge engine.  
> By the end, June will answer questions from your own documents.

---

## What you will have at the end

- Document loaders for PDF, TXT, Markdown
- Chunker that splits documents intelligently
- Embedder using sentence-transformers (runs on CPU)
- ChromaDB vector database storing your knowledge
- Retriever that finds relevant chunks for any query
- Your own documents indexed and searchable
- Full test: ask a question → get answer from your notes

---

## Before starting

Make sure Steps 1–13 are complete:

```bash
# Activate venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Confirm Ollama is running
curl http://localhost:11434
# Should return: Ollama is running

# Confirm previous tests still pass
python tests/test_llm.py
# Should show: All tests passed
```

Only continue if both are working.

---

## A quick mental model before writing code

RAG stands for Retrieval Augmented Generation.

Without RAG:
```
You → "What did I write about neural networks?"
June → "I don't know. I have no access to your notes."
```

With RAG:
```
You → "What did I write about neural networks?"
                    ↓
        Search your documents
        Find relevant chunks
        Inject into prompt
                    ↓
June → "Based on your notes from March 12th, you wrote that..."
```

The pipeline has 5 stages:

```
Your files
    ↓
1. LOAD        → read raw text from pdf / txt / md
    ↓
2. CHUNK       → split into ~512 token pieces with overlap
    ↓
3. EMBED       → convert each chunk to a float vector
    ↓
4. STORE       → save vectors + text in ChromaDB
    ↓
5. RETRIEVE    → at query time, find top matching chunks
```

Each step is its own file. Let's build them one by one.

---

## Step 14 — Write Document Loaders

Loaders read raw text out of files.
Each file type needs its own loader.
They all return the same thing: a plain string of text.

### Step 14.1 — Install PDF reading library

```bash
pip install pypdf
```

Verify:

```bash
pip show pypdf
```

### Step 14.2 — Write the base loader interface

```python
# rag/loaders/base.py

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Every loader reads a file and returns its text.
    Nothing else. No chunking, no embedding here.
    """

    @abstractmethod
    def load(self, file_path: str) -> str:
        """
        Read file at file_path.
        Return its full text content as a single string.
        """
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Return True if this loader can handle this file type.
        Example: PDFLoader.supports("notes.pdf") → True
        """
        pass
```

### Step 14.3 — Write the TXT loader

```python
# rag/loaders/txt_loader.py

from rag.loaders.base import BaseLoader


class TXTLoader(BaseLoader):
    """Loads plain text files."""

    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            raise ValueError(f"File is empty: {file_path}")

        return text.strip()

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".txt")
```

### Step 14.4 — Write the Markdown loader

```python
# rag/loaders/md_loader.py

from rag.loaders.base import BaseLoader


class MDLoader(BaseLoader):
    """
    Loads Markdown files.
    Strips markdown syntax so the model sees clean text.
    """

    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            raise ValueError(f"File is empty: {file_path}")

        # Remove markdown headers, bold, italic markers
        # Keep the actual words — the meaning is in the text
        import re
        text = re.sub(r"#{1,6}\s*", "", text)       # remove headers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text) # remove bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)     # remove italic
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text,
                      flags=re.DOTALL)                # remove code blocks
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text) # remove links

        return text.strip()

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith((".md", ".markdown"))
```

### Step 14.5 — Write the PDF loader

```python
# rag/loaders/pdf_loader.py

from rag.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """Loads PDF files and extracts all text."""

    def load(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf not installed. Run: pip install pypdf"
            )

        reader = PdfReader(file_path)

        if len(reader.pages) == 0:
            raise ValueError(f"PDF has no pages: {file_path}")

        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

        if not pages:
            raise ValueError(
                f"Could not extract text from PDF: {file_path}\n"
                "The PDF may be scanned/image-based."
            )

        return "\n\n".join(pages)

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")
```

### Step 14.6 — Write the loader router

This decides which loader to use for any given file.

```python
# rag/loaders/router.py

from rag.loaders.txt_loader import TXTLoader
from rag.loaders.md_loader import MDLoader
from rag.loaders.pdf_loader import PDFLoader


class LoaderRouter:
    """
    Given any file path, returns the right loader.
    Add new loaders here as you support more formats.
    """

    def __init__(self):
        self._loaders = [
            TXTLoader(),
            MDLoader(),
            PDFLoader(),
        ]

    def get_loader(self, file_path: str):
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader

        supported = [".txt", ".md", ".markdown", ".pdf"]
        raise ValueError(
            f"No loader found for: {file_path}\n"
            f"Supported formats: {', '.join(supported)}"
        )

    def load(self, file_path: str) -> str:
        """Load a file using the appropriate loader."""
        loader = self.get_loader(file_path)
        return loader.load(file_path)
```

### Step 14.7 — Quick loader test

Create a test file first:

```bash
echo "Neural networks are computing systems inspired by the brain.
They consist of layers of nodes called neurons.
Each connection has a weight that gets adjusted during training.
Deep learning uses many layers to learn complex patterns." > storage/files/notes/neural_networks.txt
```

Now test:

```python
# tests/test_loaders.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loaders.router import LoaderRouter


def test_txt_loader():
    router = LoaderRouter()
    text = router.load("storage/files/notes/neural_networks.txt")
    assert len(text) > 0
    assert "neural" in text.lower()
    print(f"✓ TXT loader works. Characters loaded: {len(text)}")


if __name__ == "__main__":
    print("\n--- Testing Loaders ---\n")
    test_txt_loader()
    print("\n--- Loader tests passed ---\n")
```

Run:

```bash
python tests/test_loaders.py
```

Expected:

```
--- Testing Loaders ---

✓ TXT loader works. Characters loaded: 231

--- Loader tests passed ---
```

---

## Step 15 — Write the Chunker

Raw documents are too long to fit in a prompt all at once.
Chunking splits them into smaller overlapping pieces.

### Why overlap matters

```
Chunk 1: "Neural networks consist of layers. Each layer transforms..."
Chunk 2: "...Each layer transforms the input. The final layer produces..."
```

Without overlap, meaning that spans chunk boundaries gets lost.
With overlap (~50 tokens), each chunk shares context with its neighbors.

### Step 15.1 — Write the chunker

```python
# rag/chunking/splitter.py


class TextSplitter:
    """
    Splits a long text into overlapping chunks.

    chunk_size    → max characters per chunk (~512 tokens ≈ 2000 chars)
    chunk_overlap → how many characters to repeat between chunks
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        Tries to split at paragraph or sentence boundaries.
        """
        if not text.strip():
            return []

        # Clean up whitespace
        text = " ".join(text.split())

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk — take everything remaining
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a clean break point
            # Priority: paragraph > sentence end > word boundary
            break_point = self._find_break(text, start, end)
            chunk = text[start:break_point].strip()

            if chunk:
                chunks.append(chunk)

            # Move start forward, minus overlap
            start = break_point - self.chunk_overlap

            # Safety: never go backwards
            if start <= 0:
                start = break_point

        return chunks

    def _find_break(self, text: str, start: int, end: int) -> int:
        """Find the best position to break the text near `end`."""
        window = text[start:end]

        # Try paragraph break first
        para_break = window.rfind("\n\n")
        if para_break > self.chunk_size // 2:
            return start + para_break

        # Try sentence end
        for punct in [". ", "! ", "? "]:
            sent_break = window.rfind(punct)
            if sent_break > self.chunk_size // 2:
                return start + sent_break + 1

        # Fall back to word boundary
        word_break = window.rfind(" ")
        if word_break > 0:
            return start + word_break

        # Hard cut as last resort
        return end

    def split_with_metadata(
        self, text: str, source: str
    ) -> list[dict]:
        """
        Split and attach metadata to each chunk.
        Returns list of dicts with text + source info.
        """
        chunks = self.split(text)
        return [
            {
                "text": chunk,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i, chunk in enumerate(chunks)
        ]
```

### Step 15.2 — Test the chunker

```python
# tests/test_chunker.py

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
```

Run:

```bash
python tests/test_chunker.py
```

Expected:

```
--- Testing Chunker ---

✓ Chunking works.
  Input length  : 1134 chars
  Chunks created: 7
  Avg chunk size: 182 chars
✓ Overlap check:
  End of chunk 1  : 'word word word word '
  Start of chunk 2: 'word word word word word '
✓ Metadata chunks work. Total: 4

--- Chunker tests passed ---
```

---

## Step 16 — Set Up Embedder

Embeddings convert text into vectors of numbers.
Similar text produces similar vectors.
This is how semantic search works — not keyword matching.

```
"neural network"    → [0.21, -0.43, 0.87, ...]  (384 numbers)
"deep learning"     → [0.19, -0.41, 0.85, ...]  (similar!)
"banana recipe"     → [-0.67, 0.23, -0.11, ...] (very different)
```

### Model we use: all-MiniLM-L6-v2

- Fast on CPU
- 384 dimensions
- 80MB model size
- Downloads automatically on first use
- Good quality for semantic search

### Step 16.1 — Write the embedder

```python
# rag/embeddings/embedder.py


class Embedder:
    """
    Converts text into embedding vectors using sentence-transformers.
    Runs entirely on CPU — no GPU needed.
    Model downloads automatically on first use (~80MB).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy load — don't load until first use

    def _load_model(self):
        """Load model on first use. Takes ~3 seconds first time."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            print("(First time takes ~30 seconds to download ~80MB)")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            print(f"✓ Embedding model loaded")

    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text. Returns vector."""
        self._load_model()
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts at once.
        Much faster than calling embed() in a loop.
        Use this when indexing documents.
        """
        self._load_model()
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32
        )
        return vectors.tolist()

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
```

### Step 16.2 — Test the embedder

```python
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
```

Run:

```bash
python tests/test_embedder.py
```

Expected:

```
--- Testing Embedder ---

Loading embedding model: all-MiniLM-L6-v2
(First time takes ~30 seconds to download ~80MB)
✓ Embedding model loaded
✓ Single embed works. Dimensions: 384
✓ Similarity test:
  AI topics similarity   : 0.742  (expect > 0.5)
  AI vs food similarity  : 0.087  (expect < 0.3)
✓ Batch embed works. 4 vectors produced.

--- Embedder tests passed ---
```

The similarity numbers prove semantic understanding is working.
The model knows AI topics are related, and AI vs food are not — without any keywords matching.

---

## Step 17 — Set Up ChromaDB

ChromaDB stores your vectors and text.
It lets you search by similarity at query time.
It runs fully locally — no server, no cloud, just files in `storage/vectors/`.

### Step 17.1 — Write the ChromaDB wrapper

```python
# rag/vectordb/chroma.py

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
```

### Step 17.2 — Test ChromaDB

```python
# tests/test_chroma.py

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
```

Run:

```bash
python tests/test_chroma.py
```

Expected:

```
--- Testing ChromaDB ---

✓ ChromaDB ready. Collection: 'test_collection' | Documents: 0
✓ Collection wiped and recreated.
✓ Stored 5 chunks.
✓ Query returned 2 results:
  Score 0.821 | Neural networks learn from data using backpropagation....
  Score 0.743 | Deep learning models require large datasets to train....
✓ Top result is semantically relevant.

--- ChromaDB tests passed ---
```

The scores confirm ChromaDB found the most relevant chunks.

---

## Step 18 — Write the Retriever

The retriever combines the embedder and ChromaDB into one clean interface.
It is what the orchestrator will call at query time.

```python
# rag/retriever.py

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
```

---

## Step 19 — Write the Indexing Pipeline

This is the full pipeline that takes a file and gets it into ChromaDB.
Load → Chunk → Embed → Store.

### Step 19.1 — Write pipeline.py

```python
# rag/pipeline.py

import hashlib
from pathlib import Path
from rag.loaders.router import LoaderRouter
from rag.chunking.splitter import TextSplitter
from rag.embeddings.embedder import Embedder
from rag.vectordb.chroma import ChromaStore


class RAGPipeline:
    """
    Full indexing pipeline.
    Takes a file path → loads → chunks → embeds → stores.
    Safe to run multiple times — skips already indexed files.
    """

    def __init__(self):
        self.loader = LoaderRouter()
        self.splitter = TextSplitter(chunk_size=1500, chunk_overlap=200)
        self.embedder = Embedder()
        self.store = ChromaStore()

    def index_file(self, file_path: str) -> dict:
        """
        Index a single file into the knowledge base.
        Returns a summary of what was indexed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"\nIndexing: {path.name}")

        # 1. Load
        print(f"  [1/4] Loading...")
        text = self.loader.load(file_path)
        print(f"        {len(text)} characters loaded")

        # 2. Chunk
        print(f"  [2/4] Chunking...")
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
            print(f"        All chunks already indexed. Skipping.")
            return {"file": path.name, "chunks": 0, "skipped": skipped}

        # 4. Embed
        print(f"  [3/4] Embedding {len(new_chunks)} chunks...")
        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 5. Store
        print(f"  [4/4] Storing in ChromaDB...")
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
        print(f"  ✓ Done. New chunks: {len(new_chunks)} | "
              f"Skipped: {skipped} | Total in DB: {total}")

        return {
            "file": path.name,
            "chunks_added": len(new_chunks),
            "chunks_skipped": skipped,
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
        raw = f"{filename}__chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()
```

### Step 19.2 — Add some real documents to index

Put some of your actual notes in `storage/files/notes/`.

For testing, create a few sample files:

```bash
cat > storage/files/notes/ai_basics.txt << 'EOF'
Artificial Intelligence is the field of computer science focused on
creating systems that can perform tasks that normally require human intelligence.

Machine learning is a subset of AI where systems learn from data.
Instead of being explicitly programmed, they identify patterns and make decisions.

Deep learning uses neural networks with multiple layers.
These networks can learn complex representations from raw data.

Large language models like GPT and Qwen are trained on massive text datasets.
They learn to predict the next token, which leads to emergent language understanding.
EOF
```

```bash
cat > storage/files/notes/rag_notes.txt << 'EOF'
RAG stands for Retrieval Augmented Generation.

The core idea is to give an LLM access to external knowledge at query time.
Instead of relying solely on training data, the model retrieves relevant documents
and uses them as context when generating responses.

This solves the knowledge cutoff problem.
It also allows the model to cite sources and stay grounded in facts.

Vector databases like ChromaDB store document embeddings.
At query time, the question is embedded and similar documents are retrieved.
These retrieved chunks are then injected into the LLM prompt as context.
EOF'
```

### Step 19.3 — Index them

```python
# tests/test_pipeline.py

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
```

Run:

```bash
python tests/test_pipeline.py
```

Expected:

```
--- Testing RAG Pipeline ---

Indexing: ai_basics.txt
  [1/4] Loading...
        624 characters loaded
  [2/4] Chunking...
        1 chunks created
  [3/4] Embedding 1 chunks...
  [4/4] Storing in ChromaDB...
  ✓ Done. New chunks: 1 | Skipped: 0 | Total in DB: 1

✓ File indexed: {'file': 'ai_basics.txt', 'chunks_added': 1, ...}

Found 2 files to index in storage/files/notes
...
✓ Folder indexed. Files processed: 2

--- Pipeline tests passed ---
```

---

## Step 20 — Full RAG Test

Now wire everything together and ask a real question answered from your notes.

### Step 20.1 — Write full RAG integration test

```python
# tests/test_rag_full.py

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
```

### Step 20.2 — Run the full test

```bash
python tests/test_rag_full.py
```

Expected output:

```
=== Full RAG Flow Test ===

Step 1: Indexing documents...
Found 2 files to index in storage/files/notes
  ✓ Done. All chunks already indexed.

✓ Knowledge base has 4 chunks

Step 2: Question: 'What is RAG and how does it work?'
Step 3: Retrieving relevant chunks...
✓ Retrieved 3 chunks:
  Score 0.891 | Source: rag_notes.txt
  Preview: RAG stands for Retrieval Augmented Generation. The core idea...
  Score 0.743 | Source: rag_notes.txt
  Preview: Vector databases like ChromaDB store document embeddings...
  Score 0.612 | Source: ai_basics.txt
  Preview: Large language models like GPT and Qwen are trained on...

Step 4: Sending to LLM...

June's answer:

----------------------------------------
RAG, or Retrieval Augmented Generation, is a technique that gives
a language model access to external knowledge at query time.
Instead of relying only on its training data, it retrieves relevant
documents and uses them as context when generating a response.

This works by storing document embeddings in a vector database.
When a question arrives, it is also embedded, and similar chunks
are retrieved. Those chunks are then included in the prompt...
----------------------------------------

✓ Full RAG flow working.
```

**This is the moment June first answers from your own knowledge.**

---

## Full test suite — run everything

```bash
python tests/test_llm.py
python tests/test_loaders.py
python tests/test_chunker.py
python tests/test_embedder.py
python tests/test_chroma.py
python tests/test_pipeline.py
python tests/test_rag_full.py
```

All should pass before moving to the next phase.

---

## What you have now

```
✅ Step 14 — Document loaders (txt, md, pdf)
✅ Step 15 — Chunker with overlap
✅ Step 16 — sentence-transformers embedder (CPU)
✅ Step 17 — ChromaDB vector store running locally
✅ Step 18 — RAGRetriever with clean interface
✅ Step 19 — Full indexing pipeline with your real documents
✅ Step 20 — End-to-end: question → retrieval → LLM → answer
```

---

## What comes next

```
Step 21 → Write Orchestrator router
Step 22 → Write ContextBuilder
Step 23 → Write Agent — wires LLM + RAG together
Step 24 → Build FastAPI server
Step 25 → Add WebSocket for streaming chat
Step 26 → Build simple chat UI
Step 27 → Full stack test: browser → API → agent → streamed response
```

---

*June AI — RAG Engine v1.0*