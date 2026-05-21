from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

from api.routes.chat import router as chat_router
from services import services
from orchestrator.agent import Agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    
    Initializes all singleton services once at server startup,
    preventing the expensive 30-40s initialization delay that
    was occurring on every WebSocket connection.
    
    Teardown runs on server shutdown.
    """
    print("\n" + "="*60)
    print("Starting up June AI services...")
    print("="*60)

    # Load embedding model once — this is the ~30s step
    print("Loading embedding model: all-MiniLM-L6-v2...")
    services.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Embedding model ready.")

    # Connect to ChromaDB once
    print("Connecting to ChromaDB...")
    services.chroma_client = chromadb.PersistentClient(path="storage/vectors")
    services.collection = services.chroma_client.get_or_create_collection(
        name="june_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ ChromaDB ready. Stored documents: {services.collection.count()}")

    # Build the agent once with the already-loaded dependencies
    # Agent.__init__ will print its own initialization message
    services.agent = Agent(
        embed_model=services.embed_model,
        collection=services.collection
    )
    print("="*60)
    print("Startup complete. Server ready.\n")

    yield  # Server runs here

    # Teardown (runs on shutdown)
    print("\n" + "="*60)
    print("Shutting down June AI services...")
    print("="*60)
    print("✓ Goodbye!")


app = FastAPI(
    title="June AI",
    description="Personal AI assistant API",
    version="0.1.0",
    lifespan=lifespan
)

# Allow browser to connect from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount API routes
app.include_router(chat_router, prefix="/api")

# Server UI static files
ui_path = Path("ui")
if ui_path.exists():
    app.mount("/static", StaticFiles(directory="ui"), name="static")

    @app.get("/")
    async def serve_ui():
        return FileResponse("ui/index.html")
    
@app.get("/health")
async def health():
    return {"status": "ok", "service": "June AI"}