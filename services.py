"""
Singleton services for June AI.

All long-lived objects (embedding model, ChromaDB, agent) are initialized
once at server startup and stored here. This prevents the 30-40s load delay
and massive CPU/RAM spikes that occur when reinitializing on every request.

The lifespan context manager in api/server.py initializes these services
at startup and tears them down at shutdown.
"""

from typing import Optional
from sentence_transformers import SentenceTransformer
import chromadb


class AppServices:
    """Holds all singleton service instances."""
    
    embed_model: Optional[SentenceTransformer] = None
    chroma_client: Optional[chromadb.PersistentClient] = None
    collection = None
    agent = None  # JuneAgent instance


services = AppServices()
