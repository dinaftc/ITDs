from .base import EXACT_FALLBACK, SIMILARITY_ONLY, ITDStore
from .memory import MemoryStore


def make_store(backend: str, embedder, namespace: str = "itd", *, reset: bool = False,
               dsn: str | None = None, url: str | None = None) -> ITDStore:
    """backend: memory | postgres | qdrant. Database drivers are imported lazily."""
    if backend == "memory":
        return MemoryStore(embedder)
    if backend == "postgres":
        from .postgres import PostgresStore
        return PostgresStore(embedder, dsn, namespace, reset=reset)
    if backend == "qdrant":
        from .qdrant import QdrantStore
        return QdrantStore(embedder, url, namespace, reset=reset)
    raise ValueError(f"unknown store backend: {backend}")
