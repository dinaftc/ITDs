"""Text embedders used for template similarity (Section 3.3, Retrieve)."""
from __future__ import annotations

import hashlib
import re

import numpy as np


class SentenceEmbedder:
    """all-MiniLM-L6-v2 by default, as in the paper (384 dimensions)."""

    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. No model download; used by
    the tests and the dry-run configuration, not by the paper's experiments."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


def make_embedder(name: str):
    return HashEmbedder() if name == "hash" else SentenceEmbedder(name)
