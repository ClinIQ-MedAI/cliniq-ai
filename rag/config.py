"""RAG configuration."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = _ROOT / "documents"          # source knowledge base (txt/md/pdf)
INDEX_DIR = _ROOT / "index"                  # persisted FAISS index + metadata

# Multilingual embedder — handles Arabic and English (and cross-lingual
# retrieval: an Arabic question can match an English passage). Small enough to
# embed on CPU; runs on GPU automatically if one is visible.
EMBED_MODEL = "intfloat/multilingual-e5-small"

CHUNK_CHARS = 900        # ~1 short paragraph per chunk
CHUNK_OVERLAP = 150      # keep sentence continuity across chunk boundaries
TOP_K = 4                # passages returned per query
MIN_SCORE = 0.72         # cosine floor — below this a passage is "not relevant"
