"""
Vector store for the ClinIQ RAG knowledge base.

Embeds passages with a multilingual sentence-transformer and searches them with
FAISS (cosine similarity via normalized inner-product). The index + metadata are
persisted to disk so the chatbot loads them instantly at startup instead of
re-embedding every launch.

Embedding + FAISS are optional-at-runtime: if the packages or the built index
are missing, callers get an empty result and the chatbot simply answers without
retrieval — RAG is an enhancement, never a hard dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config


@dataclass
class Passage:
    text: str
    source: str          # file the passage came from
    title: str           # human-readable document title
    score: float = 0.0   # cosine similarity to the query (set at search time)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "source": self.source,
                "title": self.title, "score": round(self.score, 4)}


class VectorStore:
    """Persisted embedding index over the knowledge-base passages."""

    _INDEX_FILE = "index.faiss"
    _META_FILE = "passages.json"

    def __init__(self, index_dir: Optional[Path] = None, model_name: Optional[str] = None):
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self.model_name = model_name or config.EMBED_MODEL
        self._model = None
        self._index = None
        self._passages: List[Passage] = []

    # ------------------------------------------------------------------ #
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: List[str], *, is_query: bool) -> "np.ndarray":  # noqa: F821
        import numpy as np
        # intfloat/e5 models want explicit "query:"/"passage:" prefixes.
        prefix = "query: " if is_query else "passage: "
        vecs = self._get_model().encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,      # cosine == inner product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    # ------------------------------------------------------------------ #
    def build(self, passages: List[Passage]) -> None:
        """Embed all passages and write the FAISS index + metadata to disk."""
        import faiss

        if not passages:
            raise ValueError("no passages to index")
        self._passages = passages
        vectors = self._embed([p.text for p in passages], is_query=False)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / self._INDEX_FILE))
        with open(self.index_dir / self._META_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                [{"text": p.text, "source": p.source, "title": p.title} for p in passages],
                fh, ensure_ascii=False, indent=2,
            )

    def load(self) -> bool:
        """Load a previously-built index. Returns False if none exists."""
        import faiss

        idx_path = self.index_dir / self._INDEX_FILE
        meta_path = self.index_dir / self._META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, encoding="utf-8") as fh:
            self._passages = [Passage(**m) for m in json.load(fh)]
        return True

    def search(self, query: str, k: int = config.TOP_K,
               min_score: float = config.MIN_SCORE) -> List[Passage]:
        """Return the top-k passages above the similarity floor."""
        if self._index is None or not self._passages:
            return []
        qv = self._embed([query], is_query=True)
        scores, idxs = self._index.search(qv, min(k, len(self._passages)))
        out: List[Passage] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or score < min_score:
                continue
            p = self._passages[int(idx)]
            out.append(Passage(text=p.text, source=p.source, title=p.title, score=float(score)))
        return out

    @property
    def size(self) -> int:
        return len(self._passages)
