"""
High-level retrieval API used by the chatbot.

`retrieve(query)` returns relevant knowledge-base passages; `grounding_block()`
formats them into a system-prompt section the LLM can cite. Everything degrades
gracefully — if the index or dependencies are missing, retrieval returns nothing
and the chatbot answers normally (just without grounded citations).
"""

from __future__ import annotations

import threading
from typing import List, Optional

from . import config
from .store import Passage, VectorStore

_store: Optional[VectorStore] = None
_lock = threading.Lock()
_load_failed = False


def _get_store() -> Optional[VectorStore]:
    global _store, _load_failed
    if _store is not None or _load_failed:
        return _store
    with _lock:
        if _store is None and not _load_failed:
            try:
                store = VectorStore()
                if store.load():
                    _store = store
                else:
                    _load_failed = True   # no index built yet
            except Exception as exc:  # noqa: BLE001 - deps missing, etc.
                print(f"[rag] retrieval disabled: {exc}")
                _load_failed = True
    return _store


def retrieve(query: str, k: int = config.TOP_K) -> List[Passage]:
    """Return up to k relevant passages for the query (empty if RAG unavailable)."""
    store = _get_store()
    if store is None or not (query or "").strip():
        return []
    try:
        return store.search(query, k=k)
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] search failed: {exc}")
        return []


def grounding_block(query: str, k: int = config.TOP_K, language: str = "en") -> str:
    """
    Build a system-prompt section from retrieved passages, or '' if none match.

    The chatbot appends this to its system message so the LLM answers from the
    knowledge base and can cite sources instead of relying on memory alone.
    """
    passages = retrieve(query, k=k)
    if not passages:
        return ""

    if language == "ar":
        header = ("مصادر طبية موثوقة من قاعدة المعرفة (استند إليها في إجابتك "
                  "واذكر المصدر عند الاقتباس، ولا تختلق معلومات غير موجودة فيها):")
    else:
        header = ("Trusted medical sources from the knowledge base (ground your "
                  "answer in these and cite the source when you use one; do not "
                  "invent facts not present here):")

    lines = [header, ""]
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p.title} ({p.source}):")
        lines.append(p.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def sources(query: str, k: int = config.TOP_K) -> List[dict]:
    """Lightweight list of matched sources (for a UI 'references' footer)."""
    return [{"title": p.title, "source": p.source, "score": round(p.score, 3)}
            for p in retrieve(query, k=k)]
