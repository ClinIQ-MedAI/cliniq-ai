"""
Build the RAG index from the knowledge-base documents.

Loads every .txt / .md / .pdf under rag/documents/, splits each into overlapping
character chunks, embeds them, and writes the FAISS index. Re-run this whenever
the documents change.

    python -m rag.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from . import config
from .store import Passage, VectorStore


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read PDF {path.name}: {exc}")
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, size: int, overlap: int) -> List[str]:
    """Overlapping character-window chunks, split on paragraph boundaries first."""
    text = text.strip()
    if not text:
        return []
    # Prefer paragraph boundaries; fall back to a sliding window for long blocks.
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= size:
                buf = para
            else:
                # Long paragraph — slide a window over it.
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + size])
                    start += size - overlap
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _title_from(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return path.stem.replace("_", " ").title()


def build_index() -> int:
    docs_dir = Path(config.DOCUMENTS_DIR)
    files = sorted(
        p for p in docs_dir.rglob("*")
        if p.suffix.lower() in (".txt", ".md", ".pdf") and p.is_file()
    )
    if not files:
        print(f"No documents found in {docs_dir} — add .txt/.md/.pdf files and re-run.")
        return 0

    passages: List[Passage] = []
    for path in files:
        text = _read_document(path)
        if not text.strip():
            continue
        title = _title_from(path, text)
        rel = str(path.relative_to(docs_dir))
        for chunk in _chunk(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP):
            passages.append(Passage(text=chunk, source=rel, title=title))
        print(f"  ✓ {rel}: {sum(1 for p in passages if p.source == rel)} chunks")

    if not passages:
        print("No text extracted from documents.")
        return 0

    print(f"\nEmbedding {len(passages)} passages with {config.EMBED_MODEL} ...")
    store = VectorStore()
    store.build(passages)
    print(f"✓ Index written to {config.INDEX_DIR} ({len(passages)} passages from {len(files)} docs)")
    return len(passages)


if __name__ == "__main__":
    sys.exit(0 if build_index() else 1)
