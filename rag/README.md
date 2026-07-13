# RAG — Retrieval-Augmented Grounding

Grounds the chatbot's medical answers in a curated knowledge base instead of the
model's memory alone. This reduces hallucination and lets answers cite a source.

```
question --> multilingual embedding --> FAISS search --> top-k passages
                                                              |
           LLM answer  <-- system prompt + grounded passages -+
```

## Layout

| File | Role |
|---|---|
| `documents/` | The knowledge base (patient-education, red-flags, medication safety, per-modality guidance, clinic FAQ) |
| `ingest.py` | Chunk + embed the documents, build the FAISS index |
| `store.py` | `VectorStore` — embed, persist, cosine search |
| `retriever.py` | `retrieve()` + `grounding_block()` used by the chatbot |
| `config.py` | Embedding model, chunk size, top-k, score floor |

**Embedding model:** `intfloat/multilingual-e5-small` — cross-lingual, so an
Arabic question retrieves the right passage even from an English document.

## Build the index

```bash
python -m rag.ingest        # writes rag/index/index.faiss + passages.json
```

## Wiring

`chatbot-app` calls `rag.grounding_block(message, language)` and appends the
retrieved, source-labelled passages to the LLM system prompt. It is opt-in and
graceful: set `RAG_ENABLED=0` to disable, and if the index or dependencies are
missing, retrieval returns nothing and the chatbot answers normally.
