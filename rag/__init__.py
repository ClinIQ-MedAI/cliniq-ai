"""
ClinIQ RAG — retrieval-augmented grounding for the chatbot.

Grounds the chatbot's medical answers in a curated knowledge base (patient-
education material, red-flag guidance, medication-safety notes) so responses are
sourced and citable instead of relying on the model's memory alone.

Public API:
    from rag import retrieve, grounding_block, sources
"""

from __future__ import annotations

from .retriever import grounding_block, retrieve, sources

__all__ = ["retrieve", "grounding_block", "sources"]
