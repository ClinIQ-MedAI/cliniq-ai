"""
ClinIQ triage worklist.

Consumes the shared results stream (`cliniq:results`) produced by the AI
workers and turns it into two things radiology AI products (Aidoc, Qure.ai)
consider table stakes:

  1. A severity-ranked worklist — critical findings (fracture, pneumothorax,
     decay) float to the top so they are reviewed first.
  2. An audit trail — every prediction is logged with its model version and
     input fingerprint for traceability.

It is a *consumer* of the queue, so it doubles as a reference implementation of
the backend side and requires no changes to the inference services.
"""

from .severity import priority_of, summarize_result, PriorityLevel
from .store import TriageStore, WorklistCase

__all__ = [
    "priority_of",
    "summarize_result",
    "PriorityLevel",
    "TriageStore",
    "WorklistCase",
]
