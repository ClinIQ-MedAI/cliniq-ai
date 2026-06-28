"""
ClinIQ audit layer.

Lightweight, dependency-free (stdlib sqlite3) audit trail for every AI
prediction, plus model-version fingerprinting. Designed to satisfy the
"production-grade / traceable" requirement without pulling in heavy MLOps stacks.

Public API:
    AuditStore            -> SQLite-backed prediction log
    model_version(path)   -> stable fingerprint for a weights file
    model_version_for_modality(modality) -> fingerprint via the known model map
"""

from .store import AuditStore, AuditEntry
from .versioning import model_version, model_version_for_modality, KNOWN_MODEL_PATHS

__all__ = [
    "AuditStore",
    "AuditEntry",
    "model_version",
    "model_version_for_modality",
    "KNOWN_MODEL_PATHS",
]
