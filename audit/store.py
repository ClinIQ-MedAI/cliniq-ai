"""
SQLite-backed audit store for AI predictions.

One row per prediction with everything needed for traceability:
who/what/when, which model version, the input fingerprint, the outcome, and
how long it took. Pure stdlib (sqlite3), thread-safe via a single lock so the
queue consumer (background thread) and the API (event loop) can share it.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "audit.db"


@dataclass
class AuditEntry:
    job_id: str
    modality: str
    status: str                     # completed | failed
    source: str = "queue"           # queue | http
    model_version: Optional[str] = None
    image_sha256: Optional[str] = None
    patient_id: Optional[str] = None
    top_finding: Optional[str] = None
    max_confidence: Optional[float] = None
    urgency: Optional[str] = None
    num_findings: Optional[int] = None
    duration_ms: Optional[float] = None
    worker: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None  # ISO; stamped on record if absent


class AuditStore:
    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or _DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id        TEXT,
                    modality      TEXT,
                    status        TEXT,
                    source        TEXT,
                    model_version TEXT,
                    image_sha256  TEXT,
                    patient_id    TEXT,
                    top_finding   TEXT,
                    max_confidence REAL,
                    urgency       TEXT,
                    num_findings  INTEGER,
                    duration_ms   REAL,
                    worker        TEXT,
                    error         TEXT,
                    created_at    TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions (created_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_patient ON predictions (patient_id, created_at)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    def record(self, entry: AuditEntry) -> int:
        data = asdict(entry)
        if not data.get("created_at"):
            data["created_at"] = datetime.now(timezone.utc).isoformat()
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
                list(data.values()),
            )
            self._conn.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------ #
    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def by_patient(self, patient_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            by_modality = self._conn.execute(
                "SELECT modality, COUNT(*) c FROM predictions GROUP BY modality"
            ).fetchall()
            by_status = self._conn.execute(
                "SELECT status, COUNT(*) c FROM predictions GROUP BY status"
            ).fetchall()
            by_urgency = self._conn.execute(
                "SELECT urgency, COUNT(*) c FROM predictions GROUP BY urgency"
            ).fetchall()
            avg_ms = self._conn.execute(
                "SELECT AVG(duration_ms) FROM predictions WHERE status='completed'"
            ).fetchone()[0]
            models = self._conn.execute(
                "SELECT DISTINCT modality, model_version FROM predictions "
                "WHERE model_version IS NOT NULL"
            ).fetchall()
        return {
            "total_predictions": total,
            "by_modality": {r["modality"]: r["c"] for r in by_modality},
            "by_status": {r["status"]: r["c"] for r in by_status},
            "by_urgency": {r["urgency"]: r["c"] for r in by_urgency},
            "avg_duration_ms": round(avg_ms, 2) if avg_ms else None,
            "model_versions": {r["modality"]: r["model_version"] for r in models},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
