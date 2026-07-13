"""
Worklist persistence (SQLite).

Holds one row per case, ordered by priority score then recency. Cases can be
acknowledged (marked reviewed) so the active worklist shrinks as clinicians
work through it — mirroring a PACS worklist.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "triage.db"


@dataclass
class WorklistCase:
    job_id: str
    modality: str
    priority_score: int
    priority_level: str
    status: str                 # completed | failed
    top_finding: Optional[str] = None
    max_confidence: Optional[float] = None
    patient_id: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[str] = None


class TriageStore:
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
                CREATE TABLE IF NOT EXISTS worklist (
                    job_id         TEXT PRIMARY KEY,
                    modality       TEXT,
                    priority_score INTEGER,
                    priority_level TEXT,
                    status         TEXT,
                    top_finding    TEXT,
                    max_confidence REAL,
                    patient_id     TEXT,
                    summary        TEXT,
                    acknowledged   INTEGER NOT NULL DEFAULT 0,
                    created_at     TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_worklist_order "
                "ON worklist (acknowledged, priority_score DESC, created_at DESC)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    def upsert(self, case: WorklistCase) -> None:
        created = case.created_at or datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO worklist
                    (job_id, modality, priority_score, priority_level, status,
                     top_finding, max_confidence, patient_id, summary, acknowledged, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    modality=excluded.modality,
                    priority_score=excluded.priority_score,
                    priority_level=excluded.priority_level,
                    status=excluded.status,
                    top_finding=excluded.top_finding,
                    max_confidence=excluded.max_confidence,
                    patient_id=excluded.patient_id,
                    summary=excluded.summary
                """,
                (case.job_id, case.modality, case.priority_score, case.priority_level,
                 case.status, case.top_finding, case.max_confidence, case.patient_id,
                 case.summary, created),
            )
            self._conn.commit()

    def worklist(self, limit: int = 100, include_acknowledged: bool = False) -> List[Dict[str, Any]]:
        clause = "" if include_acknowledged else "WHERE acknowledged = 0"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM worklist {clause} "
                "ORDER BY acknowledged ASC, priority_score DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def acknowledge(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE worklist SET acknowledged = 1 WHERE job_id = ?", (job_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT priority_level, COUNT(*) c FROM worklist "
                "WHERE acknowledged = 0 GROUP BY priority_level"
            ).fetchall()
            pending = self._conn.execute(
                "SELECT COUNT(*) FROM worklist WHERE acknowledged = 0"
            ).fetchone()[0]
        return {"pending": pending, **{r["priority_level"]: r["c"] for r in rows}}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
