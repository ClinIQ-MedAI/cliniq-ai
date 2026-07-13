"""
Critical-finding alerts.

When the triage consumer classifies a case as CRITICAL (a fracture, a
pneumothorax, ...), this fires a simple, multi-channel alert:

  * persists it to an ``alerts`` table so the dashboard can show a live feed,
  * POSTs a compact JSON payload to ``ALERT_WEBHOOK_URL`` if configured — your
    .NET backend, or a Slack/Teams incoming webhook, can receive it, and
  * prints a console banner.

Deliberately lightweight: no external alerting dependency and no broker of its
own. The webhook call uses a short timeout and never propagates errors, so a
slow or broken endpoint can't stall result consumption in the triage thread.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "triage.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AlertStore:
    """Persistence for fired alerts (own connection to the shared triage DB)."""

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
                CREATE TABLE IF NOT EXISTS alerts (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id         TEXT,
                    modality       TEXT,
                    priority_level TEXT,
                    priority_score INTEGER,
                    top_finding    TEXT,
                    patient_id     TEXT,
                    summary        TEXT,
                    delivery       TEXT,
                    acknowledged   INTEGER NOT NULL DEFAULT 0,
                    created_at     TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts (created_at DESC)"
            )
            self._conn.commit()

    def record(self, alert: Dict[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO alerts
                    (job_id, modality, priority_level, priority_score, top_finding,
                     patient_id, summary, delivery, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.get("job_id"), alert.get("modality"),
                    alert.get("priority_level"), alert.get("priority_score"),
                    alert.get("top_finding"), alert.get("patient_id"),
                    alert.get("summary"), alert.get("delivery"),
                    alert.get("created_at") or _now_iso(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 50, include_acknowledged: bool = True) -> List[Dict[str, Any]]:
        clause = "" if include_acknowledged else "WHERE acknowledged = 0"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM alerts {clause} ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def unacknowledged_count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged = 0"
            ).fetchone()[0]

    def acknowledge(self, alert_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CriticalNotifier:
    """
    Fires alerts for CRITICAL cases across store + webhook + console.

    Wire it into the results consumer as ``on_critical=notifier.notify``.
    ``webhook_url`` defaults to the ``ALERT_WEBHOOK_URL`` environment variable.
    """

    def __init__(
        self,
        store: Optional[AlertStore] = None,
        *,
        webhook_url: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.store = store or AlertStore()
        self.webhook_url = webhook_url if webhook_url is not None else os.getenv("ALERT_WEBHOOK_URL")
        self.timeout = timeout

    def _post_webhook(self, payload: Dict[str, Any]) -> str:
        if not self.webhook_url:
            return "webhook:disabled"
        try:
            import requests
            resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            return f"webhook:{resp.status_code}"
        except Exception as exc:  # noqa: BLE001 - never stall consuming on a bad endpoint
            return f"webhook:error({type(exc).__name__})"

    def notify(self, case) -> None:
        """Accepts a triage WorklistCase (or any object with the same fields)."""
        payload = {
            "type": "critical_finding",
            "job_id": getattr(case, "job_id", None),
            "modality": getattr(case, "modality", None),
            "priority_level": getattr(case, "priority_level", None),
            "priority_score": getattr(case, "priority_score", None),
            "top_finding": getattr(case, "top_finding", None),
            "patient_id": getattr(case, "patient_id", None),
            "summary": getattr(case, "summary", None),
            "timestamp": _now_iso(),
        }

        delivery = self._post_webhook(payload)

        # Console banner — visible in the triage service logs.
        print(
            "\n" + "🚨" * 20 +
            f"\n[ALERT] CRITICAL {payload['modality']} · job={str(payload['job_id'])[:8]} · "
            f"finding={payload['top_finding']} · patient={payload['patient_id']} · {delivery}\n" +
            "🚨" * 20
        )

        try:
            self.store.record({**payload, "priority_level": payload["priority_level"],
                               "delivery": delivery, "created_at": payload["timestamp"]})
        except Exception as exc:  # noqa: BLE001
            print(f"[alerts] failed to persist alert: {exc}")
