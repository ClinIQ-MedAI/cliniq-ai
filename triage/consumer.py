"""
Results consumer.

Subscribes to the shared results channel and, for every ResultMessage:
  * writes an audit row (traceability), and
  * upserts a worklist case (severity-ranked review queue).

Runs the broker consume loop in a background thread so it can live inside the
triage FastAPI app. This is the backend-side counterpart to the per-service
JobWorker.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from messaging.base import Broker
from messaging.config import QueueConfig
from messaging.schemas import ResultMessage

from audit.store import AuditStore, AuditEntry
from .severity import priority_of, summarize_result
from .store import TriageStore, WorklistCase


class ResultConsumer:
    def __init__(
        self,
        broker: Broker,
        config: QueueConfig,
        audit_store: AuditStore,
        triage_store: TriageStore,
        *,
        on_critical: Optional[Callable[[WorklistCase], None]] = None,
    ):
        self.broker = broker
        self.config = config
        self.audit = audit_store
        self.triage = triage_store
        self.on_critical = on_critical
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    def _handle(self, raw: str) -> None:
        try:
            msg = ResultMessage.from_json(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"[triage] dropping malformed result: {exc}")
            return

        summary = summarize_result(msg.result)
        score, level = priority_of(msg.result, msg.status)

        # 1) audit
        self.audit.record(AuditEntry(
            job_id=msg.job_id,
            modality=msg.modality,
            status=msg.status,
            source="queue",
            model_version=msg.model_version,
            image_sha256=msg.image_sha256,
            patient_id=msg.patient_id,
            top_finding=summary["top_finding"],
            max_confidence=summary["max_confidence"],
            urgency=level,
            num_findings=summary["num_findings"],
            duration_ms=msg.duration_ms,
            worker=msg.worker,
            error=msg.error,
            created_at=msg.finished_at,
        ))

        # 2) worklist
        case = WorklistCase(
            job_id=msg.job_id,
            modality=msg.modality,
            priority_score=score,
            priority_level=level,
            status=msg.status,
            top_finding=summary["top_finding"],
            max_confidence=summary["max_confidence"],
            patient_id=msg.patient_id,
            summary=(msg.result or {}).get("summary") if msg.result else msg.error,
            created_at=msg.finished_at,
        )
        self.triage.upsert(case)

        flag = "🔴 CRITICAL" if level == "CRITICAL" else ("🟠 HIGH" if level == "HIGH" else level)
        print(f"[triage] {flag} {msg.modality} job={msg.job_id[:8]} "
              f"score={score} finding={summary['top_finding']}")

        if level == "CRITICAL" and self.on_critical:
            try:
                self.on_critical(case)
            except Exception as exc:  # noqa: BLE001
                print(f"[triage] critical notifier failed: {exc}")

    # ------------------------------------------------------------------ #
    def run_forever(self) -> None:
        print(f"[triage] consuming '{self.config.result_channel}' "
              f"(backend={self.config.backend})")
        self.broker.consume(
            self.config.result_channel,
            self._handle,
            group=f"{self.config.prefix}-triage",
            consumer="triage-consumer",
            block_ms=self.config.block_ms,
            stop_event=self._stop,
        )

    def start_background(self) -> "ResultConsumer":
        if self._thread and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self.run_forever, name="cliniq-triage-consumer", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self.broker.close()
        except Exception:  # noqa: BLE001
            pass
