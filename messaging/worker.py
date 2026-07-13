"""
Generic job worker.

A JobWorker consumes JobMessages from one modality queue, hands the decoded
image to an inference callable, and publishes a ResultMessage back. It is broker
agnostic and runs its consume loop in a background thread so it can live inside a
FastAPI process alongside the existing HTTP endpoints.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from .base import Broker
from .config import QueueConfig
from .schemas import JobMessage, ResultMessage

# An inference callable takes raw image bytes + options and returns a result dict.
InferenceFn = Callable[[bytes, Dict], Dict]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobWorker:
    def __init__(
        self,
        broker: Broker,
        config: QueueConfig,
        modality: str,
        inference_fn: InferenceFn,
        *,
        image_fetcher: Optional[Callable[[str], bytes]] = None,
        model_version: Optional[str] = None,
    ):
        self.broker = broker
        self.config = config
        self.modality = modality
        self.inference_fn = inference_fn
        self.image_fetcher = image_fetcher
        self.model_version = model_version
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    def _resolve_image(self, job: JobMessage) -> bytes:
        if job.image_base64:
            try:
                return base64.b64decode(job.image_base64)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"invalid base64 image: {exc}")
        if job.image_url:
            if self.image_fetcher is None:
                raise ValueError("job provided image_url but no fetcher configured")
            return self.image_fetcher(job.image_url)
        raise ValueError("job has neither image_base64 nor image_url")

    def _handle_raw(self, raw: str) -> None:
        """Parse one job, run inference, publish the result. Never raises."""
        try:
            job = JobMessage.from_json(raw)
        except Exception as exc:  # noqa: BLE001 - undecodable message, drop it
            print(f"[worker:{self.modality}] dropping malformed job: {exc}")
            return

        started = time.time()
        result_msg = ResultMessage(
            job_id=job.job_id,
            modality=job.modality or self.modality,
            status="failed",
            patient_id=job.patient_id,
            worker=self.config.consumer_name(self.modality),
            model_version=self.model_version,
        )

        try:
            image_bytes = self._resolve_image(job)
            result_msg.image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            payload = self.inference_fn(image_bytes, job.options or {})
            result_msg.status = "completed"
            result_msg.result = payload
        except Exception as exc:  # noqa: BLE001
            result_msg.status = "failed"
            result_msg.error = str(exc)
            print(f"[worker:{self.modality}] job {job.job_id} failed: {exc}")

        result_msg.duration_ms = round((time.time() - started) * 1000, 2)
        result_msg.finished_at = _now_iso()

        channel = job.reply_to or self.config.result_channel
        try:
            self.broker.publish(channel, result_msg.to_json())
        except Exception as exc:  # noqa: BLE001
            print(f"[worker:{self.modality}] failed to publish result: {exc}")

    # ------------------------------------------------------------------ #
    def run_forever(self) -> None:
        channel = self.config.jobs_channel(self.modality)
        consumer = self.config.consumer_name(self.modality)
        print(
            f"[worker:{self.modality}] consuming '{channel}' "
            f"(group={self.config.group}, backend={self.config.backend}) "
            f"-> results to '{self.config.result_channel}'"
        )
        self.broker.consume(
            channel,
            self._handle_raw,
            group=self.config.group,
            consumer=consumer,
            block_ms=self.config.block_ms,
            stop_event=self._stop,
        )

    def start_background(self) -> "JobWorker":
        if self._thread and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self.run_forever,
            name=f"cliniq-worker-{self.modality}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self.broker.close()
        except Exception:  # noqa: BLE001
            pass
