"""
Redis Streams broker.

Uses Redis Streams + consumer groups (XADD / XREADGROUP / XACK) so that:
  * multiple workers can share one modality queue (load balancing),
  * messages are acknowledged only after successful handling (at-least-once),
  * the .NET backend can read the results stream with XREAD/XREADGROUP.

Works against managed Redis (Upstash, ElastiCache, etc.) over TLS.
"""

from __future__ import annotations

import threading
import time

from .base import Broker, MessageHandler

# Field name used inside each stream entry to carry the JSON payload.
_FIELD = "data"


class RedisStreamBroker(Broker):
    def __init__(self, connection_string: str, *, maxlen: int = 10000):
        from .connection import build_redis_client

        self._client = build_redis_client(connection_string, decode_responses=True)
        self._maxlen = maxlen
        self._groups_ready = set()

    # ------------------------------------------------------------------ #
    def publish(self, channel: str, message: str) -> None:
        # approximate trimming (~) is far cheaper than exact and is fine here.
        self._client.xadd(
            channel, {_FIELD: message}, maxlen=self._maxlen, approximate=True
        )

    # ------------------------------------------------------------------ #
    def _ensure_group(self, channel: str, group: str) -> None:
        key = (channel, group)
        if key in self._groups_ready:
            return
        try:
            # id="0" so a freshly created group can also drain pre-existing
            # entries; MKSTREAM creates the stream if the producer hasn't yet.
            self._client.xgroup_create(channel, group, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups_ready.add(key)

    def consume(
        self,
        channel: str,
        handler: MessageHandler,
        *,
        group: str,
        consumer: str,
        block_ms: int = 5000,
        stop_event: "threading.Event | None" = None,
    ) -> None:
        self._ensure_group(channel, group)
        stop_event = stop_event or threading.Event()

        # First drain anything already pending for THIS consumer (e.g. messages
        # delivered but not acked before a previous crash), then read new ones.
        backlog_id = "0"
        while not stop_event.is_set():
            try:
                streams = {channel: backlog_id}
                response = self._client.xreadgroup(
                    group, consumer, streams, count=10, block=block_ms
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[redis-broker] read error on {channel}: {exc}")
                time.sleep(1.0)
                continue

            if not response:
                # No backlog left -> switch to live tail and keep blocking.
                if backlog_id != ">":
                    backlog_id = ">"
                continue

            for _stream_name, entries in response:
                if not entries and backlog_id != ">":
                    # Exhausted this consumer's backlog; move to new messages.
                    backlog_id = ">"
                    continue
                for entry_id, fields in entries:
                    payload = fields.get(_FIELD)
                    try:
                        if payload is not None:
                            handler(payload)
                        # Ack only after the handler succeeds.
                        self._client.xack(channel, group, entry_id)
                    except Exception as exc:  # noqa: BLE001
                        # Leave the message pending for later reclaim/inspection.
                        print(f"[redis-broker] handler failed for {entry_id}: {exc}")

    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
