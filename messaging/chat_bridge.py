"""Chat bridge: put the ClinIQ chatbot behind the Redis queue.

The chatbot listens on 127.0.0.1:5000 on a private compute node, so the public
.NET backend cannot call it over HTTP, exactly as with the model services. This
bridge closes that gap the same way the model workers do:

    backend  --XADD-->  cliniq:chat:requests
    bridge   consumes, calls the local chatbot's /api/chat, collects the stream
    bridge   --XADD-->  cliniq:chat:results
    backend  consumes the reply

Run it beside the chatbot, inside the same GPU allocation:

    QUEUE_BACKEND=redis REDIS_CONNECTION=... python -m messaging.chat_bridge

It stays silent (and exits 0) when QUEUE_BACKEND is unset, so the launcher can
call it unconditionally.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from datetime import datetime, timezone

import requests

from .config import load_config
from .factory import get_broker
from .schemas import ChatRequest, ChatReply

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://127.0.0.1:5000")
CHAT_TIMEOUT = float(os.getenv("CHAT_BRIDGE_TIMEOUT", "120"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_chatbot(req: ChatRequest) -> str:
    """POST one turn to the chatbot and fold its NDJSON stream into one string."""
    resp = requests.post(
        f"{CHATBOT_URL}/api/chat",
        json={
            "message": req.message,
            "patient_id": req.patient_id,
            "language_preference": req.language_preference,
        },
        stream=True,
        timeout=CHAT_TIMEOUT,
    )
    resp.raise_for_status()

    text_parts, meta = [], {}
    import json

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if "chunk" in obj:
            text_parts.append(obj["chunk"])
        if obj.get("done"):
            meta = obj
    return "".join(text_parts), meta


def _handle(broker, cfg, worker_id: str, raw: str) -> None:
    req = ChatRequest.from_json(raw)
    started = time.monotonic()
    channel = req.reply_to or cfg.chat_results_channel()

    try:
        text, meta = _call_chatbot(req)
        reply = ChatReply(
            chat_id=req.chat_id,
            status="completed",
            reply=text,
            query_type=meta.get("query_type"),
            show_upload=bool(meta.get("show_upload")),
            patient_id=req.patient_id,
            worker=worker_id,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
            finished_at=_now(),
        )
    except Exception as exc:  # noqa: BLE001 — any failure becomes a failed reply
        reply = ChatReply(
            chat_id=req.chat_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            patient_id=req.patient_id,
            worker=worker_id,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
            finished_at=_now(),
        )

    broker.publish(channel, reply.to_json())
    print(f"[chat-bridge] {req.chat_id[:8]} {reply.status} -> {channel}")


def main() -> int:
    cfg = load_config()
    if not cfg.enabled:
        print("[chat-bridge] QUEUE_BACKEND not set — chat bridge disabled.")
        return 0

    broker = get_broker(cfg)
    if not broker.ping():
        print("[chat-bridge] broker unreachable — check REDIS_CONNECTION.")
        return 1

    worker_id = f"{socket.gethostname()}:chat"
    channel = cfg.chat_requests_channel()
    print(f"[chat-bridge] consuming '{channel}' (group={cfg.group}) "
          f"-> replies to '{cfg.chat_results_channel()}'")

    broker.consume(
        channel,
        lambda raw: _handle(broker, cfg, worker_id, raw),
        group=cfg.group,
        consumer=worker_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
