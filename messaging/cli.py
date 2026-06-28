"""
Command-line helper for the ClinIQ job queue.

Acts as the *backend side* of the queue so you can exercise the AI workers
without the real .NET service:

    # 1. health check
    python -m messaging.cli ping

    # 2. submit a job (the backend's producer role)
    python -m messaging.cli enqueue --modality bone --image /path/to/xray.jpg

    # 3. stream results as workers finish (the backend's consumer role)
    python -m messaging.cli listen

Requires the same QUEUE_BACKEND / REDIS_CONNECTION / RABBITMQ_URL env vars as
the services. Run `enqueue`/`listen` in separate terminals.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone

from .config import load_config
from .factory import get_broker
from .schemas import JobMessage, ResultMessage, KNOWN_MODALITIES


def _cmd_ping(args) -> int:
    cfg = load_config()
    if not cfg.enabled:
        print("QUEUE_BACKEND is not set (none). Enable redis or rabbitmq first.")
        return 1
    broker = get_broker(cfg)
    ok = broker.ping()
    print(f"backend={cfg.backend} ping={'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def _cmd_enqueue(args) -> int:
    cfg = load_config()
    broker = get_broker(cfg)

    if args.modality not in KNOWN_MODALITIES:
        print(f"warning: '{args.modality}' is not a known modality "
              f"({', '.join(sorted(KNOWN_MODALITIES))})")

    with open(args.image, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode()

    options = json.loads(args.options) if args.options else {}
    job = JobMessage(
        modality=args.modality,
        image_base64=image_b64,
        patient_id=args.patient_id,
        options=options,
        enqueued_at=datetime.now(timezone.utc).isoformat(),
    )
    broker.publish(cfg.jobs_channel(args.modality), job.to_json())
    print(f"enqueued job_id={job.job_id} -> {cfg.jobs_channel(args.modality)}")
    return 0


def _cmd_listen(args) -> int:
    cfg = load_config()
    broker = get_broker(cfg)
    channel = args.channel or cfg.result_channel
    print(f"listening on '{channel}' (backend={cfg.backend}). Ctrl-C to stop.")

    def handle(raw: str) -> None:
        msg = ResultMessage.from_json(raw)
        head = f"[{msg.status.upper()}] {msg.modality} job={msg.job_id[:8]} " \
               f"({msg.duration_ms}ms, worker={msg.worker})"
        print(head)
        if msg.status == "failed":
            print(f"    error: {msg.error}")
        elif args.verbose and msg.result:
            print("    " + json.dumps(msg.result, ensure_ascii=False)[:500])

    try:
        broker.consume(
            channel, handle,
            group=args.group or f"{cfg.prefix}-cli",
            consumer="cli-listener",
            block_ms=cfg.block_ms,
        )
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="messaging.cli", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="check broker connectivity")

    p_enq = sub.add_parser("enqueue", help="submit a job (backend producer)")
    p_enq.add_argument("--modality", required=True)
    p_enq.add_argument("--image", required=True, help="path to an image file")
    p_enq.add_argument("--patient-id", default="cli_demo")
    p_enq.add_argument("--options", default="", help='JSON, e.g. {"include_gradcam": false}')

    p_lis = sub.add_parser("listen", help="stream results (backend consumer)")
    p_lis.add_argument("--channel", default="", help="override results channel")
    p_lis.add_argument("--group", default="", help="consumer group (Redis)")
    p_lis.add_argument("--verbose", action="store_true", help="print result payloads")

    args = parser.parse_args(argv)
    return {
        "ping": _cmd_ping,
        "enqueue": _cmd_enqueue,
        "listen": _cmd_listen,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
