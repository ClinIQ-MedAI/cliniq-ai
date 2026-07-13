"""Unified HTTP request/response logging for every ClinIQ FastAPI service.

One install call per service adds a middleware that prints, per request:

    [<service>] <ISO-timestamp> <client-ip> <METHOD> <path> -> <status> (<ms>ms) <body-summary>

and, on an unhandled exception, the full traceback. Everything goes to stdout so
it lands in the SLURM log the sbatch tees, and can be tailed live.

Usage, right after the FastAPI app is created:

    from messaging.http_logging import install_request_logging
    install_request_logging(app, "bone-detect")

The middleware never prints file bytes: for multipart uploads it lists the field
and file names only; for JSON it prints the (truncated) body.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware

_MAX_BODY_SUMMARY = 400          # chars of body summary to print
_MAX_BODY_READ = 8 * 1024 * 1024  # only read <=8MB into memory to summarise
_NAME_RE = re.compile(rb'name="([^"]*)"')
_FILE_RE = re.compile(rb'filename="([^"]*)"')


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _truncate(s: str) -> str:
    return s if len(s) <= _MAX_BODY_SUMMARY else s[:_MAX_BODY_SUMMARY] + f"...(+{len(s)-_MAX_BODY_SUMMARY})"


def _summarize_body(content_type: str, body: bytes) -> str:
    if not body:
        return ""
    ct = (content_type or "").lower()
    if "application/json" in ct:
        try:
            return "body=" + _truncate(json.dumps(json.loads(body), ensure_ascii=False))
        except Exception:
            return "body=" + _truncate(body[:_MAX_BODY_SUMMARY].decode("utf-8", "replace"))
    if "multipart/form-data" in ct:
        # Field/file names only — never the file content.
        files = [m.decode("utf-8", "replace") for m in _FILE_RE.findall(body)]
        names = [m.decode("utf-8", "replace") for m in _NAME_RE.findall(body)]
        fields = [n for n in names if n not in files]
        return _truncate(f"multipart fields={fields} files={files}")
    if "application/x-www-form-urlencoded" in ct:
        return "body=" + _truncate(body[:_MAX_BODY_SUMMARY].decode("utf-8", "replace"))
    return f"body=<{len(body)} bytes {ct or 'unknown'}>"


class _RequestLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service: str):
        super().__init__(app)
        self.service = service

    async def dispatch(self, request, call_next):
        t0 = time.monotonic()
        client = request.client.host if request.client else "-"
        method = request.method
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"

        # Read the body to summarise it, then replay it so the route still gets
        # it. Skip reading oversized bodies (still replayed as-is).
        body = b""
        summary = ""
        try:
            clen = int(request.headers.get("content-length") or 0)
        except ValueError:
            clen = 0
        if method in ("POST", "PUT", "PATCH") and 0 < clen <= _MAX_BODY_READ:
            body = await request.body()
            summary = _summarize_body(request.headers.get("content-type", ""), body)

            async def _receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = _receive
        elif clen > _MAX_BODY_READ:
            summary = f"body=<{clen} bytes, not read>"

        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            dt = (time.monotonic() - t0) * 1000
            print(f"[{self.service}] {_now()} {client} {method} {path} -> 500 "
                  f"({dt:.0f}ms) EXCEPTION: {type(exc).__name__}: {exc}  {summary}",
                  flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise

        dt = (time.monotonic() - t0) * 1000
        print(f"[{self.service}] {_now()} {client} {method} {path} -> "
              f"{response.status_code} ({dt:.0f}ms) {summary}", flush=True)
        return response


def install_request_logging(app, service: str) -> None:
    """Attach the unified request/response logger to a FastAPI/Starlette app."""
    app.add_middleware(_RequestLogMiddleware, service=service)
