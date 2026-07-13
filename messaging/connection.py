"""
Connection-string helpers.

The .NET backend stores Redis credentials in StackExchange.Redis format, e.g.

    causal-leopard-72320.upstash.io:6379,password=xxxx,ssl=true

This module turns that (or a standard redis:// / rediss:// URL) into the kwargs
that redis-py expects, so the Python AI workers and the .NET backend can share a
single connection string verbatim.
"""

from __future__ import annotations

from typing import Dict, Optional
from urllib.parse import urlparse


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_redis_connection(conn: str) -> Dict[str, object]:
    """
    Parse a Redis connection string into redis-py keyword arguments.

    Accepts two forms:
      1. StackExchange.Redis / .NET:  "host:port,password=...,ssl=true"
      2. URL:                         "redis://[:pass@]host:port/db"
                                      "rediss://..." (TLS)

    Returns a dict suitable for ``redis.Redis(**kwargs)``.
    """
    conn = (conn or "").strip()
    if not conn:
        raise ValueError("Empty Redis connection string")

    # URL form -------------------------------------------------------------
    if conn.startswith(("redis://", "rediss://")):
        parsed = urlparse(conn)
        kwargs: Dict[str, object] = {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 6379,
            "ssl": parsed.scheme == "rediss",
        }
        if parsed.password:
            kwargs["password"] = parsed.password
        if parsed.username:
            kwargs["username"] = parsed.username
        if parsed.path and parsed.path.strip("/").isdigit():
            kwargs["db"] = int(parsed.path.strip("/"))
        return kwargs

    # StackExchange.Redis form --------------------------------------------
    parts = [p.strip() for p in conn.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Could not parse Redis connection string: {conn!r}")

    endpoint = parts[0]
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        port = int(port_str)
    else:
        host, port = endpoint, 6379

    kwargs = {"host": host, "port": port, "ssl": False}
    for option in parts[1:]:
        if "=" not in option:
            continue
        key, _, raw = option.partition("=")
        key = key.strip().lower()
        raw = raw.strip()
        if key == "password":
            kwargs["password"] = raw
        elif key == "user" or key == "username":
            kwargs["username"] = raw
        elif key == "ssl":
            kwargs["ssl"] = _as_bool(raw)
        elif key in ("defaultdatabase", "database", "db"):
            try:
                kwargs["db"] = int(raw)
            except ValueError:
                pass
    return kwargs


def build_redis_client(conn: str, *, decode_responses: bool = True):
    """Create a redis-py client from a connection string (lazy import)."""
    import redis  # local import so RabbitMQ-only deployments need not install redis

    kwargs = parse_redis_connection(conn)
    # Upstash and most managed Redis terminate idle TLS sockets; keepalive +
    # periodic health checks keep long-lived worker connections honest.
    kwargs.update(
        decode_responses=decode_responses,
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    return redis.Redis(**kwargs)
