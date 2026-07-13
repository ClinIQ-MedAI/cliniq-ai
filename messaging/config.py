"""
Queue configuration, resolved from environment variables.

The whole messaging layer is opt-in: if ``QUEUE_BACKEND`` is unset or ``none``,
services behave exactly as before (synchronous HTTP only) and no worker starts.

Environment variables
----------------------
QUEUE_BACKEND        redis | rabbitmq | none        (default: none)
QUEUE_PREFIX         channel namespace               (default: cliniq)
QUEUE_GROUP          consumer group name             (default: <prefix>-workers)
QUEUE_RESULT_CHANNEL where results are published     (default: <prefix>:results)
QUEUE_BLOCK_MS       broker blocking-read timeout ms (default: 5000)
QUEUE_MAXLEN         Redis stream cap (approx)        (default: 10000)

# Redis
REDIS_CONNECTION     StackExchange.Redis / .NET style string
REDIS_URL            redis:// or rediss:// URL  (alternative to REDIS_CONNECTION)

# RabbitMQ
RABBITMQ_URL         amqp://user:pass@host:5672/vhost
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


@dataclass
class QueueConfig:
    backend: str = "none"            # redis | rabbitmq | none
    prefix: str = "cliniq"
    group: str = ""
    result_channel: str = ""
    block_ms: int = 5000
    maxlen: int = 10000

    redis_connection: Optional[str] = None
    rabbitmq_url: Optional[str] = None

    consumer_id: str = field(default_factory=lambda: socket.gethostname())

    @property
    def enabled(self) -> bool:
        return self.backend in ("redis", "rabbitmq")

    def jobs_channel(self, modality: str) -> str:
        """Per-modality request queue, e.g. ``cliniq:jobs:bone``."""
        return f"{self.prefix}:jobs:{modality}"

    def consumer_name(self, modality: str) -> str:
        return f"{self.consumer_id}:{modality}"

    def chat_requests_channel(self) -> str:
        """Where the backend posts chat turns, e.g. ``cliniq:chat:requests``."""
        return f"{self.prefix}:chat:requests"

    def chat_results_channel(self) -> str:
        """Where the chat bridge posts replies, e.g. ``cliniq:chat:results``."""
        return f"{self.prefix}:chat:results"


def load_config() -> QueueConfig:
    backend = (_env("QUEUE_BACKEND", "none") or "none").lower()
    prefix = _env("QUEUE_PREFIX", "cliniq")

    # Accept either the .NET-style connection string or a plain URL.
    redis_connection = _env("REDIS_CONNECTION") or _env("REDIS_URL")

    return QueueConfig(
        backend=backend,
        prefix=prefix,
        group=_env("QUEUE_GROUP", f"{prefix}-workers"),
        result_channel=_env("QUEUE_RESULT_CHANNEL", f"{prefix}:results"),
        block_ms=int(_env("QUEUE_BLOCK_MS", "5000")),
        maxlen=int(_env("QUEUE_MAXLEN", "10000")),
        redis_connection=redis_connection,
        rabbitmq_url=_env("RABBITMQ_URL"),
    )
