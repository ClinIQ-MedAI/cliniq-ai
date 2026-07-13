"""Broker factory — selects an implementation from configuration."""

from __future__ import annotations

from typing import Optional

from .base import Broker
from .config import QueueConfig, load_config


def get_broker(config: Optional[QueueConfig] = None) -> Broker:
    """
    Build the broker selected by ``QUEUE_BACKEND``.

    Raises ValueError if the backend is disabled/unknown or the matching
    connection string is missing, so callers fail loudly at startup.
    """
    config = config or load_config()

    if config.backend == "redis":
        if not config.redis_connection:
            raise ValueError(
                "QUEUE_BACKEND=redis but no REDIS_CONNECTION / REDIS_URL is set"
            )
        from .redis_broker import RedisStreamBroker

        return RedisStreamBroker(config.redis_connection, maxlen=config.maxlen)

    if config.backend == "rabbitmq":
        if not config.rabbitmq_url:
            raise ValueError("QUEUE_BACKEND=rabbitmq but no RABBITMQ_URL is set")
        from .rabbitmq_broker import RabbitMQBroker

        return RabbitMQBroker(config.rabbitmq_url)

    raise ValueError(
        f"Queue backend {config.backend!r} is not enabled. "
        "Set QUEUE_BACKEND=redis or QUEUE_BACKEND=rabbitmq."
    )
