"""Abstract broker interface shared by the Redis and RabbitMQ implementations."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

# A consumer callback receives the raw JSON string of one message.
MessageHandler = Callable[[str], None]


class Broker(ABC):
    """
    Minimal broker contract.

    Semantics are at-least-once: a message is acknowledged only after the
    handler returns without raising. Implementations must not ack on handler
    failure, so an unhandled crash leaves the message for redelivery.
    """

    @abstractmethod
    def publish(self, channel: str, message: str) -> None:
        """Publish a single message (JSON string) to ``channel``."""

    @abstractmethod
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
        """
        Block and consume ``channel`` forever (until ``stop_event`` is set),
        invoking ``handler`` for each message and acknowledging on success.
        """

    @abstractmethod
    def ping(self) -> bool:
        """Return True if the broker connection is healthy."""

    def close(self) -> None:  # pragma: no cover - optional override
        """Release any underlying connections."""
