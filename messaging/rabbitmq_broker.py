"""
RabbitMQ broker (AMQP 0-9-1 via pika).

Uses durable queues on the default exchange with manual acknowledgements and a
prefetch of 1, so work is fairly distributed across workers and a message is
removed only after it has been processed.

pika's BlockingConnection is not thread-safe; this broker is therefore meant to
be created and used entirely within a single worker thread (publish of the
result happens on the same thread that consumed the job).
"""

from __future__ import annotations

import threading
import time

from .base import Broker, MessageHandler


class RabbitMQBroker(Broker):
    def __init__(self, url: str):
        self._url = url
        self._connection = None
        self._channel = None
        self._declared = set()
        self._connect()

    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        import pika

        params = pika.URLParameters(self._url)
        # Keep the socket alive through idle periods and slow inference.
        if params.heartbeat is None:
            params.heartbeat = 60
        params.blocked_connection_timeout = 300
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.basic_qos(prefetch_count=1)
        self._declared.clear()

    def _ensure_queue(self, queue: str) -> None:
        if queue in self._declared:
            return
        self._channel.queue_declare(queue=queue, durable=True)
        self._declared.add(queue)

    def _reconnect(self) -> None:
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:  # noqa: BLE001
            pass
        self._connect()

    # ------------------------------------------------------------------ #
    def publish(self, channel: str, message: str) -> None:
        import pika

        self._ensure_queue(channel)
        self._channel.basic_publish(
            exchange="",
            routing_key=channel,
            body=message.encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,            # persist message to disk
                content_type="application/json",
            ),
        )

    # ------------------------------------------------------------------ #
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
        # `group` is unused for RabbitMQ: a shared durable queue already gives
        # competing-consumer load balancing across workers.
        stop_event = stop_event or threading.Event()
        self._ensure_queue(channel)
        inactivity = max(0.5, block_ms / 1000.0)

        while not stop_event.is_set():
            try:
                for method, _props, body in self._channel.consume(
                    channel, inactivity_timeout=inactivity
                ):
                    if stop_event.is_set():
                        break
                    if method is None:  # inactivity tick — loop to re-check stop
                        continue
                    try:
                        handler(body.decode("utf-8") if body else "")
                        self._channel.basic_ack(method.delivery_tag)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[rabbitmq-broker] handler failed: {exc}")
                        # Drop (don't requeue) to avoid poison-message loops;
                        # the worker already publishes a failure result.
                        self._channel.basic_nack(method.delivery_tag, requeue=False)
                break  # generator exited cleanly (stop requested)
            except Exception as exc:  # noqa: BLE001
                if stop_event.is_set():
                    break
                print(f"[rabbitmq-broker] connection error on {channel}: {exc}")
                time.sleep(2.0)
                try:
                    self._reconnect()
                except Exception as reconnect_exc:  # noqa: BLE001
                    print(f"[rabbitmq-broker] reconnect failed: {reconnect_exc}")
                    time.sleep(3.0)

        try:
            self._channel.cancel()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        try:
            return bool(self._connection and self._connection.is_open)
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:  # noqa: BLE001
            pass
