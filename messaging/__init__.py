"""
ClinIQ messaging layer.

A small, pluggable job-queue abstraction that lets the AI inference services
act as asynchronous workers behind a Redis or RabbitMQ broker.

Flow:

    .NET Backend ---(JobMessage)---> [ jobs queue ] ---> ClinIQ AI worker
    .NET Backend <--(ResultMessage)-- [ results queue ] <-- ClinIQ AI worker

Public API:
    get_broker()                -> Broker (selected via QUEUE_BACKEND env)
    JobMessage, ResultMessage   -> message schemas
    JobWorker                   -> generic consume/process/publish loop
    attach_worker(app, ...)     -> wire a worker into a FastAPI service
"""

from .config import QueueConfig, load_config
from .schemas import JobMessage, ResultMessage
from .factory import get_broker
from .worker import JobWorker

__all__ = [
    "QueueConfig",
    "load_config",
    "JobMessage",
    "ResultMessage",
    "get_broker",
    "JobWorker",
]
