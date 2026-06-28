"""
FastAPI integration.

`attach_worker` lets an existing inference service become a queue worker in two
lines, reusing its current ``predict_for_llm`` route as the inference function —
no duplication of model loading or post-processing.

    from messaging.fastapi_integration import attach_worker
    attach_worker(app, modality="bone", route=predict_for_llm)

The worker only starts when QUEUE_BACKEND is set to redis/rabbitmq; otherwise the
service runs exactly as before. The worker consumes from ``cliniq:jobs:<modality>``
and publishes ResultMessages to ``cliniq:results``.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import threading
from typing import Callable, Dict, Optional

from .config import load_config
from .factory import get_broker
from .worker import JobWorker


class _UploadShim:
    """
    Duck-typed stand-in for fastapi.UploadFile.

    Supports every access pattern the ClinIQ routes use:
      * ``await file.read()``        (bone, chest, oral-classify, oral-xray)
      * ``file.file.read()``         (prescription parser)
      * ``file.content_type``        (oral-xray content-type guard)
      * ``file.filename``
    """

    def __init__(self, data: bytes, filename: str, content_type: str):
        self._data = data
        self.file = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._data

    async def seek(self, offset: int) -> None:
        self.file.seek(offset)

    async def close(self) -> None:
        pass


def _unwrap_default(param: inspect.Parameter):
    """Resolve a real default from a plain value or a FastAPI Query/FieldInfo."""
    default = param.default
    if default is inspect.Parameter.empty:
        return inspect.Parameter.empty
    # fastapi.params.Query and pydantic FieldInfo both carry a `.default`.
    inner = getattr(default, "default", None)
    if inner is not None and inner is not Ellipsis and type(default).__name__ in (
        "Query", "FieldInfo", "Form", "Body",
    ):
        return inner
    return default


def _to_dict(result) -> Dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):      # pydantic v2
        return result.model_dump()
    if hasattr(result, "dict"):            # pydantic v1
        return result.dict()
    return {"result": result}


def _make_inference_fn(route: Callable, default_content_type: str):
    """Wrap an async ``predict_for_llm``-style route as a sync inference fn."""
    sig = inspect.signature(route)
    extra_params = [
        p for name, p in sig.parameters.items()
        if name not in ("file", "self")
    ]
    # One event loop per worker thread, reused across jobs.
    _local = threading.local()

    def _loop() -> asyncio.AbstractEventLoop:
        loop = getattr(_local, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            _local.loop = loop
        return loop

    def inference_fn(image_bytes: bytes, options: Dict) -> Dict:
        upload = _UploadShim(
            image_bytes,
            filename=options.get("filename", "upload.jpg"),
            content_type=options.get("content_type", default_content_type),
        )
        kwargs = {}
        for param in extra_params:
            if param.name in options:
                kwargs[param.name] = options[param.name]
            else:
                default = _unwrap_default(param)
                if default is not inspect.Parameter.empty:
                    kwargs[param.name] = default

        result = route(upload, **kwargs)
        if inspect.isawaitable(result):
            result = _loop().run_until_complete(result)
        return _to_dict(result)

    return inference_fn


def _default_image_fetcher(url: str) -> bytes:
    import requests

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def attach_worker(
    app,
    modality: str,
    route: Callable,
    *,
    default_content_type: str = "image/jpeg",
    image_fetcher: Optional[Callable[[str], bytes]] = None,
):
    """
    Register a queue worker on a FastAPI app (no-op unless QUEUE_BACKEND is set).

    Returns a small state dict whose ``worker`` key is populated after startup.
    """
    config = load_config()
    if not config.enabled:
        # Messaging disabled — keep the service purely synchronous.
        return None

    inference_fn = _make_inference_fn(route, default_content_type)
    state: Dict[str, Optional[JobWorker]] = {"worker": None}

    # Fingerprint this service's checkpoint once, for prediction traceability.
    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[1]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from audit.versioning import model_version_for_modality
        resolved_model_version = model_version_for_modality(modality)
    except Exception:  # noqa: BLE001
        resolved_model_version = None

    @app.on_event("startup")
    async def _start_worker():  # pragma: no cover - exercised at runtime
        try:
            broker = get_broker(config)
            if not broker.ping():
                print(f"[worker:{modality}] broker ping failed — worker not started")
                return
        except Exception as exc:  # noqa: BLE001
            print(f"[worker:{modality}] queue unavailable, staying HTTP-only: {exc}")
            return
        worker = JobWorker(
            broker,
            config,
            modality,
            inference_fn,
            image_fetcher=image_fetcher or _default_image_fetcher,
            model_version=resolved_model_version,
        )
        worker.start_background()
        state["worker"] = worker

    @app.on_event("shutdown")
    async def _stop_worker():  # pragma: no cover - exercised at runtime
        worker = state.get("worker")
        if worker is not None:
            worker.stop()

    return state
