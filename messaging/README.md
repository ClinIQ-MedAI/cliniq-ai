# messaging — ClinIQ async job queue

Pluggable Redis/RabbitMQ job queue that turns the AI inference services into
asynchronous workers for the .NET backend. Opt-in via `QUEUE_BACKEND`.

```
backend ──JobMessage──▶ cliniq:jobs:<modality> ──▶ worker (in each service)
backend ◀─ResultMessage─ cliniq:results        ◀── worker
```

## Layout

| File | Role |
|---|---|
| `config.py` | env-driven `QueueConfig` (`load_config()`) |
| `connection.py` | parse `.NET`/StackExchange-style + URL Redis strings |
| `schemas.py` | `JobMessage`, `ResultMessage` |
| `base.py` | `Broker` interface |
| `redis_broker.py` | Redis Streams + consumer groups |
| `rabbitmq_broker.py` | RabbitMQ durable queues |
| `factory.py` | `get_broker()` picks impl from config |
| `worker.py` | `JobWorker` consume→infer→publish loop |
| `fastapi_integration.py` | `attach_worker(app, modality, route)` |
| `cli.py` | `ping` / `enqueue` / `listen` test harness |

## Wire a service (already done for all 5)

```python
from messaging.fastapi_integration import attach_worker
attach_worker(app, modality="bone", route=predict_for_llm)
```

No-op unless `QUEUE_BACKEND=redis|rabbitmq`. Reuses the existing
`predict_for_llm` route as the inference function — no model code duplicated.

## Enable

```bash
pip install -r messaging/requirements.txt
export QUEUE_BACKEND=redis
export REDIS_CONNECTION='host:6379,password=…,ssl=true'
python api/server.py        # service now also consumes its modality queue
```

See [../docs/QUEUE_INTEGRATION.md](../docs/QUEUE_INTEGRATION.md) for the full
backend contract and C# examples.
