# triage — Severity Worklist + Audit Trail (port 8010)

Consumes the `cliniq:results` stream produced by the AI workers and provides two
production features radiology AI products treat as table stakes:

1. **Severity-ranked worklist** — critical findings (fracture, pneumothorax,
   decay…) float to the top so they're reviewed first. Inspired by Aidoc/Qure.ai
   triage prioritization.
2. **Audit trail** — every prediction is logged with its **model version** and
   **input fingerprint** (sha256) for full traceability.

It's a pure *consumer* of the queue, so it needs **no changes to the inference
services** and doubles as a reference for the backend side.

## Run

```bash
pip install -r triage/requirements.txt
export QUEUE_BACKEND=redis
export REDIS_CONNECTION='host:6379,password=…,ssl=true'

python triage/app.py            # dashboard at http://localhost:8010
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | live HTML dashboard (auto-refresh) |
| GET | `/worklist` | severity-ranked cases (JSON) |
| POST | `/worklist/{job_id}/ack` | mark a case reviewed |
| GET | `/audit/recent` | recent audit rows |
| GET | `/audit/stats` | totals, latency, model versions |
| GET | `/health` | liveness |

## Terminal view (no web)

```bash
python -m triage.cli worklist
python -m triage.cli audit
python -m triage.cli stats
```

## How priority is scored

`severity.py` reads `urgency`, per-detection `severity`, and `ai_findings` from
each result, maps Arabic/English levels to a canonical scale, escalates on
critical keywords (fracture, pneumothorax…), and nudges by confidence →
a 0-100 `priority_score` + level (CRITICAL/HIGH/MEDIUM/LOW/INFO).

## Storage

- `data/audit.db` — `predictions` table (one row per prediction)
- `data/triage.db` — `worklist` table (one row per case, ack-able)

Both are plain SQLite (stdlib). Failed jobs are still logged and shown so nothing
is silently dropped.
