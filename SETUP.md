# ClinIQ — Setup Guide (fresh clone → running platform)

Follow these steps in order. By the end you'll have all AI services, the
chatbot, the async job queue, and the triage worklist running.

---

## 0. Prerequisites

- Linux (CUDA-capable GPU strongly recommended; CPU works but is slow)
- Python 3.10+ and `git`
- ~10 GB free disk for the bundled models (+3 GB if you download the chest model)

---

## 1. Clone

```bash
git clone https://github.com/ClinIQ-MedAI/cliniq-ai.git
cd cliniq-ai
```

---

## 2. Environment + dependencies

```bash
# create an isolated env (conda or venv)
python -m venv .venv && source .venv/bin/activate
#   or:  conda create -n cliniq python=3.10 && conda activate cliniq

# core platform deps (all services except the prescription VLM)
pip install -r requirements.txt
```

> If `torch` doesn't match your CUDA driver, install the right build from
> https://pytorch.org then re-run the line above.

The **prescription-parser** is heavy (Qwen2-VL-72B) and optional. Only if you
need it, on a 48 GB-VRAM GPU:
```bash
pip install -r prescription-parser/requirements.txt
```

---

## 3. Models

Most checkpoints ship in `models/`. Verify what's present:

```bash
ls -lh models/bone-detect/*/weights/best.pt \
       models/oral-xray/*/weights/best.pt \
       models/oral-classify/*/best_model.pth
```

| Service | Status | Action |
|---|---|---|
| bone-detect | bundled | none |
| oral-xray (detector + refiner) | bundled | none |
| oral-classify | bundled | none |
| **chest_xray (~3 GB)** | **download** | see below |
| prescription-parser | auto from HuggingFace on first request | none |

Download the chest model from the Drive folder in [README.md](README.md#model-artifacts-google-drive)
and place it at `chest_xray/outputs/checkpoints/best.pt`.

---

## 4. Configure the chatbot LLM (optional but recommended)

```bash
cp chatbot-app/.env.example chatbot-app/.env 2>/dev/null || true
# edit chatbot-app/.env and set:
#   API_KEY=<your LLM key>
#   API_BASE_URL=https://llm.jetstream-cloud.org/api/
#   MODEL=gpt-oss-120b
```

The platform still runs without an LLM key — image analysis works; only the
conversational replies degrade.

---

## 5. Run the services (synchronous mode)

Each in its own terminal:

```bash
cd bone-detect          && python api/server.py     # :8001
cd oral-xray            && python api/server.py     # :8002
cd chest_xray           && python -m api.server     # :8003
cd oral-classify        && python -m api.server     # :8004
cd prescription-parser  && python api/server.py     # :8005  (optional)
cd chatbot-app          && python app.py            # :5000
```

Open **http://127.0.0.1:5000** — upload an X-ray and you should get an analysis.

Quick health check:
```bash
for p in 8001 8002 8003 8004; do curl -s localhost:$p/health; echo; done
```

---

## 6. (Optional) Enable the async job queue + triage worklist

This connects the AI services to an external backend (e.g. .NET) via Redis,
and adds a severity-ranked worklist + audit trail.

```bash
# point every service at the same broker (set in each service's shell or .env)
export QUEUE_BACKEND=redis
export REDIS_CONNECTION='your-host:6379,password=YOUR_TOKEN,ssl=true'

# restart the services from step 5 — each now ALSO consumes its job queue,
# then start the triage dashboard:
cd triage && python app.py        # http://localhost:8010
```

Test the queue without a backend:
```bash
python -m messaging.cli ping
python -m messaging.cli enqueue --modality bone --image sample.jpg
python -m messaging.cli listen --verbose
```

See [docs/QUEUE_INTEGRATION.md](docs/QUEUE_INTEGRATION.md) for the full backend
contract (message schemas + C# examples) and [triage/README.md](triage/README.md)
for the worklist.

---

## 7. Verify

```bash
# queue health
python -m messaging.cli ping        # -> backend=redis ping=OK

# after some jobs run:
python -m triage.cli worklist       # severity-ranked cases
python -m triage.cli stats          # audit totals + model versions
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to <service> API` in chatbot | that service isn't running — start it (step 5) |
| `ModuleNotFoundError: ultralytics/timm` | re-run `pip install -r requirements.txt` |
| chest service `model not found` | download the 3 GB checkpoint (step 3) |
| queue worker not starting | `QUEUE_BACKEND`/`REDIS_CONNECTION` unset or wrong — check `messaging.cli ping` |
| `invalid username-password pair` (Redis) | wrong/expired token — copy a fresh one from the Upstash console |
