# ClinIQ ⇄ Backend Queue Integration

Asynchronous job queue connecting the **AI WorkFlow** (Python inference services)
to the **Backend** (.NET) via Redis or RabbitMQ.

```
.NET Backend ──(JobMessage)──▶  cliniq:jobs:<modality>  ──▶  ClinIQ AI worker
.NET Backend ◀─(ResultMessage)── cliniq:results        ◀──  ClinIQ AI worker
```

The backend **produces** jobs and **consumes** results. Each AI service runs an
in-process worker that consumes its modality queue and publishes results back.
The whole layer is opt-in: with `QUEUE_BACKEND` unset, services stay HTTP-only.

---

## 1. Channels

| Purpose | Name | Notes |
|---|---|---|
| Job request (per modality) | `cliniq:jobs:<modality>` | backend → worker |
| Results (shared) | `cliniq:results` | worker → backend |
| Consumer group (Redis) | `cliniq-workers` | workers; backend should use its own group |

`<modality>` ∈ `bone`, `dental_xray`, `dental_photo`, `chest`, `prescription`.

| Modality | Service | Port |
|---|---|---|
| `bone` | bone-detect | 8001 |
| `dental_xray` | oral-xray | 8002 |
| `chest` | chest_xray | 8003 |
| `dental_photo` | oral-classify | 8004 |
| `prescription` | prescription-parser | 8005 |

Names are configurable via `QUEUE_PREFIX`, `QUEUE_GROUP`, `QUEUE_RESULT_CHANNEL`.

---

## 2. Message contracts

### JobMessage (backend → worker)

```json
{
  "job_id": "3f2c…(uuid hex, optional — worker generates if absent)",
  "modality": "bone",
  "image_base64": "<base64 of the image bytes>",
  "image_url": null,
  "patient_id": "patient_demo",
  "options": { "include_gradcam": false },
  "reply_to": null,
  "enqueued_at": "2026-06-27T10:00:00Z"
}
```

- Provide **either** `image_base64` **or** `image_url` (worker fetches the URL).
- `options` is modality-specific; today only `chest` reads `include_gradcam` (bool).
- `reply_to` overrides the results channel for that one job (optional).

### ResultMessage (worker → backend)

```json
{
  "job_id": "3f2c…",
  "modality": "bone",
  "status": "completed",          // or "failed"
  "result": { …predict_for_llm payload… },
  "error": null,                   // string when status == "failed"
  "patient_id": "patient_demo",
  "worker": "host123:bone",
  "duration_ms": 842.3,
  "finished_at": "2026-06-27T10:00:01Z"
}
```

`result` is the exact JSON the service's `/predict_for_llm` HTTP endpoint returns
(detections, `annotated_image_base64`, `gradcam_image_base64`, summary, etc.), so
the backend can reuse existing parsing.

---

## 3. Transport details

### Redis (Streams)
- Jobs/results are **Redis Streams**. Publish with `XADD`, consume with a
  consumer group (`XREADGROUP` + `XACK`). At-least-once delivery.
- Works with Upstash/managed Redis over TLS — use the `…,ssl=true` connection
  string verbatim (same one the backend already stores).

### RabbitMQ
- Durable queues on the default exchange, persistent messages, manual ack,
  `prefetch=1`. Queue names are the channel names above.

---

## 4. Backend examples (C# / StackExchange.Redis)

Producer — submit a bone X-ray job:

```csharp
var redis = ConnectionMultiplexer.Connect(
    configuration.GetConnectionString("Redis"));   // host:port,password=…,ssl=true
var db = redis.GetDatabase();

var job = new {
    job_id = Guid.NewGuid().ToString("N"),
    modality = "bone",
    image_base64 = Convert.ToBase64String(imageBytes),
    patient_id = patientId,
    options = new { include_gradcam = false },
    enqueued_at = DateTime.UtcNow.ToString("o"),
};
await db.StreamAddAsync("cliniq:jobs:bone",
    new NameValueEntry[] { new("data", JsonSerializer.Serialize(job)) });
```

Consumer — read results (own consumer group):

```csharp
const string stream = "cliniq:results", group = "backend";
try { await db.StreamCreateConsumerGroupAsync(stream, group, "0-0", true); }
catch (RedisServerException e) when (e.Message.Contains("BUSYGROUP")) { }

while (true)
{
    var entries = await db.StreamReadGroupAsync(stream, group, "api", count: 10);
    foreach (var entry in entries)
    {
        var json = (string)entry["data"];
        var result = JsonSerializer.Deserialize<ResultMessage>(json);
        // …match result.job_id, push to client via SignalR/WebSocket, persist…
        await db.StreamAcknowledgeAsync(stream, group, entry.Id);
    }
}
```

> The Python workers already use the group `cliniq-workers` on the **jobs**
> streams. The backend should use a **different** group (e.g. `backend`) on the
> **results** stream so both sides ack independently.

---

## 5. Running it

```bash
# in each service environment
pip install -r messaging/requirements.txt
cp .env.queue.example .env          # set QUEUE_BACKEND + REDIS_CONNECTION

# start a service — it now also runs a worker
cd bone-detect && python api/server.py
#  -> [worker:bone] consuming 'cliniq:jobs:bone' -> results to 'cliniq:results'
```

Test without the .NET backend using the built-in CLI:

```bash
python -m messaging.cli ping
python -m messaging.cli listen --verbose          # terminal A (acts as backend)
python -m messaging.cli enqueue --modality bone --image sample_xray.jpg   # terminal B
```

---

## 6. Operational notes

- **Opt-in / safe rollout.** `QUEUE_BACKEND=none` (default) = no worker, no new
  deps required, existing HTTP behaviour untouched.
- **At-least-once.** A worker acks only after a result is produced; a crash
  mid-job leaves the message pending for redelivery. Make backend handling of a
  repeated `job_id` idempotent.
- **Failures are messages, not silence.** Inference errors come back as a
  `ResultMessage` with `status:"failed"` and an `error` string — never a dropped job.
- **GPU contention.** The worker shares the loaded model with the HTTP server in
  the same process; heavy simultaneous HTTP + queue load can serialize on the GPU.
  Run a dedicated worker instance (same `QUEUE_BACKEND`, scale horizontally) if you
  need isolation.
- **Payload size.** Images travel as base64 in the message. For very large scans
  prefer `image_url` (object storage / presigned URL) to keep the broker light.
```
