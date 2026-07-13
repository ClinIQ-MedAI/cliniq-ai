# QStash Gateway

The public HTTP face of the ClinIQ AI stack. Upstash QStash is push-based: the
.NET backend hands a job to QStash, QStash POSTs it here over HTTP, this gateway
runs the prediction synchronously and returns the result in the same response,
and QStash relays that back to the backend.

The internal model services speak multipart file uploads and return raw
predictions; QStash speaks JSON with base64 images and expects a wrapped
envelope. This gateway is the translator, so the model services stay unchanged.

## Routes

| Route | Purpose |
|---|---|
| `POST /predict_for_llm` | Scans + prescription. Routes by the `modality` field to the internal service (8001–8005), wraps the result. |
| `POST /chat` | Chatbot. Calls the chatbot's `/api/chat`, folds the NDJSON stream into one reply. |
| `GET /health` | Liveness. |

`modality` ∈ `bone`, `dental_xray`, `chest`, `dental_photo`, `prescription`.
One gateway serves all five; point every `AIServiceSettings` URL at it.

## Run

```bash
python qstash-gateway/server.py        # defaults to :8080
```

Requires the internal model services (:8001–:8005) and the chatbot (:5000) to be
reachable on localhost — i.e. run it inside the shared-GPU allocation.
