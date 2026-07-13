"""
Triage worklist service (FastAPI, port 8010).

Endpoints
  GET  /                      HTML dashboard (auto-refresh)
  GET  /worklist              severity-ranked cases (JSON)
  POST /worklist/{job_id}/ack mark a case reviewed
  GET  /audit/recent          recent audit rows (JSON)
  GET  /audit/stats           audit summary (counts, model versions)
  GET  /health                liveness

On startup it begins consuming `cliniq:results` (if QUEUE_BACKEND is set).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the consolidated cliniq packages importable regardless of CWD.
_CLINIQ_ROOT = Path(__file__).resolve().parents[1]
if str(_CLINIQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLINIQ_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from messaging.config import load_config
from messaging.factory import get_broker
from audit.store import AuditStore
from triage.store import TriageStore
from triage.consumer import ResultConsumer
from triage.alerts import AlertStore, CriticalNotifier

app = FastAPI(title="ClinIQ Triage Worklist", version="1.0.0")

# Unified request/response logging -> stdout -> SLURM log
try:
    from messaging.http_logging import install_request_logging
    install_request_logging(app, "triage")
except Exception as _log_exc:  # noqa: BLE001
    print(f"[triage] request logging not installed: {_log_exc}")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

audit_store = AuditStore()
triage_store = TriageStore()
alert_store = AlertStore()
notifier = CriticalNotifier(alert_store)   # webhook via ALERT_WEBHOOK_URL env
_state = {"consumer": None}

_LEVEL_COLOR = {
    "CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f",
    "LOW": "#3498db", "INFO": "#95a5a6", "REJECTED": "#6b7280",
}


@app.on_event("startup")
async def _start_consumer():
    cfg = load_config()
    if not cfg.enabled:
        print("[triage] QUEUE_BACKEND not set — dashboard only, no live consume.")
        return
    try:
        broker = get_broker(cfg)
        if not broker.ping():
            print("[triage] broker ping failed — consumer not started.")
            return
    except Exception as exc:  # noqa: BLE001
        print(f"[triage] queue unavailable: {exc}")
        return
    consumer = ResultConsumer(
        broker, cfg, audit_store, triage_store, on_critical=notifier.notify
    )
    consumer.start_background()
    _state["consumer"] = consumer
    if notifier.webhook_url:
        print(f"[triage] critical alerts -> webhook {notifier.webhook_url}")
    else:
        print("[triage] critical alerts -> store + console (set ALERT_WEBHOOK_URL for webhook)")


@app.on_event("shutdown")
async def _stop_consumer():
    if _state["consumer"]:
        _state["consumer"].stop()


@app.get("/health")
async def health():
    return {"status": "healthy", "consuming": _state["consumer"] is not None}


@app.get("/worklist")
async def worklist(limit: int = 100, include_acknowledged: bool = False):
    return {
        "counts": triage_store.counts(),
        "cases": triage_store.worklist(limit, include_acknowledged),
    }


@app.post("/worklist/{job_id}/ack")
async def ack(job_id: str):
    if not triage_store.acknowledge(job_id):
        raise HTTPException(status_code=404, detail="case not found")
    return {"acknowledged": job_id}


@app.get("/alerts")
async def alerts(limit: int = 50):
    return {
        "unacknowledged": alert_store.unacknowledged_count(),
        "alerts": alert_store.recent(limit),
    }


@app.post("/alerts/{alert_id}/ack")
async def ack_alert(alert_id: int):
    if not alert_store.acknowledge(alert_id):
        raise HTTPException(status_code=404, detail="alert not found")
    return {"acknowledged": alert_id}


@app.get("/audit/recent")
async def audit_recent(limit: int = 50):
    return {"entries": audit_store.recent(limit)}


@app.get("/audit/stats")
async def audit_stats():
    return audit_store.stats()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    counts = triage_store.counts()
    cases = triage_store.worklist(100)
    stats = audit_store.stats()
    recent_alerts = alert_store.recent(8)
    pending_alerts = alert_store.unacknowledged_count()

    # Prominent banner + feed for critical alerts.
    alert_banner = ""
    if pending_alerts:
        alert_banner = (
            f'<div class="alertbanner">🚨 {pending_alerts} unacknowledged '
            f'CRITICAL alert{"s" if pending_alerts != 1 else ""} — review immediately</div>'
        )
    alert_rows = "".join(
        f"""<tr>
          <td>🚨</td><td>{a.get('modality') or '—'}</td>
          <td>{a.get('top_finding') or '—'}</td>
          <td>{a.get('patient_id') or '—'}</td>
          <td><code>{a.get('delivery') or '—'}</code></td>
          <td class="sub">{(a.get('created_at') or '')[:19].replace('T',' ')}</td>
        </tr>""" for a in recent_alerts
    )

    rows = ""
    for c in cases:
        color = _LEVEL_COLOR.get(c["priority_level"], "#95a5a6")
        conf = f"{c['max_confidence']*100:.0f}%" if c.get("max_confidence") else "—"
        rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{c['priority_level']}</span></td>
          <td>{c['priority_score']}</td>
          <td>{c['modality']}</td>
          <td>{c.get('top_finding') or '—'}</td>
          <td>{conf}</td>
          <td>{c.get('patient_id') or '—'}</td>
          <td class="status-{c['status']}">{c['status']}</td>
          <td><button onclick="ack('{c['job_id']}')">✓ Reviewed</button></td>
        </tr>"""

    chips = " ".join(
        f'<span class="chip" style="background:{_LEVEL_COLOR.get(k, "#777")}">{k}: {v}</span>'
        for k, v in counts.items() if k != "pending"
    )
    models = "".join(
        f"<li><b>{m}</b>: <code>{v}</code></li>" for m, v in stats["model_versions"].items()
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>ClinIQ Triage Worklist</title>
<meta http-equiv="refresh" content="5">
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0f1419; color:#e6e6e6; }}
  header {{ padding: 16px 24px; background:#1a2332; border-bottom:1px solid #2c3e50; }}
  h1 {{ margin:0; font-size:20px; }} .sub {{ color:#8aa; font-size:13px; }}
  .bar {{ padding:12px 24px; }} .chip,.badge {{ color:#fff; padding:3px 9px; border-radius:12px; font-size:12px; font-weight:600; }}
  .chip {{ margin-right:6px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ text-align:left; padding:10px 24px; border-bottom:1px solid #222c3a; font-size:14px; }}
  th {{ color:#8aa; font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:.5px; }}
  tr:hover {{ background:#161e2b; }}
  .status-failed {{ color:#e74c3c; }} .status-completed {{ color:#2ecc71; }}
  button {{ background:#2c3e50; color:#fff; border:0; padding:5px 10px; border-radius:6px; cursor:pointer; }}
  button:hover {{ background:#3a516b; }}
  .panel {{ padding:12px 24px; color:#8aa; font-size:13px; }} code {{ color:#7fd; }}
  ul {{ margin:6px 0; }}
  .alertbanner {{ background:#e74c3c; color:#fff; font-weight:700; padding:12px 24px;
    font-size:15px; animation: pulse 1.6s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.72; }} }}
  .section {{ padding:6px 24px 2px; color:#8aa; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
</style></head><body>
<header>
  <h1>🏥 ClinIQ Triage Worklist</h1>
  <div class="sub">Severity-ranked review queue · auto-refresh 5s · {counts.get('pending',0)} pending</div>
</header>
{alert_banner}
<div class="bar">{chips or '<span class="sub">No cases yet — waiting for results…</span>'}</div>
<table>
  <thead><tr><th>Priority</th><th>Score</th><th>Modality</th><th>Top finding</th>
  <th>Conf</th><th>Patient</th><th>Status</th><th></th></tr></thead>
  <tbody>{rows}</tbody>
</table>
{f'''<div class="section">🚨 Recent critical alerts</div>
<table><thead><tr><th></th><th>Modality</th><th>Finding</th><th>Patient</th>
<th>Delivery</th><th>Time</th></tr></thead><tbody>{alert_rows}</tbody></table>''' if alert_rows else ''}
<div class="panel">
  <b>Audit:</b> {stats['total_predictions']} predictions logged ·
  avg {stats.get('avg_duration_ms') or 0} ms · by status {stats['by_status']}
  <div><b>Model versions in use:</b><ul>{models or '<li>none yet</li>'}</ul></div>
</div>
<script>
  async function ack(id) {{
    await fetch('/worklist/'+id+'/ack', {{method:'POST'}});
    location.reload();
  }}
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
