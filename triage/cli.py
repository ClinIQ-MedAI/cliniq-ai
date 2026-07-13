"""
Terminal view of the triage worklist and audit log (no web UI needed).

    python -m triage.cli worklist      # severity-ranked pending cases
    python -m triage.cli audit         # recent audit rows
    python -m triage.cli stats         # audit summary + model versions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CLINIQ_ROOT = Path(__file__).resolve().parents[1]
if str(_CLINIQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLINIQ_ROOT))

from audit.store import AuditStore
from triage.store import TriageStore


def _worklist(_args) -> int:
    store = TriageStore()
    counts = store.counts()
    print(f"Pending: {counts.get('pending', 0)}  "
          + "  ".join(f"{k}={v}" for k, v in counts.items() if k != "pending"))
    print("-" * 78)
    for c in store.worklist(50):
        conf = f"{c['max_confidence']*100:.0f}%" if c.get("max_confidence") else "—"
        print(f"[{c['priority_level']:8}] score={c['priority_score']:3}  "
              f"{c['modality']:13} {(c.get('top_finding') or '—')[:32]:32} "
              f"conf={conf:4} job={c['job_id'][:8]}")
    return 0


def _audit(_args) -> int:
    for e in AuditStore().recent(30):
        print(f"{e['created_at']}  {e['status']:9} {e['modality']:13} "
              f"{(e.get('top_finding') or '—')[:30]:30} "
              f"v={e.get('model_version') or '—'}")
    return 0


def _stats(_args) -> int:
    import json
    print(json.dumps(AuditStore().stats(), indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="triage.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worklist")
    sub.add_parser("audit")
    sub.add_parser("stats")
    args = parser.parse_args(argv)
    return {"worklist": _worklist, "audit": _audit, "stats": _stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
