#!/usr/bin/env python3
"""ClinIQ — shared-GPU stack summary.

Probes every service running inside the shared SLURM allocation and prints a
per-service Node / GPU / CUDA table, e.g.

    LLM              -> Node: lair-g2 | GPU: L40S | CUDA: Yes
    Bone Detect      -> Node: lair-g2 | GPU: L40S | CUDA: Yes
    ...
    ✓ All services are running on the same GPU allocation.

Run this INSIDE the allocation (so 127.0.0.1 reaches the services and nvidia-smi
sees the allocated GPU). The sbatch job writes the output to
.local/cliniq_shared_summary.txt so the login-node launcher can echo it too.
"""

import socket
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")  # hide urllib3/charset version noise

try:
    import requests
except Exception:  # pragma: no cover - requests is a hard dep of the stack
    requests = None


def gpu_name() -> str:
    """Short GPU model name from nvidia-smi (e.g. 'L40S'), or 'N/A' on CPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = out.stdout.strip().splitlines()[0].strip()
        # "NVIDIA L40S" -> "L40S"
        return name.replace("NVIDIA", "").strip() or "Unknown"
    except Exception:
        return "N/A"


def probe_device(port: int, paths=("/health", "/")) -> str:
    """Return the 'device' string a service reports, or '' if unreachable."""
    if requests is None:
        return ""
    for path in paths:
        try:
            r = requests.get(f"http://127.0.0.1:{port}{path}", timeout=3)
            if r.status_code == 200:
                dev = r.json().get("device")
                if dev:
                    return str(dev)
        except Exception:
            continue
    return ""


def probe_llm_cuda() -> str:
    """Return the LLM device: 'cuda' if Ollama has a model resident in VRAM."""
    if requests is None:
        return ""
    try:
        # Is Ollama even up?
        v = requests.get("http://127.0.0.1:11434/api/version", timeout=3)
        if v.status_code != 200:
            return ""
        # /api/ps reports loaded models; size_vram > 0 means it's on the GPU.
        ps = requests.get("http://127.0.0.1:11434/api/ps", timeout=3)
        if ps.status_code == 200:
            for m in ps.json().get("models", []):
                if m.get("size_vram", 0) and m["size_vram"] > 0:
                    return "cuda"
            # Up but nothing resident yet — assume GPU node if a GPU is present.
            return "cuda" if gpu_name() not in ("N/A", "Unknown") else "cpu"
    except Exception:
        return ""
    return ""


def cuda_flag(device: str) -> str:
    if not device:
        return "Unreachable"
    return "Yes" if "cuda" in device.lower() else "No"


def queue_status() -> str:
    """One line describing the async job queue (broker + triage service)."""
    import os
    from pathlib import Path

    backend = os.getenv("QUEUE_BACKEND", "none").strip().lower()
    if not backend or backend == "none":
        return "Queue: disabled (QUEUE_BACKEND unset) — services are HTTP-only"

    # Broker reachability, via the project's own messaging layer.
    ping = "?"
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from messaging.config import load_config
        from messaging.factory import get_broker

        cfg = load_config()
        ping = "OK" if get_broker(cfg).ping() else "FAILED"
    except Exception as exc:  # broker unreachable / bad creds / missing dep
        ping = f"ERROR ({type(exc).__name__})"

    # Triage worklist service.
    triage = "down"
    if requests is not None:
        try:
            r = requests.get("http://127.0.0.1:8010/health", timeout=3)
            if r.status_code == 200:
                triage = "up on :8010"
        except Exception:
            pass

    return f"Queue: {backend} (ping {ping})  |  Triage: {triage}"


def main() -> int:
    node = socket.gethostname()
    gpu = gpu_name()

    # (label, device string) — order matters for display.
    rows = [
        ("LLM", probe_llm_cuda()),
        ("Bone Detect", probe_device(8001)),
        ("Oral X-ray", probe_device(8002)),
        ("Chest X-ray", probe_device(8003)),
        ("Oral Classify", probe_device(8004)),
    ]

    print()
    print("=" * 60)
    print(f"  ClinIQ Shared-GPU Stack — node {node} ({gpu})")
    print("=" * 60)

    all_cuda = True
    any_unreachable = False
    for label, device in rows:
        flag = cuda_flag(device)
        if flag != "Yes":
            all_cuda = False
        if flag == "Unreachable":
            any_unreachable = True
        # Per-service node is this allocation's node (they all run here).
        node_disp = node if flag != "Unreachable" else "-"
        gpu_disp = gpu if flag == "Yes" else ("-" if flag == "Unreachable" else gpu)
        print(f"  {label:<16} -> Node: {node_disp} | GPU: {gpu_disp} | CUDA: {flag}")

    print("-" * 60)
    if all_cuda:
        print("  ✓ All services are running on the same GPU allocation.")
    elif any_unreachable:
        print("  ⚠ Some services are not reachable yet (still booting?).")
    else:
        print("  ⚠ Some services fell back to CPU (torch.cuda not available?).")

    # The queue is CPU-only infrastructure, so it stays out of the CUDA verdict.
    print(f"  {queue_status()}")
    print("=" * 60)
    print()
    return 0 if all_cuda else 1


if __name__ == "__main__":
    sys.exit(main())
