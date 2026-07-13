#!/bin/bash
# ClinIQ — Run x-ray models as standalone CPU services
#
# Use this if GPU SLURM jobs fail. Each model runs locally on CPU.
# Start each in a separate terminal window.
#
# Terminal 1: ./run_xray_standalone.sh bone-detect
# Terminal 2: ./run_xray_standalone.sh oral-xray
# Terminal 3: ./run_xray_standalone.sh chest-xray
# Terminal 4: ./run_xray_standalone.sh oral-classify

if [ $# -ne 1 ]; then
  echo "Usage: $0 {bone-detect|oral-xray|chest-xray|oral-classify}"
  echo ""
  echo "Run each in a separate terminal:"
  echo "  Terminal 1: $0 bone-detect"
  echo "  Terminal 2: $0 oral-xray"
  echo "  Terminal 3: $0 chest-xray"
  echo "  Terminal 4: $0 oral-classify"
  exit 1
fi

SERVICE="$1"
ROOT="$(cd "$(dirname "$0")" && pwd)"

case "$SERVICE" in
  bone-detect)
    echo "▶ Starting Bone Detect API (:8001) on CPU..."
    cd "$ROOT/bone-detect"
    CUDA_VISIBLE_DEVICES="" python api/server.py
    ;;
  oral-xray)
    echo "▶ Starting Oral X-Ray API (:8002) on CPU..."
    cd "$ROOT/oral-xray"
    CUDA_VISIBLE_DEVICES="" python api/server.py
    ;;
  chest-xray)
    echo "▶ Starting Chest X-Ray API (:8003) on CPU..."
    cd "$ROOT/chest_xray"
    CUDA_VISIBLE_DEVICES="" python -m api.server
    ;;
  oral-classify)
    echo "▶ Starting Oral Classification API (:8004) on CPU..."
    cd "$ROOT/oral-classify"
    CUDA_VISIBLE_DEVICES="" python -m api.server
    ;;
  *)
    echo "Unknown service: $SERVICE"
    exit 1
    ;;
esac
