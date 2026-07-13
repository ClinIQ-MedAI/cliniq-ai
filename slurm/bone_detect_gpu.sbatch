#!/bin/bash
# ClinIQ — Bone Detection GPU server (SLURM batch job)
#SBATCH --job-name=cliniq-bone-detect
#SBATCH --account=thetan-ai
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/bone_detect_gpu-%j.log

set -euo pipefail

ROOT="/data/user/moabouag/far/cliniq"
SERVICE_DIR="$ROOT/bone-detect"
PORT=8001
ENDPOINT_FILE="$ROOT/.local/bone_detect_endpoint.txt"

echo "==================================================================="
echo "ClinIQ Bone Detect GPU server"
echo "  node:  $(hostname)"
echo "  gpus:  ${CUDA_VISIBLE_DEVICES:-?}"
echo "  port:  $PORT"
echo "  time:  $(date)"
echo "==================================================================="

source /u/moabouag/miniforge3/etc/profile.d/conda.sh
conda activate r

# Publish endpoint
NODE_HOST="$(hostname)"
echo "$NODE_HOST:$PORT" > "$ENDPOINT_FILE"
echo "[bone_detect] endpoint published: $NODE_HOST:$PORT"

# Start API server
cd "$SERVICE_DIR"
echo "[bone_detect] starting API server..."
CUDA_VISIBLE_DEVICES=0 python -u api/server.py &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "[bone_detect] ✓ Server ready at http://$NODE_HOST:$PORT"
    break
  fi
  sleep 2
  echo "[bone_detect] waiting... ($i/30)"
done

# Keep alive
wait $SERVER_PID
