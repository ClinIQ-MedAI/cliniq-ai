#!/bin/bash
# ClinIQ — Chest X-Ray GPU server (SLURM batch job)
#SBATCH --job-name=cliniq-chest-xray
#SBATCH --account=thetan-ai
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/chest_xray_gpu-%j.log

set -euo pipefail

ROOT="/data/user/moabouag/far/cliniq"
SERVICE_DIR="$ROOT/chest_xray"
PORT=8003
ENDPOINT_FILE="$ROOT/.local/chest_xray_endpoint.txt"

echo "==================================================================="
echo "ClinIQ Chest X-Ray GPU server"
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
echo "[chest_xray] endpoint published: $NODE_HOST:$PORT"

# Start API server
cd "$SERVICE_DIR"
echo "[chest_xray] starting API server..."
CUDA_VISIBLE_DEVICES=0 python -m api.server &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "[chest_xray] ✓ Server ready at http://$NODE_HOST:$PORT"
    break
  fi
  sleep 2
  echo "[chest_xray] waiting... ($i/30)"
done

# Keep alive
wait $SERVER_PID
