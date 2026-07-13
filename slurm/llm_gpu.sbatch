#!/bin/bash
# ClinIQ — GPU LLM server (SLURM batch job)
#
# Runs the local Ollama LLM on a cluster GPU node (L40S/H100) instead of the
# CPU-only login node, so the chatbot's fallback model is fast and can be a
# much stronger model (gpt-oss:20b — same family as the primary gpt-oss-120b).
#
# The login-node LiteLLM proxy then points at this job's node over the internal
# cluster network. Model blobs live on shared NFS (cliniq/.local/ollama/data),
# so the model is already present — no download inside the GPU allocation.
#
# Submit:   sbatch slurm/llm_gpu.sbatch
# Watch:    tail -f slurm/llm_gpu-<jobid>.log
# Wire up:  ./slurm/point_to_gpu.sh          (run on the login node afterwards)
# Stop:     scancel <jobid>
#
#SBATCH --job-name=cliniq-llm
#SBATCH --account=thetan-ai
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/llm_gpu-%j.log

set -euo pipefail

MODEL="${CLINIQ_LLM_MODEL:-gpt-oss:20b}"
ROOT="/data/user/moabouag/far/cliniq"
OLLAMA_DIR="$ROOT/.local/ollama"
PORT=11434

echo "==================================================================="
echo "ClinIQ GPU LLM job"
echo "  node:  $(hostname)"
echo "  gpus:  ${CUDA_VISIBLE_DEVICES:-?}"
echo "  model: $MODEL"
echo "  time:  $(date)"
echo "==================================================================="

# Enable `conda activate` inside a non-interactive batch shell.
source /u/moabouag/miniforge3/etc/profile.d/conda.sh
conda activate r

export OLLAMA_MODELS="$OLLAMA_DIR/data"
export OLLAMA_HOST="0.0.0.0:$PORT"                    # reachable from the login node
export OLLAMA_KEEP_ALIVE="-1"                          # keep the model resident
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama:${LD_LIBRARY_PATH:-}"

# Publish this node's address so the login side can find us (shared NFS).
NODE_HOST="$(hostname)"
echo "$NODE_HOST:$PORT" > "$ROOT/.local/gpu_llm_endpoint.txt"
echo "[llm_gpu] endpoint published: $NODE_HOST:$PORT"

# Start the server (CUDA is auto-detected on a GPU node).
"$OLLAMA_DIR/bin/ollama" serve &
OLLAMA_PID=$!

# Wait for it to answer.
for i in $(seq 1 30); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/api/version" && break
  sleep 1
done

# Ensure the model is present (served from shared NFS; pulls only if missing).
if ! "$OLLAMA_DIR/bin/ollama" list | grep -q "${MODEL%%:*}"; then
  echo "[llm_gpu] model not found locally, pulling $MODEL ..."
  "$OLLAMA_DIR/bin/ollama" pull "$MODEL"
fi

# Warm the model + quick GPU benchmark so the log shows real throughput.
echo "[llm_gpu] warming + benchmarking $MODEL on GPU ..."
curl -s "http://127.0.0.1:$PORT/api/chat" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"In one sentence, why is regular medical screening important?\"}],
  \"stream\": false
}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
tok=d.get('eval_count',0); ns=d.get('eval_duration',1)
print(f\"[llm_gpu] reply: {d.get('message',{}).get('content','')[:120]}\")
print(f\"[llm_gpu] throughput: {tok/(ns/1e9):.1f} tokens/sec on GPU\")
" || echo "[llm_gpu] benchmark call failed (model may still be loading)"

echo "[llm_gpu] READY — serving $MODEL at $NODE_HOST:$PORT"
echo "[llm_gpu] on the login node run:  ./slurm/point_to_gpu.sh"

# Stay alive for the allocation so the login-node proxy can use us.
wait $OLLAMA_PID
