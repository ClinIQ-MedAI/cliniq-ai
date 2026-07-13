#!/bin/bash
# ClinIQ Platform — Multi-Service Launcher
#
# Run this script to start all AI services + chatbot + triage worklist
# with async job queue enabled (Redis or RabbitMQ).
#
# Usage:
#   ./runs.sh                    (synchronous HTTP mode)
#   QUEUE_BACKEND=redis ./runs.sh (async mode with Redis)
#   QUEUE_BACKEND=rabbitmq ./runs.sh (async mode with RabbitMQ)

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ClinIQ Platform — Multi-Service Launcher           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Check if .env exists (optional)
if [ -f .env ]; then
  echo -e "${GREEN}✓ Loading .env${NC}"
  source .env
fi

# Determine mode (sync HTTP vs async queue)
MODE="${QUEUE_BACKEND:-none}"
if [ "$MODE" = "none" ] || [ -z "$MODE" ]; then
  echo -e "${YELLOW}Mode: Synchronous HTTP only (no async queue)${NC}"
  echo -e "${YELLOW}Tip: Set QUEUE_BACKEND=redis or QUEUE_BACKEND=rabbitmq for async mode${NC}"
else
  echo -e "${YELLOW}Mode: Async Queue${NC}"
  echo -e "${YELLOW}Backend: ${MODE}${NC}"
  if [ "$MODE" = "redis" ]; then
    echo -e "${YELLOW}Redis: ${REDIS_CONNECTION:-unset (will fail)}${NC}"
  elif [ "$MODE" = "rabbitmq" ]; then
    echo -e "${YELLOW}RabbitMQ: ${RABBITMQ_URL:-unset (will fail)}${NC}"
  fi
fi
echo

# Trap Ctrl+C to kill all background processes
trap 'echo -e "${RED}Stopping all services...${NC}"; kill $(jobs -p) 2>/dev/null; exit 0' INT

# Local LLM backend — opt-in.
#   GPU=1                 → submit a SLURM GPU job and serve a strong model
#                           (gpt-oss:20b) on a cluster GPU (~30x faster). One command.
#   LOCAL_LLM_FALLBACK=1  → free local CPU model (qwen2.5:3b) as a fallback only.
# Either way a LiteLLM proxy (:4000) sits in front of the chatbot: it tries the
# remote gpt-oss-120b first and falls back to the local/GPU model on failure.
# See docs/GPU_SLURM.md and docs/LOCAL_LLM_FALLBACK.md.
if [ "$LOCAL_LLM_FALLBACK" = "1" ] || [ "$GPU" = "1" ]; then
  echo -e "${BLUE}Starting local LLM backend (LiteLLM proxy)...${NC}"

  OLLAMA_BIN="$(pwd)/.local/ollama/bin/ollama"
  OLLAMA_LIB="$(pwd)/.local/ollama/lib/ollama"
  OLLAMA_DATA="$(pwd)/.local/ollama/data"

  if [ ! -x "$OLLAMA_BIN" ]; then
    echo -e "${RED}✗ Ollama not found at $OLLAMA_BIN${NC}"
    echo -e "${YELLOW}  See docs/LOCAL_LLM_FALLBACK.md to install it, or unset LOCAL_LLM_FALLBACK${NC}"
  else
    # Load the real upstream key from chatbot-app/.env (exported so both the
    # LiteLLM config and the chatbot subprocess below can see it/its override).
    set -a
    [ -f chatbot-app/.env ] && source chatbot-app/.env
    set +a
    export JETSTREAM_API_KEY="$API_KEY"

    GPU_ENDPOINT_FILE="$(pwd)/.local/gpu_llm_endpoint.txt"

    # GPU=1 → make sure a SLURM GPU job is up: submit it and wait if needed.
    if [ "$GPU" = "1" ]; then
      if [ -f "$GPU_ENDPOINT_FILE" ] && \
         curl -s -o /dev/null --max-time 3 "http://$(cat "$GPU_ENDPOINT_FILE")/api/version"; then
        echo -e "${GREEN}  ✓ GPU LLM already running at $(cat "$GPU_ENDPOINT_FILE")${NC}"
      elif ! command -v sbatch >/dev/null 2>&1; then
        echo -e "${RED}  ✗ GPU=1 but SLURM (sbatch) not found here — using CPU fallback${NC}"
      else
        echo -e "${GREEN}  → submitting SLURM GPU job (slurm/llm_gpu.sbatch)...${NC}"
        rm -f "$GPU_ENDPOINT_FILE"
        GPU_JOBID=$(sbatch --parsable slurm/llm_gpu.sbatch 2>/dev/null || true)
        if [ -z "$GPU_JOBID" ]; then
          echo -e "${RED}  ✗ sbatch failed (check --account / GPU availability) — using CPU fallback${NC}"
        else
          echo -e "${YELLOW}  job $GPU_JOBID queued; waiting for the GPU model to load (up to ~5 min)...${NC}"
          for _i in $(seq 1 60); do
            sleep 5
            if [ -f "$GPU_ENDPOINT_FILE" ] && \
               curl -s -o /dev/null --max-time 3 "http://$(cat "$GPU_ENDPOINT_FILE")/api/version"; then
              echo -e "\n${GREEN}  ✓ GPU LLM up at $(cat "$GPU_ENDPOINT_FILE") (job $GPU_JOBID)${NC}"
              break
            fi
            printf "."
          done
          echo
          [ -f "$GPU_ENDPOINT_FILE" ] || echo -e "${YELLOW}  ! GPU job not ready yet (still queued?) — using CPU fallback for now; re-run once it's up${NC}"
        fi
      fi
    fi

    # Prefer a GPU-served model if a SLURM GPU job is up and reachable (see
    # slurm/llm_gpu.sbatch). It publishes its endpoint on shared NFS; if we can
    # reach it, use the strong GPU model (gpt-oss:20b) instead of the local
    # CPU fallback (qwen2.5:3b) — much faster, much stronger.
    GPU_CONFIG="llm-gateway/config.gpu.yaml"
    LLM_CONFIG="llm-gateway/config.yaml"
    USE_GPU_LLM=0
    if [ -f "$GPU_ENDPOINT_FILE" ] && \
       curl -s -o /dev/null --max-time 3 "http://$(cat "$GPU_ENDPOINT_FILE")/api/version"; then
      echo -e "${GREEN}  ✓ GPU LLM detected at $(cat "$GPU_ENDPOINT_FILE") — using gpt-oss:20b${NC}"
      ./slurm/point_to_gpu.sh > /tmp/cliniq_point_gpu.log 2>&1 && USE_GPU_LLM=1
      [ -f "$GPU_CONFIG" ] && LLM_CONFIG="$GPU_CONFIG"
    fi

    if [ "$USE_GPU_LLM" != "1" ]; then
      # No GPU job — fall back to the local CPU model on this node.
      if ! curl -s -o /dev/null --max-time 2 http://127.0.0.1:11434/api/version; then
        echo -e "${GREEN}  → starting Ollama server (CPU)${NC}"
        nohup env OLLAMA_MODELS="$OLLAMA_DATA" OLLAMA_FLASH_ATTENTION=0 \
          LD_LIBRARY_PATH="$OLLAMA_LIB" "$OLLAMA_BIN" serve \
          > /tmp/cliniq_ollama.log 2>&1 &
        disown
        sleep 3
      else
        echo -e "${GREEN}  ✓ Ollama already running${NC}"
      fi

      # Pull the fallback model once; no-op on subsequent runs.
      if ! OLLAMA_MODELS="$OLLAMA_DATA" LD_LIBRARY_PATH="$OLLAMA_LIB" "$OLLAMA_BIN" list 2>/dev/null | grep -q "qwen2.5:3b"; then
        echo -e "${GREEN}  → pulling qwen2.5:3b (~2GB, first run only)${NC}"
        OLLAMA_MODELS="$OLLAMA_DATA" LD_LIBRARY_PATH="$OLLAMA_LIB" "$OLLAMA_BIN" pull qwen2.5:3b
      fi

      if ! curl -s -o /dev/null --max-time 2 http://127.0.0.1:4000/health/liveliness; then
        echo -e "${GREEN}  → starting LiteLLM proxy (:4000, CPU fallback)${NC}"
        nohup litellm --config "$LLM_CONFIG" --port 4000 \
          > /tmp/cliniq_litellm.log 2>&1 &
        disown
        sleep 4
      else
        echo -e "${GREEN}  ✓ LiteLLM proxy already running${NC}"
      fi
    fi

    # Route chatbot-app through the local proxy instead of jetstream directly.
    # MODEL stays "gpt-oss-120b" — that's the model-group name in the proxy
    # config; the fallback happens inside LiteLLM, transparent to the chatbot.
    export API_BASE_URL="http://127.0.0.1:4000/"
    export API_KEY="local-fallback-gateway"
  fi
  echo
fi

echo -e "${BLUE}Starting services (each in a separate process):${NC}"
echo

# 1. Bone Detection (:8001)
echo -e "${GREEN}[1/7]${NC} bone-detect (:8001)..."
(cd bone-detect && python api/server.py) &
sleep 2

# 2. Oral X-Ray (:8002)
echo -e "${GREEN}[2/7]${NC} oral-xray (:8002)..."
(cd oral-xray && python api/server.py) &
sleep 2

# 3. Chest X-Ray (:8003)
echo -e "${GREEN}[3/7]${NC} chest_xray (:8003)..."
(cd chest_xray && python -m api.server) &
sleep 2

# 4. Oral Classify (:8004)
echo -e "${GREEN}[4/7]${NC} oral-classify (:8004)..."
(cd oral-classify && python -m api.server) &
sleep 2

# 5. Prescription Parser (:8005) — optional, heavy (VLM)
if [ "$SKIP_PRESCRIPTION" != "1" ]; then
  echo -e "${GREEN}[5/7]${NC} prescription-parser (:8005)..."
  (cd prescription-parser && python api/server.py) &
  sleep 3
else
  echo -e "${YELLOW}[5/7] prescription-parser — SKIPPED${NC}"
fi
echo

# 6. Chatbot Gateway (:5000)
echo -e "${GREEN}[6/7]${NC} chatbot-app (:5000)..."
(cd chatbot-app && python app.py) &
sleep 2

# 7. Triage Worklist (:8010) — only if queue enabled
if [ "$MODE" != "none" ]; then
  echo -e "${GREEN}[7/7]${NC} triage worklist (:8010)..."
  (cd triage && python app.py) &
  sleep 2
else
  echo -e "${YELLOW}[7/7] triage worklist — SKIPPED (no async queue)${NC}"
fi
echo

echo -e "${GREEN}✓ All services started${NC}"
echo
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Service URLs:${NC}"
echo -e "  Chatbot         : ${BLUE}http://127.0.0.1:5000${NC}"
echo -e "  Bone Detect     : ${BLUE}http://127.0.0.1:8001/docs${NC}"
echo -e "  Oral X-Ray      : ${BLUE}http://127.0.0.1:8002/docs${NC}"
echo -e "  Chest X-Ray     : ${BLUE}http://127.0.0.1:8003/docs${NC}"
echo -e "  Oral Classify   : ${BLUE}http://127.0.0.1:8004/docs${NC}"
echo -e "  Prescription    : ${BLUE}http://127.0.0.1:8005/docs${NC}"
if [ "$MODE" != "none" ]; then
  echo -e "  Triage Worklist : ${BLUE}http://127.0.0.1:8010${NC}"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo

# Keep running
wait
