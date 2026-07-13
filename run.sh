#!/bin/bash
# ClinIQ — Unified launcher for GPU-accelerated deployment
#
# Usage:
#   ./run.sh shared       All services share one GPU (simpler, cheaper)
#   ./run.sh all          Each service gets its own GPU (faster, more resources)

# test
# python -m messaging.cli enqueue --modality dental_xray --image /data/user/moabouag/far/cliniq/sample_data/dental1.jpg

# python -m messaging.cli listen
# python -m messaging.cli ping

#./run.sh shared
# ./run.sh all


set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m'

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-}"

# Banner
banner() {
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║              ClinIQ Platform — Launcher                    ║${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

# Helper: print section
section() {
  echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${YELLOW}$1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Helper: check if SLURM available
check_slurm() {
  if ! command -v sbatch >/dev/null 2>&1; then
    echo -e "${RED}✗ SLURM not available (sbatch not found)${NC}"
    echo -e "${YELLOW}  Falling back to local CPU mode${NC}"
    return 1
  fi
  return 0
}

# Helper: wait for endpoint
wait_endpoint() {
  local endpoint_file=$1
  local name=$2
  local timeout=60

  for i in $(seq 1 $timeout); do
    if [ -f "$endpoint_file" ] && curl -s -o /dev/null --max-time 2 "http://$(cat "$endpoint_file"):8001/health" 2>/dev/null; then
      return 0
    fi
    [ $((i % 10)) -eq 0 ] && echo -e "${YELLOW}  Waiting for $name... ($i/$timeout)${NC}"
    sleep 1
  done
  return 1
}

# Helper: get GPU info
get_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'Unknown')"
  else
    echo "N/A"
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED GPU MODE
# The ENTIRE stack (LLM + all x-ray models + chatbot) runs INSIDE one SLURM GPU
# allocation. Nothing is started on the login node — this launcher only submits
# the job, waits for it to come up, and prints the per-service GPU summary.
# ═══════════════════════════════════════════════════════════════════════════════
run_shared() {
  banner
  section "MODE: Shared GPU (entire stack inside ONE L40S allocation)"

  if ! check_slurm; then
    echo -e "${RED}✗ SLURM required for 'shared' mode (need a GPU allocation).${NC}"
    echo -e "${YELLOW}  Without SLURM, run the plain CPU stack:  GPU=1 SKIP_PRESCRIPTION=1 ./runs.sh${NC}"
    exit 1
  fi

  local ready_file="$ROOT/.local/cliniq_shared_ready"
  local endpoint_file="$ROOT/.local/cliniq_shared_gpu_endpoint.txt"
  local summary_file="$ROOT/.local/cliniq_shared_summary.txt"

  # Reuse an existing allocation if one is already up, else submit a fresh job.
  local existing_job
  existing_job=$(squeue -u "$USER" --name="cliniq-gpu-shared" -h --format="%i" 2>/dev/null | tail -1 || true)

  local job_id="$existing_job"
  if [ -n "$existing_job" ]; then
    echo -e "${GREEN}✓ Shared GPU job already running: $existing_job${NC}"
  else
    echo -e "${YELLOW}Submitting shared GPU job (whole stack on one GPU)...${NC}"
    rm -f "$ready_file" "$endpoint_file" "$summary_file"
    job_id=$(sbatch --parsable "$ROOT/slurm/cliniq_gpu_shared.sbatch" 2>/dev/null || echo "")
    if [ -z "$job_id" ]; then
      echo -e "${RED}✗ Failed to submit job (check --account=thetan-ai and GPU availability).${NC}"
      exit 1
    fi
    echo -e "${GREEN}✓ Job submitted: $job_id${NC}"
  fi

  # Wait for the job to publish its readiness marker (LLM + x-ray + chatbot up).
  section "Waiting for the stack to come up inside the allocation"
  echo -e "${YELLOW}This can take a few minutes (queue + model loads)...${NC}"

  local waited=0 timeout=600
  while [ ! -f "$ready_file" ] && [ $waited -lt $timeout ]; do
    # Bail early if the job left the queue without becoming ready.
    if ! squeue -j "$job_id" -h >/dev/null 2>&1 && [ -n "$job_id" ]; then
      if ! squeue -j "$job_id" -h 2>/dev/null | grep -q .; then
        echo -e "\n${RED}✗ Job $job_id is no longer in the queue and never signalled ready.${NC}"
        echo -e "${YELLOW}  Check the log: tail -n 40 slurm/cliniq_gpu_shared-$job_id.log${NC}"
        tail -n 20 "$ROOT/slurm/cliniq_gpu_shared-$job_id.log" 2>/dev/null || true
        exit 1
      fi
    fi
    [ $((waited % 15)) -eq 0 ] && echo -e "${YELLOW}  ... still waiting (${waited}s)${NC}"
    sleep 5
    waited=$((waited + 5))
  done

  if [ ! -f "$ready_file" ]; then
    echo -e "${RED}✗ Timed out after ${timeout}s waiting for the stack.${NC}"
    echo -e "${YELLOW}  Log: tail -f slurm/cliniq_gpu_shared-$job_id.log${NC}"
    exit 1
  fi

  # Show the per-service Node / GPU / CUDA table the job produced.
  section "Stack summary"
  if [ -f "$summary_file" ]; then
    cat "$summary_file"
  else
    echo -e "${YELLOW}(summary file not found; check the job log)${NC}"
  fi

  # The stack lives on the compute node. Forward its ports onto the login node's
  # localhost so http://127.0.0.1:5000 just works here (and VS Code's automatic
  # port forwarding then carries it through to the user's browser).
  local node="?"
  [ -f "$endpoint_file" ] && node="$(cat "$endpoint_file")"

  section "ClinIQ is running INSIDE the GPU allocation on: $node"
  echo -e "${GREEN}Everything (LLM + x-ray models + chatbot) shares GPU on $node.${NC}"
  echo -e "${GREEN}Nothing was started on the login node.${NC}\n"

  open_tunnel "$node"

  echo -e "\n${YELLOW}Live logs:${NC}   ${BLUE}tail -f slurm/cliniq_gpu_shared-$job_id.log${NC}"
  echo -e "${YELLOW}Stop stack:${NC}  ${BLUE}scancel $job_id${NC}"
  echo -e "${YELLOW}Close tunnel:${NC} ${BLUE}kill \$(cat .local/cliniq_tunnel.pid)${NC}\n"
}

# Forward chatbot + x-ray ports from the compute node onto this login node.
open_tunnel() {
  local node="$1"

  # If a tunnel from a previous run is already up and healthy, keep it — no need
  # to churn it. Otherwise drop any stale one (a dead tunnel still holds the -L
  # bind and would make the new ssh fail). The marker file records its PID.
  local pidfile="$ROOT/.local/cliniq_tunnel.pid"
  if curl -s -o /dev/null --max-time 3 "http://127.0.0.1:5000/" 2>/dev/null; then
    echo -e "${GREEN}✓ Tunnel already up — reusing it.${NC}"
    echo -e "  ${BLUE}http://127.0.0.1:5000${NC}"
    return 0
  fi
  if [ -f "$pidfile" ]; then
    kill "$(cat "$pidfile")" 2>/dev/null || true
    rm -f "$pidfile"
    sleep 1
  fi

  echo -e "${YELLOW}Forwarding ports from $node to this login node...${NC}"
  # setsid detaches the tunnel into its own session so it survives you closing
  # VS Code / logging out of the login node — otherwise it dies with the session
  # and :5000 stops answering until you re-run this. ServerAliveInterval keeps it
  # from going stale across network blips; the SLURM job itself is independent and
  # keeps running regardless. We background it and check the port, rather than
  # relying on ssh -f, so a failed forward is caught here.
  setsid ssh -N \
      -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L 5000:127.0.0.1:5000 \
      -L 8001:127.0.0.1:8001 \
      -L 8002:127.0.0.1:8002 \
      -L 8003:127.0.0.1:8003 \
      -L 8004:127.0.0.1:8004 \
      -L 8005:127.0.0.1:8005 \
      -L 8010:127.0.0.1:8010 \
      "$node" </dev/null >/dev/null 2>&1 &
  local tunnel_pid=$!
  disown 2>/dev/null || true
  echo "$tunnel_pid" > "$pidfile"

  sleep 3
  if curl -s -o /dev/null --max-time 5 "http://127.0.0.1:5000/" 2>/dev/null; then
    echo -e "${GREEN}✓ Tunnel up (survives logout). Open the chatbot at:${NC}"
    echo -e "  ${BLUE}http://127.0.0.1:5000${NC}"
    echo -e "${GREEN}  X-ray API docs: 127.0.0.1:8001/docs .. 8004/docs${NC}"
    if curl -s -o /dev/null --max-time 3 "http://127.0.0.1:8010/health" 2>/dev/null; then
      echo -e "${GREEN}  Triage worklist: ${BLUE}http://127.0.0.1:8010${NC}"
    fi
  else
    echo -e "${YELLOW}! Tunnel started but the chatbot did not answer yet — retry in a few seconds.${NC}"
    echo -e "${YELLOW}  Or run it manually from your laptop:${NC}"
    echo -e "  ${BLUE}ssh -L 5000:$node:5000 $USER@$(hostname)${NC}"
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SEPARATE GPU JOBS MODE
# ═══════════════════════════════════════════════════════════════════════════════
run_all() {
  banner
  section "MODE: Individual GPUs (Each service on separate L40S)"

  if ! check_slurm; then
    echo -e "${RED}✗ SLURM required for 'all' mode${NC}"
    echo -e "${YELLOW}Use './run.sh shared' for local CPU mode${NC}"
    exit 1
  fi

  local services=(
    "bone_detect_gpu:bone-detect"
    "oral_xray_gpu:oral-xray"
    "chest_xray_gpu:chest-xray"
    "oral_classify_gpu:oral-classify"
  )

  local job_ids=()

  echo -e "${YELLOW}Submitting individual GPU jobs...${NC}"

  for service_pair in "${services[@]}"; do
    local sbatch_name="${service_pair%:*}"
    local service_name="${service_pair#*:}"
    local sbatch_file="$ROOT/slurm/${sbatch_name}.sbatch"

    if [ ! -f "$sbatch_file" ]; then
      echo -e "${RED}✗ Missing: $sbatch_file${NC}"
      exit 1
    fi

    echo -n "  $service_name ... "
    local job_id=$(sbatch --parsable "$sbatch_file" 2>/dev/null || echo "")

    if [ -z "$job_id" ]; then
      echo -e "${RED}✗${NC}"
      exit 1
    fi

    echo -e "${GREEN}✓ ($job_id)${NC}"
    job_ids+=("$job_id")
  done

  section "Verifying services"

  local endpoint_files=(
    "$ROOT/.local/bone_detect_endpoint.txt"
    "$ROOT/.local/oral_xray_endpoint.txt"
    "$ROOT/.local/chest_xray_endpoint.txt"
    "$ROOT/.local/oral_classify_endpoint.txt"
  )

  local ports=(8001 8002 8003 8004)
  local service_names=("Bone Detect" "Oral X-Ray" "Chest X-Ray" "Oral Classify")

  echo -e "${YELLOW}Waiting for services to boot (~90s)...${NC}"

  local all_ready=1
  for i in "${!services[@]}"; do
    local endpoint_file="${endpoint_files[$i]}"
    local port="${ports[$i]}"
    local service_name="${service_names[$i]}"

    echo -n "  $service_name ... "

    rm -f "$endpoint_file"

    if wait_endpoint "$endpoint_file" "$service_name"; then
      local endpoint=$(cat "$endpoint_file")
      echo -e "${GREEN}✓${NC}"
      echo -e "    → ${BLUE}http://$endpoint:$port${NC}"
    else
      echo -e "${RED}✗${NC}"
      echo -e "    ${YELLOW}Still queued or booting. Check: squeue -u \$USER${NC}"
      all_ready=0
    fi
  done

  if [ "$all_ready" = "1" ]; then
    section "GPU Info"

    echo -e "${BLUE}Checking GPU allocation across nodes...${NC}"
    squeue -u "$USER" --name="cliniq*" --format="%.7i %.20j %.8T %.6D %.10N %.8c"

    echo -e "\n${GREEN}✓ All services ready for launch${NC}"
  else
    echo -e "\n${YELLOW}⚠ Some services still booting${NC}"
    echo -e "  Check logs: ${BLUE}tail -f slurm/bone_detect_gpu-*.log${NC}"
    echo -e "  Recheck:    ${BLUE}squeue -u \$USER${NC}"
  fi

  section "Starting ClinIQ on login node"
  cd "$ROOT"
  GPU=1 SKIP_PRESCRIPTION=1 ./runs.sh
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if [ -z "$MODE" ]; then
  banner
  echo -e "\n${YELLOW}Usage:${NC}"
  echo -e "  ${BLUE}./run.sh shared${NC}     All services share one GPU (simpler, cheaper)"
  echo -e "  ${BLUE}./run.sh all${NC}        Each service gets own GPU (faster, more resources)"
  echo -e "\n${YELLOW}Example:${NC}"
  echo -e "  ${BLUE}cd cliniq && ./run.sh shared${NC}\n"
  exit 0
fi

case "$MODE" in
  shared)
    run_shared
    ;;
  all)
    run_all
    ;;
  *)
    echo -e "${RED}✗ Unknown mode: $MODE${NC}"
    echo -e "${YELLOW}Use: ./run.sh {shared|all}${NC}"
    exit 1
    ;;
esac
