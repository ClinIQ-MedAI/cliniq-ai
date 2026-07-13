#!/bin/bash
# ClinIQ — Route x-ray models to their GPU SLURM jobs
#
# After starting GPU jobs with:
#   sbatch slurm/bone_detect_gpu.sbatch
#   sbatch slurm/oral_xray_gpu.sbatch
#   sbatch slurm/chest_xray_gpu.sbatch
#   sbatch slurm/oral_classify_gpu.sbatch
#
# Run this to verify they're reachable and configure runs.sh to use them.

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ROOT="/data/user/moabouag/far/cliniq"

MODELS=(
  "bone_detect:8001"
  "oral_xray:8002"
  "chest_xray:8003"
  "oral_classify:8004"
)

echo -e "${YELLOW}Checking x-ray GPU endpoints...${NC}\n"

ALL_READY=1
for model_port in "${MODELS[@]}"; do
  model="${model_port%:*}"
  port="${model_port#*:}"
  endpoint_file="$ROOT/.local/${model}_endpoint.txt"

  if [ ! -f "$endpoint_file" ]; then
    echo -e "${RED}✗ $model: endpoint file not found${NC}"
    echo -e "  ${YELLOW}Submit job: sbatch slurm/${model}_gpu.sbatch${NC}"
    ALL_READY=0
    continue
  fi

  endpoint="$(cat "$endpoint_file")"
  echo -n "  $model ($endpoint:$port) ... "

  if curl -s -o /dev/null --max-time 3 "http://$endpoint:$port/health" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${RED}✗ unreachable${NC}"
    echo -e "    ${YELLOW}Check: squeue -u \$USER | grep $model${NC}"
    echo -e "    ${YELLOW}Logs: tail -f slurm/${model}_gpu-*.log${NC}"
    ALL_READY=0
  fi
done

if [ "$ALL_READY" = "0" ]; then
  echo -e "\n${RED}Some endpoints not ready. Wait for jobs to start, then re-run this script.${NC}"
  exit 1
fi

echo -e "\n${GREEN}✓ All x-ray GPU servers ready!${NC}"
echo -e "${YELLOW}Now run:  GPU=1 SKIP_PRESCRIPTION=1 ./runs.sh${NC}"
echo -e "${GREEN}The login-node services will connect to GPU servers.${NC}\n"

# Show status summary
echo "Summary:"
for model_port in "${MODELS[@]}"; do
  model="${model_port%:*}"
  port="${model_port#*:}"
  endpoint_file="$ROOT/.local/${model}_endpoint.txt"
  if [ -f "$endpoint_file" ]; then
    endpoint="$(cat "$endpoint_file")"
    echo "  $model: http://$endpoint:$port"
  fi
done
