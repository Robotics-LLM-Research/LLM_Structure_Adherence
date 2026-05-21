#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_LIST="${1:-$PROJECT_ROOT/jobs/BTBench_evaluation/models.txt}"
EXP_ID="${2:-bt_cd_full_array_$(date +%F_%H-%M-%S)}"
TASK_IDX="${3:-none}"
MAX_BT_COUNT="${4:-3}"
CONCURRENCY="${5:-7}"

cd "$PROJECT_ROOT"

mkdir -p \
  logs/BTBench_evaluation/slurm \
  "logs/BTBench_evaluation/${EXP_ID}" \
  "jobs/BTBench_evaluation/tmp"

MODEL_COUNT="$(grep -cvE '^\s*(#|$)' "$MODEL_LIST")"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run: export HF_TOKEN='your_token_here'"
  exit 1
fi

echo "Submitting model array"
echo "Project root: $PROJECT_ROOT"
echo "Model list:   $MODEL_LIST"
echo "Model count:  $MODEL_COUNT"
echo "Exp id:       $EXP_ID"
echo "Task idx:     $TASK_IDX"
echo "Max BT count: $MAX_BT_COUNT"
echo "Concurrency:  $CONCURRENCY"

sbatch \
  --array="1-${MODEL_COUNT}%${CONCURRENCY}" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MODEL_LIST="$MODEL_LIST",EXP_ID="$EXP_ID",TASK_IDX="$TASK_IDX",MAX_BT_COUNT="$MAX_BT_COUNT" \
  jobs/BTBench_evaluation/run_model_array.slurm