#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_LIST="${1:-$PROJECT_ROOT/jobs/PDV_vs/models.txt}"
EXP_ID="${2:-pvd_bt_full_array_$(date +%F_%H-%M-%S)}"
TASK_IDX="${3:-none}"
MAX_BT_COUNT="${4:-3}"
MAX_VERIFY_COUNT="${5:-2}"
CONCURRENCY="${6:-7}"
BOT_MODEL_ID="${7:-}"
TEMPERATURE="${8:-0.0}"

cd "$PROJECT_ROOT"

mkdir -p \
  logs/PDV_vs/slurm \
  "logs/PDV_vs/${EXP_ID}" \
  "jobs/PDV_vs/tmp"

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
echo "Max verify:   $MAX_VERIFY_COUNT"
echo "Concurrency:  $CONCURRENCY"
echo "Bot model:    ${BOT_MODEL_ID:-<none>}"
echo "Temperature:  $TEMPERATURE"
echo "List format:  top_model OR top_model|bot_model"

sbatch \
  --array="1-${MODEL_COUNT}%${CONCURRENCY}" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MODEL_LIST="$MODEL_LIST",EXP_ID="$EXP_ID",TASK_IDX="$TASK_IDX",MAX_BT_COUNT="$MAX_BT_COUNT",MAX_VERIFY_COUNT="$MAX_VERIFY_COUNT",BOT_MODEL_ID="$BOT_MODEL_ID",TEMPERATURE="$TEMPERATURE" \
  jobs/PDV_vs/run_model_array.slurm
