#!/bin/bash
#SBATCH --job-name=PDV_vs_test
#SBATCH --account=abedi
#SBATCH --qos=abedi
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-user=di246213@ucf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/PDV_vs/slurm/%x-%A_%a.out
#SBATCH --error=logs/PDV_vs/slurm/%x-%A_%a.err

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Research/LLM_Structure_Adherence}"
MODEL_LIST="${MODEL_LIST:-$PROJECT_ROOT/jobs/PDV_vs/models.txt}"
EXP_ID="${EXP_ID:-pvd_bt_full_array_${SLURM_ARRAY_JOB_ID}}"
TASK_IDX="${TASK_IDX:-none}"
MAX_BT_COUNT="${MAX_BT_COUNT:-3}"
MAX_VERIFY_COUNT="${MAX_VERIFY_COUNT:-2}"
BOT_MODEL_ID="${BOT_MODEL_ID:-}"
CONTAINER_PATH="${CONTAINER_PATH:-$HOME/containers/vllm-openai-latest.sif}"

cd "$PROJECT_ROOT" || exit 1

mkdir -p \
  "logs/PDV_vs/slurm" \
  "logs/PDV_vs/${EXP_ID}"

mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODEL_LIST")

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_SPEC="${MODELS[$MODEL_INDEX]}"

# Each line can be either:
# - top_model_id
# - top_model_id|bot_model_id
TOP_MODEL_ID="$MODEL_SPEC"
BOT_MODEL_ID_FROM_SPEC=""
if [[ "$MODEL_SPEC" == *"|"* ]]; then
  IFS='|' read -r TOP_MODEL_ID BOT_MODEL_ID_FROM_SPEC <<< "$MODEL_SPEC"
fi

ACTIVE_BOT_MODEL_ID="$BOT_MODEL_ID"
if [[ -n "$BOT_MODEL_ID_FROM_SPEC" ]]; then
  ACTIVE_BOT_MODEL_ID="$BOT_MODEL_ID_FROM_SPEC"
fi

SAFE_MODEL="${TOP_MODEL_ID//\//__}"
if [[ -n "$ACTIVE_BOT_MODEL_ID" ]]; then
  SAFE_MODEL="${SAFE_MODEL}__${ACTIVE_BOT_MODEL_ID//\//__}"
fi
SAFE_MODEL="${SAFE_MODEL//:/_}"
MODEL_LOG="logs/PDV_vs/${EXP_ID}/${SAFE_MODEL}.log"

echo "Array job id: $SLURM_ARRAY_JOB_ID"
echo "Array task id: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Model list: $MODEL_LIST"
echo "Model spec: $MODEL_SPEC"
echo "Top model id: $TOP_MODEL_ID"
echo "Bot model id (active): ${ACTIVE_BOT_MODEL_ID:-<none>}"
echo "Exp id: $EXP_ID"
echo "Task idx: $TASK_IDX"
echo "Max BT count: $MAX_BT_COUNT"
echo "Max verify count: $MAX_VERIFY_COUNT"
echo "Bot model id (global): ${BOT_MODEL_ID:-<none>}"
echo "Container path: $CONTAINER_PATH"
echo "Model log: $MODEL_LOG"

module purge
module load apptainer/apptainer-1.3.3-go-1.22.5

export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"
export APPTAINERENV_HF_HOME="$PROJECT_ROOT/.hf_cache"

TASK_ARGS=()
if [[ "$TASK_IDX" != "none" && "$TASK_IDX" != "None" && -n "$TASK_IDX" ]]; then
  TASK_ARGS+=(--task-idx "$TASK_IDX")
fi

BOT_ARGS=()
if [[ -n "$ACTIVE_BOT_MODEL_ID" ]]; then
  BOT_ARGS+=(--bot-model-id "$ACTIVE_BOT_MODEL_ID")
fi

apptainer exec --cleanenv --nv \
  --bind "$HOME:$HOME" \
  "$CONTAINER_PATH" \
  bash -lc '
    project_root="$1"
    shift

    cd "$project_root" || exit 1

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_DEEP_GEMM_WARMUP=skip

    python3 jobs/PDV_vs/run_one_pvd_model.py "$@"
  ' _ \
  "$PROJECT_ROOT" \
  "$TOP_MODEL_ID" \
  "${BOT_ARGS[@]}" \
  "${TASK_ARGS[@]}" \
  --max-bt-count "$MAX_BT_COUNT" \
  --max-verify-count "$MAX_VERIFY_COUNT" \
  --exp-id "$EXP_ID" \
  2>&1 | tee "$MODEL_LOG"

status=${PIPESTATUS[0]}

echo "Exit code: $status"
nvidia-smi || true

exit "$status"
