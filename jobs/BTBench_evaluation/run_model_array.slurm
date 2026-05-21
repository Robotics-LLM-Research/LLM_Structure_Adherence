#!/bin/bash
#SBATCH --job-name=BTBench_evaluation_test
#SBATCH --account=abedi
#SBATCH --qos=abedi
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-user=di246213@ucf.edu
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --output=logs/BTBench_evaluation/slurm/%x-%A_%a.out
#SBATCH --error=logs/BTBench_evaluation/slurm/%x-%A_%a.err

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Research/LLM_Structure_Adherence}"
MODEL_LIST="${MODEL_LIST:-$PROJECT_ROOT/jobs/BTBench_evaluation/models.txt}"
EXP_ID="${EXP_ID:-bt_cd_full_array_${SLURM_ARRAY_JOB_ID}}"
TASK_IDX="${TASK_IDX:-none}"
MAX_BT_COUNT="${MAX_BT_COUNT:-3}"
USE_CD="${USE_CD:-true}"
CONTAINER_PATH="${CONTAINER_PATH:-$HOME/containers/vllm-openai-latest.sif}"

cd "$PROJECT_ROOT" || exit 1

mkdir -p \
  "logs/BTBench_evaluation/slurm" \
  "logs/BTBench_evaluation/${EXP_ID}"

mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODEL_LIST")

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_ID="${MODELS[$MODEL_INDEX]}"

SAFE_MODEL="${MODEL_ID//\//__}"
SAFE_MODEL="${SAFE_MODEL//:/_}"
MODEL_LOG="logs/BTBench_evaluation/${EXP_ID}/${SAFE_MODEL}.log"

echo "Array job id: $SLURM_ARRAY_JOB_ID"
echo "Array task id: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Model list: $MODEL_LIST"
echo "Model id: $MODEL_ID"
echo "Exp id: $EXP_ID"
echo "Task idx: $TASK_IDX"
echo "Max BT count: $MAX_BT_COUNT"
echo "Use CD: $USE_CD"
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

CD_ARGS=()
if [[ "$USE_CD" == "true" || "$USE_CD" == "True" || "$USE_CD" == "1" ]]; then
  CD_ARGS+=(--use-cd)
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

    python3 jobs/BTBench_evaluation/run_one_bt_model.py "$@"
  ' _ \
  "$PROJECT_ROOT" \
  "$MODEL_ID" \
  "${CD_ARGS[@]}" \
  "${TASK_ARGS[@]}" \
  --max-bt-count "$MAX_BT_COUNT" \
  --exp-id "$EXP_ID" \
  2>&1 | tee "$MODEL_LOG"

status=${PIPESTATUS[0]}

echo "Exit code: $status"
nvidia-smi || true

exit "$status"