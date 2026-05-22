#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_LIST="${1:-$SCRIPT_DIR/models.txt}"
EXP_ID="${2:-pvd_bt_smoke_one_task}"
TASK_IDX="${3:-0}"
MAX_BT_COUNT="${4:-3}"
MAX_VERIFY_COUNT="${5:-2}"

cd "$PROJECT_ROOT" || exit 1

mkdir -p logs/PDV_vs

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=INFO
export VLLM_DEEP_GEMM_WARMUP=skip

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set. Gated models may fail."
fi

echo "Project root: $PROJECT_ROOT"
echo "Model list:   $MODEL_LIST"
echo "Exp ID:       $EXP_ID"
echo "Task idx:     $TASK_IDX"
echo "Max BT count: $MAX_BT_COUNT"
echo "Max verify:   $MAX_VERIFY_COUNT"

while IFS= read -r model_id; do
    if [[ -z "$model_id" || "$model_id" == \#* ]]; then
        continue
    fi

    safe_model="${model_id//\//__}"
    safe_model="${safe_model//:/_}"
    timestamp="$(date +%F_%H-%M-%S)"
    log_path="logs/PDV_vs/${EXP_ID}_${safe_model}_${timestamp}.log"

    echo
    echo "======================================================================"
    echo "START: $model_id"
    echo "LOG:   $log_path"
    echo "======================================================================"

    python3 jobs/PDV_vs/run_one_pvd_model.py "$model_id" \
        --task-idx "$TASK_IDX" \
        --max-bt-count "$MAX_BT_COUNT" \
        --max-verify-count "$MAX_VERIFY_COUNT" \
        --exp-id "$EXP_ID" \
        2>&1 | tee "$log_path"

    status=${PIPESTATUS[0]}

    echo
    echo "======================================================================"
    echo "DONE: $model_id"
    echo "EXIT CODE: $status"
    echo "======================================================================"

    nvidia-smi || true
    sleep 10
done < "$MODEL_LIST"
