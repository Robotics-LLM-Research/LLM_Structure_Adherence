#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_LIST="${1:-$SCRIPT_DIR/models.txt}"
EXP_ID="${2:-pvd_bt_smoke_one_task}"
TASK_IDX="${3:-0}"
MAX_BT_COUNT="${4:-3}"
MAX_VERIFY_COUNT="${5:-2}"
BOT_MODEL_ID="${6:-}"

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
echo "Bot model:    ${BOT_MODEL_ID:-<none>}"

while IFS= read -r model_spec; do
    if [[ -z "$model_spec" || "$model_spec" == \#* ]]; then
        continue
    fi

    # Each line can be either:
    # - top_model_id
    # - top_model_id|bot_model_id
    top_model_id="$model_spec"
    bot_model_id_from_spec=""
    if [[ "$model_spec" == *"|"* ]]; then
        IFS='|' read -r top_model_id bot_model_id_from_spec <<< "$model_spec"
    fi

    active_bot_model_id="$BOT_MODEL_ID"
    if [[ -n "$bot_model_id_from_spec" ]]; then
        active_bot_model_id="$bot_model_id_from_spec"
    fi

    safe_model="${top_model_id//\//__}"
    if [[ -n "$active_bot_model_id" ]]; then
        safe_model="${safe_model}__${active_bot_model_id//\//__}"
    fi
    safe_model="${safe_model//:/_}"
    timestamp="$(date +%F_%H-%M-%S)"
    log_path="logs/PDV_vs/${EXP_ID}_${safe_model}_${timestamp}.log"

    echo
    echo "======================================================================"
    echo "START: $model_spec"
    echo "TOP:   $top_model_id"
    echo "BOT:   ${active_bot_model_id:-<none>}"
    echo "LOG:   $log_path"
    echo "======================================================================"

    BOT_ARGS=()
    if [[ -n "$active_bot_model_id" ]]; then
        BOT_ARGS+=(--bot-model-id "$active_bot_model_id")
    fi

    python3 jobs/PDV_vs/run_one_pvd_model.py "$top_model_id" \
        "${BOT_ARGS[@]}" \
        --task-idx "$TASK_IDX" \
        --max-bt-count "$MAX_BT_COUNT" \
        --max-verify-count "$MAX_VERIFY_COUNT" \
        --exp-id "$EXP_ID" \
        2>&1 | tee "$log_path"

    status=${PIPESTATUS[0]}

    echo
    echo "======================================================================"
    echo "DONE: $model_spec"
    echo "EXIT CODE: $status"
    echo "======================================================================"

    nvidia-smi || true
    sleep 10
done < "$MODEL_LIST"
