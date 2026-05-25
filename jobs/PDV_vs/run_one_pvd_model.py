import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bt_tasks_pvd import main as run_bt_tasks_pvd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PVD model in one Python process."
    )

    parser.add_argument("top_model_id")
    parser.add_argument("--bot-model-id", default=None)
    parser.add_argument("--exp-id", default="pvd_bt_smoke_one_task")
    parser.add_argument("--task-idx", type=int, default=None)
    parser.add_argument("--max-bt-count", type=int, default=3)
    parser.add_argument("--max-verify-count", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--backend", default="vllm")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_bt_tasks_pvd(
        top_model_id=args.top_model_id,
        bot_model_id=args.bot_model_id,
        max_bt_count=args.max_bt_count,
        max_verify_count=args.max_verify_count,
        tasks_idx=None if args.task_idx is None else [args.task_idx],
        exp_id=args.exp_id,
        backend=args.backend,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
