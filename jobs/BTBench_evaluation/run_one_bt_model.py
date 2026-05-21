import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bt_tasks_online import main as run_bt_tasks_online



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one BT benchmark model in one Python process."
    )

    parser.add_argument("model_id")
    parser.add_argument("--exp-id", default="bt_cd_smoke_one_task")
    parser.add_argument("--task-idx", type=int, default=None)
    parser.add_argument("--max-bt-count", type=int, default=1)
    parser.add_argument("--backend", default="vllm")

    parser.add_argument(
        "--use-cd",
        action="store_true",
        help="Enable constrained decoding.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool declarations in the chat template.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    run_bt_tasks_online(
        model_id=args.model_id,
        max_bt_count=args.max_bt_count,
        use_tools=not args.no_tools,
        use_cd=args.use_cd,
        tasks_idx=None if args.task_idx is None else [args.task_idx],
        exp_id=args.exp_id,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()