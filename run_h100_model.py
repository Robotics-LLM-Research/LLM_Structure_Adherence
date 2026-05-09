import argparse

from scripts.run_bt_tasks_dccd import main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--exp-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_id=args.model_id,
        tasks_idx=None,
        exp_id=args.exp_id,
    )
