# Scripts

Experiment drivers and one-off runners. Shared logic lives under `src/` (model loading, parsers, simulator, prompts, result helpers).

## Active

### `run_bt_tasks_online.py`

Runs the **behavior-tree task suite** with **live inference** (vLLM backend).

- Loads tasks from `src/tasks/tasks_100.json`.
- For each task, starts a chat in the `bt_tasks` prompt mode, injects the JSON schema sample, and queries the model up to `MAX_BT_COUNT` (3) times per episode.
- Parses each reply with `parse_bt_output`; on structural errors, appends feedback and retries. On valid trees, runs `simulate_bt_plan` with the task’s `world` and `task_type`, then appends outcome feedback if the task is not yet successful.
- Writes one file per task under `results/<experiment>/<model_id>/tasks/task_<task_id>.json`, plus an aggregate `main_results.json`.
- **Resume:** if a task file already exists, that index is skipped so long runs can continue after interruption.

**How to run:** there is no CLI in this file; call `main` from Python (see `jobs/job-bt_online_all_tasks.py`), for example:

```python
from scripts.run_bt_tasks_online import main

main(model_id="Qwen/Qwen2.5-7B-Instruct", tasks_idx=None, exp_id="my_exp")
```

Optional `tasks_idx` limits which task indices to run; `exp_id` groups outputs under `results/<exp_id>/`.

---

## Archive (`archive/`)

Older or alternate setups kept for reference. Most of these assume imports like `import src...` resolve correctly: **run from the repository root** and set `PYTHONPATH` to that root (for example `export PYTHONPATH=$PWD` on bash, or `$env:PYTHONPATH = (Get-Location)` in PowerShell before `python scripts/archive/<script>.py`). A few files also prepend a computed path to `sys.path`; if imports fail after moving scripts into `archive/`, point `PYTHONPATH` at the repo root.

| Script | Purpose |
|--------|--------|
| `run_wall_steps.py` | **Wall-crossing, step-by-step:** one JSON action per turn, `simulate_action_step`, schemas from `STEP_SCHEMAS`, feedback from `wall_steps`. |
| `run_wall_path.py` | **Wall-crossing, full path:** single response with a full plan, `simulate_action_plan`, `PATH_SCHEMAS`. |
| `run_wall_bt.py` | **Wall-crossing, behavior tree:** fixed `WALL_TASK_ENV`, `wall_bt` prompts, `simulate_bt_plan`; typically one BT per episode (`MAX_BT_COUNT = 1`). |
| `run_bt_tasks_offline.py` | **No model calls:** reads saved strings from `src/tasks/ground_truth_responses.json`, matches `task_id` to `tasks_100.json`, parses with `parse_bt_output`, simulates with `simulate_bt_plan`, and writes aggregate JSON under `results/` (run id from `format_run_timestamp`). |
| `multi_dog_targets.py` | **Multi-robot hardware loop:** stepwise multi-dog JSON commands, HTTP pose refresh over `DOG_PORTS`, `execute_multi_dog_commands` / `parse_multi_dog_step_output`; uses `.env` / `HF_TOKEN` and an `EXPERIMENTS` grid at the bottom of the file. |

Example:

```bash
cd /path/to/LLM_Structure_Adherence
export PYTHONPATH=$PWD
python scripts/archive/run_wall_steps.py
```
