# LLM Structure Adherence Research

This repository evaluates how well vision-language and instruction-tuned models follow required JSON output schemas while solving simple robot-navigation tasks. Experiments compare step-by-step action generation, full-path planning, and behavior-tree outputs, then score both structural adherence and task success.

## What This Repo Contains

- `src/`: shared prompts, schema definitions, parser normalization, simulator logic, model loading, tool declarations, and result helpers.
- `scripts/`: runnable experiment drivers. See [`scripts/README.md`](scripts/README.md) for a per-script description.
- `jobs/`: batch drivers and SLURM templates for multi-model or cluster runs (e.g. `job-bt_online_all_tasks.py`, `job_bt_online_tasks.slurm`).
- `assets/`: static environment assets such as `wall_crossing_env.png`.
- `results/`: saved run outputs and experiment summaries.
- `notebooks/`: analysis and visualization notebooks.

## Experiment Flow

1. A runner script selects prompts, schema variants, and model settings.
2. `src/model.py` builds the prompt messages and queries the model.
3. `src/parsers/` validates the raw JSON against the requested schema.
4. `src/simulator.py` applies actions or behavior-tree plans in the task environment.
5. `src/utils.py` writes raw run details and aggregate summaries.

## Core Modules

- `src/schemas/`: schema package split by concern (`base`, `step`, `path`, `bt`, `multi_dog`).
- `src/prompts/`: prompt package split by scenario (`wall_path`, `wall_steps`, `wall_bt`, `bt_tasks`, `multi_dog_step`, etc.) with centralized mode maps in `factory.py`.
- `src/model.py`: Hugging Face / vLLM model loading, message construction, and generation.
- `src/parsers/`: validation and normalization into shared action models.
- `src/simulator.py`: collision checks, movement updates, BT execution, and success detection.
- `src/utils.py`: prompt selection, timestamp formatting, and result persistence.
- `src/tools.py`: serialized tool declarations for prompt injection.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

For notebook-focused work, `requirements-colab.txt` provides the Colab-oriented dependency set.

Set environment variables in `.env` when needed:

```env
HF_TOKEN=your_huggingface_token
```

## Running Experiments

**Primary behavior-tree benchmark (online inference):** invoke `main` from `scripts/run_bt_tasks_online.py` (see [`scripts/README.md`](scripts/README.md)). The batch driver `jobs/job-bt_online_all_tasks.py` loops several model IDs and a fixed `exp_id`.

**Legacy wall-crossing drivers** (step, path, BT) and **offline scoring** of saved BT JSON live under `scripts/archive/`; see [`scripts/README.md`](scripts/README.md).

For cluster jobs, adjust `jobs/job_bt_online_tasks.slurm` to your environment (Conda module, paths, partition), then submit with `sbatch`. The SLURM script sets `PYTHONPATH` to the repo and runs the Python module named in that file.

## Outputs

- Per-config raw runs are written under `results/<run_id>/<model_id>/...` (layout depends on the script; BT online runs use `results/<exp_id>/<model_id>/tasks/` plus `main_results.json`).
- Experiment-level summaries are written per model and experiment id.
- Each saved payload includes both structure-adherence metrics and task-completion metrics where applicable.

## Notes

- Task definitions for the BT suite are documented in `src/tasks/README.md`.
- Step-sequence runs (archived wall scripts) allow corrective feedback after malformed outputs.
- Full-path runs test whether a model can produce an entire valid action plan in one response.
