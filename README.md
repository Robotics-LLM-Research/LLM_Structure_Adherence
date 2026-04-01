# LLM Structure Adherence Research

This repository evaluates how well vision-language and instruction-tuned models follow required JSON output schemas while solving a simple robot-navigation task. The experiments compare step-by-step action generation against full-path planning, then score both structural adherence and task success.

## What This Repo Contains

- `src/`: shared prompts, schema definitions, parser normalization, simulator logic, model loading, tool declarations, and result helpers.
- `scripts/run_step_sequence.py`: evaluates one action at a time with simulator feedback after each turn.
- `scripts/run_full_path.py`: evaluates full plans generated in a single response.
- `assets/`: static environment assets such as `wall_crossing_env.png`.
- `results/`: saved run outputs and experiment summaries.
- `notebooks/`: analysis and visualization notebooks.
- `run.slurm`: cluster entry point for the step-sequence experiment.

## Experiment Flow

1. A runner script selects prompts, schema variants, and model settings.
2. `src/model.py` builds the prompt messages and queries the model.
3. `src/parser.py` validates the raw JSON against the requested schema.
4. `src/simulator.py` applies actions in the wall-crossing environment.
5. `src/utils.py` writes raw run details and aggregate summaries.

## Core Modules

- `src/schemas/`: schema package split by concern (`base`, `step`, `path`, `multi_dog`).
- `src/prompts/`: prompt package split by scenario (`single_path`, `single_step`, `multi_dog_step`) with centralized mode maps.
- `src/model.py`: Hugging Face model loading, message construction, and generation.
- `src/parser.py`: validation and normalization into shared action models.
- `src/simulator.py`: collision checks, movement updates, and success detection.
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

Run the step-sequence experiment:

```bash
python scripts/run_step_sequence.py
```

Run the full-path experiment:

```bash
python scripts/run_full_path.py
```

The SLURM entry point in `run.slurm` activates the expected Conda environment, sets `PYTHONPATH=$PWD`, and launches the step-sequence script.

## Outputs

- Per-config raw runs are written under `results/<run_id>/<model_id>/...`.
- Experiment-level summaries are written per model and image setting.
- Each saved payload includes both structure-adherence metrics and task-completion metrics.

## Notes

- The environment is a simple 2D wall-crossing task with optional image context.
- Step-sequence runs allow corrective feedback after malformed outputs.
- Full-path runs test whether a model can produce an entire valid action plan in one response.
