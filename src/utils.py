import json
from typing import Any
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_llm_raw_output(raw: str) -> str:
    """
    Make raw LLM output easier to read in saved logs.

    Behavior:
    1) Strip surrounding whitespace and markdown code fences.
    2) If the model returned JSON directly, pretty-print it.
    3) If the model returned a JSON-encoded string containing JSON, unwrap it.
    4) If parsing still fails, do a best-effort cleanup of escaped characters
       like \\n and \\" without pretending broken JSON is valid.
    """
    if raw is None:
        return ""

    text = str(raw).strip()
    if not text:
        return ""

    # Remove markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    def _try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    # Unwrap up to a few times in case the model returned a quoted JSON string.
    for _ in range(3):
        parsed = _try_parse(text)

        if parsed is None:
            break

        if isinstance(parsed, str):
            text = parsed.strip()
            continue

        return json.dumps(parsed, indent=2, ensure_ascii=False)

    # Try extracting a JSON object/array from surrounding extra text.
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)

        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            parsed = _try_parse(candidate)

            if parsed is not None:
                return json.dumps(parsed, indent=2, ensure_ascii=False)

    # Best-effort readability cleanup for invalid/truncated JSON.
    text = (
        text
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "    ")
        .replace('\\"', '"')
        .replace("\\/", "/")
    )

    cleaned_lines = []
    previous_blank = False

    for line in text.splitlines():
        cleaned = line.rstrip()

        if not cleaned.strip():
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        previous_blank = False
        cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines).strip()

def format_run_timestamp(when: datetime | None = None) -> str:
    """ Run folder name: YYYY-MM-DD_HH-MM-SS """
    dt = when or datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

def save_results(
    exp_config: dict[str, Any], 
    results: dict[str, Any], 
) -> Path:
    run_id = exp_config["run_id"]
    RUN_DIR = RESULTS_DIR / run_id
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = RUN_DIR / exp_config["model_id"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    uses_image = exp_config["uses_image"]
    filename = "with_image" if uses_image else "without_image"

    out_path = MODEL_DIR / f"{filename}.json"
    out_path.write_text(json.dumps(results, indent=2))

def save_run_config(
    run_id: str,
    model_id: str,
    uses_image: bool,
    prompt_id: str,
    schema_id: str,
    runs: list[dict[str, Any]],
) -> Path:
    root = RESULTS_DIR / run_id
    root.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = root / model_id
    image_folder = "with_image" if uses_image else "without_image"
    out_dir = MODEL_DIR / image_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{prompt_id}_{schema_id}.json"
    payload = {
        "userprompt_id": prompt_id,
        "schema_id": schema_id,
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
