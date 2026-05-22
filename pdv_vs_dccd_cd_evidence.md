# PDV vs DCCD/CD: Evidence Summary

## Scope and data used

This memo compares:

1. **PDV**: `results/pdv_bt_all100`
2. **DCCD (archived run)**: `results/archive/dccd_bt_all_tasks`
3. **CD baseline**: `results/bt_cd_all100_parallel`

Primary model-overlap comparisons were done on common models only.

---

## Executive conclusion

PDV is **not consistently better** than DCCD/CD on task completion.  
It is often faster, but success is mixed and regresses on specific task types, especially `move_to_closest_target`.

---

## 1) Aggregate evidence: PDV vs DCCD (common models)

Common models: `Qwen2.5-14B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen3-14B`, `granite-3.3-8b-instruct`.

Computed comparison:

- Mean task-completion delta (PDV - DCCD): **-1.50 percentage points**
- Mean structure-adherence delta (PDV - DCCD): **-0.905 points**
- Mean latency ratio (DCCD / PDV): **2.69x** (PDV faster)

Per-model task completion deltas (PDV - DCCD):

- `Qwen2.5-14B-Instruct`: **+2.0**
- `Qwen2.5-7B-Instruct`: **-8.0**
- `Qwen3-14B`: **-3.0**
- `granite-3.3-8b-instruct`: **+3.0**

Task-type average deltas (PDV - DCCD, common models):

- `go_to_target_pct`: **+3.8**
- `face_target_pct`: **+1.2**
- `go_around_obstacle_pct`: **+1.2**
- `go_to_multiple_targets_pct`: **-1.2**
- `move_to_closest_target_pct`: **-12.5**  <-- biggest deficit

Interpretation: PDV’s largest weakness versus DCCD is `move_to_closest_target`, enough to erase gains elsewhere.

---

## 2) Aggregate evidence: PDV vs CD baseline (`bt_cd_all100_parallel`)

Common models: `Qwen2.5-14B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen3-14B`, `gemma-2-9b-it`, `granite-3.3-8b-instruct`.

Computed comparison:

- Mean task-completion delta (PDV - CD): **+0.40 points** (essentially flat)
- Mean structure-adherence delta (PDV - CD): **-0.227 points**
- Mean speed ratio (CD / PDV): **~1.00x** (no clear latency advantage overall)

Per-model task completion deltas (PDV - CD):

- `Qwen2.5-14B-Instruct`: **-9.0**
- `Qwen2.5-7B-Instruct`: **+5.0**
- `Qwen3-14B`: **-1.0**
- `gemma-2-9b-it`: **-3.0**
- `granite-3.3-8b-instruct`: **+10.0**

Interpretation: against CD baselines, PDV is mixed rather than dominant.

---

## 3) Response-level evidence (why PDV fails more often)

### A) Repeated non-improving plans in PDV (`move_to_closest_target`)

From `results/pdv_bt_all100/Qwen2.5-7B-Instruct/tasks/task_move_to_closest_target_v8.json`:

- `bt_count = 3`, `task_completion = false`
- Planner repeats near-identical plan across BTs:
  - `rotate_spot(0)` then `move_spot(4.1)` (or `4.123`)
- Observations stay:
  - `"target_ahead": false`
  - `"at_goal": false`
- Final spots progress linearly away from start without success:
  - BT1: `x=4.1`
  - BT2: `x=8.2`
  - BT3: `x=12.3`

This is a classic “repeat same failed action template” pattern.

### B) Decoder output can allow runaway forward motion

From `results/pdv_bt_all100/Qwen2.5-7B-Instruct/tasks/task_move_to_closest_target_v17.json`:

- Planner repeatedly gives:
  - `move_spot(4.5)`
- Behavior tree branch includes `obstacle_ahead == false` then move.
- Execution results:
  - BT1 final spot: `x=288.0`
  - BT2 final spot: `x=576.0`
  - BT3 final spot: `x=864.0`
- Task fails with massive overshoot:
  - `"task_completion": false`
  - final spot `{"x": 864.0, "y": 0.0, "heading": 0.0}`

This is strong evidence that some decoded trees permit repeated movement ticks before replanning/termination.

### C) DCCD sometimes recovers with heading change + short step

From `results/archive/dccd_bt_all_tasks/Qwen2.5-7B-Instruct/tasks/task_move_to_closest_target_v1.json`:

- `bt_count = 2`, `task_completion = true`
- First BT fails, but second BT changes action mix:
  - `move_spot(1.0)` + `rotate_spot(90.0)` + `move_spot(0.5)`
- Second BT result:
  - observations include `"at_goal": true`
  - success true, final spot near target (`x=2.5`, `y=-0.5`)

This indicates better short-step correction behavior after failure in this case.

---

## 4) Quantified pattern differences on `Qwen2.5-7B-Instruct`

For `move_to_closest_target` tasks only:

- Success rate:
  - **PDV: 1/20**
  - **DCCD: 6/20**
- Avg BTs per task:
  - PDV: **2.9**
  - DCCD: **2.45**
- Avg replans per task:
  - PDV: **2.2**
  - DCCD: **1.1**
- Tasks with a single repeated move distance across BT attempts:
  - PDV: **11**
  - DCCD: **5**
- Rotation diversity (`degrees` unique count):
  - PDV: **3** (`-45, 0, 45`)
  - DCCD: **7** (`-180, -90, -45, 0, 45, 90, 180`)
- Tree-level replan-requested rate:
  - PDV: **75.9%** (44/58 trees)
  - DCCD: **45.8%** (22/48 trees)
- Tree-level success rate:
  - PDV: **1.7%** (1/58)
  - DCCD: **12.5%** (6/48)

For `face_target` tasks (`Qwen2.5-7B-Instruct`):

- Success:
  - PDV: **7/20**
  - DCCD: **10/20**
- `move_spot` action frequency:
  - PDV: **29**
  - DCCD: **6**

This suggests PDV over-moves on a rotation-dominant task.

---

## Root-cause hypothesis supported by evidence

1. **PDV planner often repeats similar rotate+move pairs after failure** instead of changing strategy.
2. **PDV decoded trees can allow repeated motion execution before forcing clean termination/replan**, causing overshoot in some tasks.
3. **`move_to_closest_target` is the main regression axis** (largest negative task-type delta).
4. **For `face_target`, PDV includes unnecessary movement more often**, likely reducing completion reliability.

---

## Suggested prompt-level fixes (minimal, high-impact)

1. Explicitly ban repeating the same failed rotate+move pair.
2. For `move_to_closest_target`, enforce smaller bounded moves and heading change before move when `target_ahead=false`.
3. For `face_target`, default to rotation-only plans.
4. Decoder rule: after `move_spot`, require `at_goal` check or immediate reachable `call_llm` (avoid move-only terminal sequences).

These fixes target the exact failure signatures shown in the task outputs above.

