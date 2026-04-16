# Tasks

Benchmark tasks are configured in `tasks.json`. Each task should set `task_type` (and `world` with `obstacles` and `targets`) so the simulator in `src/simulator.py` can evaluate success.

Targets and obstacles are axis-aligned rectangles: `x1`, `y1`, `x2`, `y2`.

---

## Task catalog

### 1. `go_to_target`

**Spec:** Spot needs to have a final position overlap the target bounds. No obstacle.

**Simulator:** `task_type` is `go_to_target`. Success uses `_check_go_to_target`: exactly **one** target in `world.targets`; the robot’s \((x, y)\) must lie inside that rectangle. Use an empty `obstacles` list in JSON for a no-obstacle scenario. Collisions and observations still come from whatever obstacles you define.

---

### 2. `go_around_obstacle`

**Spec:** Spot needs to have a final position overlap the target bounds. Obstacle in environment.

**Simulator:** `task_type` is `go_around_obstacle`. **Same success rule** as `go_to_target` (one target, final pose inside it). The difference is **world layout**: add rectangles under `obstacles` so moves can collide and ray-based obstacle observations apply. Success does not depend on obstacle count—only on ending inside the goal.

---

### 3. `face_target`

**Spec:** Spot must rotate to point in the direction of the target.

**Simulator:** `task_type` is `face_target`. Exactly **one** target. `_check_face_target` compares the robot’s **forward heading** (same convention as `move_spot`) to the **bearing** toward the **center** of the target. If the smallest angular difference is within `DEFAULT_FACE_TOLERANCE_DEG` (10° in `simulator.py`), success is true. If the robot is on top of the target center, success is treated as true (degenerate case).

---

### 4. `move_to_closest_target`

**Spec:** Out of multiple targets, spot must go towards the closest one from his starting position.

**Simulator:** `task_type` is `move_to_closest_target`. At least **two** targets. The “closest” target is the one whose **center** has minimum Euclidean distance to **`INITIAL_SPOT_STATE`** \((0, 0)\) at episode start—not the robot’s pose after replans. Success is true when the final pose lies **inside that chosen target’s bounds** only.

---

### 5. `go_to_multiple_targets`

**Spec:** With multiple targets, spot must visit all of them.

**Simulator:** `task_type` is `go_to_multiple_targets`. At least **two** targets. The simulator keeps a mutable **`visited`** set of target indices. Whenever success or observations are updated, any target whose bounds contain the current \((x, y)\) is marked visited. **`at_goal`** (and full task success) is true when `visited` contains every target index at least once—not necessarily all at the same time. `get_visited()` in `simulator.py` creates and seeds `visited` from the starting pose before actions run.

---

## Configuration notes

- `task_type` must be one of the five strings above; unknown values raise in `_check_task_success`.
- For behavior-tree runs, `task_env` passed into `simulate_bt_plan` should include `task_type`, `obstacles`, and `targets` (see `scripts/run_bt_tasks.py`, which merges `task_type` from the task record with `world`).
