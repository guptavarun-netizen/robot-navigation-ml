"""Run 100 deterministic, fixed-goal trials of the pygame navigation controller.

Run with: .venv/bin/python run_navigation_trials.py
Outputs: results/navigation_trials.csv and results/navigation_trial_steps.csv
"""
import csv
import math
import random
import shutil
from pathlib import Path

from simulator import (
    ACTION_MAP, FPS, GOAL_RADIUS, MAX_TRIAL_TIME_S, MODEL_PATH, RESULTS_DIR,
    ROBOT_SPEED, TURN_SPEED, WALLS, WIDTH, HEIGHT, choose_safe_heading,
    collides, dataframe_for_model, label_key, load_model, sense_robot,
)

ROOT = Path(__file__).resolve().parent
WEBSITE_RESULTS = ROOT / "website" / "assets" / "results"
START = (115.0, 115.0)
# Preserve the original ten goals first, then add a reproducible, spatially
# spread set of collision-free targets across the same simulator field.
ORIGINAL_GOALS = [(950, 620), (150, 620), (600, 150), (700, 600), (400, 600),
                  (950, 500), (150, 500), (700, 200), (400, 500), (950, 250)]
TRIAL_COUNT = 100
GOAL_SEED = 20261117
SUMMARY_FIELDS = ["trial_id", "goal_x_px", "goal_y_px", "status", "goal_reached",
                  "steps", "elapsed_s", "path_length_px", "straight_line_px",
                  "path_efficiency", "collision_count", "mean_inference_ms"]
STEP_FIELDS = ["trial_id", "time_s", "x_px", "y_px", "heading_rad", "goal_x_px",
               "goal_y_px", "predicted_class", "inference_ms", "collision_count",
               "path_length_px", "goal_reached", "trial_status"]


def build_goal_batch():
    goals = list(ORIGINAL_GOALS)
    candidates = [
        (x, y)
        for y in range(85, HEIGHT - 84, 20)
        for x in range(85, WIDTH - 84, 20)
        if not collides((x, y), WALLS) and math.dist(START, (x, y)) >= 250
    ]
    random.Random(GOAL_SEED).shuffle(candidates)
    for candidate in candidates:
        if all(math.dist(candidate, chosen) >= 45 for chosen in goals):
            goals.append(candidate)
            if len(goals) == TRIAL_COUNT:
                break
    if len(goals) != TRIAL_COUNT:
        raise RuntimeError(f"Could only create {len(goals)} of {TRIAL_COUNT} unique free-space goals")
    return goals


def run_trial(model, trial_id, goal):
    position = list(START)
    heading = 0.0
    collisions = 0
    path_length = 0.0
    inference_ms = []
    rows = []
    dt = 1.0 / FPS
    status = "TIMEOUT"
    steps = 0
    for steps in range(1, int(MAX_TRIAL_TIME_S * FPS) + 1):
        features, _ = sense_robot(position, heading, WALLS)
        sensor_df = dataframe_for_model(features, model)
        import time
        started = time.perf_counter()
        prediction = model.predict(sensor_df)[0]
        latency = (time.perf_counter() - started) * 1000
        inference_ms.append(latency)
        label = label_key(prediction)
        speed_mult, ml_turn = ACTION_MAP.get(label, (0.0, 0))
        goal_heading = math.atan2(goal[1] - position[1], goal[0] - position[0])
        goal_error = math.atan2(math.sin(goal_heading - heading), math.cos(goal_heading - heading))
        desired = heading + 0.75 * goal_error + 0.25 * (ml_turn * TURN_SPEED * dt)
        speed = ROBOT_SPEED * max(speed_mult, 0.35)
        safe_heading, can_move = choose_safe_heading(position, desired, speed, dt, WALLS)
        if can_move:
            new_position = [position[0] + math.cos(safe_heading) * speed * dt,
                            position[1] + math.sin(safe_heading) * speed * dt]
            if collides(new_position, WALLS):
                collisions += 1
            else:
                path_length += math.dist(position, new_position)
                position = new_position
                heading = safe_heading
        else:
            heading += math.radians(30) * dt
        reached = math.dist(position, goal) <= GOAL_RADIUS
        if reached:
            status = "SUCCESS"
        rows.append({"trial_id": trial_id, "time_s": f"{steps * dt:.3f}",
                     "x_px": f"{position[0]:.2f}", "y_px": f"{position[1]:.2f}",
                     "heading_rad": f"{heading:.4f}", "goal_x_px": goal[0],
                     "goal_y_px": goal[1], "predicted_class": label,
                     "inference_ms": f"{latency:.4f}", "collision_count": collisions,
                     "path_length_px": f"{path_length:.2f}", "goal_reached": int(reached),
                     "trial_status": status if reached or steps == int(MAX_TRIAL_TIME_S * FPS) else "RUNNING"})
        if reached:
            break
    straight = math.dist(START, goal)
    summary = {"trial_id": trial_id, "goal_x_px": goal[0], "goal_y_px": goal[1],
               "status": status, "goal_reached": int(status == "SUCCESS"),
               "steps": steps, "elapsed_s": f"{steps / FPS:.3f}",
               "path_length_px": f"{path_length:.2f}", "straight_line_px": f"{straight:.2f}",
               # Measure progress to the actual terminal position. Because a
               # goal has a 20 px acceptance radius, using the goal center can
               # make a nominal path-efficiency ratio exceed one.
               "path_efficiency": f"{math.dist(START, position) / path_length:.4f}" if path_length else "0",
               "collision_count": collisions,
               "mean_inference_ms": f"{sum(inference_ms) / len(inference_ms):.4f}"}
    return summary, rows


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train models first: .venv/bin/python train_models.py")
    model = load_model()
    goals = build_goal_batch()
    all_summaries, all_steps = [], []
    for trial_id, goal in enumerate(goals, 1):
        if collides(START, WALLS) or collides(goal, WALLS):
            raise ValueError(f"Invalid start or goal in trial {trial_id}: {goal}")
        summary, steps = run_trial(model, trial_id, goal)
        all_summaries.append(summary)
        all_steps.extend(steps)
        print(f"Trial {trial_id:03d}/{len(goals)} {summary['status']}: {summary['steps']} steps, "
              f"{summary['path_length_px']} px, {summary['collision_count']} collisions")
    RESULTS_DIR.mkdir(exist_ok=True)
    WEBSITE_RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = [("navigation_trials.csv", SUMMARY_FIELDS, all_summaries),
               ("navigation_trial_steps.csv", STEP_FIELDS, all_steps)]
    for name, fields, rows in outputs:
        path = RESULTS_DIR / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        shutil.copy2(path, WEBSITE_RESULTS / name)
    print(f"Completed {len(all_summaries)} trials using goal seed {GOAL_SEED}; results saved under results/ and website/assets/results/")


if __name__ == "__main__":
    main()
