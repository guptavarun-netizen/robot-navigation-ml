"""Evaluate goal-conditioned ML actions on held-out BARN environments.

This is a simulated-map generalization experiment. BARN reference paths are
used as next-action labels; range readings are ray casts from its inflated
occupancy maps. It is deliberately reported separately from UCI and physical
TurtleBot sensor results.
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from train_models import create_models

ROOT = Path(__file__).resolve().parent
BARN = ROOT / "data/raw/BARN_dataset"
OUT = ROOT / "results/barn_environment_generalization.csv"
WEB_OUT = ROOT / "website/assets/results/barn_environment_generalization.csv"
ROWS, COLS, MAX_RAY = 30, 30, 15
RAYS = [(math.cos(2 * math.pi * i / 24), math.sin(2 * math.pi * i / 24)) for i in range(24)]
MOVES = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
MOVE_TO_LABEL = {move: str(i) for i, move in enumerate(MOVES)}
FEATURES = ["goal_row", "goal_col", "position_row", "position_col"] + [f"range_{i}" for i in range(24)]


def ray_ranges(occupancy, row, col):
    readings = []
    for dr, dc in RAYS:
        distance = MAX_RAY
        for step in np.arange(0.5, MAX_RAY + 0.5, 0.5):
            rr, cc = int(round(row + dr * step)), int(round(col + dc * step))
            if rr < 0 or rr >= ROWS or cc < 0 or cc >= COLS or occupancy[rr, cc] != 0:
                distance = float(step)
                break
        readings.append(distance / MAX_RAY)
    return readings


def state_features(occupancy, row, col, goal):
    return [
        (goal[0] - row) / ROWS, (goal[1] - col) / COLS,
        row / ROWS, col / COLS,
        *ray_ranges(occupancy, row, col),
    ]


def collect_maps():
    records, starts_goals = [], {}
    for map_id in range(300):
        occupancy = np.load(BARN / f"cspace_files/cspace_{map_id}.npy")
        path = np.load(BARN / f"path_files/path_{map_id}.npy").astype(int)
        # Retain path order while removing repeated consecutive points.
        path = np.asarray([p for i, p in enumerate(path) if i == 0 or not np.array_equal(p, path[i - 1])])
        if len(path) < 2:
            continue
        goal = tuple(path[-1])
        starts_goals[map_id] = (tuple(path[0]), goal)
        for current, following in zip(path[:-1], path[1:]):
            move = tuple((following - current).tolist())
            if move not in MOVE_TO_LABEL:
                continue
            row, col = map(int, current)
            records.append((map_id, state_features(occupancy, row, col, goal), MOVE_TO_LABEL[move]))
    return records, starts_goals


def rollout(model, occupancy, start, goal, max_steps=180):
    position = tuple(start)
    visited = {position}
    length = 0
    for step in range(1, max_steps + 1):
        if position == goal:
            return True, step - 1, length, 0
        features = pd.DataFrame([state_features(occupancy, *position, goal)], columns=FEATURES)
        label = str(model.predict(features)[0])
        dr, dc = MOVES[int(label)]
        candidate = (position[0] + dr, position[1] + dc)
        if not (0 <= candidate[0] < ROWS and 0 <= candidate[1] < COLS) or occupancy[candidate] != 0:
            return False, step, length, 1
        position = candidate
        length += math.sqrt(2) if dr and dc else 1
        if position in visited:
            return False, step, length, 0
        visited.add(position)
    return position == goal, max_steps, length, 0


def main():
    records, starts_goals = collect_maps()
    rng = np.random.default_rng(42)
    map_ids = np.array(sorted(starts_goals))
    rng.shuffle(map_ids)
    split = int(len(map_ids) * 0.8)
    train_ids, test_ids = set(map_ids[:split]), set(map_ids[split:])
    train = [row for row in records if row[0] in train_ids]
    test = [row for row in records if row[0] in test_ids]
    X_train = pd.DataFrame([row[1] for row in train], columns=FEATURES)
    y_train = pd.Series([row[2] for row in train])
    X_test = pd.DataFrame([row[1] for row in test], columns=FEATURES)
    y_test = pd.Series([row[2] for row in test])
    heldout_states = pd.Series([row[0] for row in test])
    definitions = create_models()
    rows = []
    for name, definition in definitions.items():
        estimator = clone(definition).fit(X_train, y_train)
        prediction = estimator.predict(X_test)
        per_map = []
        for map_id in sorted(test_ids):
            indices = heldout_states[heldout_states == map_id].index.to_numpy()
            if len(indices):
                per_map.append(f1_score(y_test.iloc[indices], prediction[indices], average="macro", zero_division=0))
        scores, collisions, moves = [], 0, 0
        for map_id in sorted(test_ids):
            occupancy = np.load(BARN / f"cspace_files/cspace_{map_id}.npy")
            start, goal = starts_goals[map_id]
            reached, decisions, _, contact = rollout(estimator, occupancy, start, goal)
            scores.append(int(reached)); collisions += contact; moves += decisions
        rows.append({
            "model": name,
            "train_maps": len(train_ids),
            "heldout_maps": len(test_ids),
            "train_path_states": len(train),
            "heldout_path_states": len(test),
            "heldout_action_accuracy": accuracy_score(y_test, prediction),
            "heldout_action_macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
            "mean_per_map_macro_f1": float(np.mean(per_map)),
            "route_successes": int(sum(scores)),
            "route_success_rate": float(np.mean(scores)),
            "collision_failures": collisions,
            "mean_rollout_decisions": float(moves / len(test_ids)),
        })
    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    WEB_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    result.to_csv(WEB_OUT, index=False)
    print(f"BARN maps: {len(map_ids)} total; {len(train_ids)} train; {len(test_ids)} held out")
    print(f"Path states: {len(train)} train; {len(test)} held out")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
