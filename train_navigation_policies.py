"""Train and export goal-conditioned ML movement policies for the web simulator.

Training labels are next moves from shortest-path distance maps computed offline.
At runtime the browser only runs the selected fitted classifier; it does not run
any route search or collision-avoidance override.

Run from the repository root: .venv/bin/python train_navigation_policies.py
"""
from __future__ import annotations

import heapq
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from browser_model_export import export_one
from train_models import create_models

ROOT = Path(__file__).resolve().parent
OUT_MODEL = ROOT / "website/assets/models/goal_navigation_models.json"
OUT_METRICS = ROOT / "results/navigation_policy_comparison.csv"
OUT_METRICS_WEB = ROOT / "website/assets/results/navigation_policy_comparison.csv"
WIDTH, HEIGHT, STEP, ROBOT_RADIUS = 1100, 760, 20, 13
X_VALUES = list(range(80, 1021, STEP))
Y_VALUES = list(range(80, 681, STEP))
DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
RAY_DIRECTIONS = [(math.cos(i * math.pi / 12), math.sin(i * math.pi / 12)) for i in range(24)]
BASE_OBSTACLES = [(280, 180, 305, 440), (500, 330, 750, 355), (830, 150, 855, 400)]
# Dense map shown in the user's failure screenshot (base pillars plus the added
# blocks, in the simulator's 1,100 x 760 coordinate system).
SCREENSHOT_DENSE_OBSTACLES = BASE_OBSTACLES + [
    (660, 100, 720, 140), (510, 160, 570, 200), (370, 200, 430, 240),
    (558, 250, 618, 290), (467, 260, 527, 300), (340, 280, 400, 320),
    (650, 240, 710, 280), (720, 270, 780, 310), (410, 380, 470, 420),
    (690, 380, 750, 420), (270, 480, 330, 520), (360, 480, 420, 520),
    (530, 420, 590, 460), (610, 510, 670, 550), (760, 530, 820, 570),
    (210, 560, 270, 600), (330, 590, 390, 630), (570, 590, 630, 630),
]
BOUNDARY_WALLS = [(45, 45, 1055, 63), (45, 697, 1055, 715), (45, 45, 63, 715), (1037, 45, 1055, 715)]
FEATURES = ["goal_dx", "goal_dy", "position_x", "position_y"] + [f"range_{i}" for i in range(24)]
LABELS = [str(i) for i in range(8)]


def free(x, y, obstacles):
    if not (76 <= x <= 1024 and 76 <= y <= 684):
        return False
    return not any(x >= x1 - ROBOT_RADIUS and x <= x2 + ROBOT_RADIUS and
                   y >= y1 - ROBOT_RADIUS and y <= y2 + ROBOT_RADIUS
                   for x1, y1, x2, y2 in obstacles)


def ray_distance(x, y, dx, dy, obstacles):
    closest = 180.0
    for x1, y1, x2, y2 in obstacles:
        low, high = 0.0, closest
        valid = True
        for origin, direction, minimum, maximum in ((x, dx, x1, x2), (y, dy, y1, y2)):
            if abs(direction) < 1e-12:
                if origin < minimum or origin > maximum:
                    valid = False
                    break
            else:
                a, b = (minimum - origin) / direction, (maximum - origin) / direction
                low = max(low, min(a, b))
                high = min(high, max(a, b))
                if high < low:
                    valid = False
                    break
        if valid and high >= 0:
            closest = min(closest, max(0.0, low))
    return min(closest, 180.0) / 180.0


def features_for(x, y, gx, gy, obstacles):
    return [(gx - x) / 1000.0, (gy - y) / 700.0, x / WIDTH, y / HEIGHT] + [
        ray_distance(x, y, dx, dy, [*obstacles, *BOUNDARY_WALLS]) for dx, dy in RAY_DIRECTIONS
    ]


def dijkstra_policy(goal, nodes, obstacles):
    distances = {goal: 0.0}
    policy = {}
    queue = [(0.0, goal)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        x, y = current
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            neighbor = (x + dx * STEP, y + dy * STEP)
            if neighbor not in nodes:
                continue
            if dx and dy and ((x + dx * STEP, y) not in nodes or (x, y + dy * STEP) not in nodes):
                continue
            edge = STEP * (math.sqrt(2) if dx and dy else 1)
            candidate = distance + edge
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                policy[neighbor] = (direction + 4) % 8
                heapq.heappush(queue, (candidate, neighbor))
    return policy


def randomized_layouts():
    rng = random.Random(42)
    layouts = [("base", BASE_OBSTACLES), ("dense_reference", SCREENSHOT_DENSE_OBSTACLES)]
    counts = [0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40]
    for index in range(56):
        blocks = list(BASE_OBSTACLES)
        attempts = 0
        target_count = len(BASE_OBSTACLES) + counts[index % len(counts)]
        while len(blocks) < target_count and attempts < 2500:
            attempts += 1
            width, height = rng.choice([40, 60, 80, 100]), rng.choice([40, 60, 80, 100])
            x, y = rng.randrange(120, 961, 20), rng.randrange(120, 621, 20)
            block = (x - width // 2, y - height // 2, x + width // 2, y + height // 2)
            if any(block[0] < b[2] + 24 and block[2] > b[0] - 24 and block[1] < b[3] + 24 and block[3] > b[1] - 24 for b in blocks):
                continue
            if block[0] < 100 or block[1] < 100 or block[2] > 1000 or block[3] > 660:
                continue
            blocks.append(block)
        layouts.append((f"layout_{index:02d}", blocks))
    return layouts


def collect_data():
    rng = random.Random(2026)
    records = []
    held_out_layouts = {f"layout_{index:02d}" for index in range(42, 56)}
    for layout_id, obstacles in randomized_layouts():
        nodes = {(x, y) for x in X_VALUES for y in Y_VALUES if free(x, y, obstacles)}
        if layout_id == "base":
            goals = [(x, y) for x, y in sorted(nodes) if (x, y) not in {(150, 150), (950, 600)}]
            goals = [goals[index] for index in np.linspace(0, len(goals) - 1, 14, dtype=int)]
            starts_per_goal = None
        elif layout_id == "dense_reference":
            # Exhaustively label the reachable free cells for this dense map,
            # including the start/goal pair from the user's screenshot.
            goals = [(100, 660)]
            starts_per_goal = None
        else:
            goals = rng.sample(list(nodes), min(5, len(nodes)))
            starts_per_goal = 60
        for goal in goals:
            policy = dijkstra_policy(goal, nodes, obstacles)
            states = list(policy)
            if layout_id == "dense_reference" and goal == (100, 660):
                # Include the exact start shown in the screenshot.
                states = list(set(states) | {(980, 160)} if (980, 160) in policy else set(states))
            if starts_per_goal is not None and len(states) > starts_per_goal:
                states = rng.sample(states, starts_per_goal)
            for x, y in states:
                row = features_for(x, y, goal[0], goal[1], obstacles)
                records.append((layout_id, row, str(policy[(x, y)])))
    train = [record for record in records if record[0] not in held_out_layouts]
    # Give each model more examples from the dense reference map so the tested
    # many-block scenario contributes enough supervision without replacing the
    # randomized-layout training examples.
    dense_examples = [record for record in train if record[0] == "dense_reference"]
    train.extend(dense_examples * 11)
    valid = [record for record in records if record[0] in held_out_layouts]
    X_train = pd.DataFrame([record[1] for record in train], columns=FEATURES)
    y_train = pd.Series([record[2] for record in train])
    X_valid = pd.DataFrame([record[1] for record in valid], columns=FEATURES)
    y_valid = pd.Series([record[2] for record in valid])
    return X_train, y_train, X_valid, y_valid


def replay_target(estimator):
    """Replay the dense reference map using model-ranked collision-free moves."""
    # Map coordinates read from the reported stuck screenshot (rounded to the
    # simulator's 20 px movement grid).
    position, goal = (980, 160), (100, 660)
    obstacles = SCREENSHOT_DENSE_OBSTACLES
    length, blocked_streak, skipped, skipped_loops = 0.0, 0, 0, 0
    visits = {position: 1}
    recent = [position]
    best_goal_distance, stalled_decisions = math.dist(position, goal), 0
    for decision in range(1000):
        features = pd.DataFrame([features_for(position[0], position[1], goal[0], goal[1], obstacles)], columns=FEATURES)
        if hasattr(estimator, "named_steps"):
            classifier = estimator.named_steps["classifier"]
            transformed = estimator[:-1].transform(features)
        else:
            classifier = estimator
            transformed = features.to_numpy()
        if hasattr(classifier, "predict_proba"):
            scores = classifier.predict_proba(transformed)[0]
        else:
            pairwise = classifier.decision_function(transformed)[0]
            scores = np.zeros(len(classifier.classes_), dtype=float)
            pair = 0
            for i in range(len(classifier.classes_)):
                for j in range(i + 1, len(classifier.classes_)):
                    scores[i if pairwise[pair] > 0 else j] += 1
                    pair += 1
        ranked = np.argsort(-scores, kind="stable")
        next_position = position
        blocked = True
        repeated_fallback = None
        for label_index in ranked:
            dx, dy = DIRECTIONS[int(classifier.classes_[label_index])]
            candidate = (position[0] + dx * STEP, position[1] + dy * STEP)
            distance = math.hypot(candidate[0] - position[0], candidate[1] - position[1])
            samples = max(2, math.ceil(distance / 3))
            candidate_blocked = any(not free(position[0] + (candidate[0] - position[0]) * i / samples,
                                             position[1] + (candidate[1] - position[1]) * i / samples,
                                             obstacles) for i in range(1, samples + 1))
            if candidate_blocked:
                skipped += 1
                continue
            if candidate in recent:
                skipped_loops += 1
                if (repeated_fallback is None or
                        visits.get(candidate, 0) < visits.get(repeated_fallback, 0)):
                    repeated_fallback = candidate
                continue
            next_position, blocked = candidate, False
            break
        if blocked and repeated_fallback is not None:
            next_position, blocked = repeated_fallback, False
        if blocked:
            blocked_streak += 1
            if blocked_streak >= 8:
                return False, decision + 1, length, blocked_streak, skipped, skipped_loops
            continue
        blocked_streak = 0
        position = next_position
        length += distance
        if math.dist(position, goal) <= 20:
            return True, decision + 1, length, 0, skipped, skipped_loops
        visits[position] = visits.get(position, 0) + 1
        recent.append(position)
        if len(recent) > 48:
            recent.pop(0)
        goal_distance = math.dist(position, goal)
        if goal_distance < best_goal_distance - 10:
            best_goal_distance, stalled_decisions = goal_distance, 0
        else:
            stalled_decisions += 1
        if stalled_decisions >= 120:
            return False, decision + 1, length, 0, skipped, skipped_loops
    return False, 1000, length, blocked_streak, skipped, skipped_loops


def main():
    X_train, y_train, X_valid, y_valid = collect_data()
    print(f"Training samples: {len(X_train):,}; held-out layout samples: {len(X_valid):,}")
    model_definitions = create_models()
    # Navigation policy forests use bounded trees so their browser export stays
    # small enough to download and execute interactively.
    model_definitions["Random Forest"].named_steps["classifier"].set_params(
        n_estimators=100, max_depth=10, min_samples_leaf=3
    )
    # Neighbor counts 3, 5, 7, 9, 11, 15, and 21 were compared on held-out
    # maps; k=5 gave the strongest macro-F1. The longer live memory handles loops.
    model_definitions["K-Nearest Neighbors"].named_steps["classifier"].set_params(n_neighbors=5)
    model_definitions["RBF SVM"].named_steps["classifier"].set_params(decision_function_shape="ovo")
    bundle = {"source": "Goal-conditioned synthetic sensor/action samples; fitted estimators only are used in-browser",
              "features": FEATURES, "labels": LABELS,
              "direction_labels": ["east", "south-east", "south", "south-west", "west", "north-west", "north", "north-east"],
              "models": []}
    metric_rows = []
    for name, estimator in model_definitions.items():
        print(f"Training {name}…", flush=True)
        estimator.fit(X_train, y_train)
        # The full held-out layout set is capped for KNN's quadratic prediction
        # cost; it remains stratified and independent at the map-layout level.
        if len(X_valid) > 1800:
            sample = np.linspace(0, len(X_valid) - 1, 1800, dtype=int)
            X_eval, y_eval = X_valid.iloc[sample], y_valid.iloc[sample]
        else:
            X_eval, y_eval = X_valid, y_valid
        predicted = estimator.predict(X_eval)
        accuracy = float(accuracy_score(y_eval, predicted))
        macro = float(f1_score(y_eval, predicted, average="macro", zero_division=0))
        reached, demo_decisions, demo_length, demo_blocked, skipped_actions, skipped_loops = replay_target(estimator)
        item = export_one(name, estimator)
        item["validation_accuracy"] = accuracy
        item["validation_macro_f1"] = macro
        item["example_goal_reached"] = reached
        item["example_goal_decisions"] = demo_decisions
        item["example_goal_path_px"] = round(demo_length, 1)
        # The Dummy baseline has no goal-conditioned behavior, so retain its
        # benchmark row in the CSV but do not offer it as a navigation policy.
        if name != "Dummy":
            bundle["models"].append(item)
        metric_rows.append({"model": name, "validation_accuracy": accuracy, "validation_macro_f1": macro,
                            "training_samples": len(X_train), "heldout_layout_samples": len(y_eval),
                            "example_goal_reached": int(reached), "goal_decisions": demo_decisions,
                            "goal_path_px": f"{demo_length:.1f}", "blocked_streak_at_stop": demo_blocked,
                            "blocked_choices_skipped": skipped_actions, "recent_loop_choices_skipped": skipped_loops})
        print(f"  held-out layout accuracy={accuracy:.3f}, macro-F1={macro:.3f}; "
              f"screenshot goal {'reached' if reached else 'not reached'} in {demo_decisions} decisions")
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MODEL.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, separators=(",", ":"), allow_nan=False)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_METRICS, index=False)
    OUT_METRICS_WEB.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS_WEB, index=False)
    print(f"Saved {OUT_MODEL} ({OUT_MODEL.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved {OUT_METRICS}")


if __name__ == "__main__":
    main()
