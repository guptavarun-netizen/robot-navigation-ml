"""Offline dropout replay on physically recorded TurtleBot3 LiDAR/control data.

The experiment does not create a hardware failure. It masks channels in a
recorded real-robot log and measures command-class prediction on a chronological
tail holdout. The bag contains one recording, so this is not run-wise transfer.
"""
from bisect import bisect_right
from pathlib import Path
import math

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from train_models import create_models

ROOT = Path(__file__).resolve().parent
BAG = ROOT / "data/raw/turtlebot3_alternative/TB3_WAFFLE_SLAM.bag"
OUT = ROOT / "results/turtlebot_recorded_dropout.csv"
OUT_WEB = ROOT / "website/assets/results/turtlebot_recorded_dropout.csv"
SEED = 42
SECTORS = 24


def extract_recorded_examples():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    scans, commands = [], []
    with Reader(BAG) as reader:
        connections = [c for c in reader.connections if c.topic in {"/scan", "/cmd_vel_rc100"}]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            stamp = timestamp / 1e9
            if connection.topic == "/scan":
                values = np.asarray(msg.ranges, dtype=float)
                values[~np.isfinite(values)] = float(msg.range_max)
                values[(values < msg.range_min) | (values > msg.range_max)] = float(msg.range_max)
                # 360 scan beams become 24 fixed 15-degree sector minima.
                groups = np.array_split(values, SECTORS)
                features = [float(np.min(group)) if len(group) else float(msg.range_max) for group in groups]
                scans.append((stamp, features))
            else:
                commands.append((stamp, float(msg.linear.x), float(msg.angular.z)))
    commands.sort()
    command_times = [row[0] for row in commands]
    rows = []
    for stamp, features in scans:
        index = bisect_right(command_times, stamp) - 1
        if index < 0 or stamp - commands[index][0] > 0.5:
            continue
        _, linear, angular = commands[index]
        if abs(linear) < 0.01:
            action = "STOP"
        elif angular > 0.10:
            action = "TURN_LEFT"
        elif angular < -0.10:
            action = "TURN_RIGHT"
        else:
            action = "FORWARD"
        rows.append({"timestamp_s": stamp, **{f"sector_{i:02d}": value for i, value in enumerate(features)},
                     "linear_command_m_s": linear, "angular_command_rad_s": angular, "action": action})
    return pd.DataFrame(rows)


def main():
    data = extract_recorded_examples()
    feature_names = [f"sector_{i:02d}" for i in range(SECTORS)]
    cut = int(len(data) * 0.8)
    train, test = data.iloc[:cut], data.iloc[cut:]
    X_train, X_test = train[feature_names], test[feature_names]
    y_train, y_test = train.action, test.action
    rng = np.random.default_rng(SEED)
    results = []
    for name, definition in create_models().items():
        model = clone(definition).fit(X_train, y_train)
        for rate in [0.0, 0.25, 0.50, 0.75]:
            scores = []
            repeats = 10 if rate else 1
            for repeat in range(repeats):
                values = X_test.to_numpy(copy=True)
                if rate:
                    mask = rng.random(values.shape) < rate
                    values[mask] = np.nan
                prediction = model.predict(pd.DataFrame(values, columns=feature_names))
                scores.append((f1_score(y_test, prediction, average="macro", zero_division=0),
                               balanced_accuracy_score(y_test, prediction), accuracy_score(y_test, prediction)))
            results.append({"model": name, "dropout_fraction": rate, "condition": f"{rate:.0%}",
                            "repeats": repeats, "macro_f1_mean": float(np.mean([s[0] for s in scores])),
                            "macro_f1_std": float(np.std([s[0] for s in scores], ddof=1)) if repeats > 1 else 0.0,
                            "balanced_accuracy_mean": float(np.mean([s[1] for s in scores])),
                            "accuracy_mean": float(np.mean([s[2] for s in scores])),
                            "recording_rows": len(data), "train_rows": len(train), "test_rows": len(test),
                            "test_start_time_s": float(test.timestamp_s.iloc[0]), "test_end_time_s": float(test.timestamp_s.iloc[-1])})
    output = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_WEB.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    output.to_csv(OUT_WEB, index=False)
    print(f"Recorded scans aligned to active commands: {len(data)}; train={len(train)}; chronological test={len(test)}")
    print("Action distribution:", data.action.value_counts().to_dict())
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
