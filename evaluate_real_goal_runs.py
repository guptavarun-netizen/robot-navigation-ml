"""Evaluate goal-conditioned action models with held-out physical run IDs.

Requires goal_conditioned_steps.csv created by extract_robot_goal_runs.py.
No rows from one run can appear in both train and test partitions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from train_models import create_models


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "real_robot_runs" / "goal_conditioned_steps.csv"
OUT = ROOT / "results" / "real_goal_run_validation.csv"
WEB_OUT = ROOT / "website" / "assets" / "results" / "real_goal_run_validation.csv"
FEATURES = ["goal_forward_m", "goal_left_m"] + [f"range_{i:02d}" for i in range(24)]
MAX_FOLDS = 5


def main():
    if not DATA.exists():
        raise SystemExit("No real-run data yet. Record at least two distinct runs, then run extract_robot_goal_runs.py.")
    data = pd.read_csv(DATA)
    groups = data.run_id.astype(str)
    n_runs = groups.nunique()
    if n_runs < 2:
        raise SystemExit(f"Need at least two independent run IDs for held-out-run evaluation; found {n_runs}.")
    n_folds = min(n_runs, MAX_FOLDS)
    splitter = GroupKFold(n_splits=n_folds)
    results = []
    for name, definition in create_models().items():
        scores_f1, scores_acc, test_runs = [], [], []
        for train_idx, test_idx in splitter.split(data[FEATURES], data.action, groups):
            model = clone(definition).fit(data.iloc[train_idx][FEATURES], data.iloc[train_idx].action.astype(str))
            truth = data.iloc[test_idx].action.astype(str)
            pred = model.predict(data.iloc[test_idx][FEATURES]).astype(str)
            scores_f1.append(f1_score(truth, pred, average="macro", zero_division=0))
            scores_acc.append(accuracy_score(truth, pred))
            test_runs.extend(data.iloc[test_idx].run_id.astype(str).unique())
        results.append({
            "model": name, "validation": "grouped held-out physical runs",
            "runs_available": n_runs, "group_folds": n_folds,
            "macro_f1_mean": float(np.mean(scores_f1)), "macro_f1_std": float(np.std(scores_f1, ddof=1)),
            "accuracy_mean": float(np.mean(scores_acc)), "accuracy_std": float(np.std(scores_acc, ddof=1)),
            "distinct_test_runs_seen": len(set(test_runs)), "rows": len(data),
            "goal_labels": "operator goal transformed to robot frame; action from recorded cmd_vel",
        })
    output = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    WEB_OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    output.to_csv(WEB_OUT, index=False)
    print(output[["model", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "distinct_test_runs_seen"]].to_string(index=False))
    print(f"Saved {OUT} and {WEB_OUT}")


if __name__ == "__main__":
    main()
