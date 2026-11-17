# Robot Navigation ML

Classical machine learning experiments for robot wall following, a 2D goal
navigation simulator, and analysis tools for TurtleBot3 sensor recordings.

> **Research prototype:** The classifier is trained on the UCI Wall-Following
> Robot Navigation dataset. The pygame simulator converts its own ray casts
> into approximate sensor values and uses a hand-written class-to-action map,
> plus collision avoidance. This is useful for demonstration and software
> experimentation; it is not a validated controller for a physical robot.

## What is included

- `train_models.py` loads the local UCI dataset, compares baseline and
  classical classifiers, evaluates on a stratified holdout set, and writes
  model and evaluation artifacts.
- `simulator.py` runs a pygame 2D environment using `models/best_model.joblib`
  and logs goal trials to `results/simulation_log.csv`.
- `run_navigation_trials.py` runs 100 repeatable scenarios with unique,
  collision-free start/goal pairs (the original 10 goals plus 90 additional
  goals), using the same trained model, controller, map, 60 Hz time step, and
  60-second timeout. They are software scenarios, not independent physical
  robot runs. It writes summary and
  per-step trajectory CSVs to
  `results/navigation_trials.csv` and `results/navigation_trial_steps.csv`,
  and refreshes the website's bundled copies.
- `scripts/` contains TurtleBot3 ROS bag extraction, data cleaning, plotting,
  trajectory checks, speed validation, and sensor alignment analysis.
- `record_robot_trial.py`, `extract_robot_goal_runs.py`, and
  `evaluate_real_goal_runs.py` capture separately identified physical TurtleBot3
  runs, attach operator-entered goals and sensor-condition notes, align scans,
  odometry, and commands, then evaluate with run-grouped folds. They require a
  ROS 1 robot environment and new data collection; no new runs are included.
- `docs/real_robot_collection_protocol.md` describes supervised collection,
  safety constraints, and the public SCAND dataset; it distinguishes actual
  hardware trials from software fault simulation.
- `barn_game/` contains a separate Panda3D manual navigation demo.
- `website/` is a self-contained research website built from the project
  outputs, with interactive model, noise, and simulation result explorers,
  plus a 2D model-controlled navigation comparison. All five goal-conditioned
  fitted policies run together from one start and goal; their paths and
  descriptive outcome statistics are shown side by side. There is no route
  search, and local collision and recent-position checks skip blocked or
  oscillating choices in favor of each model's next-ranked choice.
- `data/` contains the UCI wall-following data used for training. The larger
  BARN maps and TurtleBot3 ROS bag are kept out of Git; see their local data
  READMEs and the scripts' source links for download/access instructions. The
  processed TurtleBot3 summaries used by the site are included under
  `website/assets/`.

## Publish the research website

GitHub Pages deploys the static contents of `website/` through the workflow in
`.github/workflows/pages.yml`. After pushing to `main`, enable **Settings → Pages
→ Build and deployment → GitHub Actions** if Pages is not enabled for the
repository. The published site is <https://guptavarun-netizen.github.io/robot-navigation-ml/>.

## AI-use disclosure and citations

AI assistance was used to organize parts of the project, implement and revise
the website, and summarize repository-generated results. Dataset cleaning,
model fitting, evaluation metrics, and simulator outcomes are produced by the
listed project scripts. Interpretations should be checked against the saved
tables, figures, source code, and cited dataset documentation.

External datasets and technical references:

- UCI Wall-Following Robot Navigation Data:
  <https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data>
- BARN benchmark paper and dataset page:
  <https://www.cs.utexas.edu/~xiao/Research/BARN/BARN.html>
- BARN dataset repository:
  <https://github.com/miquelramirez/BARN_dataset>
- TurtleBot3 recorded dataset repository:
  <https://github.com/guilhermelawless/turtlebot3_datasets>
- SCAND real-robot navigation dataset:
  <https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html>
- NavWareSet multi-run navigation dataset:
  <https://anr-navware.github.io/navwareset/>
- Three.js documentation for the website's animated background:
  <https://threejs.org/docs/>

## Requirements and setup

Python 3.10 or newer is recommended. Create and activate a virtual environment,
then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The TurtleBot3 bag extractor additionally needs `rosbags`; the separate 3D
game needs `panda3d`. Install those only if using those features:

```bash
python -m pip install rosbags    # extract a ROS 1 bag without a ROS install
python -m pip install panda3d    # run barn_game/main.py
```

## Train and evaluate

From the repository root:

```bash
python train_models.py
```

The checked-in ARFF file is used directly, so training does not need to fetch
data from the internet. If it is missing, the loader falls back to
`data/raw/uci_wall_following_24.csv`. Outputs are written under `models/` and
`results/`, including the selected estimator, model comparison, confusion
matrix, class distribution, and a Gaussian sensor-noise robustness curve.

The model comparison evaluates a Dummy baseline, Logistic Regression, RBF SVM,
K-Nearest Neighbors, Random Forest, and Gradient Boosting. It selects the
estimator with the best mean five-fold training-set macro-F1. The holdout set
is used for final metrics and plots.
The sensor robustness suite compares all six models on the same held-out rows
under Gaussian noise, random per-reading dropout, and reduced channel counts
(24, 18, 12, and 6 available sensors). Each condition uses 10 deterministic
perturbation repeats; missing inputs are handled by each pipeline's median
imputer, fitted only on training rows. Summary means and standard deviations
are written to `results/sensor_robustness.csv`; repeat-level scores are in
`results/sensor_robustness_repeats.csv`. These simulated corruptions are
sensitivity tests, not physical sensor-failure experiments.

Training also produces `results/ordered_holdout_comparison.csv`: models are
refit on the first 80% of source rows and scored on the final contiguous 20%.
The UCI file has no run/trajectory identifiers, so this ordered-block check is
not a verified run-wise split and cannot establish new-run generalization.

Two separate evaluations add evidence without changing the UCI task:

```bash
python train_barn_environment_eval.py
python evaluate_turtlebot_recorded_dropout.py
```

The BARN evaluation derives next-move labels from reference paths, trains on
240 map IDs, and evaluates action metrics and closed-loop goal completion on
60 held-out maps. It is a simulated environment-transfer test, not physical
run-wise validation. The source benchmark contains 300 simulated navigation
environments and map/path assets; see the [BARN benchmark page](https://www.cs.utexas.edu/~xiao/Research/BARN/BARN.html).

The TurtleBot3 evaluation derives four coarse action labels from recorded
velocity commands, aggregates each real LiDAR scan into 24 sector readings,
and holds out the final chronological 20% of the single available bag. It
reports all six models under 0%, 25%, 50%, and 75% randomly masked channels;
nonzero dropout levels use 10 fixed-seed repeats. This is a recorded-data
corruption replay, not an injected physical sensor failure or independent-run
test. Both scripts write results to `results/` and copy website-ready CSVs to
`website/assets/results/`.

To collect new physical goal demonstrations, source a ROS 1 environment with
`rosbag` running and use the project recorder. Each invocation creates a
separate bag and JSON metadata containing its unique run ID and operator-set
goal. The workflow is described in `docs/real_robot_collection_protocol.md`:

```bash
python record_robot_trial.py --goal-x 1.5 --goal-y 0.8 --operator student1
python extract_robot_goal_runs.py
python evaluate_real_goal_runs.py
```

The evaluator requires at least two distinct run IDs and groups complete runs
into train/test partitions. It predicts coarse velocity-command labels from
LiDAR sectors and goal-relative features. The current workspace has no new
bags from this workflow, so it cannot produce physical run-held-out scores
until new runs are actually recorded. A run's sensor-fault label is manually
entered metadata, not a fault verified by code. Hardware fault tests must
follow the supervised collection protocol; the recorder never injects faults.

To train the five goal-conditioned movement policies used by the interactive
simulation, run:

```bash
python train_navigation_policies.py
```

This creates synthetic goal-and-sensor examples on 56 randomized layouts with
up to 35 added blocks, plus the dense screenshot layout; it does not train on
the UCI wall-following data. Each policy receives 24 simulated range readings,
the goal offset, and the robot's normalized position. Offline shortest-path
distance maps provide the training move labels. The browser loads the fitted
classifiers and performs no route search. Validation scores and a replay of the dense reference goal are saved
in `results/navigation_policy_comparison.csv`; the generated model bundle is
`website/assets/models/goal_navigation_models.json`. The Dummy constant
baseline remains in the CSV comparison but is not selectable as a route policy
because its action does not depend on the goal.

## View the research website

Serve the static website from the repository root:

```bash
python -m http.server 8000
```

Open <http://localhost:8000/website/>. See `website/README.md` for how to refresh its
bundled data and figures after regenerating results.

## Run the 2D simulator

Train first if `models/best_model.joblib` is absent, then run:

```bash
python simulator.py
```

Click free space to set a goal. Use **Space** to pause/resume, **S** to toggle
sensor rays, **R** to reset, and **Esc** to quit. Trials are logged to
`results/simulation_log.csv`; starting a new simulator session replaces that
CSV. Copy it elsewhere first if you need to retain an earlier session.

For a controlled, reproducible comparison instead of manually operated goals,
run:

```bash
python run_navigation_trials.py
```

The batch uses the fixed goals in that script and starts every run at `(115,
115)` with heading `0`. Trial order is deterministic; per-sample inference
latency is measured on the current machine and is not a hardware benchmark.
The trajectory CSV records every simulation step, while the summary CSV
records terminal outcomes, path length/directness, and collision events. The
batch is still only a software simulation: its synthetic sensors and
class-to-action mapping are provisional, and zero simulated contacts do not
establish physical safety.

## TurtleBot3 analysis

The extraction script expects the bag at
`data/raw/turtlebot3_alternative/TB3_WAFFLE_SLAM.bag` and writes processed CSVs
to `data/processed/turtlebot3/`:

```bash
python scripts/extract_turtlebot_data.py
python scripts/analyze_turtlebot_data.py
python scripts/plot_turtlebot_data.py
python scripts/check_trajectory.py
python scripts/clean_turtlebot_data.py
python scripts/validate_turtlebot_speed.py
python scripts/scripts/compare_turtlebot_sensors.py
python scripts/scripts/validate_alignment.py
```

These are standalone analyses and should be run in the listed order when
starting from the raw bag. If the processed files already exist, skip
extraction. The cleaning thresholds and timestamp matching tolerance are
research choices documented in their respective scripts and may need adjusting
for another recording.

## BARN and 3D demo

`data/raw/BARN_dataset/` is kept separate from UCI classifier training. The
map-held-out evaluation is reproduced with `python train_barn_environment_eval.py`.
The Panda3D demo is launched with:

```bash
python barn_game/main.py
```

Move using WASD or the arrow keys, press **L** to toggle LiDAR rays, and **Esc**
to quit.

## Data and artifact notes

- Keep raw inputs under `data/raw/` and derived analysis products under
  `data/processed/` or `results/`.
- The serialized sklearn model is a Python pickle-based artifact. Load only
  model files you trust, and regenerate it with the current environment when
  changing scikit-learn versions.
- TurtleBot3 extraction requires `rosbags` and a compatible bag file. The BARN
  assets and bag files may have separate upstream licenses; see the included
  dataset license/readme files before redistribution.
