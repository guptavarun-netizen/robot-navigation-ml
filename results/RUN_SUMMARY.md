# Project run summary

Batch workflows completed with the repository virtual environment.

## Model training

- Dataset: 5,456 rows, 24 sensor features, four classes.
- Selected estimator: Gradient Boosting.
- Compared candidates: Dummy baseline, Logistic Regression, RBF SVM, K-Nearest
  Neighbors (5 neighbors), Random Forest, and Gradient Boosting.
- Holdout accuracy: 0.9963.
- Holdout macro-F1: 0.9944.
- Five-fold training macro-F1: 0.9932 (standard deviation 0.0039).
- Noise sensitivity macro-F1 at 0.00, 0.02, 0.05, 0.10, and 0.20 feature
  standard deviations: 0.9944, 0.9522, 0.8748, 0.7892, and 0.6450.

## Expanded sensor robustness and ordered holdout

All six classifiers were scored on the same 1,092 held-out observations under
three sensor perturbation experiments: Gaussian noise at 0, 0.02, 0.05, 0.10,
and 0.20 feature standard deviations; random per-reading dropout at 0%, 25%,
50%, and 75%; and 24, 18, 12, or 6 available sensor channels. Each corruption
condition was repeated 10 times with a fixed seed sequence. Missing readings
were passed as NaN and imputed by each model's training-fitted median imputer.
Summary mean/standard deviation rows are in `sensor_robustness.csv`; individual
repeats are in `sensor_robustness_repeats.csv`.

At 25% random dropout, mean macro-F1 was 0.831 for Random Forest and 0.819 for
Gradient Boosting, versus clean scores of 0.990 and 0.994. With only 6 of 24
channels available, mean macro-F1 was about 0.406 for Gradient Boosting, 0.405
for KNN, and 0.400 for Random Forest. These are simulated missing-input
sensitivity checks, not measurements of hardware sensor failures.

An additional ordered-block check fit on rows 1–4,364 and evaluated rows
4,365–5,456, preserving source order. Gradient Boosting achieved 0.995 macro-F1
and Random Forest 0.986. This is not run-wise validation: the public file does
not provide run or trajectory IDs, and contiguous rows cannot be assumed to be
independent trajectories. Full scores are in `ordered_holdout_comparison.csv`.

Training outputs: `model_comparison.csv`, `best_confusion_matrix.png`,
`class_distribution.png`, `noise_robustness.csv`, and
`noise_robustness.png` in this folder. The fitted model is in
`../models/best_model.joblib`.

## Additional environment-transfer and recorded-data checks

`train_barn_environment_eval.py` derives action targets from each BARN
reference path, splits by map ID (240 train maps / 60 held-out maps), then
reports both held-out action scores and closed-loop replay outcomes. On the 60
held-out maps, Random Forest reached 39 goals (65.0%) and Gradient Boosting
reached 43 (71.7%); the other learned policies reached 4 (Logistic Regression),
12 (RBF SVM), and 2 (KNN). The Dummy baseline reached none. This indicates
material differences in closed-loop behavior despite reasonable step-level
classification scores. These are simulated maps and reference routes, not
physical robot validation. Full per-model scores are in
`barn_environment_generalization.csv`.

`evaluate_turtlebot_recorded_dropout.py` uses all 1,772 scans that could be
paired with a preceding velocity command in the available TurtleBot3 bag. It
trains on the first 1,417 observations and tests on the chronological final
355. The labels are coarse velocity-command classes, not goal-route labels.
Clean held-out macro-F1 was 0.727 for Random Forest, 0.695 for RBF SVM, 0.691
for Gradient Boosting, 0.636 for KNN, 0.579 for Logistic Regression, and
0.402 for Dummy. At 75% masked sectors, scores decreased to 0.422, 0.500,
0.434, 0.452, 0.481, and 0.402, respectively. Nonzero mask levels use 10
repeats; the clean condition is one unchanged evaluation. This is replay on a
single real sensor recording with simulated channel masking, not a physical
sensor-failure trial or new-run test. Full repeat summaries are in
`turtlebot_recorded_dropout.csv`.

The website now includes both reports and links to their CSVs. The remaining
evidence that cannot be honestly generated from the files available here is a
physical sensor-fault experiment, goal-conditioned route demonstrations from
real robot runs, and each student's personal contribution/defense. The last
item has a worksheet at `../docs/individual_defense_template.md` for the team
to complete.

## TurtleBot3 processing and analysis

The raw ROS bag was extracted successfully:

- LiDAR scans: 1,772 rows.
- Odometry: 10,190 rows.
- Velocity commands: 10,369 rows.
- Cleaned odometry: 8,091 rows.
- Sensor-aligned observations: 1,761 rows; 11 LiDAR scans had no complete
  timestamp match within the configured tolerance.
- Commanded versus recorded speed correlation: 0.9888.
- Mean absolute speed difference: 0.00181 m/s.
- Position-derived versus recorded speed correlation after cleaning: 0.7401.

Processed source datasets were copied to `data/turtlebot3/`. The UCI CSV used
for training was copied to `data/robot_navigation/`. Plots and analysis CSVs
are in `turtlebot3/`.

## Data quality note

The original odometry stream contains many very short timestamp intervals.
Cleaning removes intervals outside 10–100 ms, but 111 retained-to-retained
intervals exceed 100 ms because the filter evaluates each original adjacent
pair before removing rows. The trajectory analysis reports very large
instantaneous speeds for the short intervals. Treat position-derived speed
statistics with care; the separate recorded odometry velocity remains within
the expected range for this recording.

## Interactive applications

Both `simulator.py` and `barn_game/main.py` were launched in desktop windows
after the batch run. The simulator writes its active trials to
`simulation_log.csv`; the pre-launch log was preserved as
`simulation_log_previous.csv`. Click a free-space destination in pygame to
start a trial. The Panda3D demo uses WASD or arrow keys; **L** toggles LiDAR.
