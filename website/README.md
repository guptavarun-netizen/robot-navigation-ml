# Research website

This is a self-contained static website for the robot navigation ML project.
It bundles snapshots of project result tables, figures, the UCI CSV, and
TurtleBot3 processed data under `assets/`.

From the repository root, run:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/website/>. Serving over HTTP is required
because the interactive comparison reads CSV files with `fetch`. The site
has no build step or package install. The animated background loads the Three.js
ES module from jsDelivr; when the CDN or WebGL is unavailable, the research site
and its non-WebGL animations remain usable.

To refresh its data after regenerating results, copy the updated CSV and image
files from `results/` into the matching folders under `assets/`. Update the
research text and run summary if the evaluation protocol or reported metrics
change. The source repository remains the canonical version of the training
pipeline and analysis scripts.

The site includes model ranking controls, a selectable all-model sensor robustness chart, an
interactive 100-trial bar chart with selectable time, distance, or path
directness, an outcome donut, and a model-controlled 2D simulation. The planner
section lets visitors place shared start and goal points, edit obstacles, and
run all five goal-conditioned ML policies together on the same map, with at
most 15 user-added blocks. It draws each trajectory and compares moves, path
length, simulated time, contacts, skipped choices, and goal outcome, with
position-time, goalward-speed, and validation graphs, descriptive summary
statistics, and a per-run CSV download. No route search runs in the browser; local collision
checks and recent-position memory help prevent blocked moves and loops while
trying each model's next-ranked actions. The
policies use synthetic goal and sensor examples, not UCI wall-following data.
They receive 24 simulated range readings, the goal offset, and robot position,
and are trained on 56 randomized layouts (up to 35 added blocks) plus a dense
reference map.
Offline shortest-path distance maps provide their move labels; they are
validated on held-out obstacle layouts and can still fail on an unseen route or
an unreachable edited map. The policy comparison CSV also reports whether each
model reached the dense reference goal in a replay.

The simulation section compares 100 deterministic fixed-start goal scenarios,
lets readers inspect each trajectory, and provides the summary and per-step
CSVs. The first ten goals are retained and 90 additional collision-free goals
are generated with seed `20261117`. These are software trials on one map, not
independent physical robot runs. Regenerate them from the repository root with
`python run_navigation_trials.py`.

Regenerate the goal-conditioned policy models and comparison metrics from the
repository root with `python train_navigation_policies.py`. It writes
`assets/models/goal_navigation_models.json` and
`results/navigation_policy_comparison.csv`.

The UCI results include all six classifiers under Gaussian noise, random
per-reading dropout, and 24/18/12/6 available sensor channels. Each condition
uses 10 perturbation repeats. The site links summary and repeat-level CSVs and
shows a supplementary ordered-block holdout comparison. The source dataset
has no run IDs, so this is not presented as verified run-wise validation.
Regenerate these with `python train_models.py`; copy the resulting robustness
and ordered-holdout CSVs from `results/` into `website/assets/results/` when
refreshing the bundled site.

Supplementary evidence is regenerated with `python train_barn_environment_eval.py`
and `python evaluate_turtlebot_recorded_dropout.py`. Both scripts copy their
tables into `website/assets/results/`. The BARN table reports held-out
simulated map IDs; the TurtleBot3 table reports channel masking on one recorded
robot bag, not an injected physical sensor fault. The team should complete the
contribution and defense worksheet at `docs/individual_defense_template.md`.

The TurtleBot3 section also links a physical-run collection workflow. Start
with `python record_robot_trial.py --goal-x X --goal-y Y`, then run
`python extract_robot_goal_runs.py` and `python evaluate_real_goal_runs.py`.
That final command requires at least two independent bags and writes
`real_goal_run_validation.csv` for the site. No extra bags are currently in
this workspace, so the website does not show fabricated run-held-out scores.

The `Next model experiments` section suggests ways to improve data coverage,
rare-class performance, model selection, and temporal context. These are
proposals only; the website does not claim that they have improved measured
accuracy. Individual contribution and defense work is handled separately by
the student and is not presented as a website feature.

The visual treatment uses layered depth, pointer-responsive card tilt, a
scanning sensor overlay, and restrained motion on result elements. A shared
Three.js scene adds a rotating wireframe network, linked data nodes, particles,
mouse parallax, and a gentle scroll zoom across the page. Rendering is capped
near 30 frames per second and paused when the tab is hidden or no page section
is visible. Browser `prefers-reduced-motion` settings disable continuous scene
motion and the card transforms. The `Project explainer` control (or `/` shortcut) opens
an information-only console that does not execute code or open files; commands include `help`, `about`, `projects`, `skills`,
`models`, `results`, `limits`, and `clear`.

The site remains static HTML, CSS, and JavaScript so its existing research
widgets work without a frontend build step. The Three.js module is loaded at
runtime from a pinned CDN release rather than bundled locally.
