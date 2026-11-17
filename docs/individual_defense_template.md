# Individual contribution and defense worksheet

**Individual-project draft:** This worksheet is filled from the repository's
code and recorded outputs. Review it before submission, correct anything that
does not match the work you personally did, and make sure you can explain the
choices and results in your own words.

## Student

- Name: Varun Kumar Gupta
- Project role: Sole student; project design, data analysis, model training,
  simulation, website, and documentation
- Date: 17 November 2026

## Contribution record

| Project area | What I personally changed or produced | File, commit, or artifact | How I checked it |
|---|---|---|---|
| Data acquisition and quality | Used the UCI Wall-Following Robot Navigation data as the primary classification dataset and analyzed the TurtleBot3 recording separately. Summarized row meaning, sensor columns, class balance, missingness, duplicates, and timestamp alignment. | `train_models.py`; `scripts/analyze_turtlebot_data.py`; `results/RUN_SUMMARY.md`; `data/processed/turtlebot3/` | Compared dataset dimensions and class counts with the loaded files; checked missing/duplicate features and reviewed the TurtleBot scan, odometry, and command alignment summaries. |
| Preprocessing and split design | Built training pipelines with median imputation; scaled Logistic Regression, RBF SVM, and K-Nearest Neighbors inputs. Used a stratified 80/20 holdout and five-fold stratified cross-validation on training data, keeping the holdout for final scoring. | `train_models.py`; `results/ordered_holdout_comparison.csv` | Reviewed the pipeline and split code; compared training-CV selection scores with the held-out results and the separate ordered-block check. |
| Model training and comparison | Compared a most-frequent Dummy baseline with Logistic Regression, RBF SVM, K-Nearest Neighbors, Random Forest, and Gradient Boosting. Reported accuracy, balanced accuracy, macro-F1, confusion matrices, and inference timing; selected by training-fold CV macro-F1. | `train_models.py`; `models/`; `results/model_comparison.csv`; `results/best_confusion_matrix.png` | Checked all models on the same held-out rows, compared them with the baseline, and inspected per-class errors in the confusion matrix. |
| Robustness or navigation evaluation | Evaluated noise, reading dropout, and reduced sensor-channel availability in repeated software experiments. Also analyzed BARN held-out maps, replayed masked channels on one TurtleBot3 recording, and recorded 100 distinct Pygame start/goal scenarios using the same simulated map and controller. These are software or recorded-data evaluations, not physical sensor-failure trials. | `train_models.py`; `train_barn_environment_eval.py`; `evaluate_turtlebot_recorded_dropout.py`; `run_navigation_trials.py`; `results/sensor_robustness_repeats.csv`; `results/barn_environment_generalization.csv`; `results/turtlebot_recorded_dropout.csv`; `results/navigation_trials.csv`; `results/navigation_trial_steps.csv` | Compared model and trial CSVs, inspected representative trajectories and failure outcomes, and kept the UCI classification, BARN simulation, TurtleBot3 log replay, and Pygame navigation results as separate experiments. |
| Website or visualization | Built an interactive research website with model comparison, sensor-stress charts, error analysis, a 2D ML navigation comparison, trial charts, model discussion, and an information-only project explainer terminal. | `website/index.html`; `website/app.js`; `website/tech-theme.css`; `website/network-scene.js`; `website/assets/` | Checked that the page links to its bundled result files and that interactive charts and controls use the project outputs described in the site. |
| Reproducibility and documentation | Documented installation, data access, training and simulator commands, preprocessing, limitations, citations, and AI assistance. Bundled result tables and figures for the website. | `README.md`; `requirements.txt`; `docs/`; `website/README.md`; `website/assets/results/` | Followed the documented file layout and checked that the named scripts, dependencies, result artifacts, source links, and AI-use disclosure are present. |

## Defense prompts

1. **Project question and observations.** I ask whether classical classifiers can predict the UCI Wall-Following Robot movement class from 24 ultrasonic readings, how their class-balanced scores change under simulated sensor corruption, and whether learned actions help a separate simulated controller reach goals. A UCI row is one sensor snapshot with a movement label. A TurtleBot3 row is a recorded scan or motion/command measurement aligned by time. A BARN example is a sensor/action sample from a reference path on a particular simulated map. These datasets have different sensors and labels, so I analyze them separately.

2. **Metrics and baseline.** Macro-F1 gives each movement class equal weight, which matters because the classes are imbalanced; accuracy alone can hide weak performance on uncommon classes. The most-frequent Dummy baseline has holdout macro-F1 about 0.144. Gradient Boosting reached about 0.994 macro-F1 and 0.996 accuracy on this random held-out snapshot split. These scores describe this dataset split, not safe navigation or performance on a new physical run.

3. **Leakage control.** I split the observations once with stratification into 80% training and 20% held-out test data. Imputation and scaling are inside model pipelines so they are fitted from training folds only. Five-fold stratified cross-validation on the training portion selects the candidate by mean macro-F1. The held-out test portion is reserved for the final comparison and is not used for model selection.

4. **Error and limitation.** Gradient Boosting made four errors in the held-out confusion matrix: two class-2 and two class-3 examples were predicted as class 1 (forward). The UCI data has no run IDs, so a random row split may put related observations on both sides. Its high score therefore does not establish generalization to new robot runs. The navigation simulations also use synthetic range sensing and a hand-coded controller layer, so their goal outcomes do not validate the UCI classifier as a safe route planner.

5. **BARN evaluation.** Splitting by map ID keeps paths from the same environment out of both training and test data, so the held-out maps provide a more meaningful check of transfer to unseen simulated layouts than randomly splitting individual path steps. The test still uses simulated maps and reference routes; it does not involve a physical robot, real sensor faults, or a real-world safety evaluation.

6. **TurtleBot3 dropout replay.** I used 1,772 scans paired with preceding velocity-command labels, held out the chronological final 355 observations, and masked saved LiDAR sector values in software for dropout conditions. Masking a recorded value is not the same as disconnecting, degrading, or faulting a physical sensor. The bag is one recording and its labels are command classes, not goal-conditioned routes.

7. **Modeling decision and predicted change.** The current candidate models use fixed parameters, with model selection by training-fold CV macro-F1. A useful next change is to compare a class-weighted Random Forest with the current unweighted version. I expect it may improve recall for less common actions, while possibly reducing accuracy on common classes. I would test that expectation with the same training-only cross-validation, then compare macro-F1, balanced accuracy, per-class recall, and the untouched test results.

## Small-change explanation

- **Change I would make:** Add a class-weighted Random Forest candidate to the existing training comparison.
- **Expected effect and reason:** Giving minority classes more influence may improve their recall and macro-F1, but could trade off some overall accuracy or majority-class precision. This is a hypothesis, not a measured result.
- **Evidence I would collect to check that expectation:** Use training-only cross-validation for selection, then report the single held-out test comparison with macro-F1, balanced accuracy, per-class precision/recall, and a confusion matrix. I would compare it directly with the existing unweighted Random Forest and Dummy baseline.
