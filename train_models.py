
from pathlib import Path
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

for directory in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# 2. DOWNLOAD AND LOAD PUBLIC DATA
# --------------------------------------------------

def load_dataset():
    from scipy.io import arff

    print("\nLoading downloaded robot navigation dataset...")

    dataset_path = (
        ROOT / "data" / "raw" / "wall-robot-navigation.arff"
    )

    csv_path = DATA_DIR / "uci_wall_following_24.csv"

    if not dataset_path.exists() and csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Class" not in df.columns:
            raise ValueError(f"Expected a 'Class' column in {csv_path}")
        X = df.drop(columns="Class").apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        y = df["Class"].astype(str).str.strip()
        print(f"Dataset loaded from CSV: {csv_path}")
        return X, y

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Also checked CSV fallback: {csv_path}"
        )

    # Read the ARFF dataset.
    records, metadata = arff.loadarff(dataset_path)

    # Convert records to a pandas DataFrame.
    df = pd.DataFrame(records)

    # Decode text labels if the ARFF reader returns bytes.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].apply(
            lambda value: (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else value
            )
        )

    # The last column is the movement class.
    target_column = df.columns[-1]

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].astype(str).str.strip()

    # Ensure features are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Save a CSV copy for convenient inspection.
    local_data = X.copy()
    local_data["Class"] = y.values

    local_data.to_csv(
        DATA_DIR / "uci_wall_following_24.csv",
        index=False,
    )

    print(f"Dataset loaded: {X.shape[0]} observations")
    print(f"Sensor features: {X.shape[1]}")
    print(f"Target column: {target_column}")

    return X, y

# --------------------------------------------------
# 3. INSPECT DATA
# --------------------------------------------------

def inspect_dataset(X, y):
    print("\n" + "=" * 55)
    print("DATASET INSPECTION")
    print("=" * 55)

    print(f"Number of observations: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Missing feature values: {X.isna().sum().sum()}")
    print(f"Duplicate feature rows: {X.duplicated().sum()}")

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nClass proportions:")
    print(y.value_counts(normalize=True).round(4))

    print("\nFeature summary:")
    print(X.describe().round(3))

    # Plot class distribution.
    y.value_counts().plot(kind="bar")
    plt.title("UCI Wall-Following Class Distribution")
    plt.xlabel("Movement class")
    plt.ylabel("Number of observations")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "class_distribution.png",
        dpi=200,
    )
    plt.close()

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining observations:", len(X_train))
    print("Testing observations:", len(X_test))

    return X_train, X_test, y_train, y_test

# --------------------------------------------------
# 5. DEFINE CLASSICAL ML MODELS
# --------------------------------------------------

def create_models():
    models = {
        "Dummy": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "RBF SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
            )),
        ]),

        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                n_jobs=-1,
            )),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    return models

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    fitted_models = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for name, model in models.items():
        print("\n" + "-" * 55)
        print("Training:", name)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        # Time predictions over the complete test set.
        start = time.perf_counter()
        predictions = model.predict(X_test)
        prediction_time = time.perf_counter() - start

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        balanced_acc = balanced_accuracy_score(
            y_test,
            predictions,
        )

        # Cross-validation is performed on training data only.
        cv_scores = cross_val_score(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        result = {
            "model": name,
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
            "test_balanced_accuracy": balanced_acc,
            "cv_macro_f1_mean": cv_scores.mean(),
            "cv_macro_f1_std": cv_scores.std(),
            "training_seconds": training_time,
            "prediction_ms_per_sample": (
                prediction_time / len(X_test) * 1000
            ),
        }

        results.append(result)
        fitted_models[name] = model

        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Macro-F1:           {macro_f1:.4f}")
        print(f"Balanced accuracy:  {balanced_acc:.4f}")
        print(
            f"CV Macro-F1:        {cv_scores.mean():.4f} "
            f"+/- {cv_scores.std():.4f}"
        )
        print(f"Training time:      {training_time:.3f} s")
        print(
            f"Prediction latency: "
            f"{result['prediction_ms_per_sample']:.4f} ms/sample"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "cv_macro_f1_mean",
        ascending=False,
    )

    results_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
    )

    return results_df, fitted_models

# --------------------------------------------------
# 7. SAVE BEST MODEL AND REPORT
# --------------------------------------------------

def save_best_model(results_df, fitted_models,
                    X_test, y_test):
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    predictions = best_model.predict(X_test)

    print("\n" + "=" * 55)
    print("SELECTED MODEL:", best_name)
    print("=" * 55)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    # Save the confusion matrix.
    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=35,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / "best_confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    print("Saved model to:", model_path)
    print("Saved results to:", RESULT_DIR)

    return best_name, best_model

# --------------------------------------------------
# 8. ROBUSTNESS TEST: GAUSSIAN SENSOR NOISE
# --------------------------------------------------

def run_noise_test(model, X_train, X_test, y_test):
    """
    Illustrative noise test.
    Noise is added to copies of the held-out test features.
    Sigma is measured in each feature's standard deviations.
    """

    # Estimate perturbation scales from training data only.
    scales = X_train.std(axis=0).fillna(0).to_numpy()
    X_array = X_test.to_numpy(dtype=float)

    noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20]
    rng = np.random.default_rng(RANDOM_STATE)

    records = []

    for sigma in noise_levels:
        corrupted = (
            X_array
            + rng.normal(
                loc=0.0,
                scale=sigma * scales,
                size=X_array.shape,
            )
        )

        corrupted_df = pd.DataFrame(
            corrupted,
            columns=X_test.columns,
        )

        predictions = model.predict(corrupted_df)

        score = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        records.append({
            "noise_sigma_feature_std": sigma,
            "macro_f1": score,
        })

        print(
            f"Noise={sigma:.2f} feature std | "
            f"Macro-F1={score:.4f}"
        )

    noise_df = pd.DataFrame(records)
    noise_df.to_csv(
        RESULT_DIR / "noise_robustness.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        noise_df["noise_sigma_feature_std"],
        noise_df["macro_f1"],
        marker="o",
    )
    plt.xlabel("Noise level (feature standard deviations)")
    plt.ylabel("Macro-F1")
    plt.title("Model Robustness to Sensor Noise")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "noise_robustness.png",
        dpi=200,
    )
    plt.close()


def run_robustness_suite(fitted_models, X_train, X_test, y_test):
    """Compare all fitted UCI models under noise and missing sensor inputs.

    Each condition is repeated with independent deterministic masks/noise.
    Missing channels are represented as NaN so the training-fitted median
    imputer supplies a train-only replacement at prediction time.
    """
    rng = np.random.default_rng(RANDOM_STATE + 100)
    feature_count = X_test.shape[1]
    raw = X_test.to_numpy(dtype=float)
    scales = X_train.std(axis=0).fillna(0).to_numpy()
    repeats = 10
    conditions = []
    for sigma in [0.0, 0.02, 0.05, 0.10, 0.20]:
        conditions.append(("gaussian_noise", sigma, f"{sigma:.2f}σ"))
    for rate in [0.0, 0.25, 0.50, 0.75]:
        conditions.append(("random_dropout", rate, f"{int(rate * 100)}%"))
    for available in [24, 18, 12, 6]:
        conditions.append(("available_sensors", available, str(available)))

    records = []
    for experiment, amount, label in conditions:
        for repeat in range(repeats):
            if experiment == "gaussian_noise":
                corrupted = raw + rng.normal(0, amount * scales, size=raw.shape)
            elif experiment == "random_dropout":
                corrupted = raw.copy()
                missing = rng.random(raw.shape) < amount
                corrupted[missing] = np.nan
            else:
                corrupted = raw.copy()
                keep = rng.choice(feature_count, size=int(amount), replace=False)
                dropped = np.ones(feature_count, dtype=bool)
                dropped[keep] = False
                corrupted[:, dropped] = np.nan

            frame = pd.DataFrame(corrupted, columns=X_test.columns)
            for name, model in fitted_models.items():
                predictions = model.predict(frame)
                records.append({
                    "experiment": experiment,
                    "condition": label,
                    "condition_value": amount,
                    "repeat": repeat + 1,
                    "model": name,
                    "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
                })

    raw_results = pd.DataFrame(records)
    summary = (raw_results.groupby(["experiment", "condition", "condition_value", "model"])
               .macro_f1.agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std"}))
    raw_results.to_csv(RESULT_DIR / "sensor_robustness_repeats.csv", index=False)
    summary.to_csv(RESULT_DIR / "sensor_robustness.csv", index=False)


def run_ordered_holdout(X, y, models):
    """Supplement the random split with a final-contiguous-row holdout.

    Source data has no trajectory/run identifiers. This ordered split is a
    drift stress-test only and must not be described as run-wise validation.
    """
    cut = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    rows = []
    for name, definition in models.items():
        model = clone(definition).fit(X_train, y_train)
        prediction = model.predict(X_test)
        rows.append({
            "model": name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
            "test_accuracy": accuracy_score(y_test, prediction),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, prediction),
            "test_start_row": cut,
            "test_end_row_exclusive": len(X),
        })
    pd.DataFrame(rows).to_csv(RESULT_DIR / "ordered_holdout_comparison.csv", index=False)

# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

def main():
    try:
        X, y = load_dataset()
    except Exception as error:
        print("\nDataset download failed.")
        print("Check your internet connection and installation.")
        print("Error:", error)
        return

    inspect_dataset(X, y)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    models = create_models()

    results_df, fitted_models = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    best_name, best_model = save_best_model(
        results_df,
        fitted_models,
        X_test,
        y_test,
    )

    run_noise_test(best_model, X_train, X_test, y_test)
    run_robustness_suite(fitted_models, X_train, X_test, y_test)
    run_ordered_holdout(X, y, create_models())

    print("\nPipeline completed.")
    print("Best candidate:", best_name)
    print("Check the results/ and models/ folders.")

if __name__ == "__main__":
    main()
