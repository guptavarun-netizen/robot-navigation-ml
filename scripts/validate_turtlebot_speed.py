from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path(
    "data/processed/turtlebot3/odometry_cleaned.csv"
)

OUTPUT_DIR = Path("results/turtlebot3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load and sort the dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df = (
    df.sort_values("timestamp_s")
    .reset_index(drop=True)
)

print("ORIGINAL DATA SUMMARY")
print("=" * 50)
print(f"Rows loaded: {len(df)}")

# --------------------------------------------------
# 2. Recalculate time intervals and displacement
# --------------------------------------------------

df["dt_validation_s"] = df["timestamp_s"].diff()

dx = df["x_m"].diff()
dy = df["y_m"].diff()

df["position_change_validation_m"] = np.sqrt(
    dx**2 + dy**2
)

# --------------------------------------------------
# 3. Calculate position-derived speed
# --------------------------------------------------

df["position_speed_m_s"] = (
    df["position_change_validation_m"]
    / df["dt_validation_s"]
)

# --------------------------------------------------
# 4. Calculate recorded odometry speed magnitude
# --------------------------------------------------

df["recorded_speed_m_s"] = np.sqrt(
    df["linear_velocity_x_m_s"] ** 2
    + df["linear_velocity_y_m_s"] ** 2
)

# --------------------------------------------------
# 5. Check data validity
# --------------------------------------------------

valid_interval = (
    np.isfinite(df["dt_validation_s"])
    & (df["dt_validation_s"] > 0)
)

valid_position = (
    np.isfinite(df["x_m"])
    & np.isfinite(df["y_m"])
)

valid_velocity = (
    np.isfinite(df["linear_velocity_x_m_s"])
    & np.isfinite(df["linear_velocity_y_m_s"])
)

valid_speed = (
    np.isfinite(df["position_speed_m_s"])
    & np.isfinite(df["recorded_speed_m_s"])
)

valid_mask = (
    valid_interval
    & valid_position
    & valid_velocity
    & valid_speed
)

valid = df.loc[valid_mask].copy()

print("\nDATA QUALITY SUMMARY")
print("=" * 50)
print(f"Rows loaded: {len(df)}")
print(f"Valid rows compared: {len(valid)}")
print(f"Rows excluded: {len(df) - len(valid)}")

print("\nRecalculated time interval statistics (s):")
print(df["dt_validation_s"].describe())

print("\nIntervals below 1 ms:")
print((df["dt_validation_s"] < 0.001).sum())

print("\nIntervals below 10 ms:")
print((df["dt_validation_s"] < 0.010).sum())

print("\nIntervals above 100 ms:")
print((df["dt_validation_s"] > 0.100).sum())

# --------------------------------------------------
# 6. Calculate validation metrics
# --------------------------------------------------

if len(valid) < 2:
    raise ValueError(
        "Insufficient valid observations for speed validation."
    )

position_speed = valid["position_speed_m_s"].to_numpy()
recorded_speed = valid["recorded_speed_m_s"].to_numpy()

errors = position_speed - recorded_speed

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
bias = np.mean(errors)

correlation = np.corrcoef(
    position_speed,
    recorded_speed
)[0, 1]

print("\nVALIDATION METRICS")
print("=" * 50)

print(f"Mean absolute error (m/s): {mae:.6f}")
print(f"Root mean squared error (m/s): {rmse:.6f}")
print(f"Mean signed error (m/s): {bias:.6f}")
print(f"Pearson correlation: {correlation:.6f}")

print("\nPosition-derived speed (m/s):")
print(valid["position_speed_m_s"].describe())

print("\nRecorded odometry speed (m/s):")
print(valid["recorded_speed_m_s"].describe())

# --------------------------------------------------
# 7. Save the comparison data
# --------------------------------------------------

csv_path = OUTPUT_DIR / "speed_validation.csv"

valid.to_csv(
    csv_path,
    index=False
)

# --------------------------------------------------
# 8. Plot the speed comparison
# --------------------------------------------------

elapsed_time = (
    valid["timestamp_s"]
    - valid["timestamp_s"].iloc[0]
)

plt.figure(figsize=(12, 6))

plt.plot(
    elapsed_time,
    valid["position_speed_m_s"],
    label="Position-derived speed",
    linewidth=1
)

plt.plot(
    elapsed_time,
    valid["recorded_speed_m_s"],
    label="Recorded odometry speed",
    linewidth=1
)

plt.xlabel("Elapsed time (s)")
plt.ylabel("Speed (m/s)")
plt.title("TurtleBot3 Odometry Speed Validation")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = OUTPUT_DIR / "speed_validation.png"

plt.savefig(
    plot_path,
    dpi=200
)

plt.close()

print("\nSaved files:")
print(csv_path)
print(plot_path)