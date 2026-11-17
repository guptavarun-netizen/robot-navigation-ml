from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. File paths and thresholds
# --------------------------------------------------

INPUT_FILE = Path("data/processed/turtlebot3/odometry.csv")
OUTPUT_FILE = Path(
    "data/processed/turtlebot3/odometry_cleaned.csv"
)

MIN_INTERVAL = 0.010  # seconds
MAX_INTERVAL = 0.100  # seconds

# --------------------------------------------------
# 2. Load and sort the original dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df = (
    df.sort_values("timestamp_s")
    .reset_index(drop=True)
)

print("ORIGINAL DATA")
print("-" * 45)
print(f"Original records: {len(df)}")

# --------------------------------------------------
# 3. Calculate original timestamp intervals
# --------------------------------------------------

df["original_dt_s"] = df["timestamp_s"].diff()

valid_interval = df["original_dt_s"].between(
    MIN_INTERVAL,
    MAX_INTERVAL,
    inclusive="both",
)

# Preserve the first record
valid_interval.iloc[0] = True

# --------------------------------------------------
# 4. Filter records
# --------------------------------------------------

cleaned = (
    df.loc[valid_interval]
    .copy()
    .reset_index(drop=True)
)

# --------------------------------------------------
# 5. Recalculate intervals after filtering
# --------------------------------------------------

cleaned["dt_s"] = cleaned["timestamp_s"].diff()

# --------------------------------------------------
# 6. Calculate position changes
# --------------------------------------------------

cleaned["dx_m"] = cleaned["x_m"].diff()
cleaned["dy_m"] = cleaned["y_m"].diff()

cleaned["position_change_m"] = np.sqrt(
    cleaned["dx_m"] ** 2
    + cleaned["dy_m"] ** 2
)

# --------------------------------------------------
# 7. Calculate position-derived speed
# --------------------------------------------------

valid_dt = cleaned["dt_s"] > 0

cleaned["calculated_speed_m_s"] = np.nan

cleaned.loc[valid_dt, "calculated_speed_m_s"] = (
    cleaned.loc[valid_dt, "position_change_m"]
    / cleaned.loc[valid_dt, "dt_s"]
)

# First row has no preceding retained record
cleaned.loc[
    0,
    [
        "dx_m",
        "dy_m",
        "position_change_m",
        "calculated_speed_m_s",
    ],
] = np.nan

# --------------------------------------------------
# 8. Save the cleaned dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

cleaned.to_csv(
    OUTPUT_FILE,
    index=False,
)

# --------------------------------------------------
# 9. Report cleaning statistics
# --------------------------------------------------

print("\nCLEANING SUMMARY")
print("-" * 45)

print(f"Original records: {len(df)}")
print(f"Retained records: {len(cleaned)}")
print(f"Excluded records: {len(df) - len(cleaned)}")

print(f"\nSaved to: {OUTPUT_FILE}")

print("\nOriginal interval statistics (s):")
print(df["original_dt_s"].describe())

print("\nIntervals between retained records (s):")
print(cleaned["dt_s"].describe())

print("\nCalculated speed statistics (m/s):")
print(cleaned["calculated_speed_m_s"].describe())

print("\nMissing values:")
print(cleaned.isna().sum())