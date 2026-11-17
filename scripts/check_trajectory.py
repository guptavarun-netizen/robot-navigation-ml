import numpy as np
import pandas as pd
from pathlib import Path

path = Path("data/processed/turtlebot3/odometry.csv")
df = pd.read_csv(path)

# Calculate the change between consecutive odometry positions.
df["dx"] = df["x_m"].diff()
df["dy"] = df["y_m"].diff()

df["position_change_m"] = np.hypot(
    df["dx"],
    df["dy"],
)

# Calculate elapsed time between consecutive records.
df["dt_s"] = df["timestamp_s"].diff()

# Estimate the implied displacement rate.
df["implied_speed_m_s"] = (
    df["position_change_m"] / df["dt_s"]
)

print("Number of odometry records:", len(df))
print("\nTime interval statistics (seconds):")
print(df["dt_s"].describe().round(5))

print("\nPosition-change statistics (metres):")
print(df["position_change_m"].describe().round(5))

print("\nLargest 15 consecutive position changes:")
columns = [
    "timestamp_s",
    "x_m",
    "y_m",
    "dt_s",
    "position_change_m",
    "implied_speed_m_s",
]

print(
    df.nlargest(15, "position_change_m")[columns]
    .to_string(index=False)
)

print("\nRecords with position changes above 0.10 m:")
jumps = df[df["position_change_m"] > 0.10]
print("Count:", len(jumps))

if len(jumps):
    print(jumps[columns].to_string(index=False))

out = Path("results/turtlebot3")
out.mkdir(parents=True, exist_ok=True)

jumps.to_csv(out / "trajectory_jump_candidates.csv", index=False)

print("\nSaved results/turtlebot3/trajectory_jump_candidates.csv")

# Additional timestamp-quality analysis
print("\n" + "=" * 60)
print("TIMESTAMP QUALITY ANALYSIS")
print("=" * 60)

valid_dt = df["dt_s"].dropna()

print("\nIntervals below 1 millisecond:")
print((valid_dt < 0.001).sum())

print("\nIntervals below 10 milliseconds:")
print((valid_dt < 0.010).sum())

print("\nIntervals above 100 milliseconds:")
print((valid_dt > 0.100).sum())

print("\nIntervals above 200 milliseconds:")
print((valid_dt > 0.200).sum())

print("\nIntervals between 10 and 100 milliseconds:")
print(
    ((valid_dt >= 0.010) & (valid_dt <= 0.100)).sum()
)

print("\nImplied speeds above 0.5 m/s:")
fast = df[df["implied_speed_m_s"] > 0.5]

print("Count:", len(fast))

if len(fast):
    print(
        fast[
            [
                "timestamp_s",
                "dt_s",
                "position_change_m",
                "implied_speed_m_s",
            ]
        ].to_string(index=False)
    )