from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_DIR = Path("data/processed/turtlebot3")
OUTPUT_DIR = Path("results/turtlebot3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ODOM_FILE = DATA_DIR / "odometry_cleaned.csv"
LIDAR_FILE = DATA_DIR / "scan_summary.csv"
COMMAND_FILE = DATA_DIR / "velocity_commands.csv"

# Maximum permitted timestamp difference when matching records.
# This is a provisional value and will be checked against sampling intervals.
TIME_TOLERANCE = 0.10  # seconds


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

odom = pd.read_csv(ODOM_FILE)
lidar = pd.read_csv(LIDAR_FILE)
commands = pd.read_csv(COMMAND_FILE)

# Sort each dataset chronologically.
odom = odom.sort_values("timestamp_s").reset_index(drop=True)
lidar = lidar.sort_values("timestamp_s").reset_index(drop=True)
commands = commands.sort_values("timestamp_s").reset_index(drop=True)

print("DATASET SIZES")
print("=" * 50)
print(f"Odometry records: {len(odom)}")
print(f"LiDAR records: {len(lidar)}")
print(f"Velocity-command records: {len(commands)}")


# --------------------------------------------------
# 3. Calculate recorded odometry speed
# --------------------------------------------------

odom["recorded_speed_m_s"] = np.sqrt(
    odom["linear_velocity_x_m_s"] ** 2
    + odom["linear_velocity_y_m_s"] ** 2
)

# Retain only the columns needed for comparison.
odom_selected = odom[
    [
        "timestamp_s",
        "x_m",
        "y_m",
        "recorded_speed_m_s",
        "angular_velocity_z_rad_s",
    ]
].copy()


# --------------------------------------------------
# 4. Rename columns to avoid ambiguity
# --------------------------------------------------

lidar_selected = lidar[
    ["timestamp_s", "min_distance_m"]
].copy()

lidar_selected = lidar_selected.rename(
    columns={"timestamp_s": "lidar_timestamp_s"}
)

commands_selected = commands[
    ["timestamp_s", "linear_x_m_s", "angular_z_rad_s"]
].copy()

commands_selected = commands_selected.rename(
    columns={"timestamp_s": "command_timestamp_s"}
)


# --------------------------------------------------
# 5. Match LiDAR measurements with odometry
# --------------------------------------------------

merged = pd.merge_asof(
    lidar_selected.sort_values("lidar_timestamp_s"),
    odom_selected.sort_values("timestamp_s"),
    left_on="lidar_timestamp_s",
    right_on="timestamp_s",
    direction="nearest",
    tolerance=TIME_TOLERANCE,
)

# Match velocity commands with the corresponding LiDAR records.
merged = pd.merge_asof(
    merged.sort_values("lidar_timestamp_s"),
    commands_selected.sort_values("command_timestamp_s"),
    left_on="lidar_timestamp_s",
    right_on="command_timestamp_s",
    direction="nearest",
    tolerance=TIME_TOLERANCE,
)

# Keep only records with all required measurements.
required_columns = [
    "min_distance_m",
    "recorded_speed_m_s",
    "linear_x_m_s",
    "angular_z_rad_s",
]

valid = merged.dropna(subset=required_columns).copy()

print("\nTIMESTAMP ALIGNMENT")
print("=" * 50)
print(f"LiDAR records before alignment: {len(lidar)}")
print(f"Records matched across all datasets: {len(valid)}")
print(f"Unmatched LiDAR records: {len(lidar) - len(valid)}")


# --------------------------------------------------
# 6. Compare commanded and recorded linear speed
# --------------------------------------------------

valid["command_speed_m_s"] = valid["linear_x_m_s"].abs()

valid["speed_error_m_s"] = (
    valid["recorded_speed_m_s"] - valid["command_speed_m_s"]
)

print("\nVELOCITY COMPARISON")
print("=" * 50)

if len(valid) > 1:
    correlation = valid[
        ["command_speed_m_s", "recorded_speed_m_s"]
    ].corr().iloc[0, 1]

    print(f"Commanded vs recorded speed correlation: {correlation:.4f}")

print(
    "Mean absolute speed difference (m/s):",
    valid["speed_error_m_s"].abs().mean(),
)

print(
    "Median absolute speed difference (m/s):",
    valid["speed_error_m_s"].abs().median(),
)


# --------------------------------------------------
# 7. Examine obstacle distance and speed
# --------------------------------------------------

if len(valid) > 1:
    obstacle_speed_correlation = valid[
        ["min_distance_m", "recorded_speed_m_s"]
    ].corr().iloc[0, 1]

    print(
        "\nCorrelation between minimum LiDAR distance "
        "and recorded speed:",
        round(obstacle_speed_correlation, 4),
    )


# --------------------------------------------------
# 8. Save aligned data
# --------------------------------------------------

output_csv = OUTPUT_DIR / "aligned_sensor_data.csv"
valid.to_csv(output_csv, index=False)


# --------------------------------------------------
# 9. Plot commanded and recorded speeds
# --------------------------------------------------

if not valid.empty:
    elapsed_time = (
        valid["lidar_timestamp_s"] - valid["lidar_timestamp_s"].iloc[0]
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        elapsed_time,
        valid["command_speed_m_s"],
        label="Commanded linear speed",
        linewidth=1,
    )

    plt.plot(
        elapsed_time,
        valid["recorded_speed_m_s"],
        label="Recorded odometry speed",
        linewidth=1,
    )

    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("TurtleBot3 Commanded vs Recorded Speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "command_vs_recorded_speed.png",
        dpi=200,
    )

    plt.close()


# --------------------------------------------------
# 10. Plot obstacle distance against speed
# --------------------------------------------------

if len(valid) > 1:
    plt.figure(figsize=(8, 5))

    plt.scatter(
        valid["min_distance_m"],
        valid["recorded_speed_m_s"],
        alpha=0.5,
    )

    plt.xlabel("Minimum LiDAR distance (m)")
    plt.ylabel("Recorded odometry speed (m/s)")
    plt.title("Obstacle Distance vs Robot Speed")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "obstacle_distance_vs_speed.png",
        dpi=200,
    )

    plt.close()


# --------------------------------------------------
# 11. Report saved files
# --------------------------------------------------

print("\nSAVED OUTPUTS")
print("=" * 50)
print(output_csv)
print(OUTPUT_DIR / "command_vs_recorded_speed.png")
print(OUTPUT_DIR / "obstacle_distance_vs_speed.png")