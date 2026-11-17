from pathlib import Path
import pandas as pd

base = Path("data/processed/turtlebot3")

files = {
    "LiDAR": base / "scan_summary.csv",
    "Odometry": base / "odometry.csv",
    "Velocity commands": base / "velocity_commands.csv",
}

for name, path in files.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate timestamps:",
          df["timestamp_s"].duplicated().sum())

    print("\nNumeric summary:")
    print(df.describe().round(4).to_string())

print("\n" + "=" * 65)
print("Timestamp coverage")
print("=" * 65)

for name, path in files.items():
    df = pd.read_csv(path)
    print(
        f"{name}: "
        f"{df['timestamp_s'].min():.3f} to "
        f"{df['timestamp_s'].max():.3f} seconds"
    )