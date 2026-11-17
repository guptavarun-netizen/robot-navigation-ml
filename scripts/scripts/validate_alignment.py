from pathlib import Path
import pandas as pd

file_path = Path("results/turtlebot3/aligned_sensor_data.csv")

print("=" * 60)
print("TURTLEBOT3 ALIGNED DATA VALIDATION")
print("=" * 60)

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path)

print("\n1. DATASET SIZE")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")

print("\n2. MISSING VALUES")
print(df.isna().sum())

print("\n3. TIMESTAMP COVERAGE")
timestamp_columns = [
    col for col in df.columns
    if "timestamp" in col.lower() and pd.api.types.is_numeric_dtype(df[col])
]

for col in timestamp_columns:
    values = df[col].dropna()
    if not values.empty:
        print(
            f"{col}: start={values.min():.6f}, "
            f"end={values.max():.6f}, "
            f"duration={values.max() - values.min():.6f} seconds"
        )

print("\n4. DUPLICATE ROWS")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n5. NUMERIC SUMMARY")
print(df.describe().T.to_string())

print("\n6. VELOCITY ERROR")
if {"command_speed_m_s", "speed_error_m_s"}.issubset(df.columns):
    print(f"Mean absolute error: {df['speed_error_m_s'].abs().mean():.6f} m/s")
    print(f"Median absolute error: {df['speed_error_m_s'].abs().median():.6f} m/s")
else:
    print("Expected velocity columns were not found.")

print("\n7. ALIGNMENT CHECK")
if "lidar_timestamp_s" in df.columns and "command_timestamp_s" in df.columns:
    difference = (
        df["lidar_timestamp_s"] - df["command_timestamp_s"]
    ).abs()

    print(f"Mean absolute timestamp difference: {difference.mean():.6f} s")
    print(f"Maximum absolute timestamp difference: {difference.max():.6f} s")
else:
    print("Required timestamp columns were not found.")

print("\nValidation finished.")