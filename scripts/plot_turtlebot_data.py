from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("data/processed/turtlebot3")
OUT = Path("results/turtlebot3")
OUT.mkdir(parents=True, exist_ok=True)

lidar = pd.read_csv(BASE / "scan_summary.csv")
odom = pd.read_csv(BASE / "odometry.csv")
commands = pd.read_csv(BASE / "velocity_commands.csv")

# Convert Unix timestamps to elapsed time in seconds.
t0 = min(
    lidar["timestamp_s"].min(),
    odom["timestamp_s"].min(),
    commands["timestamp_s"].min(),
)

for df in (lidar, odom, commands):
    df["elapsed_s"] = df["timestamp_s"] - t0

# 1. Robot trajectory from odometry.
plt.figure(figsize=(8, 6))
plt.plot(odom["x_m"], odom["y_m"])
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("TurtleBot3 Recorded Trajectory")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "robot_trajectory.png", dpi=200)
plt.close()

# 2. Minimum obstacle distance from each LiDAR scan.
plt.figure(figsize=(10, 5))
plt.plot(lidar["elapsed_s"], lidar["min_distance_m"])
plt.xlabel("Elapsed time (s)")
plt.ylabel("Minimum valid LiDAR distance (m)")
plt.title("Minimum Obstacle Distance Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "lidar_min_distance.png", dpi=200)
plt.close()

# 3. Recorded velocity commands.
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax[0].plot(commands["elapsed_s"], commands["linear_x_m_s"])
ax[0].set_ylabel("Linear velocity (m/s)")
ax[0].grid(True)

ax[1].plot(commands["elapsed_s"], commands["angular_z_rad_s"])
ax[1].set_xlabel("Elapsed time (s)")
ax[1].set_ylabel("Angular velocity (rad/s)")
ax[1].grid(True)

fig.suptitle("Recorded TurtleBot3 Velocity Commands")
fig.tight_layout()
fig.savefig(OUT / "velocity_commands.png", dpi=200)
plt.close(fig)

print(f"Plots saved to: {OUT.resolve()}")
for path in sorted(OUT.glob("*.png")):
    print(path.name)