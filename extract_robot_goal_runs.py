"""Convert supervised ROS 1 goal-run bags to a model-ready sensor/action CSV."""
import json
from bisect import bisect_right
from pathlib import Path
import math

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "data" / "real_robot_runs"
OUT = RUN_DIR / "goal_conditioned_steps.csv"
QUALITY_OUT = RUN_DIR / "run_sensor_quality.csv"
QUALITY_WEB_OUT = ROOT / "website" / "assets" / "results" / "real_run_sensor_quality.csv"
SECTORS = 24
MAX_COMMAND_GAP_S = 0.5
FEATURES = [f"range_{i:02d}" for i in range(SECTORS)]


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def action_from_command(linear, angular):
    if abs(linear) < 0.01 and abs(angular) < 0.08:
        return "STOP"
    if abs(angular) >= 0.08:
        return "TURN_LEFT" if angular > 0 else "TURN_RIGHT"
    return "FORWARD" if linear > 0 else "REVERSE"


def sector_ranges(scan):
    values = np.asarray(scan.ranges, dtype=float)
    values[~np.isfinite(values)] = float(scan.range_max)
    values[(values < scan.range_min) | (values > scan.range_max)] = float(scan.range_max)
    chunks = np.array_split(values, SECTORS)
    return [float(np.min(chunk)) if len(chunk) else float(scan.range_max) for chunk in chunks]


def process_run(meta_path, typestore):
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    bag_path = meta_path.parent / metadata["bag_file"]
    if not bag_path.exists():
        raise FileNotFoundError(f"Missing bag for {metadata['run_id']}: {bag_path}")
    scans, odometry, commands = [], [], []
    with Reader(bag_path) as reader:
        wanted = {"/scan", "/odom", "/cmd_vel_rc100"}
        connections = [c for c in reader.connections if c.topic in wanted]
        for connection, stamp_ns, raw in reader.messages(connections=connections):
            msg = typestore.deserialize_ros1(raw, connection.msgtype)
            stamp = stamp_ns / 1e9
            if connection.topic == "/scan":
                scans.append((stamp, msg, sector_ranges(msg)))
            elif connection.topic == "/odom":
                pose = msg.pose.pose
                odometry.append((stamp, float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation)))
            else:
                commands.append((stamp, float(msg.linear.x), float(msg.angular.z)))

    odometry.sort(key=lambda row: row[0])
    commands.sort(key=lambda row: row[0])
    odom_times = [row[0] for row in odometry]
    command_times = [row[0] for row in commands]
    goal_x, goal_y = float(metadata["goal_x_m"]), float(metadata["goal_y_m"])
    rows = []
    for stamp, scan, ranges in scans:
        oi = bisect_right(odom_times, stamp) - 1
        ci = bisect_right(command_times, stamp) - 1
        if oi < 0 or ci < 0 or stamp - odom_times[oi][0] > MAX_COMMAND_GAP_S or stamp - command_times[ci][0] > MAX_COMMAND_GAP_S:
            continue
        _, x, y, yaw = odometry[oi]
        _, linear, angular = commands[ci]
        dx, dy = goal_x - x, goal_y - y
        # Transform goal offset into the robot's current body frame.
        goal_forward = (math.cos(yaw) * dx + math.sin(yaw) * dy)
        goal_left = (-math.sin(yaw) * dx + math.cos(yaw) * dy)
        row = {
            "run_id": metadata["run_id"], "timestamp_s": stamp,
            "x_m": x, "y_m": y, "yaw_rad": yaw,
            "goal_x_m": goal_x, "goal_y_m": goal_y,
            "goal_forward_m": goal_forward, "goal_left_m": goal_left,
            "goal_distance_m": math.hypot(dx, dy),
            "linear_command_m_s": linear, "angular_command_rad_s": angular,
            "action": action_from_command(linear, angular),
            "fault_condition": metadata.get("fault_condition", "unreported"),
        }
        row.update({feature: value for feature, value in zip(FEATURES, ranges)})
        rows.append(row)
    scan_times = np.asarray([row[0] for row in scans], dtype=float)
    intervals = np.diff(scan_times)
    intervals = intervals[intervals > 0]
    typical_period = float(np.median(intervals)) if len(intervals) else float("nan")
    gap_count = int(np.sum(intervals > max(3.0 * typical_period, 0.2))) if len(intervals) else 0
    quality = {
        "run_id": metadata["run_id"],
        "fault_condition": metadata.get("fault_condition", "unreported"),
        "operator_fault_annotation": metadata.get("fault_notes", ""),
        "scan_messages": len(scans),
        "aligned_action_rows": len(rows),
        "recorded_duration_s": float(scan_times[-1] - scan_times[0]) if len(scan_times) > 1 else 0.0,
        "median_scan_period_s": typical_period,
        "scan_gaps_over_3x_median": gap_count,
        "fault_label_source": "operator annotation; not verified by code",
    }
    return rows, quality


def main():
    if not RUN_DIR.exists():
        raise SystemExit(f"No run folder found: {RUN_DIR}")
    meta_files = sorted(RUN_DIR.glob("*.json"))
    if not meta_files:
        raise SystemExit(f"No run metadata found in {RUN_DIR}; record a run first.")
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows, quality_rows = [], []
    for path in meta_files:
        run_rows, quality = process_run(path, typestore)
        rows.extend(run_rows)
        quality_rows.append(quality)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    quality_frame = pd.DataFrame(quality_rows)
    quality_frame.to_csv(QUALITY_OUT, index=False)
    QUALITY_WEB_OUT.parent.mkdir(parents=True, exist_ok=True)
    quality_frame.to_csv(QUALITY_WEB_OUT, index=False)
    if not rows:
        raise SystemExit(f"No synchronized scan/odometry/command rows found. Sensor quality summary saved to {QUALITY_OUT}.")
    frame = pd.DataFrame(rows).sort_values(["run_id", "timestamp_s"])
    frame.to_csv(OUT, index=False)
    print(f"Wrote {len(frame):,} aligned rows from {frame.run_id.nunique()} independent run IDs: {OUT}")
    print(frame.groupby(["fault_condition", "action"]).size().unstack(fill_value=0).to_string())
    print(f"Wrote scan coverage and gap summary: {QUALITY_OUT}")


if __name__ == "__main__":
    main()
