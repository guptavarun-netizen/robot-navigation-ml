"""Record one supervised TurtleBot3 navigation run and its goal metadata.

This uses ROS 1 rosbag and records only the project's existing TurtleBot3
topics. Sensor fault conditions are operator annotations; this program does
not disable or alter hardware.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "data" / "real_robot_runs"
TOPICS = ["/scan", "/odom", "/cmd_vel_rc100"]
FAULTS = ["normal", "lidar_blocked", "lidar_disconnected", "other"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-x", type=float, required=True, help="Goal x in the odom frame, metres")
    parser.add_argument("--goal-y", type=float, required=True, help="Goal y in the odom frame, metres")
    parser.add_argument("--fault-condition", choices=FAULTS, default="normal")
    parser.add_argument("--fault-notes", default="", help="Operator-recorded setup; do not infer a fault label")
    parser.add_argument("--operator", default="unknown")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_operator = re.sub(r"[^A-Za-z0-9_-]", "_", args.operator)[:32] or "operator"
    run_id = f"{stamp}_{safe_operator}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bag_path = (args.output_dir / f"{run_id}.bag").resolve()
    meta_path = args.output_dir / f"{run_id}.json"
    metadata = {
        "run_id": run_id,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator,
        "robot": "TurtleBot3 Waffle Pi",
        "goal_x_m": args.goal_x,
        "goal_y_m": args.goal_y,
        "fault_condition": args.fault_condition,
        "fault_notes": args.fault_notes,
        "topics": TOPICS,
        "bag_file": bag_path.name,
        "sensor_fault_is_operator_reported": True,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    command = ["rosbag", "record", "-O", str(bag_path), *TOPICS]
    print(f"Run ID: {run_id}")
    print(f"Metadata: {meta_path}")
    print("Start the robot's supervised goal run. Press Ctrl+C to stop recording.")
    print("Keep the robot in a secured test area with an operator at the emergency stop.")
    try:
        process = subprocess.Popen(command)
        try:
            process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
    except FileNotFoundError:
        meta_path.unlink(missing_ok=True)
        print("rosbag was not found. Source the ROS 1 environment, then retry.", file=sys.stderr)
        return 2
    print(f"Saved bag: {bag_path}")
    print("Next: python extract_robot_goal_runs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
