import csv
import math
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

BAG = Path("data/raw/turtlebot3_alternative/TB3_WAFFLE_SLAM.bag")
OUT = Path("data/processed/turtlebot3")
OUT.mkdir(parents=True, exist_ok=True)

typestore = get_typestore(Stores.ROS1_NOETIC)

scan_rows = []
odom_rows = []
cmd_rows = []

with Reader(BAG) as reader:
    available = {c.topic for c in reader.connections}

    print("Available topics:")
    for topic in sorted(available):
        print(" ", topic)

    connections = [
        c for c in reader.connections
        if c.topic in {"/scan", "/odom", "/cmd_vel_rc100"}
    ]

    for connection, timestamp, rawdata in reader.messages(
        connections=connections
    ):
        msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        time_s = timestamp / 1e9

        if connection.topic == "/scan":
            ranges = [
                float(r) for r in msg.ranges
                if math.isfinite(r)
                and msg.range_min <= r <= msg.range_max
            ]

            scan_rows.append({
                "timestamp_s": time_s,
                "beam_count": len(msg.ranges),
                "valid_beams": len(ranges),
                "min_distance_m": min(ranges) if ranges else float("nan"),
                "mean_distance_m": (
                    sum(ranges) / len(ranges) if ranges else float("nan")
                ),
                "angle_min_rad": msg.angle_min,
                "angle_increment_rad": msg.angle_increment,
                "range_min_m": msg.range_min,
                "range_max_m": msg.range_max,
            })

        elif connection.topic == "/odom":
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            w = msg.twist.twist.angular

            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z),
            )

            odom_rows.append({
                "timestamp_s": time_s,
                "x_m": p.x,
                "y_m": p.y,
                "yaw_rad": yaw,
                "linear_velocity_x_m_s": v.x,
                "linear_velocity_y_m_s": v.y,
                "angular_velocity_z_rad_s": w.z,
            })

        elif connection.topic == "/cmd_vel_rc100":
            cmd_rows.append({
                "timestamp_s": time_s,
                "linear_x_m_s": msg.linear.x,
                "angular_z_rad_s": msg.angular.z,
            })


def write_csv(filename, rows):
    if not rows:
        print(f"WARNING: No messages extracted for {filename}")
        return

    path = OUT / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}: {len(rows)} rows")


write_csv("scan_summary.csv", scan_rows)
write_csv("odometry.csv", odom_rows)
write_csv("velocity_commands.csv", cmd_rows)

print("\nExtraction complete.")