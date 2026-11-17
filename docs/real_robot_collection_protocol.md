# Real robot run and sensor-fault collection

This workflow records data; it does not control the robot or inject faults.
Only a trained operator should run a robot. Use an open, controlled test area,
low speed, an unobstructed emergency stop, and a safety observer. Do not
disconnect or obstruct a sensor while the robot is moving. Perform any
hardware fault condition with drive motors disabled first; only conduct a
faulted navigation run if the robot owner has approved a safe test procedure.

## Collect independent goal demonstrations

1. Bring up the TurtleBot3 ROS 1 topics `/scan`, `/odom`, and `/cmd_vel_rc100`.
2. Choose one reachable goal in the `odom` frame and record its x/y in metres.
3. Record a new run with a separate rosbag and metadata file:

   ```bash
   python record_robot_trial.py --goal-x 1.5 --goal-y 0.8 --operator student1
   ```

4. Drive the robot to that goal under supervision. Press Ctrl+C only after the
   run ends. Repeat for different goals/routes; every invocation creates a
   separate run ID. Do not copy or rename the same bag to simulate more runs.
5. Convert scan/odometry/command messages into aligned rows and per-run scan
   coverage/gap summaries:

   ```bash
   python extract_robot_goal_runs.py
   ```

6. After at least two genuinely independent runs, train and evaluate on
   held-out run IDs:

   ```bash
   python evaluate_real_goal_runs.py
   ```

The feature vector contains 24 LiDAR sector minima and the goal offset in the
robot frame. Labels are coarse actions derived from the recorded velocity
command. The group split keeps all rows from a run together. It evaluates
command imitation on held-out runs; it does not by itself establish safe
closed-loop goal completion. The goal coordinate is operator-entered and must
match the odometry frame used by the bag.

## Record a sensor condition

`record_robot_trial.py` accepts `--fault-condition normal|lidar_blocked|
lidar_disconnected|other` plus `--fault-notes`. The annotation is the
operator's record of the physical setup; the script cannot verify that the
sensor fault occurred. Include both normal and fault-condition runs, with
separate run IDs and goals. Do not label software-masked channels as a physical
fault. A disconnected LiDAR may publish no scans; that absence is itself part
of the bag and will not produce ordinary scan-feature rows.

`data/real_robot_runs/run_sensor_quality.csv` records scan count, median scan
period, aligned-row count, and large scan gaps by run. A gap is a diagnostic
signal only: it does not prove a hardware fault, since network, recording, or
clock problems can produce the same symptom.

For a defensible fault study, retain the original bags and write down the
intervention, start/end times, safety setup, scan frequency, number of missing
scans, and navigation outcome. Keep motors disabled for sensor obstruction or
disconnection unless the robot owner approves a separate controlled mobility
test. Report each fault condition separately; never merge faulted rows into
normal training without explicitly describing that design.

## Public demonstration dataset

SCAND is a complementary public dataset with 138 real robot trajectories,
LiDAR, joystick commands, and odometry from Jackal and Spot robots. Its
trajectories can support run-held-out imitation experiments, but the dataset's
task is socially compliant teleoperation, not this project's exact
goal-conditioned obstacle task. Dataset downloads are hosted by the Texas Data
Repository; access can depend on that service. See the [official dataset
page](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html).
