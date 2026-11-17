"""Click-to-set-goal 2D robot simulator using a saved sklearn classifier.

Run from project root:
    python simulator.py

Controls:
    Left-click free space: start a new trial at the fixed initial position
    SPACE: pause/resume | S: toggle sensors | R: reset robot | ESC: quit

Research note:
    Sensor scaling and class-to-action mapping are provisional. The safety-heading
    logic can independently prevent collisions, so this is a proof-of-concept,
    not a validated physical robot controller.
"""

from pathlib import Path
import csv
import math
import time

import joblib
import numpy as np
import pandas as pd
import pygame


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1100, 760
FPS = 60

ROBOT_RADIUS = 13
SENSOR_COUNT = 24
MAX_SENSOR_DISTANCE = 180
SENSOR_STEP_DEG = 360 / SENSOR_COUNT

ROBOT_SPEED = 95.0
TURN_SPEED = 2.0
GOAL_RADIUS = 20

# Maximum duration for a single trial.
MAX_TRIAL_TIME_S = 60.0


# ============================================================
# 2. CLASS-TO-ACTION MAPPING
# ============================================================

# Tuple format: (linear speed multiplier, turn direction).
# Turn direction: -1 = right, 0 = straight, +1 = left.
#
# These speeds and turns are heuristic controller settings.
# They are not learned directly from the UCI dataset.

ACTION_MAP = {
    "1": (1.00, 0),     # Move Forward
    "2": (0.65, -1),    # Slight Right Turn
    "3": (0.30, -1),    # Sharp Right Turn
    "4": (0.65, +1),    # Slight Left Turn
}


# ============================================================
# 3. ENVIRONMENT AND OBSTACLES
# ============================================================

WALLS = [
    pygame.Rect(45, 45, WIDTH - 90, 18),
    pygame.Rect(45, HEIGHT - 63, WIDTH - 90, 18),
    pygame.Rect(45, 45, 18, HEIGHT - 90),
    pygame.Rect(WIDTH - 63, 45, 18, HEIGHT - 90),

    pygame.Rect(280, 180, 25, 260),
    pygame.Rect(500, 330, 250, 25),
    pygame.Rect(830, 150, 25, 250),
]

SENSOR_COLUMNS = [f"V{i}" for i in range(1, SENSOR_COUNT + 1)]


# ============================================================
# 4. LOAD THE TRAINED MACHINE-LEARNING MODEL
# ============================================================

def load_model():
    """Load the saved sklearn model and validate its feature count."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run `python train_models.py` first."
        )

    model = joblib.load(MODEL_PATH)

    expected = getattr(model, "n_features_in_", None)

    if expected is not None and expected != SENSOR_COUNT:
        raise ValueError(
            f"Model expects {expected} features; "
            f"simulator creates {SENSOR_COUNT}."
        )

    return model


# ============================================================
# 5. SIMULATED ROBOT SENSORS
# ============================================================

def ray_distance(origin, angle, walls):
    """Return the distance to the first obstacle along a sensor ray."""

    ox, oy = origin

    dx = math.cos(angle)
    dy = math.sin(angle)

    for distance in range(
        0,
        MAX_SENSOR_DISTANCE + 1,
        2
    ):
        x = int(ox + dx * distance)
        y = int(oy + dy * distance)

        if any(
            wall.collidepoint(x, y)
            for wall in walls
        ):
            return float(distance)

    return float(MAX_SENSOR_DISTANCE)


def sense_robot(position, heading, walls):
    """Generate 24 synthetic sensor readings and their visual rays."""

    values = []
    rays = []

    for i in range(SENSOR_COUNT):

        angle = (
            heading
            + math.radians(i * SENSOR_STEP_DEG)
        )

        distance_px = ray_distance(
            position,
            angle,
            walls
        )

        rays.append((angle, distance_px))

        # Provisional mapping from pixel distance to sensor value.
        # This does not reproduce the calibration of the UCI sensors.
        sensor_value = (
            0.4
            + 4.6 * min(
                distance_px / MAX_SENSOR_DISTANCE,
                1.0
            )
        )

        values.append(sensor_value)

    return (
        np.asarray(values, dtype=float).reshape(1, -1),
        rays
    )


# ============================================================
# 6. COLLISION DETECTION
# ============================================================

def collides(position, walls):
    """Check whether the robot's circular footprint intersects a wall."""

    x, y = position

    robot_rect = pygame.Rect(
        round(x - ROBOT_RADIUS),
        round(y - ROBOT_RADIUS),
        ROBOT_RADIUS * 2,
        ROBOT_RADIUS * 2,
    )

    return any(
        robot_rect.colliderect(wall)
        for wall in walls
    )


# ============================================================
# 7. PREDICTION INPUT PREPARATION
# ============================================================

def label_key(label):
    """Convert a model prediction into a string label."""

    if isinstance(label, bytes):
        label = label.decode("utf-8")

    return str(label).strip()


def dataframe_for_model(sensor_values, model):
    """Create a DataFrame with the feature names expected by the model."""

    values = np.asarray(
        sensor_values,
        dtype=float
    ).reshape(1, -1)

    columns = list(
        getattr(
            model,
            "feature_names_in_",
            SENSOR_COLUMNS
        )
    )

    if len(columns) != values.shape[1]:
        raise ValueError(
            f"Model expects {len(columns)} features; "
            f"got {values.shape[1]}."
        )

    return pd.DataFrame(
        values,
        columns=columns
    )


# ============================================================
# 8. VISUALIZATION HELPER
# ============================================================

def draw_text(
    screen,
    font,
    text,
    x,
    y,
    color=(235, 239, 245)
):
    """Draw text on the simulator screen."""

    screen.blit(
        font.render(text, True, color),
        (x, y)
    )


# ============================================================
# 9. GOAL VALIDATION
# ============================================================

def valid_goal(point, walls):
    """Check that the selected goal is in a valid free-space location."""

    x, y = point

    margin = ROBOT_RADIUS + 5

    if not (
        45 + margin <= x <= WIDTH - 45 - margin
    ):
        return False

    if not (
        45 + margin <= y <= HEIGHT - 45 - margin
    ):
        return False

    # Check internal obstacles.
    for wall in walls[4:]:

        expanded_wall = wall.inflate(
            2 * margin,
            2 * margin
        )

        if expanded_wall.collidepoint(x, y):
            return False

    return True


# ============================================================
# 10. SAFETY-HEADING LOGIC
# ============================================================

def safe_candidate(position, heading, speed, dt, walls):
    """Check the proposed short path for potential wall intersections."""

    travel = speed * dt

    samples = max(
        2,
        math.ceil(travel / 3)
    )

    for i in range(1, samples + 1):

        fraction = i / samples

        candidate = (
            position[0]
            + math.cos(heading) * travel * fraction,

            position[1]
            + math.sin(heading) * travel * fraction,
        )

        if collides(candidate, walls):
            return False

    return True


def choose_safe_heading(
    position,
    desired_heading,
    speed,
    dt,
    walls
):
    """Choose the first candidate heading whose short path is safe."""

    offsets = [
        0,
        math.radians(20),
        -math.radians(20),
        math.radians(40),
        -math.radians(40),
        math.radians(60),
        -math.radians(60),
        math.radians(90),
        -math.radians(90),
        math.radians(135),
        -math.radians(135),
        math.pi,
    ]

    for offset in offsets:

        candidate_heading = (
            desired_heading + offset
        )

        if safe_candidate(
            position,
            candidate_heading,
            speed,
            dt,
            walls
        ):
            return candidate_heading, True

    return desired_heading, False


# ============================================================
# 11. MAIN SIMULATION
# ============================================================

def main():

    model = load_model()

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "Classical ML Robot Navigation — Controlled Trials"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "monospace",
        17
    )

    small_font = pygame.font.SysFont(
        "monospace",
        14
    )

    # --------------------------------------------------------
    # Initial robot state
    # --------------------------------------------------------

    initial_position = [115.0, 115.0]

    position = initial_position[:]
    heading = 0.0

    goal = None
    paused = False
    show_sensors = True

    # --------------------------------------------------------
    # Trial variables
    # --------------------------------------------------------

    trial_id = 0

    collisions = 0
    steps = 0

    goal_reached = False
    trial_finished = False
    trial_status = "WAITING"

    last_label = "—"
    latency_ms = 0.0
    path_length = 0.0

    start_time = None
    trial_elapsed_s = 0.0

    inference_times = []

    # --------------------------------------------------------
    # CSV logging setup
    # --------------------------------------------------------

    log_path = (
        RESULTS_DIR / "simulation_log.csv"
    )

    # A new simulator session overwrites the previous log.
    # Back up the previous CSV first if you need to preserve it.

    log_file = open(
        log_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    logger = csv.writer(log_file)

    logger.writerow([
        "trial_id",
        "time_s",
        "x_px",
        "y_px",
        "heading_rad",
        "goal_x_px",
        "goal_y_px",
        "predicted_class",
        "inference_ms",
        "collision_count",
        "path_length_px",
        "goal_reached",
        "trial_status",
    ])

    # --------------------------------------------------------
    # Reset helper for a new independent trial
    # --------------------------------------------------------

    def reset_trial_state():

        nonlocal position
        nonlocal heading
        nonlocal collisions
        nonlocal steps
        nonlocal goal_reached
        nonlocal trial_finished
        nonlocal trial_status
        nonlocal last_label
        nonlocal latency_ms
        nonlocal path_length
        nonlocal start_time
        nonlocal trial_elapsed_s
        nonlocal inference_times

        position = initial_position[:]
        heading = 0.0

        collisions = 0
        steps = 0

        goal_reached = False
        trial_finished = False
        trial_status = "RUNNING"

        last_label = "—"
        latency_ms = 0.0
        path_length = 0.0

        start_time = time.perf_counter()
        trial_elapsed_s = 0.0

        inference_times = []

    # --------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------

    running = True

    try:

        while running:

            dt = min(
                clock.tick(FPS) / 1000.0,
                0.05
            )

            # =================================================
            # 12. EVENT HANDLING
            # =================================================

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                # ---------------------------------------------
                # Click to start a new trial
                # ---------------------------------------------

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                ):

                    if (
                        valid_goal(event.pos, WALLS)
                        and not collides(event.pos, WALLS)
                    ):

                        goal = [
                            float(event.pos[0]),
                            float(event.pos[1])
                        ]

                        trial_id += 1

                        # Start each trial from identical conditions.
                        reset_trial_state()

                # ---------------------------------------------
                # Keyboard controls
                # ---------------------------------------------

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors

                    elif event.key == pygame.K_r:

                        # Reset the robot and wait for a new goal.
                        position = initial_position[:]
                        heading = 0.0

                        goal = None
                        paused = False

                        collisions = 0
                        steps = 0

                        goal_reached = False
                        trial_finished = False
                        trial_status = "WAITING"

                        last_label = "—"
                        latency_ms = 0.0
                        path_length = 0.0

                        start_time = None
                        trial_elapsed_s = 0.0

                        inference_times = []

            # =================================================
            # 13. MODEL PREDICTION AND ROBOT MOVEMENT
            # =================================================

            if (
                not paused
                and goal is not None
                and not trial_finished
            ):

                if start_time is None:
                    start_time = time.perf_counter()

                trial_elapsed_s = (
                    time.perf_counter() - start_time
                )

                # Generate the robot's synthetic sensor readings.
                features, _ = sense_robot(
                    position,
                    heading,
                    WALLS
                )

                # Prepare the model input.
                sensor_df = dataframe_for_model(
                    features,
                    model
                )

                # Measure inference latency.
                t0 = time.perf_counter()

                prediction = model.predict(
                    sensor_df
                )[0]

                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                inference_times.append(
                    latency_ms
                )

                last_label = label_key(
                    prediction
                )

                # Obtain the action associated with the predicted class.
                speed_mult, ml_turn = ACTION_MAP.get(
                    last_label,
                    (0.0, 0)
                )

                # ---------------------------------------------
                # Calculate desired direction toward the goal
                # ---------------------------------------------

                goal_heading = math.atan2(
                    goal[1] - position[1],
                    goal[0] - position[0]
                )

                goal_error = math.atan2(
                    math.sin(goal_heading - heading),
                    math.cos(goal_heading - heading)
                )

                ml_turn_delta = (
                    ml_turn * TURN_SPEED * dt
                )

                desired_heading = (
                    heading
                    + 0.75 * goal_error
                    + 0.25 * ml_turn_delta
                )

                speed = (
                    ROBOT_SPEED
                    * max(speed_mult, 0.35)
                )

                # ---------------------------------------------
                # Apply safety-heading logic
                # ---------------------------------------------

                safe_heading, can_move = (
                    choose_safe_heading(
                        position,
                        desired_heading,
                        speed,
                        dt,
                        WALLS
                    )
                )

                # ---------------------------------------------
                # Update robot position
                # ---------------------------------------------

                if can_move:

                    old_position = position[:]

                    new_position = [
                        position[0]
                        + math.cos(safe_heading) * speed * dt,

                        position[1]
                        + math.sin(safe_heading) * speed * dt,
                    ]

                    if collides(new_position, WALLS):

                        collisions += 1

                    else:

                        position = new_position
                        heading = safe_heading

                        path_length += math.dist(
                            old_position,
                            position
                        )

                else:

                    # Rotate slowly if no candidate heading is safe.
                    heading += (
                        math.radians(30) * dt
                    )

                steps += 1

                trial_elapsed_s = (
                    time.perf_counter() - start_time
                )

                # ---------------------------------------------
                # Determine trial outcome
                # ---------------------------------------------

                if (
                    math.dist(position, goal)
                    <= GOAL_RADIUS
                ):

                    goal_reached = True
                    trial_finished = True
                    trial_status = "SUCCESS"

                elif (
                    trial_elapsed_s
                    >= MAX_TRIAL_TIME_S
                ):

                    trial_finished = True
                    trial_status = "TIMEOUT"

                # ---------------------------------------------
                # Log this simulation step
                # ---------------------------------------------

                logger.writerow([
                    trial_id,
                    f"{trial_elapsed_s:.3f}",
                    f"{position[0]:.2f}",
                    f"{position[1]:.2f}",
                    f"{heading:.4f}",
                    f"{goal[0]:.1f}",
                    f"{goal[1]:.1f}",
                    last_label,
                    f"{latency_ms:.4f}",
                    collisions,
                    f"{path_length:.2f}",
                    int(goal_reached),
                    (
                        trial_status
                        if trial_finished
                        else "RUNNING"
                    ),
                ])

                log_file.flush()

            elif (
                goal is not None
                and start_time is not None
            ):

                trial_elapsed_s = (
                    time.perf_counter() - start_time
                )

            # =================================================
            # 14. DRAW ENVIRONMENT
            # =================================================

            _, rays = sense_robot(
                position,
                heading,
                WALLS
            )

            screen.fill(
                (20, 25, 34)
            )

            # Draw walls.
            for wall in WALLS:

                pygame.draw.rect(
                    screen,
                    (120, 132, 150),
                    wall,
                    border_radius=3
                )

            # ---------------------------------------------
            # Draw sensor rays
            # ---------------------------------------------

            if show_sensors:

                for angle, distance in rays:

                    end = (
                        int(
                            position[0]
                            + math.cos(angle) * distance
                        ),

                        int(
                            position[1]
                            + math.sin(angle) * distance
                        ),
                    )

                    pygame.draw.line(
                        screen,
                        (65, 170, 190),
                        (
                            int(position[0]),
                            int(position[1])
                        ),
                        end,
                        1
                    )

                    pygame.draw.circle(
                        screen,
                        (245, 190, 80),
                        end,
                        2
                    )

            # ---------------------------------------------
            # Draw goal
            # ---------------------------------------------

            if goal is not None:

                gc = (
                    int(goal[0]),
                    int(goal[1])
                )

                pygame.draw.circle(
                    screen,
                    (245, 195, 70),
                    gc,
                    GOAL_RADIUS,
                    2
                )

                pygame.draw.circle(
                    screen,
                    (245, 195, 70),
                    gc,
                    4
                )

                pygame.draw.line(
                    screen,
                    (245, 195, 70),
                    (
                        int(position[0]),
                        int(position[1])
                    ),
                    gc,
                    1
                )

            # ---------------------------------------------
            # Draw robot body and heading
            # ---------------------------------------------

            center = (
                int(position[0]),
                int(position[1])
            )

            pygame.draw.circle(
                screen,
                (75, 190, 125),
                center,
                ROBOT_RADIUS
            )

            pygame.draw.circle(
                screen,
                (235, 245, 240),
                center,
                ROBOT_RADIUS,
                2
            )

            nose = (
                int(
                    position[0]
                    + math.cos(heading) * 24
                ),

                int(
                    position[1]
                    + math.sin(heading) * 24
                ),
            )

            pygame.draw.line(
                screen,
                (255, 255, 255),
                center,
                nose,
                4
            )

            # =================================================
            # 15. STATUS DISPLAY
            # =================================================

            if goal is None:

                distance_text = "Goal: click free space"

            else:

                distance_text = (
                    f"Goal distance: "
                    f"{math.dist(position, goal):.1f}px"
                )

            if goal is None:

                status = "WAITING FOR GOAL"

            elif paused:

                status = "PAUSED"

            elif trial_status == "SUCCESS":

                status = "GOAL REACHED"

            elif trial_status == "TIMEOUT":

                status = "TRIAL TIMEOUT"

            elif trial_finished:

                status = "TRIAL FINISHED"

            else:

                status = "RUNNING"

            pygame.draw.rect(
                screen,
                (12, 16, 23),
                (70, 75, 550, 215),
                border_radius=8
            )

            draw_text(
                screen,
                font,
                "CLASSICAL ML ROBOT NAVIGATION",
                84,
                88
            )

            draw_text(
                screen,
                small_font,
                f"Trial ID: {trial_id}",
                84,
                115
            )

            draw_text(
                screen,
                small_font,
                f"Predicted class: {last_label}",
                84,
                138
            )

            draw_text(
                screen,
                small_font,
                f"Inference: {latency_ms:.3f} ms",
                84,
                161
            )

            draw_text(
                screen,
                small_font,
                f"Collisions: {collisions} | Steps: {steps}",
                84,
                184
            )

            draw_text(
                screen,
                small_font,
                f"Path length: {path_length:.1f}px",
                84,
                207
            )

            draw_text(
                screen,
                small_font,
                distance_text,
                84,
                230
            )

            draw_text(
                screen,
                small_font,
                (
                    f"Elapsed: {trial_elapsed_s:.2f}s"
                    f" | Status: {status}"
                ),
                84,
                253
            )

            draw_text(
                screen,
                small_font,
                (
                    "Click goal | SPACE pause | "
                    "S sensors | R reset | ESC quit"
                ),
                70,
                HEIGHT - 32,
                (200, 210, 220)
            )

            pygame.display.flip()

    finally:

        log_file.close()
        pygame.quit()


if __name__ == "__main__":
    main()