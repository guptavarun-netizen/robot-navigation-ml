import math

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Vec4,
    LineSegs,
)


class RobotNavigationGame(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()
        self.setBackgroundColor(0.07, 0.09, 0.13, 1)

        self.robot_radius = 0.45
        self.heading = 0.0
        self.speed = 4.0
        self.turn_speed = 100.0

        self.show_lidar = True
        self.lidar_range = 7.0
        self.lidar_rays = 72
        self.collisions = 0

        self.keys = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
        }

        # Rectangular obstacles: x, y, half-width, half-depth
        self.obstacles = [
            (-3, 1, 1.6, 0.7),
            (2, 3, 0.7, 2.0),
            (3, -3, 2.0, 0.6),
            (-5, -4, 0.7, 2.0),
        ]

        self.setup_lighting()
        self.create_arena()
        self.create_robot()
        self.create_goal()
        self.setup_controls()
        self.setup_dashboard()

        self.taskMgr.add(self.update_game, "update_game")

    def setup_lighting(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.6, 0.6, 0.65, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.9, 0.9, 0.85, 1))
        sun_node = self.render.attachNewNode(sun)
        sun_node.setHpr(-35, -55, 0)
        self.render.setLight(sun_node)

    def add_box(self, position, scale, color, parent=None):
        box = self.loader.loadModel("models/box")
        box.reparentTo(parent or self.render)
        box.setPos(*position)
        box.setScale(*scale)
        box.setColor(*color)
        return box

    def create_arena(self):
        self.add_box(
            (0, 0, -0.25), (12, 12, 0.15),
            (0.22, 0.26, 0.31, 1)
        )

        # Grid lines
        for i in range(-10, 11, 2):
            self.add_box(
                (i, 0, -0.07), (0.025, 11, 0.015),
                (0.38, 0.42, 0.47, 1)
            )
            self.add_box(
                (0, i, -0.07), (11, 0.025, 0.015),
                (0.38, 0.42, 0.47, 1)
            )

        # Boundary walls
        self.add_box(
            (0, 11, 0.8), (11, 0.25, 0.8),
            (0.55, 0.58, 0.62, 1)
        )
        self.add_box(
            (0, -11, 0.8), (11, 0.25, 0.8),
            (0.55, 0.58, 0.62, 1)
        )
        self.add_box(
            (11, 0, 0.8), (0.25, 11, 0.8),
            (0.55, 0.58, 0.62, 1)
        )
        self.add_box(
            (-11, 0, 0.8), (0.25, 11, 0.8),
            (0.55, 0.58, 0.62, 1)
        )

        for x, y, hx, hy in self.obstacles:
            self.add_box(
                (x, y, 0.7), (hx, hy, 0.7),
                (0.68, 0.35, 0.23, 1)
            )

    def create_robot(self):
        self.robot = self.add_box(
            (-8, -8, 0.3), (0.45, 0.55, 0.28),
            (0.05, 0.65, 0.95, 1)
        )

        # Direction marker
        self.add_box(
            (0, 0.85, 0.3), (0.12, 0.35, 0.10),
            (1, 0.9, 0.1, 1), parent=self.robot
        )

    def create_goal(self):
        self.goal_position = (8.0, 8.0)

        self.goal = self.add_box(
            (8, 8, 0.45), (0.4, 0.4, 0.45),
            (0.1, 0.95, 0.25, 1)
        )

    def setup_controls(self):
        for key in ["w", "arrow_up"]:
            self.accept(key, self.set_key, ["forward", True])
            self.accept(f"{key}-up", self.set_key, ["forward", False])

        for key in ["s", "arrow_down"]:
            self.accept(key, self.set_key, ["backward", True])
            self.accept(f"{key}-up", self.set_key, ["backward", False])

        for key in ["a", "arrow_left"]:
            self.accept(key, self.set_key, ["left", True])
            self.accept(f"{key}-up", self.set_key, ["left", False])

        for key in ["d", "arrow_right"]:
            self.accept(key, self.set_key, ["right", True])
            self.accept(f"{key}-up", self.set_key, ["right", False])

        self.accept("l", self.toggle_lidar)
        self.accept("escape", self.userExit)

    def setup_dashboard(self):
        self.dashboard = OnscreenText(
            text="",
            pos=(-1.30, 0.91),
            scale=0.042,
            fg=(1, 1, 1, 1),
            align=0,
            mayChange=True,
        )

        self.instructions = OnscreenText(
            text="WASD/Arrows: Move | L: Toggle LiDAR | ESC: Quit",
            pos=(0, -0.94),
            scale=0.043,
            fg=(0.7, 0.9, 1, 1),
            mayChange=True,
        )

    def set_key(self, key, value):
        self.keys[key] = value

    def toggle_lidar(self):
        self.show_lidar = not self.show_lidar

    def get_position(self):
        p = self.robot.getPos()
        return p.x, p.y

    def is_collision(self, x, y):
        # Keep the robot inside the arena.
        if abs(x) > 10.0 or abs(y) > 10.0:
            return True

        # Approximate each obstacle as a rectangle expanded
        # by the robot radius.
        for ox, oy, hx, hy in self.obstacles:
            if (
                abs(x - ox) < hx + self.robot_radius
                and abs(y - oy) < hy + self.robot_radius
            ):
                return True

        return False

    def cast_ray(self, x, y, angle):
        step = 0.05
        distance = step

        while distance <= self.lidar_range:
            rx = x + math.cos(angle) * distance
            ry = y + math.sin(angle) * distance

            if self.is_collision(rx, ry):
                return distance

            distance += step

        return self.lidar_range

    def draw_lidar(self):
        x, y = self.get_position()

        lines = LineSegs("lidar")
        lines.setThickness(1.5)

        distances = []

        for i in range(self.lidar_rays):
            relative_angle = 2 * math.pi * i / self.lidar_rays
            angle = math.radians(self.heading) + relative_angle

            distance = self.cast_ray(x, y, angle)
            distances.append(distance)

            end_x = x + math.cos(angle) * distance
            end_y = y + math.sin(angle) * distance

            if distance < 1.5:
                lines.setColor(1, 0.15, 0.1, 1)
            else:
                lines.setColor(0.1, 0.9, 0.35, 1)

            lines.moveTo(x, y, 0.55)
            lines.drawTo(end_x, end_y, 0.08)

        node = self.render.attachNewNode(lines.create())
        return node, distances

    def update_game(self, task):
        dt = min(globalClock.getDt(), 0.05)

        if self.keys["left"]:
            self.heading += self.turn_speed * dt
        if self.keys["right"]:
            self.heading -= self.turn_speed * dt

        self.robot.setH(self.heading)

        forward = int(self.keys["forward"]) - int(self.keys["backward"])
        angle = math.radians(self.heading)

        dx = math.sin(angle) * self.speed * dt * forward
        dy = math.cos(angle) * self.speed * dt * forward

        x, y = self.get_position()

        # Resolve movement independently on each axis.
        blocked = False

        if not self.is_collision(x + dx, y):
            x += dx
        elif dx != 0:
            blocked = True

        if not self.is_collision(x, y + dy):
            y += dy
        elif dy != 0:
            blocked = True

        if blocked and forward != 0:
            self.collisions += 1

        self.robot.setPos(x, y, 0.3)

        # Update the following camera.
        self.camera.setPos(x, y - 22, 18)
        self.camera.lookAt(x, y, 0)

        # Rebuild LiDAR rays each frame.
        if hasattr(self, "lidar_node"):
            self.lidar_node.removeNode()

        self.lidar_node = None
        distances = []

        if self.show_lidar:
            self.lidar_node, distances = self.draw_lidar()

        nearest = min(distances) if distances else float("nan")
        detected = sum(d < 2.0 for d in distances)

        gx, gy = self.goal_position
        goal_distance = math.hypot(gx - x, gy - y)

        self.dashboard.setText(
            f"ROBOT NAVIGATION LAB\n"
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Heading: {self.heading % 360:.1f} deg\n"
            f"LiDAR rays: {self.lidar_rays if self.show_lidar else 0}\n"
            f"Nearest obstacle: {nearest:.2f} m\n"
            f"Rays detecting nearby obstacles: {detected}\n"
            f"Blocked movements: {self.collisions}\n"
            f"Goal distance: {goal_distance:.2f} m"
        )

        return Task.cont


game = RobotNavigationGame()
game.run()
