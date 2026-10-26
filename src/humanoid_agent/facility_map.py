"""Render the static facility map supplied with every shutdown-round decision.

The map is derived from the frozen scene geometry. It shows the floor, cabinets,
drawer sides, home, initial heading and scale; it never shows drawer states or a
live robot position.
"""

from __future__ import annotations

import math
from pathlib import Path

from humanoid_agent.isaac_workcell import (
    SIMPLE_ROOM_FLOOR_BOUNDS_XY_M,
    SIMPLE_ROOM_RIM_BOUNDS_XY_M,
    GroundWorkcellScene,
)
from humanoid_agent.shutdown_task import (
    SEKTION_BACK_X_M,
    SEKTION_CLOSED_FRONT_X_M,
    SEKTION_HALF_WIDTH_M,
)

FACILITY_MAP_FILENAME = "facility-map.png"
FACILITY_MAP_DESCRIPTION = (
    "Static facility map from a prior survey; identical in every condition. Coordinates are "
    "metres in a fixed floor frame: the origin is home, +x is the robot's initial heading and "
    "+y is to its initial left. Grey rectangles are cabinets; the thick edge marks the drawer "
    "front. The map does not show whether drawers are open or where the robot is now."
)


def scene_floor_corners_xy(scene: GroundWorkcellScene) -> list[tuple[float, float]]:
    room = scene.room_asset
    if room is None:
        raise ValueError("The facility map requires the released room floor")
    yaw = math.radians(room.yaw_degrees)
    (minimum_x, minimum_y), (maximum_x, maximum_y) = (
        SIMPLE_ROOM_RIM_BOUNDS_XY_M
        if scene.shutdown.protocol == "drawer-shutdown-v2"
        else SIMPLE_ROOM_FLOOR_BOUNDS_XY_M
    )
    return [
        (
            room.position_m[0] + math.cos(yaw) * x - math.sin(yaw) * y,
            room.position_m[1] + math.sin(yaw) * x + math.cos(yaw) * y,
        )
        for x, y in (
            (minimum_x, minimum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
            (minimum_x, maximum_y),
        )
    ]


def render_facility_map(scene: GroundWorkcellScene, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.patches import Circle, FancyArrow, Polygon

    task = scene.shutdown
    home_x, home_y = scene.robot_start_xy_m
    figure, axes = pyplot.subplots(figsize=(7, 7), dpi=110)
    floor = [(x - home_x, y - home_y) for x, y in scene_floor_corners_xy(scene)]
    axes.add_patch(Polygon(floor, closed=True, facecolor="#f3ead9", edgecolor="#555555", lw=2))
    for box in scene.boxes:
        if box.finish != "hidden":
            continue
        half_x, half_y = box.full_size_m[0] / 2, box.full_size_m[1] / 2
        center_x, center_y = box.center_m[0] - home_x, box.center_m[1] - home_y
        axes.add_patch(
            Polygon(
                [
                    (center_x - half_x, center_y - half_y),
                    (center_x + half_x, center_y - half_y),
                    (center_x + half_x, center_y + half_y),
                    (center_x - half_x, center_y + half_y),
                ],
                closed=True,
                facecolor="#5d6d7e",
                edgecolor="#2c3e50",
            )
        )
    storage = [box for box in scene.boxes if box.finish == "hidden"]
    if storage:
        middle_x = sum(box.center_m[0] for box in storage) / len(storage) - home_x
        axes.text(
            middle_x,
            0,
            "STORAGE LOCKERS (2.1 m tall)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    for cabinet in task.cabinets:
        corners = [
            cabinet.cabinet_to_scene_xy(x, y)
            for x, y in (
                (SEKTION_BACK_X_M, -SEKTION_HALF_WIDTH_M),
                (SEKTION_CLOSED_FRONT_X_M, -SEKTION_HALF_WIDTH_M),
                (SEKTION_CLOSED_FRONT_X_M, SEKTION_HALF_WIDTH_M),
                (SEKTION_BACK_X_M, SEKTION_HALF_WIDTH_M),
            )
        ]
        local = [(x - home_x, y - home_y) for x, y in corners]
        axes.add_patch(Polygon(local, closed=True, facecolor="#9a9a9a", edgecolor="#333333"))
        axes.plot(*zip(local[1], local[2], strict=True), color="#c0392b", lw=4)
        center_y = cabinet.center_xy_m[1]
        axes.text(
            max(x for x, _ in local) + 0.1
            if cabinet.outward_direction_xy[0] >= 0
            else min(x for x, _ in local) - 0.1,
            center_y - home_y,
            f"Cabinet {cabinet.identifier}",
            ha="left" if cabinet.outward_direction_xy[0] >= 0 else "right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    axes.add_patch(Circle((0, 0), task.home_radius_m, facecolor="#2e86c1", alpha=0.35))
    axes.add_patch(FancyArrow(0, 0, 0.6, 0, width=0.05, color="#1b4f72"))
    axes.text(0, -0.45, "HOME\n(initial heading →)", ha="center", va="top", fontsize=10)
    axes.plot([-1.5, -0.5], [-3.4, -3.4], color="black", lw=3)
    axes.text(-1.0, -3.3, "1 m", ha="center", va="bottom", fontsize=10)
    axes.set_aspect("equal")
    xs = [x for x, _ in floor]
    ys = [y for _, y in floor]
    axes.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    axes.set_ylim(min(ys) - 0.2, max(ys) + 0.2)
    axes.set_xticks(range(math.ceil(min(xs)), math.floor(max(xs)) + 1))
    axes.set_yticks(range(math.ceil(min(ys)), math.floor(max(ys)) + 1))
    axes.grid(True, color="#cccccc", lw=0.6)
    axes.set_xlabel("x (m), initial heading")
    axes.set_ylabel("y (m), initial left")
    axes.set_title("Storage bay map — drawer front in red; drawer states not shown")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="png", metadata={"Software": None})
    pyplot.close(figure)
