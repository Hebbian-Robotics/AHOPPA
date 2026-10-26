"""Private drawer-shutdown conditions, scoring tolerances and public instructions.

The scene, map and instruction are identical across conditions with the same
geometry. Only the private initial drawer positions differ; they are applied at
reset and never supplied to the model.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, StringConstraints, model_validator

from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel

CabinetIdentifier = Literal["A", "B", "C", "D"]
TwoCabinetCondition = Literal["none-open", "first-open", "second-open", "both-open"]
# Storage-round conditions name the open drawers, e.g. "open-B" or "open-A-D".
OpenSetCondition = Annotated[str, StringConstraints(pattern=r"^open(-[A-D]){1,2}$")]
ShutdownCondition = TwoCabinetCondition | OpenSetCondition
SHUTDOWN_CONDITIONS: tuple[TwoCabinetCondition, ...] = (
    "none-open",
    "first-open",
    "second-open",
    "both-open",
)
STORAGE_CABINETS: tuple[CabinetIdentifier, ...] = ("A", "B", "C", "D")
# None, each single drawer, and each pair: zero to two of four drawers open.
STORAGE_CONDITIONS: tuple[ShutdownCondition, ...] = (
    "none-open",
    *(f"open-{identifier}" for identifier in STORAGE_CABINETS),
    *(f"open-{first}-{second}" for first, second in itertools.combinations(STORAGE_CABINETS, 2)),
)

# NVIDIA's released Sektion cabinet, also used by Isaac Lab's cabinet tasks.
SEKTION_CABINET_USD = "Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
TASK_DRAWER_JOINT = "drawer_top_joint"
TASK_DRAWER_BODIES = ("drawer_top", "drawer_handle_top")
UNUSED_CABINET_JOINTS = ("drawer_bottom_joint", "door_left_joint", "door_right_joint")
# Measured from the composed Isaac 6.2 asset (metres, cabinet frame, front +X).
SEKTION_BASE_HALF_HEIGHT_M = 0.3841
SEKTION_CLOSED_FRONT_X_M = 0.3248
SEKTION_HANDLE_FRONT_X_M = 0.3628
SEKTION_HALF_WIDTH_M = 0.3819
SEKTION_BACK_X_M = -0.3050
SEKTION_TOP_DRAWER_FRONT_Z_M = (0.2553, 0.3791)
SEKTION_DRAWER_TRAVEL_M = 0.4
# Supported standing hold required by the return-home check.
MINIMUM_STANDING_ROOT_HEIGHT_M = 0.65
MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S = 0.1
MINIMUM_UPRIGHT_COMPONENT = 0.5


class ShutdownCabinet(ValidatedModel):
    identifier: CabinetIdentifier
    center_xy_m: ImmutableSequence[float] = Field(min_length=2, max_length=2)
    # 180 degrees turns the drawer front toward -X, facing the home area.
    yaw_degrees: float = 180.0
    initially_open: bool

    @property
    def base_center_m(self) -> tuple[float, float, float]:
        return (self.center_xy_m[0], self.center_xy_m[1], SEKTION_BASE_HALF_HEIGHT_M)

    def cabinet_to_scene_xy(self, x_m: float, y_m: float) -> tuple[float, float]:
        yaw = math.radians(self.yaw_degrees)
        return (
            self.center_xy_m[0] + math.cos(yaw) * x_m - math.sin(yaw) * y_m,
            self.center_xy_m[1] + math.sin(yaw) * x_m + math.cos(yaw) * y_m,
        )

    @property
    def outward_direction_xy(self) -> tuple[float, float]:
        yaw = math.radians(self.yaw_degrees)
        return (math.cos(yaw), math.sin(yaw))

    def drawer_front_xy_m(self, drawer_position_m: float) -> tuple[float, float]:
        return self.cabinet_to_scene_xy(SEKTION_CLOSED_FRONT_X_M + drawer_position_m, 0.0)


class ShutdownTask(ValidatedModel):
    # v2 adds the storage layout's drawer-viewed evidence; scoring is otherwise unchanged.
    protocol: Literal["drawer-shutdown-v1", "drawer-shutdown-v2"] = "drawer-shutdown-v1"
    condition: ShutdownCondition
    cabinets: ImmutableSequence[ShutdownCabinet] = Field(min_length=1, max_length=4)
    open_drawer_position_m: PositiveFloat = Field(default=0.2, lt=SEKTION_DRAWER_TRAVEL_M)
    # Passive slide damping retained from Isaac Lab's cabinet configuration.
    drawer_damping_n_s_per_m: PositiveFloat = 1.0
    # Dry slide friction (N), applied by Newton as MuJoCo Warp frictionloss. The
    # upstream asset has none; 5 N is mu=0.1 runner friction under the imported
    # 5.28 kg drawer and handle. Without it the drawer coasts and bounces back
    # off its nearly elastic closed limit.
    drawer_static_friction_n: float = Field(default=5.0, ge=0, le=20)
    # Closed-stop damping (N*s/m) authored as newton:limitDamping before import.
    # Newton's default 10 N*s/m against its 1e4 N/m limit stiffness is a damping
    # ratio near 0.02 for this drawer, so the stop rebounded almost elastically;
    # 460 is critical damping, 2*sqrt(1e4 * 5.28), a dead stop.
    drawer_limit_damping_n_s_per_m: PositiveFloat = 460.0
    closed_tolerance_m: PositiveFloat = Field(default=0.02, lt=0.1)
    settled_speed_m_per_s: PositiveFloat = 0.01
    disturbance_tolerance_m: PositiveFloat = Field(default=0.02, lt=0.1)
    hand_contact_threshold_n: PositiveFloat = 0.5
    hold_seconds: PositiveFloat = 1.0
    home_radius_m: PositiveFloat = 0.25

    @model_validator(mode="after")
    def require_condition_matching_drawers(self) -> ShutdownTask:
        identifiers = [cabinet.identifier for cabinet in self.cabinets]
        if identifiers != list(STORAGE_CABINETS[: len(identifiers)]):
            raise ValueError("Cabinets are ordered A, B, C, D")
        expected = condition_open_drawers(self.condition)
        if not expected <= set(identifiers):
            raise ValueError("The condition opens a drawer that is not in the scene")
        if {cabinet.identifier for cabinet in self.cabinets if cabinet.initially_open} != expected:
            raise ValueError("Initial drawer states differ from the named condition")
        if (self.protocol == "drawer-shutdown-v2") != (len(self.cabinets) == 4):
            raise ValueError("The four-cabinet storage round uses drawer-shutdown-v2")
        if self.closed_tolerance_m >= self.open_drawer_position_m / 2:
            raise ValueError("Closed tolerance must clearly separate open from closed")
        return self

    def cabinet(self, identifier: str) -> ShutdownCabinet:
        return next(cabinet for cabinet in self.cabinets if cabinet.identifier == identifier)

    def initial_drawer_position_m(self, cabinet: ShutdownCabinet) -> float:
        return self.open_drawer_position_m if cabinet.initially_open else 0.0

    @property
    def requires_manipulation(self) -> bool:
        return any(cabinet.initially_open for cabinet in self.cabinets)


def condition_open_drawers(condition: ShutdownCondition) -> set[str]:
    match condition:
        case "none-open":
            return set()
        case "first-open":
            return {"A"}
        case "second-open":
            return {"B"}
        case "both-open":
            return {"A", "B"}
        case _:
            return set(condition.split("-")[1:])


DEFAULT_CABINET_CENTERS_XY_M: tuple[tuple[float, float], ...] = ((2.9, 1.0), (2.9, -1.0))


def make_shutdown_task(condition: ShutdownCondition, *, cabinet_count: int = 2) -> ShutdownTask:
    """Fixed geometry; only the private initial drawer states vary by condition."""
    if cabinet_count not in {1, 2}:
        raise ValueError("The shutdown round uses one or two cabinets")
    open_drawers = condition_open_drawers(condition)
    return ShutdownTask(
        condition=condition,
        cabinets=[
            ShutdownCabinet(
                identifier=identifier,
                center_xy_m=list(center),
                initially_open=initially_open,
            )
            for identifier, center, initially_open in zip(
                ("A", "B"),
                DEFAULT_CABINET_CENTERS_XY_M,
                ("A" in open_drawers, "B" in open_drawers),
                strict=True,
            )
        ][:cabinet_count],
    )


# Storage round: cabinets back onto the walls, drawer fronts facing into the room.
# A and B stand on the east wall behind the locker block; C and D on the west wall
# behind home. Each back stops 2 cm inside the flat floor collider (scene x -2.12 to
# 4.95): cabinets standing on the curved skirting mesh coincided with robot falls.
STORAGE_CABINET_LAYOUT: dict[CabinetIdentifier, tuple[tuple[float, float], float]] = {
    "A": ((4.623, 1.5), 180.0),
    "B": ((4.623, -1.5), 180.0),
    "C": ((-1.795, 2.0), 0.0),
    "D": ((-1.795, -2.0), 0.0),
}


def make_storage_round_task(condition: ShutdownCondition) -> ShutdownTask:
    """Four wall cabinets; only which 0-2 drawers start open varies by condition."""
    open_drawers = condition_open_drawers(condition)
    return ShutdownTask(
        protocol="drawer-shutdown-v2",
        condition=condition,
        cabinets=[
            ShutdownCabinet(
                identifier=identifier,
                center_xy_m=list(center),
                yaw_degrees=yaw,
                initially_open=identifier in open_drawers,
            )
            for identifier, (center, yaw) in STORAGE_CABINET_LAYOUT.items()
        ],
    )


def balanced_condition_schedule(
    repetitions: int,
    seed: int,
    conditions: tuple[ShutdownCondition, ...] = SHUTDOWN_CONDITIONS,
) -> list[ShutdownCondition]:
    """Each repetition block contains every condition once in a seeded order."""
    if repetitions < 1:
        raise ValueError("A schedule requires at least one repetition")
    generator = random.Random(seed)
    schedule: list[ShutdownCondition] = []
    for _ in range(repetitions):
        block = list(conditions)
        generator.shuffle(block)
        schedule.extend(block)
    return schedule


def shutdown_instruction(task: ShutdownTask) -> str:
    names = [f"cabinet {cabinet.identifier}" for cabinet in task.cabinets]
    cabinet_names = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (
        f"End-of-shift shutdown round. Check the top drawer of {cabinet_names} shown on the "
        "facility map. Close any of these drawers that are open by pushing the drawer front "
        "with your hand. Leave closed drawers alone. Then return to the home position marked "
        "on the map and declare completion. Some, all or none of the drawers may be open; the "
        "map does not show drawer states. Use the cameras to check each drawer; you may verify "
        "a drawer from a distance if it is clearly visible. "
        + (
            "Drawer fronts face into the room from the walls; some cannot be seen from home "
            "and are hidden behind the storage lockers, so use the map to plan where to look. "
            if task.protocol == "drawer-shutdown-v2"
            else ""
        )
        + f"A drawer counts as closed when it is within {task.closed_tolerance_m * 100:g} cm of "
        "fully closed and has stopped moving. Push only with the hand: do not use the torso "
        "or other body parts, and do not pull drawers open. "
        f"Home means the point on the floor directly below the pelvis is within "
        f"{task.home_radius_m:g} m of the initial position, standing upright and still for "
        f"{task.hold_seconds:g} seconds: pelvis height at least 0.65 m, pelvis up within 60 "
        "degrees of world up, horizontal and absolute vertical speed at most 0.1 m/s, both feet "
        "supported, knees off the floor, and both knee angles at most 0.5 rad. "
        "No task-state, drawer-position or object-coordinate tool is available. Move around "
        "furniture rather than through it. You choose the route and all SONIC motion targets; "
        "no automatic navigation, drawer-closing or inspection skill is supplied."
    )
