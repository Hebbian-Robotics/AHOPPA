"""Instantiate the drawer-shutdown room, cabinets and fixed marker geometry in Isaac.

Scene coordinates configure physics and rendering. This module is not part of
the model observation interface.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveFloat, model_validator

from humanoid_agent.shutdown_task import ShutdownTask

Position = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Dimensions = Annotated[list[PositiveFloat], Field(min_length=3, max_length=3)]
Color = Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=3, max_length=3)]
WORKCELL_GROUND_HALF_EXTENT_M = 10.0
# "hidden" boxes are invisible colliders standing in for visual-only furniture.
SurfaceFinish = Literal["painted", "hidden"]

WorkcellAssetName = Literal[
    "simple_room",
    "simple_room_without_table",
    "simready_locker_double_bench",
]
ROOM_ASSETS: frozenset[str] = frozenset({"simple_room", "simple_room_without_table"})
# NVIDIA SimReady Central (huggingface.co/datasets/nvidia/simready-assets, CC-BY-4.0),
# revision 524747d9, copied to each host. Loaded visual-only; scene boxes collide.
SIMREADY_ASSET_REVISION = "524747d9b5e1f21b5b18b56041788c01d745f1a1"
SIMREADY_ASSET_PATHS: dict[str, str] = {
    "simready_locker_double_bench": (
        "storage_locker_double_bench_b03/simready_usd/sm_storage_locker_double_bench_b03_01.usd"
    ),
}
SIMREADY_VISUAL_ONLY_VARIANTS = {"Newton": "Disabled", "PhysX": "Disabled", "MuJoCo": "Disabled"}
USD_ASSET_PATHS: dict[WorkcellAssetName, str] = {
    "simple_room": "Environments/Simple_Room/simple_room.usd",
    "simple_room_without_table": "Environments/Simple_Room/simple_room.usd",
}
SIMPLE_ROOM_FLOOR_PATH = "Towel_Room01_floor_bottom_218/Towel_Room01_floor_bottom"
# Measured from the fully composed Isaac 6.2 room, in metres; checked after spawning.
SIMPLE_ROOM_FLOOR_TOP_M = -0.7695705931503188
# Walking-floor collider bounds in room coordinates, from the same composed inspection.
SIMPLE_ROOM_FLOOR_BOUNDS_XY_M = (
    (-3.9446376681781317, -2.7983224423354303),
    (3.9294780615833815, 4.269690400170449),
)
# Inner faces of the walls' 0.19 m skirting rims in room coordinates, from the same
# composed geometry; wall-mounted cabinets stand between these and the floor collider.
SIMPLE_ROOM_RIM_BOUNDS_XY_M = ((-4.32, -3.22), (4.32, 4.67))

# Each filtered channel measures one robot body against one task collider.
# These are private evaluator measurements, never model observations.
TASK_CONTACT_PAIRS = (
    ("left_hand_target", "left_wrist_yaw_link", "{ENV_REGEX_NS}/Task_target_button/geometry/mesh"),
    (
        "right_hand_target",
        "right_wrist_yaw_link",
        "{ENV_REGEX_NS}/Task_target_button/geometry/mesh",
    ),
    ("pelvis_floor", "pelvis", "{TASK_TERRAIN_COLLIDER}"),
    ("torso_floor", "torso_link", "{TASK_TERRAIN_COLLIDER}"),
    ("left_foot_floor", "left_ankle_roll_link", "{TASK_TERRAIN_COLLIDER}"),
    ("left_knee_floor", "left_knee_link", "{TASK_TERRAIN_COLLIDER}"),
    ("right_knee_floor", "right_knee_link", "{TASK_TERRAIN_COLLIDER}"),
    (
        "right_foot_floor",
        "right_ankle_roll_link",
        "{TASK_TERRAIN_COLLIDER}",
    ),
)


class WorkcellBox(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-z_]+$")
    center_m: Position
    full_size_m: Dimensions
    color: Color
    finish: SurfaceFinish = "painted"


class WorkcellAsset(BaseModel):
    """Released USD furniture, with its own upstream materials and collision geometry."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-z_]+$")
    asset: WorkcellAssetName
    position_m: Position
    yaw_degrees: FiniteFloat = 0


# room-floor-mesh: the Simple Room's thin floor box carries the robot (earlier scenes).
# flat-ground-collider: the standard 10 cm flat ground (friction 1, no restitution)
# carries the robot and the room's floor meshes do not collide; walls and skirting
# still do. The released SONIC planner spun and fell on the room floor box but not on
# the flat ground collider under identical commands.
FloorSupport = Literal["room-floor-mesh", "flat-ground-collider"]


class GroundWorkcellScene(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    robot_start_xy_m: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)] = Field(
        default_factory=lambda: [0.0, 0.0]
    )
    dex3_hands: bool = False
    wrist_cameras: bool = False
    # Earlier tasks' definitions; current scene files keep them explicitly null.
    transfer: None = None
    mobility: None = None
    inspection: None = None
    shutdown: ShutdownTask
    assets: list[WorkcellAsset] = Field(default_factory=list)
    floor_support: FloorSupport = "room-floor-mesh"
    # Give the model turn_in_place_degrees and cap raw facing changes at 45 degrees.
    stepped_turn_tool: bool = False
    # Keep the commanded facing within a small lead of the measured heading, so a raw
    # facing change is approached gradually like the released gamepad's stick turning.
    leashed_facing: bool = False
    boxes: list[WorkcellBox]
    button_center_m: Position
    button_radius_m: PositiveFloat
    button_color: Color

    @model_validator(mode="after")
    def require_consistent_shutdown_scene(self) -> GroundWorkcellScene:
        names = [item.name for item in [*self.boxes, *self.assets]]
        if len(names) != len(set(names)):
            raise ValueError("Scene asset and box names must be unique")
        if sum(asset.asset in ROOM_ASSETS for asset in self.assets) > 1:
            raise ValueError("A scene may contain only one room")
        if not self.dex3_hands:
            raise ValueError("Drawer closure contact is measured on articulated Dex3 hands")
        for cabinet in self.shutdown.cabinets:
            open_front = cabinet.drawer_front_xy_m(self.shutdown.open_drawer_position_m)
            if math.dist(self.robot_start_xy_m, open_front) <= self.shutdown.home_radius_m + 1:
                raise ValueError("Home must leave clear space in front of every drawer")
        return self

    @property
    def room_asset(self) -> WorkcellAsset | None:
        return next((asset for asset in self.assets if asset.asset in ROOM_ASSETS), None)

    @property
    def floor_collider_path(self) -> str:
        room = self.room_asset
        return (
            "{ENV_REGEX_NS}/Asset_" + room.name + "/" + SIMPLE_ROOM_FLOOR_PATH
            if room is not None and self.floor_support == "room-floor-mesh"
            else "/World/Ground/geometry/mesh"
        )


def task_contact_evidence(environment: Any, environment_index: int = 0) -> dict[str, Any]:
    """Read native filtered normal forces, including contacts between control ticks."""
    import numpy as np

    pairs = {}
    for sensor_name, _, _ in TASK_CONTACT_PAIRS:
        sensor = environment.scene["task_contact_" + sensor_name]
        data = sensor.data
        if data.force_matrix_w is None or data.force_matrix_w_history is None:
            raise RuntimeError(f"Filtered task contact data unavailable: {sensor_name}")
        forces = data.force_matrix_w.torch[environment_index].cpu().numpy()
        history = data.force_matrix_w_history.torch[environment_index].cpu().numpy()
        if forces.shape != (1, 1, 3) or history.shape != (environment.cfg.decimation, 1, 1, 3):
            raise RuntimeError(f"Unexpected task contact layout: {sensor_name}")
        if not np.isfinite(forces).all() or not np.isfinite(history).all():
            raise ValueError(f"Nonfinite task contact data: {sensor_name}")
        pairs[sensor_name] = {
            "normal_force_world_n": forces[0, 0].tolist(),
            "peak_normal_force_since_previous_control_tick_n": float(
                np.max(np.linalg.norm(history, axis=-1))
            ),
        }
    return {"protocol": "isaac-filtered-normal-forces-v1", "pairs": pairs}


def task_contact_tensor_fields(environment: Any) -> dict[str, Any]:
    """Expose contact tensors for packing with the current tick's other fields."""
    fields = {}
    for sensor_name, _, _ in TASK_CONTACT_PAIRS:
        data = environment.scene["task_contact_" + sensor_name].data
        if data.force_matrix_w is None or data.force_matrix_w_history is None:
            raise RuntimeError(f"Filtered task contact data unavailable: {sensor_name}")
        fields[sensor_name + "_forces"] = data.force_matrix_w.torch
        fields[sensor_name + "_history"] = data.force_matrix_w_history.torch
    return fields


def task_contact_evidence_batch(
    environment: Any, *, snapshot: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Read all robot contact channels in one host transfer, preserving history."""
    import numpy as np

    from humanoid_agent.runtime.state import copy_tensor_fields_to_cpu

    if snapshot is None:
        snapshot = copy_tensor_fields_to_cpu(task_contact_tensor_fields(environment))
    robot_count = next(iter(snapshot.values())).shape[0]
    results = []
    for environment_index in range(robot_count):
        pairs = {}
        for sensor_name, _, _ in TASK_CONTACT_PAIRS:
            forces = snapshot[sensor_name + "_forces"][environment_index]
            history = snapshot[sensor_name + "_history"][environment_index]
            if forces.shape != (1, 1, 3) or history.shape != (environment.cfg.decimation, 1, 1, 3):
                raise RuntimeError(f"Unexpected task contact layout: {sensor_name}")
            if not np.isfinite(forces).all() or not np.isfinite(history).all():
                raise ValueError(f"Nonfinite task contact data: {sensor_name}")
            pairs[sensor_name] = {
                "normal_force_world_n": forces[0, 0].tolist(),
                "peak_normal_force_since_previous_control_tick_n": float(
                    np.max(np.linalg.norm(history, axis=-1))
                ),
            }
        results.append({"protocol": "isaac-filtered-normal-forces-v1", "pairs": pairs})
    return results
