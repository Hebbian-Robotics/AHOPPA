"""Read batched scoring poses without depending on a runnable task driver."""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.runtime.evaluation import GroundTaskMeasurements
from humanoid_agent.runtime.state import copy_tensor_fields_to_cpu, recorded_state_from_newton
from humanoid_agent.shutdown_task import (
    SEKTION_BASE_HALF_HEIGHT_M,
    SEKTION_TOP_DRAWER_FRONT_Z_M,
    TASK_DRAWER_JOINT,
    CabinetIdentifier,
    ShutdownTask,
)
from humanoid_agent.shutdown_visibility import PlanOccluder, front_in_view


@dataclass(frozen=True)
class TaskMeasurementSnapshot:
    root_states: NDArray[np.floating]
    body_positions: NDArray[np.floating]
    body_quaternions_xyzw: NDArray[np.floating]
    joint_positions: NDArray[np.floating] | None = None
    shutdown_fields: dict[str, NDArray[np.floating]] | None = None
    cabinet_joint_names: list[str] | None = None
    shutdown_contact_body_names: list[str] | None = None


def task_measurement_tensor_fields(
    robot: Any, *, include_joint_positions: bool = False
) -> dict[str, Any]:
    return {
        **({"joint_positions": robot.data.joint_pos.torch} if include_joint_positions else {}),
        "root_states": robot.data.root_state_w.torch,
        "body_positions": robot.data.body_link_pos_w.torch,
        "body_quaternions_xyzw": robot.data.body_link_quat_w.torch,
    }


def shutdown_tensor_fields(scene: Any, task: ShutdownTask) -> dict[str, Any]:
    """Cabinet joint state and explicit normal-force contact histories for scoring."""
    fields = {}
    for cabinet in task.cabinets:
        articulation = scene["cabinet_" + cabinet.identifier]
        fields[f"cabinet_{cabinet.identifier}_joint_positions"] = articulation.data.joint_pos.torch
        fields[f"cabinet_{cabinet.identifier}_joint_velocities"] = articulation.data.joint_vel.torch
        data = scene["shutdown_drawer_contacts_" + cabinet.identifier].data
        if data.normal_force_matrix_w_history is None:
            raise RuntimeError("Filtered drawer normal-force history is unavailable")
        fields[f"cabinet_{cabinet.identifier}_drawer_normal_force_history"] = (
            data.normal_force_matrix_w_history.torch
        )
    return fields


# Dex3 wrist (with its merged palm) and finger bodies count as the hand.
HAND_BODY_NAME = re.compile(r"(left|right)_(wrist_yaw|hand_\w+)_link")


def cabinet_joint_positions(
    fields: dict[str, Any], task: ShutdownTask, joint_names: list[str], environment_index: int
) -> dict[CabinetIdentifier, dict[str, float]]:
    return {
        cabinet.identifier: {
            name: float(value)
            for name, value in zip(
                joint_names,
                fields[f"cabinet_{cabinet.identifier}_joint_positions"][environment_index],
                strict=True,
            )
        }
        for cabinet in task.cabinets
    }


def shutdown_measurements(
    fields: dict[str, Any],
    task: ShutdownTask,
    joint_names: list[str],
    contact_body_names: list[str],
    environment_index: int,
) -> dict[str, float]:
    """Drawer slide state and peak summed normal force since the previous 20 ms tick."""
    drawer_index = joint_names.index(TASK_DRAWER_JOINT)
    hand_bodies = np.asarray(
        [bool(HAND_BODY_NAME.fullmatch(name)) for name in contact_body_names], dtype=bool
    )
    if not hand_bodies.any() or hand_bodies.all():
        raise ValueError("Drawer contact sensing must cover hand and other robot bodies")
    measured = {}
    for cabinet in task.cabinets:
        prefix = "drawer_" + cabinet.identifier.lower()
        key = f"cabinet_{cabinet.identifier}"
        measured[prefix + "_position_m"] = float(
            fields[key + "_joint_positions"][environment_index, drawer_index]
        )
        measured[prefix + "_velocity_m_per_s"] = float(
            fields[key + "_joint_velocities"][environment_index, drawer_index]
        )
        history = np.asarray(fields[key + "_drawer_normal_force_history"][environment_index])
        if history.ndim != 4 or history.shape[1] != len(contact_body_names):
            raise ValueError("Unexpected drawer contact history layout")
        # (history, sensing bodies, filter shapes, xyz) -> per-body force per physics step.
        per_body = np.linalg.norm(history, axis=-1).sum(axis=2)
        for kind, mask in (("hand", hand_bodies), ("body", np.logical_not(hand_bodies))):
            measured[f"{prefix}_{kind}_normal_force_n"] = float(per_body[:, mask].sum(axis=1).max())
    return measured


TOP_DRAWER_FRONT_CENTER_HEIGHT_M = (
    SEKTION_BASE_HALF_HEIGHT_M + sum(SEKTION_TOP_DRAWER_FRONT_Z_M) / 2
)


def storage_occluders(scene: GroundWorkcellScene) -> list[PlanOccluder]:
    """Scene boxes taller than a standing camera hide what is behind them in plan view."""
    return [
        PlanOccluder(
            center_xy_m=(box.center_m[0], box.center_m[1]),
            half_size_xy_m=(box.full_size_m[0] / 2, box.full_size_m[1] / 2),
            top_z_m=box.center_m[2] + box.full_size_m[2] / 2,
        )
        for box in scene.boxes
        if box.center_m[2] + box.full_size_m[2] / 2 > 1.5
    ]


def read_task_measurement_snapshot(
    robot: Any, *, include_joint_positions: bool = False
) -> TaskMeasurementSnapshot:
    """Read all robots' scoring poses with one device-to-host synchronization."""
    fields = copy_tensor_fields_to_cpu(
        task_measurement_tensor_fields(robot, include_joint_positions=include_joint_positions)
    )
    return TaskMeasurementSnapshot(**fields)


def measurements(
    robot: Any,
    scene: GroundWorkcellScene,
    evidence: dict[str, Any],
    *,
    environment_index: int = 0,
    environment_origin_m: Any = None,
    snapshot: TaskMeasurementSnapshot | None = None,
) -> GroundTaskMeasurements:
    if snapshot is None:
        snapshot = read_task_measurement_snapshot(robot, include_joint_positions=True)
    root = recorded_state_from_newton(snapshot.root_states[environment_index])
    origin = np.zeros(3) if environment_origin_m is None else np.asarray(environment_origin_m)
    robot_start_world_xy_m = origin[:2] + np.asarray(scene.robot_start_xy_m)
    if snapshot.joint_positions is None:
        raise ValueError("Standing checks require measured knee joint angles")
    standing_measurements = {}
    for side in ("left", "right"):
        standing_measurements[side + "_knee_angle_rad"] = float(
            snapshot.joint_positions[
                environment_index, robot.joint_names.index(side + "_knee_joint")
            ]
        )
        for body in ("foot", "knee"):
            standing_measurements[f"{side}_{body}_floor_force_n"] = evidence["pairs"][
                f"{side}_{body}_floor"
            ]["peak_normal_force_since_previous_control_tick_n"]
    if (
        snapshot.shutdown_fields is None
        or snapshot.cabinet_joint_names is None
        or snapshot.shutdown_contact_body_names is None
    ):
        raise ValueError("Shutdown scoring requires measured cabinet joints and contacts")
    drawer_measurements = shutdown_measurements(
        snapshot.shutdown_fields,
        scene.shutdown,
        snapshot.cabinet_joint_names,
        snapshot.shutdown_contact_body_names,
        environment_index,
    )
    if scene.shutdown.protocol == "drawer-shutdown-v2":
        torso = robot.body_names.index("torso_link")
        torso_position = snapshot.body_positions[environment_index, torso] - origin
        torso_quaternion = snapshot.body_quaternions_xyzw[environment_index, torso]
        occluders = storage_occluders(scene)
        for cabinet in scene.shutdown.cabinets:
            prefix = "drawer_" + cabinet.identifier.lower()
            front = cabinet.drawer_front_xy_m(drawer_measurements[prefix + "_position_m"])
            drawer_measurements[prefix + "_front_in_view"] = float(
                front_in_view(
                    torso_position,
                    torso_quaternion,
                    np.asarray([*front, TOP_DRAWER_FRONT_CENTER_HEIGHT_M]),
                    np.asarray(cabinet.outward_direction_xy),
                    occluders,
                )
            )
    return GroundTaskMeasurements(
        **drawer_measurements,
        **standing_measurements,
        root_vertical_speed_m_per_s=float(root[9]),
        root_height_m=float(root[2]),
        root_distance_from_start_m=float(np.linalg.norm(root[:2] - robot_start_world_xy_m)),
        horizontal_speed_m_per_s=float(np.linalg.norm(root[7:9])),
        gravity_z=-float(Rotation.from_quat(root[3:7], scalar_first=True).as_matrix()[2, 2]),
        pelvis_floor_force_n=evidence["pairs"]["pelvis_floor"][
            "peak_normal_force_since_previous_control_tick_n"
        ],
        torso_floor_force_n=evidence["pairs"]["torso_floor"][
            "peak_normal_force_since_previous_control_tick_n"
        ],
    )
