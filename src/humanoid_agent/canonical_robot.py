"""Shared SONIC target frames and kinematics from the pinned Newton robot URDF."""

from __future__ import annotations

import hashlib
import importlib
import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, NonNegativeFloat, PositiveFloat
from scipy.spatial.transform import Rotation

from humanoid_agent.paths import PROJECT_ROOT


class ControllerConstants(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    default_angles: Annotated[list[FiniteFloat], Field(min_length=29, max_length=29)]
    action_scale: Annotated[list[PositiveFloat], Field(min_length=29, max_length=29)]
    kp: Annotated[list[PositiveFloat], Field(min_length=29, max_length=29)]
    kd: Annotated[list[NonNegativeFloat], Field(min_length=29, max_length=29)]


CANONICAL_URDF_SHA256 = "6e107391d015ab27438026bb953b21ef8fa6a02305e65393985908941d7fd98d"


def canonical_urdf_path() -> Path:
    source = Path(os.environ.get("SONIC_SOURCE", PROJECT_ROOT / "artifacts/upstream-assets/SONIC"))
    return Path(
        os.environ.get(
            "SONIC_ROBOT_URDF",
            source / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf",
        )
    )


def require_canonical_urdf(path: Path) -> bytes:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != CANONICAL_URDF_SHA256:
        raise ValueError("Robot URDF differs from the pinned SONIC/Newton definition")
    return source


def build_encoder_model(urdf_path: Path | None = None) -> tuple[Any, Any]:
    mujoco: Any = importlib.import_module("mujoco")
    root = ElementTree.fromstring(require_canonical_urdf(urdf_path or canonical_urdf_path()))
    settings = ElementTree.SubElement(root, "mujoco")
    ElementTree.SubElement(settings, "compiler", fusestatic="false", discardvisual="true")
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for geometry in link.findall(tag):
                link.remove(geometry)
    # Geometry is unnecessary for FK. All link/joint transforms remain from the
    # original URDF, including fixed links; this model never steps dynamics.
    specification = mujoco.MjSpec.from_string(ElementTree.tostring(root, encoding="unicode"))
    specification.body("pelvis").add_freejoint(name="root")
    model = specification.compile()
    return model, mujoco.MjData(model)


def imu_mounts(urdf_path: Path) -> dict[str, tuple[str, tuple[float, ...]]]:
    """Read identity-oriented fixed IMU mounts from the canonical robot definition."""
    root = ElementTree.fromstring(require_canonical_urdf(urdf_path))
    mounts = {}
    for name in ("pelvis", "torso"):
        joint = root.find(f"joint[@name='imu_in_{name}_joint']")
        if joint is None or joint.attrib["type"] != "fixed":
            raise ValueError(f"Missing fixed {name} IMU mount")
        origin = joint.find("origin")
        parent = joint.find("parent")
        if (
            origin is None
            or parent is None
            or any(float(value) != 0 for value in origin.attrib.get("rpy", "0 0 0").split())
        ):
            raise ValueError(f"Unsupported {name} IMU mount orientation")
        mounts[name] = (
            parent.attrib["link"],
            tuple(float(value) for value in origin.attrib["xyz"].split()),
        )
    return mounts


JOINT_NAMES = [
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
JOINT_NAMES += ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
JOINT_NAMES += [
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
HAND_TRACKING_POINTS = (
    ("left_wrist_yaw_link", (0.18, -0.025, 0.0)),
    ("right_wrist_yaw_link", (0.18, 0.025, 0.0)),
)
BODY_NAMES = [body_name for body_name, _ in HAND_TRACKING_POINTS] + ["torso_link"]
POINT_OFFSETS = np.asarray([offset for _, offset in HAND_TRACKING_POINTS] + [(0, 0, 0.35)])


def heading_rotation(data: Any) -> Rotation:
    root_rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
    yaw = np.arctan2(root_rotation.as_matrix()[1, 0], root_rotation.as_matrix()[0, 0])
    return Rotation.from_euler("z", yaw)


def measured_tracking_points(model: Any, data: Any) -> tuple[Any, Any, Any]:
    heading = heading_rotation(data)
    positions, orientations, world_positions = [], [], []
    for body_name, offset in zip(BODY_NAMES, POINT_OFFSETS, strict=True):
        body_identifier = model.body(body_name).id
        body_rotation = Rotation.from_matrix(data.xmat[body_identifier].reshape(3, 3))
        world_position = data.xpos[body_identifier] + body_rotation.apply(offset)
        world_positions.append(world_position)
        positions.append(heading.inv().apply(world_position - data.qpos[:3]))
        local_rotation = heading.inv() * body_rotation
        if not isinstance(local_rotation, Rotation):
            raise TypeError("Tracking orientation conversion failed")
        orientations.append(local_rotation.as_quat(scalar_first=True))
    return np.asarray(positions), np.asarray(orientations), np.asarray(world_positions)
