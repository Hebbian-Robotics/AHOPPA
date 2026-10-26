"""A virtual robot-mounted RGB-D sensor with explicit calibration assumptions."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from functools import cache
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.canonical_robot import canonical_urdf_path, require_canonical_urdf

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# HomeBody c09fcf39468e7bab43d744458a53ef1b1265cd6b: Camera.horizontal_fov
# and the torso -> d435_joint transform in assets/robot/g1_29dof_with_hand.urdf.
VERTICAL_FOV_DEGREES = float(
    np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(69.0) / 2) * IMAGE_HEIGHT / IMAGE_WIDTH))
)
CAMERA_POSITION_TORSO_M = [0.0576235, 0.01753, 0.42987]
CAMERA_DOWNWARD_PITCH_DEGREES = 47.6
MINIMUM_DEPTH_M = 0.25
MAXIMUM_DEPTH_M = 5.0
DEPTH_GRID_ROWS = 12
DEPTH_GRID_COLUMNS = 16


def calibration() -> dict[str, Any]:
    focal_length = IMAGE_HEIGHT / (2 * np.tan(np.deg2rad(VERTICAL_FOV_DEGREES) / 2))
    pitch = np.deg2rad(CAMERA_DOWNWARD_PITCH_DEGREES)
    forward = np.asarray([np.cos(pitch), 0.0, -np.sin(pitch)])
    up = np.asarray([np.sin(pitch), 0.0, np.cos(pitch)])
    right = np.cross(forward, up)
    return {
        "resolution_width_height": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "intrinsics_fx_fy_cx_cy": [
            float(focal_length),
            float(focal_length),
            IMAGE_WIDTH / 2,
            IMAGE_HEIGHT / 2,
        ],
        "camera_body": "torso_link",
        "camera_position_torso_m": CAMERA_POSITION_TORSO_M,
        "optical_to_torso_rotation": np.column_stack([right, -up, forward]).tolist(),
        "optical_axes": "x right, y down, z forward",
        "depth_kind": "optical-axis z in metres; not Euclidean range",
        "depth_valid_interval_m": [MINIMUM_DEPTH_M, MAXIMUM_DEPTH_M],
        "depth_grid": {
            "rows": DEPTH_GRID_ROWS,
            "columns": DEPTH_GRID_COLUMNS,
            "aggregation": "median of valid aligned depth pixels per equal-sized image tile",
            "invalid": None,
        },
        "assumptions": (
            "Virtual aligned RGB-D camera, not verified factory G1 calibration. "
            "Real deployment must use measured intrinsics, mounting transform and alignment. "
            "No depth noise, stereo failures or lens distortion are simulated."
        ),
    }


def depth_tiles(depth_m: Any) -> list[list[float | None]]:
    """Reduce aligned depth as a hardware-applicable numeric observation."""
    depth = np.asarray(depth_m)
    if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("Depth image does not match the declared camera resolution")
    result = []
    for row in range(DEPTH_GRID_ROWS):
        values = []
        for column in range(DEPTH_GRID_COLUMNS):
            tile = depth[
                row * IMAGE_HEIGHT // DEPTH_GRID_ROWS : (row + 1) * IMAGE_HEIGHT // DEPTH_GRID_ROWS,
                column * IMAGE_WIDTH // DEPTH_GRID_COLUMNS : (column + 1)
                * IMAGE_WIDTH
                // DEPTH_GRID_COLUMNS,
            ]
            valid = tile[np.isfinite(tile) & (tile >= MINIMUM_DEPTH_M) & (tile <= MAXIMUM_DEPTH_M)]
            values.append(round(float(np.median(valid)), 3) if valid.size else None)
        result.append(values)
    return result


@cache
def canonical_palm_attachment(side: str) -> tuple[np.ndarray, Rotation]:
    """Read the fixed palm frame that Newton merges into the wrist rigid body."""
    root = ElementTree.fromstring(require_canonical_urdf(canonical_urdf_path()))
    joint = root.find(f"joint[@name='{side}_hand_palm_joint']")
    if joint is None or joint.attrib["type"] != "fixed":
        raise ValueError("Wrist camera requires the canonical fixed palm attachment")
    origin = joint.find("origin")
    parent = joint.find("parent")
    child = joint.find("child")
    if (
        origin is None
        or parent is None
        or child is None
        or parent.attrib["link"] != f"{side}_wrist_yaw_link"
        or child.attrib["link"] != f"{side}_hand_palm_link"
    ):
        raise ValueError("Wrist camera palm attachment has unexpected frames")
    return (
        np.asarray([float(value) for value in origin.attrib["xyz"].split()]),
        Rotation.from_euler("xyz", [float(value) for value in origin.attrib["rpy"].split()]),
    )


def wrist_calibration(side: str) -> dict[str, Any]:
    """Compose the Unitree optical preset through the canonical fixed palm frame."""
    if side not in {"left", "right"}:
        raise ValueError("Wrist camera side must be left or right")
    palm_position, palm_rotation = canonical_palm_attachment(side)
    optical_position = [-0.04012, -0.07441 if side == "left" else 0.07441, 0.15711]
    optical_rotation = Rotation.from_quat([0.00539, 0.86024, 0.0424, 0.50809], scalar_first=True)
    focal_length_pixels = IMAGE_WIDTH * 12.0 / 20.0
    return {
        "protocol": "unitree-dex3-wrist-camera-v1",
        "camera_body": f"{side}_wrist_yaw_link",
        "position_body_m": (palm_position + palm_rotation.apply(optical_position)).tolist(),
        "optical_to_body_rotation": (
            palm_rotation.as_matrix() @ optical_rotation.as_matrix()
        ).tolist(),
        "resolution_width_height": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "intrinsics_fx_fy_cx_cy": [
            focal_length_pixels,
            focal_length_pixels,
            IMAGE_WIDTH / 2,
            IMAGE_HEIGHT / 2,
        ],
        "pinhole": {
            "focal_length": 12.0,
            "horizontal_aperture": 20.0,
            "focus_distance": 400.0,
            "clipping_range": (0.1, 1.0e5),
        },
        "upstream": {
            "repository": "unitreerobotics/unitree_sim_isaaclab",
            "revision": "e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc",
            "preset": f"CameraPresets.{side}_dex3_wrist_camera",
            "parent_link": f"{side}_hand_camera_base_link",
            "asset_revision": "394cf2448f8a9ed815c77c701a761f3d1ff1c8fb",
            "frame_equivalence": "Canonical SONIC hand_palm_joint equals Unitree hand_camera_base_joint relative to wrist_yaw_link",
        },
        "optical_axes": "ROS: x right, y down, z forward",
        "sensor": "RGB only; depth queries are available on the head camera only",
        "assumptions": "Published Unitree simulator preset and Dex3 attachment geometry, not measured calibration of our hardware. Physical deployment requires added cameras and measured optical calibration. Camera mass, cable forces and lens distortion are not simulated.",
    }
