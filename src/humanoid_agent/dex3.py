"""Dex3 joint ordering and articulation from the pinned SONIC robot definition."""

from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Annotated

from pydantic import Field, FiniteFloat, model_validator

from humanoid_agent.artifact_io import atomic_write_text
from humanoid_agent.canonical_robot import require_canonical_urdf
from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel

HAND_JOINT_PARTS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
HAND_JOINT_NAMES = {
    side: [f"{side}_hand_{part}_joint" for part in HAND_JOINT_PARTS] for side in ("left", "right")
}
HandVector = Annotated[ImmutableSequence[FiniteFloat], Field(min_length=7, max_length=7)]


class HandMotorCommand(ValidatedModel):
    positions: HandVector
    velocities: HandVector
    feedforward_torques: HandVector
    stiffness: HandVector
    damping: HandVector

    @model_validator(mode="after")
    def require_nonnegative_gains(self) -> HandMotorCommand:
        if min(self.stiffness) < 0 or min(self.damping) < 0:
            raise ValueError("Hand motor gains must be nonnegative")
        return self


class HandJointObservation(ValidatedModel):
    positions: HandVector
    velocities: HandVector


class HandEncoderState(ValidatedModel):
    positions: HandVector
    velocities: HandVector
    accelerations: HandVector
    measured_torques: HandVector


def validate_hand_joint_targets(
    left: HandVector | None,
    right: HandVector | None,
    limits: dict[str, dict[str, float]] | None,
) -> None:
    for side, targets in (("left", left), ("right", right)):
        if targets is None:
            continue
        if limits is None:
            raise ValueError("Finger targets require articulated hand feedback")
        for name, target in zip(HAND_JOINT_NAMES[side], targets, strict=True):
            if not limits[name]["lower"] <= target <= limits[name]["upper"]:
                raise ValueError(f"Finger target exceeds the pinned joint limits: {name}")


def hand_joint_limits(urdf: Path) -> dict[str, dict[str, float]]:
    root = ElementTree.fromstring(require_canonical_urdf(urdf))
    limits = {}
    for joint_name in (*HAND_JOINT_NAMES["left"], *HAND_JOINT_NAMES["right"]):
        joint = root.find(f"joint[@name='{joint_name}']")
        limit = None if joint is None else joint.find("limit")
        if limit is None:
            raise ValueError(f"Pinned robot has no limits for {joint_name}")
        limits[joint_name] = {
            name: float(limit.attrib[name]) for name in ("lower", "upper", "effort", "velocity")
        }
    return limits


def articulated_hand_urdf(urdf: Path) -> Path:
    """Enable the 14 existing finger joints; preserve body transforms and inertias.

    The training URDF fixes these joints despite retaining their axes and limits.
    SONIC's companion g1_29dof_with_hand.urdf uses revolute joints with these same
    names and limits. Absolute mesh paths let the derived file live outside the
    upstream checkout. Physics and rendering use this same generated definition.
    """
    root = ElementTree.fromstring(require_canonical_urdf(urdf))
    hand_joint_limits(urdf)
    for joint_name in (*HAND_JOINT_NAMES["left"], *HAND_JOINT_NAMES["right"]):
        joint = root.find(f"joint[@name='{joint_name}']")
        if joint is None or joint.attrib["type"] != "fixed":
            raise ValueError(f"Unexpected pinned finger joint: {joint_name}")
        joint.set("type", "revolute")
    description = urdf.parents[2]
    for mesh in root.iter("mesh"):
        filename = mesh.attrib["filename"]
        if not filename.startswith("package://robot_description/"):
            raise ValueError(f"Unexpected pinned mesh URI: {filename}")
        path = description / filename.removeprefix("package://robot_description/")
        if not path.is_file() or path.stat().st_size < 256:
            raise ValueError(f"Missing or unusable robot mesh: {path}")
        mesh.set("filename", str(path.resolve()))
    source = ElementTree.tostring(root, encoding="unicode")
    digest = hashlib.sha256(source.encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / "humanoid-agent-robot-assets" / digest
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "g1-dex3.urdf"
    if not destination.exists():
        atomic_write_text(destination, source)
    elif destination.read_text() != source:
        raise ValueError("Content-addressed robot definition was modified")
    return destination
