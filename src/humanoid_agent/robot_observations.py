"""Allowlisted model observations that have explicit hardware counterparts."""

from __future__ import annotations

from pydantic import FiniteFloat, model_validator

from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel, Vector3

PROTOCOL = "egocentric-rgbd-encoders-imu-v1"


class RobotSensors(ValidatedModel):
    """Ideal sensor emulation; no world localization or evaluator outputs."""

    joint_names: ImmutableSequence[str]
    joint_positions_rad: ImmutableSequence[FiniteFloat]
    joint_velocities_rad_per_s: ImmutableSequence[FiniteFloat]
    imu_angular_velocity_body_rad_per_s: Vector3
    imu_gravity_direction_body: Vector3

    @model_validator(mode="after")
    def require_complete_named_joint_measurements(self) -> RobotSensors:
        if (
            not len(self.joint_names)
            == len(self.joint_positions_rad)
            == len(self.joint_velocities_rad_per_s)
        ):
            raise ValueError("Joint names, positions and velocities must have matching lengths")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Joint sensor names must be unique")
        return self

    def in_joint_order(self, joint_names: list[str]) -> RobotSensors:
        if len(set(joint_names)) != len(joint_names) or set(joint_names) != set(self.joint_names):
            raise ValueError("Sensor joint names differ from the calibrated robot")
        original_indices = {name: index for index, name in enumerate(self.joint_names)}
        return self.model_copy(
            update={
                "joint_names": list(joint_names),
                "joint_positions_rad": [
                    self.joint_positions_rad[original_indices[name]] for name in joint_names
                ],
                "joint_velocities_rad_per_s": [
                    self.joint_velocities_rad_per_s[original_indices[name]] for name in joint_names
                ],
            }
        )
