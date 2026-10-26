"""Reuse pinned staging and device buffers for batched native motor commands."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from humanoid_agent.dex3 import HAND_JOINT_NAMES
from humanoid_agent.runtime.sdk import SonicMotorCommand


class BatchedMotorTargets:
    """Upload changed commands together, retaining independent robot/joint order."""

    fields = ("positions", "velocities", "feedforward_torques", "stiffness", "damping")

    def __init__(self, robot: Any, motor_indices: Sequence[int], robot_count: int) -> None:
        self.robot = robot
        self.robot_count = robot_count
        self.dex3_hands = len(robot.joint_names) == 43
        all_indices = list(motor_indices)
        if self.dex3_hands:
            all_indices.extend(
                robot.joint_names.index(name)
                for side in ("left", "right")
                for name in HAND_JOINT_NAMES[side]
            )
        self.inverse_indices = np.argsort(all_indices)
        torch: Any = importlib.import_module("torch")
        shape = (len(self.fields), robot_count, len(all_indices))
        self.host = torch.empty(shape, dtype=torch.float32, device="cpu", pin_memory=True)
        self.host_values = self.host.numpy()
        self.device = torch.empty(shape, dtype=torch.float32, device="cuda:0")
        self.last_commands: tuple[SonicMotorCommand, ...] = ()
        self.last_gains: tuple[tuple[tuple[float, ...], ...], ...] = ()

    def apply(self, commands: Sequence[SonicMotorCommand]) -> bool:
        if len(commands) != self.robot_count:
            raise ValueError("Motor targets require one command per simulated robot")
        current_commands = tuple(commands)
        if current_commands == self.last_commands:
            return False
        if self.dex3_hands and any(
            command.left_hand is None or command.right_hand is None for command in commands
        ):
            raise ValueError("Articulated hands require native motor commands for both hands")
        gains = tuple(
            tuple(
                tuple(getattr(part, field))
                for part in (command, command.left_hand, command.right_hand)
                if part is not None
                for field in ("stiffness", "damping")
            )
            for command in commands
        )
        gains_changed = gains != self.last_gains
        field_count = len(self.fields) if gains_changed else 3
        for field_index, field_name in enumerate(self.fields[:field_count]):
            for environment_index, command in enumerate(commands):
                np.take(
                    [
                        value
                        for part in (command, command.left_hand, command.right_hand)
                        if part is not None
                        for value in getattr(part, field_name)
                    ]
                    if self.dex3_hands
                    else getattr(command, field_name),
                    self.inverse_indices,
                    out=self.host_values[field_index, environment_index],
                )
        # Blocking copy completes before the host staging area is reused. The
        # device views retain stable addresses across all physics steps.
        self.device[:field_count].copy_(self.host[:field_count])
        if gains_changed:
            self.robot.write_joint_stiffness_to_sim(self.device[3])
            self.robot.write_joint_damping_to_sim(self.device[4])
        self.robot.set_joint_position_target(self.device[0])
        self.robot.set_joint_velocity_target(self.device[1])
        self.robot.set_joint_effort_target(self.device[2])
        self.last_commands = current_commands
        self.last_gains = gains
        return True
