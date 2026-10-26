"""Timed binary VR grasp commands using the pinned SONIC PICO preset solver."""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from humanoid_agent.dex3 import HAND_JOINT_NAMES, HandVector
from humanoid_agent.finite_command import FiniteCommandIdentity
from humanoid_agent.paths import PROJECT_ROOT
from humanoid_agent.validated_types import ValidatedModel

SOLVER_SHA256 = "1819bc011d022e2e9567e5209bc8c5b4f3f328c1275ab5cfeb0a8e0a2e9b9c25"
PICO_STREAMER_SHA256 = "728557bea52ee8dddea0d8cc6f54d8a74779d071bd6bd193de194080b90e897c"
HAND_CONTROL_PROTOCOL = "pico-trigger-timed-ramp-v1"


class GraspCommand(ValidatedModel):
    action: Literal["open", "close"]
    start_seconds: FiniteFloat = Field(ge=0)
    transition_seconds: FiniteFloat = Field(ge=0.02, default=0.4)


class ResolvedGrasp(ValidatedModel):
    command: GraspCommand
    target_positions: HandVector
    mapping_protocol: Literal["pico-trigger-middle-v1"] = "pico-trigger-middle-v1"
    solver_sha256: Literal["1819bc011d022e2e9567e5209bc8c5b4f3f328c1275ab5cfeb0a8e0a2e9b9c25"] = (
        SOLVER_SHA256
    )
    streamer_sha256: Literal["728557bea52ee8dddea0d8cc6f54d8a74779d071bd6bd193de194080b90e897c"] = (
        PICO_STREAMER_SHA256
    )


class HandTransitionInitials(ValidatedModel):
    left: HandVector | None = None
    right: HandVector | None = None


@cache
def load_pinned_hand_mapper(sonic_source: Path) -> tuple[Any, Any]:
    """Import the upstream solver; isolate its VR-only trigger encoder by source pin."""
    solver_path = sonic_source / "gear_sonic/utils/teleop/solver/hand/g1_gripper_ik_solver.py"
    streamer_path = sonic_source / "gear_sonic/scripts/pico_manager_thread_server.py"
    for path, expected_hash in (
        (solver_path, SOLVER_SHA256),
        (streamer_path, PICO_STREAMER_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Upstream hand mapping differs from pinned SONIC: {path}")
    module = importlib.import_module("gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver")
    if module.__file__ is None or Path(module.__file__).resolve() != solver_path.resolve():
        raise ValueError("Imported hand solver is not from the configured SONIC checkout")
    definitions: list[ast.stmt] = [
        statement
        for statement in ast.parse(streamer_path.read_bytes()).body
        if isinstance(statement, ast.FunctionDef) and statement.name == "generate_finger_data"
    ]
    if len(definitions) != 1:
        raise ValueError("Pinned upstream trigger encoder is missing or ambiguous")
    # This ten-line function lives in the executable PICO application, which
    # imports hardware/tracking dependencies. Execute its exact pinned definition;
    # importing that application would burden the model operator with VR runtime.
    namespace: dict[str, Any] = {"np": np}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(streamer_path), "exec"),
        namespace,
    )
    return module.G1GripperInverseKinematicsSolver, namespace["generate_finger_data"]


def sonic_source_path() -> Path:
    return Path(os.environ.get("SONIC_SOURCE", PROJECT_ROOT / "artifacts/upstream-assets/SONIC"))


def upstream_hand_positions(
    side: Literal["left", "right"],
    action: Literal["open", "close"],
    *,
    sonic_source: Path | None = None,
) -> list[float]:
    solver_type, generate_finger_data = load_pinned_hand_mapper(sonic_source or sonic_source_path())
    finger_data = generate_finger_data(side, float(action == "close"), 0.0)
    targets = np.asarray(solver_type(side=side)({"position": finger_data}), dtype=float)
    if targets.shape != (7,) or not np.isfinite(targets).all():
        raise ValueError("Upstream hand solver returned invalid targets")
    return targets.tolist()


def validate_grasp_timing(
    command: GraspCommand | None, duration_seconds: float, *, hands_available: bool
) -> None:
    if command is None:
        return
    if not hands_available:
        raise ValueError("Grasp commands require articulated hand feedback")
    execution_duration_seconds = max(1, round(duration_seconds / 0.02)) * 0.02
    if (
        command.start_seconds + command.transition_seconds
        > min(duration_seconds, execution_duration_seconds) + 1e-9
    ):
        raise ValueError("Grasp transition must finish within the requested motion duration")


class FiniteGraspPublication(FiniteCommandIdentity):
    """Validate resolved schedules at ingress, including saved or direct publications."""

    control_ticks: int = Field(ge=1)
    left_grasp_schedule: ResolvedGrasp | None = None
    right_grasp_schedule: ResolvedGrasp | None = None

    @model_validator(mode="after")
    def require_grasp_within_motion(self) -> FiniteGraspPublication:
        try:
            duration_seconds = self.control_ticks * 0.02
        except OverflowError as error:
            raise ValueError("Control ticks must describe a finite motion duration") from error
        for schedule in (self.left_grasp_schedule, self.right_grasp_schedule):
            validate_grasp_timing(
                schedule.command if schedule is not None else None,
                duration_seconds,
                hands_available=True,
            )
        return self


def resolve_grasp(
    command: GraspCommand | None,
    side: Literal["left", "right"],
    *,
    duration_seconds: float,
    limits: dict[str, dict[str, float]] | None,
) -> ResolvedGrasp | None:
    validate_grasp_timing(command, duration_seconds, hands_available=limits is not None)
    if command is None:
        return None
    targets = upstream_hand_positions(side, command.action)
    if limits is None:
        raise ValueError("Missing hand limits")
    for name, target in zip(HAND_JOINT_NAMES[side], targets, strict=True):
        if not limits[name]["lower"] <= target <= limits[name]["upper"]:
            raise ValueError(f"Upstream preset exceeds canonical finger limits: {name}")
    return ResolvedGrasp(command=command, target_positions=targets)


class HandTransitionSampler:
    """Our elapsed-time ramp; native measured-state clipping remains downstream."""

    def __init__(
        self,
        left: ResolvedGrasp | None,
        right: ResolvedGrasp | None,
        initials: HandTransitionInitials,
    ) -> None:
        self.left = left
        self.right = right
        self.initials = initials
        if (left is not None and initials.left is None) or (
            right is not None and initials.right is None
        ):
            raise ValueError("Timed grasp requires measured finger positions at admission")

    def sample(self, elapsed_seconds: float) -> tuple[HandVector | None, HandVector | None]:
        outputs = []
        for schedule, initial in (
            (self.left, self.initials.left),
            (self.right, self.initials.right),
        ):
            if schedule is None:
                outputs.append(None)
                continue
            if initial is None:
                raise ValueError("Missing measured hand transition start")
            fraction = max(
                0.0,
                min(
                    1.0,
                    (elapsed_seconds - schedule.command.start_seconds)
                    / schedule.command.transition_seconds,
                ),
            )
            outputs.append(
                tuple(
                    (1 - fraction) * start + fraction * end
                    for start, end in zip(initial, schedule.target_positions, strict=True)
                )
            )
        return outputs[0], outputs[1]
