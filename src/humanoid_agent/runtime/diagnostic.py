"""Configure model-free controller checks for the production episode owner."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.runtime.shutdown_reference import SHUTDOWN_REFERENCE_DURATION_SECONDS
from humanoid_agent.runtime.stepped_turn import TURN_CHECK_DURATION_SECONDS

if TYPE_CHECKING:
    from humanoid_agent.cli import EpisodeOptions


def scripted_episode_options(arguments: EpisodeOptions) -> EpisodeOptions:
    """Change the command source and optional rendering workload, retaining episode ownership."""
    if (
        not math.isfinite(arguments.diagnostic_phase_seconds)
        or not 1 <= arguments.diagnostic_phase_seconds <= 30
    ):
        raise ValueError("Diagnostic phases must last 1–30 seconds")
    if arguments.task_scene is None:
        raise ValueError("Controller checks require --task-scene to exercise the production scene")
    if (
        not math.isfinite(arguments.capture_interval_seconds)
        or arguments.capture_interval_seconds < 0.05
    ):
        raise ValueError("Camera capture interval must be finite and at least 50 ms")
    # Every controller check runs in a validated drawer-shutdown scene.
    GroundWorkcellScene.model_validate_json(arguments.task_scene.read_text())
    return replace(
        arguments,
        episode_kind="controller-check",
        wall_seconds=arguments.diagnostic_phase_seconds
        if arguments.diagnostic_motion == "standing"
        else TURN_CHECK_DURATION_SECONDS
        if arguments.diagnostic_motion == "turn-check"
        else SHUTDOWN_REFERENCE_DURATION_SECONDS,
    )


def run_native_controller_check(arguments: EpisodeOptions) -> None:
    """Keep the public diagnostic alias on the same supervised owner as model episodes."""
    from humanoid_agent.runtime.episode import run_episode

    run_episode(scripted_episode_options(arguments))


def episode_command_protocol(
    arguments: EpisodeOptions,
) -> Literal["sonic-native-planner-v2", "sonic-native-planner-v3"]:
    """Identify the actual model schema or model-free diagnostic wire contract."""
    return (
        "sonic-native-planner-v2"
        if arguments.episode_kind == "controller-check"
        else "sonic-native-planner-v3"
    )
