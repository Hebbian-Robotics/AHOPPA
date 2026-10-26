"""Run the native SONIC task physics or its separately supervised model operator."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from humanoid_agent.logging_config import configure_logging
from humanoid_agent.model_api_configuration import ApiRetryPolicy, ApiServiceTier, ReasoningEffort
from humanoid_agent.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


@dataclass(kw_only=True)
class EpisodeOptions:
    role: Literal[
        "episode",
        "group",
        "shared-physics",
        "operator",
        "physics",
        "camera",
        "sdk",
        "scene-check",
        "shared-scene-check",
        "shared-camera-check",
        "native-controller-check",
        "analyze",
        "render",
        "render-worker",
    ] = "episode"
    provider: Literal["sol", "astra", "opus", "gemini"] = "sol"
    providers: Annotated[
        tuple[Literal["sol", "astra", "opus", "gemini"], ...], Parameter(consume_multiple=True)
    ] = ()
    # Hybrid head/hand keyframes with native SONIC locomotion is the only interface.
    mode: Literal["hybrid"] = "hybrid"
    simulator: Literal["isaac"] = "isaac"
    output: Path
    wall_seconds: float = 600
    external_camera: bool = True
    physics_directory: Path | None = None
    facility_map: Path | None = None
    stepped_turns: bool = False
    leashed_facing: bool = False
    episode_directory: Path | None = None
    frames_per_second: int = 20
    num_environments: int = 1
    physics_cpu_set: str = "2-5"
    operator_cpu_set: str = "0-1"
    sdk_cpu_set: str = "0-1"
    io_cpu_set: str = "0-1"
    controller_cpu_set: str = "6-9"
    camera_cpu_set: str = "10-13"
    episode_kind: Literal["model", "controller-check"] = "model"
    diagnostic_camera: Literal["off", "on"] = "off"
    diagnostic_phase_seconds: float = 5.0
    diagnostic_motion: Literal[
        "standard",
        "standing",
        "turn-check",
    ] = "standard"
    capture_interval_seconds: float = 0.5
    native_workspace: Path = Path("/home/ubuntu/frontier-motion-native-v1")
    isaac_workspace: Path = Path("/home/ubuntu/frontier-motion-newton-v1")
    system_libraries: Path = Path(
        "/home/ubuntu/frontier-motion-isaac-v1/system-libs/usr/lib/x86_64-linux-gnu"
    )
    sonic_constants: Path = ROOT / "runtime-assets/sonic_constants.json"
    task_scene: Path | None = None
    decorative_assets: Path | None = None
    command_directory: Path | None = None
    # Omitted for an episode: the scene's shutdown instruction. The owner passes it on.
    task_instruction: str | None = None
    api_service_tier: ApiServiceTier = ApiServiceTier.STANDARD
    api_request_timeout_seconds: float = 300
    api_retry_policy: ApiRetryPolicy = ApiRetryPolicy.BOUNDED
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    model_max_output_tokens: int = 12000


def run_task(arguments: Annotated[EpisodeOptions, Parameter(name="*")]) -> None:
    if not 1 <= arguments.model_max_output_tokens <= 65536:
        raise ValueError("Model output token ceiling must be between 1 and 65536")
    if not 0 < arguments.wall_seconds <= 3600:
        raise ValueError("Episode wall budget must be from zero to one hour")
    if arguments.role == "render":
        if arguments.episode_directory is None:
            raise ValueError("Postprocessing requires --episode-directory")
        from humanoid_agent.runtime.episode import create_host_process_environment

        subprocess.run(
            [
                str(arguments.isaac_workspace / "IsaacLab/.venv/bin/python"),
                "-m",
                "humanoid_agent",
                "--role",
                "render-worker",
                "--output",
                str(arguments.output),
                "--episode-directory",
                str(arguments.episode_directory),
                "--frames-per-second",
                str(arguments.frames_per_second),
                "--isaac-workspace",
                str(arguments.isaac_workspace),
                "--native-workspace",
                str(arguments.native_workspace),
                "--system-libraries",
                str(arguments.system_libraries),
                "--sonic-constants",
                str(arguments.sonic_constants),
            ],
            env=create_host_process_environment(arguments, arguments.sonic_constants),
            check=True,
        )
    elif arguments.role in {"analyze", "render-worker"}:
        if arguments.episode_directory is None:
            raise ValueError("Postprocessing requires --episode-directory")
        if not 1 <= arguments.frames_per_second <= 60:
            raise ValueError("Replay frame rate must be between 1 and 60")
        from humanoid_agent.runtime.postprocess import analyze_episode, render_episode

        if arguments.role == "analyze":
            analyze_episode(arguments.episode_directory, arguments.output)
        else:
            render_episode(
                arguments.episode_directory,
                arguments.output,
                arguments.isaac_workspace / "SONIC",
                arguments.frames_per_second,
            )
    elif arguments.role == "episode":
        from humanoid_agent.runtime.episode import run_episode

        run_episode(arguments)
    elif arguments.role == "group":
        from humanoid_agent.runtime.group import run_episode_group

        run_episode_group(arguments)
    elif arguments.role in {"physics", "shared-physics"}:
        if arguments.task_scene is None:
            raise ValueError("Shared physics requires a task scene")
        from humanoid_agent.runtime.shared import run_episode_physics

        run_episode_physics(arguments)
    elif arguments.role == "native-controller-check":
        from humanoid_agent.runtime.diagnostic import run_native_controller_check

        run_native_controller_check(arguments)
    elif arguments.role in {"shared-scene-check", "shared-camera-check"}:
        if arguments.task_scene is None or not 2 <= arguments.num_environments <= 16:
            raise ValueError("Shared checks require a task scene and 2–16 environments")
        from humanoid_agent.runtime.checks import (
            check_shared_camera_batch,
            check_shared_workcell_scene,
        )

        asset_root = arguments.isaac_workspace / "SONIC/gear_sonic/data/assets"
        if arguments.role == "shared-scene-check":
            check_shared_workcell_scene(
                asset_root, arguments.task_scene, arguments.output, arguments.num_environments
            )
        else:
            check_shared_camera_batch(
                asset_root,
                arguments.task_scene,
                arguments.output,
                arguments.num_environments,
                arguments.decorative_assets,
            )
    elif arguments.role == "scene-check":
        if arguments.task_scene is None:
            raise ValueError("Scene preflight requires --task-scene")
        from humanoid_agent.runtime.checks import check_workcell_scene

        check_workcell_scene(
            arguments.isaac_workspace / "SONIC/gear_sonic/data/assets",
            arguments.task_scene,
            arguments.output,
        )
    else:
        from humanoid_agent.runtime.operator import run_isaac_model_operator

        run_isaac_model_operator(arguments)


def select_role(*, role: str = "episode") -> None:
    """Select the worker before parsing its distinct command options."""


def main(tokens: list[str] | None = None) -> None:
    role_app = App(default_command=select_role, help_flags=(), suppress_keyboard_interrupt=False)
    _, bound, remaining, _ = role_app.parse_known_args(tokens)
    role = bound.arguments.get("role", "episode")
    if role == "sdk":
        from humanoid_agent.runtime.sdk_worker import main as worker
    elif role == "camera":
        from humanoid_agent.runtime.camera_worker import main as worker
    else:
        App(default_command=run_task, help=__doc__, suppress_keyboard_interrupt=False)(tokens)
        return
    App(default_command=worker, help=worker.__module__, suppress_keyboard_interrupt=False)(
        remaining
    )


def entrypoint() -> None:
    configure_logging()
    main()


if __name__ == "__main__":
    entrypoint()
