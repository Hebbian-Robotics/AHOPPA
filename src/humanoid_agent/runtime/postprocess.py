"""Explicit local analysis and offline replay; neither changes episode outcomes."""

from __future__ import annotations

import importlib
import math
import shutil
import tempfile
import time
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.experiment_records import load_episode_record, validate_physical_record
from humanoid_agent.presentation.publish_isaac_operator import (
    CAMERA_COMPOSITION_PROTOCOL,
    compose_native_capture,
    open_recorded_wrist_views,
)
from humanoid_agent.presentation.recorded_motion import observer_calibration_from_capture
from humanoid_agent.recording import iter_records, read_record, recording_sha256
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.sensor_feedback import EncoderKinematics


def analyze_episode(episode: Path, output: Path) -> None:
    """Summarize recorded physical evidence without inferring model success."""
    terminal = load_episode_record(episode)
    physical_result = validate_physical_record(episode, terminal)
    physics = episode / "physics"
    rows = list(iter_records(physics / "physical-trace.jsonl"))
    if not rows:
        raise ValueError("Episode has no physical trace")
    output.mkdir(parents=True, exist_ok=False)
    result = physical_result.model_dump(mode="json")
    atomic_json(
        output / "analysis.json",
        {
            "episode": str(episode.resolve()),
            "execution_status": terminal.status,
            "simulation_validity": terminal.simulation_validity.model_dump(mode="json"),
            "task_outcome": read_record(physics / "task-outcome.json"),
            "physics_result": result,
            "sampled_root_height_minimum_m": min(row["root_position"][2] for row in rows),
            "sampled_trace_rows": len(rows),
            "sampled_maximum_wall_minus_simulation_seconds": max(
                row["wall_seconds"] - row["simulation_seconds"] for row in rows
            ),
            "scope": "One recorded episode; sampled trace quantities are not all-tick bounds or statistical estimates",
        },
    )


def render_episode(episode: Path, output: Path, asset_root: Path, frames_per_second: int) -> None:
    """Replay saved poses at wall-clock speed with no task physics or model calls."""
    from humanoid_agent.runtime.camera import NativeCameraRenderer, NativeVisualState
    from humanoid_agent.runtime.environment import (
        create_newton_simulation_configuration,
        launch_newton_runtime,
        register_newton_scene_schemas,
    )

    terminal = load_episode_record(episode)
    physical_result = validate_physical_record(episode, terminal)
    physics = episode / "physics"
    rows = list(iter_records(physics / "physical-trace.jsonl"))
    if not rows or any(b["wall_seconds"] <= a["wall_seconds"] for a, b in pairwise(rows)):
        raise ValueError("Offline replay requires a nonempty chronological physical trace")
    elapsed = physical_result.elapsed_wall_seconds
    if elapsed <= 0:
        raise ValueError("Episode has no positive wall-clock duration")
    output.mkdir(parents=True, exist_ok=False)
    register_newton_scene_schemas(rendering=True)
    configuration = create_newton_simulation_configuration()
    imageio: Any = importlib.import_module("imageio.v2")
    renderer = None
    writers = []
    mapping = []
    layout_width_height = [1280, 640]
    encoder = EncoderKinematics(episode / "inputs/robot.urdf")
    receipt_stream = iter_records(physics / "executed-inputs.jsonl")
    pending_receipt_line = next(receipt_stream, None)
    latest_receipt = None
    try:
        with (
            launch_newton_runtime(configuration),
            tempfile.TemporaryDirectory(prefix="humanoid-agent-replay-") as temporary,
        ):
            temporary_output = Path(temporary) / "frames"
            appearance = episode / "inputs/appearance"
            renderer = NativeCameraRenderer(
                asset_root,
                episode / "inputs/scene.json",
                appearance if appearance.exists() else None,
                temporary_output,
                simulation_configuration=configuration,
                retain_capture_media=False,
            )
            layout_width_height = [1280, 1160 if renderer.wrist_cameras else 640]
            view_names = [
                "onboard",
                "observer",
                *(f"{side}_wrist" for side in renderer.wrist_cameras),
            ]
            for name in view_names:
                writers.append(
                    imageio.get_writer(
                        str(output / f"{name}.mp4"),
                        fps=frames_per_second,
                        codec="libx264",
                        macro_block_size=1,
                        ffmpeg_params=["-threads", "2"],
                    )
                )
            writers.append(
                imageio.get_writer(
                    str(output / "combined.mp4"),
                    fps=frames_per_second,
                    codec="libx264",
                    macro_block_size=1,
                    ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
                )
            )
            selected_index = 0
            images = None
            previous_index = None
            for frame_index in range(math.ceil(elapsed * frames_per_second)):
                wall_seconds = frame_index / frames_per_second
                while (
                    selected_index + 1 < len(rows)
                    and rows[selected_index + 1]["wall_seconds"] <= wall_seconds
                ):
                    selected_index += 1
                row = rows[selected_index]
                if selected_index != previous_index:
                    identifier = f"frame_{selected_index:06d}"
                    capture = renderer.capture(
                        NativeVisualState(
                            identifier=identifier,
                            episode_id="offline_replay",
                            episode_deadline_elapsed_wall_seconds=elapsed,
                            simulation_seconds=row["simulation_seconds"],
                            elapsed_wall_seconds=row["wall_seconds"],
                            source_monotonic_seconds=time.monotonic(),
                            joint_names=row["joint_names"],
                            joint_positions=row["joint_positions"],
                            joint_velocities=row["joint_velocities"],
                            root_state_world=row["root_state"],
                            cabinet_joint_positions=row.get("cabinet_joint_positions"),
                            indicator_activated=row["activated"],
                        )
                    )
                    snapshot = temporary_output / identifier
                    while pending_receipt_line is not None:
                        receipt = pending_receipt_line
                        if receipt["elapsed_wall_seconds"] > row["wall_seconds"]:
                            break
                        latest_receipt = receipt
                        pending_receipt_line = next(receipt_stream, None)
                    observation = IsaacObservation.model_validate_json(
                        (snapshot / "observation.json").read_text()
                    )
                    with (
                        Image.open(snapshot / "onboard.png") as onboard,
                        Image.open(snapshot / "observer.png") as external,
                        ExitStack() as image_context,
                    ):
                        wrist_images = open_recorded_wrist_views(snapshot, capture, image_context)
                        canvas, _ = compose_native_capture(
                            episode,
                            capture,
                            observation.robot_sensors.in_joint_order(
                                JOINT_NAMES
                            ).joint_positions_rad,
                            latest_receipt,
                            onboard,
                            external,
                            observer_calibration_from_capture(capture),
                            encoder,
                            representation="Offline pose replay; these additional frames were not model observations.",
                            wrist_images=wrist_images,
                        )
                        images = [
                            np.asarray(onboard.convert("RGB")),
                            np.asarray(external.convert("RGB")),
                            *(
                                np.asarray(wrist_images[name].convert("RGB"))
                                for name in view_names[2:]
                            ),
                            np.asarray(canvas),
                        ]
                    shutil.rmtree(snapshot)
                    previous_index = selected_index
                for writer, image in zip(writers, images or [], strict=True):
                    writer.append_data(image)
                mapping.append(
                    {
                        "video_seconds": wall_seconds,
                        "source_wall_seconds": row["wall_seconds"],
                        "command_id": row["command_id"],
                    }
                )
            renderer.close()
            renderer = None
            shutil.copyfile(temporary_output / "records.mcap", output / "records.mcap")
    finally:
        if renderer is not None:
            renderer.close()
        for writer in writers:
            writer.close()
        receipt_stream.close()
    atomic_json(
        output / "render.json",
        {
            "episode": str(episode.resolve()),
            "source_trace_sha256": recording_sha256(physics / "physical-trace.jsonl"),
            "composition_protocol": CAMERA_COMPOSITION_PROTOCOL,
            "views": view_names,
            "layout_width_height": layout_width_height,
            "frames_per_second": frames_per_second,
            "frames": mapping,
            "task_physics_steps": 0,
            "model_requests": 0,
            "scope": "Offline pose replay for presentation; these frames are not model-request images",
            "temporary_pose_images_retained": False,
        },
    )
