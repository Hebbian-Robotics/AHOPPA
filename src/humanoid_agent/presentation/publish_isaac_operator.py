"""Publish paired native-camera recordings with synchronized SONIC references.

One composition path serves original captures and offline pose replay. Overlays
are presentation derivatives; original request PNGs are never modified.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, closing
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from cyclopts import App
from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES, measured_tracking_points
from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.hand_control import HandTransitionInitials
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.presentation import recorded_motion
from humanoid_agent.presentation.media_probe import probe_video
from humanoid_agent.presentation.publish_model_inputs import publish_model_inputs
from humanoid_agent.presentation.recorded_motion import (
    ObserverCalibration,
    observer_calibration_from_capture,
    project_world_points,
    recorded_targets_world,
)
from humanoid_agent.recording import (
    iter_record_bytes,
    iter_records,
    read_record,
    read_record_bytes,
    record_exists,
    recording_sha256,
)
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.sensor_feedback import EncoderKinematics
from humanoid_agent.sonic_planner_input import (
    SonicPlannerTimeline,
    motion_timeline,
    parse_motion_publication,
)

COLORS = ((66, 166, 255), (255, 158, 64), (208, 140, 255))
CAMERA_COMPOSITION_PROTOCOL = "robot-camera-views-sonic-targets-v2"
PRESENTATION_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
if not PRESENTATION_FONT_PATH.is_file():
    PRESENTATION_FONT_PATH = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
FONT = ImageFont.truetype(str(PRESENTATION_FONT_PATH), 15)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parsed_text(response: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed := response.get("output_parsed"), dict):
        return parsed
    messages = [
        content["text"]
        for item in response["output"]
        if item["type"] == "message"
        for content in item["content"]
        if content["type"] == "output_text"
    ]
    if len(messages) != 1:
        raise ValueError("Expected one structured operator decision per response")
    return json.loads(messages[0])


def compose_recorded_frame(
    onboard: Image.Image,
    external: Image.Image,
    *,
    calibration: ObserverCalibration,
    targets_world: np.ndarray | None,
    measured_world: np.ndarray,
    path_world: np.ndarray | None,
    source_seconds: float,
    command_id: str | None,
    reference_age_seconds: float | None,
    representation: str,
    control_mode: str = "Control mode not specified",
    wrist_images: Mapping[str, Image.Image] | None = None,
) -> Image.Image:
    """Keep both views visible and put annotations only on the external panel."""
    if command_id is not None and targets_world is None:
        raise ValueError("An active command must have target-overlay evidence")
    wrist_images = wrist_images or {}
    if wrist_images and set(wrist_images) != {"left_wrist", "right_wrist"}:
        raise ValueError("Wrist-camera recordings require both declared wrist views")
    canvas = Image.new("RGB", (1280, 1160 if wrist_images else 640), "#101010")
    canvas.paste(external.convert("RGB").resize((640, 360)), (0, 100))
    canvas.paste(onboard.convert("RGB").resize((640, 480)), (640, 40))
    draw = ImageDraw.Draw(canvas)
    for index, (name, wrist_image) in enumerate(sorted(wrist_images.items())):
        horizontal_offset = index * 640
        draw.text(
            (horizontal_offset + 10, 650),
            f"{name.replace('_', ' ').capitalize()} RGB (unannotated)",
            fill="white",
            font=FONT,
        )
        canvas.paste(wrist_image.convert("RGB").resize((640, 480)), (horizontal_offset, 680))
    draw.text((10, 10), "External RGB + requested targets", fill="white", font=FONT)
    draw.text((650, 10), "Onboard RGB (unannotated)", fill="white", font=FONT)
    draw.text((12, 42), control_mode, fill="white", font=FONT)
    external_overlay = canvas.crop((0, 0, 640, 460))
    draw = ImageDraw.Draw(external_overlay)
    if targets_world is not None:
        points = project_world_points(calibration, targets_world)
        if path_world is not None:
            paths = project_world_points(calibration, path_world.reshape(-1, 3)).reshape(-1, 3, 2)
            for index, color in enumerate(COLORS):
                draw.line([tuple(point) for point in paths[:, index]], fill=color, width=2)
        measured = project_world_points(calibration, measured_world)
        for index, color in enumerate(COLORS):
            horizontal, vertical = points[index]
            draw.ellipse(
                (horizontal - 6, vertical - 6, horizontal + 6, vertical + 6),
                fill=color,
                outline="black",
                width=2,
            )
            horizontal, vertical = measured[index]
            draw.ellipse(
                (horizontal - 7, vertical - 7, horizontal + 7, vertical + 7),
                outline="white",
                width=2,
            )
    canvas.paste(external_overlay.crop((0, 100, 640, 460)), (0, 100))
    draw = ImageDraw.Draw(canvas)
    for index, (name, color) in enumerate(
        zip(("Left wrist", "Right wrist", "Head / torso"), COLORS, strict=True)
    ):
        draw.text((12 + index * 205, 480), name, fill=color, font=FONT)
    draw.text(
        (12, 505),
        "Filled: requested  |  Rings: measured  |  Lines: local waypoints",
        fill="white",
        font=FONT,
    )
    draw.text(
        (12, 548),
        f"Capture {source_seconds:.3f} s  |  {command_id or 'native reference'}",
        fill="white",
        font=FONT,
    )
    age = (
        "none published yet"
        if reference_age_seconds is None
        else f"{reference_age_seconds:.3f} s old at capture"
    )
    draw.text(
        (12, 571), f"Reference: {age}. Overlays were not model inputs.", fill="white", font=FONT
    )
    draw.text((12, 600), representation, fill="white", font=FONT)
    return canvas


def measured_tracking_world(
    encoder: EncoderKinematics, joint_positions_rad: Sequence[float], root: Sequence[float]
) -> np.ndarray:
    encoder.data.qpos[:] = 0
    encoder.data.qpos[:7] = root
    encoder.data.qpos[encoder.position_indices] = joint_positions_rad
    mujoco: Any = importlib.import_module("mujoco")
    mujoco.mj_forward(encoder.model, encoder.data)
    return np.asarray(measured_tracking_points(encoder.model, encoder.data)[2])


@lru_cache(maxsize=256)
def load_recorded_waypoint_timeline(episode: Path, command_id: str) -> SonicPlannerTimeline:
    """Read immutable completed-episode inputs once; never cache changing poses."""
    ledger = episode / "physics/command-ledger"
    command_path = ledger / "commands" / f"{command_id}.json"
    if not record_exists(command_path):
        command_path = episode / "operator" / command_id / "command.json"
    if not record_exists(command_path):
        raise FileNotFoundError(f"Executed command lacks retained waypoints: {command_id}")
    publication = parse_motion_publication(read_record_bytes(command_path))
    basis = read_record(ledger / "controller-observations" / f"{publication.observation_id}.json")
    initials_path = ledger / "hand-initials" / f"{command_id}.json"
    initials = (
        HandTransitionInitials.model_validate_json(read_record_bytes(initials_path))
        if record_exists(initials_path)
        else None
    )
    return motion_timeline(publication, basis["pelvis_quaternion_wxyz"], hand_initials=initials)


def command_path_world(
    episode: Path, command_id: str | None, root: Sequence[float]
) -> np.ndarray | None:
    if command_id is None:
        return None
    timeline = load_recorded_waypoint_timeline(episode, command_id)
    publication = timeline.publication
    paths = []
    for elapsed in np.linspace(0, timeline.duration_seconds, len(publication.poses)):
        sample = timeline.sample(float(elapsed), root[3:7]).model_dump(mode="json")
        paths.append(recorded_targets_world(sample, root))
    return np.asarray(paths)


def compose_native_capture(
    episode: Path,
    capture: dict[str, Any],
    joint_positions_rad: Sequence[float],
    receipt: dict[str, Any] | None,
    onboard: Image.Image,
    external: Image.Image,
    calibration: ObserverCalibration,
    encoder: EncoderKinematics,
    *,
    representation: str,
    wrist_images: Mapping[str, Image.Image] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    audit = capture["private_camera_audit"]
    root = audit["robot_root_position_world"] + audit["robot_root_quaternion_wxyz"]
    command_id = None if receipt is None else receipt["command_id"]
    targets = None if receipt is None else recorded_targets_world(receipt["sample"], root)
    measured = measured_tracking_world(encoder, joint_positions_rad, root)
    control_mode = (
        "Initial hold: no published reference"
        if receipt is None
        else "Native planner reference: head/hands and locomotion"
        if command_id is None
        else "Hybrid control: head/hands poses and locomotion"
    )
    path = command_path_world(episode, command_id, root)
    canvas = compose_recorded_frame(
        onboard,
        external,
        calibration=calibration,
        targets_world=targets,
        measured_world=measured,
        path_world=path,
        source_seconds=capture["source_elapsed_wall_seconds"],
        command_id=command_id,
        reference_age_seconds=None
        if receipt is None
        else capture["source_elapsed_wall_seconds"] - receipt["elapsed_wall_seconds"],
        representation=representation,
        control_mode=control_mode,
        wrist_images=wrist_images,
    )
    return canvas, {
        "command_id": command_id,
        "targets_world_m": None if targets is None else targets.tolist(),
        "waypoints_world_m": None if path is None else path.tolist(),
        "control_mode": control_mode,
    }


def open_recorded_wrist_views(
    snapshot: Path, capture: dict[str, Any], image_context: ExitStack
) -> dict[str, Image.Image]:
    """Open every declared wrist image and verify its retained capture hash."""
    images = {}
    for name, audit in capture.get("private_wrist_camera_audits", {}).items():
        if name not in {"left", "right"}:
            raise ValueError("Unknown wrist camera in capture evidence")
        path = snapshot / f"{name}_wrist.png"
        if file_sha256(path) != audit["rgb_sha256"]:
            raise ValueError("Wrist image differs from its original capture")
        images[f"{name}_wrist"] = image_context.enter_context(Image.open(path))
    return images


def publish_recording(
    episode: Path,
    destination: Path,
    observer_calibration: Path | None = None,
    *,
    replay: Path | None = None,
) -> None:
    """Use exact retained captures and SDK send timestamps; never rerun task physics."""
    if any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("Recording publication requires ffmpeg and ffprobe on PATH")
    if replay is not None:
        publish_replayed_recording(episode, replay, destination)
        return
    result = read_record(episode / "physics/result.json")
    duration = float(result["elapsed_wall_seconds"])
    camera_root = episode / "physics/camera-output"
    capture_path = camera_root / "capture-records.jsonl"
    captures = sorted(
        iter_records(capture_path), key=lambda record: record["source_elapsed_wall_seconds"]
    )
    captures = [
        capture for capture in captures if capture["source_elapsed_wall_seconds"] < duration
    ]
    if not captures:
        raise ValueError("No recorded camera pairs; render or capture both views before publishing")
    calibration = (
        ObserverCalibration.model_validate_json(observer_calibration.read_text())
        if observer_calibration is not None
        else observer_calibration_from_capture(captures[0])
    )
    if (calibration.image_width, calibration.image_height, calibration.composite_header_pixels) != (
        640,
        360,
        100,
    ):
        raise ValueError("Recording calibration must describe the 640x360 external panel at y=100")
    # Keep only the latest SDK receipt at each camera snapshot; long episodes
    # can contain hundreds of megabytes of dense reference evidence.
    selected = []
    receipt_path = episode / "physics/executed-inputs.jsonl"
    with closing(iter_record_bytes(receipt_path)) as stream:
        upcoming = next(stream, None)
        latest = None
        for capture in captures:
            while upcoming is not None:
                record = json.loads(upcoming)
                if record["elapsed_wall_seconds"] > capture["source_elapsed_wall_seconds"]:
                    break
                latest = record
                upcoming = next(stream, None)
            selected.append(latest)
    destination.mkdir(parents=True, exist_ok=True)
    frames_directory = destination / "recording-frames"
    frames_directory.mkdir(exist_ok=True)
    encoder = EncoderKinematics(episode / "inputs/robot.urdf")
    frame_records = []
    poster_written = False
    concat = []
    for index, (capture, receipt) in enumerate(zip(captures, selected, strict=True)):
        snapshot = camera_root / Path(capture["snapshot_directory"]).name
        for filename, field in (
            ("onboard.png", "onboard_rgb_sha256"),
            ("observer.png", "observer_rgb_sha256"),
        ):
            if file_sha256(snapshot / filename) != capture[field]:
                raise ValueError(f"Original camera hash mismatch: {snapshot / filename}")
        observation = IsaacObservation.model_validate_json(
            (snapshot / "observation.json").read_text()
        )
        with (
            Image.open(snapshot / "onboard.png") as onboard,
            Image.open(snapshot / "observer.png") as external,
            ExitStack() as image_context,
        ):
            canvas, overlay = compose_native_capture(
                episode,
                capture,
                observation.robot_sensors.in_joint_order(JOINT_NAMES).joint_positions_rad,
                receipt,
                onboard,
                external,
                calibration,
                encoder,
                representation="Original paired captures held until next capture; not continuous motion footage.",
                wrist_images=open_recorded_wrist_views(snapshot, capture, image_context),
            )
        frame_path = frames_directory / f"{index:06d}.png"
        canvas.save(frame_path)
        start = 0 if index == 0 else capture["source_elapsed_wall_seconds"]
        end = (
            captures[index + 1]["source_elapsed_wall_seconds"]
            if index + 1 < len(captures)
            else duration
        )
        concat.extend([f"file '{frame_path.resolve()}'", f"duration {end - start:.9f}"])
        frame_records.append(
            {
                "source_elapsed_seconds": capture["source_elapsed_wall_seconds"],
                "source_capture_sha256": hashlib.sha256(
                    json.dumps(capture, sort_keys=True).encode()
                ).hexdigest(),
                "reference_elapsed_seconds": None
                if receipt is None
                else receipt["elapsed_wall_seconds"],
                **overlay,
                "frame_sha256": file_sha256(frame_path),
            }
        )
        if index == 0 or (overlay["command_id"] is not None and not poster_written):
            canvas.save(destination / "poster-targets.jpg", quality=95)
            poster_written = overlay["command_id"] is not None
    last_frame_path = frames_directory / f"{len(captures) - 1:06d}.png"
    concat.append(f"file '{last_frame_path.resolve()}'")
    concat_path = frames_directory / "concat.txt"
    concat_path.write_text("\n".join(concat) + "\n")
    video = destination / "recorded-targets.partial.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            "fps=4,tpad=stop_mode=clone:stop_duration=0.25",
            "-frames:v",
            str(math.ceil(duration * 4)),
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = probe_video(video).video
    if (
        probe.frame_count != math.ceil(duration * 4)
        or abs(probe.duration_seconds - duration) > 0.251
    ):
        raise ValueError("Recording does not preserve the full episode duration")
    final_video = destination / "recorded-targets.mp4"
    video.replace(final_video)
    video = final_video
    atomic_json(
        destination / "recording-audit.json",
        {
            "protocol": CAMERA_COMPOSITION_PROTOCOL,
            "publisher_source_sha256": file_sha256(Path(__file__)),
            "target_geometry_source_sha256": file_sha256(Path(recorded_motion.__file__)),
            "episode": str(episode.resolve()),
            "video_sha256": file_sha256(video),
            "capture_records_sha256": recording_sha256(capture_path),
            "sdk_receipts_sha256": recording_sha256(receipt_path),
            "calibration": calibration.model_dump(),
            "calibration_sha256": file_sha256(observer_calibration)
            if observer_calibration is not None
            else None,
            "frames": frame_records,
            "model_inputs_modified": False,
            "task_physics_steps": 0,
            "model_requests": 0,
        },
    )
    finish_recording_publication(
        episode, destination, video.name, "poster-targets.jpg", probe.duration_seconds, replay=False
    )


def finish_recording_publication(
    episode: Path,
    destination: Path,
    video_name: str,
    poster_name: str,
    duration_seconds: float,
    *,
    replay: bool,
) -> None:
    result = json.loads((episode / "physics/result.json").read_text())
    status_path = destination / "status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {
            "status": "completed",
            "message": f"Execution ended: {result['finish_reason']}. Task success: {result['task_success']}. Maximum lag: {1000 * result['maximum_physics_clock_lag_seconds']:.1f} ms.",
        }
    )
    status.update(
        video=f"transfer/{destination.name}/{video_name}?v={file_sha256(destination / video_name)[:12]}",
        poster=f"transfer/{destination.name}/{poster_name}?v={file_sha256(destination / poster_name)[:12]}",
    )
    atomic_json(status_path, status)
    if (episode / "operator").exists():
        publish_model_inputs(
            episode / "operator", destination, asset_prefix=f"transfer/{destination.name}"
        )
    publish_recording_metadata(episode, destination, duration_seconds, replay=replay)


def publish_replayed_recording(episode: Path, replay: Path, destination: Path) -> None:
    render = json.loads((replay / "render.json").read_text())
    trace = episode / "physics/physical-trace.jsonl"
    if (
        render.get("source_trace_sha256") != recording_sha256(trace)
        or render.get("composition_protocol") != CAMERA_COMPOSITION_PROTOCOL
    ):
        raise ValueError(
            "Replay lacks matching trace and compositor provenance; render with the current pipeline"
        )
    source_video = replay / "combined.mp4"
    probe = probe_video(source_video).video
    if (
        probe.frame_count != len(render["frames"])
        or float(probe.frame_rate) != render["frames_per_second"]
        or [probe.width, probe.height] != render["layout_width_height"]
    ):
        raise ValueError("Replay frame count, rate or paired-view layout differs from its audit")
    destination.mkdir(parents=True, exist_ok=True)
    temporary_video = destination / "recorded-targets.partial.mp4"
    recompose_replay_recording(episode, replay, render, temporary_video)
    refreshed_probe = probe_video(temporary_video).video
    if (
        refreshed_probe.frame_count != probe.frame_count
        or refreshed_probe.duration_seconds != probe.duration_seconds
        or (refreshed_probe.width, refreshed_probe.height) != (probe.width, probe.height)
    ):
        raise ValueError("Refreshed annotations changed replay dimensions or clock")
    video = destination / "recorded-targets.mp4"
    temporary_video.replace(video)
    poster = destination / "poster-targets.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(probe.duration_seconds / 2),
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(poster),
        ],
        check=True,
    )
    atomic_json(
        destination / "recording-audit.json",
        {
            "protocol": CAMERA_COMPOSITION_PROTOCOL,
            "source": "offline-pose-replay",
            "render": render,
            "publisher_source_sha256": file_sha256(Path(__file__)),
            "geometry_source_sha256": file_sha256(Path(recorded_motion.__file__)),
            "raw_view_sha256": {
                name: file_sha256(replay / f"{name}.mp4") for name in render["views"]
            },
            "video_sha256": file_sha256(video),
            "task_physics_steps": 0,
            "model_requests": 0,
            "model_inputs_modified": False,
        },
    )
    finish_recording_publication(
        episode, destination, video.name, poster.name, probe.duration_seconds, replay=True
    )


def recompose_replay_recording(
    episode: Path, replay: Path, render: dict[str, Any], destination: Path
) -> None:
    """Refresh the common overlay from retained raw views, without GPU rendering."""
    imageio: Any = importlib.import_module("imageio.v2")
    view_names = render["views"]
    if view_names not in (
        ["onboard", "observer"],
        ["onboard", "observer", "left_wrist", "right_wrist"],
    ):
        raise ValueError("Replay must retain the declared robot camera views in order")
    for name in view_names:
        view = probe_video(replay / f"{name}.mp4").video
        if (
            view.frame_count != len(render["frames"])
            or float(view.frame_rate) != render["frames_per_second"]
        ):
            raise ValueError("Raw replay views do not match the retained frame clock")
    capture_path = (
        replay / "capture-records.jsonl"
        if (replay / "records.mcap").exists()
        else replay / "replay-captures.jsonl"
    )
    captures = {
        capture["source_elapsed_wall_seconds"]: capture for capture in iter_records(capture_path)
    }
    required_times = {frame["source_wall_seconds"] for frame in render["frames"]}
    poses = {}
    with closing(iter_record_bytes(episode / "physics/physical-trace.jsonl")) as trace_stream:
        for line in trace_stream:
            row = json.loads(line)
            if row["wall_seconds"] in required_times:
                poses[row["wall_seconds"]] = row
    encoder = EncoderKinematics(episode / "inputs/robot.urdf")
    readers = [imageio.get_reader(str(replay / f"{name}.mp4")) for name in view_names]
    writer = imageio.get_writer(
        str(destination),
        fps=render["frames_per_second"],
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
    )
    try:
        with closing(iter_record_bytes(episode / "physics/executed-inputs.jsonl")) as receipts:
            upcoming = next(receipts, None)
            latest = None
            for frame in render["frames"]:
                source_seconds = frame["source_wall_seconds"]
                while upcoming is not None:
                    receipt = json.loads(upcoming)
                    if receipt["elapsed_wall_seconds"] > source_seconds:
                        break
                    latest = receipt
                    upcoming = next(receipts, None)
                pose = poses[source_seconds]
                joint_positions = [
                    pose["joint_positions"][pose["joint_names"].index(name)] for name in JOINT_NAMES
                ]
                canvas, _ = compose_native_capture(
                    episode,
                    captures[source_seconds],
                    joint_positions,
                    latest,
                    Image.fromarray(readers[0].get_next_data()),
                    Image.fromarray(readers[1].get_next_data()),
                    observer_calibration_from_capture(captures[source_seconds]),
                    encoder,
                    representation="Offline pose replay; these additional frames were not model observations.",
                    wrist_images={
                        name: Image.fromarray(reader.get_next_data())
                        for name, reader in zip(view_names[2:], readers[2:], strict=True)
                    },
                )
                writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def model_decision_timeline(operator: Path, duration_seconds: float) -> list[dict[str, Any]]:
    clock_path = operator / "episode-clock.json"
    if not clock_path.exists():
        return []
    clock = EpisodeClock.model_validate_json(clock_path.read_text())
    release = clock.deadline_monotonic_seconds - clock.deadline_elapsed_wall_seconds
    decisions = []
    for response_path in sorted(operator.glob("*/model.response.json")):
        timing_path = response_path.parent / "model-timing.json"
        if not timing_path.exists():
            continue
        response = json.loads(response_path.read_text()).get("output_parsed")
        if not response or "subtasks" not in response:
            continue
        response_at = (
            json.loads(timing_path.read_text())["response_received_monotonic_seconds"] - release
        )
        if not 0 <= response_at <= duration_seconds:
            continue
        decisions.append(
            {
                "identifier": response_path.parent.name,
                "at": response_at,
                "progress": response.get("observed_progress", "Initial plan received."),
                "subtasks": response["subtasks"],
            }
        )
    return sorted(decisions, key=lambda decision: decision["at"])


def publish_recording_metadata(
    episode: Path, destination: Path, video_duration_seconds: float, *, replay: bool = False
) -> None:
    prefix = f"transfer/{destination.name}"
    physical_duration_seconds = json.loads((episode / "physics/result.json").read_text())[
        "elapsed_wall_seconds"
    ]
    decisions = model_decision_timeline(episode / "operator", physical_duration_seconds)
    clock = EpisodeClock.model_validate_json(
        read_record_bytes(episode / "physics/episode-clock.json")
    )
    # Preserve UUID links already published for these episodes.
    record = {
        "identifier": str(uuid.uuid5(uuid.NAMESPACE_URL, f"webolt/{clock.episode_id}")),
        "archived": False,
        "title": destination.name.replace("-", " "),
        "statusPath": f"{prefix}/status.json",
        "description": "Recorded native SONIC episode; see its retained physical outcome and timing classification.",
        "videoDescription": "Recorded head/external views and any enabled wrist views, with requested targets, measured rings and local waypoint paths on the external view. Annotations were not model inputs. Sparse captures hold until the next capture; playback preserves the episode clock.",
        "durationSeconds": video_duration_seconds,
        "phases": [{"name": "start", "at": 0.0}]
        + [{"name": decision["identifier"], "at": decision["at"]} for decision in decisions],
        "decisions": decisions,
    }
    if replay:
        replay_frame_rate = json.loads((destination / "recording-audit.json").read_text())[
            "render"
        ]["frames_per_second"]
        record["videoDescription"] = (
            f"Uniform {replay_frame_rate:g} fps offline replay of recorded poses with all enabled camera views, requested targets, measured rings and local waypoint paths. Each frame carries an explicit control-mode label. These additional frames were not model observations; original request images remain in the image panel."
        )
    if (destination / "model-inputs.json").exists():
        record["modelInputsDirectory"] = prefix
    manifest_path = destination.parent / "manifest.json"
    if manifest_path.exists():
        with (destination.parent / ".recording-publication.lock").open("a") as publication_lock:
            fcntl.flock(publication_lock.fileno(), fcntl.LOCK_EX)
            manifest = json.loads(manifest_path.read_text())
            previous = next(
                (
                    item
                    for item in manifest["records"]
                    if item["statusPath"] == record["statusPath"]
                ),
                None,
            )
            if previous is not None:
                # Preserve reviewed names, UUID links and integration-defect warnings.
                record = {
                    **record,
                    **previous,
                    **{
                        name: record[name]
                        for name in ("durationSeconds", "phases", "decisions", "videoDescription")
                    },
                }
                manifest["records"] = [
                    record if item is previous else item for item in manifest["records"]
                ]
            else:
                manifest["records"].insert(0, record)
            atomic_json(manifest_path, manifest)
    atomic_json(destination / "presentation-record.json", record)


def main(
    episode: Path,
    destination: Path,
    *,
    observer_calibration: Path | None = None,
    replay: Path | None = None,
) -> None:
    publish_recording(episode, destination, observer_calibration, replay=replay)


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
