"""Execution-clock image selection and ordinary bounded action-history records."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from humanoid_agent.finite_command import Identifier
from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel

OBSERVATION_INTERFACE_PROTOCOL = "paired-action-history-initial-reference-v2"
RECENT_ACTION_HISTORY_LENGTH = 3
HistoryRole = Literal["start", "midpoint", "current/end"]


class ActionExecutionWindow(ValidatedModel):
    action_id: Identifier
    started_elapsed_seconds: FiniteFloat = Field(ge=0)
    endpoint_elapsed_seconds: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def require_ordered_window(self) -> ActionExecutionWindow:
        if self.endpoint_elapsed_seconds < self.started_elapsed_seconds:
            raise ValueError("Action endpoint precedes first native publication")
        return self

    def capture_identifier(self, role: Literal["start", "midpoint"]) -> str:
        return f"{self.action_id}_history_{role}"

    def capture_time(self, role: Literal["start", "midpoint"]) -> float:
        return (
            self.started_elapsed_seconds
            if role == "start"
            else (self.started_elapsed_seconds + self.endpoint_elapsed_seconds) / 2
        )


class ActionFrameContext(ValidatedModel):
    execution: ActionExecutionWindow
    role: HistoryRole
    requested_elapsed_seconds: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def require_execution_clock_target(self) -> ActionFrameContext:
        expected = (
            self.execution.endpoint_elapsed_seconds
            if self.role == "current/end"
            else self.execution.capture_time(self.role)
        )
        if abs(self.requested_elapsed_seconds - expected) > 1e-9:
            raise ValueError("History target differs from the action execution clock")
        return self


class ImageFrame(ValidatedModel):
    observation_id: Identifier
    action_id: Identifier | None
    role: HistoryRole | Literal["episode/start"]
    captured_elapsed_seconds: FiniteFloat = Field(ge=0)
    requested_elapsed_seconds: FiniteFloat = Field(ge=0)


class CameraImage(ValidatedModel):
    camera: Literal["onboard", "external", "left_wrist", "right_wrist"]
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @property
    def label(self) -> str:
        return f"{self.camera.replace('_', ' ')} RGB"


class ImagePair(ValidatedModel):
    frames: ImmutableSequence[ImageFrame] = Field(min_length=1)
    onboard_path: str
    external_path: str
    onboard_sha256: str
    external_sha256: str
    wrist_views: ImmutableSequence[CameraImage] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def require_distinct_wrist_views(self) -> ImagePair:
        names = [view.camera for view in self.wrist_views]
        if len(set(names)) != len(names) or any(
            name not in {"left_wrist", "right_wrist"} for name in names
        ):
            raise ValueError("Additional camera images must be distinct wrist views")
        return self

    @property
    def views(self) -> list[CameraImage]:
        return [
            CameraImage(camera="onboard", path=self.onboard_path, sha256=self.onboard_sha256),
            CameraImage(camera="external", path=self.external_path, sha256=self.external_sha256),
            *self.wrist_views,
        ]

    @property
    def current(self) -> bool:
        return any(frame.role == "current/end" for frame in self.frames)


def prepare_image_pairs(
    observations: list[tuple[Path, ImageFrame]], destination: Path
) -> list[ImagePair]:
    """Retain originals; upload an identical synchronized image pair only once.

    Duplicate pairs retain all source roles/timestamps and the most recent files.
    No contact/evaluator signals participate in selection.
    """
    pairs: dict[tuple[tuple[str, str], ...], ImagePair] = {}
    previous_time = -1.0
    episode_id = None
    for directory, frame in observations:
        raw = json.loads((directory / "observation.json").read_text())
        identity = raw["observation_identity"]
        if (
            abs(raw["elapsed_wall_seconds"] - frame.captured_elapsed_seconds) > 1e-9
            or identity["identifier"] != frame.observation_id
            or abs(identity["captured_elapsed_wall_seconds"] - frame.captured_elapsed_seconds)
            > 1e-9
        ):
            raise ValueError("Image frame identity or timestamp differs from its sensor record")
        if episode_id is not None and identity["episode_id"] != episode_id:
            raise ValueError("Visual history mixes episodes")
        episode_id = identity["episode_id"]
        if frame.captured_elapsed_seconds < previous_time:
            raise ValueError("Visual history frames must be chronological")
        previous_time = frame.captured_elapsed_seconds
        if not raw.get("external_camera_available"):
            raise ValueError("Visual history requires synchronized onboard and external RGB")
        paths = directory / "rgb.png", directory / "external.png"
        hashes = (
            hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        )
        wrist_views = [
            CameraImage(
                camera=name,
                path=(directory / f"{name}.png").relative_to(destination).as_posix(),
                sha256=hashlib.sha256((directory / f"{name}.png").read_bytes()).hexdigest(),
            )
            for name in raw.get("wrist_camera_names", [])
        ]
        image_identity = (
            ("onboard", hashes[0]),
            ("external", hashes[1]),
            *((view.camera, view.sha256) for view in wrist_views),
        )
        preceding = pairs.pop(image_identity, None)
        pair = ImagePair(
            frames=[*(preceding.frames if preceding else []), frame],
            onboard_path=paths[0].relative_to(destination).as_posix(),
            external_path=paths[1].relative_to(destination).as_posix(),
            onboard_sha256=hashes[0],
            external_sha256=hashes[1],
            wrist_views=wrist_views,
        )
        pairs[image_identity] = pair
    return list(pairs.values())


def image_pair_content(pairs: list[ImagePair], destination: Path) -> list[dict[str, str]]:
    """Interleave timestamp/view labels with every unmodified camera image."""
    content = []
    for pair in pairs:
        content.append(
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "synchronized_pair": [
                            frame.model_dump(mode="json") for frame in pair.frames
                        ],
                        "view_order": [view.label for view in pair.views],
                        "identical_pairs_deduplicated": len(pair.frames) > 1,
                    }
                ),
            }
        )
        for view in pair.views:
            image_path = (destination / view.path).resolve()
            if not image_path.is_relative_to(destination.resolve()):
                raise ValueError("Model image path escapes the observation record")
            image_bytes = image_path.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != view.sha256:
                raise ValueError("Model image differs from its recorded camera hash")
            encoded = base64.b64encode(image_bytes).decode()
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
    return content


def collect_visual_history(
    current_directory: Path, observations_directory: Path, *, include_episode_start: bool = False
) -> list[ImagePair]:
    """Retain scheduled action frames and an optional fixed initial visual reference."""
    import shutil

    raw = json.loads((current_directory / "observation.json").read_text())
    context = (
        ActionFrameContext.model_validate(raw["action_frame"]) if raw.get("action_frame") else None
    )
    selected = []
    if include_episode_start and raw["observation_identity"]["identifier"] != "initial":
        source = observations_directory / "initial"
        initial = json.loads((source / "observation.json").read_text())
        destination = current_directory / "history" / "initial"
        destination.mkdir(parents=True, exist_ok=False)
        for filename in (
            "rgb.png",
            "external.png",
            "observation.json",
            *(f"{name}.png" for name in initial.get("wrist_camera_names", [])),
        ):
            shutil.copyfile(source / filename, destination / filename)
        selected.append(
            (
                destination,
                ImageFrame(
                    observation_id=initial["observation_identity"]["identifier"],
                    action_id=None,
                    role="episode/start",
                    captured_elapsed_seconds=initial["elapsed_wall_seconds"],
                    requested_elapsed_seconds=initial["elapsed_wall_seconds"],
                ),
            )
        )
    if context is not None and context.role != "current/end":
        raise ValueError("Current model observation must be the post-action endpoint")
    if context is not None:
        for role in ("start", "midpoint"):
            identifier = context.execution.capture_identifier(role)
            source = observations_directory / identifier
            frame_record = json.loads((source / "observation.json").read_text())
            frame_context = ActionFrameContext.model_validate(frame_record["action_frame"])
            if frame_context.execution != context.execution or frame_context.role != role:
                raise ValueError("Visual history frame belongs to a different action or phase")
            destination = current_directory / "history" / identifier
            destination.mkdir(parents=True, exist_ok=False)
            for filename in (
                "rgb.png",
                "external.png",
                "observation.json",
                *(f"{name}.png" for name in frame_record.get("wrist_camera_names", [])),
            ):
                shutil.copyfile(source / filename, destination / filename)
            selected.append(
                (
                    destination,
                    ImageFrame(
                        observation_id=identifier,
                        action_id=context.execution.action_id,
                        role=role,
                        captured_elapsed_seconds=frame_record["elapsed_wall_seconds"],
                        requested_elapsed_seconds=frame_context.requested_elapsed_seconds,
                    ),
                )
            )
    selected.append(
        (
            current_directory,
            ImageFrame(
                observation_id=raw["observation_identity"]["identifier"],
                action_id=context.execution.action_id if context else None,
                role="current/end",
                captured_elapsed_seconds=raw["elapsed_wall_seconds"],
                requested_elapsed_seconds=context.requested_elapsed_seconds
                if context
                else raw["elapsed_wall_seconds"],
            ),
        )
    )
    return prepare_image_pairs(selected, current_directory)
