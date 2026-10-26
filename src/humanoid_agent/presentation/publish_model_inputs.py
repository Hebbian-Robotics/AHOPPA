"""Publish original uploaded camera images separately from presentation footage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.observation_history import ImageFrame, ImagePair


class ModelInputImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ModelInputPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: list[ImageFrame]
    images: list[ModelInputImage]


class ModelInputSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str
    captured_wall_seconds: FiniteFloat = Field(ge=0)
    request_wall_seconds: FiniteFloat = Field(ge=0)
    images: list[ModelInputImage]
    pairs: list[ModelInputPair] = Field(default_factory=list)
    previous_action_feedback: dict[str, Any] | None = None


def publish_model_inputs(
    operator: Path, destination: Path, *, asset_prefix: str | None = None
) -> None:
    """Copy saved request images without resizing, overlays or video compression."""
    snapshots = []
    initial_image = operator / "initial/rgb.png"
    for timing_path in sorted(operator.glob("*/model-timing.json")):
        request_directory = timing_path.parent
        prompt_path = request_directory / "prompt.json"
        prompt = json.loads(prompt_path.read_text())
        observation = prompt["observation"]
        history_path = request_directory / "visual-history.json"
        pairs = (
            [
                ImagePair.model_validate(pair)
                for pair in json.loads(history_path.read_text())["pairs"]
            ]
            if history_path.exists()
            else []
        )
        if not pairs and observation["image_order"] != ["initial onboard", "current onboard"]:
            raise ValueError(
                f"Unsupported image order in historical request {prompt_path}; do not guess model inputs"
            )
        timing = json.loads(timing_path.read_text())
        published_images = []
        published_pairs = []
        if pairs:
            for pair in pairs:
                pair_images = []
                for view in pair.views:
                    source = (request_directory / view.path).resolve()
                    if not source.is_relative_to(request_directory.resolve()):
                        raise ValueError("Model image path escapes its saved request")
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    if digest != view.sha256:
                        raise ValueError("Saved model image differs from its request manifest")
                    relative_asset = Path("model-inputs") / f"{digest}.png"
                    exported_path = destination / relative_asset
                    exported_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, exported_path)
                    pair_images.append(
                        ModelInputImage(
                            label=view.label[0].upper() + view.label[1:],
                            path=f"{asset_prefix or destination.name}/{relative_asset.as_posix()}",
                            sha256=digest,
                        )
                    )
                published_pairs.append(ModelInputPair(frames=list(pair.frames), images=pair_images))
                published_images.extend(pair_images)
        else:
            for label, image_path in [
                ("Initial onboard image", initial_image),
                ("Current onboard image", request_directory / "rgb.png"),
            ]:
                image_bytes = image_path.read_bytes()
                digest = hashlib.sha256(image_bytes).hexdigest()
                relative_path = Path("model-inputs") / f"{digest}.png"
                exported_path = destination / relative_path
                exported_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, exported_path)
                if hashlib.sha256(exported_path.read_bytes()).hexdigest() != digest:
                    raise ValueError("Exported model image differs from the saved upload")
                published_images.append(
                    ModelInputImage(
                        label=label,
                        path=f"{asset_prefix or destination.name}/{relative_path.as_posix()}",
                        sha256=digest,
                    )
                )
        snapshots.append(
            ModelInputSnapshot(
                identifier=request_directory.name,
                captured_wall_seconds=observation["robot_feedback"]["captured_elapsed_seconds"]
                if pairs
                else observation["elapsed_wall_seconds"],
                request_wall_seconds=timing["model_request_wall_seconds"],
                images=published_images,
                pairs=published_pairs,
                previous_action_feedback=prompt.get("recent_decisions", [{}])[-1].get("feedback")
                if prompt.get("recent_decisions")
                else None,
            )
        )
    atomic_json(
        destination / "model-inputs.json",
        {
            "provenance": "Original PNG upload files, identified using saved prompts and operator image order; byte hashes verified on export. Provider-internal image processing is not observable. Only completed request groups with recorded timing are included.",
            "snapshots": [snapshot.model_dump() for snapshot in snapshots],
        },
    )


def main(
    cli_operator: Annotated[Path, Parameter(name="operator")],
    cli_destination: Annotated[Path, Parameter(name="destination")],
    /,
) -> None:
    publish_model_inputs(cli_operator, cli_destination)


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
