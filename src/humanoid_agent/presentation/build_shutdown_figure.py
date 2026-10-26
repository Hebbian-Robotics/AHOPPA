"""Compose a shutdown-round figure from the exact images supplied to the model.

Each panel is one retained request image (onboard ``rgb.png`` or ``external.png``)
or the episode's frozen facility map; nothing is re-rendered. The provenance
file records each source path, SHA-256 and elapsed time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyclopts import App
from PIL import Image, ImageDraw, ImageFont

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.facility_map import FACILITY_MAP_FILENAME
from humanoid_agent.logging_config import configure_logging

PANEL_HEIGHT = 360


def panel_source(episode: Path, panel: str) -> tuple[Path, float | None]:
    """``map`` or ``REQUEST:onboard|external`` under the episode's operator records."""
    if panel == "map":
        return episode / "inputs" / FACILITY_MAP_FILENAME, None
    request, view = panel.split(":")
    directory = episode / "operator" / request
    observation = json.loads((directory / "observation.json").read_text())
    filename = {"onboard": "rgb.png", "external": "external.png"}[view]
    return directory / filename, float(observation["elapsed_wall_seconds"])


def main(episode: Path, /, *panel_labels: str, output: Path, label_font_size: int = 22) -> None:
    """Write OUTPUT (PNG) and OUTPUT.provenance.json from ``SOURCE=LABEL`` panels."""
    if output.exists():
        raise FileExistsError(output)
    images = []
    provenance = []
    for panel_label in panel_labels:
        panel, label = panel_label.split("=", 1)
        path, elapsed = panel_source(episode, panel)
        data = path.read_bytes()
        with Image.open(path) as source:
            image = source.convert("RGB")
        width = round(image.width * PANEL_HEIGHT / image.height)
        image = image.resize((width, PANEL_HEIGHT), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=label_font_size)
        text = label if elapsed is None else f"{label}  t={elapsed:.0f} s"
        box = draw.textbbox((10, 8), text, font=font)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(255, 255, 255))
        draw.text((10, 8), text, fill=(0, 0, 0), font=font)
        images.append(image)
        provenance.append(
            {
                "panel": panel,
                "label": label,
                "source": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "elapsed_wall_seconds": elapsed,
            }
        )
    gap = 8
    strip = Image.new(
        "RGB",
        (sum(image.width for image in images) + gap * (len(images) - 1), PANEL_HEIGHT),
        "white",
    )
    left = 0
    for image in images:
        strip.paste(image, (left, 0))
        left += image.width + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output)
    atomic_json(
        output.with_suffix(".provenance.json"),
        {
            "episode": str(episode),
            "panels": provenance,
            "model_inputs_unmodified_except_resize": True,
            "label_font_size": label_font_size,
        },
    )


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
