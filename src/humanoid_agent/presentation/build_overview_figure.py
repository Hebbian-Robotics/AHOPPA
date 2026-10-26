"""Compose the editable method figure from one recorded model request and its execution.

Every scene image is a retained model input; nothing is re-rendered. The plan,
targets and motor commands come from the same request's response, command and
physical trace, so all panels describe one executed decision.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
from cyclopts import App
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.observation_history import ImagePair
from humanoid_agent.recording import iter_records, recording_location, recording_sha256

TRACKING_POINT_STYLES = {
    "left_wrist": ("Left wrist", "#2277bb"),
    "right_wrist": ("Right wrist", "#dc7633"),
    "head": ("Head", "#8954ad"),
}
TRACKING_POINT_ORDER = tuple(TRACKING_POINT_STYLES)
HISTORY_ROLE_LABELS = {
    "episode/start": "Start",
    "start": "Chunk start",
    "midpoint": "Midpoint",
    "current/end": "Current",
}
SUBTASK_SHORT_LABELS = {
    "inspect_A": "Inspect drawer A",
    "close_A_if_needed": "Close A if open",
    "inspect_B": "Inspect drawer B",
    "close_B_if_needed": "Close B if open",
    "return_home": "Return home",
}
TASK_INSTRUCTION_SHORT = "“Close open drawers, then return home.”"
MOTOR_PLOT_HOLD_SECONDS = 2.4
PLOTTED_MOTOR_JOINTS = (
    ("left_shoulder_pitch_joint", "Shoulder", "#2277bb"),
    ("left_elbow_joint", "Elbow", "#dc7633"),
    ("left_hip_pitch_joint", "Hip", "#26897e"),
    ("left_knee_joint", "Knee", "#8954ad"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embedded_image(path: Path, x: float, y: float, width: float, height: float) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<image x="{x}" y="{y}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{encoded}"/>'


def image_aspect_ratio(path: Path) -> float:
    with Image.open(path) as image:
        return image.width / image.height


def project_heading_points(
    heading_points: Any, kinematic_feedback: dict[str, Any], intrinsic_matrix: Any
) -> Any | None:
    """Project pelvis-heading-frame points into the onboard image, or None if any is behind it."""
    optical_to_heading = np.asarray(kinematic_feedback["camera_optical_to_heading_rotation"])
    optical_origin = np.asarray(kinematic_feedback["camera_optical_origin_m"])
    optical_points = (np.asarray(heading_points) - optical_origin) @ optical_to_heading
    if np.any(optical_points[:, 2] <= 0):
        return None
    projected = optical_points @ np.asarray(intrinsic_matrix).T
    return projected[:, :2] / projected[:, 2:3]


def target_overlay(
    request_directory: Path, *, x: float, y: float, width: float, height: float
) -> tuple[str, list[str]]:
    """Dashed path from each measured tracking point through the commanded keyframes."""
    kinematic_feedback = json.loads((request_directory / "kinematic-feedback.json").read_text())
    observation = json.loads((request_directory / "observation.json").read_text())
    command = json.loads((request_directory / "command.json").read_text())
    with Image.open(request_directory / "rgb.png") as image:
        image_width, image_height = image.size
    elements = []
    drawn_point_names = []
    for point_index, point_name in enumerate(TRACKING_POINT_ORDER):
        _, color = TRACKING_POINT_STYLES[point_name]
        heading_points = [kinematic_feedback["tracking_points_m"][point_name]] + [
            pose["positions_heading_frame"][3 * point_index : 3 * point_index + 3]
            for pose in command["poses"]
        ]
        pixels = project_heading_points(
            heading_points, kinematic_feedback, observation["camera_intrinsic_matrix"]
        )
        if pixels is None or not np.any(
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image_width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image_height)
        ):
            # The torso-mounted head point is usually behind or above the head camera.
            continue
        pixels[:, 0] = x + pixels[:, 0] * width / image_width
        pixels[:, 1] = y + pixels[:, 1] * height / image_height
        drawn_point_names.append(point_name)
        path = "M" + " L".join(f"{px:.2f} {py:.2f}" for px, py in pixels)
        elements.append(
            f'<path d="{path}" fill="none" stroke="white" stroke-width="6" opacity="0.85"/><path d="{path}" fill="none" stroke="{color}" stroke-width="2.6" stroke-dasharray="6 4"/>'
        )
        start, endpoint = pixels[0], pixels[-1]
        elements.append(
            f'<circle cx="{start[0]:.2f}" cy="{start[1]:.2f}" r="5" fill="white" stroke="{color}" stroke-width="2.4"/>'
            f'<circle cx="{endpoint[0]:.2f}" cy="{endpoint[1]:.2f}" r="6" fill="{color}" stroke="white" stroke-width="1.6"/>'
        )
    return "\n".join(elements), drawn_point_names


def command_trace_rows(episode: Path, request_identifier: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in iter_records(episode / "physics" / "physical-trace.jsonl")
        if row["command_id"] == request_identifier and row["native_command"] is not None
    ]
    if not rows:
        raise ValueError(f"No native motor commands were recorded for {request_identifier}")
    return rows


def draw_motor_command_plot(
    rows: list[dict[str, Any]], chunk_seconds: float, assets: Path, *, plotted_seconds: float
) -> None:
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]}
    )
    first_wall = rows[0]["wall_seconds"]
    rows = [row for row in rows if row["wall_seconds"] - first_wall <= plotted_seconds]
    times = np.asarray([row["wall_seconds"] - first_wall for row in rows])
    positions = np.asarray([row["native_command"]["positions"] for row in rows])
    figure, axes = plt.subplots(4, 1, figsize=(2.5, 3.1), sharex=True)
    for axis, (joint_name, label, color) in zip(axes, PLOTTED_MOTOR_JOINTS, strict=True):
        axis.axvspan(0, chunk_seconds, color="#dceaf7", linewidth=0)
        axis.plot(times, positions[:, JOINT_NAMES.index(joint_name)], color=color, linewidth=2)
        axis.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=6)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#bdc4cc")
        axis.tick_params(labelsize=8, colors="#687481", length=2)
    axes[-1].set_xlabel("Time (s)", fontsize=9, color="#53606b")
    figure.subplots_adjust(left=0.35, right=0.96, top=0.96, bottom=0.16, hspace=0.43)
    figure.savefig(assets / "motor-commands.png", dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def verified_history_pairs(request_directory: Path) -> list[ImagePair]:
    visual_history = json.loads((request_directory / "visual-history.json").read_text())
    pairs = [ImagePair.model_validate(pair) for pair in visual_history["pairs"]]
    if not 1 <= len(pairs) <= 4:
        raise ValueError("The overview requires one to four recorded synchronized pairs")
    for pair in pairs:
        for relative_path, expected_hash in (
            (pair.onboard_path, pair.onboard_sha256),
            (pair.external_path, pair.external_sha256),
        ):
            original = (request_directory / relative_path).resolve()
            if not original.is_relative_to(request_directory.resolve()):
                raise ValueError("Observation image escapes its saved request directory")
            if sha256_file(original) != expected_hash:
                raise ValueError("Observation image differs from its original request manifest")
    return pairs


def subtask_label(identifier: str) -> str:
    """``inspect_B_A`` -> ``Inspect B, A``; single letters name cabinets and stay uppercase."""
    words = identifier.split("_")
    head = [words[0].capitalize()] + [word.lower() for word in words[1:] if len(word) > 1]
    cabinets = [word.upper() for word in words[1:] if len(word) == 1]
    return " ".join(head) + (" " + ", ".join(cabinets) if cabinets else "")


def main(
    *,
    episode: Path,
    request: str,
    execution_request: str,
    output: Path,
    execution_external_crop: tuple[int, int, int, int] = (250, 90, 730, 450),
    execution_view: Literal["external-with-onboard-inset", "onboard"] = (
        "external-with-onboard-inset"
    ),
) -> None:
    """Build the overview from REQUEST's inputs and EXECUTION_REQUEST's images.

    EXECUTION_REQUEST should be the request that follows REQUEST, so its
    images show the result of executing REQUEST's command. The execution panel
    shows its external image cropped to EXECUTION_EXTERNAL_CROP (left, top,
    right, bottom pixels; 4:3) with its onboard image inset. EXECUTION_VIEW
    "onboard" shows only the onboard image, for scenes whose fixed external
    camera faces away from the interaction.
    """
    operator = episode / "operator"
    request_directory = operator / request
    execution_directory = operator / execution_request
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    shutil.rmtree(assets, ignore_errors=True)
    assets.mkdir()

    decision = json.loads((request_directory / "model.response.json").read_text())["output_parsed"]
    command = json.loads((request_directory / "command.json").read_text())
    if command["identifier"] != request:
        raise ValueError("The request command belongs to another decision")
    trace_rows = command_trace_rows(episode, request)
    draw_motor_command_plot(
        trace_rows,
        decision["duration_seconds"],
        assets,
        plotted_seconds=decision["duration_seconds"] + MOTOR_PLOT_HOLD_SECONDS,
    )

    pairs = verified_history_pairs(request_directory)
    row_pitch = 84
    image_height = 70.0
    history_images = []
    for pair_index, pair in enumerate(pairs):
        row_top = 202 + pair_index * row_pitch
        left = 32.0
        for view_index, relative_path in enumerate((pair.onboard_path, pair.external_path)):
            original = request_directory / relative_path
            asset = assets / f"history-{pair_index}-{view_index}.png"
            shutil.copyfile(original, asset)
            image_width = image_height * image_aspect_ratio(asset)
            history_images.append(embedded_image(asset, left, row_top, image_width, image_height))
            left += image_width + 6
        roles = " / ".join(HISTORY_ROLE_LABELS.get(frame.role, frame.role) for frame in pair.frames)
        history_images.append(
            f'<text x="{left + 2:.1f}" y="{row_top + image_height / 2 + 6:.1f}" class="history-label">{html.escape(roles)}</text>'
        )

    shutil.copyfile(request_directory / "rgb.png", assets / "targets.png")
    shutil.copyfile(execution_directory / "rgb.png", assets / "execution.png")
    with Image.open(execution_directory / "external.png") as external:
        external.crop(execution_external_crop).save(assets / "execution-external.png")
    targets_box = (705.0, 196.0, 320.0, 320.0 / image_aspect_ratio(assets / "targets.png"))
    onboard_only = execution_view == "onboard"
    execution_image = assets / ("execution.png" if onboard_only else "execution-external.png")
    execution_box = (1385.0, 196.0, 380.0, 380.0 / image_aspect_ratio(execution_image))
    inset_width = 132.0
    inset_height = inset_width / image_aspect_ratio(assets / "execution.png")
    inset_box = (
        execution_box[0] + 8,
        execution_box[1] + execution_box[3] - inset_height - 8,
        inset_width,
        inset_height,
    )
    images = {
        "targets": embedded_image(assets / "targets.png", *targets_box),
        "execution": embedded_image(execution_image, *execution_box),
        "execution_inset": ""
        if onboard_only
        else embedded_image(assets / "execution.png", *inset_box),
        "motor": embedded_image(assets / "motor-commands.png", 1098, 219, 204, 253),
    }
    overlays, drawn_point_names = target_overlay(
        request_directory,
        x=targets_box[0],
        y=targets_box[1],
        width=targets_box[2],
        height=targets_box[3],
    )
    legend = []
    legend_x = (
        865.0
        - sum(30 + 9.5 * len(TRACKING_POINT_STYLES[name][0]) for name in drawn_point_names) / 2
    )
    for point_name in drawn_point_names:
        label, color = TRACKING_POINT_STYLES[point_name]
        legend.append(
            f'<circle cx="{legend_x}" cy="482" r="5.5" fill="{color}"/><text x="{legend_x + 11}" y="488" font-size="19" fill="#444">{label}</text>'
        )
        legend_x += 30 + 9.5 * len(label)
    motion = (
        f"Walk {command['speed_m_per_s']:.1f} m/s · {decision['duration_seconds']:.1f} s chunk"
        if command["locomotion_mode"] != 0
        else f"Stand · {decision['duration_seconds']:.1f} s chunk"
    )

    inset_markup = (
        ""
        if onboard_only
        else f'''<rect x="{inset_box[0] - 2.5}" y="{inset_box[1] - 2.5}" width="{inset_box[2] + 5}" height="{inset_box[3] + 5:.3f}" rx="3" fill="white"/>
 {images["execution_inset"]}
 <rect x="{inset_box[0] + 3}" y="{inset_box[1] + 3}" width="68" height="20" rx="3" fill="white" opacity="0.9"/>
 <text x="{inset_box[0] + 37}" y="{inset_box[1] + 18}" text-anchor="middle" font-size="15" fill="#444">Onboard</text>'''
    )
    plan_rows = []
    for index, subtask in enumerate(decision["subtasks"]):
        label = SUBTASK_SHORT_LABELS.get(
            subtask["identifier"], subtask_label(subtask["identifier"])
        )
        active = subtask["identifier"] == decision["active_subtask_id"]
        vertical = 278 + index * 36
        if active:
            plan_rows.append(
                f'<rect x="425" y="{vertical - 24}" width="195" height="33" rx="4" fill="#dceaf7"/>'
            )
        color = "#205986" if active else "#333333"
        plan_rows.append(
            f'<text x="438" y="{vertical}" font-size="20" fill="{color}">{index + 1}</text><text x="460" y="{vertical}" font-size="18.5" fill="{color}">{html.escape(label)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="740" viewBox="0 0 1800 740" role="img" aria-labelledby="title description">
<title id="title">AHOPPA closed-loop humanoid operation</title>
<desc id="description">One task instruction starts the LLM, which updates its subtask plan and sends head and wrist targets with a walking command through SONIC to the G1. All images are recorded model inputs from one cabinet inspection episode; the targets and motor commands are that request's recorded command.</desc>
<defs>
 <marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="8" markerHeight="8" orient="auto"><path d="M2 2 L10 6 L2 10 Z" fill="#333333"/></marker>
 <clipPath id="target-clip"><rect x="{targets_box[0]}" y="{targets_box[1]}" width="{targets_box[2]}" height="{targets_box[3]:.3f}" rx="3"/></clipPath>
 <clipPath id="execution-clip"><rect x="{execution_box[0]}" y="{execution_box[1]}" width="{execution_box[2]}" height="{execution_box[3]:.3f}" rx="3"/></clipPath>
 <style>text {{ font-family:"Times New Roman",Times,"Liberation Serif",serif; fill:#171717; }} .label {{ font-size:29px; font-weight:700; }} .small {{ font-size:21px; fill:#444; }} .history-label {{ font-size:17px; fill:#444; }} .panel {{ fill:white; stroke:#202020; stroke-width:2; }} .flow {{ fill:none; stroke:#333; stroke-width:2.4; marker-end:url(#arrow); }}</style>
</defs>
<rect width="1800" height="740" fill="white"/>
<g id="task-instruction">
 <rect class="panel" x="327" y="16" width="400" height="80" rx="9"/>
 <text x="527" y="43" text-anchor="middle" class="small">Task instruction</text>
 <text x="527" y="76" text-anchor="middle" font-size="24">{html.escape(TASK_INSTRUCTION_SHORT)}</text>
 <path class="flow" d="M527 96 V121"/>
</g>
<g id="paired-observation-history">
 <rect class="panel" x="20" y="125" width="340" height="485" rx="10"/>
 <text x="190" y="159" text-anchor="middle" class="label">Observation history</text>
 <text x="79" y="180" text-anchor="middle" font-size="17">Onboard</text>
 <text x="79" y="196" text-anchor="middle" font-size="15" fill="#444">RGB-D</text>
 <text x="194" y="180" text-anchor="middle" font-size="17">External</text>
 <text x="194" y="196" text-anchor="middle" font-size="15" fill="#444">RGB</text>
 {"".join(history_images)}
 <path d="M38 540 H342" stroke="#b8c5ce" stroke-width="1"/>
 <text x="190" y="566" text-anchor="middle" font-size="22" font-weight="700">Measured robot feedback</text>
 <text x="190" y="593" text-anchor="middle" class="small">Head · wrists · hands · body tilt</text>
</g>
<g id="planning-and-control">
 <rect class="panel" x="390" y="125" width="950" height="485" rx="10"/>
 <g id="llm-and-evolving-plan">
  <text x="522.5" y="165" text-anchor="middle" class="label">Multimodal LLM</text>
  <rect x="410" y="195" width="225" height="307" rx="6" fill="#edf3fa" stroke="#a0b4c6" stroke-width="1.2"/>
  <text x="522.5" y="229" text-anchor="middle" font-size="21" font-weight="700">Plan + update subtasks</text>
  {"".join(plan_rows)}
  <text x="522.5" y="539" text-anchor="middle" font-size="25" font-weight="700">Next motion chunk</text>
 </g>
 <g id="head-and-wrist-targets">
  <text x="865" y="165" text-anchor="middle" class="label">Head + wrist targets</text>
  <g clip-path="url(#target-clip)">{images["targets"]}{overlays}</g>
  {"".join(legend)}
  <text x="865" y="527" text-anchor="middle" class="small">Pelvis-frame positions + orientations</text>
  <text x="865" y="555" text-anchor="middle" class="small">{html.escape(motion)}</text>
 </g>
 <g id="sonic-motor-control">
  <text x="1200" y="165" text-anchor="middle" class="label">SONIC</text>
  <rect x="1090" y="195" width="220" height="307" rx="6" fill="#edf3fa" stroke="#a0b4c6" stroke-width="1.2"/>
  {images["motor"]}
  <text x="1200" y="539" text-anchor="middle" font-size="25" font-weight="700">Target tracking</text>
  <text x="1200" y="568" text-anchor="middle" class="small">Motor commands</text>
 </g>
</g>
<g id="g1-execution">
 <rect class="panel" x="1370" y="125" width="410" height="485" rx="10"/>
 <text x="1575" y="165" text-anchor="middle" class="label">Robot execution</text>
 <g clip-path="url(#execution-clip)">{images["execution"]}</g>
 {inset_markup}
 <rect x="{execution_box[0] + 6}" y="{execution_box[1] + 6}" width="72" height="22" rx="3" fill="white" opacity="0.9"/>
 <text x="{execution_box[0] + 42}" y="{execution_box[1] + 22}" text-anchor="middle" font-size="16" fill="#444">{"Onboard" if onboard_only else "External"}</text>
 <text x="1575" y="530" text-anchor="middle" class="small">Drawer pushed closed by hand</text>
 <text x="1575" y="566" text-anchor="middle" font-size="27" font-weight="700">G1 in Newton</text>
 <text x="1575" y="595" text-anchor="middle" class="small">Physics continues during inference</text>
</g>
<g id="forward-control-flow">
 <path class="flow" d="M365 350 H403"/><path class="flow" d="M643 350 H697"/>
 <path class="flow" d="M1033 350 H1082"/><path class="flow" d="M1318 350 H1365"/>
</g>
<g id="motor-feedback">
 <path class="flow" d="M1480 610 V640 H1200 V612"/>
 <text x="1340" y="668" text-anchor="middle" class="small">Motor feedback</text>
</g>
<g id="next-observation">
 <path class="flow" d="M1670 610 V710 H190 V612" stroke-dasharray="7 5"/>
 <rect x="683" y="693" width="440" height="34" fill="white"/>
 <text x="903" y="719" text-anchor="middle" font-size="26">New images + action feedback</text>
</g>
</svg>"""
    source = output / "system-overview.svg"
    source.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    subprocess.run(
        [
            "rsvg-convert",
            "-w",
            "3000",
            "-o",
            str(output / "system-overview.png"),
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsvg-convert",
            "-f",
            "pdf",
            "-w",
            "180mm",
            "-o",
            str(output / "system-overview.pdf"),
            str(source),
        ],
        check=True,
    )

    trace_path = episode / "physics" / "physical-trace.jsonl"
    recording = recording_location(trace_path)
    request_files = [
        request_directory / name
        for name in (
            "prompt.json",
            "model.response.json",
            "command.json",
            "observation.json",
            "kinematic-feedback.json",
            "visual-history.json",
            "rgb.png",
        )
    ] + [
        execution_directory / "rgb.png",
        execution_directory / "external.png",
        episode / "inputs" / "scene.json",
    ]
    execution_scope = (
        "The execution panel is the next request's onboard image."
        if onboard_only
        else "The execution panel is the next request's external image, cropped to the robot and cabinets, with its onboard image inset; both were captured together."
    )
    atomic_json(
        output / "overview-provenance.json",
        {
            "episode": str(episode),
            "request": request,
            "execution_request": execution_request,
            "execution_view": execution_view,
            "execution_external_crop_left_top_right_bottom": list(execution_external_crop),
            "scope": f"All scene images are unmodified (resized or cropped) model inputs from one episode. The targets panel is the request's onboard image with that request's recorded head/wrist command projected through its recorded intrinsics and onboard camera kinematics; targets are drawn in the request-time pelvis frame, so the walking displacement during the chunk is not shown. Points that do not project into the onboard image are omitted. {execution_scope} The motor plot shows the recorded SONIC native position commands while this command was active; the shaded interval is the commanded chunk duration, followed by the first {MOTOR_PLOT_HOLD_SECONDS:.1f} s of the hold until the next command.",
            "labels": "Task and subtask labels are shortened from the recorded prompt and model plan. Subtask status is model-reported, not an independent physical score.",
            "source_sha256": {
                str(path.relative_to(episode)): sha256_file(path) for path in request_files
            },
            "history_pairs": [pair.model_dump(mode="json") for pair in pairs],
            "motor_command_trace": {
                "recording": str(recording[0] if recording else trace_path),
                "recording_sha256": recording_sha256(trace_path),
                "command_id": request,
                "rows": len(trace_rows),
                "first_wall_seconds": trace_rows[0]["wall_seconds"],
                "last_wall_seconds": trace_rows[-1]["wall_seconds"],
                "plotted_joints": [joint for joint, _, _ in PLOTTED_MOTOR_JOINTS],
                "cabinet_joint_positions_first": trace_rows[0]["cabinet_joint_positions"],
                "cabinet_joint_positions_last": trace_rows[-1]["cabinet_joint_positions"],
            },
            "asset_sha256": {path.name: sha256_file(path) for path in sorted(assets.glob("*.png"))},
            "builder_source_sha256": sha256_file(Path(__file__)),
        },
    )


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
