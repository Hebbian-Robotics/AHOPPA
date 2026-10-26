"""Render private native-controller states and export allowlisted camera feedback.

This module owns a separate Isaac application. Its pose input is private render
state, not an agent observation. Rendering never advances task physics.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter, model_validator
from scipy.spatial.transform import Rotation

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.command_admission import ObservationIdentity
from humanoid_agent.dex3 import HAND_JOINT_NAMES, HandJointObservation
from humanoid_agent.finite_command import Identifier
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.observation_history import ActionFrameContext
from humanoid_agent.process_lifecycle import process_record
from humanoid_agent.recording import EpisodeRecorder
from humanoid_agent.robot_camera import calibration, depth_tiles, wrist_calibration
from humanoid_agent.robot_observations import RobotSensors
from humanoid_agent.runtime.cpu import CpuAllocation
from humanoid_agent.runtime.environment import create_newton_workcell_scene
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.runtime.state import TensorReadbackBuffers, copy_tensor_fields_to_cpu
from humanoid_agent.shutdown_task import CabinetIdentifier
from humanoid_agent.validated_types import ValidatedModel

CAMERA_STARTUP_BUDGET_SECONDS = 900

JointVector = Annotated[list[FiniteFloat], Field(min_length=29, max_length=43)]
RootState = Annotated[list[FiniteFloat], Field(min_length=13, max_length=13)]


@dataclass(frozen=True)
class CameraBatchSnapshot:
    """Owned CPU arrays; saving evidence must not access live device tensors."""

    images: dict[str, NDArray[Any]]
    geometry: dict[str, NDArray[Any]]


class NativeVisualState(BaseModel):
    """Private simulator state used only to reconstruct images and audit scoring."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    identifier: str = Field(pattern=r"^[a-z0-9_-]+$")
    episode_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    episode_deadline_elapsed_wall_seconds: FiniteFloat = Field(gt=0)
    simulation_seconds: FiniteFloat = Field(ge=0)
    elapsed_wall_seconds: FiniteFloat = Field(ge=0)
    source_monotonic_seconds: FiniteFloat = Field(ge=0)
    joint_names: Annotated[list[str], Field(min_length=29, max_length=43)]
    joint_positions: JointVector
    joint_velocities: JointVector
    root_state_world: RootState
    # Articulated cabinet joints by cabinet, so replays show the recorded drawer motion.
    cabinet_joint_positions: dict[CabinetIdentifier, dict[str, FiniteFloat]] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    indicator_activated: bool = False
    action_frame: ActionFrameContext | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def require_valid_physical_pose(self) -> NativeVisualState:
        expected_names = set(JOINT_NAMES)
        if len(self.joint_names) == 43:
            expected_names.update(HAND_JOINT_NAMES["left"])
            expected_names.update(HAND_JOINT_NAMES["right"])
        if set(self.joint_names) != expected_names or len(set(self.joint_names)) != len(
            self.joint_names
        ):
            raise ValueError("Render state requires the canonical 29 body or 43 body/finger joints")
        if not len(self.joint_names) == len(self.joint_positions) == len(self.joint_velocities):
            raise ValueError("Render joint names, positions and velocities must match")
        if not np.isclose(np.linalg.norm(self.root_state_world[3:7]), 1, atol=1e-3, rtol=0):
            raise ValueError("Private render state requires a unit wxyz root quaternion")
        return self

    def robot_sensors(self) -> RobotSensors:
        root_rotation = Rotation.from_quat(self.root_state_world[3:7], scalar_first=True)
        indices = [self.joint_names.index(name) for name in JOINT_NAMES]
        return RobotSensors(
            joint_names=JOINT_NAMES,
            joint_positions_rad=[self.joint_positions[index] for index in indices],
            joint_velocities_rad_per_s=[self.joint_velocities[index] for index in indices],
            imu_angular_velocity_body_rad_per_s=root_rotation.inv()
            .apply(self.root_state_world[10:13])
            .tolist(),
            imu_gravity_direction_body=root_rotation.inv().apply([0, 0, -1]).tolist(),
        )


class NativeVisualBatch(BaseModel):
    """Private ordered snapshots; each environment retains its own episode identity."""

    model_config = ConfigDict(extra="forbid")
    identifier: Identifier
    states: list[NativeVisualState] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_independent_episodes(self) -> NativeVisualBatch:
        if len({state.episode_id for state in self.states}) != len(self.states):
            raise ValueError("Shared capture requires independent episode identities")
        return self

    def require_matching_receipt(self, receipt: CameraCaptureCompleted) -> None:
        """A tiled capture cannot acknowledge another batch or swap robot identities."""
        expected = [
            (index, state.episode_id, state.identifier) for index, state in enumerate(self.states)
        ]
        received = [
            (capture.environment_index, capture.episode_id, capture.identifier)
            for capture in receipt.captures
        ]
        if receipt.identifier != self.identifier or received != expected:
            raise ValueError("Camera receipt swaps, omits or reassigns a robot observation")


class CameraCaptureRequest(ValidatedModel):
    kind: Literal["capture"] = "capture"
    state: NativeVisualState | NativeVisualBatch
    reply_to: Path
    observation_replies: list[Path | None] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def require_matching_reply_routes(self) -> CameraCaptureRequest:
        if self.observation_replies and (
            not isinstance(self.state, NativeVisualBatch)
            or len(self.observation_replies) != len(self.state.states)
        ):
            raise ValueError("Observation reply routes must match the complete camera batch")
        return self


class CameraReady(ValidatedModel):
    kind: Literal["ready"] = "ready"


class CameraCaptureIdentity(ValidatedModel):
    environment_index: int = Field(ge=0, lt=16)
    episode_id: Identifier
    identifier: Identifier


class CameraCaptureCompleted(ValidatedModel):
    kind: Literal["captured"] = "captured"
    identifier: Identifier
    captures: list[CameraCaptureIdentity] = Field(min_length=1, max_length=16)


CAMERA_EVENT_ADAPTER = TypeAdapter(
    Annotated[CameraReady | CameraCaptureCompleted, Field(discriminator="kind")]
)


class CameraStopRequest(ValidatedModel):
    kind: Literal["stop"] = "stop"


CAMERA_REQUEST_ADAPTER = TypeAdapter(
    Annotated[CameraCaptureRequest | CameraStopRequest, Field(discriminator="kind")]
)


class NativeCameraRenderer:
    """Pose replay only: the owner must create the Isaac application first."""

    def __init__(
        self,
        isaac_source: Path,
        scene_path: Path,
        decorative_assets: Path | None,
        output: Path,
        *,
        simulation_configuration: Any,
        num_environments: int = 1,
        render_observer: bool = True,
        observer_resolution: tuple[int, int] = (960, 540),
        retain_capture_media: bool = True,
    ) -> None:
        self.output = output
        self.retain_capture_media = retain_capture_media
        self.num_environments = num_environments
        self.readback_buffers = TensorReadbackBuffers()
        output.mkdir(parents=True, exist_ok=False)
        self.torch: Any = importlib.import_module("torch")
        simulation: Any = importlib.import_module("isaaclab.sim")
        sensors: Any = importlib.import_module("isaaclab.sensors")
        renderer_types: Any = importlib.import_module("isaaclab_ov.renderers")
        self.simulation = simulation.SimulationContext(simulation_configuration)
        self.scene_definition = GroundWorkcellScene.model_validate_json(scene_path.read_text())
        if decorative_assets is not None:
            manifest = json.loads((decorative_assets / "manifest.json").read_text())
            if manifest["scene_sha256"] != hashlib.sha256(scene_path.read_bytes()).hexdigest():
                raise ValueError("Industrial visuals belong to another physical scene")
        renderer_configuration = renderer_types.OVRTXRendererCfg(
            log_file_path=str(output / "ovrtx.log")
        )
        camera_calibration = calibration()
        camera_configuration = sensors.CameraCfg(
            prim_path="{ENV_REGEX_NS}/AgentCamera",
            renderer_cfg=renderer_configuration,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            update_period=0,
            update_latest_camera_pose=True,
            spawn=simulation.PinholeCameraCfg(
                focal_length=24,
                horizontal_aperture=24 * 640 / camera_calibration["intrinsics_fx_fy_cx_cy"][0],
                clipping_range=(0.05, 10),
            ),
        )
        camera_configurations = {"onboard_camera": camera_configuration}
        if self.scene_definition.wrist_cameras:
            for side in ("left", "right"):
                wrist_mount = wrist_calibration(side)
                camera_configurations[f"{side}_wrist_camera"] = sensors.CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{side.title()}WristCamera",
                    renderer_cfg=renderer_configuration,
                    height=480,
                    width=640,
                    data_types=["rgb"],
                    update_period=0,
                    update_latest_camera_pose=True,
                    spawn=simulation.PinholeCameraCfg(**wrist_mount["pinhole"]),
                )
        if render_observer:
            observer_configuration = sensors.CameraCfg(
                prim_path="{ENV_REGEX_NS}/ObserverCamera",
                renderer_cfg=renderer_configuration,
                height=observer_resolution[1],
                width=observer_resolution[0],
                data_types=["rgb"],
                update_period=0,
                update_latest_camera_pose=True,
                spawn=simulation.PinholeCameraCfg(
                    focal_length=18,
                    horizontal_aperture=24,
                    clipping_range=(0.05, 20),
                ),
            )
            camera_configurations["observer_camera"] = observer_configuration
        self.scene = create_newton_workcell_scene(
            isaac_source / "gear_sonic/data/assets",
            self.scene_definition,
            contact_sensors=False,
            display_collision_boxes=decorative_assets is None,
            num_environments=num_environments,
            writable_marker_material=True,
            decorative_assets=decorative_assets,
            camera_configurations=camera_configurations,
        )
        self.robot = self.scene["robot"]
        self.marker_material = self.scene["marker_material"]
        self.camera = self.scene["onboard_camera"]
        self.wrist_cameras = (
            {side: self.scene[f"{side}_wrist_camera"] for side in ("left", "right")}
            if self.scene_definition.wrist_cameras
            else {}
        )
        self.observer = self.scene["observer_camera"] if render_observer else None
        sunlight = simulation.DistantLightCfg(intensity=1800, color=(1, 0.97, 0.92))
        sunlight.func(
            "/World/Sunlight",
            sunlight,
            orientation=tuple(Rotation.from_euler("xyz", [-35, 20, -30], degrees=True).as_quat()),
        )
        skylight = simulation.DomeLightCfg(intensity=450, color=(0.78, 0.84, 0.9))
        skylight.func("/World/Skylight", skylight)
        self.simulation.reset()
        self.initial_simulation_seconds = float(
            self.simulation.get_physics_step_count() * self.simulation.get_physics_dt()
        )
        self.robot.reset()
        self.robot.update(0.005)
        # Fixed corner installation view covering home and both drawer fronts.
        observer_eye = [-1.5, -2.9, 2.5]
        observer_look_at = [2.2, 0.0, 0.55]
        if self.scene_definition.shutdown.protocol == "drawer-shutdown-v2":
            # High on the west wall between C and D, looking east at home only: the
            # lockers hide A and B, and C and D sit behind the camera.
            observer_eye = [-2.25, 0.0, 2.4]
            observer_look_at = [0.8, 0.0, 0.4]
        if self.observer is not None:
            self.observer.set_world_poses_from_view(
                self.torch.tensor([observer_eye], device="cuda:0") + self.scene.env_origins,
                self.torch.tensor([observer_look_at], device="cuda:0") + self.scene.env_origins,
            )
        for _ in range(5):
            self.simulation.render()
        self.rendered_states = 0
        self.scene_path = scene_path
        self.scene_definition = GroundWorkcellScene.model_validate_json(scene_path.read_text())
        self.recorder = EpisodeRecorder(output)
        atomic_json(
            output / "manifest.json",
            {
                "operation": "Pose replay in a separate Isaac camera worker; no task physics steps",
                "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                "camera_calibration": camera_calibration,
                "wrist_camera_calibrations": {
                    side: wrist_calibration(side) for side in self.wrist_cameras
                },
                "observer_camera": {"eye": observer_eye, "look_at": observer_look_at}
                if self.observer is not None
                else None,
                "model_feedback": "RGB, aligned metric depth, encoders and ideal IMU only",
                "startup_render_warmup_frames": 5,
                "model_requests": 0,
            },
        )

    def capture(self, state: NativeVisualState) -> dict[str, Any]:
        """Preserve the one-episode interface used by existing workers and replay."""
        if self.num_environments != 1:
            raise ValueError("A multi-environment renderer requires a complete capture batch")
        return self.capture_batch([state])[0]

    def capture_batch(self, states: list[NativeVisualState]) -> list[dict[str, Any]]:
        """Render all independent snapshots together through upstream tiled cameras."""
        if len(states) != self.num_environments:
            raise ValueError("Capture batch must contain one snapshot per environment")
        started_at = time.monotonic()
        colors = [
            (0.15, 0.8, 0.25)
            if state.indicator_activated
            else tuple(self.scene_definition.button_color)
            for state in states
        ]
        self.marker_material.write_channels(
            [self.marker_material],
            {"color": self.torch.tensor([colors], dtype=self.torch.float32, device="cuda:0")},
        )
        roots = []
        positions = []
        velocities = []
        for state in states:
            if set(state.joint_names) != set(self.robot.joint_names):
                raise ValueError("Camera worker and physics robot joint names differ")
            indices = [state.joint_names.index(name) for name in self.robot.joint_names]
            roots.append(np.asarray(state.root_state_world)[[0, 1, 2, 4, 5, 6, 3]])
            positions.append(np.asarray(state.joint_positions)[indices])
            velocities.append(np.asarray(state.joint_velocities)[indices])
        torch = self.torch
        self.robot.write_root_pose_to_sim_index(
            root_pose=torch.tensor(np.asarray(roots), dtype=torch.float32, device="cuda:0")
        )
        self.robot.write_joint_position_to_sim_index(
            position=torch.tensor(np.asarray(positions), dtype=torch.float32, device="cuda:0")
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.tensor(np.asarray(velocities), dtype=torch.float32, device="cuda:0")
        )
        for cabinet in self.scene_definition.shutdown.cabinets:
            articulation = self.scene["cabinet_" + cabinet.identifier]
            recorded = []
            for state in states:
                if state.cabinet_joint_positions is None:
                    raise ValueError("Shutdown replay requires recorded cabinet joints")
                joints = state.cabinet_joint_positions[cabinet.identifier]
                recorded.append([joints[name] for name in articulation.joint_names])
            articulation.write_joint_position_to_sim_index(
                position=torch.tensor(np.asarray(recorded), dtype=torch.float32, device="cuda:0")
            )
        self.simulation.forward()
        self.robot.update(0.005)
        camera_calibration = calibration()
        torso = self.robot.body_names.index(camera_calibration["camera_body"])
        torso_snapshot = copy_tensor_fields_to_cpu(
            {
                "positions": self.robot.data.body_link_pos_w.torch[:, torso],
                "quaternions": self.robot.data.body_link_quat_w.torch[:, torso],
            },
            buffers=self.readback_buffers,
        )
        torso_positions = torso_snapshot["positions"]
        torso_quaternions = torso_snapshot["quaternions"]
        camera_calibration = calibration()
        upstream_math: Any = importlib.import_module("isaaclab.utils.math")
        eye_tensor, optical_quaternions = upstream_math.combine_frame_transforms(
            torch.tensor(torso_positions, dtype=torch.float32),
            torch.tensor(torso_quaternions, dtype=torch.float32),
            torch.tensor(
                [camera_calibration["camera_position_torso_m"]] * self.num_environments,
                dtype=torch.float32,
            ),
            torch.tensor(
                np.tile(
                    Rotation.from_matrix(
                        np.asarray(camera_calibration["optical_to_torso_rotation"])
                    ).as_quat(),
                    (self.num_environments, 1),
                ),
                dtype=torch.float32,
            ),
        )
        for environment_index in range(self.num_environments):
            torso_rotation = Rotation.from_quat(torso_quaternions[environment_index])
            independent_eye = torso_positions[environment_index] + torso_rotation.apply(
                camera_calibration["camera_position_torso_m"]
            )
            independent_rotation = torso_rotation.as_matrix() @ np.asarray(
                camera_calibration["optical_to_torso_rotation"]
            )
            if not np.allclose(
                eye_tensor[environment_index].numpy(), independent_eye, atol=1e-6, rtol=0
            ) or not np.allclose(
                Rotation.from_quat(optical_quaternions[environment_index].numpy()).as_matrix(),
                independent_rotation,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    "Upstream optical transform disagrees with independent calibration"
                )
        self.camera.set_world_poses(eye_tensor, optical_quaternions, convention="ros")
        for side, wrist_camera in self.wrist_cameras.items():
            mount = wrist_calibration(side)
            body_index = self.robot.body_names.index(mount["camera_body"])
            body_positions = self.robot.data.body_link_pos_w.torch[:, body_index]
            body_quaternions = self.robot.data.body_link_quat_w.torch[:, body_index]
            wrist_eyes, wrist_rotations = upstream_math.combine_frame_transforms(
                body_positions,
                body_quaternions,
                torch.tensor(
                    [mount["position_body_m"]] * self.num_environments,
                    dtype=torch.float32,
                    device=body_positions.device,
                ),
                torch.tensor(
                    np.tile(
                        Rotation.from_matrix(mount["optical_to_body_rotation"]).as_quat(),
                        (self.num_environments, 1),
                    ),
                    dtype=torch.float32,
                    device=body_positions.device,
                ),
            )
            wrist_camera.set_world_poses(wrist_eyes, wrist_rotations, convention="ros")

        self.simulation.render()
        self.simulation.render()
        sensor_types: Any = importlib.import_module("isaaclab.sensors")
        active_cameras = [self.camera, *self.wrist_cameras.values()]
        if self.observer is not None:
            active_cameras.append(self.observer)
        sensor_types.SensorBase.update_batch(active_cameras, 0.5)
        batch_snapshot = self._read_capture_batch()
        return [
            self._write_capture(
                state,
                environment_index,
                started_at,
                eye_tensor[environment_index].numpy(),
                Rotation.from_quat(optical_quaternions[environment_index].numpy()),
                batch_snapshot,
            )
            for environment_index, state in enumerate(states)
        ]

    def _read_capture_batch(self) -> CameraBatchSnapshot:
        """Complete unprojection and batched readback before any image encoding."""
        upstream_math: Any = importlib.import_module("isaaclab.utils.math")
        depth = self.camera.data.output["distance_to_image_plane"].torch[:, :, :, 0]
        intrinsics = self.camera.data.intrinsic_matrices.torch
        # Retain the pinned upstream per-view unprojection and x-major layout.
        optical_points = self.torch.stack(
            [
                upstream_math.unproject_depth(depth[index], intrinsics[index], is_ortho=True)
                .reshape(640, 480, 3)
                .permute(1, 0, 2)
                for index in range(self.num_environments)
            ]
        )
        geometry_fields = {
            "depth": depth,
            "intrinsics": intrinsics,
            "optical_points": optical_points,
            "camera_positions": self.camera.data.pos_w.torch,
            "camera_quaternions_xyzw": self.camera.data.quat_w_ros.torch,
            "robot_positions": self.robot.data.root_link_pos_w.torch,
        }
        if self.observer is not None:
            geometry_fields.update(
                {
                    "observer_positions": self.observer.data.pos_w.torch,
                    "observer_quaternions_xyzw": self.observer.data.quat_w_ros.torch,
                    "observer_intrinsics": self.observer.data.intrinsic_matrices.torch,
                }
            )
        geometry = copy_tensor_fields_to_cpu(geometry_fields, buffers=self.readback_buffers)
        for side, camera in self.wrist_cameras.items():
            camera_geometry = copy_tensor_fields_to_cpu(
                {
                    f"{side}_wrist_camera_positions": camera.data.pos_w.torch,
                    f"{side}_wrist_camera_quaternions_xyzw": camera.data.quat_w_ros.torch,
                    f"{side}_wrist_intrinsics": camera.data.intrinsic_matrices.torch,
                },
                buffers=self.readback_buffers,
            )
            geometry.update(camera_geometry)
        image_fields = {"rgb": self.camera.data.output["rgb"].torch[:, :, :, :3]}
        image_fields.update(
            {
                f"{side}_wrist_rgb": camera.data.output["rgb"].torch[:, :, :, :3]
                for side, camera in self.wrist_cameras.items()
            }
        )
        if self.observer is not None:
            image_fields["observer_rgb"] = self.observer.data.output["rgb"].torch[:, :, :, :3]
        return CameraBatchSnapshot(
            images=copy_tensor_fields_to_cpu(image_fields, buffers=self.readback_buffers),
            geometry=geometry,
        )

    def _write_capture(
        self,
        state: NativeVisualState,
        environment_index: int,
        started_at: float,
        eye: Any,
        optical_rotation: Rotation,
        batch_snapshot: CameraBatchSnapshot,
    ) -> dict[str, Any]:
        camera_calibration = calibration()
        rgb = batch_snapshot.images["rgb"][environment_index]
        depth = batch_snapshot.geometry["depth"][environment_index]
        intrinsics = batch_snapshot.geometry["intrinsics"][environment_index]
        # An upward-facing camera can legitimately see only background after
        # a fall. Preserve unavailable depth rather than masking the physical
        # outcome with a rendering infrastructure error.
        if np.any(np.isfinite(depth) & (depth < 0)):
            raise ValueError("Rendered depth contains malformed measurements")
        expected_intrinsics = camera_calibration["intrinsics_fx_fy_cx_cy"]
        if not np.allclose(
            intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]], expected_intrinsics, atol=1e-3, rtol=0
        ):
            raise ValueError("Rendered intrinsics differ from the declared RGB-D calibration")
        snapshot_root = (
            self.output if self.num_environments == 1 else self.output / f"env_{environment_index}"
        )
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_root / state.identifier
        snapshot.mkdir(exist_ok=False)
        image: Any = importlib.import_module("PIL.Image")
        image.fromarray(rgb).save(snapshot / "onboard.png")
        wrist_audits = {}
        for side in self.wrist_cameras:
            wrist_rgb = batch_snapshot.images[f"{side}_wrist_rgb"][environment_index]
            image.fromarray(wrist_rgb).save(snapshot / f"{side}_wrist.png")
            wrist_intrinsics = batch_snapshot.geometry[f"{side}_wrist_intrinsics"][
                environment_index
            ]
            declared = wrist_calibration(side)["intrinsics_fx_fy_cx_cy"]
            if not np.allclose(
                wrist_intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]], declared, atol=1e-3, rtol=0
            ):
                raise ValueError("Wrist camera intrinsics differ from declared calibration")
            wrist_audits[side] = {
                "position_world": batch_snapshot.geometry[f"{side}_wrist_camera_positions"][
                    environment_index
                ].tolist(),
                "optical_quaternion_xyzw": batch_snapshot.geometry[
                    f"{side}_wrist_camera_quaternions_xyzw"
                ][environment_index].tolist(),
                "rgb_sha256": hashlib.sha256(
                    (snapshot / f"{side}_wrist.png").read_bytes()
                ).hexdigest(),
            }

        if self.observer is not None:
            external_rgb = batch_snapshot.images["observer_rgb"][environment_index]
            image.fromarray(external_rgb).save(snapshot / "observer.png")
        np.save(snapshot / "depth-m.npy", depth)
        optical_points = batch_snapshot.geometry["optical_points"][environment_index]
        projection_errors = []
        inverse_intrinsics = np.linalg.inv(intrinsics)
        for pixel_y, pixel_x in ((120, 160), (120, 480), (240, 320), (360, 160), (360, 480)):
            measured_depth = depth[pixel_y, pixel_x]
            if np.isfinite(measured_depth):
                expected_point = (
                    inverse_intrinsics @ np.asarray([pixel_x, pixel_y, 1.0]) * measured_depth
                )
                projection_errors.append(
                    float(np.linalg.norm(optical_points[pixel_y, pixel_x] - expected_point))
                )
        if projection_errors and max(projection_errors) > 1e-5:
            raise ValueError("Upstream optical projection failed independent pixel/depth audit")
        np.save(snapshot / "optical-points-m.npy", optical_points)
        observation = IsaacObservation(
            action_frame=state.action_frame,
            simulation_seconds=state.simulation_seconds,
            elapsed_wall_seconds=state.elapsed_wall_seconds,
            robot_sensors=state.robot_sensors(),
            hand_joint_sensors={
                side: HandJointObservation(
                    positions=[
                        state.joint_positions[state.joint_names.index(name)] for name in names
                    ],
                    velocities=[
                        state.joint_velocities[state.joint_names.index(name)] for name in names
                    ],
                )
                for side, names in HAND_JOINT_NAMES.items()
            }
            if len(state.joint_names) == 43
            else None,
            external_camera_available=self.observer is not None,
            wrist_camera_names=["left_wrist", "right_wrist"] if self.wrist_cameras else [],
            depth_tiles_m=depth_tiles(depth),
            camera_intrinsic_matrix=intrinsics.tolist(),
            observation_identity=ObservationIdentity(
                episode_id=state.episode_id,
                identifier=state.identifier,
                captured_elapsed_wall_seconds=state.elapsed_wall_seconds,
                episode_deadline_elapsed_wall_seconds=state.episode_deadline_elapsed_wall_seconds,
            ),
        )
        atomic_json(snapshot / "observation.json", observation.model_dump())
        captured_at = time.monotonic()
        worker_simulation_seconds = float(
            self.simulation.get_physics_step_count() * self.simulation.get_physics_dt()
        )
        if worker_simulation_seconds != self.initial_simulation_seconds:
            raise RuntimeError("Camera rendering unexpectedly advanced its private physics clock")
        record = {
            "identifier": state.identifier,
            "episode_id": state.episode_id,
            "environment_index": environment_index,
            "simulation_seconds": state.simulation_seconds,
            "source_elapsed_wall_seconds": state.elapsed_wall_seconds,
            "source_monotonic_seconds": state.source_monotonic_seconds,
            "captured_monotonic_seconds": captured_at,
            "capture_wall_seconds": captured_at - started_at,
            "source_age_at_capture_seconds": captured_at - state.source_monotonic_seconds,
            "private_render_state_sha256": hashlib.sha256(
                state.model_dump_json().encode()
            ).hexdigest(),
            "onboard_rgb_sha256": hashlib.sha256(
                (snapshot / "onboard.png").read_bytes()
            ).hexdigest(),
            "observer_rgb_sha256": hashlib.sha256(
                (snapshot / "observer.png").read_bytes()
            ).hexdigest()
            if self.observer is not None
            else None,
            "depth_sha256": hashlib.sha256((snapshot / "depth-m.npy").read_bytes()).hexdigest(),
            "snapshot_directory": str(snapshot),
            "optical_points_sha256": hashlib.sha256(
                (snapshot / "optical-points-m.npy").read_bytes()
            ).hexdigest(),
            "projection_audit_maximum_error_m": max(projection_errors)
            if projection_errors
            else None,
            "projection_audit_valid_pixels": len(projection_errors),
            "private_wrist_camera_audits": wrist_audits,
            "private_camera_audit": {
                "expected_optical_position_world": eye.tolist(),
                "measured_optical_position_world": batch_snapshot.geometry["camera_positions"][
                    environment_index
                ].tolist(),
                "expected_optical_quaternion_ros_wxyz": optical_rotation.as_quat(
                    scalar_first=True
                ).tolist(),
                "measured_optical_quaternion_ros_wxyz": batch_snapshot.geometry[
                    "camera_quaternions_xyzw"
                ][environment_index][[3, 0, 1, 2]].tolist(),
                "robot_root_position_world": batch_snapshot.geometry["robot_positions"][
                    environment_index
                ].tolist(),
                "robot_root_quaternion_wxyz": state.root_state_world[3:7],
                "camera_mount_body": camera_calibration["camera_body"],
                "task_physics_steps_in_worker": 0,
                "worker_simulation_seconds": worker_simulation_seconds,
                "initial_worker_simulation_seconds": self.initial_simulation_seconds,
            },
        }
        if self.observer is not None:
            record["private_observer_camera_audit"] = {
                "optical_position_world": batch_snapshot.geometry["observer_positions"][
                    environment_index
                ].tolist(),
                "optical_quaternion_ros_wxyz": batch_snapshot.geometry["observer_quaternions_xyzw"][
                    environment_index
                ][[3, 0, 1, 2]].tolist(),
                "intrinsic_matrix": batch_snapshot.geometry["observer_intrinsics"][
                    environment_index
                ].tolist(),
                "resolution_width_height": list(
                    batch_snapshot.images["observer_rgb"][environment_index].shape[1::-1]
                ),
            }
        self.recorder.write("capture-records.jsonl", record)
        if self.num_environments > 1:
            self.recorder.write(f"env_{environment_index}/capture-records.jsonl", record)
        if self.retain_capture_media:
            for artifact in sorted(snapshot.iterdir()):
                self.recorder.attach(
                    artifact.relative_to(self.output).as_posix(),
                    artifact,
                    "image/png"
                    if artifact.suffix == ".png"
                    else "application/json"
                    if artifact.suffix == ".json"
                    else "application/octet-stream",
                    image_topic=f"/images/env_{environment_index}/{artifact.stem}",
                    source_timestamp_ns=round(state.source_monotonic_seconds * 1_000_000_000),
                )
        self.rendered_states += 1
        return record

    def close(self) -> None:
        self.recorder.close()


class NativeCameraProcess:
    """Owned camera application; the physics owner publishes private pose snapshots."""

    def __init__(
        self,
        python_executable: Path,
        isaac_source: Path,
        scene_path: Path,
        decorative_assets: Path | None,
        directory: Path,
        *,
        watchdog_seconds: float = 900,
        num_environments: int = 1,
        batch_captures: bool = False,
        external_camera: bool = False,
    ) -> None:
        self.startup_started_monotonic_seconds = time.monotonic()
        self.startup_deadline_monotonic_seconds = (
            self.startup_started_monotonic_seconds + CAMERA_STARTUP_BUDGET_SECONDS
        )
        self.directory = directory
        self.state_path = directory / "camera-state.json"
        self.sender = LocalPushSender(self.state_path)
        self.ready_path = directory / "camera-ready.json"
        self.events = LocalPullReceiver(self.ready_path)
        self.initial_batch: NativeVisualBatch | None = None
        self.output = directory / "camera-output"
        self.log = (directory / "camera-worker.log").open("w")
        command = [
            "taskset",
            "-c",
            CpuAllocation.from_environment().camera,
            str(python_executable),
            "-m",
            "humanoid_agent",
            "--role",
            "camera",
            "--isaac-source",
            str(isaac_source),
            "--scene",
            str(scene_path),
            "--state",
            str(self.state_path),
            "--output",
            str(self.output),
            "--ready",
            str(self.ready_path),
            "--watchdog-seconds",
            str(watchdog_seconds),
            "--num-environments",
            str(num_environments),
        ]
        if external_camera:
            command.append("--external-camera")
        if batch_captures:
            command.append("--batch-captures")
        if decorative_assets is not None:
            command.extend(["--decorative-assets", str(decorative_assets)])
        try:
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
            )
        except BaseException:
            self.sender.close()
            self.events.close()
            self.log.close()
            raise
        try:
            atomic_json(
                directory / "camera-worker-process.json",
                process_record(self.process.pid),
            )
            event = self._wait_for_startup_event("camera readiness")
            if not isinstance(event, CameraReady):
                raise ValueError("Camera sent capture completion before becoming ready")
        except BaseException:
            # Cancellation must release a child even before construction returns.
            self.close()
            raise

    def _wait_for_startup_event(self, description: str) -> CameraReady | CameraCaptureCompleted:
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("Camera worker failed startup; inspect camera-worker.log")
            if time.monotonic() >= self.startup_deadline_monotonic_seconds:
                raise TimeoutError(f"Camera startup exceeded its budget waiting for {description}")
            payload = self.events.receive(timeout_ms=100)
            if payload is not None:
                return CAMERA_EVENT_ADAPTER.validate_json(payload)

    def wait_for_initial_batch(self) -> None:
        # The first RGB-D frame can compile shaders after scene setup reports
        # ready. Both stages consume the same startup budget, before release.
        if self.initial_batch is None:
            raise ValueError("Initial camera batch must be published before waiting")
        event = self._wait_for_startup_event("initial capture")
        if not isinstance(event, CameraCaptureCompleted):
            raise ValueError("Expected initial camera capture completion")
        self.initial_batch.require_matching_receipt(event)
        atomic_json(
            self.directory / "camera-startup.json",
            {
                "elapsed_seconds": time.monotonic() - self.startup_started_monotonic_seconds,
                "budget_seconds": CAMERA_STARTUP_BUDGET_SECONDS,
                "initial_rgbd_ready": True,
                "episode_started": False,
            },
        )

    def publish(self, state: NativeVisualState | NativeVisualBatch) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("Camera worker exited during physics; inspect camera-worker.log")
        if isinstance(state, NativeVisualBatch) and state.identifier == "initial":
            self.initial_batch = state
        self.sender.send(
            CameraCaptureRequest(state=state, reply_to=self.ready_path).model_dump_json().encode()
        )

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.sender.send(CameraStopRequest().model_dump_json().encode())
            except TimeoutError:
                from humanoid_agent.runtime.episode import stop_process

                stop_process(self.process, self.directory)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                from humanoid_agent.runtime.episode import stop_process

                stop_process(self.process, self.directory)
        self.sender.close()
        self.events.close()
        self.log.close()
        if self.process.returncode != 0:
            raise RuntimeError(
                f"Camera worker exited {self.process.returncode}; inspect camera-worker.log"
            )
