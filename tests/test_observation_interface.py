"""Action-clock image history and robot feedback retain observable evidence only."""

import base64
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.observation_history import (
    ActionExecutionWindow,
    ActionFrameContext,
    HistoryRole,
    ImageFrame,
    collect_visual_history,
    image_pair_content,
    prepare_image_pairs,
)
from humanoid_agent.recording import read_record
from humanoid_agent.robot_feedback import concise_robot_feedback, previous_action_feedback
from humanoid_agent.runtime.actor import RobotEpisode
from humanoid_agent.runtime.camera import (
    CameraCaptureCompleted,
    CameraCaptureIdentity,
    NativeVisualState,
)
from humanoid_agent.runtime.episode_io import (
    BeginEpisode,
    CaptureSnapshot,
    EpisodeIO,
    ObservationReady,
)
from humanoid_agent.runtime.local_transport import LocalPushSender
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.sensor_feedback import DepthSnapshot
from humanoid_agent.sonic_planner_input import PlannerTrackingPose, SonicPlannerPublication


def observation_record(identifier: str = "current", captured: float = 3.0) -> IsaacObservation:
    return IsaacObservation.model_validate(
        {
            "simulation_seconds": captured,
            "elapsed_wall_seconds": captured,
            "robot_sensors": {
                "joint_names": ["example_joint"],
                "joint_positions_rad": [0],
                "joint_velocities_rad_per_s": [0],
                "imu_angular_velocity_body_rad_per_s": [0, 0, 0],
                "imu_gravity_direction_body": [0, 0, -1],
            },
            "depth_tiles_m": [[None] * 16 for _ in range(12)],
            "camera_intrinsic_matrix": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
            "external_camera_available": True,
            "observation_identity": {
                "episode_id": "episode",
                "identifier": identifier,
                "captured_elapsed_wall_seconds": captured,
                "episode_deadline_elapsed_wall_seconds": 30,
            },
        }
    )


def measured_kinematics() -> dict:
    return {
        "coordinate_frame": "heading-relative pelvis frame",
        "tracking_points_m": {
            "left_wrist": [0.123456789, 0.2, 0.1],
            "right_wrist": [0.3, -0.2, 0.1],
            "head": [0, 0, 0.4],
        },
        "tracking_orientations_wxyz": {
            name: [1, 0, 0, 0] for name in ("left_wrist", "right_wrist", "head")
        },
        "base_roll_pitch_radians": [0.01, -0.02],
        "camera_optical_origin_m": [0, 0, 0],
        "camera_optical_to_heading_rotation": np.eye(3).tolist(),
        "wrist_camera_kinematics": {"private_unused_transform": "must not appear"},
    }


class ObservationInterfaceTests(unittest.TestCase):
    def test_wrist_changes_survive_history_selection_and_reach_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = ActionExecutionWindow(
                action_id="reach", started_elapsed_seconds=0, endpoint_elapsed_seconds=2
            )
            expected_images = []
            roles: tuple[HistoryRole, ...] = ("start", "midpoint", "current/end")
            for index, role in enumerate(roles):
                identifier = (
                    execution.capture_identifier(role) if role != "current/end" else "current"
                )
                directory = root / identifier
                directory.mkdir()
                record = observation_record(identifier, float(index)).model_copy(
                    update={
                        "wrist_camera_names": ["left_wrist", "right_wrist"],
                        "action_frame": ActionFrameContext(
                            execution=execution, role=role, requested_elapsed_seconds=index
                        ),
                    }
                )
                atomic_json(directory / "observation.json", record.model_dump(mode="json"))
                for filename, content in (
                    ("rgb.png", b"unchanged head"),
                    ("external.png", b"unchanged external"),
                    ("left_wrist.png", b"left hand" + bytes([index])),
                    ("right_wrist.png", b"right hand" + bytes([index])),
                ):
                    (directory / filename).write_bytes(content)
                    expected_images.append(content)
            pairs = collect_visual_history(root / "current", root)
            self.assertEqual(len(pairs), 3)
            content = image_pair_content(pairs, root / "current")
            self.assertEqual(
                [
                    base64.b64decode(item["image_url"].split(",", 1)[1])
                    for item in content
                    if item["type"] == "input_image"
                ],
                expected_images,
            )
            self.assertEqual(
                json.loads(content[0]["text"])["view_order"],
                ["onboard RGB", "external RGB", "left wrist RGB", "right wrist RGB"],
            )
            (root / "current/left_wrist.png").write_bytes(b"changed after selection")
            with self.assertRaisesRegex(ValueError, "recorded camera hash"):
                image_pair_content(pairs, root / "current")

    def test_later_request_retains_initial_images_without_relabelling_them_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_images = []
            for identifier, captured in (("initial", 0.0), ("current", 3.0)):
                directory = root / identifier
                directory.mkdir()
                atomic_json(
                    directory / "observation.json",
                    observation_record(identifier, captured).model_dump(mode="json"),
                )
                for filename in ("rgb.png", "external.png"):
                    content = f"{identifier}:{filename}".encode()
                    (directory / filename).write_bytes(content)
                    expected_images.append(content)
            initial = collect_visual_history(root / "initial", root, include_episode_start=True)
            self.assertEqual(len(initial), 1)
            pairs = collect_visual_history(root / "current", root, include_episode_start=True)
            self.assertEqual(
                [pair.frames[0].role for pair in pairs], ["episode/start", "current/end"]
            )
            self.assertFalse(pairs[0].current)
            self.assertTrue(pairs[1].current)
            self.assertEqual(
                [pair.frames[0].captured_elapsed_seconds for pair in pairs], [0.0, 3.0]
            )
            self.assertEqual(
                [
                    base64.b64decode(item["image_url"].split(",", 1)[1])
                    for item in image_pair_content(pairs, root / "current")
                    if item["type"] == "input_image"
                ],
                expected_images,
            )
            self.assertEqual((root / "initial/rgb.png").read_bytes(), expected_images[0])

    def test_compact_feedback_retains_copy_precision_without_raw_joint_arrays(self) -> None:
        record = observation_record()
        feedback = concise_robot_feedback(record, measured_kinematics(), 4.25)
        self.assertEqual(
            feedback["copy_current_pose"]["tracking_points_m"]["left_wrist"][0], 0.123457
        )
        self.assertEqual(feedback["observation_age_seconds_at_prompt"], 1.25)
        self.assertNotIn("joint_positions_rad", json.dumps(feedback))
        self.assertNotIn("wrist_camera_kinematics", json.dumps(feedback))
        self.assertEqual(feedback["body_tilt_degrees"], {"roll": 0.57, "pitch": -1.15})

    def test_endpoint_errors_are_calculated_from_measured_pose_including_quaternion_sign(
        self,
    ) -> None:
        kinematics = measured_kinematics()
        kinematics["tracking_points_m"]["right_wrist"] = [0.33, -0.16, 0.1]
        kinematics["tracking_orientations_wxyz"]["right_wrist"] = (
            Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True).tolist()
        )
        kinematics["tracking_orientations_wxyz"]["head"] = [-1, 0, 0, 0]
        publication = SonicPlannerPublication(
            identifier="reach",
            source_response_sha256="a" * 64,
            episode_id="episode",
            observation_id="initial",
            observation_elapsed_wall_seconds=0,
            valid_until_elapsed_wall_seconds=30,
            request_task_completion=False,
            control_ticks=100,
            locomotion_mode=0,
            movement_direction_heading_frame=[1, 0, 0],
            facing_yaw_radians=0,
            speed_m_per_s=0,
            body_height_m=-1,
            poses=[
                PlannerTrackingPose(
                    positions_heading_frame=[0.1, 0.2, 0.1, 0.3, -0.2, 0.1, 0, 0, 0.4],
                    orientations_heading_frame_wxyz=[1, 0, 0, 0] * 3,
                )
            ]
            * 4,
        )
        feedback = previous_action_feedback(
            publication,
            ActionExecutionWindow(
                action_id="reach", started_elapsed_seconds=0.02, endpoint_elapsed_seconds=2
            ),
            observation_record(),
            kinematics,
        )
        self.assertEqual(feedback["tracking"]["right_wrist"]["position_error_m"], 0.05)
        self.assertEqual(feedback["tracking"]["right_wrist"]["orientation_error_degrees"], 90)
        self.assertEqual(feedback["tracking"]["head"]["orientation_error_degrees"], 0)
        self.assertEqual(feedback["sample_after_execution_endpoint_seconds"], 1)

    def test_depth_reports_selected_pixels_in_the_heading_frame_only(self) -> None:
        depth = np.ones((480, 640))
        optical = np.zeros((480, 640, 3))
        optical[:, :, 2] = 1
        snapshot = DepthSnapshot(depth, measured_kinematics(), 1, 1, optical_points_m=optical)
        result = snapshot.query('{"pixels":[{"x":320,"y":240}]}')
        self.assertEqual(result["points"][0]["point_heading_pelvis_m"], [0, 0, 1])
        self.assertNotIn("point_action_frame_m", result["points"][0])
        self.assertNotIn("action_coordinate_frame", result)

    def test_history_pairs_are_ordered_synchronized_deduplicated_and_byte_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            roles: tuple[HistoryRole, ...] = ("start", "midpoint", "current/end")
            for index, role in enumerate(roles):
                path = root / str(index)
                path.mkdir()
                record = observation_record(f"frame_{index}", float(index))
                atomic_json(path / "observation.json", record.model_dump(mode="json"))
                (path / "rgb.png").write_bytes(b"onboard" + bytes([index if index < 2 else 0]))
                (path / "external.png").write_bytes(
                    b"external" + bytes([index if index < 2 else 0])
                )
                frames.append(
                    (
                        path,
                        ImageFrame(
                            observation_id=f"frame_{index}",
                            action_id="action",
                            role=role,
                            captured_elapsed_seconds=index,
                            requested_elapsed_seconds=index,
                        ),
                    )
                )
            pairs = prepare_image_pairs(frames, root)
            self.assertEqual(len(pairs), 2)
            self.assertTrue(pairs[-1].current)
            self.assertEqual([frame.role for frame in pairs[-1].frames], ["start", "current/end"])
            self.assertEqual(
                sum(item["type"] == "input_image" for item in image_pair_content(pairs, root)), 4
            )
            with self.assertRaisesRegex(ValueError, "chronological"):
                prepare_image_pairs(frames[::-1], root)
            raw = json.loads((frames[1][0] / "observation.json").read_text())
            raw["observation_identity"]["episode_id"] = "other"
            atomic_json(frames[1][0] / "observation.json", raw)
            with self.assertRaisesRegex(ValueError, "mixes episodes"):
                prepare_image_pairs(frames, root)

    def test_short_action_end_waits_for_matching_native_receipt_and_both_history_frames(
        self,
    ) -> None:
        actor = RobotEpisode.__new__(RobotEpisode)
        actor.observation_history_enabled = True
        actor.stream = MagicMock()
        actor.stream.active_timeline.publication.identifier = "new"
        actor.execution_window = ActionExecutionWindow(
            action_id="old", started_elapsed_seconds=0, endpoint_elapsed_seconds=1
        )
        actor.captured_history_roles = {"start", "midpoint"}
        self.assertFalse(actor.current_capture_ready())
        actor.execution_window = ActionExecutionWindow(
            action_id="new", started_elapsed_seconds=2, endpoint_elapsed_seconds=2
        )
        actor.captured_history_roles = set()
        self.assertFalse(actor.current_capture_ready())
        actor.captured_history_roles = {"start", "midpoint"}
        self.assertTrue(actor.current_capture_ready())

    def test_capture_schedule_uses_execution_clock_and_retains_requested_vs_actual_times(
        self,
    ) -> None:
        actor = RobotEpisode.__new__(RobotEpisode)
        actor.observation_history_enabled = True
        actor.execution_window = ActionExecutionWindow(
            action_id="reach", started_elapsed_seconds=1, endpoint_elapsed_seconds=5
        )
        actor.captured_history_roles = set()
        actor.finish_reason = None
        actor.io = MagicMock()
        self.assertIsNone(actor.due_history_frame(0.9))
        context = actor.due_history_frame(1.03)
        self.assertIsNotNone(context)
        assert context is not None
        state = NativeVisualState(
            identifier="capture",
            episode_id="episode",
            episode_deadline_elapsed_wall_seconds=30,
            simulation_seconds=1,
            elapsed_wall_seconds=1.03,
            source_monotonic_seconds=101.03,
            joint_names=JOINT_NAMES,
            joint_positions=[0] * 29,
            joint_velocities=[0] * 29,
            root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        captured = actor.begin_history_capture(state, context, 101.03)
        assert captured.action_frame is not None
        self.assertEqual(captured.action_frame.requested_elapsed_seconds, 1)
        self.assertEqual(captured.elapsed_wall_seconds, 1.03)
        self.assertFalse(actor.io.submit.call_args.args[0].publish_ready)
        self.assertIsNone(actor.due_history_frame(2.9))
        midpoint = actor.due_history_frame(3)
        assert midpoint is not None
        self.assertEqual(midpoint.role, "midpoint")

    @unittest.skipUnless(sys.platform == "linux", "Evidence worker uses Linux affinity")
    def test_history_queue_publishes_only_current_ready_and_drains_original_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "physics"
            output.mkdir()
            commands = root / "commands"
            commands.mkdir()
            for identifier in ("start", "midpoint", "current"):
                capture = output / "camera-output" / identifier
                capture.mkdir(parents=True)
                for filename in (
                    "onboard.png",
                    "observer.png",
                    "depth-m.npy",
                    "optical-points-m.npy",
                ):
                    (capture / filename).write_bytes(filename.encode())
                atomic_json(
                    capture / "observation.json",
                    observation_record(identifier).model_dump(mode="json"),
                )
            worker = EpisodeIO(output, commands)
            camera_replies = LocalPushSender(output / "camera-events")
            try:
                worker.submit(
                    BeginEpisode(
                        EpisodeClock(
                            deadline_monotonic_seconds=time.monotonic() + 30,
                            episode_id="episode",
                            elapsed_wall_seconds=0,
                            deadline_elapsed_wall_seconds=30,
                        ),
                        {},
                        capture_initial_observation=False,
                    )
                )
                for identifier in ("start", "midpoint", "current"):
                    worker.submit(
                        CaptureSnapshot(
                            identifier,
                            {},
                            time.monotonic(),
                            publish_state=False,
                            publish_ready=identifier == "current",
                        )
                    )
                for identifier in ("start", "midpoint", "current"):
                    camera_replies.send(
                        CameraCaptureCompleted(
                            identifier=identifier,
                            captures=[
                                CameraCaptureIdentity(
                                    environment_index=0,
                                    episode_id="episode",
                                    identifier=identifier,
                                )
                            ],
                        )
                        .model_dump_json()
                        .encode()
                    )
                deadline = time.monotonic() + 5
                events = []
                while time.monotonic() < deadline and (not events):
                    events.extend(worker.events())
                    time.sleep(0.01)
                self.assertEqual(
                    [event.identifier for event in events if isinstance(event, ObservationReady)],
                    ["current"],
                )
            finally:
                camera_replies.close()
                worker.close()
            self.assertEqual(
                read_record(output / "observation-ready.json")["identifier"], "current"
            )
            for identifier in ("start", "midpoint", "current"):
                self.assertTrue((output / "observations" / identifier / "external.png").is_file())
