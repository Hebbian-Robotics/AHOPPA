"""Finite native planner commands preserve model authority and endpoint behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from humanoid_agent.command_admission import AdmissionReason, ObservationIdentity
from humanoid_agent.posture_control import PlannedIsaacChunk
from humanoid_agent.sonic_model_command import prepare_planner_publication
from humanoid_agent.sonic_planner_input import (
    ControllerObservation,
    NativePlannerSample,
    PlannerTrackingPose,
    SonicPlannerCommandStream,
    SonicPlannerProtocol,
    SonicPlannerPublication,
)
from tests.recording_fixtures import write_test_record


class SonicPlannerInputTests(unittest.TestCase):
    def test_restored_controller_basis_must_match_its_registered_observation(self) -> None:
        for mutation in ("renamed", "different_identity", "unregistered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stream = SonicPlannerCommandStream(root, "episode_a", record=write_test_record)
                observation = ControllerObservation(
                    identity=ObservationIdentity(
                        episode_id="episode_a",
                        identifier="initial",
                        captured_elapsed_wall_seconds=0,
                        episode_deadline_elapsed_wall_seconds=60,
                    ),
                    pelvis_quaternion_wxyz=[1, 0, 0, 0],
                )
                stream.register_observation(observation)
                SonicPlannerCommandStream(root, "episode_a", record=write_test_record)
                basis_path = stream.basis_directory / "initial.json"
                if mutation == "renamed":
                    basis_path.rename(stream.basis_directory / "other.json")
                elif mutation == "different_identity":
                    changed = observation.model_copy(
                        update={
                            "identity": observation.identity.model_copy(
                                update={"captured_elapsed_wall_seconds": 1}
                            )
                        }
                    )
                    basis_path.write_text(changed.model_dump_json())
                else:
                    (stream.ledger.observation_directory / "initial.json").unlink()
                with self.assertRaisesRegex(ValueError, "IMU basis differs"):
                    SonicPlannerCommandStream(root, "episode_a", record=write_test_record)

    def test_model_decision_preserves_numerical_keyframes_and_rejects_discontinuous_start(
        self,
    ) -> None:
        body_names = ("left_wrist", "right_wrist", "head")
        positions = {
            "left_wrist": [0.1, 0.2, -0.1],
            "right_wrist": [0.1, -0.2, -0.1],
            "head": [0, 0, 0.4],
        }
        orientations = {name: [1, 0, 0, 0] for name in body_names}
        poses = [{name: list(value) for name, value in positions.items()} for _ in range(4)]
        poses[-1]["right_wrist"][0] = 0.3
        decision = PlannedIsaacChunk.model_validate(
            {
                "subtasks": [
                    {
                        "identifier": "reach",
                        "description": "Reach",
                        "completion_condition": "Hand at target",
                        "status": "active",
                    }
                ],
                "active_subtask_id": "reach",
                "observed_progress": "Ready",
                "request_task_completion": False,
                "walking": False,
                "speed_m_per_s": 0.2,
                "movement_direction": [1, 0, 0],
                "facing_yaw_radians": 0,
                "poses": poses,
                "orientations": [orientations] * 4,
                "duration_seconds": 1.01,
                "locomotion_posture": "kneel_both",
                "body_height_m": 0.45,
            }
        )
        identity = ObservationIdentity(
            episode_id="episode_a",
            identifier="snapshot_1",
            captured_elapsed_wall_seconds=10,
            episode_deadline_elapsed_wall_seconds=60,
        )
        kinematics = {
            "tracking_points_m": positions,
            "tracking_orientations_wxyz": orientations,
        }
        publication = prepare_planner_publication(
            decision, identity, kinematics, "motion_1", "a" * 64
        )
        self.assertEqual(publication.control_ticks, 50)
        self.assertEqual(publication.locomotion_mode, 5)
        self.assertEqual(publication.speed_m_per_s, 0)
        self.assertEqual(publication.body_height_m, 0.45)
        self.assertEqual(
            list(publication.poses[-1].positions_heading_frame),
            [0.1, 0.2, -0.1, 0.3, -0.2, -0.1, 0, 0, 0.4],
        )
        self.assertEqual(publication.observation_id, "snapshot_1")
        self.assertEqual(publication.protocol, "sonic-native-planner-v2")
        long_decision = decision.model_copy(update={"duration_seconds": 30.0})
        self.assertEqual(
            prepare_planner_publication(
                long_decision, identity, kinematics, "long_motion", "c" * 64
            ).control_ticks,
            1500,
        )
        for duration in [50.1, 1e308]:
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(ValueError, "remaining episode budget"),
            ):
                prepare_planner_publication(
                    decision.model_copy(update={"duration_seconds": duration}),
                    identity,
                    kinematics,
                    "too_long",
                    "d" * 64,
                )
        legacy_publication = prepare_planner_publication(
            decision,
            identity,
            kinematics,
            "legacy_motion",
            "e" * 64,
            command_protocol="sonic-native-planner-v1",
        )
        self.assertEqual(legacy_publication.protocol, "sonic-native-planner-v1")
        for duration in [0.399, 4.001]:
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "Legacy v1"):
                prepare_planner_publication(
                    decision.model_copy(update={"duration_seconds": duration}),
                    identity,
                    kinematics,
                    "legacy_invalid",
                    "f" * 64,
                    command_protocol="sonic-native-planner-v1",
                )
        with self.assertRaisesRegex(ValidationError, "20–200"):
            legacy_publication.model_copy(update={"control_ticks": 500})
        invalid_payload = decision.model_dump()
        invalid_payload["poses"][0]["right_wrist"][0] = 0.5
        invalid_decision = PlannedIsaacChunk.model_validate(invalid_payload)
        with self.assertRaisesRegex(ValueError, "First keyframe"):
            prepare_planner_publication(
                invalid_decision, identity, kinematics, "motion_2", "b" * 64
            )

    def test_deadline_truncates_v2_motion_without_compressing_keyframes(self) -> None:
        protocols: tuple[SonicPlannerProtocol, ...] = (
            "sonic-native-planner-v1",
            "sonic-native-planner-v2",
        )
        for protocol in protocols:
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory:
                stream = SonicPlannerCommandStream(
                    Path(directory), "episode_a", record=write_test_record
                )
                stream.register_observation(
                    ControllerObservation(
                        identity=ObservationIdentity(
                            episode_id="episode_a",
                            identifier="snapshot_1",
                            captured_elapsed_wall_seconds=10,
                            episode_deadline_elapsed_wall_seconds=60,
                        ),
                        pelvis_quaternion_wxyz=[1, 0, 0, 0],
                    )
                )
                command = SonicPlannerPublication(
                    protocol=protocol,
                    identifier="motion_1",
                    source_response_sha256="a" * 64,
                    episode_id="episode_a",
                    observation_id="snapshot_1",
                    observation_elapsed_wall_seconds=10,
                    valid_until_elapsed_wall_seconds=60,
                    request_task_completion=False,
                    control_ticks=200,
                    locomotion_mode=1,
                    movement_direction_heading_frame=[1, 0, 0],
                    facing_yaw_radians=0,
                    speed_m_per_s=0.3,
                    body_height_m=-1,
                    poses=[
                        PlannerTrackingPose(
                            positions_heading_frame=[index / 10, 0.2, 0.4] * 3,
                            orientations_heading_frame_wxyz=[1, 0, 0, 0] * 3,
                        )
                        for index in range(4)
                    ],
                )
                self.assertEqual(stream.activate(command, 59).reason, AdmissionReason.ACCEPTED)
                current = stream.sample(59.5, [1, 0, 0, 0])
                assert isinstance(current, NativePlannerSample)
                self.assertFalse(current.endpoint_hold)
                np.testing.assert_allclose(current.vr_position_body_frame, [0.0375, 0.2, 0.4] * 3)
                truncated = protocol == "sonic-native-planner-v2"
                self.assertEqual(stream.endpoint_reached(60), truncated)
                held = stream.sample(61, [1, 0, 0, 0])
                assert isinstance(held, NativePlannerSample)
                self.assertEqual(held.endpoint_hold, truncated)
                self.assertEqual(held.speed_m_per_s, 0 if truncated else 0.3)
                np.testing.assert_allclose(
                    held.vr_position_body_frame,
                    [0.075 if truncated else 0.15, 0.2, 0.4] * 3,
                )
                self.assertEqual(
                    stream.activate(command.model_copy(update={"identifier": "late"}), 63).reason,
                    AdmissionReason.EXPIRED,
                )

    def test_finite_walk_stops_once_and_holds_the_model_endpoint_during_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stream = SonicPlannerCommandStream(
                Path(directory), "episode_a", record=write_test_record
            )
            stream.register_observation(
                ControllerObservation(
                    identity=ObservationIdentity(
                        episode_id="episode_a",
                        identifier="snapshot_1",
                        captured_elapsed_wall_seconds=10,
                        episode_deadline_elapsed_wall_seconds=60,
                    ),
                    pelvis_quaternion_wxyz=[2**-0.5, 0, 0, 2**-0.5],
                )
            )
            command = SonicPlannerPublication(
                identifier="motion_1",
                source_response_sha256="a" * 64,
                episode_id="episode_a",
                observation_id="snapshot_1",
                observation_elapsed_wall_seconds=10,
                valid_until_elapsed_wall_seconds=40,
                request_task_completion=True,
                control_ticks=50,
                locomotion_mode=1,
                movement_direction_heading_frame=[1, 0, 0],
                facing_yaw_radians=0,
                speed_m_per_s=0.3,
                body_height_m=-1,
                poses=[
                    PlannerTrackingPose(
                        positions_heading_frame=[index / 10, 0.2, 0.4] * 3,
                        orientations_heading_frame_wxyz=[1, 0, 0, 0] * 3,
                    )
                    for index in range(4)
                ],
            )
            self.assertEqual(stream.activate(command, 39.9).reason, AdmissionReason.ACCEPTED)
            self.assertEqual(stream.activate(command, 40).reason, AdmissionReason.DUPLICATE)
            with patch.object(
                stream.active_timeline,
                "sample",
                side_effect=AssertionError("interpolation during deadline check"),
            ):
                self.assertFalse(stream.endpoint_reached(40.4))
                self.assertTrue(stream.endpoint_reached(45))
            current = stream.sample(40.4, [0, 0, 0, 1])
            assert isinstance(current, NativePlannerSample)
            self.assertFalse(current.endpoint_hold)
            self.assertEqual(current.locomotion_mode, 1)
            np.testing.assert_allclose(current.movement_direction_world, [0, 1, 0], atol=1e-12)
            np.testing.assert_allclose(current.facing_direction_world, [0, 1, 0], atol=1e-12)
            np.testing.assert_allclose(
                current.vr_position_body_frame, [0.15, 0.2, 0.4] * 3, atol=1e-12
            )
            held = stream.sample(45, [1, 0, 0, 0])
            assert isinstance(held, NativePlannerSample)
            self.assertTrue(held.endpoint_hold)
            self.assertEqual(held.locomotion_mode, 0)
            self.assertEqual(held.speed_m_per_s, 0)
            np.testing.assert_allclose(held.vr_position_body_frame, [0.3, 0.2, 0.4] * 3)
            # A reconnect cannot start the already admitted motion again.
            reopened = SonicPlannerCommandStream(
                Path(directory), "episode_a", record=write_test_record
            )
            self.assertEqual(reopened.activate(command, 46).reason, AdmissionReason.DUPLICATE)
            self.assertIsNone(reopened.sample(46, [1, 0, 0, 0]))
            self.assertFalse(reopened.endpoint_reached(46))

            kneel = command.model_copy(
                update={
                    "identifier": "motion_2",
                    "locomotion_mode": 5,
                    "speed_m_per_s": 0,
                    "body_height_m": 0.45,
                    "valid_until_elapsed_wall_seconds": 60,
                }
            )
            self.assertEqual(stream.activate(kneel, 46).reason, AdmissionReason.ACCEPTED)
            self.assertEqual(stream.activate(command, 46.2).reason, AdmissionReason.DUPLICATE)
            self.assertEqual(
                stream.activate(
                    command.model_copy(update={"request_task_completion": False}), 46.3
                ).reason,
                AdmissionReason.IDENTIFIER_CONFLICT,
            )
            with self.assertRaisesRegex(ValueError, "cannot be interrupted"):
                stream.activate(command.model_copy(update={"identifier": "motion_3"}), 46.5)
            # Expiry holds the requested kneeling posture, not an implicit stand.
            tilted_hold = stream.sample(50, [2**-0.5, 2**-0.5, 0, 0])
            assert isinstance(tilted_hold, NativePlannerSample)
            self.assertTrue(tilted_hold.endpoint_hold)
            self.assertEqual(tilted_hold.locomotion_mode, 5)
            self.assertEqual(tilted_hold.body_height_m, 0.45)
            np.testing.assert_allclose(
                tilted_hold.vr_position_body_frame, [0.3, 0.4, -0.2] * 3, atol=1e-12
            )
            np.testing.assert_allclose(
                tilted_hold.vr_orientation_body_frame_wxyz,
                [2**-0.5, -(2**-0.5), 0, 0] * 3,
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
