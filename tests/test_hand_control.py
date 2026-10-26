"""Grasp switches preserve model timing, measured starts and historical evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from humanoid_agent.command_admission import AdmissionReason, ObservationIdentity
from humanoid_agent.dex3 import HAND_JOINT_NAMES
from humanoid_agent.hand_control import (
    GraspCommand,
    HandTransitionInitials,
    load_pinned_hand_mapper,
    validate_grasp_timing,
)
from humanoid_agent.posture_control import PlannedIsaacChunk, PlannedTeleopIsaacChunk
from humanoid_agent.runtime.sdk_worker import HandReferenceLatch
from humanoid_agent.sonic_model_command import prepare_planner_publication
from humanoid_agent.sonic_planner_input import (
    ControllerObservation,
    SonicPlannerCommandStream,
    SonicPlannerPublication,
    motion_timeline,
    parse_motion_publication,
)
from tests.recording_fixtures import write_test_record


def mapped_positions(side: str, action: str) -> list[float]:
    return [0.0] * 7 if action == "open" else [0.2 if side == "left" else -0.2] * 7


class HandControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = patch(
            "humanoid_agent.hand_control.upstream_hand_positions", side_effect=mapped_positions
        )
        self.mapping.start()
        self.addCleanup(self.mapping.stop)
        self.positions = {
            "left_wrist": [0.1, 0.2, -0.1],
            "right_wrist": [0.1, -0.2, -0.1],
            "head": [0, 0, 0.4],
        }
        self.orientations = {name: [1, 0, 0, 0] for name in self.positions}
        self.planning = {
            "subtasks": [
                {
                    "identifier": "reach",
                    "description": "Reach",
                    "completion_condition": "Contact",
                    "status": "active",
                }
            ],
            "active_subtask_id": "reach",
            "observed_progress": "Ready",
            "request_task_completion": False,
        }
        self.hybrid = {
            **self.planning,
            "duration_seconds": 2.0,
            "walking": False,
            "speed_m_per_s": 0.2,
            "movement_direction": [1, 0, 0],
            "facing_yaw_radians": 0,
            "poses": [self.positions] * 4,
            "orientations": [self.orientations] * 4,
            "locomotion_posture": "standing",
            "body_height_m": 0.75,
        }
        self.identity = ObservationIdentity(
            episode_id="episode",
            identifier="initial",
            captured_elapsed_wall_seconds=0,
            episode_deadline_elapsed_wall_seconds=10,
        )
        self.limits = {
            name: {"lower": -2.0, "upper": 2.0}
            for names in HAND_JOINT_NAMES.values()
            for name in names
        }
        self.kinematics = {
            "tracking_points_m": self.positions,
            "tracking_orientations_wxyz": self.orientations,
        }
        self.initials = HandTransitionInitials(left=[0.1] * 7, right=[-0.1] * 7)

    def publication(self) -> SonicPlannerPublication:
        command = PlannedTeleopIsaacChunk.model_validate(
            {
                **self.hybrid,
                "right_grasp": {"action": "close", "start_seconds": 1, "transition_seconds": 1},
                "left_grasp": {"action": "open", "start_seconds": 0, "transition_seconds": 0.4},
            }
        )
        return prepare_planner_publication(
            command, self.identity, self.kinematics, "reach", "a" * 64, hand_limits=self.limits
        )

    def test_deferred_closure_independent_hands_and_truncated_deadline(self) -> None:
        publication = self.publication()
        self.assertEqual(publication.protocol, "sonic-native-planner-v3")
        timeline = motion_timeline(
            publication, [1, 0, 0, 0], hand_initials=self.initials, remaining_episode_seconds=1.5
        )
        np.testing.assert_allclose(
            np.asarray(timeline.sample(0.8, [1, 0, 0, 0]).right_hand_joints), [-0.1] * 7
        )
        np.testing.assert_allclose(
            np.asarray(timeline.sample(0.8, [1, 0, 0, 0]).left_hand_joints), [0] * 7
        )
        np.testing.assert_allclose(
            np.asarray(timeline.sample(1.5, [1, 0, 0, 0]).right_hand_joints), [-0.15] * 7
        )
        np.testing.assert_allclose(
            np.asarray(timeline.sample(9, [1, 0, 0, 0]).right_hand_joints), [-0.15] * 7
        )
        with self.assertRaisesRegex(ValueError, "measured"):
            motion_timeline(publication, [1, 0, 0, 0])

    def test_admission_duplicate_keeps_original_measured_initials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            stream = SonicPlannerCommandStream(
                episode / "physics/command-ledger", "episode", record=write_test_record
            )
            stream.register_observation(
                ControllerObservation(identity=self.identity, pelvis_quaternion_wxyz=[1, 0, 0, 0])
            )
            publication = self.publication()
            self.assertEqual(
                stream.activate(publication, 0, hand_initials=self.initials).reason,
                AdmissionReason.ACCEPTED,
            )
            self.assertEqual(
                stream.activate(
                    publication,
                    0.2,
                    hand_initials=HandTransitionInitials(left=[0] * 7, right=[0] * 7),
                ).reason,
                AdmissionReason.DUPLICATE,
            )
            self.assertEqual(
                json.loads(
                    (episode / "physics/command-ledger/hand-initials/reach.json").read_text()
                ),
                self.initials.model_dump(mode="json"),
            )

    def test_rejected_grasp_commands_do_not_require_motion_preparation_or_consume_valid_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stream = SonicPlannerCommandStream(Path(directory), "episode", record=write_test_record)
            stream.register_observation(
                ControllerObservation(identity=self.identity, pelvis_quaternion_wxyz=[1, 0, 0, 0])
            )
            publication = self.publication()
            for identifier, changes, elapsed, reason in (
                ("foreign", {"episode_id": "other_episode"}, 0, AdmissionReason.WRONG_EPISODE),
                ("expired", {}, 10, AdmissionReason.EXPIRED),
                (
                    "mismatch",
                    {"observation_elapsed_wall_seconds": 1},
                    0,
                    AdmissionReason.OBSERVATION_MISMATCH,
                ),
            ):
                with self.subTest(reason=reason):
                    rejected = publication.model_copy(update={"identifier": identifier, **changes})
                    self.assertEqual(stream.activate(rejected, elapsed).reason, reason)
                    self.assertIsNone(stream.sample(elapsed, [1, 0, 0, 0]))
            with self.assertRaisesRegex(ValueError, "measured admission"):
                stream.activate(publication, 0)
            self.assertNotIn(publication.identifier, stream.ledger.admissions)
            self.assertEqual(
                stream.activate(publication, 0, hand_initials=self.initials).reason,
                AdmissionReason.ACCEPTED,
            )

    def test_timing_and_missing_hardware_fail_before_execution(self) -> None:
        for duration in (1.009, 0.9):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "finish"):
                validate_grasp_timing(
                    GraspCommand(action="close", start_seconds=0.6, transition_seconds=0.409),
                    duration,
                    hands_available=True,
                )
        with self.assertRaisesRegex(ValueError, "feedback"):
            validate_grasp_timing(
                GraspCommand(action="open", start_seconds=0), 1, hands_available=False
            )

    def test_direct_publications_cannot_schedule_grasps_after_motion_completion(self) -> None:
        publication = self.publication().model_dump(mode="json")
        parsed = parse_motion_publication(json.dumps(publication))
        timeline = motion_timeline(parsed, [1, 0, 0, 0], hand_initials=self.initials)
        np.testing.assert_allclose(
            np.asarray(timeline.sample(2, [1, 0, 0, 0]).right_hand_joints), [-0.2] * 7
        )
        for shortened_ticks in (99, 50):
            with self.assertRaisesRegex(ValidationError, "Grasp transition must finish"):
                parse_motion_publication(
                    json.dumps({**publication, "control_ticks": shortened_ticks})
                )
        with self.assertRaisesRegex(ValidationError, "finite motion duration"):
            parse_motion_publication(json.dumps({**publication, "control_ticks": 10**400}))

    def test_legacy_json_preserves_original_keys(self) -> None:
        publication = prepare_planner_publication(
            PlannedIsaacChunk.model_validate(self.hybrid),
            self.identity,
            self.kinematics,
            "legacy",
            "c" * 64,
        )
        raw = publication.model_dump(mode="json")
        self.assertNotIn("left_grasp_schedule", raw)
        self.assertNotIn("right_grasp_schedule", raw)
        self.assertEqual(parse_motion_publication(json.dumps(raw)).model_dump(mode="json"), raw)

    def test_relay_retains_explicit_grip_without_changing_legacy_omission(self) -> None:
        timeline = motion_timeline(self.publication(), [1, 0, 0, 0], hand_initials=self.initials)
        close = timeline.sample(2, [1, 0, 0, 0])
        latch = HandReferenceLatch()
        latch.resolve(close)
        hold = close.model_copy(update={"right_hand_joints": None, "left_hand_joints": None})
        resolved = latch.resolve(hold)
        np.testing.assert_allclose(np.asarray(resolved.right_hand_joints), [-0.2] * 7)
        np.testing.assert_allclose(np.asarray(resolved.left_hand_joints), [0] * 7)
        legacy = hold.model_copy(update={"retain_uncommanded_hand_targets": False})
        self.assertIsNone(latch.resolve(legacy).right_hand_joints)
        release = hold.model_copy(update={"right_hand_joints": [0] * 7})
        np.testing.assert_allclose(np.asarray(latch.resolve(release).right_hand_joints), [0] * 7)
        np.testing.assert_allclose(np.asarray(latch.resolve(hold).right_hand_joints), [0] * 7)

    def test_changed_upstream_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            path = source / "gear_sonic/utils/teleop/solver/hand/g1_gripper_ik_solver.py"
            path.parent.mkdir(parents=True)
            path.write_text("raise RuntimeError('untrusted mapping must never execute')")
            with self.assertRaisesRegex(ValueError, "differs from pinned"):
                load_pinned_hand_mapper(source)
