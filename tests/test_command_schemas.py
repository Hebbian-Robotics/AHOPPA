"""Executable decisions preserve exact JSON values and remain valid after parsing."""

import json
import operator
import unittest
from typing import Any

from pydantic import ValidationError

from humanoid_agent.posture_control import PlannedIsaacChunk
from humanoid_agent.robot_observations import RobotSensors
from humanoid_agent.runtime.sdk import SonicMotorCommand
from humanoid_agent.sonic_planner_input import NativePlannerSample


def decision_payload() -> dict[str, Any]:
    return {
        "subtasks": [
            {
                "identifier": "reach",
                "description": "Reach",
                "completion_condition": "Hand at target",
                "status": "active",
            }
        ],
        "active_subtask_id": "reach",
        "observed_progress": "Starting",
        "request_task_completion": False,
        "walking": False,
        "speed_m_per_s": 0.3,
        "movement_direction": [1, 0, 0],
        "facing_yaw_radians": 0,
        "poses": [
            {"left_wrist": [0, 0, 0], "right_wrist": [0, 0, 0], "head": [0, 0, 0.4]}
            for _ in range(4)
        ],
        "orientations": [
            {"left_wrist": [1, 0, 0, 0], "right_wrist": [1, 0, 0, 0], "head": [1, 0, 0, 0]}
            for _ in range(4)
        ],
        "left_hand_joints": None,
        "right_hand_joints": None,
        "duration_seconds": 1,
        "locomotion_posture": "standing",
        "body_height_m": 0.45,
    }


class CommandSchemaTests(unittest.TestCase):
    def test_sensor_reordering_preserves_values_and_copied_updates_reject_mismatched_joints(
        self,
    ) -> None:
        sensors = RobotSensors(
            joint_names=["left", "right"],
            joint_positions_rad=[0.1, 0.2],
            joint_velocities_rad_per_s=[0.3, 0.4],
            imu_angular_velocity_body_rad_per_s=[0, 0, 0],
            imu_gravity_direction_body=[0, 0, -1],
        )
        reordered = sensors.in_joint_order(["right", "left"])
        self.assertEqual(list(reordered.joint_positions_rad), [0.2, 0.1])
        self.assertEqual(list(reordered.joint_velocities_rad_per_s), [0.4, 0.3])
        self.assertEqual(list(sensors.joint_names), ["left", "right"])
        with self.assertRaisesRegex(ValidationError, "matching lengths"):
            sensors.model_copy(update={"joint_positions_rad": [0.1]})
        with self.assertRaisesRegex(ValidationError, "unique"):
            sensors.model_copy(update={"joint_names": ["left", "left"]})

    def test_model_decisions_reject_coercion_and_unknown_fields_at_every_level(self) -> None:
        for field, value in [
            ("walking", "false"),
            ("walking", 0),
            ("request_task_completion", "true"),
            ("speed_m_per_s", "0.3"),
            ("duration_seconds", True),
            ("movement_direction", ["1", 0, 0]),
            ("body_height_m", float("nan")),
            ("unknown_command", True),
        ]:
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                PlannedIsaacChunk.model_validate_json(
                    json.dumps({**decision_payload(), field: value})
                )
        for field in ("subtasks", "poses", "orientations"):
            payload = decision_payload()
            payload[field][0]["unknown_command"] = True
            with self.subTest(nested=field), self.assertRaises(ValidationError):
                PlannedIsaacChunk.model_validate_json(json.dumps(payload))

    def test_parsed_commands_are_isolated_from_inputs_and_exported_values(self) -> None:
        payload = decision_payload()
        command = PlannedIsaacChunk.model_validate(payload)
        original = command.model_dump()
        self.assertEqual(original, payload)
        self.assertEqual(json.loads(command.model_dump_json()), payload)
        self.assertEqual(PlannedIsaacChunk.model_validate_json(command.model_dump_json()), command)
        payload["poses"][0]["head"][2] = 100
        exported = command.model_dump()
        exported["orientations"][0]["head"][0] = 100
        self.assertEqual(command.model_dump(), original)
        # Deliberately attempt operations forbidden by the public immutable types.
        with self.assertRaises(TypeError):
            operator.setitem(command.poses[0].head, 2, 100)  # ty: ignore[no-matching-overload]
        with self.assertRaises(TypeError):
            operator.setitem(command.orientations[0].head, 0, 100)  # ty: ignore[no-matching-overload]
        with self.assertRaises(ValidationError):
            command.subtasks[0].status = "unknown"  # ty: ignore[invalid-assignment]
        with self.assertRaises(ValidationError):
            command.walking = True  # ty: ignore[invalid-assignment]

    def test_copied_decisions_revalidate_nested_values_and_cross_field_invariants(self) -> None:
        command = PlannedIsaacChunk.model_validate(decision_payload())
        for update in [
            {"duration_seconds": "1"},
            {"poses": []},
            {"active_subtask_id": "missing"},
            {"walking": True, "locomotion_posture": "kneel_both"},
            {"unknown_command": True},
            {"orientations": [{"left_wrist": [0, 0, 0, 0]}] * 4},
        ]:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                command.model_copy(update=update)
        walking = command.model_copy(update={"walking": True, "speed_m_per_s": 0.4})
        self.assertTrue(walking.walking)
        self.assertFalse(command.walking)
        self.assertEqual(command.model_copy(deep=True), command)

    def test_sdk_motor_updates_cannot_bypass_physical_command_constraints(self) -> None:
        positions = [0.0] * 29
        command = SonicMotorCommand(
            positions=positions,
            velocities=[0.0] * 29,
            feedforward_torques=[0.0] * 29,
            stiffness=[1.0] * 29,
            damping=[0.1] * 29,
        )
        positions[0] = 100
        self.assertEqual(command.positions[0], 0)
        for update in [{"positions": [0.0]}, {"stiffness": [-1.0] * 29}, {"damping": ["0.1"] * 29}]:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                command.model_copy(update=update)

    def test_native_samples_require_integer_modes_and_unit_tracking_orientations(self) -> None:
        sample = NativePlannerSample(
            locomotion_mode=0,
            movement_direction_world=[1, 0, 0],
            facing_direction_world=[1, 0, 0],
            speed_m_per_s=-1,
            body_height_m=-1,
            vr_position_body_frame=[0.0] * 9,
            vr_orientation_body_frame_wxyz=[1, 0, 0, 0] * 3,
            endpoint_hold=False,
        )
        for update in [
            {"locomotion_mode": True},
            {"locomotion_mode": 1.0},
            {"locomotion_mode": "1"},
            {"vr_orientation_body_frame_wxyz": [0.0] * 12},
        ]:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                NativePlannerSample.model_validate_json(
                    json.dumps({**sample.model_dump(), **update})
                )
