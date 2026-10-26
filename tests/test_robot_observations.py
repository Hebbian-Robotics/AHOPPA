"""Verify the model boundary cannot expose evaluator truth or localization."""

import unittest

import numpy as np
from pydantic import ValidationError

from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.posture_control import (
    MotionTiming,
    PlannedPostureChunk,
    TrackingOrientations,
    command_timing,
    planner_mode_and_height,
    planner_movement_speed,
)
from humanoid_agent.robot_camera import IMAGE_HEIGHT, IMAGE_WIDTH, depth_tiles
from humanoid_agent.robot_observations import RobotSensors
from humanoid_agent.sensor_feedback import DepthSnapshot


class RobotObservationTests(unittest.TestCase):
    def test_tracking_rotations_reject_invalid_quaternions_before_execution(self) -> None:
        valid = {name: [1, 0, 0, 0] for name in ["left_wrist", "right_wrist", "head"]}
        self.assertEqual(list(TrackingOrientations.model_validate(valid).right_wrist), [1, 0, 0, 0])
        for invalid in [[0, 0, 0, 0], [2, 0, 0, 0], [float("nan"), 0, 0, 0]]:
            with self.subTest(quaternion=invalid), self.assertRaises(ValidationError):
                TrackingOrientations.model_validate({**valid, "right_wrist": invalid})

    def test_depth_grid_masks_invalid_ranges_without_disclosing_scene_labels(self) -> None:
        depth = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 2.0)
        depth[:40, :40] = np.nan
        depth[:40, 40:80] = 9.0
        depth[40:80, :40] = 0.1
        tiles = depth_tiles(depth)
        self.assertIsNone(tiles[0][0])
        self.assertIsNone(tiles[0][1])
        self.assertIsNone(tiles[1][0])
        self.assertEqual(tiles[1][1], 2.0)
        self.assertEqual((len(tiles), len(tiles[0])), (12, 16))

    def test_native_joint_order_preserves_measurements_and_rejects_missing_joints(
        self,
    ) -> None:
        sensors = RobotSensors(
            joint_names=JOINT_NAMES,
            joint_positions_rad=np.linspace(-0.1, 0.1, len(JOINT_NAMES)).tolist(),
            joint_velocities_rad_per_s=[0.0] * len(JOINT_NAMES),
            imu_angular_velocity_body_rad_per_s=[0.0, 0.0, 0.0],
            imu_gravity_direction_body=[0.0, 0.0, -1.0],
        )
        reordered = sensors.in_joint_order(list(reversed(JOINT_NAMES)))
        self.assertEqual(
            list(reordered.joint_positions_rad), list(reversed(sensors.joint_positions_rad))
        )
        self.assertEqual(list(reordered.joint_names), list(reversed(JOINT_NAMES)))
        with self.assertRaisesRegex(ValueError, "differ from the calibrated robot"):
            sensors.in_joint_order(JOINT_NAMES[:-1])
        with self.assertRaisesRegex(ValidationError, "matching lengths"):
            RobotSensors.model_validate({**sensors.model_dump(), "joint_positions_rad": [0.0]})

    def test_depth_queries_return_sensor_surfaces_in_the_declared_robot_frame(self) -> None:
        depth = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 2.0)
        depth[0, 0] = np.nan
        depth[240, 320] = 1.0
        snapshot = DepthSnapshot(
            depth,
            {
                "camera_optical_origin_m": [1.0, 2.0, 3.0],
                "camera_optical_to_heading_rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            },
            simulation_time=4.0,
            wall_time=4.1,
            optical_points_m=np.dstack((np.zeros_like(depth), np.zeros_like(depth), depth)),
        )
        result = snapshot.query('{"pixels":[{"x":320,"y":240},{"x":0,"y":0}]}')
        np.testing.assert_allclose(result["points"][0]["point_heading_pelvis_m"], [2, 2, 3])
        self.assertEqual(result["points"][0]["valid_samples"], 1)
        self.assertIsNone(result["points"][1]["point_heading_pelvis_m"])
        self.assertEqual(result["snapshot_wall_elapsed_seconds"], 4.1)

    def test_depth_query_rejects_unobservable_coordinates_and_extra_fields(self) -> None:
        snapshot = DepthSnapshot(
            np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 2.0),
            {
                "camera_optical_origin_m": [0.0, 0.0, 0.0],
                "camera_optical_to_heading_rotation": np.eye(3).tolist(),
            },
            simulation_time=0,
            wall_time=0,
            optical_points_m=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        )
        for arguments in [
            '{"pixels":[{"x":640,"y":0}]}',
            '{"pixels":[{"x":1.5,"y":0}]}',
            '{"pixels":[{"x":0,"y":0}],"target_world_xyz":[1,2,3]}',
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                snapshot.query(arguments)

    def test_posture_and_duration_commands_preserve_numeric_authority(self) -> None:
        command_fields = {
            "walking": False,
            "speed_m_per_s": 0.6,
            "movement_direction": [1, 0, 0],
            "facing_yaw_radians": 0,
            "poses": [{"left_wrist": [0, 0, 0], "right_wrist": [0, 0, 0], "head": [0, 0, 0]}] * 4,
            "subtasks": [
                {
                    "identifier": "reach",
                    "description": "Reach",
                    "completion_condition": "Reached",
                    "status": "active",
                }
            ],
            "active_subtask_id": "reach",
            "observed_progress": "Starting",
            "request_task_completion": False,
            "locomotion_posture": "standing",
            "body_height_m": 0.45,
            "duration_seconds": 2.414,
        }
        for posture in ["standing", "squat", "kneel_both", "kneel_single"]:
            command = PlannedPostureChunk.model_validate(
                {**command_fields, "locomotion_posture": posture}
            )
            self.assertEqual(planner_movement_speed(command), 0.0)
        walking = PlannedPostureChunk.model_validate(
            {**command_fields, "walking": True, "speed_m_per_s": 0.35}
        )
        self.assertEqual(planner_movement_speed(walking), 0.35)
        self.assertEqual(planner_mode_and_height(walking), (1, -1.0))
        squat = PlannedPostureChunk.model_validate(
            {**command_fields, "locomotion_posture": "squat"}
        )
        self.assertEqual(planner_mode_and_height(squat), (4, 0.45))
        for invalid_fields in [
            {"walking": True, "speed_m_per_s": float("nan")},
            {"walking": True, "locomotion_posture": "kneel_both"},
        ]:
            with self.subTest(fields=invalid_fields), self.assertRaises(ValidationError):
                PlannedPostureChunk.model_validate({**command_fields, **invalid_fields})
        ticks, keyframe_times = command_timing(MotionTiming(duration_seconds=2.414))
        self.assertEqual(ticks, 121)
        self.assertAlmostEqual(keyframe_times[-1], 2.42)
        for duration, expected_ticks in [(0.001, 1), (0.1, 5), (10.0, 500), (60.0, 3000)]:
            self.assertEqual(
                command_timing(MotionTiming(duration_seconds=duration))[0], expected_ticks
            )
        for invalid_duration in [float("nan"), float("inf"), 0, -1]:
            with self.assertRaises(ValidationError):
                MotionTiming(duration_seconds=invalid_duration)


if __name__ == "__main__":
    unittest.main()
