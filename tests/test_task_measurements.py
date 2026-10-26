"""Private shutdown scoring reads named encoders, floor contacts and episode-local poses."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanoid_agent.canonical_robot import BODY_NAMES, POINT_OFFSETS
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.runtime.evaluation import GroundTaskEvaluation
from humanoid_agent.runtime.measurements import TaskMeasurementSnapshot, measurements
from humanoid_agent.shutdown_task import TASK_DRAWER_JOINT

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CABINET_JOINTS = ["drawer_bottom_joint", TASK_DRAWER_JOINT]
CONTACT_BODIES = ["right_wrist_yaw_link", "torso_link"]


def storage_scene() -> GroundWorkcellScene:
    return GroundWorkcellScene.model_validate_json(
        (CONFIGS / "shutdown-storage-open-A.json").read_text()
    )


def snapshot_at(
    scene: GroundWorkcellScene,
    root_world: list[float],
    *,
    knee_angles: tuple[float, float] = (0.2, 0.2),
    drawer_a_position_m: float = 0.2,
    hand_force_n: float = 0.0,
) -> TaskMeasurementSnapshot:
    shutdown_fields = {}
    for cabinet in scene.shutdown.cabinets:
        position = drawer_a_position_m if cabinet.identifier == "A" else 0.0
        history = np.zeros((1, 4, len(CONTACT_BODIES), 1, 3))
        if cabinet.identifier == "A":
            history[0, 1, 0, 0] = [0, 0, hand_force_n]
        shutdown_fields[f"cabinet_{cabinet.identifier}_joint_positions"] = np.asarray(
            [[0.0, position]]
        )
        shutdown_fields[f"cabinet_{cabinet.identifier}_joint_velocities"] = np.zeros((1, 2))
        shutdown_fields[f"cabinet_{cabinet.identifier}_drawer_normal_force_history"] = history
    return TaskMeasurementSnapshot(
        root_states=np.asarray([root_world]),
        body_positions=np.asarray([[[0.0, 0.0, 1.0]]]) + np.asarray(root_world[:3]),
        body_quaternions_xyzw=np.asarray([[[0, 0, 0, 1]]]),
        joint_positions=np.asarray([[knee_angles[1], knee_angles[0]]]),
        shutdown_fields=shutdown_fields,
        cabinet_joint_names=CABINET_JOINTS,
        shutdown_contact_body_names=CONTACT_BODIES,
    )


def contact_evidence(**forces: float) -> dict[str, object]:
    names = (
        "pelvis_floor",
        "torso_floor",
        "left_foot_floor",
        "right_foot_floor",
        "left_knee_floor",
        "right_knee_floor",
    )
    return {
        "pairs": {
            name: {"peak_normal_force_since_previous_control_tick_n": forces.get(name, 0)}
            for name in names
        }
    }


class TaskMeasurementsTests(unittest.TestCase):
    robot = SimpleNamespace(
        body_names=["torso_link"], joint_names=["right_knee_joint", "left_knee_joint"]
    )

    def test_named_encoders_floor_contacts_and_drawer_forces_are_episode_local(self) -> None:
        scene = storage_scene()
        origin = [10.0, -5.0, 0.0]
        root = [10.25, -4.8, 0.76, 0, 0, 0, 1, 0, 0, 0.03, 0, 0, 0]
        measured = measurements(
            self.robot,
            scene,
            contact_evidence(left_foot_floor=15, right_foot_floor=20, left_knee_floor=40),
            environment_origin_m=origin,
            snapshot=snapshot_at(scene, root, knee_angles=(1.6, 1.7), hand_force_n=3.0),
        )
        self.assertAlmostEqual(measured.root_distance_from_start_m, np.hypot(0.25, 0.2))
        self.assertAlmostEqual(measured.root_vertical_speed_m_per_s, 0.03)
        self.assertEqual(measured.left_knee_angle_rad, 1.6)
        self.assertEqual(measured.right_knee_angle_rad, 1.7)
        self.assertEqual(measured.left_knee_floor_force_n, 40)
        self.assertEqual(measured.right_foot_floor_force_n, 20)
        self.assertEqual(measured.drawer("A"), (0.2, 0.0, 3.0, 0.0))
        self.assertEqual(measured.drawer("B"), (0.0, 0.0, 0.0, 0.0))

    def test_return_home_uses_the_configured_start_in_each_environment(self) -> None:
        scene = storage_scene()
        evidence = contact_evidence(left_foot_floor=30, right_foot_floor=30)
        for environment_origin_m in (None, [10.0, -5.0, 0.0]):
            origin = (
                np.zeros(3) if environment_origin_m is None else np.asarray(environment_origin_m)
            )
            for root_local_xy_m, expected_distance_m, expected_returned in (
                ([0.0, 0.0], 0.0, True),
                ([0.25, 0.0], 0.25, True),
                ([0.3, 0.4], 0.5, False),
            ):
                with self.subTest(origin=environment_origin_m, root=root_local_xy_m):
                    position = origin + np.asarray([*root_local_xy_m, 0.8])
                    root = [*position, 0, 0, 0, 1, *([0] * 6)]
                    measured = measurements(
                        self.robot,
                        scene,
                        evidence,
                        environment_origin_m=environment_origin_m,
                        snapshot=snapshot_at(scene, root, drawer_a_position_m=0.0),
                    )
                    self.assertAlmostEqual(measured.root_distance_from_start_m, expected_distance_m)
                    evaluation = GroundTaskEvaluation(scene.shutdown)
                    for _ in range(100):
                        evaluation.update(measured)
                    self.assertEqual(evaluation.physical_criterion_met, expected_returned)

    def test_tracking_points_keep_their_canonical_offsets(self) -> None:
        self.assertEqual(BODY_NAMES, ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"])
        np.testing.assert_array_equal(
            POINT_OFFSETS, [[0.18, -0.025, 0], [0.18, 0.025, 0], [0, 0, 0.35]]
        )
