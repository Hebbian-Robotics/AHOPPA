"""Released robot geometry keeps encoder FK, camera mounting and Dex3 fingers consistent."""

import importlib
import os
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.canonical_robot import (
    JOINT_NAMES,
    ControllerConstants,
    build_encoder_model,
    canonical_urdf_path,
    measured_tracking_points,
)
from humanoid_agent.dex3 import HAND_JOINT_NAMES, articulated_hand_urdf, hand_joint_limits
from humanoid_agent.paths import PROJECT_ROOT
from humanoid_agent.robot_observations import RobotSensors
from humanoid_agent.sensor_feedback import EncoderKinematics

# MuJoCo exposes its public API through compiled extension modules without stubs.
mujoco: Any = importlib.import_module("mujoco")


def load_controller_constants() -> ControllerConstants:
    constants_path = Path(
        os.environ.get(
            "SONIC_CONSTANTS", PROJECT_ROOT / "artifacts/upstream-assets/sonic_constants.json"
        )
    )
    return ControllerConstants.model_validate_json(constants_path.read_text())


class EncoderKinematicsTests(unittest.TestCase):
    def test_verified_encoder_model_preserves_measurements_across_joint_permutations(self) -> None:
        sensors = RobotSensors(
            joint_names=JOINT_NAMES,
            joint_positions_rad=load_controller_constants().default_angles,
            joint_velocities_rad_per_s=[0.0] * len(JOINT_NAMES),
            imu_angular_velocity_body_rad_per_s=[0.0, 0.0, 0.0],
            imu_gravity_direction_body=[0.0, 0.0, -1.0],
        )
        kinematics = EncoderKinematics()
        reordered = sensors.in_joint_order(list(reversed(JOINT_NAMES)))
        self.assertEqual(kinematics.measure(sensors), kinematics.measure(reordered))

    def test_camera_mount_follows_measured_waist_pitch_without_an_independent_neck(self) -> None:
        encoder = EncoderKinematics()
        angles = list(load_controller_constants().default_angles)
        gravity = [0.0, 0.0, -1.0]
        measurements = []
        for waist_pitch in (0.0, np.deg2rad(30)):
            angles[JOINT_NAMES.index("waist_pitch_joint")] = waist_pitch
            sensors = RobotSensors(
                joint_names=JOINT_NAMES,
                joint_positions_rad=angles,
                joint_velocities_rad_per_s=[0.0] * 29,
                imu_angular_velocity_body_rad_per_s=[0.0] * 3,
                imu_gravity_direction_body=gravity,
            )
            measurement = encoder.measure(sensors)
            measurements.append(measurement)
            head_rotation = Rotation.from_quat(
                measurement["tracking_orientations_wxyz"]["head"], scalar_first=True
            )
            optical_rotation = np.asarray(measurement["camera_optical_to_heading_rotation"])
            np.testing.assert_allclose(
                head_rotation.inv().as_matrix() @ optical_rotation,
                encoder.calibration["optical_to_torso_rotation"],
                atol=1e-12,
            )
            expected_displacement = head_rotation.apply([0.0576235, 0.01753, 0.07987])
            np.testing.assert_allclose(
                np.subtract(
                    measurement["camera_optical_origin_m"], measurement["tracking_points_m"]["head"]
                ),
                expected_displacement,
                atol=1e-12,
            )
        self.assertAlmostEqual(
            measurements[1]["camera_downward_pitch_degrees"]
            - measurements[0]["camera_downward_pitch_degrees"],
            30,
        )
        self.assertGreater(
            measurements[1]["camera_optical_origin_m"][0],
            measurements[0]["camera_optical_origin_m"][0],
        )


class ArticulatedHandAssetTests(unittest.TestCase):
    def test_enabling_fingers_preserves_body_fk_and_moves_the_finger_links(self) -> None:
        original_model, original_data = build_encoder_model()
        urdf = canonical_urdf_path()
        root = ElementTree.fromstring(articulated_hand_urdf(urdf).read_bytes())
        settings = ElementTree.SubElement(root, "mujoco")
        ElementTree.SubElement(settings, "compiler", fusestatic="false", discardvisual="true")
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for geometry in link.findall(tag):
                    link.remove(geometry)
        specification = mujoco.MjSpec.from_string(ElementTree.tostring(root, encoding="unicode"))
        specification.body("pelvis").add_freejoint(name="root")
        articulated_model = specification.compile()
        articulated_data = mujoco.MjData(articulated_model)
        self.assertEqual(articulated_model.nv - original_model.nv, 14)
        angles = load_controller_constants().default_angles
        limits = hand_joint_limits(urdf)
        finger_before = None
        for closed in (False, True):
            for model, data in (
                (original_model, original_data),
                (articulated_model, articulated_data),
            ):
                data.qpos[:7] = [0.3, -0.2, 0.76, 1, 0, 0, 0]
                for name, angle in zip(JOINT_NAMES, angles, strict=True):
                    data.qpos[model.jnt_qposadr[model.joint(name).id]] = angle
            for side in ("left", "right"):
                for name in HAND_JOINT_NAMES[side]:
                    joint = articulated_model.joint(name).id
                    limit = limits[name]
                    articulated_data.qpos[articulated_model.jnt_qposadr[joint]] = (
                        (limit["lower"] + limit["upper"]) / 2 if closed else 0
                    )
            mujoco.mj_forward(original_model, original_data)
            mujoco.mj_forward(articulated_model, articulated_data)
            for original, articulated in zip(
                measured_tracking_points(original_model, original_data),
                measured_tracking_points(articulated_model, articulated_data),
                strict=True,
            ):
                np.testing.assert_allclose(original, articulated, atol=1e-12, rtol=0)
            finger = articulated_data.xpos[
                articulated_model.body("right_hand_index_1_link").id
            ].copy()
            if finger_before is None:
                finger_before = finger
            else:
                self.assertGreater(float(np.linalg.norm(finger - finger_before)), 0.01)


if __name__ == "__main__":
    unittest.main()
