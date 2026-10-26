"""Keep upstream parameter extraction isolated and preserve the Newton mapping."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from humanoid_agent.runtime.robot import (
    create_sonic_newton_robot_configuration,
    read_sonic_robot_parameters,
)

# Deliberately includes imports and a post-configuration assignment that must not run.
SOURCE_CONFIGURATION = b"""
import unavailable_simulator
STIFFNESS = 120.0
G1_CYLINDER_MODEL_12_DEX_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78), joint_pos={".*hip_pitch.*": -0.2}, joint_vel={".*": 0.0}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_.*", ".*knee.*"],
            effort_limit_sim={".*hip_.*": 88.0, ".*knee.*": 139.0},
            velocity_limit_sim=32.0, stiffness=STIFFNESS, damping=4.0, armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*"], effort_limit_sim=25.0,
            velocity_limit_sim={".*shoulder.*": 37.0}, stiffness=40.0, damping=2.0,
        ),
    },
)
must_not_execute = unavailable_after_configuration()
"""


class CapturedArticulation(SimpleNamespace):
    InitialStateCfg = SimpleNamespace


class SonicRobotConfigurationTests(unittest.TestCase):
    def test_hash_mismatch_rejects_source_before_executing_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "g1.py"
            source_path.write_bytes(SOURCE_CONFIGURATION)
            with self.assertRaisesRegex(ValueError, "differs from pinned revision"):
                read_sonic_robot_parameters(source_path)

    def test_extraction_and_newton_mapping_preserve_upstream_parameters(self) -> None:
        joint_drive = type("JointDriveCfg", (SimpleNamespace,), {"PDGainsCfg": SimpleNamespace})
        modules = {
            "isaaclab.sim": SimpleNamespace(
                UrdfFileCfg=SimpleNamespace,
                UrdfConverterCfg=SimpleNamespace(JointDriveCfg=joint_drive),
            ),
            "isaaclab.actuators": SimpleNamespace(ImplicitActuatorCfg=SimpleNamespace),
            "isaaclab.assets": SimpleNamespace(ArticulationCfg=CapturedArticulation),
            "isaaclab_newton.sim.schemas": SimpleNamespace(NewtonArticulationCfg=SimpleNamespace),
        }
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch(
                "humanoid_agent.runtime.robot.SONIC_CONFIGURATION_SHA256",
                hashlib.sha256(SOURCE_CONFIGURATION).hexdigest(),
            ),
            patch("humanoid_agent.runtime.robot.require_canonical_urdf"),
            patch(
                "humanoid_agent.runtime.robot.importlib.import_module",
                side_effect=modules.__getitem__,
            ),
        ):
            workspace = Path(temporary_directory)
            asset_root = workspace / "gear_sonic/data/assets"
            source_path = workspace / "gear_sonic/envs/manager_env/robots/g1.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(SOURCE_CONFIGURATION)
            parameters = read_sonic_robot_parameters(source_path)
            original_actuators = {
                name: vars(configuration).copy()
                for name, configuration in parameters.actuators.items()
            }
            configuration = create_sonic_newton_robot_configuration(asset_root)
            self.assertEqual(configuration.init_state.pos, (0.0, 0.0, 0.78))
            self.assertEqual(configuration.init_state.joint_pos, {".*hip_pitch.*": -0.2})
            self.assertEqual(configuration.init_state.joint_vel, {".*": 0.0})
            self.assertEqual(configuration.soft_joint_pos_limit_factor, 0.9)
            self.assertEqual(
                vars(configuration.spawn.joint_drive.gains), {"stiffness": 0.0, "damping": 0.0}
            )
            self.assertTrue(configuration.spawn.articulation_props.self_collision_enabled)
            self.assertTrue(configuration.spawn.activate_contact_sensors)
            self.assertFalse(configuration.spawn.fix_base)
            self.assertFalse(configuration.spawn.run_multi_physics_conversion)
            self.assertEqual(
                configuration.spawn.asset_path,
                str(asset_root / "robot_description/urdf/g1/main.urdf"),
            )
            for name, expected_fields in original_actuators.items():
                expected_newton_fields = expected_fields.copy()
                expected_newton_fields["joint_effort_limit"] = expected_newton_fields.pop(
                    "effort_limit_sim"
                )
                expected_newton_fields["joint_velocity_limit"] = expected_newton_fields.pop(
                    "velocity_limit_sim"
                )
                self.assertEqual(vars(configuration.actuators[name]), expected_newton_fields)
            self.assertEqual(parameters.actuators["legs"].stiffness, 120.0)
            self.assertEqual(parameters.actuators["legs"].armature, 0.01)

    def test_missing_or_wrong_configuration_has_an_explicit_contract_error(self) -> None:
        for source, message in (
            (b"OTHER_ROBOT = 1", "expected robot configuration assignment"),
            (b"G1_CYLINDER_MODEL_12_DEX_CFG = 1", "unexpected constructor"),
        ):
            with (
                self.subTest(source=source),
                tempfile.TemporaryDirectory() as temporary_directory,
                patch(
                    "humanoid_agent.runtime.robot.SONIC_CONFIGURATION_SHA256",
                    hashlib.sha256(source).hexdigest(),
                ),
            ):
                source_path = Path(temporary_directory) / "g1.py"
                source_path.write_bytes(source)
                with self.assertRaisesRegex(ValueError, message):
                    read_sonic_robot_parameters(source_path)
