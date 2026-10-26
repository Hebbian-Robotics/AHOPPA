"""Adapt the hash-pinned SONIC configuration to Newton without copying parameters."""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from humanoid_agent.canonical_robot import require_canonical_urdf
from humanoid_agent.dex3 import HAND_JOINT_NAMES, articulated_hand_urdf, hand_joint_limits

SONIC_CONFIGURATION_SHA256 = "42f9b4394eb21512735da4b1ca78c2d0546c4221c7bdaf8b3a186d8dc5588ba9"


class SourceConfiguration(SimpleNamespace):
    """Capture upstream constructor arguments without importing its old simulator."""

    InitialStateCfg = SimpleNamespace


def read_sonic_robot_parameters(source_path: Path) -> SourceConfiguration:
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SONIC_CONFIGURATION_SHA256:
        raise ValueError("SONIC robot configuration differs from pinned revision 45c03fac")
    assignments: list[ast.stmt] = []
    configuration_found = False
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            assignments.append(statement)
            if any(
                isinstance(target, ast.Name) and target.id == "G1_CYLINDER_MODEL_12_DEX_CFG"
                for target in statement.targets
            ):
                configuration_found = True
                break
    if not configuration_found:
        raise ValueError("Pinned SONIC source lacks the expected robot configuration assignment")
    configuration_types = SimpleNamespace(
        UrdfFileCfg=SimpleNamespace,
        RigidBodyPropertiesCfg=SimpleNamespace,
        ArticulationRootPropertiesCfg=SimpleNamespace,
        UrdfConverterCfg=SimpleNamespace(
            JointDriveCfg=type(
                "SourceJointDrive", (SimpleNamespace,), {"PDGainsCfg": SimpleNamespace}
            )
        ),
    )
    namespace: dict[str, Any] = {
        "__builtins__": {},
        "sim_utils": configuration_types,
        "ArticulationCfg": SourceConfiguration,
        "ImplicitActuatorCfg": SimpleNamespace,
    }
    # Execute only the assignments in the exact trusted upstream file. Imports,
    # training integrations and the later action-scale loop are not executed.
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), str(source_path), "exec"), namespace
    )
    parameters = namespace["G1_CYLINDER_MODEL_12_DEX_CFG"]
    if not isinstance(parameters, SourceConfiguration):
        raise ValueError("Pinned SONIC robot configuration has an unexpected constructor")
    return parameters


def create_sonic_newton_robot_configuration(asset_root: Path, *, dex3_hands: bool = False) -> Any:
    simulation: Any = importlib.import_module("isaaclab.sim")
    actuators: Any = importlib.import_module("isaaclab.actuators")
    assets: Any = importlib.import_module("isaaclab.assets")
    schemas: Any = importlib.import_module("isaaclab_newton.sim.schemas")
    source = asset_root.parents[2] / "gear_sonic/envs/manager_env/robots/g1.py"
    parameters = read_sonic_robot_parameters(source)
    require_canonical_urdf(asset_root / "robot_description/urdf/g1/main.urdf")
    newton_actuators = {}
    for group, original in parameters.actuators.items():
        fields = vars(original).copy()
        fields["joint_effort_limit"] = fields.pop("effort_limit_sim")
        fields["joint_velocity_limit"] = fields.pop("velocity_limit_sim")
        newton_actuators[group] = actuators.ImplicitActuatorCfg(**fields)
    urdf_path = asset_root / "robot_description/urdf/g1/main.urdf"
    if dex3_hands:
        limits = hand_joint_limits(urdf_path)
        newton_actuators["dex3_hands"] = actuators.ImplicitActuatorCfg(
            joint_names_expr=[*HAND_JOINT_NAMES["left"], *HAND_JOINT_NAMES["right"]],
            joint_effort_limit={name: limit["effort"] for name, limit in limits.items()},
            joint_velocity_limit={name: limit["velocity"] for name, limit in limits.items()},
            stiffness=1.5,
            damping=0.1,
        )
        # Initial gains match Dex3Hands::sizeCommand. Live gains are taken from
        # the unchanged native driver's DDS commands, not this startup value.
        urdf_path = articulated_hand_urdf(urdf_path)
    return assets.ArticulationCfg(
        spawn=simulation.UrdfFileCfg(
            fix_base=False,
            asset_path=str(urdf_path),
            activate_contact_sensors=True,
            joint_drive=simulation.UrdfConverterCfg.JointDriveCfg(
                gains=simulation.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    **vars(parameters.spawn.joint_drive.gains)
                )
            ),
            articulation_props=schemas.NewtonArticulationCfg(
                self_collision_enabled=parameters.spawn.articulation_props.enabled_self_collisions
            ),
            run_multi_physics_conversion=False,
            self_collision=True,
        ),
        init_state=assets.ArticulationCfg.InitialStateCfg(**vars(parameters.init_state)),
        soft_joint_pos_limit_factor=parameters.soft_joint_pos_limit_factor,
        actuators=newton_actuators,
    )
