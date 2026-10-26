"""Build the pinned Newton GPU scene with released SONIC robot parameters."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

from scipy.spatial.transform import Rotation

from humanoid_agent.canonical_robot import imu_mounts
from humanoid_agent.isaac_workcell import (
    SIMPLE_ROOM_FLOOR_PATH,
    SIMREADY_ASSET_PATHS,
    SIMREADY_ASSET_REVISION,
    SIMREADY_VISUAL_ONLY_VARIANTS,
    TASK_CONTACT_PAIRS,
    USD_ASSET_PATHS,
    WORKCELL_GROUND_HALF_EXTENT_M,
    GroundWorkcellScene,
)
from humanoid_agent.runtime.robot import create_sonic_newton_robot_configuration
from humanoid_agent.shutdown_task import (
    SEKTION_CABINET_USD,
    TASK_DRAWER_BODIES,
    TASK_DRAWER_JOINT,
    UNUSED_CABINET_JOINTS,
    ShutdownTask,
)

logger = logging.getLogger(__name__)


# The room's floor meshes; with flat-ground support they stay visual only.
SIMPLE_ROOM_FLOOR_COLLIDER_PATHS = (
    SIMPLE_ROOM_FLOOR_PATH,
    "Towel_Room01_florr_edge_262/Towel_Room01_florr_edge",
    "Floor/SM_Template_Map_Floor",
)


def _room_asset_configuration_type(
    assets: Any, *, remove_table: bool, floor_collision: bool
) -> Any:
    """Apply USD overrides before Isaac Lab snapshots the physics prototypes."""
    suppressed_names = ("GroundPlane", "table_low_327") if remove_table else ("GroundPlane",)

    class RoomAssetConfiguration(assets.AssetBaseCfg):
        def _post_spawn(self, stage: Any) -> None:
            super()._post_spawn(stage)
            simulation: Any = importlib.import_module("isaaclab.sim")
            room_paths = simulation.find_matching_prim_paths(self.prim_path)
            if not room_paths:
                raise ValueError("Spawned room is missing: " + self.prim_path)
            for room_path in room_paths:
                for relative_path in suppressed_names:
                    path = room_path + "/" + relative_path
                    prim = stage.GetPrimAtPath(path)
                    if not prim.IsValid():
                        raise ValueError(
                            "Authored room layout differs from the inspected asset: " + path
                        )
                    prim.SetActive(False)
                if floor_collision:
                    continue
                physics: Any = importlib.import_module("pxr.UsdPhysics")
                for relative_path in SIMPLE_ROOM_FLOOR_COLLIDER_PATHS:
                    prim = stage.GetPrimAtPath(room_path + "/" + relative_path)
                    if not prim.IsValid():
                        raise ValueError("Room floor collider is missing: " + relative_path)
                    physics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)

    return RoomAssetConfiguration


def simready_asset_root() -> Path:
    """Host copy of the pinned SimReady revision; override with HUMANOID_AGENT_SIMREADY_ROOT."""
    root = Path(
        os.environ.get(
            "HUMANOID_AGENT_SIMREADY_ROOT",
            f"/home/ubuntu/webolt/runtime-assets/simready-assets-{SIMREADY_ASSET_REVISION[:8]}",
        )
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Pinned SimReady assets are missing: {root}")
    return root


def _visual_only_asset_configuration_type(assets: Any) -> Any:
    """Strip physics from a visual-only asset before Newton imports it, then verify."""

    class VisualOnlyAssetConfiguration(assets.AssetBaseCfg):
        def _post_spawn(self, stage: Any) -> None:
            super()._post_spawn(stage)
            simulation: Any = importlib.import_module("isaaclab.sim")
            usd: Any = importlib.import_module("pxr.Usd")
            physics: Any = importlib.import_module("pxr.UsdPhysics")
            paths = simulation.find_matching_prim_paths(self.prim_path)
            if not paths:
                raise ValueError("Spawned visual asset is missing: " + self.prim_path)
            for path in paths:
                # The released physics-disabled variant still authors door and lock bodies;
                # strip every physics schema so Newton imports this asset as visuals only.
                while instances := [
                    prim for prim in usd.PrimRange(stage.GetPrimAtPath(path)) if prim.IsInstance()
                ]:
                    for prim in instances:
                        prim.SetInstanceable(False)
                for prim in list(usd.PrimRange(stage.GetPrimAtPath(path))):
                    if prim.IsA(physics.Joint):
                        prim.SetActive(False)
                        continue
                    for schema in (
                        physics.RigidBodyAPI,
                        physics.MassAPI,
                        physics.ArticulationRootAPI,
                    ):
                        if prim.HasAPI(schema):
                            prim.RemoveAPI(schema)
                    if prim.HasAPI(physics.CollisionAPI):
                        physics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                for prim in usd.PrimRange(stage.GetPrimAtPath(path), usd.TraverseInstanceProxies()):
                    if prim.HasAPI(physics.CollisionAPI) and (
                        physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
                    ):
                        raise ValueError(
                            "Visual-only asset retains a collider: " + str(prim.GetPath())
                        )
                    if prim.HasAPI(physics.RigidBodyAPI) or prim.IsA(physics.Joint):
                        raise ValueError(
                            "Visual-only asset retains physics: " + str(prim.GetPath())
                        )

    return VisualOnlyAssetConfiguration


def register_newton_scene_schemas(*, rendering: bool = False) -> None:
    """Register installed USD import schemas without activating PhysX physics."""
    importlib.import_module("newton_usd_schemas")
    installed_schemas: Any = importlib.import_module("ovphysx")
    plugins: Any = importlib.import_module("pxr.Plug")
    for schema_path in installed_schemas.codeless_schema_paths():
        plugins.Registry().RegisterPlugins(str(schema_path))
    if rendering:
        renderer: Any = importlib.import_module("ovrtx")
        renderer.register_schema_paths()


def create_newton_simulation_configuration() -> Any:
    """Select CUDA explicitly; contact capacity is fixed before world allocation."""
    upstream: Any = importlib.import_module("isaaclab_tasks.core.velocity.config.g1.flat_env_cfg")
    configuration = upstream.G1FlatEnvCfg().sim
    configuration.device = "cuda:0"
    configuration.dt = 0.005
    # SONIC consumes 200 Hz physics observations. Keep the validated single-step
    # integration and allocate contacts for the workcell, beyond the flat task.
    configuration.physics = configuration.physics.default
    configuration.physics.num_substeps = 1
    configuration.physics.solver_cfg.nconmax = 1000
    configuration.physics.solver_cfg.njmax = 6000
    configuration.physics.use_cuda_graph = True
    configuration.render_interval = 2**31 - 1
    return configuration


def launch_newton_runtime(configuration: Any) -> Any:
    """Use the upstream headless launcher, including its lifecycle cleanup."""
    launcher: Any = importlib.import_module("isaaclab.app")
    return launcher.launch_simulation(configuration, {"device": "cuda:0", "viz": "none"})


def _create_newton_robot_ground_configuration(
    asset_root: Path,
    num_environments: int,
    environment_spacing_m: float,
    *,
    dex3_hands: bool = False,
) -> Any:
    simulation: Any = importlib.import_module("isaaclab.sim")
    assets: Any = importlib.import_module("isaaclab.assets")
    scenes: Any = importlib.import_module("isaaclab.scene")
    material_configurations: Any = importlib.import_module(
        "isaaclab.sim.spawners.materials.physics_materials_cfg"
    )
    if not 1 <= num_environments <= 16 or environment_spacing_m < 12:
        raise ValueError("Workcell batches require 1–16 environments spaced at least 12 metres")
    configuration = scenes.InteractiveSceneCfg(
        num_envs=num_environments,
        env_spacing=2.0 if num_environments == 1 else environment_spacing_m,
    )
    robot_configuration = create_sonic_newton_robot_configuration(asset_root, dex3_hands=dex3_hands)
    robot_configuration.prim_path = "{ENV_REGEX_NS}/Robot"
    configuration.robot = robot_configuration
    configuration.ground = assets.AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=simulation.CuboidCfg(
            size=(
                2 * WORKCELL_GROUND_HALF_EXTENT_M + environment_spacing_m * (num_environments - 1),
            )
            * 2
            + (0.1,),
            collision_props=simulation.UsdPhysicsCollisionCfg(),
            physics_material=material_configurations.RigidBodyMaterialBaseCfg(
                static_friction=1, dynamic_friction=1, restitution=0
            ),
            visual_material=simulation.PreviewSurfaceCfg(
                diffuse_color=(0.32, 0.34, 0.33), roughness=0.95
            ),
        ),
        init_state=assets.AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.05)),
    )
    return configuration


def _add_imu_sensor_configuration(configuration: Any, asset_root: Path) -> None:
    sensors: Any = importlib.import_module("isaaclab.sensors")
    configuration.lazy_sensor_update = False
    for name, (parent_body, position) in imu_mounts(
        asset_root / "robot_description/urdf/g1/main.urdf"
    ).items():
        body_path = (
            "{ENV_REGEX_NS}/Robot/Geometry/pelvis"
            if parent_body == "pelvis"
            else "{ENV_REGEX_NS}/Robot/Geometry/.*/" + parent_body
        )
        setattr(
            configuration,
            "imu_" + name,
            sensors.ImuCfg(
                prim_path=body_path,
                update_period=0,
                offset=sensors.ImuCfg.OffsetCfg(pos=position),
            ),
        )


def _add_contact_sensor_configuration(
    configuration: Any, *, floor_collider_path: str = "/World/Ground/geometry/mesh"
) -> None:
    sensors: Any = importlib.import_module("isaaclab.sensors")
    configuration.lazy_sensor_update = False
    for sensor_name, body_name, _ in TASK_CONTACT_PAIRS:
        body_path = (
            "{ENV_REGEX_NS}/Robot/Geometry/pelvis"
            if body_name == "pelvis"
            else "{ENV_REGEX_NS}/Robot/Geometry/.*/" + body_name
        )
        contact_shape = (
            floor_collider_path
            if "floor" in sensor_name
            else "{ENV_REGEX_NS}/Task_target_button/geometry/mesh"
        )
        setattr(
            configuration,
            "task_contact_" + sensor_name,
            sensors.ContactSensorCfg(
                prim_path=body_path,
                update_period=0,
                history_length=4,
                filter_shape_prim_expr=[contact_shape],
            ),
        )


# Every articulated G1 body; measurements split hand bodies from the rest by name.
SHUTDOWN_ROBOT_BODY_PATTERN = "{ENV_REGEX_NS}/Robot/Geometry/(pelvis|.*/.*_link)"


def _cabinet_configuration_type(assets: Any, limit_damping_n_s_per_m: float) -> Any:
    """Author the task drawer's stop damping before Newton imports the articulation."""

    class CabinetConfiguration(assets.ArticulationCfg):
        def _post_spawn(self, stage: Any) -> None:
            super()._post_spawn(stage)
            simulation: Any = importlib.import_module("isaaclab.sim")
            sdf: Any = importlib.import_module("pxr.Sdf")
            author_path = (
                self.spawn.spawn_path
                if self.spawn is not None and self.spawn.spawn_path is not None
                else self.prim_path
            )
            cabinet_paths = simulation.find_matching_prim_paths(author_path)
            if not cabinet_paths:
                raise ValueError("Spawned cabinet is missing: " + author_path)
            for cabinet_path in cabinet_paths:
                joint = stage.GetPrimAtPath(cabinet_path + "/sektion/" + TASK_DRAWER_JOINT)
                if not joint.IsValid():
                    raise ValueError("Authored cabinet differs from the inspected asset")
                joint.CreateAttribute("newton:limitDamping", sdf.ValueTypeNames.Float).Set(
                    limit_damping_n_s_per_m
                )

    return CabinetConfiguration


def _add_shutdown_cabinet_configuration(configuration: Any, task: ShutdownTask) -> None:
    """Reuse Isaac Lab's Sektion cabinet; only the task drawer loses its closing drive."""
    assets: Any = importlib.import_module("isaaclab.assets")
    upstream: Any = importlib.import_module("isaaclab_tasks.core.cabinet.cabinet_env_cfg")
    upstream_cabinet = upstream.CABINET_CFG
    upstream_drawers = upstream_cabinet.actuators["drawers"]
    if not upstream_cabinet.spawn.usd_path.endswith("/" + SEKTION_CABINET_USD):
        raise ValueError("Upstream cabinet asset differs from the inspected Sektion cabinet")
    cabinet_configuration_type = _cabinet_configuration_type(
        assets, task.drawer_limit_damping_n_s_per_m
    )
    for cabinet in task.cabinets:
        setattr(
            configuration,
            "cabinet_" + cabinet.identifier,
            cabinet_configuration_type(
                spawn=upstream_cabinet.spawn,
                prim_path="{ENV_REGEX_NS}/Cabinet_" + cabinet.identifier,
                init_state=upstream_cabinet.init_state.replace(
                    pos=cabinet.base_center_m,
                    rot=tuple(
                        Rotation.from_euler("z", cabinet.yaw_degrees, degrees=True).as_quat()
                    ),
                    joint_pos={
                        **dict.fromkeys(UNUSED_CABINET_JOINTS, 0.0),
                        TASK_DRAWER_JOINT: task.initial_drawer_position_m(cabinet),
                    },
                ),
                actuators={
                    # Passive slide: no position drive can close the drawer without contact.
                    "task_drawer": upstream_drawers.replace(
                        joint_names_expr=[TASK_DRAWER_JOINT],
                        stiffness=0.0,
                        damping=task.drawer_damping_n_s_per_m,
                        friction=task.drawer_static_friction_n,
                    ),
                    # Other doors and drawers keep their upstream closing drives.
                    "unused_drawer": upstream_drawers.replace(
                        joint_names_expr=["drawer_bottom_joint"]
                    ),
                    "doors": upstream_cabinet.actuators["doors"],
                },
            ),
        )


def _add_shutdown_contact_configuration(configuration: Any, task: ShutdownTask) -> None:
    sensors: Any = importlib.import_module("isaaclab.sensors")
    for cabinet in task.cabinets:
        cabinet_root = "{ENV_REGEX_NS}/Cabinet_" + cabinet.identifier
        # Only drawer and handle contact can move the slide; the fixed carcass cannot.
        setattr(
            configuration,
            "shutdown_drawer_contacts_" + cabinet.identifier,
            sensors.ContactSensorCfg(
                prim_path=SHUTDOWN_ROBOT_BODY_PATTERN,
                update_period=0,
                history_length=4,
                filter_shape_prim_expr=[f"{cabinet_root}/{body}/.*" for body in TASK_DRAWER_BODIES],
            ),
        )


def create_newton_workcell_scene(
    asset_root: Path,
    scene_definition: GroundWorkcellScene,
    *,
    contact_sensors: bool,
    imu_sensors: bool | None = None,
    display_collision_boxes: bool,
    num_environments: int = 1,
    environment_spacing_m: float = 12.0,
    writable_marker_material: bool = False,
    decorative_assets: Path | None = None,
    camera_configurations: dict[str, Any] | None = None,
) -> Any:
    """Register physical assets with InteractiveScene so Newton imports the world."""
    simulation: Any = importlib.import_module("isaaclab.sim")
    assets: Any = importlib.import_module("isaaclab.assets")
    scenes: Any = importlib.import_module("isaaclab.scene")
    configuration = _create_newton_robot_ground_configuration(
        asset_root, num_environments, environment_spacing_m, dex3_hands=scene_definition.dex3_hands
    )
    if scene_definition.room_asset is not None:
        if scene_definition.floor_support == "flat-ground-collider":
            # The standard flat ground carries the robot under the room's visual floor.
            configuration.ground.spawn.visible = False
        else:
            # The authored room owns support; do not add a second ground surface.
            configuration.ground = None
    for box in scene_definition.boxes:
        setattr(
            configuration,
            "task_" + box.name,
            assets.AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Task_" + box.name,
                spawn=simulation.CuboidCfg(
                    size=tuple(box.full_size_m),
                    collision_props=simulation.UsdPhysicsCollisionCfg(),
                    visual_material=simulation.PreviewSurfaceCfg(diffuse_color=tuple(box.color)),
                    # Decorative bundles replace painted boxes; hidden boxes are colliders only.
                    visible=box.finish != "hidden" and display_collision_boxes,
                ),
                init_state=assets.AssetBaseCfg.InitialStateCfg(pos=tuple(box.center_m)),
            ),
        )
    if scene_definition.assets:
        nucleus: Any = importlib.import_module("isaaclab.utils.assets")
        for furniture in scene_definition.assets:
            asset_configuration_type = (
                _room_asset_configuration_type(
                    assets,
                    remove_table=furniture.asset == "simple_room_without_table",
                    floor_collision=scene_definition.floor_support == "room-floor-mesh",
                )
                if furniture.asset in {"simple_room", "simple_room_without_table"}
                else _visual_only_asset_configuration_type(assets)
                if furniture.asset in SIMREADY_ASSET_PATHS
                else assets.AssetBaseCfg
            )
            spawn = (
                simulation.UsdFileCfg(
                    usd_path=f"{simready_asset_root()}/{SIMREADY_ASSET_PATHS[furniture.asset]}",
                    variants=dict(SIMREADY_VISUAL_ONLY_VARIANTS),
                )
                if furniture.asset in SIMREADY_ASSET_PATHS
                else simulation.UsdFileCfg(
                    usd_path=f"{nucleus.ISAAC_NUCLEUS_DIR}/{USD_ASSET_PATHS[furniture.asset]}"
                )
            )
            # Use the same USD spawner as Isaac Lab's official spawn_prims tutorial.
            setattr(
                configuration,
                "asset_" + furniture.name,
                asset_configuration_type(
                    prim_path="{ENV_REGEX_NS}/Asset_" + furniture.name,
                    spawn=spawn,
                    init_state=assets.AssetBaseCfg.InitialStateCfg(
                        pos=tuple(furniture.position_m),
                        rot=tuple(
                            Rotation.from_euler("z", furniture.yaw_degrees, degrees=True).as_quat()
                        ),
                    ),
                ),
            )
    _add_shutdown_cabinet_configuration(configuration, scene_definition.shutdown)
    configuration.task_button = assets.AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Task_target_button",
        spawn=simulation.SphereCfg(
            radius=scene_definition.button_radius_m,
            collision_props=simulation.UsdPhysicsCollisionCfg(),
            visual_material=simulation.PreviewSurfaceCfg(
                diffuse_color=tuple(scene_definition.button_color)
            ),
            visible=True,
        ),
        init_state=assets.AssetBaseCfg.InitialStateCfg(pos=tuple(scene_definition.button_center_m)),
    )
    if writable_marker_material:
        configuration.marker_material = assets.VisualMaterialCfg(
            prim_path="{ENV_REGEX_NS}/Task_target_button/geometry/material",
            spawn=None,
            channels=("color",),
        )
    if decorative_assets is not None:
        configuration.industrial_visuals = assets.AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/IndustrialVisuals",
            spawn=simulation.UsdFileCfg(
                usd_path=str(decorative_assets / "industrial-workcell.usda")
            ),
        )
    enable_imu_sensors = contact_sensors if imu_sensors is None else imu_sensors
    if enable_imu_sensors:
        _add_imu_sensor_configuration(configuration, asset_root)
    if contact_sensors:
        _add_contact_sensor_configuration(
            configuration, floor_collider_path=scene_definition.floor_collider_path
        )
        _add_shutdown_contact_configuration(configuration, scene_definition.shutdown)
    for camera_name, camera_configuration in (camera_configurations or {}).items():
        setattr(configuration, camera_name, camera_configuration)
    scene = scenes.InteractiveScene(configuration)
    room = scene_definition.room_asset
    if room is not None:
        suppressed_paths = ["GroundPlane"]
        if room.asset == "simple_room_without_table":
            suppressed_paths.append("table_low_327")
        for environment_path in scene.env_prim_paths:
            root = environment_path + "/Asset_" + room.name
            for relative_path in suppressed_paths:
                prim = scene.stage.GetPrimAtPath(root + "/" + relative_path)
                if not prim.IsValid():
                    raise ValueError(
                        "Authored room layout differs from the inspected asset: "
                        + root
                        + "/"
                        + relative_path
                    )
                if prim.IsActive():
                    raise ValueError(
                        "Room override was not applied before scene assembly: "
                        + str(prim.GetPath())
                    )
            floor = scene.stage.GetPrimAtPath(root + "/" + SIMPLE_ROOM_FLOOR_PATH)
            if not floor.IsValid():
                raise ValueError("Authored room floor collider is missing")
            geometry: Any = importlib.import_module("pxr.UsdGeom")
            usd: Any = importlib.import_module("pxr.Usd")
            bounds = (
                geometry.BBoxCache(usd.TimeCode.Default(), ["default", "render", "proxy"])
                .ComputeWorldBound(floor)
                .ComputeAlignedRange()
            )
            if abs(bounds.GetMax()[2]) > 1e-5:
                raise ValueError(
                    "Imported room floor is not aligned with the task's zero-height floor"
                )
        # USD visibility/activity after scene assembly does not establish what
        # Newton imported. Inspect the retained upstream source prototype too.
        newton_cloner: Any = importlib.import_module("isaaclab_newton.cloner")
        source_path = scene.env_prim_paths[0] + "/Asset_" + room.name
        prototype = newton_cloner.copy_newton_clone_source(source_path)
        unexpected_colliders = [
            label
            for label in prototype.shape_label
            if any(name in label.split("/") for name in suppressed_paths)
        ]
        if unexpected_colliders:
            raise ValueError(
                "Suppressed room geometry remains in Newton: " + repr(unexpected_colliders)
            )
        if scene_definition.floor_support == "flat-ground-collider":
            newton: Any = importlib.import_module("newton")
            # Match full room-relative paths: Floor2-5 reuse the floor mesh name for walls.
            colliding_floor = [
                label
                for label, flags in zip(prototype.shape_label, prototype.shape_flags, strict=True)
                if any(label.endswith("/" + path) for path in SIMPLE_ROOM_FLOOR_COLLIDER_PATHS)
                and flags & newton.ShapeFlags.COLLIDE_SHAPES
            ]
            if colliding_floor:
                raise ValueError(
                    "Room floor meshes still collide in Newton: " + repr(colliding_floor)
                )
        logger.info(
            "Verified %d imported room shapes exclude %s before physics release",
            len(prototype.shape_label),
            suppressed_paths,
        )
    return scene
