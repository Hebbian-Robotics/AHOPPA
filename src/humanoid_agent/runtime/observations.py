"""Allowlist remote Isaac sensor snapshots before they enter a model request."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from humanoid_agent.command_admission import EpisodeClock, ObservationIdentity
from humanoid_agent.dex3 import HandJointObservation
from humanoid_agent.observation_history import ActionFrameContext
from humanoid_agent.robot_camera import DEPTH_GRID_COLUMNS, DEPTH_GRID_ROWS
from humanoid_agent.robot_observations import RobotSensors, Vector3

DepthTileRow = Annotated[
    list[FiniteFloat | None], Field(min_length=DEPTH_GRID_COLUMNS, max_length=DEPTH_GRID_COLUMNS)
]


class IsaacObservation(BaseModel):
    """Unknown record fields are discarded so evaluator additions stay private."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    action_frame: ActionFrameContext | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    simulation_seconds: FiniteFloat = Field(ge=0)
    elapsed_wall_seconds: FiniteFloat = Field(ge=0)
    robot_sensors: RobotSensors
    hand_joint_sensors: dict[str, HandJointObservation] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    wrist_camera_names: list[Literal["left_wrist", "right_wrist"]] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    external_camera_available: bool = Field(default=False, exclude_if=lambda value: not value)
    depth_tiles_m: Annotated[
        list[DepthTileRow], Field(min_length=DEPTH_GRID_ROWS, max_length=DEPTH_GRID_ROWS)
    ]
    camera_intrinsic_matrix: Annotated[list[Vector3], Field(min_length=3, max_length=3)]
    observation_identity: ObservationIdentity

    @classmethod
    def parse_requested_snapshot(
        cls, payload: str | bytes, *, identifier: str, clock: EpisodeClock
    ) -> Self:
        observation = cls.model_validate_json(payload)
        identity = observation.observation_identity
        if (
            identity.identifier != identifier
            or identity.episode_id != clock.episode_id
            or identity.episode_deadline_elapsed_wall_seconds != clock.deadline_elapsed_wall_seconds
            or identity.captured_elapsed_wall_seconds != observation.elapsed_wall_seconds
        ):
            raise ValueError(
                "Camera snapshot identity or clock differs from the requested observation"
            )
        return observation
