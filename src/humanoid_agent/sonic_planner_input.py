"""Finite model commands for the unchanged SONIC planner and VR input interface.

Only model keyframe timing, coordinate conversion and finite-command expiry
belong here. SONIC owns lower-body context, inference, replanning and blending.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    FiniteFloat,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
    model_validator,
)
from scipy.spatial.transform import Rotation, Slerp

from humanoid_agent.command_admission import (
    AdmissionReason,
    CommandAdmission,
    CommandAdmissionLedger,
    ObservationIdentity,
)
from humanoid_agent.dex3 import HandVector
from humanoid_agent.hand_control import (
    FiniteGraspPublication,
    HandTransitionInitials,
    HandTransitionSampler,
)
from humanoid_agent.recording import read_record_bytes, record_paths
from humanoid_agent.stepped_turn import SteppedTurn
from humanoid_agent.validated_types import (
    ImmutableSequence,
    TrackingQuaternions,
    UnitQuaternion,
    ValidatedModel,
    Vector3,
)

PositionTargets = Annotated[ImmutableSequence[FiniteFloat], Field(min_length=9, max_length=9)]
OrientationTargets = TrackingQuaternions
SonicPlannerProtocol = Literal[
    "sonic-native-planner-v1", "sonic-native-planner-v2", "sonic-native-planner-v3"
]
CURRENT_SONIC_PLANNER_PROTOCOL: SonicPlannerProtocol = "sonic-native-planner-v2"


def require_integer_locomotion_mode(value: object) -> object:
    # Literal integer validation otherwise accepts equal booleans and floats.
    if type(value) is not int:
        raise ValueError("Locomotion mode must be an integer")
    return value


LocomotionMode = Annotated[Literal[0, 1, 4, 5, 6], BeforeValidator(require_integer_locomotion_mode)]


class PlannerTrackingPose(ValidatedModel):
    positions_heading_frame: PositionTargets
    orientations_heading_frame_wxyz: OrientationTargets


class SonicPlannerPublication(FiniteGraspPublication):
    """Locomotion uses observation heading; tracking uses live pelvis heading.

    Tracking targets remain pelvis-relative as the robot turns. Locomotion's
    direction/facing are anchored once to the identified observation's IMU.
    """

    protocol: SonicPlannerProtocol = CURRENT_SONIC_PLANNER_PROTOCOL
    locomotion_mode: LocomotionMode
    movement_direction_heading_frame: Vector3
    facing_yaw_radians: FiniteFloat = Field(ge=-3.141593, le=3.141593)
    speed_m_per_s: FiniteFloat = Field(ge=0, le=0.8)
    body_height_m: FiniteFloat
    left_hand_joints: HandVector | None = None
    right_hand_joints: HandVector | None = None
    poses: ImmutableSequence[PlannerTrackingPose] = Field(min_length=4, max_length=4)
    # A stepped in-place turn relative to the measured heading at activation.
    turn_in_place_radians: FiniteFloat | None = Field(
        default=None, ge=-math.pi - 1e-6, le=math.pi + 1e-6, exclude_if=lambda value: value is None
    )
    # Approach the requested facing with the command held near the measured heading.
    leashed_facing: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_serializer(mode="wrap")
    def serialize_legacy_protocol(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        values = handler(self)
        if self.protocol != "sonic-native-planner-v3":
            values.pop("left_grasp_schedule", None)
            values.pop("right_grasp_schedule", None)
        return values

    @model_validator(mode="after")
    def require_consistent_locomotion(self) -> SonicPlannerPublication:
        if self.protocol != "sonic-native-planner-v3" and (
            self.left_grasp_schedule is not None or self.right_grasp_schedule is not None
        ):
            raise ValueError("Timed grasp schedules require planner protocol v3")
        if self.protocol == "sonic-native-planner-v3" and (
            self.left_hand_joints is not None or self.right_hand_joints is not None
        ):
            raise ValueError("Teleop grasp protocol rejects individual model finger targets")
        if self.protocol == "sonic-native-planner-v1" and not 20 <= self.control_ticks <= 200:
            raise ValueError("Legacy v1 commands require 20–200 control ticks")
        if self.locomotion_mode != 1 and self.speed_m_per_s != 0:
            raise ValueError("Only standing locomotion permits walking speed")
        if self.turn_in_place_radians is not None and (
            self.locomotion_mode != 0 or self.facing_yaw_radians != 0
        ):
            raise ValueError("An in-place turn stands still and sets no separate facing change")
        if self.locomotion_mode in {0, 1}:
            if self.body_height_m != -1:
                raise ValueError("Standing uses the released default height")
        elif not 0.2 <= self.body_height_m <= 0.8:
            raise ValueError("Static postures require a height from 0.2 to 0.8 m")
        return self


SonicMotionPublication = SonicPlannerPublication
MOTION_PUBLICATION_ADAPTER = TypeAdapter(SonicMotionPublication)


def parse_motion_publication(payload: str | bytes) -> SonicMotionPublication:
    return MOTION_PUBLICATION_ADAPTER.validate_json(payload)


def motion_timeline(
    publication: SonicMotionPublication,
    observation_pelvis_quaternion_wxyz: Sequence[float],
    *,
    remaining_episode_seconds: float | None = None,
    hand_initials: HandTransitionInitials | None = None,
) -> SonicPlannerTimeline:
    timeline = SonicPlannerTimeline(
        publication,
        observation_pelvis_quaternion_wxyz,
        remaining_episode_seconds=remaining_episode_seconds,
    )
    if publication.left_grasp_schedule is not None or publication.right_grasp_schedule is not None:
        if hand_initials is None:
            raise ValueError("Timed grasp requires measured admission finger positions")
        timeline.hand_sampler = HandTransitionSampler(
            publication.left_grasp_schedule, publication.right_grasp_schedule, hand_initials
        )
    return timeline


class NativePlannerSample(ValidatedModel):
    """Arguments for upstream build_planner_message, with no packed observations."""

    locomotion_mode: LocomotionMode
    movement_direction_world: Vector3
    facing_direction_world: Vector3
    speed_m_per_s: FiniteFloat
    body_height_m: FiniteFloat
    left_hand_joints: HandVector | None = None
    right_hand_joints: HandVector | None = None
    vr_position_body_frame: PositionTargets
    vr_orientation_body_frame_wxyz: OrientationTargets
    endpoint_hold: bool
    retain_uncommanded_hand_targets: bool = False


def imu_rotation(quaternion_wxyz: Sequence[float]) -> Rotation:
    values = np.asarray(quaternion_wxyz, dtype=float)
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or not np.isclose(np.linalg.norm(values), 1, rtol=0, atol=0.001)
    ):
        raise ValueError("Controller IMU requires a finite unit wxyz quaternion")
    return Rotation.from_quat(values, scalar_first=True)


def heading_rotation(rotation: Rotation) -> Rotation:
    matrix = rotation.as_matrix()
    return Rotation.from_euler("z", math.atan2(matrix[1, 0], matrix[0, 0]))


# Released gamepad stick turning changes facing by at most 0.02 rad per 50 Hz tick;
# held within this lead of the measured heading, the planner never sees a large jump.
FACING_LEAD_RADIANS = math.radians(15.0)


class SonicPlannerTimeline:
    """Hold a finite model endpoint while native closed-loop control continues."""

    def __init__(
        self,
        publication: SonicPlannerPublication,
        observation_pelvis_quaternion_wxyz: Sequence[float],
        *,
        remaining_episode_seconds: float | None = None,
    ) -> None:
        self.publication = SonicPlannerPublication.model_validate(publication)
        self.hand_sampler: HandTransitionSampler | None = None
        self.duration_seconds = publication.control_ticks * 0.02
        self.execution_duration_seconds = self.duration_seconds
        if (
            publication.protocol != "sonic-native-planner-v1"
            and remaining_episode_seconds is not None
        ):
            if not math.isfinite(remaining_episode_seconds):
                raise ValueError("Remaining episode time must be finite")
            self.execution_duration_seconds = min(
                self.duration_seconds, max(0, remaining_episode_seconds)
            )
        self.keyframe_times = np.linspace(0, self.duration_seconds, 4)
        self.positions = np.asarray(
            [pose.positions_heading_frame for pose in publication.poses]
        ).reshape(4, 3, 3)
        orientations = np.asarray(
            [pose.orientations_heading_frame_wxyz for pose in publication.poses]
        ).reshape(4, 3, 4)
        self.orientation_interpolators = [
            Slerp(
                self.keyframe_times,
                Rotation.from_quat(orientations[:, body], scalar_first=True),
            )
            for body in range(3)
        ]
        observation_heading = heading_rotation(imu_rotation(observation_pelvis_quaternion_wxyz))
        self.movement_world = observation_heading.apply(
            publication.movement_direction_heading_frame
        ).tolist()
        self.facing_world = observation_heading.apply(
            [math.cos(publication.facing_yaw_radians), math.sin(publication.facing_yaw_radians), 0]
        ).tolist()
        self.stepped_turn: SteppedTurn | None = None
        self.facing_target_radians = math.atan2(self.facing_world[1], self.facing_world[0])

    def sample(
        self, elapsed_seconds: float, pelvis_quaternion_wxyz: Sequence[float]
    ) -> NativePlannerSample:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Command elapsed time must be finite and nonnegative")
        local_time = min(elapsed_seconds, self.execution_duration_seconds)
        endpoint_hold = elapsed_seconds >= self.execution_duration_seconds
        root_rotation = imu_rotation(pelvis_quaternion_wxyz)
        body_from_heading = cast(Rotation, root_rotation.inv() * heading_rotation(root_rotation))
        positions_heading = np.asarray(
            [
                [
                    np.interp(local_time, self.keyframe_times, self.positions[:, body, axis])
                    for axis in range(3)
                ]
                for body in range(3)
            ]
        )
        orientations_body = [
            cast(Rotation, body_from_heading * interpolator(local_time))
            .as_quat(scalar_first=True)
            .tolist()
            for interpolator in self.orientation_interpolators
        ]
        mode = self.publication.locomotion_mode
        if self.publication.leashed_facing:
            forward = heading_rotation(root_rotation).apply([1.0, 0.0, 0.0])
            heading = math.atan2(forward[1], forward[0])
            lead = math.remainder(self.facing_target_radians - heading, math.tau)
            commanded = heading + max(-FACING_LEAD_RADIANS, min(FACING_LEAD_RADIANS, lead))
            self.facing_world = [math.cos(commanded), math.sin(commanded), 0.0]
        if self.publication.turn_in_place_radians is not None and not endpoint_hold:
            forward = heading_rotation(root_rotation).apply([1.0, 0.0, 0.0])
            heading = math.atan2(forward[1], forward[0])
            if self.stepped_turn is None:
                self.stepped_turn = SteppedTurn(
                    heading, self.publication.turn_in_place_radians, elapsed_seconds
                )
            self.stepped_turn.update(elapsed_seconds, heading)
            self.facing_world = self.stepped_turn.facing_xy()
        left_hand, right_hand = (
            self.hand_sampler.sample(local_time)
            if self.hand_sampler is not None
            else (self.publication.left_hand_joints, self.publication.right_hand_joints)
        )
        return NativePlannerSample(
            locomotion_mode=0 if endpoint_hold and mode == 1 else mode,
            movement_direction_world=self.movement_world,
            facing_direction_world=self.facing_world,
            speed_m_per_s=0 if endpoint_hold else self.publication.speed_m_per_s,
            body_height_m=self.publication.body_height_m,
            vr_position_body_frame=body_from_heading.apply(positions_heading).reshape(-1).tolist(),
            vr_orientation_body_frame_wxyz=np.asarray(orientations_body).reshape(-1).tolist(),
            endpoint_hold=endpoint_hold,
            retain_uncommanded_hand_targets=self.publication.protocol == "sonic-native-planner-v3",
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )


def publish_native_planner_sample(
    publisher: Any, upstream_sender: Any, sample: NativePlannerSample
) -> None:
    """Delegate wire encoding to the pinned released Python message builder."""
    publisher.send(
        upstream_sender.build_planner_message(
            mode=sample.locomotion_mode,
            movement=list(sample.movement_direction_world),
            facing=list(sample.facing_direction_world),
            speed=sample.speed_m_per_s,
            height=sample.body_height_m,
            left_hand_position=list(sample.left_hand_joints)
            if sample.left_hand_joints is not None
            else None,
            right_hand_position=list(sample.right_hand_joints)
            if sample.right_hand_joints is not None
            else None,
            vr_3pt_position=list(sample.vr_position_body_frame),
            vr_3pt_orientation=list(sample.vr_orientation_body_frame_wxyz),
        )
    )


class ControllerObservation(ValidatedModel):
    """Private IMU basis bound to a published sensor observation; no translation."""

    identity: ObservationIdentity
    pelvis_quaternion_wxyz: UnitQuaternion


class SonicPlannerCommandStream:
    """Single physics-owner command admission and endpoint publication.

    An accepted command starts once. Publication refreshes the upstream input's
    timeout while leaving its planner lifecycle untouched. Episode termination
    and independent physical scoring remain the caller's responsibility.
    """

    def __init__(
        self,
        directory: Path,
        episode_id: str,
        *,
        record: Callable[[Path, BaseModel], None],
        restore_records: bool = True,
    ) -> None:
        self.record = record
        self.ledger = CommandAdmissionLedger(
            directory, episode_id, record=record, restore_records=restore_records
        )
        self.basis_directory = directory / "controller-observations"
        self.observations: dict[str, ControllerObservation] = {}
        for path in record_paths(self.basis_directory) if restore_records else ():
            observation = ControllerObservation.model_validate_json(read_record_bytes(path))
            if (
                observation.identity.identifier != path.stem
                or self.ledger.observations.get(path.stem) != observation.identity
            ):
                raise ValueError("Stored controller IMU basis differs from its ledger observation")
            self.observations[path.stem] = observation
        self.active_timeline: SonicPlannerTimeline | None = None
        self.active_started_elapsed_seconds: float | None = None

    def register_observation(self, observation: ControllerObservation) -> None:
        self.ledger.register_observation(observation.identity)
        path = self.basis_directory / f"{observation.identity.identifier}.json"
        if observation.identity.identifier in self.observations:
            previous = self.observations[observation.identity.identifier]
            if previous != observation:
                raise ValueError("An observation's controller IMU basis cannot be reassigned")
            return
        self.record(path, observation)
        self.observations[observation.identity.identifier] = observation

    def activate(
        self,
        publication: SonicMotionPublication,
        elapsed_seconds: float,
        *,
        hand_initials: HandTransitionInitials | None = None,
    ) -> CommandAdmission:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Command activation time must be finite and nonnegative")
        publication = MOTION_PUBLICATION_ADAPTER.validate_python(publication)
        if publication.identifier in self.ledger.admissions:
            return self.ledger.admit(publication, elapsed_seconds)
        if (
            self.active_timeline is not None
            and self.active_started_elapsed_seconds is not None
            and elapsed_seconds
            < self.active_started_elapsed_seconds + self.active_timeline.execution_duration_seconds
        ):
            raise ValueError("An active finite command cannot be interrupted by another command")
        if self.ledger.assess(publication, elapsed_seconds).reason != AdmissionReason.ACCEPTED:
            return self.ledger.admit(publication, elapsed_seconds)
        timeline = None
        if publication.observation_id in self.observations:
            observation = self.observations[publication.observation_id]
            timeline = motion_timeline(
                publication,
                observation.pelvis_quaternion_wxyz,
                remaining_episode_seconds=(
                    observation.identity.episode_deadline_elapsed_wall_seconds - elapsed_seconds
                ),
                hand_initials=hand_initials,
            )
        elif publication.observation_id in self.ledger.observations:
            raise RuntimeError("Registered observation lacks its controller IMU basis")
        admission = self.ledger.admit(publication, elapsed_seconds)
        if admission.reason != AdmissionReason.ACCEPTED:
            return admission
        if timeline is None:
            raise RuntimeError(
                "Admitted observation lacks its controller IMU basis; stop execution"
            )
        self.record(
            self.ledger.directory / "commands" / f"{publication.identifier}.json", publication
        )
        if timeline.hand_sampler is not None:
            hand_path = self.ledger.directory / "hand-initials" / f"{publication.identifier}.json"
            self.record(hand_path, timeline.hand_sampler.initials)
        self.active_timeline = timeline
        self.active_started_elapsed_seconds = elapsed_seconds
        return admission

    def sample(
        self, elapsed_seconds: float, pelvis_quaternion_wxyz: Sequence[float]
    ) -> NativePlannerSample | None:
        if self.active_timeline is None or self.active_started_elapsed_seconds is None:
            return None
        return self.active_timeline.sample(
            elapsed_seconds - self.active_started_elapsed_seconds, pelvis_quaternion_wxyz
        )

    def endpoint_reached(self, elapsed_seconds: float) -> bool:
        """Completion polling needs the deadline, not interpolated tracking poses."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Command elapsed time must be finite and nonnegative")
        if self.active_timeline is None or self.active_started_elapsed_seconds is None:
            return False
        return (
            elapsed_seconds - self.active_started_elapsed_seconds
            >= self.active_timeline.execution_duration_seconds
        )
