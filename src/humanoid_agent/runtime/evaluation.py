"""Private physical drawer-shutdown measurements and versioned outcome criteria."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from humanoid_agent.shutdown_task import (
    MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S,
    MINIMUM_STANDING_ROOT_HEIGHT_M,
    MINIMUM_UPRIGHT_COMPONENT,
    CabinetIdentifier,
    ShutdownTask,
)
from humanoid_agent.task_balance import (
    GROUND_REACHING_BALANCE,
    forbidden_floor_contact,
)
from humanoid_agent.task_outcome import EpisodeEndReason


@dataclass(frozen=True)
class GroundTaskMeasurements:
    root_height_m: float
    root_distance_from_start_m: float
    horizontal_speed_m_per_s: float
    gravity_z: float
    pelvis_floor_force_n: float
    torso_floor_force_n: float
    root_vertical_speed_m_per_s: float = 0
    left_knee_angle_rad: float = 0
    right_knee_angle_rad: float = 0
    left_foot_floor_force_n: float = 0
    right_foot_floor_force_n: float = 0
    left_knee_floor_force_n: float = 0
    right_knee_floor_force_n: float = 0
    drawer_a_position_m: float = 0
    drawer_a_velocity_m_per_s: float = 0
    drawer_a_hand_normal_force_n: float = 0
    drawer_a_body_normal_force_n: float = 0
    drawer_a_front_in_view: float = 0
    drawer_b_position_m: float = 0
    drawer_b_velocity_m_per_s: float = 0
    drawer_b_hand_normal_force_n: float = 0
    drawer_b_body_normal_force_n: float = 0
    drawer_b_front_in_view: float = 0
    drawer_c_position_m: float = 0
    drawer_c_velocity_m_per_s: float = 0
    drawer_c_hand_normal_force_n: float = 0
    drawer_c_body_normal_force_n: float = 0
    drawer_c_front_in_view: float = 0
    drawer_d_position_m: float = 0
    drawer_d_velocity_m_per_s: float = 0
    drawer_d_hand_normal_force_n: float = 0
    drawer_d_body_normal_force_n: float = 0
    drawer_d_front_in_view: float = 0

    def drawer(self, identifier: str) -> tuple[float, float, float, float]:
        prefix = "drawer_" + identifier.lower()
        return (
            getattr(self, prefix + "_position_m"),
            getattr(self, prefix + "_velocity_m_per_s"),
            getattr(self, prefix + "_hand_normal_force_n"),
            getattr(self, prefix + "_body_normal_force_n"),
        )


ClosingContact = Literal["hand", "other_body", "none"]


@dataclass
class DrawerProgress:
    """Private per-drawer evidence; closure is attributed to the latest contact source."""

    initial_position_m: float | None = None
    final_position_m: float = 0
    maximum_position_m: float = 0
    closed_hold_ticks: int = 0
    hand_contact_ticks: int = 0
    other_body_contact_ticks: int = 0
    latest_contact: ClosingContact = "none"
    closing_displacement_m: dict[ClosingContact, float] = field(
        default_factory=lambda: {"hand": 0.0, "other_body": 0.0, "none": 0.0}
    )
    previous_position_m: float | None = None
    latest_velocity_m_per_s: float = 0
    viewed_ticks: int = 0
    first_viewed_tick: int | None = None
    latest_hand_normal_force_n: float = 0
    latest_other_body_normal_force_n: float = 0


class GroundTaskEvaluation:
    """Advance once per 20 ms of physics, independently of model completion."""

    def __init__(self, shutdown: ShutdownTask) -> None:
        self.shutdown = shutdown
        self.drawers = {cabinet.identifier: DrawerProgress() for cabinet in shutdown.cabinets}
        self.update_ticks = 0
        self.first_physical_completion_seconds: float | None = None
        self.activated = False
        self.physical_criterion_met = False
        self.fallen = False
        self.return_hold_ticks = 0

    def check_fall(self, measurements: GroundTaskMeasurements) -> bool:
        if not np.isfinite(list(vars(measurements).values())).all():
            raise ValueError("Task measurements must be finite")
        self.fallen = GROUND_REACHING_BALANCE.is_fallen(
            measurements.root_height_m, measurements.gravity_z
        ) or forbidden_floor_contact(
            measurements.pelvis_floor_force_n, measurements.torso_floor_force_n
        )
        return self.fallen

    def update(self, measurements: GroundTaskMeasurements) -> None:
        self.check_fall(measurements)
        self.update_ticks += 1
        task = self.shutdown
        for identifier, progress in self.drawers.items():
            position, velocity, hand_force, body_force = measurements.drawer(identifier)
            if progress.initial_position_m is None:
                progress.initial_position_m = position
            progress.final_position_m = position
            if getattr(measurements, "drawer_" + identifier.lower() + "_front_in_view") > 0:
                progress.viewed_ticks += 1
                if progress.first_viewed_tick is None:
                    progress.first_viewed_tick = self.update_ticks
            progress.latest_velocity_m_per_s = velocity
            progress.latest_hand_normal_force_n = hand_force
            progress.latest_other_body_normal_force_n = body_force
            progress.maximum_position_m = max(progress.maximum_position_m, position)
            hand_contact = hand_force >= task.hand_contact_threshold_n
            body_contact = body_force >= task.hand_contact_threshold_n
            progress.hand_contact_ticks += hand_contact
            progress.other_body_contact_ticks += body_contact
            # Simultaneous contact is not a clean hand push; coasting keeps the last source.
            if body_contact:
                progress.latest_contact = "other_body"
            elif hand_contact:
                progress.latest_contact = "hand"
            if progress.previous_position_m is not None:
                closing = progress.previous_position_m - position
                if closing > 0:
                    progress.closing_displacement_m[progress.latest_contact] += closing
            progress.previous_position_m = position
            settled = (
                position <= task.closed_tolerance_m and abs(velocity) <= task.settled_speed_m_per_s
            )
            progress.closed_hold_ticks = progress.closed_hold_ticks + 1 if settled else 0
        self.activated = all(
            progress.closed_hold_ticks * 0.02 >= task.hold_seconds
            for progress in self.drawers.values()
        )
        self.return_hold_ticks = self.return_hold_ticks + 1 if self.is_returned(measurements) else 0
        self.physical_criterion_met = self.return_hold_ticks * 0.02 >= task.hold_seconds
        if self.physical_criterion_met and self.first_physical_completion_seconds is None:
            self.first_physical_completion_seconds = self.update_ticks * 0.02

    def drawer_disturbed(self, identifier: CabinetIdentifier) -> bool:
        task = self.shutdown
        progress = self.drawers[identifier]
        if progress.initial_position_m is None:
            return False
        return (
            not task.cabinet(identifier).initially_open
            and progress.maximum_position_m - progress.initial_position_m
            > task.disturbance_tolerance_m
        )

    def is_returned(self, measurements: GroundTaskMeasurements) -> bool:
        return (
            self.activated
            and measurements.root_distance_from_start_m <= self.shutdown.home_radius_m
            and self.is_standing_still(measurements)
        )

    def is_standing_still(self, measurements: GroundTaskMeasurements) -> bool:
        """The supported upright hold that the return-home check requires."""
        return (
            not self.fallen
            and measurements.root_height_m >= MINIMUM_STANDING_ROOT_HEIGHT_M
            and measurements.gravity_z < -MINIMUM_UPRIGHT_COMPONENT
            and measurements.horizontal_speed_m_per_s <= MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
            and abs(measurements.root_vertical_speed_m_per_s) <= MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
            and min(measurements.left_foot_floor_force_n, measurements.right_foot_floor_force_n) > 0
            and max(measurements.left_knee_floor_force_n, measurements.right_knee_floor_force_n)
            == 0
            and max(measurements.left_knee_angle_rad, measurements.right_knee_angle_rad) <= 0.5
        )

    def terminal_reason(
        self,
        measurements: GroundTaskMeasurements,
        *,
        completion_requested: bool = False,
        deadline_reached: bool = False,
    ) -> EpisodeEndReason | None:
        self.check_fall(measurements)
        if not self.is_returned(measurements):
            self.return_hold_ticks = 0
            self.physical_criterion_met = False
        if self.fallen:
            return EpisodeEndReason.ROBOT_FELL
        if deadline_reached:
            return EpisodeEndReason.WALL_TIME_LIMIT
        if completion_requested:
            return EpisodeEndReason.AGENT_DECLARED_COMPLETION
        return None
