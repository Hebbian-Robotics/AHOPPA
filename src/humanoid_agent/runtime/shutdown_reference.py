"""Privileged drawer-closing feasibility reference for the production controller check.

This source reads the frozen scene, measured robot root, measured hand tracking
points and the private drawer positions. It is not an agent and never writes
simulator state: every motion is a native SONIC planner sample, and drawers move
only through physical contact.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.runtime.evaluation import GroundTaskMeasurements
from humanoid_agent.shutdown_task import (
    MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S,
    SEKTION_BASE_HALF_HEIGHT_M,
    SEKTION_TOP_DRAWER_FRONT_Z_M,
    ShutdownCabinet,
    ShutdownTask,
)
from humanoid_agent.shutdown_visibility import PlanOccluder, segment_crosses_rectangle
from humanoid_agent.sonic_planner_input import NativePlannerSample, heading_rotation

ShutdownReferenceStage = Literal[
    "settle",
    "walk",
    "stop",
    "raise_hands",
    "push",
    "press",
    "settle_drawer",
    "ease_off",
    "lower_hands",
    "back_away",
    "verify",
    "walk_home",
    "stop_home",
    "done",
]

SHUTDOWN_REFERENCE_DURATION_SECONDS = 360.0
STAGE_TIMEOUT_SECONDS = 30.0
# Walks around storage blocks between wall cabinets take longer than other stages.
WALK_STAGE_TIMEOUT_SECONDS = 75.0
# Route corners keep the pelvis clear of tall obstacles despite stopping overshoot
# and the released planner's lateral sway of about 0.2 m.
OBSTACLE_CLEARANCE_M = 1.0
DETOUR_ARRIVAL_RADIUS_M = 0.3
WALKING_SPEED_M_PER_S = 0.3
BACKING_SPEED_M_PER_S = 0.2
# Pelvis-to-drawer-front standoffs along the cabinet facing direction.
APPROACH_STANDOFF_M = 0.42
MINIMUM_PUSH_STANDOFF_M = 0.26
BACK_AWAY_STANDOFF_M = 0.65
# Hand tracking points held ahead of the pelvis in the heading frame while pushing.
PUSH_HAND_FORWARD_M = 0.30
# Stop walking this far from closed and finish by extending the hands, so the
# drawer is not driven into its stop at walking speed.
PRESS_START_POSITION_M = 0.07
PRESS_EXTENSION_M = 0.12
PRESS_SECONDS = 2.5
PUSH_HAND_LATERAL_M = 0.18
HAND_RAMP_SECONDS = 3.0
# Withdraw the hands slightly from the closed front before lowering them.
EASE_OFF_DISTANCE_M = 0.06
EASE_OFF_SECONDS = 1.5
ARRIVAL_RADIUS_M = 0.08
# The released stop transition carried the robot 0.2-0.3 m past the point where
# a stop was requested; request it early, then correct a residual error.
STOPPING_LEAD_M = 0.25
HOME_CORRECTION_RADIUS_M = 0.2
MAXIMUM_HOME_CORRECTIONS = 3
# Walk facing the route to a staging point straight out from the drawer, then walk
# straight in: the released slow-walk planner did not sidestep short lateral offsets.
STAGING_STANDOFF_M = 1.0
STAGING_ARRIVAL_RADIUS_M = 0.15
CORRIDOR_HALF_WIDTH_M = 0.12
MAXIMUM_CLOSING_ATTEMPTS = 3
# Facing snaps, crab-walking and turning in place all preceded uncommanded
# 400-900 deg/s yaw spins and falls. On route legs the reference steers in arcs:
# the commanded facing turns toward the leg at a bounded rate and the robot walks
# where it faces. Only the final alignment at a drawer turns in place.
MAXIMUM_FACING_TURN_RATE_RADIANS_PER_SECOND = np.radians(30.0)
# The planner turned slower than any fixed rate tried; falls followed 22-61 degree
# command-to-heading errors. The command stays within this lead of the measured heading.
MAXIMUM_FACING_LEAD_RADIANS = np.radians(15.0)
# The released planner sways about 0.2 m sideways, so aiming at a nearby point
# swings the command; each leg instead keeps its bearing with a bounded
# cross-track correction, and a point counts as reached once passed along the leg.
CROSS_TRACK_GAIN_RADIANS_PER_M = 1.0
MAXIMUM_CROSS_TRACK_CORRECTION_RADIANS = np.radians(10.0)
# Arcs leave the leg line, so a waypoint counts as passed only near that line, and
# steering aims at the waypoint itself until this close, then holds the leg bearing.
LEG_PASS_CROSS_TRACK_M = 0.35
BEARING_HOLD_DISTANCE_M = 0.5


@dataclass(frozen=True)
class ShutdownVisit:
    cabinet: ShutdownCabinet

    @property
    def facing_xy(self) -> np.ndarray:
        return -np.asarray(self.cabinet.outward_direction_xy)


def inflated_corners(occluder: PlanOccluder, clearance: float) -> list[np.ndarray]:
    half = np.asarray(occluder.half_size_xy_m) + clearance
    center = np.asarray(occluder.center_xy_m)
    return [center + half * np.asarray(sign) for sign in ((1, 1), (1, -1), (-1, 1), (-1, -1))]


def plan_route(
    start: np.ndarray, goal: np.ndarray, occluders: list[PlanOccluder]
) -> list[np.ndarray]:
    """Shortest corner-to-corner path around inflated obstacles, ending at the goal."""
    blocked = [
        PlanOccluder(
            occluder.center_xy_m,
            (
                occluder.half_size_xy_m[0] + OBSTACLE_CLEARANCE_M - 0.05,
                occluder.half_size_xy_m[1] + OBSTACLE_CLEARANCE_M - 0.05,
            ),
            occluder.top_z_m,
        )
        for occluder in occluders
    ]
    nodes = [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]
    for occluder in occluders:
        nodes.extend(inflated_corners(occluder, OBSTACLE_CLEARANCE_M))

    def clear(first: np.ndarray, second: np.ndarray) -> bool:
        return not any(segment_crosses_rectangle(first, second, box) for box in blocked)

    distance = {0: 0.0}
    previous: dict[int, int] = {}
    pending = {0}
    done: set[int] = set()
    while pending:
        current = min(pending, key=lambda index: distance[index])
        pending.remove(current)
        done.add(current)
        if current == 1:
            break
        for index, node in enumerate(nodes):
            if index in done or not clear(nodes[current], node):
                continue
            candidate = distance[current] + float(np.linalg.norm(node - nodes[current]))
            if candidate < distance.get(index, float("inf")):
                distance[index] = candidate
                previous[index] = current
                pending.add(index)
    if 1 not in distance:
        raise ValueError("No obstacle-free route between the requested points")
    indices = [1]
    while indices[0] != 0:
        indices.insert(0, previous[indices[0]])
    return [nodes[index] for index in indices[1:]]


def route_length(start: np.ndarray, goal: np.ndarray, occluders: list[PlanOccluder]) -> float:
    points = [np.asarray(start, dtype=float), *plan_route(start, goal, occluders)]
    return sum(float(np.linalg.norm(b - a)) for a, b in itertools.pairwise(points))


def greedy_visit_order(
    visits: list[ShutdownVisit], home_xy_m: np.ndarray, occluders: list[PlanOccluder]
) -> list[ShutdownVisit]:
    """Visit the nearest remaining staging point by route length, starting from home."""
    remaining = list(visits)
    ordered = []
    position = np.asarray(home_xy_m, dtype=float)
    while remaining:
        staging = {
            id(visit): np.asarray(visit.cabinet.drawer_front_xy_m(0.0))
            - visit.facing_xy * STAGING_STANDOFF_M
            for visit in remaining
        }
        nearest = min(
            remaining, key=lambda visit: route_length(position, staging[id(visit)], occluders)
        )
        ordered.append(nearest)
        remaining.remove(nearest)
        position = staging[id(nearest)]
    return ordered


class ShutdownReferenceControl:
    """Measured-progress visits to every drawer, pushes open ones closed, returns home.

    A stage that does not progress within its timeout halts the reference and
    holds its last native sample; the failure is retained, never skipped.
    """

    def __init__(
        self,
        neutral: NativePlannerSample,
        task: ShutdownTask,
        home_xy_m: list[float],
        occluders: list[PlanOccluder] | None = None,
    ) -> None:
        self.neutral = neutral
        self.task = task
        self.home_xy_m = np.asarray(home_xy_m, dtype=float)
        self.occluders = occluders or []
        self.visits = greedy_visit_order(
            [ShutdownVisit(cabinet) for cabinet in task.cabinets], self.home_xy_m, self.occluders
        )
        self.detour_xy_m: list[np.ndarray] = []
        self.visit_index = 0
        self.current_stage: ShutdownReferenceStage = "settle"
        self.stage_started_seconds = 0.0
        self.condition_started_seconds: float | None = None
        self.failure_reason: str | None = None
        self.transitions: list[dict[str, float | str]] = []
        self.latest_measurement: GroundTaskMeasurements | None = None
        self.approach_xy_m: np.ndarray | None = None
        self.staging_xy_m: np.ndarray | None = None
        self.home_corrections = 0
        self.stopping_lead_m = STOPPING_LEAD_M
        self.hand_ramp_start_heading_m: np.ndarray | None = None
        self.minimum_push_standoff_m: float | None = None
        self.closing_attempts = {cabinet.identifier: 0 for cabinet in task.cabinets}
        self.reopened_after_release_m: dict[str, list[float]] = {
            cabinet.identifier: [] for cabinet in task.cabinets
        }
        self._last_sample = neutral
        self._commanded_facing_radians: float | None = None
        self._previous_sample_seconds: float | None = None
        self.facing_error_radians = 0.0
        self._leg_start_xy_m: np.ndarray | None = None
        self._leg_target_xy_m: np.ndarray | None = None
        self._neutral_hands_heading_m = (
            np.asarray(neutral.vr_position_body_frame).reshape(3, 3)[:2].copy()
        )

    @property
    def duration_seconds(self) -> float:
        return SHUTDOWN_REFERENCE_DURATION_SECONDS

    def phase(self, elapsed_seconds: float) -> str:
        del elapsed_seconds
        visit = (
            self.visits[self.visit_index].cabinet.identifier
            if self.visit_index < len(self.visits)
            else "home"
        )
        label = f"{self.current_stage}-{visit}"
        return f"halted-{label}-{self.failure_reason}" if self.failure_reason else label

    def drawer_position_m(self, measured: GroundTaskMeasurements) -> float:
        return measured.drawer(self.visits[self.visit_index].cabinet.identifier)[0]

    def drawer_velocity_m_per_s(self, measured: GroundTaskMeasurements) -> float:
        return measured.drawer(self.visits[self.visit_index].cabinet.identifier)[1]

    def front_standoff_m(self, root_xy: np.ndarray, measured: GroundTaskMeasurements) -> float:
        visit = self.visits[self.visit_index]
        front = np.asarray(visit.cabinet.drawer_front_xy_m(self.drawer_position_m(measured)))
        return float(np.dot(front - root_xy, visit.facing_xy))

    def advance(self, stage: ShutdownReferenceStage, elapsed_seconds: float) -> None:
        self.transitions.append(
            {"from": self.phase(elapsed_seconds), "to": stage, "elapsed_seconds": elapsed_seconds}
        )
        self.current_stage = stage
        self.stage_started_seconds = elapsed_seconds
        self.condition_started_seconds = None
        self.hand_ramp_start_heading_m = None

    def halt(self, reason: str) -> NativePlannerSample:
        """Stop locomotion and keep the reached upper-body targets."""
        self.failure_reason = reason
        self._last_sample = self._last_sample.model_copy(
            update={"locomotion_mode": 0, "speed_m_per_s": 0.0}
        )
        return self._last_sample

    def held(self, condition: bool, elapsed_seconds: float, hold_seconds: float) -> bool:
        if not condition:
            self.condition_started_seconds = None
            return False
        if self.condition_started_seconds is None:
            self.condition_started_seconds = elapsed_seconds
        return elapsed_seconds - self.condition_started_seconds >= hold_seconds

    def sample(
        self,
        elapsed_seconds: float,
        root_state_wxyz: list[float],
        hand_points_m: list[list[float]],
    ) -> NativePlannerSample:
        measured = self.latest_measurement
        if measured is None or self.failure_reason is not None or self.current_stage == "done":
            return self._last_sample
        root = np.asarray(root_state_wxyz)
        root_xy = root[:2]
        root_rotation = Rotation.from_quat(root[3:7], scalar_first=True)
        heading = heading_rotation(root_rotation)
        body_from_heading = cast(Rotation, root_rotation.inv() * heading)
        hands_heading = heading.inv().apply(np.asarray(hand_points_m) - root[:3])
        stage_elapsed = elapsed_seconds - self.stage_started_seconds
        timeout = (
            WALK_STAGE_TIMEOUT_SECONDS
            if self.current_stage in {"walk", "walk_home"}
            else STAGE_TIMEOUT_SECONDS
        )
        if stage_elapsed >= timeout:
            return self.halt("progress-timeout")
        stable = (
            measured.root_height_m >= 0.70
            and measured.gravity_z < -0.90
            and measured.horizontal_speed_m_per_s <= MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
            and abs(measured.root_vertical_speed_m_per_s) <= MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
        )
        visiting = self.visit_index < len(self.visits)
        facing_xy = (
            self.visits[self.visit_index].facing_xy
            if visiting and self.current_stage != "settle"
            else np.asarray([1.0, 0.0])
        )
        if self._commanded_facing_radians is None:
            initial = heading.apply([1.0, 0.0, 0.0])
            self._commanded_facing_radians = float(np.arctan2(initial[1], initial[0]))
        match self.current_stage:
            case "settle":
                if self.held(stable, elapsed_seconds, 2.0):
                    self.advance("walk", elapsed_seconds)
            case "walk" | "walk_home":
                if self.approach_xy_m is None:
                    if visiting:
                        front = np.asarray(
                            self.visits[self.visit_index].cabinet.drawer_front_xy_m(
                                self.drawer_position_m(measured)
                            )
                        )
                        self.approach_xy_m = front - facing_xy * APPROACH_STANDOFF_M
                        offset = root_xy - self.approach_xy_m
                        lateral = abs(float(offset[0] * facing_xy[1] - offset[1] * facing_xy[0]))
                        behind = -float(np.dot(offset, facing_xy))
                        # Already in the straight corridor in front of this drawer: walk in.
                        self.staging_xy_m = (
                            None
                            if lateral <= CORRIDOR_HALF_WIDTH_M
                            and 0 <= behind <= STAGING_STANDOFF_M
                            else front - facing_xy * STAGING_STANDOFF_M
                        )
                    else:
                        self.approach_xy_m = self.home_xy_m.copy()
                        self.staging_xy_m = None
                    first_target = (
                        self.staging_xy_m if self.staging_xy_m is not None else self.approach_xy_m
                    )
                    self.detour_xy_m = plan_route(root_xy, first_target, self.occluders)[:-1]
                    # Short corrections stop proportionally earlier than full walks.
                    self.stopping_lead_m = min(
                        STOPPING_LEAD_M,
                        0.5 * float(np.linalg.norm(self.approach_xy_m - root_xy)),
                    )
                if (
                    self.detour_xy_m
                    and self.leg_remaining_m(root_xy, self.detour_xy_m[0])
                    <= DETOUR_ARRIVAL_RADIUS_M
                ):
                    self.detour_xy_m.pop(0)
                if (
                    not self.detour_xy_m
                    and self.staging_xy_m is not None
                    and self.leg_remaining_m(root_xy, self.staging_xy_m) <= STAGING_ARRIVAL_RADIUS_M
                ):
                    self.staging_xy_m = None
                if self.staging_xy_m is None and (
                    self.leg_remaining_m(root_xy, self.approach_xy_m)
                    <= max(ARRIVAL_RADIUS_M, self.stopping_lead_m)
                ):
                    self.advance("stop" if visiting else "stop_home", elapsed_seconds)
            case "stop":
                if self.held(stable, elapsed_seconds, 1.0):
                    needs_push = self.drawer_position_m(measured) > self.task.closed_tolerance_m
                    if needs_push:
                        self.closing_attempts[self.visits[self.visit_index].cabinet.identifier] += 1
                    self.advance("raise_hands" if needs_push else "verify", elapsed_seconds)
            case "raise_hands":
                if stage_elapsed >= HAND_RAMP_SECONDS + 1.0 and stable:
                    self.advance("push", elapsed_seconds)
            case "push":
                standoff = self.front_standoff_m(root_xy, measured)
                if standoff > APPROACH_STANDOFF_M + 0.25:
                    # Not at the drawer: re-approach instead of walking with raised hands.
                    self.approach_xy_m = None
                    self.advance("walk", elapsed_seconds)
                    return self.native_sample(
                        elapsed_seconds, root, root_xy, facing_xy, hands_heading, body_from_heading
                    )
                self.minimum_push_standoff_m = (
                    standoff
                    if self.minimum_push_standoff_m is None
                    else min(self.minimum_push_standoff_m, standoff)
                )
                if self.drawer_position_m(measured) <= PRESS_START_POSITION_M:
                    self.advance("press", elapsed_seconds)
                elif standoff <= MINIMUM_PUSH_STANDOFF_M:
                    return self.halt("reach-limit-before-closure")
            case "press":
                if self.drawer_position_m(measured) <= self.task.closed_tolerance_m and (
                    stage_elapsed >= PRESS_SECONDS
                ):
                    self.advance("settle_drawer", elapsed_seconds)
                elif stage_elapsed >= PRESS_SECONDS + 1.0:
                    self.advance("push", elapsed_seconds)
            case "settle_drawer":
                settled = (
                    self.drawer_position_m(measured) <= self.task.closed_tolerance_m
                    and abs(self.drawer_velocity_m_per_s(measured))
                    <= self.task.settled_speed_m_per_s
                )
                if self.held(settled and stable, elapsed_seconds, 1.0):
                    self.advance("ease_off", elapsed_seconds)
            case "ease_off":
                if stage_elapsed >= EASE_OFF_SECONDS + 0.5:
                    self.advance("lower_hands", elapsed_seconds)
            case "back_away":
                # Unused: the released planner kept drifting backward after reverse walking.
                self.advance("verify", elapsed_seconds)
            case "lower_hands":
                if stage_elapsed >= HAND_RAMP_SECONDS and stable:
                    self.advance("verify", elapsed_seconds)
            case "verify":
                if self.held(stable, elapsed_seconds, 2.0):
                    identifier = self.visits[self.visit_index].cabinet.identifier
                    position = self.drawer_position_m(measured)
                    self.approach_xy_m = None
                    if position > self.task.closed_tolerance_m:
                        self.reopened_after_release_m[identifier].append(position)
                        if self.closing_attempts[identifier] >= MAXIMUM_CLOSING_ATTEMPTS:
                            return self.halt("drawer-reopened-after-release")
                        self.advance("walk", elapsed_seconds)
                    else:
                        self.visit_index += 1
                        self.advance(
                            "walk" if self.visit_index < len(self.visits) else "walk_home",
                            elapsed_seconds,
                        )
            case "stop_home":
                if self.held(stable, elapsed_seconds, 2.0):
                    error = float(np.linalg.norm(self.home_xy_m - root_xy))
                    if error <= HOME_CORRECTION_RADIUS_M:
                        self.advance("done", elapsed_seconds)
                    elif self.home_corrections >= MAXIMUM_HOME_CORRECTIONS:
                        return self.halt("home-not-reached")
                    else:
                        self.home_corrections += 1
                        self.approach_xy_m = None
                        self.advance("walk_home", elapsed_seconds)
            case "done":
                pass
        return self.native_sample(
            elapsed_seconds, root, root_xy, facing_xy, hands_heading, body_from_heading
        )

    def native_sample(
        self,
        elapsed_seconds: float,
        root: np.ndarray,
        root_xy: np.ndarray,
        facing_xy: np.ndarray,
        hands_heading: np.ndarray,
        body_from_heading: Rotation,
    ) -> NativePlannerSample:
        stage = self.current_stage
        stage_elapsed = elapsed_seconds - self.stage_started_seconds
        positions = np.asarray(self.neutral.vr_position_body_frame).reshape(3, 3).copy()
        push_hands = np.asarray(
            [
                [PUSH_HAND_FORWARD_M, PUSH_HAND_LATERAL_M, 0.0],
                [PUSH_HAND_FORWARD_M, -PUSH_HAND_LATERAL_M, 0.0],
            ]
        )
        push_hands[:, 2] = (
            sum(SEKTION_TOP_DRAWER_FRONT_Z_M) / 2 + SEKTION_BASE_HALF_HEIGHT_M - root[2]
        )
        if stage in {"raise_hands", "lower_hands"}:
            if self.hand_ramp_start_heading_m is None:
                self.hand_ramp_start_heading_m = hands_heading.copy()
            fraction = min(1.0, stage_elapsed / HAND_RAMP_SECONDS)
            blend = fraction * fraction * (3 - 2 * fraction)
            goal = push_hands if stage == "raise_hands" else self._neutral_hands_heading_m
            positions[:2] = self.hand_ramp_start_heading_m * (1 - blend) + goal * blend
        elif stage == "push":
            positions[:2] = push_hands
        elif stage in {"press", "settle_drawer"}:
            fraction = 1.0 if stage == "settle_drawer" else min(1.0, stage_elapsed / PRESS_SECONDS)
            pressed = push_hands.copy()
            pressed[:, 0] += PRESS_EXTENSION_M * fraction * fraction * (3 - 2 * fraction)
            positions[:2] = pressed
        elif stage == "back_away":
            if self.hand_ramp_start_heading_m is None:
                self.hand_ramp_start_heading_m = hands_heading.copy()
            positions[:2] = self.hand_ramp_start_heading_m
        elif stage == "ease_off":
            fraction = min(1.0, stage_elapsed / EASE_OFF_SECONDS)
            eased = push_hands.copy()
            eased[:, 0] += PRESS_EXTENSION_M - EASE_OFF_DISTANCE_M * fraction * fraction * (
                3 - 2 * fraction
            )
            positions[:2] = eased
        walking_direction: np.ndarray | None = None
        steering_in_arcs = False
        speed = WALKING_SPEED_M_PER_S
        if stage in {"walk", "walk_home"} and self.approach_xy_m is not None:
            target = (
                self.detour_xy_m[0]
                if self.detour_xy_m
                else self.staging_xy_m
                if self.staging_xy_m is not None
                else self.approach_xy_m
            )
            leg_direction = self.leg_direction(root_xy, target)
            walking_direction = self.corrected_walking_direction(root_xy, leg_direction)
            # Steer along the route, but do not turn around for a short final correction.
            steering_in_arcs = stage == "walk" or float(np.linalg.norm(target - root_xy)) > 0.6
            if steering_in_arcs:
                offset = target - root_xy
                distance = float(np.linalg.norm(offset))
                facing_xy = (
                    offset / distance if distance > BEARING_HOLD_DISTANCE_M else walking_direction
                )
        elif stage == "push":
            walking_direction = facing_xy.copy()
        elif stage == "back_away":
            walking_direction = -facing_xy
            speed = BACKING_SPEED_M_PER_S
        measured_heading = heading_rotation(Rotation.from_quat(root[3:7], scalar_first=True)).apply(
            [1.0, 0.0, 0.0]
        )
        facing_xy = self.slew_facing(
            elapsed_seconds,
            facing_xy,
            float(np.arctan2(measured_heading[1], measured_heading[0])),
        )
        if steering_in_arcs:
            walking_direction = facing_xy.copy()
        walking = walking_direction is not None
        if walking_direction is not None:
            walking_direction = walking_direction / max(
                float(np.linalg.norm(walking_direction)), 1e-9
            )
        orientations = Rotation.from_quat(
            np.asarray(self.neutral.vr_orientation_body_frame_wxyz).reshape(3, 4), scalar_first=True
        )
        facing_world = [float(facing_xy[0]), float(facing_xy[1]), 0.0]
        self._last_sample = self.neutral.model_copy(
            update={
                "locomotion_mode": 1 if walking else 0,
                "speed_m_per_s": speed if walking else 0.0,
                "movement_direction_world": [*walking_direction.tolist(), 0.0]
                if walking_direction is not None
                else self.neutral.movement_direction_world,
                "facing_direction_world": facing_world,
                "vr_position_body_frame": body_from_heading.apply(positions).reshape(-1).tolist(),
                "vr_orientation_body_frame_wxyz": cast(Rotation, body_from_heading * orientations)
                .as_quat(scalar_first=True)
                .reshape(-1)
                .tolist(),
                "left_hand_joints": [0.0] * 7,
                "right_hand_joints": [0.0] * 7,
                "endpoint_hold": False,
            }
        )
        return self._last_sample

    def leg_direction(self, root_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        """Bearing of the leg toward a waypoint, fixed when the waypoint is first targeted."""
        if self._leg_target_xy_m is None or not np.array_equal(self._leg_target_xy_m, target_xy):
            self._leg_start_xy_m = root_xy.copy()
            self._leg_target_xy_m = np.asarray(target_xy, dtype=float).copy()
        assert self._leg_start_xy_m is not None
        leg = self._leg_target_xy_m - self._leg_start_xy_m
        if float(np.linalg.norm(leg)) < 1e-6:
            leg = self._leg_target_xy_m - root_xy
        return leg / max(float(np.linalg.norm(leg)), 1e-9)

    def leg_remaining_m(self, root_xy: np.ndarray, target_xy: np.ndarray) -> float:
        """Distance still to walk along the leg near its line, else straight-line distance."""
        direction = self.leg_direction(root_xy, target_xy)
        offset = target_xy - root_xy
        cross_track = abs(float(offset[0] * direction[1] - offset[1] * direction[0]))
        if cross_track <= LEG_PASS_CROSS_TRACK_M:
            return float(np.dot(offset, direction))
        return float(np.linalg.norm(offset))

    def corrected_walking_direction(
        self, root_xy: np.ndarray, leg_direction: np.ndarray
    ) -> np.ndarray:
        """Leg bearing turned back toward the leg line by a bounded cross-track correction."""
        assert self._leg_start_xy_m is not None
        offset = root_xy - self._leg_start_xy_m
        cross_track = float(leg_direction[0] * offset[1] - leg_direction[1] * offset[0])
        correction = float(
            np.clip(
                -CROSS_TRACK_GAIN_RADIANS_PER_M * cross_track,
                -MAXIMUM_CROSS_TRACK_CORRECTION_RADIANS,
                MAXIMUM_CROSS_TRACK_CORRECTION_RADIANS,
            )
        )
        cosine, sine = np.cos(correction), np.sin(correction)
        return np.asarray(
            [
                cosine * leg_direction[0] - sine * leg_direction[1],
                sine * leg_direction[0] + cosine * leg_direction[1],
            ]
        )

    def slew_facing(
        self, elapsed_seconds: float, target_xy: np.ndarray, measured_heading_radians: float
    ) -> np.ndarray:
        """Turn the commanded facing toward the target at the bounded rate, leashed to
        the measured heading."""
        assert self._commanded_facing_radians is not None
        interval = (
            0.0
            if self._previous_sample_seconds is None
            else max(0.0, elapsed_seconds - self._previous_sample_seconds)
        )
        self._previous_sample_seconds = elapsed_seconds
        target = float(np.arctan2(target_xy[1], target_xy[0]))
        error = (target - self._commanded_facing_radians + np.pi) % (2 * np.pi) - np.pi
        step = MAXIMUM_FACING_TURN_RATE_RADIANS_PER_SECOND * interval
        self._commanded_facing_radians += float(np.clip(error, -step, step))
        lead = (self._commanded_facing_radians - measured_heading_radians + np.pi) % (
            2 * np.pi
        ) - np.pi
        self._commanded_facing_radians = measured_heading_radians + float(
            np.clip(lead, -MAXIMUM_FACING_LEAD_RADIANS, MAXIMUM_FACING_LEAD_RADIANS)
        )
        remaining = (target - self._commanded_facing_radians + np.pi) % (2 * np.pi) - np.pi
        self.facing_error_radians = abs(remaining)
        return np.asarray(
            [np.cos(self._commanded_facing_radians), np.sin(self._commanded_facing_radians)]
        )

    def summary(self) -> dict[str, object]:
        return {
            "phase": self.phase(0.0),
            "failure_reason": self.failure_reason,
            "visits_completed": self.visit_index,
            "minimum_push_standoff_m": self.minimum_push_standoff_m,
            "closing_attempts": self.closing_attempts,
            "home_corrections": self.home_corrections,
            "reopened_after_release_m": self.reopened_after_release_m,
            "transitions": self.transitions,
            "model_requests": 0,
        }
