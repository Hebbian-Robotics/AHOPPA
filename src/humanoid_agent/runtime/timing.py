"""Bounded diagnostic timing; no simulator actions or clock policy changes."""

from __future__ import annotations

import math
from array import array
from bisect import bisect_left
from collections import deque
from itertools import pairwise
from typing import Any

from pydantic import FiniteFloat

from humanoid_agent.validated_types import ValidatedModel

# Fixed logarithmic bins give percentile upper bounds with at most 5% bucket
# width above 1 microsecond. Overflow uses the observed maximum as its bound.
DURATION_BOUNDARIES = tuple(1e-6 * 1.05**index for index in range(380))


class DurationSummary:
    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = budget_seconds
        self.histogram = array("Q", [0]) * (len(DURATION_BOUNDARIES) + 1)
        self.count = 0
        self.total_seconds = 0.0
        self.maximum_seconds = 0.0
        self.over_budget_count = 0

    def observe(self, duration_seconds: float) -> None:
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("Timing duration must be finite and nonnegative")
        self.histogram[bisect_left(DURATION_BOUNDARIES, duration_seconds)] += 1
        self.count += 1
        self.total_seconds += duration_seconds
        self.maximum_seconds = max(self.maximum_seconds, duration_seconds)
        self.over_budget_count += duration_seconds > self.budget_seconds

    def percentile_upper_bound(self, fraction: float) -> float | None:
        if not self.count:
            return None
        remaining = math.ceil(self.count * fraction)
        for index, count in enumerate(self.histogram):
            remaining -= count
            if remaining <= 0:
                return (
                    min(DURATION_BOUNDARIES[index], self.maximum_seconds)
                    if index < len(DURATION_BOUNDARIES)
                    else self.maximum_seconds
                )
        raise RuntimeError("Timing histogram count is inconsistent")

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_seconds": self.total_seconds / self.count if self.count else None,
            "p95_seconds": self.percentile_upper_bound(0.95),
            "p99_seconds": self.percentile_upper_bound(0.99),
            "maximum_seconds": self.maximum_seconds if self.count else None,
            "over_budget_count": self.over_budget_count,
            "budget_seconds": self.budget_seconds,
            "percentile_method": "nearest-rank histogram upper bound; 5% bins above 1 us",
        }


INITIAL_TICK_RETENTION = 10


class EpisodeTiming:
    stage_names = (
        "sensors",
        "actuation",
        "physics_and_update",
        "tick_before_trace_and_sleep",
        "evaluation_and_trace",
        "tick_work_before_sleep",
    )

    def __init__(self, physics_period_seconds: float) -> None:
        self.stages = {name: DurationSummary(physics_period_seconds) for name in self.stage_names}
        self.physics_period_seconds = physics_period_seconds
        self.lag_spikes: deque[dict[str, Any]] = deque(maxlen=128)
        self.lag_spike_count = 0
        self.maximum_lag_tick: dict[str, Any] | None = None
        # The release tick and its successors, which the bounded spike buffer can evict.
        self.initial_ticks: list[dict[str, Any]] = []

    def observe_tick(
        self,
        durations: dict[str, float],
        *,
        elapsed_wall_seconds: float,
        simulation_seconds: float,
        lag_seconds: float,
        lag_increase_seconds: float,
        events: list[str],
        command_id: str | None,
    ) -> None:
        for name, summary in self.stages.items():
            summary.observe(durations[name])
        new_maximum_lag = (
            self.maximum_lag_tick is None or lag_seconds > self.maximum_lag_tick["lag_seconds"]
        )
        # Record spikes, not every tick spent recovering from one stall.
        is_lag_spike = (
            durations["tick_work_before_sleep"] > self.physics_period_seconds
            or lag_increase_seconds > self.physics_period_seconds
        )
        retain_initial = len(self.initial_ticks) < INITIAL_TICK_RETENTION
        if new_maximum_lag or is_lag_spike or retain_initial:
            tick_context = {
                "elapsed_wall_seconds": elapsed_wall_seconds,
                "simulation_seconds": simulation_seconds,
                "lag_seconds": lag_seconds,
                "lag_increase_seconds": lag_increase_seconds,
                "events": list(events),
                "command_id": command_id,
                "durations_seconds": dict(durations),
            }
            if retain_initial:
                self.initial_ticks.append(tick_context)
            if new_maximum_lag:
                self.maximum_lag_tick = tick_context
            if is_lag_spike:
                self.lag_spike_count += 1
                self.lag_spikes.append(tick_context)

    def summary(self) -> dict[str, Any]:
        stages = {name: summary.summary() for name, summary in self.stages.items()}
        return {
            "protocol": "bounded-episode-timing-v1",
            "all": {"ticks": self.stages["tick_work_before_sleep"].count, **stages},
            "free_phase": {"ticks": self.stages["tick_work_before_sleep"].count, **stages},
            "lag_spike_count": self.lag_spike_count,
            "retained_lag_spikes": list(self.lag_spikes),
            "maximum_lag_tick": self.maximum_lag_tick,
            "initial_ticks": self.initial_ticks,
            "lag_spike_retention": "last 128 ticks with work or lag increase above physics period",
            "scope": "Active physics ticks only; work includes trace output, excludes pacing sleep and timing aggregation; no explicit CUDA synchronization",
        }


class CommandTimingEvidence(ValidatedModel):
    """Private sidecar on the same host monotonic clock; never model input."""

    identifier: str
    episode_id: str
    observation_id: str
    source_response_sha256: str
    observation_received_monotonic_seconds: FiniteFloat
    request_started_monotonic_seconds: FiniteFloat
    response_received_monotonic_seconds: FiniteFloat
    publication_prepared_monotonic_seconds: FiniteFloat

    def execution_breakdown(
        self, *, observation_monotonic_seconds: float, execution_monotonic_seconds: float
    ) -> dict[str, Any]:
        boundaries = (
            observation_monotonic_seconds,
            self.observation_received_monotonic_seconds,
            self.request_started_monotonic_seconds,
            self.response_received_monotonic_seconds,
            self.publication_prepared_monotonic_seconds,
            execution_monotonic_seconds,
        )
        if any(not math.isfinite(value) for value in boundaries) or any(
            later < earlier for earlier, later in pairwise(boundaries)
        ):
            return {"status": "invalid_clock_order"}
        return {
            "status": "measured",
            "snapshot_to_operator_receive_seconds": boundaries[1] - boundaries[0],
            "operator_preparation_seconds": boundaries[2] - boundaries[1],
            "model_request_and_tools_seconds": boundaries[3] - boundaries[2],
            "response_to_publication_prepared_seconds": boundaries[4] - boundaries[3],
            "publication_prepared_to_first_planner_send_seconds": boundaries[5] - boundaries[4],
            "observation_age_at_first_planner_send_seconds": boundaries[5] - boundaries[0],
            "execution_boundary": "first native planner send; not motor receipt or physical response",
        }
