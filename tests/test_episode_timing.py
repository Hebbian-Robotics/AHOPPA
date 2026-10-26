"""Timing diagnostics retain slow work and distinguish observation-age components."""

import json
import unittest

from humanoid_agent.runtime.timing import CommandTimingEvidence, DurationSummary, EpisodeTiming


class EpisodeTimingTests(unittest.TestCase):
    def test_tail_bounds_and_budget_counts_include_rare_stalls(self) -> None:
        summary = DurationSummary(0.005)
        for duration in [0.001] * 98 + [0.01, 0.5]:
            summary.observe(duration)
        result = summary.summary()
        self.assertEqual(result["count"], 100)
        self.assertAlmostEqual(result["mean_seconds"], 0.00608)
        self.assertEqual(result["over_budget_count"], 2)
        self.assertEqual(result["maximum_seconds"], 0.5)
        self.assertGreaterEqual(result["p99_seconds"], 0.01)
        self.assertLessEqual(result["p99_seconds"], 0.0105)
        self.assertGreaterEqual(result["p95_seconds"], 0.001)
        self.assertLessEqual(result["p95_seconds"], 0.00105)
        json.dumps(result, allow_nan=False)

    def test_zero_empty_overflow_and_invalid_measurements(self) -> None:
        summary = DurationSummary(0.005)
        self.assertIsNone(summary.summary()["p99_seconds"])
        summary.observe(0)
        self.assertEqual(summary.summary()["p99_seconds"], 0)
        summary.observe(1000)
        self.assertEqual(summary.summary()["p99_seconds"], 1000)
        for invalid in [-0.1, float("nan"), float("inf")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summary.observe(invalid)
        self.assertEqual(summary.summary()["count"], 2)

    def test_recording_overruns_and_wakeup_lag_keep_bounded_event_context(self) -> None:
        timing = EpisodeTiming(0.005)
        durations = {
            "sensors": 0.0005,
            "actuation": 0.0001,
            "physics_and_update": 0.0014,
            "tick_before_trace_and_sleep": 0.002,
            "evaluation_and_trace": 0.006,
            "tick_work_before_sleep": 0.008,
        }
        for tick_index in range(200):
            timing.observe_tick(
                durations,
                elapsed_wall_seconds=tick_index * 0.008,
                simulation_seconds=tick_index * 0.005,
                lag_seconds=tick_index * 0.003,
                lag_increase_seconds=0.003,
                events=["trace_serialization_write_and_flush"],
                command_id="motion",
            )
        result = timing.summary()
        self.assertEqual(result["all"]["tick_work_before_sleep"]["over_budget_count"], 200)
        self.assertEqual(result["all"]["tick_before_trace_and_sleep"]["over_budget_count"], 0)
        self.assertEqual(result["lag_spike_count"], 200)
        self.assertEqual(len(result["retained_lag_spikes"]), 128)
        self.assertAlmostEqual(result["retained_lag_spikes"][0]["simulation_seconds"], 0.36)
        short_durations = dict.fromkeys(durations, 0.001)
        timing.observe_tick(
            short_durations,
            elapsed_wall_seconds=2,
            simulation_seconds=1,
            lag_seconds=1,
            lag_increase_seconds=0.02,
            events=["camera_snapshot_requested"],
            command_id=None,
        )
        result = timing.summary()
        self.assertEqual(result["lag_spike_count"], 201)
        self.assertEqual(result["retained_lag_spikes"][-1]["events"], ["camera_snapshot_requested"])
        for tick_index in range(129):
            timing.observe_tick(
                short_durations,
                elapsed_wall_seconds=3 + tick_index * 0.005,
                simulation_seconds=2 + tick_index * 0.005,
                lag_seconds=0.1,
                lag_increase_seconds=0.02,
                events=["later_wakeup"],
                command_id="later_motion",
            )
        result = timing.summary()
        self.assertEqual(result["maximum_lag_tick"]["elapsed_wall_seconds"], 2)
        self.assertEqual(result["maximum_lag_tick"]["lag_seconds"], 1)
        self.assertEqual(result["maximum_lag_tick"]["events"], ["camera_snapshot_requested"])
        self.assertEqual(result["retained_lag_spikes"][0]["events"], ["later_wakeup"])
        json.dumps(result, allow_nan=False)

    def test_observation_age_components_sum_to_first_send_age(self) -> None:
        evidence = CommandTimingEvidence(
            identifier="motion",
            episode_id="episode",
            observation_id="snapshot",
            source_response_sha256="a" * 64,
            observation_received_monotonic_seconds=102,
            request_started_monotonic_seconds=103,
            response_received_monotonic_seconds=110,
            publication_prepared_monotonic_seconds=111,
        )
        result = evidence.execution_breakdown(
            observation_monotonic_seconds=100, execution_monotonic_seconds=113
        )
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["snapshot_to_operator_receive_seconds"], 2)
        self.assertEqual(result["operator_preparation_seconds"], 1)
        self.assertEqual(result["model_request_and_tools_seconds"], 7)
        self.assertEqual(result["response_to_publication_prepared_seconds"], 1)
        self.assertEqual(result["publication_prepared_to_first_planner_send_seconds"], 2)
        self.assertEqual(result["observation_age_at_first_planner_send_seconds"], 13)
        self.assertEqual(
            evidence.execution_breakdown(
                observation_monotonic_seconds=104, execution_monotonic_seconds=113
            ),
            {"status": "invalid_clock_order"},
        )
