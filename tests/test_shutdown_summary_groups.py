"""Storage-round summaries group by model and effort and keep startup falls apart."""

import unittest

from humanoid_agent.analysis.summarize_shutdown_round import group_label, latex_table, summarize


def row(group: str, **values: object) -> dict[str, object]:
    provider, _, effort = group.partition("-")
    return {
        "provider": provider,
        "reasoning_effort": effort,
        "group": group,
        "condition": "open-A",
        "requires_manipulation": True,
        "execution_status": "completed",
        "timing_valid": True,
        "maximum_physics_lag_seconds": 0.05,
        "end_reason": "agent_declared_completion",
        "declared_completion": True,
        "physical_criterion_met": True,
        "task_success": True,
        "physical_completion_observed": True,
        "declaration_seconds": 900.0,
        "open_drawers": 1,
        "open_drawers_closed": 1,
        "open_drawers_closed_by_hand": 1,
        "closed_drawers_disturbed": 0,
        "drawers_viewed": 4,
        "drawer_count": 4,
        "startup_fall": False,
        **values,
    }


class ShutdownSummaryGroupTests(unittest.TestCase):
    def test_published_success_rate_includes_api_failed_starts(self) -> None:
        episode_rows = [
            row("gemini-low"),
            row("gemini-low"),
            row("gemini-low", task_success=False),
            row("gemini-low", task_success=False),
            *[
                row("gemini-low", execution_status="infrastructure_error", task_success=None)
                for _ in range(3)
            ],
        ]
        report = latex_table(summarize(episode_rows))
        self.assertIn("Gemini 4 Argon (low) & 2/7 &", report)

    def test_efforts_are_summarized_separately(self) -> None:
        summary = summarize([row("astra-low"), row("astra-high", task_success=False)])
        self.assertEqual(summary["astra-low"]["manipulation"]["task_success"], 1)
        self.assertEqual(summary["astra-high"]["manipulation"]["task_success"], 0)
        self.assertEqual(group_label("astra-high"), "GPT-6 Astra (high)")

    def test_startup_falls_and_full_inspection_are_counted(self) -> None:
        summary = summarize(
            [
                row("sol-high", end_reason="robot_fell", task_success=False, startup_fall=True),
                row("sol-high", end_reason="robot_fell", task_success=False, drawers_viewed=2),
                row("sol-high"),
            ]
        )["sol-high"]["manipulation"]
        self.assertEqual(summary["falls"], 2)
        self.assertEqual(summary["startup_falls"], 1)
        self.assertEqual(summary["all_drawers_viewed"], 2)


if __name__ == "__main__":
    unittest.main()
