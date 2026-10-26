"""Derive the project page's shutdown-round result data from the frozen summary.

Each manipulation start is assigned exactly one end outcome. The counts are
checked against the summary's per-provider totals, so the page cannot drift
from the reported table.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Literal

from cyclopts import App

from humanoid_agent.analysis.summarize_api_costs import CostReport
from humanoid_agent.analysis.summarize_shutdown_round import group_label
from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.logging_config import configure_logging

EpisodeOutcome = Literal[
    "success",
    "declared_without_success",
    "fell",
    "startup_fall",
    "time_limit",
    "infrastructure_error",
]
EPISODE_OUTCOMES: tuple[EpisodeOutcome, ...] = (
    "success",
    "declared_without_success",
    "fell",
    "startup_fall",
    "time_limit",
    "infrastructure_error",
)


def episode_outcome(row: dict[str, Any]) -> EpisodeOutcome:
    match (row["execution_status"], row["end_reason"], row["task_success"]):
        case ("infrastructure_error", _, _):
            return "infrastructure_error"
        case ("completed", _, True):
            return "success"
        case ("completed", "agent_declared_completion", False):
            return "declared_without_success"
        case ("completed", "robot_fell", False):
            return "startup_fall" if row.get("startup_fall") else "fell"
        case ("completed", "wall_time_limit", False):
            return "time_limit"
        case unknown:
            raise ValueError(f"Unclassified shutdown episode ending: {unknown}")


def provider_results(provider: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict:
    manipulation_rows = [row for row in rows if row["requires_manipulation"]]
    outcomes = dict.fromkeys(EPISODE_OUTCOMES, 0)
    for row in manipulation_rows:
        outcomes[episode_outcome(row)] += 1
    manipulation = summary["manipulation"]
    control = summary["control"]
    expected: dict[EpisodeOutcome, int] = {
        "success": manipulation["task_success"],
        "fell": manipulation["falls"] - manipulation.get("startup_falls", 0),
        "startup_fall": manipulation.get("startup_falls", 0),
        "time_limit": manipulation["timeouts"],
        "infrastructure_error": manipulation["started"] - manipulation["completed"],
    }
    if (
        any(outcomes[outcome] != count for outcome, count in expected.items())
        or sum(outcomes.values()) != len(manipulation_rows)
        or len(manipulation_rows) != manipulation["started"]
    ):
        raise ValueError(f"Episode outcomes disagree with the summary totals for {provider}")
    request_seconds = [
        row["median_request_seconds"] for row in rows if row["median_request_seconds"] is not None
    ]
    return {
        "provider": provider,
        "label": group_label(provider),
        "manipulation": {
            "started": manipulation["started"],
            "completed": manipulation["completed"],
            "outcomes": outcomes,
            "physical_completion_observed": manipulation["physical_completion_observed"],
            "open_drawers": manipulation["open_drawers"],
            "open_drawers_closed": manipulation["open_drawers_closed"],
            "declared_without_physical_completion": manipulation[
                "declared_without_physical_completion"
            ],
            "successful_completion_seconds": sorted(manipulation["successful_completion_seconds"]),
            "all_drawers_viewed": manipulation.get("all_drawers_viewed", 0),
        },
        "control": {
            "started": control["started"],
            "completed": control["completed"],
            "task_success": control["task_success"],
        },
        "median_request_seconds": statistics.median(request_seconds),
    }


def main(
    summary_path: Path,
    /,
    *,
    output: Path,
    budget_seconds: float = 1800.0,
    attempt_metrics: Path | None = None,
) -> None:
    """Write OUTPUT (JSON) from the shutdown SUMMARY_PATH written by summarize_shutdown_round.

    BUDGET_SECONDS is the frozen schedule's episode budget.
    """
    summary = json.loads(summary_path.read_text())
    if summary["schema_version"] != "webolt-shutdown-summary-v1":
        raise ValueError("Unsupported shutdown summary schema")
    rows_by_provider: dict[str, list[dict[str, Any]]] = {}
    for row in summary["episodes"]:
        rows_by_provider.setdefault(row.get("group", row["provider"]), []).append(row)
    providers = [
        provider_results(provider, rows_by_provider[provider], groups)
        for provider, groups in summary["providers"].items()
    ]
    providers.sort(
        key=lambda result: (
            -result["manipulation"]["outcomes"]["success"] / result["manipulation"]["started"],
            result["label"],
        )
    )
    if attempt_metrics is not None:
        metrics = CostReport.model_validate_json(attempt_metrics.read_text())
        if metrics.summary_sha256 != hashlib.sha256(summary_path.read_bytes()).hexdigest():
            raise ValueError("Attempt metrics and outcomes use different source summaries")
        metrics_by_group = {group.group: group for group in metrics.groups}
        for provider in providers:
            timing = metrics_by_group[provider["provider"]].attempt_timing
            provider["attempt_timing"] = timing.model_dump(mode="json") if timing else None
    completed_lags = [
        row["maximum_physics_lag_seconds"]
        for row in summary["episodes"]
        if row["maximum_physics_lag_seconds"] is not None
    ]
    atomic_json(
        output,
        {
            "schema_version": "webolt-shutdown-results-v1",
            "source": {
                "summary": str(summary_path),
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "attempt_metrics_sha256": hashlib.sha256(attempt_metrics.read_bytes()).hexdigest()
                if attempt_metrics
                else None,
            },
            "episode_budget_seconds": budget_seconds,
            "outcomes": list(EPISODE_OUTCOMES),
            "providers": providers,
            "timing": {
                "completed_episodes": len(completed_lags),
                "timing_valid_episodes": sum(row["timing_valid"] for row in summary["episodes"]),
                "median_peak_physics_lag_seconds": statistics.median(completed_lags),
                "maximum_peak_physics_lag_seconds": max(completed_lags),
            },
            "limits": summary["limits"],
        },
    )


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
