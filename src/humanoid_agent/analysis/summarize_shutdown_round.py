"""Describe frozen shutdown-round cohorts, keeping controls and failure kinds separate.

Each cohort manifest covers one drawer condition (one scene hash). Every manifest
is first validated by the cohort analyzer; this summary then reads the verified
records. It reports all started episodes, and separately the timing-valid ones,
rather than discarding timing-invalid physical outcomes.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from cyclopts import App

from humanoid_agent.analysis.analyze_episode_cohort import analyze_episode_cohort
from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.experiment_records import (
    CohortManifest,
    MeasuredValidity,
    PhysicsResult,
    load_episode_record,
)
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.recording import read_record_bytes
from humanoid_agent.shutdown_task import ShutdownTask


def episode_row(directory: Path, provider: str, task: ShutdownTask) -> dict[str, Any]:
    """``task`` comes from the manifest's frozen scene, so failed starts keep their condition."""
    episode = load_episode_record(directory)
    outcome = episode.task_outcome
    shutdown = outcome.shutdown if outcome is not None else None
    validity = episode.simulation_validity
    result = (
        PhysicsResult.model_validate_json(read_record_bytes(directory / "physics/result.json"))
        if episode.status == "completed"
        else None
    )
    request_seconds = [
        json.loads(path.read_text())["model_request_wall_seconds"]
        for path in sorted((directory / "operator").glob("*/model-timing.json"))
    ]
    operator_configuration = directory / "operator/operator-configuration.json"
    reasoning_effort = (
        json.loads(operator_configuration.read_text()).get("reasoning_effort")
        if operator_configuration.exists()
        else None
    )
    # A fall before any model command executed is a startup failure, not model behaviour.
    executed_commands = list((directory / "operator").glob("*/command.json"))
    open_drawers = (
        [drawer for drawer in shutdown.drawers if drawer.initially_open] if shutdown else []
    )
    closed_drawers = (
        [drawer for drawer in shutdown.drawers if not drawer.initially_open] if shutdown else []
    )
    return {
        "directory": str(directory),
        "provider": provider,
        "reasoning_effort": reasoning_effort,
        "group": f"{provider}-{reasoning_effort}" if reasoning_effort else provider,
        "condition": task.condition,
        "requires_manipulation": task.requires_manipulation,
        "execution_status": episode.status,
        "timing_valid": isinstance(validity, MeasuredValidity) and validity.status == "valid",
        "maximum_physics_lag_seconds": validity.maximum_physics_lag_seconds
        if isinstance(validity, MeasuredValidity)
        else None,
        "end_reason": outcome.end_reason.value if outcome is not None else None,
        "declared_completion": outcome.agent_requested_completion if outcome else None,
        "physical_criterion_met": outcome.physical_criterion_met if outcome else None,
        "task_success": outcome.task_success if outcome else None,
        "physical_completion_observed": shutdown.first_physical_completion_seconds is not None
        if shutdown
        else None,
        "first_physical_completion_seconds": shutdown.first_physical_completion_seconds
        if shutdown
        else None,
        "declaration_seconds": shutdown.completion_declaration_elapsed_seconds
        if shutdown
        else None,
        "open_drawers": len(open_drawers),
        "open_drawers_closed": sum(drawer.closed_and_settled for drawer in open_drawers),
        "open_drawers_closed_by_hand": sum(drawer.closed_by_hand for drawer in open_drawers),
        "closed_drawers_disturbed": sum(drawer.disturbed for drawer in closed_drawers),
        "drawers_viewed": sum(bool(drawer.front_viewed) for drawer in shutdown.drawers)
        if shutdown
        else None,
        "drawer_count": len(shutdown.drawers) if shutdown else None,
        "startup_fall": outcome is not None
        and outcome.end_reason.value == "robot_fell"
        and not executed_commands,
        "other_body_contact_seconds": sum(
            drawer.other_body_contact_seconds for drawer in shutdown.drawers
        )
        if shutdown
        else None,
        "elapsed_wall_seconds": result.elapsed_wall_seconds if result else None,
        "model_requests": len(request_seconds),
        "median_request_seconds": statistics.median(request_seconds) if request_seconds else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def describe(selected: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [row for row in selected if row["execution_status"] == "completed"]
        return {
            "started": len(selected),
            "completed": len(completed),
            "timing_valid": sum(row["timing_valid"] for row in completed),
            "task_success": sum(bool(row["task_success"]) for row in completed),
            "task_success_timing_valid": sum(
                bool(row["task_success"]) and row["timing_valid"] for row in completed
            ),
            "physical_criterion_met_at_end": sum(
                bool(row["physical_criterion_met"]) for row in completed
            ),
            "physical_completion_observed": sum(
                bool(row["physical_completion_observed"]) for row in completed
            ),
            "declared_completion": sum(bool(row["declared_completion"]) for row in completed),
            "declared_without_physical_completion": sum(
                bool(row["declared_completion"]) and not row["physical_criterion_met"]
                for row in completed
            ),
            "open_drawers": sum(row["open_drawers"] for row in completed),
            "open_drawers_closed": sum(row["open_drawers_closed"] for row in completed),
            "open_drawers_closed_by_hand": sum(
                row["open_drawers_closed_by_hand"] for row in completed
            ),
            "closed_drawers_disturbed": sum(row["closed_drawers_disturbed"] for row in completed),
            "falls": sum(row["end_reason"] == "robot_fell" for row in completed),
            "startup_falls": sum(bool(row.get("startup_fall")) for row in completed),
            "all_drawers_viewed": sum(
                row.get("drawers_viewed") is not None
                and row.get("drawers_viewed") == row.get("drawer_count")
                for row in completed
            ),
            "timeouts": sum(row["end_reason"] == "wall_time_limit" for row in completed),
            "successful_completion_seconds": sorted(
                row["declaration_seconds"] for row in completed if row["task_success"]
            ),
            "maximum_physics_lag_seconds": sorted(
                row["maximum_physics_lag_seconds"]
                for row in completed
                if row["maximum_physics_lag_seconds"] is not None
            ),
        }

    by_provider: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"manipulation": [], "control": []}
    )
    for row in rows:
        by_provider[row.get("group", row["provider"])][
            "manipulation" if row["requires_manipulation"] else "control"
        ].append(row)
    return {
        provider: {kind: describe(selected) for kind, selected in groups.items()}
        for provider, groups in sorted(by_provider.items())
    }


PROVIDER_LABELS = {
    "sol": "GPT-6.1 Sol",
    "astra": "GPT-6 Astra",
    "opus": "Claude Opus 5.5",
    "gemini": "Gemini 4 Argon",
}


def group_label(group: str) -> str:
    """``provider`` or ``provider-effort`` as a display label."""
    provider, _, effort = group.partition("-")
    label = PROVIDER_LABELS.get(provider, provider)
    return f"{label} ({effort})" if effort else label


def latex_table(
    providers: dict[str, Any],
    *,
    api_costs: dict[str, tuple[Decimal, Decimal, int]] | None = None,
    successful_api_costs: dict[str, tuple[Decimal, Decimal] | None] | None = None,
    attempt_times: dict[str, tuple[float | None, float | None]] | None = None,
) -> str:
    """Use all starts for success rates and completed episodes for physical counts."""
    lines = [
        r"\begin{tabular}{lcccccccc}"
        if attempt_times is not None
        else (r"\begin{tabular}{lcccccc}" if api_costs is None else r"\begin{tabular}{lccccccc}"),
        r"\toprule",
        "Model & Success & Physical & Closed & Seen & False & Control"
        + (r" & Est. \$" if api_costs is not None else "")
        + (" & Episode (s)" if attempt_times is not None else "")
        + r" \\",
        r"\midrule",
    ]
    for provider, groups in providers.items():
        manipulation = groups.get("manipulation")
        control = groups.get("control")
        if manipulation is None:
            continue
        cost_suffix = ""
        if api_costs is not None:
            minimum, maximum, unreported = api_costs[provider]
            cost_text = f"{minimum:.2f}" if minimum == maximum else f"{minimum:.2f}--{maximum:.2f}"
            if successful_api_costs is not None:
                successful_cost = successful_api_costs[provider]
                if successful_cost is None:
                    cost_text += r" / --$^{*}$"
                else:
                    success_minimum, success_maximum = successful_cost
                    success_text = (
                        f"{success_minimum:.2f}"
                        if success_minimum == success_maximum
                        else f"{success_minimum:.2f}--{success_maximum:.2f}"
                    )
                    cost_text += f" / {success_text}"
            elif unreported:
                cost_text += r"$^{*}$"
            cost_suffix = f" & {cost_text}"
        label = group_label(provider)
        if attempt_times is not None:
            model, _, effort = provider.partition("-")
            label = f"{model.title()} ({effort})"
            all_seconds, successful_seconds = attempt_times[provider]
            all_text = "--" if all_seconds is None else f"{all_seconds:.0f}"
            successful_text = (
                r"--$^{*}$" if successful_seconds is None else f"{successful_seconds:.0f}"
            )
            cost_suffix += f" & {all_text} / {successful_text}"
        lines.append(
            f"{label} & "
            f"{manipulation['task_success']}/{manipulation['started']} & "
            f"{manipulation['physical_completion_observed']} & "
            f"{manipulation['open_drawers_closed']} & "
            f"{manipulation.get('all_drawers_viewed', 0)} & "
            f"{manipulation['declared_without_physical_completion']} & "
            + (f"{control['task_success']}/{control['started']}" if control else "--")
            + cost_suffix
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(manifests: list[Path], /, *, output: Path) -> None:
    """Validate each frozen manifest, then write rows and a per-provider summary."""
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for manifest_path in manifests:
        analyze_episode_cohort(manifest_path)
        manifest = CohortManifest.model_validate_json(manifest_path.read_text())
        for entry in manifest.episodes:
            directory = (manifest_path.parent / entry.directory).resolve()
            scene = GroundWorkcellScene.model_validate_json(
                (directory / "inputs/scene.json").read_text()
            )
            if (
                hashlib.sha256((directory / "inputs/scene.json").read_bytes()).hexdigest()
                != manifest.protocol.scene_sha256
            ):
                raise ValueError(
                    f"Episode scene differs from the frozen shutdown scene: {directory}"
                )
            rows.append(episode_row(directory, entry.provider, scene.shutdown))
    providers = summarize(rows)
    output.with_suffix(".tex").write_text(latex_table(providers))
    atomic_json(
        output,
        {
            "schema_version": "webolt-shutdown-summary-v1",
            "manifests": [str(path) for path in manifests],
            "episodes": rows,
            "providers": providers,
            "limits": "Counts describe the declared starts in the supplied frozen manifests; "
            "controls with no open drawer are reported separately; frames and decisions are not "
            "independent trials; timing-invalid episodes are retained and labelled.",
        },
    )


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
