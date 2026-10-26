"""Validate every declared episode before describing a model-comparison cohort."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from cyclopts import App

from humanoid_agent.experiment_records import (
    CohortManifest,
    MeasuredValidity,
    load_episode_record,
    validate_frozen_inputs,
    validate_frozen_source,
    validate_physical_record,
)
from humanoid_agent.logging_config import configure_logging


def analyze_episode_cohort(manifest_path: Path) -> dict[str, object]:
    manifest = CohortManifest.model_validate_json(manifest_path.read_text())
    directories = [
        (manifest_path.parent / entry.directory).resolve(strict=True) for entry in manifest.episodes
    ]
    if len(set(directories)) != len(directories):
        raise ValueError("Each started episode must appear exactly once in the cohort")
    rows = []
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "started": 0,
            "infrastructure_error": 0,
            "cancelled": 0,
            "invalid_or_unknown_simulation": 0,
            "valid_completed": 0,
            "verified_success": 0,
        }
    )
    for assignment, directory in zip(manifest.episodes, directories, strict=True):
        episode = load_episode_record(directory)
        if (
            episode.protocol != manifest.protocol
            or episode.provider != assignment.provider
            or episode.episode_kind != "model"
        ):
            raise ValueError(f"Episode differs from frozen cohort assignment: {directory}")
        validate_frozen_source(directory, manifest.protocol.source_manifest_sha256)
        validate_frozen_inputs(directory, manifest.protocol)
        if episode.status == "completed":
            validate_physical_record(directory, episode)
        valid = (
            isinstance(episode.simulation_validity, MeasuredValidity)
            and episode.simulation_validity.status == "valid"
        )
        completed = episode.status == "completed" and valid
        success = bool(completed and episode.task_outcome and episode.task_outcome.task_success)
        condition = counts[assignment.condition]
        condition["started"] += 1
        condition["infrastructure_error"] += episode.status == "infrastructure_error"
        condition["cancelled"] += episode.status == "cancelled"
        condition["invalid_or_unknown_simulation"] += not valid
        condition["valid_completed"] += completed
        condition["verified_success"] += success
        rows.append(
            {
                "directory": str(directory),
                "condition": assignment.condition,
                "provider": assignment.provider,
                "execution_status": episode.status,
                "simulation_status": episode.simulation_validity.status,
                "verified_success": success,
            }
        )
    return {
        "schema_version": "webolt-cohort-analysis-v1",
        "episodes": rows,
        "conditions": dict(counts),
        "limits": "Every declared episode retained; infrastructure and simulation-validity counts can overlap. Only completed, timing-valid episodes count as verified successes. Counts are descriptive, not an inferential ranking. Manifest completeness must be established before launching the study.",
    }


def main(manifest: Path, /) -> None:
    print(json.dumps(analyze_episode_cohort(manifest), indent=2))


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
