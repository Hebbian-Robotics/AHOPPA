"""Comparisons reject incompatible evidence and retain unsuccessful started attempts."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from humanoid_agent.analysis.analyze_episode_cohort import analyze_episode_cohort
from humanoid_agent.artifact_io import atomic_json, canonical_json_sha256
from humanoid_agent.experiment_records import (
    EpisodeIORecord,
    EpisodeProtocol,
    EpisodeRecord,
    validate_frozen_inputs,
)
from humanoid_agent.recording import EpisodeRecorder, recording_sha256
from humanoid_agent.runtime.episode import write_terminal_episode
from humanoid_agent.runtime.postprocess import analyze_episode
from tests.recording_fixtures import shutdown_task_outcome_record


class EpisodeRecordTests(unittest.TestCase):
    def test_truncated_recording_still_produces_an_infrastructure_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare_cohort(root)
            episode = root / "success"
            original = EpisodeRecord.model_validate_json((episode / "episode.json").read_text())
            (episode / "episode.json").unlink()
            recorder = EpisodeRecorder(episode / "physics")
            recorder.write("result.json", {"status": "completed"})
            recorder.close()
            recorder.path.write_bytes(recorder.path.read_bytes()[:-12])
            terminal = write_terminal_episode(
                episode,
                provider=original.provider,
                episode_kind=original.episode_kind,
                protocol=original.protocol,
                status="infrastructure_error",
                error="Writer crashed",
                cleanup_errors=[],
                component_exit_codes={"physics": 1},
            )
            self.assertEqual(terminal.status, "infrastructure_error")
            self.assertTrue(terminal.cleanup_errors)
            self.assertEqual(
                EpisodeRecord.model_validate_json((episode / "episode.json").read_text()), terminal
            )

    def test_terminal_writer_preserves_incomplete_or_conflicting_completion_as_failure(
        self,
    ) -> None:
        for mutation in (
            "none",
            "outcome",
            "result",
            "recording",
            "conflicting_result",
            "corrupt_outcome",
            "corrupt_result",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.prepare_cohort(root)
                episode = root / "success"
                record_path = episode / "episode.json"
                original = EpisodeRecord.model_validate_json(record_path.read_text())
                record_path.unlink()
                if mutation in {"outcome", "result", "recording"}:
                    filename = {
                        "outcome": "task-outcome.json",
                        "result": "result.json",
                        "recording": "io-summary.json",
                    }[mutation]
                    (episode / "physics" / filename).unlink()
                elif mutation == "conflicting_result":
                    result_path = episode / "physics/result.json"
                    result = json.loads(result_path.read_text())
                    result["task_success"] = False
                    atomic_json(result_path, result)
                elif mutation == "corrupt_outcome":
                    (episode / "physics/task-outcome.json").write_text("malformed outcome")
                elif mutation == "corrupt_result":
                    (episode / "physics/result.json").write_text("malformed result")
                terminal = write_terminal_episode(
                    episode,
                    provider=original.provider,
                    episode_kind=original.episode_kind,
                    protocol=original.protocol,
                    status="completed",
                    error=None,
                    cleanup_errors=[],
                    component_exit_codes=dict[str, int | None](
                        original.component_exit_codes.items()
                    ),
                )
                persisted = EpisodeRecord.model_validate_json(record_path.read_text())
                self.assertEqual(persisted, terminal)
                if mutation == "none":
                    self.assertEqual(terminal.status, "completed")
                else:
                    self.assertEqual(terminal.status, "infrastructure_error")
                    self.assertTrue(terminal.cleanup_errors)
                    if mutation not in {"outcome", "corrupt_outcome"}:
                        self.assertEqual(terminal.task_outcome, original.task_outcome)
                    if mutation == "corrupt_outcome":
                        self.assertEqual(terminal.simulation_validity, original.simulation_validity)

    def test_frozen_appearance_rejects_malformed_file_maps_and_escaping_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare_cohort(root)
            episode = root / "success"
            record = EpisodeRecord.model_validate_json((episode / "episode.json").read_text())
            assert record.protocol is not None
            appearance = episode / "inputs/appearance"
            appearance.mkdir()
            asset = appearance / "visual.usda"
            asset.write_bytes(b"appearance")
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            outside = root / "outside.usda"
            outside.write_bytes(asset.read_bytes())
            manifest_path = appearance / "manifest.json"
            for files in (
                {"visual.usda": digest},
                [],
                {"visual.usda": "invalid digest"},
                {"../../../outside.usda": digest},
            ):
                with self.subTest(files=files):
                    atomic_json(manifest_path, {"files": files, "mesh_count": 1})
                    protocol = record.protocol.model_copy(
                        update={
                            "appearance_manifest_sha256": hashlib.sha256(
                                manifest_path.read_bytes()
                            ).hexdigest()
                        }
                    )
                    if files == {"visual.usda": digest}:
                        validate_frozen_inputs(episode, protocol)
                    else:
                        with self.assertRaises(ValueError):
                            validate_frozen_inputs(episode, protocol)

    def test_completed_record_requires_episode_kind_and_model_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare_cohort(root)
            record = json.loads((root / "success/episode.json").read_text())
            for changes in ({"episode_kind": None}, {"provider": None}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    EpisodeRecord.model_validate({**record, **changes})

    def test_new_io_completion_requires_drained_captures_and_preserves_legacy_records(self) -> None:
        record = {
            "schema_version": "webolt-episode-io-v2",
            "status": "completed",
            "error": None,
            "operations_processed": 4,
            "operations_submitted": 4,
            "interrupted_capture": None,
            "queued_captures_at_shutdown": 0,
        }
        EpisodeIORecord.model_validate(record)
        for changes in ({"interrupted_capture": "pending"}, {"queued_captures_at_shutdown": 1}):
            with self.assertRaisesRegex(ValueError, "accepted operations"):
                EpisodeIORecord.model_validate({**record, **changes})
        legacy = {
            **record,
            "schema_version": "webolt-episode-io-v1",
            "interrupted_capture": "historical",
        }
        del legacy["queued_captures_at_shutdown"]
        self.assertEqual(EpisodeIORecord.model_validate(legacy).queued_captures_at_shutdown, 0)

    def prepare_cohort(self, root: Path) -> Path:
        sources = {"runner.py": hashlib.sha256(b"saved source").hexdigest()}
        protocol = EpisodeProtocol(
            source_manifest_sha256=canonical_json_sha256(sources),
            scene_sha256=hashlib.sha256(b"scene").hexdigest(),
            appearance_manifest_sha256=None,
            task_instruction_sha256=hashlib.sha256(b"task").hexdigest(),
            wall_seconds=20,
            execution_topology="single-robot",
            gpu_scheduling="driver-default",
            sensor_protocol="egocentric-rgbd-wrist-rgb-kinematics-depth-tools-v4",
            command_protocol="sonic-native-planner-v1",
            completion_protocol="agent-declaration-then-independent-verification-v1",
            balance_contact_protocol="ground-reaching-filtered-floor-contact-v2",
            clock_policy="wall-clock-lag-v1",
            sonic_constants_sha256=hashlib.sha256(b"constants").hexdigest(),
            robot_urdf_sha256=hashlib.sha256(b"robot").hexdigest(),
            api_service_tier="standard",
            api_request_timeout_seconds=180,
            project_lock_sha256=hashlib.sha256(b"project lock").hexdigest(),
            physics_lock_sha256=hashlib.sha256(b"physics lock").hexdigest(),
        )
        assignments = []
        for name, status, lag in (
            ("success", "completed", 0.01),
            ("invalid", "completed", 0.2),
            ("failed", "infrastructure_error", 0.01),
        ):
            directory = root / name
            for subdirectory in ("source", "inputs", "physics"):
                (directory / subdirectory).mkdir(parents=True)
            (directory / "source/runner.py").write_bytes(b"saved source")
            for filename, data in (
                ("scene.json", b"scene"),
                ("sonic_constants.json", b"constants"),
                ("robot.urdf", b"robot"),
                ("uv.lock", b"project lock"),
                ("physics-uv.lock", b"physics lock"),
            ):
                (directory / "inputs" / filename).write_bytes(data)
            atomic_json(directory / "source-manifest.json", sources)
            validity = {
                "status": "valid" if lag <= 0.1 else "invalid_physics_lag",
                "maximum_physics_lag_seconds": lag,
                "limit_seconds": 0.1,
                "policy": "wall-clock-lag-v1",
            }
            outcome = shutdown_task_outcome_record(
                end_reason="agent_declared_completion",
                physical_criterion_met=True,
                agent_requested_completion=True,
                task_success=True,
            )
            record = {
                "schema_version": "webolt-episode-v3",
                "provider": "sol",
                "episode_kind": "model",
                "protocol": protocol.model_dump(mode="json"),
                "status": status,
                "error": "operator failed" if status == "infrastructure_error" else None,
                "cleanup_errors": [],
                "component_exit_codes": {
                    "physics": 0,
                    "operator": 1 if status == "infrastructure_error" else 0,
                },
                "task_outcome": outcome,
                "simulation_validity": validity,
                "analysis": "not_run",
                "rendering": "not_run",
                "publication": "not_run",
            }
            atomic_json(
                directory / "episode.json",
                EpisodeRecord.model_validate(record).model_dump(mode="json"),
            )
            atomic_json(directory / "physics/task-outcome.json", outcome)
            atomic_json(
                directory / "physics/io-summary.json",
                {
                    "schema_version": "webolt-episode-io-v1",
                    "status": "completed",
                    "error": None,
                    "operations_processed": 2,
                    "operations_submitted": 2,
                    "interrupted_capture": None,
                },
            )
            atomic_json(
                directory / "physics/result.json",
                {
                    "finish_reason": "agent_declared_completion",
                    "status": "completed",
                    "elapsed_wall_seconds": 20,
                    "task_success": True,
                    "physics_steps": 4000,
                    "trace_rows": 400,
                    "distinct_native_commands_applied": 1000,
                    "control_started": True,
                    "physics_started_after_controller_ready": True,
                    "maximum_physics_clock_lag_seconds": lag,
                    "simulation_validity": validity,
                    "maximum_command_age_seconds": 0.01,
                    "physics_process_model_requests": 0,
                    "agent_requested_completion": True,
                },
            )
            assignments.append({"directory": name, "condition": "sol", "provider": "sol"})
        manifest = root / "cohort.json"
        atomic_json(
            manifest,
            {
                "schema_version": "webolt-cohort-v3",
                "protocol": protocol.model_dump(mode="json"),
                "episodes": assignments,
            },
        )
        return manifest

    def test_episode_analysis_uses_archived_outcome_without_supervisor_projections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare_cohort(root)
            episode = root / "success"
            physics = episode / "physics"
            recorder = EpisodeRecorder(physics)
            for name in ("task-outcome.json", "result.json"):
                projection = physics / name
                recorder.write(name, json.loads(projection.read_text()))
                projection.unlink()
            recorder.write(
                "physical-trace.jsonl",
                {"root_position": [0, 0, 0.76], "wall_seconds": 1, "simulation_seconds": 1},
            )
            recorder.close()
            summary = json.loads((physics / "io-summary.json").read_text())
            atomic_json(
                physics / "io-summary.json",
                {
                    **summary,
                    "schema_version": "webolt-episode-io-v3",
                    "recording_sha256": recording_sha256(recorder.path),
                },
            )
            output = root / "analysis"
            analyze_episode(episode, output)
            analysis = json.loads((output / "analysis.json").read_text())
            self.assertTrue(analysis["task_outcome"]["task_success"])
            self.assertEqual(analysis["sampled_trace_rows"], 1)

    def test_invalid_simulation_and_operator_failure_remain_in_started_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare_cohort(Path(directory))
            summary = analyze_episode_cohort(manifest)
            self.assertEqual(
                summary["conditions"],
                {
                    "sol": {
                        "started": 3,
                        "infrastructure_error": 1,
                        "cancelled": 0,
                        "invalid_or_unknown_simulation": 1,
                        "valid_completed": 1,
                        "verified_success": 1,
                    }
                },
            )

    def test_completed_mps_episode_requires_successful_service_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_cohort(root)
            record = json.loads((root / "success/episode.json").read_text())
            record["protocol"]["gpu_scheduling"] = "cuda-mps"
            for exit_code in (None, 1):
                record["component_exit_codes"]["gpu_scheduling"] = exit_code
                with self.assertRaises(ValueError):
                    EpisodeRecord.model_validate(record)
            record["component_exit_codes"]["gpu_scheduling"] = 0
            self.assertEqual(EpisodeRecord.model_validate(record).status, "completed")

    def test_missing_versions_incompatible_protocols_and_mutated_sources_abort_comparison(
        self,
    ) -> None:
        for mutation in (
            "version",
            "protocol",
            "topology",
            "gpu_scheduling",
            "source",
            "missing",
            "duplicate",
            "nan",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.prepare_cohort(root)
                episode_path = root / "success/episode.json"
                record = json.loads(episode_path.read_text())
                if mutation == "version":
                    record.pop("schema_version")
                elif mutation == "protocol":
                    record["protocol"]["wall_seconds"] = 21
                elif mutation == "topology":
                    record["protocol"]["execution_topology"] = "shared-physics"
                elif mutation == "gpu_scheduling":
                    record["protocol"]["gpu_scheduling"] = "cuda-mps"
                    record["component_exit_codes"]["gpu_scheduling"] = 0
                elif mutation == "source":
                    (root / "success/source/runner.py").write_text("modified")
                elif mutation == "missing":
                    record.pop("task_outcome")
                elif mutation == "nan":
                    record["simulation_validity"]["maximum_physics_lag_seconds"] = float("nan")
                elif mutation == "duplicate":
                    cohort = json.loads(manifest.read_text())
                    cohort["episodes"].append(cohort["episodes"][0])
                    atomic_json(manifest, cohort)
                episode_path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    analyze_episode_cohort(manifest)
