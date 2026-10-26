"""Recorded heading transforms and camera projection must agree on displayed targets."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from humanoid_agent.presentation.publish_isaac_operator import (
    compose_recorded_frame,
    model_decision_timeline,
    publish_recording_metadata,
)
from humanoid_agent.presentation.publish_model_inputs import publish_model_inputs
from humanoid_agent.presentation.recorded_motion import (
    ObserverCalibration,
    project_world_points,
)


class ObserverProjectionTests(unittest.TestCase):
    def test_standard_recording_keeps_both_views_and_requires_active_targets(self) -> None:
        onboard = Image.new("RGB", (640, 480), (91, 17, 32))
        external = Image.new("RGB", (640, 360), (31, 62, 93))
        camera = ObserverCalibration(
            eye_world_m=[0, -4, 2],
            look_at_world_m=[0, 0, 2],
            focal_length_pixels=100,
            image_width=640,
            image_height=360,
            composite_header_pixels=100,
        )
        arguments: dict[str, Any] = {
            "calibration": camera,
            "measured_world": np.asarray([[-1, 0, 1.8], [1, 0, 1.8], [0, 0, 2.8]]),
            "path_world": None,
            "source_seconds": 3.5,
            "command_id": "reach",
            "reference_age_seconds": 0.02,
            "representation": "Original captures",
        }
        rendered = compose_recorded_frame(
            onboard,
            external,
            targets_world=np.asarray([[-1, 0, 2], [1, 0, 2], [0, 0, 3]]),
            control_mode="Hybrid control: head/hands poses and locomotion",
            **arguments,
        )
        np.testing.assert_array_equal(
            np.asarray(rendered.crop((640, 40, 1280, 520))), np.asarray(onboard)
        )
        self.assertEqual(rendered.getpixel((295, 280)), (66, 166, 255))
        self.assertEqual(external.getpixel((295, 180)), (31, 62, 93))
        wrists = {
            "left_wrist": Image.new("RGB", (640, 480), (12, 87, 36)),
            "right_wrist": Image.new("RGB", (640, 480), (83, 14, 65)),
        }
        four_views = compose_recorded_frame(
            onboard,
            external,
            targets_world=np.asarray([[-1, 0, 2], [1, 0, 2], [0, 0, 3]]),
            wrist_images=wrists,
            **arguments,
        )
        self.assertEqual(four_views.size, (1280, 1160))
        for index, name in enumerate(("left_wrist", "right_wrist")):
            np.testing.assert_array_equal(
                np.asarray(four_views.crop((index * 640, 680, (index + 1) * 640, 1160))),
                np.asarray(wrists[name]),
            )
        np.testing.assert_array_equal(
            np.asarray(four_views.crop((640, 40, 1280, 520))), np.asarray(onboard)
        )
        with self.assertRaisesRegex(ValueError, "both declared wrist views"):
            compose_recorded_frame(
                onboard,
                external,
                targets_world=np.asarray([[-1, 0, 2], [1, 0, 2], [0, 0, 3]]),
                wrist_images={"left_wrist": wrists["left_wrist"]},
                **arguments,
            )
        with self.assertRaisesRegex(ValueError, "target-overlay evidence"):
            compose_recorded_frame(onboard, external, targets_world=None, **arguments)

    def test_commanded_targets_rotate_into_the_same_observer_frame_as_the_robot(self) -> None:
        camera = ObserverCalibration(
            eye_world_m=[0, -4, 2],
            look_at_world_m=[0, 0, 2],
            focal_length_pixels=100,
            image_width=200,
            image_height=100,
            composite_header_pixels=40,
        )
        projected = project_world_points(
            camera, np.asarray([[-1, 0, 2], [1, 0, 2], [0, 0, 3]], dtype=float)
        )
        # Known world points are four metres in front of the calibrated observer.
        # The head is one metre higher. Header pixels shift every displayed y by 40.
        np.testing.assert_allclose(projected, [[75, 90], [125, 90], [100, 65]], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "in front"):
            project_world_points(camera, np.asarray([[0.0, -5.0, 2.0]]))


class ModelInputPublicationTests(unittest.TestCase):
    def test_recording_metadata_uses_response_clock_and_keeps_reviewed_warnings(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gallery = root / "gallery"
            gallery.mkdir()
            manifest = {
                "records": [
                    {
                        "statusPath": "transfer/first/status.json",
                        "identifier": "00000000-0000-4000-8000-000000000001",
                        "description": "Known integration defect; do not score.",
                        "title": "Reviewed title",
                        "archived": False,
                    }
                ],
                "checks": [],
            }
            (gallery / "manifest.json").write_text(json.dumps(manifest))
            episodes = []
            for name in ("first", "second"):
                episode = root / name
                (episode / "physics").mkdir(parents=True)
                (episode / "physics/result.json").write_text(
                    json.dumps(
                        {
                            "finish_reason": "wall_time_limit",
                            "status": "completed",
                            "elapsed_wall_seconds": 10,
                            "task_success": False,
                            "physics_steps": 2000,
                            "trace_rows": 200,
                            "distinct_native_commands_applied": 500,
                            "control_started": True,
                            "physics_started_after_controller_ready": True,
                            "maximum_physics_clock_lag_seconds": 0.07,
                            "simulation_validity": {
                                "status": "valid",
                                "maximum_physics_lag_seconds": 0.07,
                                "limit_seconds": 0.1,
                                "policy": "wall-clock-lag-v1",
                            },
                            "maximum_command_age_seconds": 0.02,
                            "physics_process_model_requests": 0,
                            "agent_requested_completion": False,
                        }
                    )
                )
                operator = episode / "operator"
                operator.mkdir()
                clock = {
                    "episode_id": name,
                    "deadline_monotonic_seconds": 110,
                    "elapsed_wall_seconds": 0,
                    "deadline_elapsed_wall_seconds": 10,
                }
                for clock_path in (
                    operator / "episode-clock.json",
                    episode / "physics/episode-clock.json",
                ):
                    clock_path.write_text(json.dumps(clock))
                for identifier, response_time in (
                    ("initial", 102),
                    ("feedback_000", 105),
                    ("late", 110.1),
                ):
                    request = operator / identifier
                    request.mkdir()
                    (request / "model-timing.json").write_text(
                        json.dumps({"response_received_monotonic_seconds": response_time})
                    )
                    (request / "model.response.json").write_text(
                        json.dumps(
                            {
                                "output_parsed": {
                                    "subtasks": [
                                        {
                                            "identifier": "reach",
                                            "description": "Reach",
                                            "status": "active",
                                        }
                                    ]
                                }
                            }
                        )
                    )
                destination = gallery / name
                destination.mkdir()
                episodes.append((episode, destination))
            self.assertEqual(
                [
                    decision["at"]
                    for decision in model_decision_timeline(episodes[0][0] / "operator", 10)
                ],
                [2, 5],
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(
                    executor.map(
                        lambda episode_destination: publish_recording_metadata(
                            episode_destination[0], episode_destination[1], 10.25
                        ),
                        episodes,
                    )
                )
            published = json.loads((gallery / "manifest.json").read_text())["records"]
            self.assertEqual(len(published), 2)
            first = next(
                record
                for record in published
                if record["statusPath"] == "transfer/first/status.json"
            )
            self.assertEqual(first["identifier"], "00000000-0000-4000-8000-000000000001")
            self.assertEqual(first["description"], "Known integration defect; do not score.")
            self.assertEqual(first["title"], "Reviewed title")
            self.assertEqual([decision["at"] for decision in first["decisions"]], [2, 5])

    def test_paired_history_exports_original_views_and_refuses_modified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operator = root / "operator"
            request = operator / "current"
            request.mkdir(parents=True)
            onboard, external = b"original onboard", b"original external"
            wrists = {"left_wrist": b"original left wrist", "right_wrist": b"original right wrist"}
            for name, image_bytes in wrists.items():
                (request / f"{name}.png").write_bytes(image_bytes)
            (request / "rgb.png").write_bytes(onboard)
            (request / "external.png").write_bytes(external)
            (request / "model-timing.json").write_text(
                json.dumps({"model_request_wall_seconds": 1.2})
            )
            (request / "prompt.json").write_text(
                json.dumps(
                    {
                        "observation": {"robot_feedback": {"captured_elapsed_seconds": 4.0}},
                        "recent_decisions": [{"feedback": {"action_id": "action_001"}}],
                    }
                )
            )
            (request / "visual-history.json").write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "frames": [
                                    {
                                        "observation_id": "current",
                                        "action_id": "action_001",
                                        "role": "current/end",
                                        "captured_elapsed_seconds": 4.0,
                                        "requested_elapsed_seconds": 4.0,
                                    }
                                ],
                                "onboard_path": "rgb.png",
                                "external_path": "external.png",
                                "onboard_sha256": hashlib.sha256(onboard).hexdigest(),
                                "external_sha256": hashlib.sha256(external).hexdigest(),
                                "wrist_views": [
                                    {
                                        "camera": name,
                                        "path": f"{name}.png",
                                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                                    }
                                    for name, image_bytes in wrists.items()
                                ],
                            }
                        ]
                    }
                )
            )
            destination = root / "published"
            publish_model_inputs(operator, destination)
            snapshot = json.loads((destination / "model-inputs.json").read_text())["snapshots"][0]
            self.assertEqual(snapshot["previous_action_feedback"], {"action_id": "action_001"})
            self.assertEqual(snapshot["pairs"][0]["frames"][0]["role"], "current/end")
            self.assertEqual(
                [(root / image["path"]).read_bytes() for image in snapshot["pairs"][0]["images"]],
                [onboard, external, *wrists.values()],
            )
            (request / "right_wrist.png").write_bytes(b"re-rendered replacement")
            with self.assertRaisesRegex(ValueError, "differs from its request manifest"):
                publish_model_inputs(operator, destination)

    def test_preserves_original_image_bytes_and_refuses_unknown_image_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operator = root / "operator"
            initial = operator / "initial"
            feedback = operator / "feedback_000"
            initial.mkdir(parents=True)
            feedback.mkdir()
            (initial / "rgb.png").write_bytes(b"initial uploaded image")
            (feedback / "rgb.png").write_bytes(b"current uploaded image")
            prompt = {
                "observation": {
                    "image_order": ["initial onboard", "current onboard"],
                    "elapsed_wall_seconds": 12.5,
                }
            }
            (feedback / "prompt.json").write_text(json.dumps(prompt))
            (feedback / "model-timing.json").write_text(
                json.dumps({"model_request_wall_seconds": 8.2})
            )
            destination = root / "published"
            publish_model_inputs(operator, destination)
            manifest = json.loads((destination / "model-inputs.json").read_text())
            snapshot = manifest["snapshots"][0]
            self.assertEqual(snapshot["captured_wall_seconds"], 12.5)
            self.assertEqual(snapshot["request_wall_seconds"], 8.2)
            self.assertEqual(
                [(root / image["path"]).read_bytes() for image in snapshot["images"]],
                [b"initial uploaded image", b"current uploaded image"],
            )
            prompt["observation"]["image_order"] = ["external camera"]
            (feedback / "prompt.json").write_text(json.dumps(prompt))
            with self.assertRaisesRegex(ValueError, "Unsupported image order"):
                publish_model_inputs(operator, destination)
