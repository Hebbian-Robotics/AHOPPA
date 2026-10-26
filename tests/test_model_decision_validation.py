"""Rejected numerical decisions stay unexecuted and can be corrected before the deadline."""

import json
import os
import tempfile
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import patch

import anthropic
import httpx2
import numpy as np
from openai import APIStatusError, OpenAI
from pydantic import TypeAdapter

from humanoid_agent.model_api_configuration import ApiRetryPolicy, ReasoningEffort
from humanoid_agent.model_contracts import JsonObject, ModelRequest, NativeContinuation
from humanoid_agent.motion_types import MotionUsage
from humanoid_agent.posture_control import TrackingOrientations
from humanoid_agent.sensor_feedback import DEPTH_TOOL, DepthSnapshot
from humanoid_agent.sensor_tool_client import SensorRequestStopped, request_with_sensor_tools
from humanoid_agent.task_model_client import (
    ApiServiceTier,
    SensorFunctionCall,
    TaskGeneration,
    TaskModelClient,
)
from humanoid_agent.task_planning import TaskPlan


@dataclass
class FakeModelGenerator:
    respond: Callable[[ModelRequest], TaskGeneration]

    def generate(self, request: ModelRequest) -> TaskGeneration:
        return self.respond(request)


class DecisionValidationTests(unittest.TestCase):
    def test_openai_depth_tool_uses_supported_references_and_preserves_pixel_limits(self) -> None:
        def model_api(request: httpx2.Request) -> httpx2.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["reasoning"]["effort"], "high")
            pixels = payload["tools"][0]["parameters"]["properties"]["pixels"]
            if "$ref" in pixels and ("minItems" in pixels or "maxItems" in pixels):
                return httpx2.Response(
                    400,
                    json={"error": {"message": "$ref cannot have minItems/maxItems siblings"}},
                )
            self.assertEqual(pixels["type"], "array")
            self.assertEqual((pixels["minItems"], pixels["maxItems"]), (1, 32))
            return httpx2.Response(
                200,
                json={
                    "id": "resp_depth_schema",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-6.1-sol",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
            )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            client = TaskModelClient("sol", reasoning_effort=ReasoningEffort.HIGH)
        self.addCleanup(client.client.close)
        transport = httpx2.Client(transport=httpx2.MockTransport(model_api))
        self.addCleanup(transport.close)
        assert isinstance(client.client, (OpenAI, anthropic.Anthropic))
        client.client = client.client.with_options(http_client=transport)
        self.addCleanup(client.client.close)
        response = client.generate(
            ModelRequest.model_validate(
                {
                    "text_format": TaskPlan,
                    "input": [],
                    "tools": [DEPTH_TOOL],
                    "max_output_tokens": 100,
                }
            )
        )
        self.assertEqual(response.raw["id"], "resp_depth_schema")

    def test_unsupported_flex_and_invalid_timeouts_fail_before_credentials_are_read(self) -> None:
        with self.assertRaisesRegex(ValueError, "Opus Messages has no Flex tier"):
            TaskModelClient("opus", service_tier=ApiServiceTier.FLEX)
        for request_timeout in [0, -1, float("inf"), float("nan")]:
            with (
                self.subTest(request_timeout=request_timeout),
                self.assertRaisesRegex(ValueError, "finite and positive"),
            ):
                TaskModelClient("sol", request_timeout_seconds=request_timeout)

    def test_absolute_deadline_caps_api_timeout_and_expired_request_is_not_dispatched(self) -> None:
        observed_timeouts = []
        plan = {
            "subtasks": [
                {
                    "identifier": "reach",
                    "description": "Reach the marker",
                    "completion_condition": "Marker changes colour",
                    "status": "active",
                }
            ]
        }

        def model_api(request: httpx2.Request) -> httpx2.Response:
            self.assertEqual(json.loads(request.content)["output_config"]["effort"], "low")
            observed_timeouts.append(request.extensions["timeout"]["read"])
            return httpx2.Response(
                200,
                json={
                    "id": "msg_deadline",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "content": [{"type": "text", "text": json.dumps(plan)}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}):
            client = TaskModelClient("opus", request_timeout_seconds=180)
        self.addCleanup(client.client.close)
        transport = httpx2.Client(transport=httpx2.MockTransport(model_api))
        self.addCleanup(transport.close)
        assert isinstance(client.client, (OpenAI, anthropic.Anthropic))
        client.client = client.client.with_options(http_client=transport)
        self.addCleanup(client.client.close)
        request = {"text_format": TaskPlan, "input": [], "max_output_tokens": 100}
        client.generate(
            ModelRequest.model_validate(
                {**request, "deadline_monotonic_seconds": time.monotonic() + 2}
            )
        )
        self.assertEqual(len(observed_timeouts), 1)
        self.assertGreater(observed_timeouts[0], 0)
        self.assertLessEqual(observed_timeouts[0], 2)
        with self.assertRaisesRegex(TimeoutError, "deadline reached"):
            client.generate(
                ModelRequest.model_validate(
                    {**request, "deadline_monotonic_seconds": time.monotonic() - 1}
                )
            )
        self.assertEqual(len(observed_timeouts), 1)

    def test_malformed_depth_arguments_can_be_corrected_and_raw_evidence_is_retained(self) -> None:
        requests: list[dict[str, Any]] = []
        request_deadline = time.monotonic() + 30
        malformed_arguments = '{"pixels": ['
        corrected_arguments = '{"pixels": [{"x": 320, "y": 240}]}'
        plan = {
            "subtasks": [
                {
                    "identifier": "reach",
                    "description": "Reach the marker",
                    "completion_condition": "Marker changes colour",
                    "status": "active",
                }
            ]
        }

        def network_response(request: ModelRequest) -> TaskGeneration:
            self.assertEqual(request.deadline_monotonic_seconds, request_deadline)
            requests.append(request.model_dump())
            if len(requests) <= 2:
                arguments = malformed_arguments if len(requests) == 1 else corrected_arguments
                call = SensorFunctionCall(
                    call_id=f"depth_{len(requests)}",
                    name="query_onboard_depth",
                    arguments=arguments,
                )
                return TaskGeneration(
                    "test-model",
                    None,
                    MotionUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    {},
                    [call],
                    [],
                )
            return TaskGeneration(
                "test-model",
                request.text_format.model_validate(plan),
                MotionUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                {},
                [],
                [],
            )

        kinematics = {
            "camera_optical_origin_m": [0, 0, 0],
            "camera_optical_to_heading_rotation": np.eye(3).tolist(),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            response = request_with_sensor_tools(
                FakeModelGenerator(network_response),
                directory / "motion.response.json",
                DepthSnapshot(
                    np.ones((480, 640)), kinematics, 0, 0, optical_points_m=np.ones((480, 640, 3))
                ),
                Event(),
                ModelRequest(
                    text_format=TaskPlan, input=[], deadline_monotonic_seconds=request_deadline
                ),
                deadline_monotonic_seconds=request_deadline + 30,
            )
            assert response.output_parsed is not None
            self.assertEqual(response.output_parsed.model_dump(), plan)
            feedback = json.loads(requests[1]["input"][0]["output"])
            self.assertEqual(feedback["error"], "invalid_depth_query")
            corrected_measurement = json.loads(requests[2]["input"][-1]["output"])
            self.assertEqual(corrected_measurement["points"][0]["depth_optical_z_m"], 1.0)
            transcript = json.loads((directory / "motion.tool-00.sensor-tools.json").read_text())
            self.assertEqual(transcript["calls"][0]["arguments_raw"], malformed_arguments)

    def test_rejected_quaternion_is_recorded_then_corrected_without_changing_its_values(
        self,
    ) -> None:
        requests: list[dict[str, Any]] = []
        invalid = {
            "left_wrist": [1, 0, 0, 0],
            "right_wrist": [0.99, 0, 0.1, 0],
            "head": [1, 0, 0, 0],
        }
        corrected = {**invalid, "right_wrist": [1, 0, 0, 0]}

        def network_response(request: ModelRequest) -> TaskGeneration:
            requests.append(request.model_dump())
            payload = invalid if len(requests) == 1 else corrected
            assistant = {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(payload)}],
            }
            return TaskGeneration(
                "claude-opus-5-5",
                request.text_format.model_validate(payload),
                MotionUsage(input_tokens=11, output_tokens=7, total_tokens=18),
                TypeAdapter(JsonObject).validate_python({"content": assistant["content"]}),
                [],
                [NativeContinuation.model_validate({"provider": "opus", "items": [assistant]})],
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "motion.response.json"
            response = request_with_sensor_tools(
                FakeModelGenerator(network_response),
                output,
                DepthSnapshot(
                    np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                ),
                Event(),
                ModelRequest.model_validate(
                    {
                        "text_format": TrackingOrientations,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Generate orientations"}
                                ],
                            }
                        ],
                    }
                ),
                enable_depth_queries=False,
            )
            assert response.output_parsed is not None
            self.assertEqual(response.output_parsed.model_dump(), corrected)
            self.assertEqual(response.validation_rounds, 1)
            self.assertEqual(response.request_count, 2)
            self.assertEqual(response.usage.total_tokens, 36)
            rejected = json.loads(
                (Path(temporary_directory) / "motion.attempt-000.response.json").read_text()
            )
            self.assertEqual(rejected["output_parsed"], invalid)
            feedback = json.loads(requests[1]["input"][-1]["content"][0]["text"])
            self.assertFalse(feedback["command_executed"])
            self.assertEqual(feedback["error"], "invalid_robot_decision")
            self.assertEqual(feedback["details"][0]["loc"], ["right_wrist"])
            self.assertEqual(
                requests[1]["text_format"].model_json_schema(),
                TrackingOrientations.model_json_schema(),
            )
            self.assertEqual(response.model_dump()["model_request_count"], 2)

    def test_native_anthropic_request_preserves_the_decision_schema(self) -> None:
        plan = {
            "subtasks": [
                {
                    "identifier": "reach",
                    "description": "Reach the marker",
                    "completion_condition": "Marker changes colour",
                    "status": "active",
                }
            ]
        }

        def model_api(request: httpx2.Request) -> httpx2.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["service_tier"], "standard_only")
            self.assertEqual(payload["output_config"]["effort"], "high")
            # Reject a missing action contract as a provider would. The SDK's
            # TypeAdapter path previously turned the transport wrapper into {}.
            schema = payload["output_config"]["format"]["schema"]
            self.assertIn("subtasks", schema.get("properties", {}))
            return httpx2.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "content": [{"type": "text", "text": json.dumps(plan)}],
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                },
            )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}):
            client = TaskModelClient("opus", reasoning_effort=ReasoningEffort.HIGH)
        self.addCleanup(client.client.close)
        transport = httpx2.Client(transport=httpx2.MockTransport(model_api))
        self.addCleanup(transport.close)
        assert isinstance(client.client, (OpenAI, anthropic.Anthropic))
        client.client = client.client.with_options(http_client=transport)
        self.addCleanup(client.client.close)
        with tempfile.TemporaryDirectory() as temporary_directory:
            response = request_with_sensor_tools(
                client,
                Path(temporary_directory) / "plan.response.json",
                DepthSnapshot(
                    np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                ),
                Event(),
                ModelRequest.model_validate(
                    {
                        "text_format": TaskPlan,
                        "input": [
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": "Plan the task"}],
                            }
                        ],
                        "max_output_tokens": 6000,
                    }
                ),
                enable_depth_queries=False,
            )
            assert response.output_parsed is not None
            self.assertEqual(response.output_parsed.model_dump(), plan)
            self.assertEqual(response.model_dump()["reasoning_effort"], "high")

    def test_episode_stop_prevents_a_correction_request_after_rejection(self) -> None:
        stop_requested = Event()

        def network_response(request: ModelRequest) -> TaskGeneration:
            stop_requested.set()
            payload = {
                "left_wrist": [1, 0, 0, 0],
                "right_wrist": [0, 0, 0, 0],
                "head": [1, 0, 0, 0],
            }
            return TaskGeneration(
                "gemini-4-argon",
                request.text_format.model_validate(payload),
                MotionUsage(input_tokens=11, output_tokens=7, total_tokens=18),
                {"output_text": json.dumps(payload)},
                [],
                [],
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(SensorRequestStopped):
                request_with_sensor_tools(
                    FakeModelGenerator(network_response),
                    Path(temporary_directory) / "motion.response.json",
                    DepthSnapshot(
                        np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                    ),
                    stop_requested,
                    ModelRequest.model_validate({"text_format": TrackingOrientations, "input": []}),
                    enable_depth_queries=False,
                )
            self.assertFalse(
                (Path(temporary_directory) / "motion.attempt-001.timing.json").exists()
            )
            self.assertTrue(
                (Path(temporary_directory) / "motion.attempt-000.validation.json").exists()
            )

    def test_retryable_failure_reuses_the_snapshot_and_records_both_attempts(self) -> None:
        for status, code in [(429, "rate_limit"), (400, "content_blocked")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                requests: list[ModelRequest] = []
                response_path = Path(temporary) / "motion.response.json"

                def generate(
                    request: ModelRequest,
                    observed_requests: list[ModelRequest] = requests,
                    response_status: int = status,
                    response_code: str = code,
                ) -> TaskGeneration:
                    observed_requests.append(request)
                    if len(observed_requests) == 1:
                        raise APIStatusError(
                            "temporary provider failure",
                            response=httpx2.Response(
                                response_status,
                                request=httpx2.Request("POST", "https://example.test"),
                            ),
                            body={"error": {"code": response_code}},
                        )
                    return TaskGeneration(
                        "test-model",
                        request.text_format.model_validate(
                            {
                                "subtasks": [
                                    {
                                        "identifier": "inspect",
                                        "description": "Inspect drawers",
                                        "completion_condition": "Drawers observed",
                                        "status": "active",
                                    }
                                ]
                            }
                        ),
                        MotionUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        {},
                        [],
                        [],
                    )

                result = request_with_sensor_tools(
                    FakeModelGenerator(generate),
                    response_path,
                    DepthSnapshot(
                        np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                    ),
                    Event(),
                    ModelRequest(text_format=TaskPlan, input=[]),
                    enable_depth_queries=False,
                    api_retry_policy=ApiRetryPolicy.BOUNDED,
                )
                self.assertEqual(result.request_count, 2)
                self.assertEqual(requests[0].input_record(), requests[1].input_record())
                self.assertEqual(
                    (Path(temporary) / "motion.attempt-000.input.json").read_bytes(),
                    (Path(temporary) / "motion.attempt-001.input.json").read_bytes(),
                )
                states = [
                    json.loads(
                        (Path(temporary) / f"motion.attempt-{index:03d}.timing.json").read_text()
                    )["state"]
                    for index in range(2)
                ]
                self.assertEqual(states, ["failed", "received"])
                self.assertTrue((Path(temporary) / "motion.attempt-000.retry.json").exists())

    def test_persistent_filter_and_invalid_requests_end_with_the_original_failure(self) -> None:
        for status, code, expected_attempts in [
            (400, "content_blocked", 2),
            (400, "invalid_request", 1),
            (401, "invalid_key", 1),
        ]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                attempts = 0
                failure = APIStatusError(
                    "provider failure",
                    response=httpx2.Response(
                        status, request=httpx2.Request("POST", "https://example.test")
                    ),
                    body={"error": {"code": code}},
                )

                def generate(
                    request: ModelRequest, failure_to_raise: APIStatusError = failure
                ) -> TaskGeneration:
                    nonlocal attempts
                    attempts += 1
                    raise failure_to_raise

                with self.assertRaises(APIStatusError) as raised:
                    request_with_sensor_tools(
                        FakeModelGenerator(generate),
                        Path(temporary) / "motion.response.json",
                        DepthSnapshot(
                            np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                        ),
                        Event(),
                        ModelRequest(text_format=TaskPlan, input=[]),
                        enable_depth_queries=False,
                        api_retry_policy=ApiRetryPolicy.BOUNDED,
                    )
                self.assertIs(raised.exception, failure)
                self.assertEqual(attempts, expected_attempts)
                self.assertEqual(
                    len(list(Path(temporary).glob("*.timing.json"))), expected_attempts
                )

    def test_remaining_deadline_prevents_a_retry_that_cannot_fit(self) -> None:
        attempts = 0

        def generate(request: ModelRequest) -> TaskGeneration:
            nonlocal attempts
            attempts += 1
            raise APIStatusError(
                "busy",
                response=httpx2.Response(
                    429, request=httpx2.Request("POST", "https://example.test")
                ),
                body={},
            )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(APIStatusError):
                request_with_sensor_tools(
                    FakeModelGenerator(generate),
                    Path(temporary) / "motion.response.json",
                    DepthSnapshot(
                        np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                    ),
                    Event(),
                    ModelRequest(
                        text_format=TaskPlan,
                        input=[],
                        deadline_monotonic_seconds=time.monotonic() + 0.1,
                    ),
                    enable_depth_queries=False,
                    api_retry_policy=ApiRetryPolicy.BOUNDED,
                )
            self.assertEqual(attempts, 1)
            self.assertEqual(list(Path(temporary).glob("*.retry.json")), [])

    def test_provider_timeout_remains_distinct_from_a_local_episode_stop(self) -> None:
        def network_response(request: ModelRequest) -> TaskGeneration:
            raise TimeoutError("provider timeout")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TimeoutError, "provider timeout") as failure:
                request_with_sensor_tools(
                    FakeModelGenerator(network_response),
                    Path(directory) / "motion.response.json",
                    DepthSnapshot(
                        np.ones((480, 640)), {}, 0, 0, optical_points_m=np.ones((480, 640, 3))
                    ),
                    Event(),
                    ModelRequest.model_validate({"text_format": TrackingOrientations, "input": []}),
                    enable_depth_queries=False,
                )
            self.assertNotIsInstance(failure.exception, SensorRequestStopped)


if __name__ == "__main__":
    unittest.main()
