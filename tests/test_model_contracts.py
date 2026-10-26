"""Typed model adapters preserve native evidence while checking application inputs."""

import json
import os
import unittest
from unittest.mock import patch

from google import genai
from google.genai.interactions import Interaction
from pydantic import ValidationError

from humanoid_agent.model_contracts import (
    InputText,
    JsonObject,
    ModelRequest,
    NativeContinuation,
    ObservationMessage,
    ToolResultMessage,
)
from humanoid_agent.sensor_feedback import DEPTH_TOOL
from humanoid_agent.task_model_client import TaskModelClient, native_conversation
from humanoid_agent.task_planning import TaskPlan


class ModelContractTests(unittest.TestCase):
    def test_request_rejects_invalid_configuration_and_copied_updates(self) -> None:
        for invalid in (
            {"max_output_tokens": True},
            {"max_output_tokens": 0},
            {"parallel_tool_calls": True},
            {"deadline_monotonic_seconds": float("nan")},
            {"tools": [{**DEPTH_TOOL, "name": "move_robot"}]},
            {"input": [{"role": "assistant", "content": []}]},
            {"unexpected": "field"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                ModelRequest.model_validate({"input": [], "text_format": TaskPlan, **invalid})
        request = ModelRequest(input=[], text_format=TaskPlan)
        with self.assertRaises(ValidationError):
            request.model_copy(update={"max_output_tokens": 0})

    def test_input_record_keeps_schema_and_opaque_native_continuations(self) -> None:
        signed: JsonObject = {
            "type": "future_native_block",
            "signature": "opaque",
            "future": {"version": 7},
        }
        continuation = NativeContinuation(provider="gemini", items=[signed])
        request = ModelRequest(
            input=[
                ObservationMessage(content=[InputText(text="Inspect the snapshot")]),
                continuation,
            ],
            text_format=TaskPlan,
        )
        record = request.input_record()
        self.assertEqual(record["text_format_schema"], TaskPlan.model_json_schema())
        self.assertEqual(native_conversation([continuation], "gemini"), [signed])
        self.assertEqual(json.loads(json.dumps(record))["input"][1]["items"], [signed])

    def test_openai_continuation_drops_only_response_lifecycle_fields(self) -> None:
        continuation = NativeContinuation(
            provider="sol",
            items=[
                {
                    "type": "function_call",
                    "id": "fc_a",
                    "call_id": "call_a",
                    "name": "query_onboard_depth",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
        )
        transcript = native_conversation(
            [continuation, ToolResultMessage(call_id="call_a", output='{"depth_m": 1}')], "sol"
        )
        self.assertNotIn("status", transcript[0])
        self.assertEqual(transcript[0]["call_id"], transcript[1]["call_id"])
        self.assertEqual(continuation.items[0]["status"], "completed")

    def test_google_native_step_types_and_signed_blocks_survive(self) -> None:
        response = Interaction.model_validate(
            {
                "id": "interaction_a",
                "status": "completed",
                "model": "gemini-4-argon",
                "steps": [
                    {"type": "thought", "signature": "signed-state", "summary": []},
                    {
                        "type": "function_call",
                        "id": "depth_a",
                        "name": "query_onboard_depth",
                        "arguments": {"pixels": [{"x": 1, "y": 2}]},
                    },
                ],
                "usage": {"total_input_tokens": 2, "total_output_tokens": 3, "total_tokens": 5},
            }
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key"}):
            client = TaskModelClient("gemini")
        assert isinstance(client.client, genai.Client)
        self.addCleanup(client.client.close)
        with patch.object(client.client.interactions, "create", return_value=response):
            generated = client.generate(ModelRequest(input=[], text_format=TaskPlan))
        self.assertEqual(generated.usage.total_tokens, 5)
        self.assertEqual(generated.output[0].call_id, "depth_a")
        self.assertIsNone(generated.output_parsed)
        original_steps = response.model_dump(mode="json", exclude_none=True)["steps"]
        self.assertEqual(native_conversation(generated.continuation, "gemini"), original_steps)
        self.assertEqual(
            generated.model_dump()["raw_response"],
            response.model_dump(mode="json", exclude_none=True),
        )
