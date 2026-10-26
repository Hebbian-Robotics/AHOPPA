"""Protect sensor evidence and native reasoning across provider tool continuations."""

import unittest

from pydantic import TypeAdapter

from humanoid_agent.model_contracts import ConversationItem
from humanoid_agent.motion_types import MotionUsage
from humanoid_agent.presentation.publish_isaac_operator import parsed_text
from humanoid_agent.task_model_client import (
    TaskGeneration,
    native_conversation,
    recorded_reasoning_summary,
)
from humanoid_agent.task_planning import TaskPlan


class ReasoningSummaryTests(unittest.TestCase):
    def test_each_provider_shape_yields_readable_summaries_without_signature_noise(self) -> None:
        records = {
            "opus": {
                "raw_response": {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Reach for the brick first.",
                            "signature": "sig",
                        },
                        {"type": "thinking", "thinking": "", "signature": "sig"},
                        {"type": "text", "text": "{}"},
                    ]
                }
            },
            "gemini": {
                "raw_response": {
                    "steps": [
                        {
                            "type": "thought",
                            "signature": "sig",
                            "summary": [{"type": "text", "text": "Check the table height."}],
                        },
                        {"type": "thought", "signature": "sig", "summary": []},
                    ]
                }
            },
            "sol": {
                "output": [
                    {
                        "type": "reasoning",
                        "encrypted_content": "opaque",
                        "summary": [{"type": "summary_text", "text": "Plan the approach."}],
                    },
                    {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
                ]
            },
        }
        expected = {
            "opus": ["Reach for the brick first."],
            "gemini": ["Check the table height."],
            "sol": ["Plan the approach."],
        }
        for provider, record in records.items():
            with self.subTest(provider=provider):
                self.assertEqual(recorded_reasoning_summary(record), expected[provider])

    def test_unknown_block_shapes_are_ignored_rather_than_raising(self) -> None:
        record = {"raw_response": {"content": [{"type": "future_block"}, "not-a-block"]}}
        self.assertEqual(recorded_reasoning_summary(record), [])


class NativeConversationTests(unittest.TestCase):
    def test_native_validated_decision_survives_recording_and_video_export(self) -> None:
        plan = TaskPlan.model_validate(
            {
                "subtasks": [
                    {
                        "identifier": "approach",
                        "description": "Approach the marker",
                        "completion_condition": "Marker is in reach",
                        "status": "active",
                    }
                ]
            }
        )
        for model in ["claude-opus-5-5", "gemini-4-argon"]:
            with self.subTest(model=model):
                recorded = TaskGeneration(
                    model,
                    plan,
                    MotionUsage(input_tokens=10, output_tokens=20, total_tokens=30),
                    {},
                    [],
                    [],
                ).model_dump()
                self.assertEqual(parsed_text(recorded), plan.model_dump())

    def test_same_observation_images_and_text_reach_both_native_providers(self) -> None:
        observation = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Current sensor snapshot"},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,b3RoZXI="},
            ],
        }
        opus_content = native_conversation(
            TypeAdapter(list[ConversationItem]).validate_python([observation]), "opus"
        )[0]["content"]
        gemini_content = native_conversation(
            TypeAdapter(list[ConversationItem]).validate_python([observation]), "gemini"
        )[0]["content"]
        self.assertEqual(opus_content[0], gemini_content[0])
        self.assertEqual(
            [part["source"]["data"] for part in opus_content[1:]],
            [part["data"] for part in gemini_content[1:]],
        )
        self.assertEqual(opus_content[2]["source"]["media_type"], "image/jpeg")
        self.assertEqual(gemini_content[2]["mime_type"], "image/jpeg")

    def test_signed_native_reasoning_and_tool_call_order_survive_continuation(self) -> None:
        for provider in ("opus", "gemini"):
            with self.subTest(provider=provider):
                native_items = (
                    [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "signed-native-state",
                                },
                                {
                                    "type": "tool_use",
                                    "id": "depth-1",
                                    "name": "query_onboard_depth",
                                    "input": {"pixels": [{"x": 42, "y": 21}]},
                                },
                            ],
                        }
                    ]
                    if provider == "opus"
                    else [
                        {"type": "thought", "signature": "signed-native-state"},
                        {
                            "type": "function_call",
                            "id": "depth-1",
                            "name": "query_onboard_depth",
                            "arguments": {"pixels": [{"x": 42, "y": 21}]},
                        },
                    ]
                )
                result = native_conversation(
                    TypeAdapter(list[ConversationItem]).validate_python(
                        [
                            {
                                "type": "native_continuation",
                                "provider": provider,
                                "items": native_items,
                            },
                            {
                                "type": "function_call_output",
                                "call_id": "depth-1",
                                "output": '{"depth_m": 1.25}',
                            },
                        ]
                    ),
                    provider,
                )
                self.assertEqual(result[: len(native_items)], native_items)
                tool_result = result[-1]["content"][0] if provider == "opus" else result[-1]
                self.assertEqual(
                    tool_result["tool_use_id" if provider == "opus" else "call_id"], "depth-1"
                )
                self.assertEqual(
                    tool_result["content"]
                    if provider == "opus"
                    else tool_result["result"][0]["text"],
                    '{"depth_m": 1.25}',
                )

    def test_cross_provider_continuations_fail_before_a_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "different provider"):
            native_conversation(
                TypeAdapter(list[ConversationItem]).validate_python(
                    [{"type": "native_continuation", "provider": "opus", "items": []}]
                ),
                "gemini",
            )


if __name__ == "__main__":
    unittest.main()
