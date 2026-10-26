"""Native multimodal generation clients for the shared robot task schemas."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import anthropic
import openai
from anthropic.types import MessageParam, ToolChoiceAutoParam, ToolParam
from google import genai
from google.genai.interactions import FunctionCallStep, Interaction
from openai import OpenAI, omit, pydantic_function_tool
from openai.types.responses import FunctionToolParam, ResponseInputParam

from humanoid_agent.model_api_configuration import ApiServiceTier as ApiServiceTier
from humanoid_agent.model_api_configuration import ReasoningEffort
from humanoid_agent.model_contracts import (
    ConversationItem,
    InputText,
    ModelRequest,
    NativeContinuation,
    ObservationMessage,
    ToolResultMessage,
)
from humanoid_agent.model_contracts import (
    ModelProvider as ModelProvider,
)
from humanoid_agent.model_contracts import (
    SensorFunctionCall as SensorFunctionCall,
)
from humanoid_agent.model_contracts import (
    TaskGeneration as TaskGeneration,
)
from humanoid_agent.motion_types import MotionUsage
from humanoid_agent.sensor_feedback import DEPTH_TOOL, DepthQuery

MODEL_IDENTIFIERS: dict[ModelProvider, str] = {
    "sol": "gpt-6.1-sol",
    "astra": "gpt-6-astra",
    "opus": "claude-opus-5-5",
    "gemini": "gemini-4-argon",
}


def read_api_key(environment_variable: str, credential_path: Path) -> str:
    value = os.environ.get(environment_variable)
    credential = value.strip() if value is not None else credential_path.read_text().strip()
    if not credential:
        raise ValueError(f"Empty API credential: set {environment_variable}")
    return credential


def recorded_reasoning_summary(record: dict[str, Any]) -> list[str]:
    """Collect the model-written reasoning summaries from one recorded response.

    No provider returns its raw chain of thought; these are the model's own
    summaries of it. Unknown item shapes are ignored rather than raising, so a
    provider's future block type cannot interrupt a recorded episode.
    """
    native = record.get("raw_response") or record
    summaries: list[str] = []
    for key in ("content", "steps", "output"):
        for item in native.get(key) or []:
            if not isinstance(item, dict):
                continue
            match item.get("type"):
                case "thinking":
                    text = item.get("thinking") or ""
                case "thought" | "reasoning":
                    text = "\n".join(
                        block.get("text") or ""
                        for block in item.get("summary") or []
                        if isinstance(block, dict)
                    )
                case _:
                    continue
            if text.strip():
                summaries.append(text)
    return summaries


def native_conversation(
    conversation: Sequence[ConversationItem], provider: ModelProvider
) -> list[dict[str, Any]]:
    """Translate typed application messages while retaining native blocks verbatim."""
    translated: list[dict[str, Any]] = []
    for item in conversation:
        if isinstance(item, NativeContinuation):
            if item.provider != provider:
                raise ValueError("A native continuation belongs to a different provider")
            if provider in {"sol", "astra"}:
                translated.extend(openai_response_item_as_input(block) for block in item.items)
            else:
                translated.extend(item.items)
        elif isinstance(item, ToolResultMessage):
            if provider in {"sol", "astra"}:
                translated.append(item.model_dump())
            elif provider == "opus":
                result = {
                    "type": "tool_result",
                    "tool_use_id": item.call_id,
                    "content": item.output,
                }
                if translated and translated[-1].get("role") == "user":
                    translated[-1]["content"].append(result)
                else:
                    translated.append({"role": "user", "content": [result]})
            else:
                matching_calls = [
                    step
                    for step in translated
                    if step.get("type") == "function_call" and step["id"] == item.call_id
                ]
                if len(matching_calls) != 1:
                    raise ValueError("A native tool result must match exactly one function call")
                translated.append(
                    {
                        "type": "function_result",
                        "call_id": item.call_id,
                        "name": matching_calls[0]["name"],
                        "result": [{"type": "text", "text": item.output}],
                    }
                )
        elif isinstance(item, ObservationMessage):
            if provider in {"sol", "astra"}:
                translated.append(item.model_dump(mode="json", exclude_none=True))
                continue
            content = []
            for part in item.content:
                if isinstance(part, InputText):
                    content.append({"type": "text", "text": part.text})
                else:
                    header, image = part.image_url.split(",", 1)
                    if not header.startswith("data:image/") or not header.endswith(";base64"):
                        raise ValueError("Native model images must be base64 image data URLs")
                    media_type = header.removeprefix("data:").removesuffix(";base64")
                    content.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": image},
                        }
                        if provider == "opus"
                        else {"type": "image", "data": image, "mime_type": media_type}
                    )
            translated.append(
                {"role": "user", "content": content}
                if provider == "opus"
                else {"type": "user_input", "content": content}
            )
    return translated


def openai_response_item_as_input(item: dict[str, Any]) -> dict[str, Any]:
    """Keep signed continuation data but omit response-only lifecycle fields."""
    if item["type"] == "reasoning":
        fields = ["type", "id", "summary", "content", "encrypted_content"]
    elif item["type"] == "function_call":
        fields = ["type", "id", "call_id", "name", "arguments"]
    elif item["type"] == "message":
        return {
            "type": "message",
            "role": item["role"],
            "content": [
                {key: value for key, value in content.items() if key in {"type", "text", "refusal"}}
                for content in item["content"]
            ],
        }
    else:
        raise ValueError("Unexpected model output item in the sensor-only tool protocol")
    return {field: item[field] for field in fields if item.get(field) is not None}


def retryable_api_failure(error: BaseException) -> Literal["transient", "content_blocked"] | None:
    """Classify native SDK failures without retrying arbitrary bad requests."""
    module = type(error).__module__
    if module not in {"openai", "anthropic", "google.genai"} and not module.startswith(
        ("openai.", "anthropic.", "google.genai.")
    ):
        return None
    if isinstance(error, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return "transient"
    if module.startswith("google.genai.") and any(
        parent.__name__ in {"APIConnectionError", "APITimeoutError"}
        for parent in type(error).__mro__
    ):
        return "transient"
    status = getattr(error, "status_code", getattr(error, "code", None))
    if isinstance(status, int) and (status in {408, 409, 429} or 500 <= status <= 599):
        return "transient"
    body = getattr(error, "body", None)
    if status == 400 and isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict) and detail.get("code") == "content_blocked":
            return "content_blocked"
    return None


class TaskModelClient:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        service_tier: ApiServiceTier = ApiServiceTier.STANDARD,
        request_timeout_seconds: float = 300,
        reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    ) -> None:
        if provider == "opus" and service_tier == ApiServiceTier.FLEX:
            raise ValueError("Opus Messages has no Flex tier; use standard or an offline batch job")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("API request timeout must be finite and positive")
        self.provider = provider
        self.service_tier = service_tier
        self.native_service_tier = (
            "standard_only"
            if provider == "opus"
            else "default"
            if provider in {"sol", "astra"} and service_tier == ApiServiceTier.STANDARD
            else service_tier.value
        )
        self.request_timeout_seconds = request_timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.model = MODEL_IDENTIFIERS[provider]
        self.client: OpenAI | anthropic.Anthropic | genai.Client
        if provider in {"sol", "astra"}:
            self.api_path = "openai.responses"
            self.client = OpenAI(
                api_key=read_api_key(
                    "OPENAI_API_KEY", Path.home() / ".config/controller-diversity/openai-api-key"
                ),
                timeout=request_timeout_seconds,
                max_retries=0,
            )
        elif provider == "opus":
            self.api_path = "anthropic.messages"
            self.client = anthropic.Anthropic(
                api_key=read_api_key(
                    "ANTHROPIC_API_KEY", Path.home() / ".config/frontier-motion/anthropic-api-key"
                ),
                timeout=request_timeout_seconds,
                max_retries=0,
            )
        elif provider == "gemini":
            self.api_path = "google.interactions"
            self.client = genai.Client(
                api_key=read_api_key(
                    "GEMINI_API_KEY", Path.home() / ".config/frontier-motion/gemini-api-key"
                ),
                http_options={"retry_options": {"attempts": 0}},
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def generate(self, request: ModelRequest) -> TaskGeneration:
        timeout = self.request_timeout_seconds
        if request.deadline_monotonic_seconds is not None:
            timeout = min(timeout, request.deadline_monotonic_seconds - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Episode deadline reached before API dispatch")
        if self.provider in {"sol", "astra"}:
            if not isinstance(self.client, OpenAI):
                raise TypeError("OpenAI provider requires an OpenAI SDK client")
            tool_schemas = {DEPTH_TOOL["name"]: DepthQuery}
            tools = [
                FunctionToolParam(
                    type="function",
                    name=tool.name,
                    description=tool.description,
                    parameters=pydantic_function_tool(tool_schemas[tool.name])["function"][
                        "parameters"
                    ],
                    strict=tool.strict,
                )
                for tool in request.tools
            ]
            # The application's typed messages are serialized at this SDK boundary.
            # Native continuation dictionaries stay opaque so future signed fields
            # are not stripped by revalidating against a closed SDK TypedDict.
            response = self.client.responses.parse(
                model=self.model,
                input=cast(
                    ResponseInputParam, native_conversation(list(request.input), self.provider)
                ),
                text_format=request.text_format,
                max_output_tokens=request.max_output_tokens,
                tools=tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                include=list(request.include),
                timeout=timeout,
                reasoning={"effort": self.reasoning_effort.value, "summary": "auto"},
                service_tier="default" if self.service_tier == ApiServiceTier.STANDARD else "flex",
                store=request.store if request.store is not None else omit,
            )
            raw = response.model_dump(mode="json")
            usage = response.usage
            return TaskGeneration(
                response.model,
                response.output_parsed,
                MotionUsage(
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
                raw,
                [
                    SensorFunctionCall(
                        call_id=item.call_id, name=item.name, arguments=item.arguments
                    )
                    for item in response.output
                    if item.type == "function_call"
                ],
                [
                    NativeContinuation(
                        provider=self.provider,
                        items=[item.model_dump(by_alias=True) for item in response.output],
                    )
                ],
                reasoning_effort=self.reasoning_effort,
                original_record=raw,
            )
        if self.provider == "opus":
            return self._generate_anthropic(request, timeout)
        return self._generate_google(request, timeout)

    def inference_configuration(self) -> dict[str, Any]:
        return {
            "requested_service_tier": self.service_tier.value,
            "reasoning_effort": self.reasoning_effort.value,
            "native_requested_service_tier": self.native_service_tier,
            "request_timeout_seconds": self.request_timeout_seconds,
            "native_sdk_automatic_retries": False,
            "automatic_tier_fallback": False,
            # Model-written summaries, not the raw chain of thought, which no provider returns.
            "reasoning_summaries_requested": True,
        }

    def _generate_anthropic(self, request: ModelRequest, timeout: float) -> TaskGeneration:
        tool_choice = request.tool_choice
        # Opus 5.5 requires automatic tool choice with its always-on thinking.
        # Never disable thinking to force a sensor call during an integration check.
        if tool_choice not in {"auto", "none"}:
            raise ValueError(
                "Opus 5.5 supports automatic sensor-tool choice; forced tool use is unsupported"
            )
        tools = [] if tool_choice == "none" else list(request.tools)
        # TypeAdapter would regenerate the permissive transport model schema.
        # Send the original decision schema explicitly through the native API.
        if not isinstance(self.client, anthropic.Anthropic):
            raise TypeError("Anthropic provider requires an Anthropic SDK client")
        response = self.client.messages.create(
            model=self.model,
            service_tier="standard_only",
            timeout=timeout,
            max_tokens=request.max_output_tokens,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={
                "effort": self.reasoning_effort.value,
                "format": {
                    "type": "json_schema",
                    "schema": anthropic.transform_schema(request.text_format.model_json_schema()),
                },
            },
            messages=cast(list[MessageParam], native_conversation(list(request.input), "opus")),
            tools=[
                ToolParam(
                    name=tool.name,
                    description=tool.description,
                    input_schema=anthropic.transform_schema(tool.parameters),
                    strict=tool.strict,
                )
                for tool in tools
            ]
            if tools
            else anthropic.omit,
            tool_choice=ToolChoiceAutoParam(type="auto", disable_parallel_tool_use=True)
            if tools
            else anthropic.omit,
        )
        raw = response.model_dump(mode="json", exclude_none=True)
        calls = [
            SensorFunctionCall(call_id=block.id, name=block.name, arguments=json.dumps(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        input_tokens = (
            response.usage.input_tokens
            + (response.usage.cache_read_input_tokens or 0)
            + (response.usage.cache_creation_input_tokens or 0)
        )
        return TaskGeneration(
            response.model,
            None
            if calls
            else request.text_format.model_validate_json(
                "".join(block.text for block in response.content if block.type == "text")
            ),
            MotionUsage(
                input_tokens=input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=input_tokens + response.usage.output_tokens,
            ),
            raw,
            calls,
            [
                NativeContinuation(
                    provider="opus", items=[{"role": "assistant", "content": raw["content"]}]
                )
            ],
            reasoning_effort=self.reasoning_effort,
        )

    def _generate_google(self, request: ModelRequest, timeout: float) -> TaskGeneration:
        schema = request.text_format
        tool_choice = request.tool_choice
        tools = list(request.tools) if tool_choice != "none" else []
        if not isinstance(self.client, genai.Client):
            raise TypeError("Gemini provider requires a Google SDK client")
        response = self.client.interactions.create(
            model=self.model,
            service_tier=self.native_service_tier,
            input=native_conversation(list(request.input), "gemini"),
            store=False,
            stream=False,
            timeout=timeout,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
            tools=[tool.model_dump(exclude={"strict"}) for tool in tools],
            generation_config={
                "thinking_level": self.reasoning_effort.value,
                "thinking_summaries": "auto",
                "max_output_tokens": request.max_output_tokens,
                "tool_choice": "any" if tool_choice == "required" else tool_choice,
            },
        )
        if not isinstance(response, Interaction):
            raise TypeError("Non-streaming generation returned a stream")
        raw = response.model_dump(mode="json", exclude_none=True)
        usage = response.usage
        steps = raw.get("steps") or []
        calls = []
        for step in response.steps or []:
            if isinstance(step, FunctionCallStep):
                if not step.id or not step.name or step.arguments is None:
                    raise ValueError("Native function call requires identity, name and arguments")
                calls.append(
                    SensorFunctionCall(
                        call_id=step.id, name=step.name, arguments=json.dumps(step.arguments)
                    )
                )
        return TaskGeneration(
            response.model or self.model,
            None
            if calls or response.output_text is None
            else schema.model_validate_json(response.output_text),
            MotionUsage(
                input_tokens=(usage.total_input_tokens or 0) if usage else 0,
                output_tokens=(usage.total_output_tokens or 0) if usage else 0,
                total_tokens=(usage.total_tokens or 0) if usage else 0,
            ),
            raw,
            calls,
            [NativeContinuation(provider="gemini", items=steps)],
            reasoning_effort=self.reasoning_effort,
        )
