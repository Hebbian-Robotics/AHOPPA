"""Typed requests and results shared by native model adapters and the tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field, FiniteFloat, JsonValue

from humanoid_agent.model_api_configuration import ReasoningEffort
from humanoid_agent.motion_types import MotionUsage
from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel

type ModelProvider = Literal["sol", "astra", "opus", "gemini"]
type JsonObject = dict[str, JsonValue]


class InputText(ValidatedModel):
    type: Literal["input_text"] = "input_text"
    text: str


class InputImage(ValidatedModel):
    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["auto", "low", "high"] | None = None


class ObservationMessage(ValidatedModel):
    role: Literal["user"] = "user"
    content: ImmutableSequence[InputText | InputImage]


class ToolResultMessage(ValidatedModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class NativeContinuation(ValidatedModel):
    type: Literal["native_continuation"] = "native_continuation"
    provider: ModelProvider
    # Native blocks are opaque evidence, including signatures and future fields.
    items: ImmutableSequence[JsonObject]


type ConversationItem = ObservationMessage | ToolResultMessage | NativeContinuation


class SensorTool(ValidatedModel):
    type: Literal["function"] = "function"
    name: Literal["query_onboard_depth", "preview_robot_pose"]
    description: str
    parameters: JsonObject
    strict: Literal[True] = True


class ModelRequest(ValidatedModel):
    input: ImmutableSequence[ConversationItem]
    text_format: type[BaseModel]
    max_output_tokens: int = Field(default=12000, gt=0, le=65536)
    tools: ImmutableSequence[SensorTool] = ()
    tool_choice: Literal["auto", "none", "required"] = "auto"
    parallel_tool_calls: Literal[False] = False
    include: ImmutableSequence[Literal["reasoning.encrypted_content"]] = ()
    store: bool | None = None
    deadline_monotonic_seconds: FiniteFloat | None = None

    def input_record(self) -> JsonObject:
        values = self.model_dump(mode="json", exclude={"text_format"}, exclude_none=True)
        return {**values, "text_format_schema": self.text_format.model_json_schema()}


class SensorFunctionCall(ValidatedModel):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


@dataclass
class TaskGeneration:
    model: str
    output_parsed: BaseModel | None
    usage: MotionUsage
    raw: JsonObject
    output: list[SensorFunctionCall]
    continuation: list[NativeContinuation]
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    # Preserve the existing OpenAI recording shape while normalizing the
    # application's result interface. Other providers retain their native wrapper.
    original_record: JsonObject | None = None

    def model_dump(self) -> JsonObject:
        if self.original_record is not None:
            return self.original_record
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort.value,
            "usage": self.usage.model_dump(),
            "raw_response": self.raw,
            "output_parsed": self.output_parsed.model_dump()
            if self.output_parsed is not None
            else None,
            "output": [item.model_dump() for item in self.output],
            "native_continuation": [item.model_dump(mode="json") for item in self.continuation],
        }


class ModelGenerator(Protocol):
    def generate(self, request: ModelRequest) -> TaskGeneration: ...
