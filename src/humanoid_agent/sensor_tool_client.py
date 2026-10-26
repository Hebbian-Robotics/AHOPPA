"""Let the model inspect a frozen sensor snapshot before emitting SONIC references."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    stop_any,
    stop_before_delay,
    stop_when_event_set,
    wait_fixed,
)
from tenacity.nap import sleep_using_event

from humanoid_agent.artifact_io import atomic_json, canonical_json_sha256
from humanoid_agent.model_api_configuration import (
    RECORDED_API_MAXIMUM_ATTEMPTS,
    RECORDED_API_RETRY_WAIT_SECONDS,
    ApiRetryPolicy,
)
from humanoid_agent.model_contracts import (
    ConversationItem,
    InputText,
    ModelGenerator,
    ModelRequest,
    ObservationMessage,
    SensorTool,
    TaskGeneration,
    ToolResultMessage,
)
from humanoid_agent.model_decision_validation import transport_payload_schema
from humanoid_agent.motion_types import MotionUsage
from humanoid_agent.sensor_feedback import DEPTH_TOOL, DepthSnapshot
from humanoid_agent.task_model_client import recorded_reasoning_summary, retryable_api_failure


class SensorRequestStopped(TimeoutError):
    """The local episode stop event was reached between native requests."""


@dataclass
class SensorToolResponse:
    native_response: TaskGeneration
    usage: MotionUsage
    tool_rounds: int
    validated_output: BaseModel | None = None
    validation_rounds: int = 0
    request_count: int = 0

    @property
    def model(self) -> str:
        return self.native_response.model

    @property
    def output_parsed(self) -> BaseModel | None:
        return self.validated_output

    def model_dump(self) -> dict[str, Any]:
        native = self.native_response.model_dump()
        return {
            **native,
            "reasoning_summary": recorded_reasoning_summary(native),
            "request_group_usage": self.usage.model_dump(),
            "perception_tool_rounds": self.tool_rounds,
            "validation_correction_rounds": self.validation_rounds,
            "model_request_count": self.request_count,
            "output_parsed": self.validated_output.model_dump()
            if self.validated_output is not None
            else None,
            "motion_authority": (
                "LLM-generated numerical references; perception tools are read-only"
            ),
        }


def conversation_record(conversation: list[ConversationItem]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json", exclude_none=True) for item in conversation]


def request_with_sensor_tools(
    client: ModelGenerator,
    response_path: Path,
    snapshot: DepthSnapshot,
    stop_requested: Event,
    request: ModelRequest,
    *,
    enable_depth_queries: bool = True,
    api_retry_policy: ApiRetryPolicy = ApiRetryPolicy.NONE,
    deadline_monotonic_seconds: float | None = None,
    validate_output: Callable[[BaseModel], None] | None = None,
) -> SensorToolResponse:
    deadlines = [
        deadline
        for deadline in (request.deadline_monotonic_seconds, deadline_monotonic_seconds)
        if deadline is not None
    ]
    deadline_monotonic_seconds = min(deadlines) if deadlines else None
    allowed_tools = [SensorTool.model_validate(DEPTH_TOOL)] if enable_depth_queries else []
    allowed_tool_names = {tool.name for tool in allowed_tools}
    conversation = list(request.input)
    totals = MotionUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    tool_rounds = 0
    validation_rounds = 0
    request_count = 0
    decision_schema = request.text_format
    perception_path = response_path.with_suffix(".perception.json")
    perception_path.write_text(json.dumps({"queries": [], "read_only": True}) + "\n")
    while not stop_requested.is_set():
        if (
            deadline_monotonic_seconds is not None
            and time.monotonic() >= deadline_monotonic_seconds
        ):
            stop_requested.set()
            break
        current_request = request.model_copy(
            update={
                "input": conversation,
                "text_format": transport_payload_schema(decision_schema),
                "tools": allowed_tools,
                "parallel_tool_calls": False,
                "tool_choice": (
                    (request.tool_choice if tool_rounds == 0 else "auto")
                    if allowed_tools
                    else "none"
                ),
                "include": ["reasoning.encrypted_content"],
                "deadline_monotonic_seconds": deadline_monotonic_seconds
                if deadline_monotonic_seconds is not None
                else request.deadline_monotonic_seconds,
            }
        )
        attempt_prefix = response_path

        def generate_recorded_attempt(native_request: ModelRequest) -> TaskGeneration:
            nonlocal request_count, attempt_prefix
            if stop_requested.is_set():
                raise SensorRequestStopped("Episode stopped before an API attempt")
            attempt_prefix = response_path.with_name(
                response_path.name.removesuffix(".response.json") + f".attempt-{request_count:03d}"
            )
            atomic_json(
                attempt_prefix.with_suffix(attempt_prefix.suffix + ".input.json"),
                native_request.input_record(),
            )
            started_at = time.monotonic()
            request_count += 1
            attempt_record = {
                "request_number": request_count,
                "started_monotonic_seconds": started_at,
                "input_sha256": canonical_json_sha256(conversation_record(conversation)),
            }
            atomic_json(
                attempt_prefix.with_suffix(attempt_prefix.suffix + ".timing.json"),
                {**attempt_record, "state": "started"},
            )
            try:
                native_response = client.generate(native_request)
            except BaseException as error:
                interrupted = (
                    stop_requested.is_set()
                    or isinstance(error, (KeyboardInterrupt, SystemExit))
                    or (
                        deadline_monotonic_seconds is not None
                        and time.monotonic() >= deadline_monotonic_seconds
                    )
                )
                atomic_json(
                    attempt_prefix.with_suffix(attempt_prefix.suffix + ".timing.json"),
                    {
                        **attempt_record,
                        "state": "interrupted" if interrupted else "failed",
                        "elapsed_seconds": time.monotonic() - started_at,
                        "exception_type": type(error).__name__,
                        "api_failure_category": retryable_api_failure(error),
                    },
                )
                if interrupted and not isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise SensorRequestStopped(
                        "Absolute episode deadline reached during API request"
                    ) from error
                raise
            atomic_json(
                attempt_prefix.with_suffix(attempt_prefix.suffix + ".response.json"),
                native_response.model_dump(),
            )
            atomic_json(
                attempt_prefix.with_suffix(attempt_prefix.suffix + ".timing.json"),
                {
                    **attempt_record,
                    "state": "received",
                    "elapsed_seconds": time.monotonic() - started_at,
                },
            )
            return native_response

        def record_retry(state: RetryCallState) -> None:
            error = state.outcome.exception() if state.outcome is not None else None
            retry_prefix = response_path.with_name(
                response_path.name.removesuffix(".response.json")
                + f".attempt-{request_count - 1:03d}"
            )
            atomic_json(
                retry_prefix.with_suffix(retry_prefix.suffix + ".retry.json"),
                {
                    "policy": api_retry_policy.value,
                    "failure_category": retryable_api_failure(error) if error else None,
                    "next_attempt": state.attempt_number + 1,
                    "wait_seconds": state.next_action.sleep if state.next_action else 0,
                    "same_input_sha256": canonical_json_sha256(conversation_record(conversation)),
                },
            )

        stopping = stop_any(
            stop_after_attempt(RECORDED_API_MAXIMUM_ATTEMPTS), stop_when_event_set(stop_requested)
        )
        if deadline_monotonic_seconds is not None:
            stopping = stop_any(
                stopping,
                stop_before_delay(max(0, deadline_monotonic_seconds - time.monotonic())),
            )
        native_response = Retrying(
            retry=retry_if_exception(
                lambda error: (
                    api_retry_policy == ApiRetryPolicy.BOUNDED
                    and retryable_api_failure(error) is not None
                )
            ),
            stop=stopping,
            wait=wait_fixed(RECORDED_API_RETRY_WAIT_SECONDS),
            sleep=sleep_using_event(stop_requested),
            before_sleep=record_retry,
            reraise=True,
        )(generate_recorded_attempt, current_request)
        if native_response.usage:
            totals.input_tokens += native_response.usage.input_tokens
            totals.output_tokens += native_response.usage.output_tokens
            totals.total_tokens += native_response.usage.total_tokens
        tool_calls = [item for item in native_response.output if item.type == "function_call"]
        if tool_calls and not allowed_tools:
            raise ValueError("Model attempted a perception tool disabled by its assigned condition")
        if not tool_calls:
            if native_response.output_parsed is None:
                raise ValueError(
                    "Model returned no structured decision; inspect the retained raw response"
                )
            try:
                validated_output = decision_schema.model_validate(
                    native_response.output_parsed.model_dump()
                )
                if validate_output is not None:
                    validate_output(validated_output)
            except ValueError as error:
                feedback = {
                    "error": "invalid_robot_decision",
                    "details": error.errors(
                        include_input=False, include_context=False, include_url=False
                    )
                    if isinstance(error, ValidationError)
                    else [{"message": str(error)}],
                    "command_executed": False,
                    "instruction": "Correct the rejected decision and return the full required schema. No values were changed for you. The observation and depth tools still refer to the same frozen snapshot; physics continues and the episode deadline has not been extended.",
                }
                atomic_json(
                    attempt_prefix.with_suffix(attempt_prefix.suffix + ".validation.json"), feedback
                )
                conversation.extend(native_response.continuation)
                conversation.append(
                    ObservationMessage(content=[InputText(text=json.dumps(feedback))])
                )
                validation_rounds += 1
                atomic_json(
                    perception_path,
                    {
                        "queries": snapshot.queries,
                        "read_only": True,
                        "request_group_usage": totals.model_dump(),
                        "validation_correction_rounds": validation_rounds,
                    },
                )
                continue
            atomic_json(
                perception_path,
                {
                    "queries": snapshot.queries,
                    "read_only": True,
                    "request_group_usage": totals.model_dump(),
                    "validation_correction_rounds": validation_rounds,
                },
            )
            return SensorToolResponse(
                native_response,
                totals,
                tool_rounds,
                validated_output,
                validation_rounds,
                request_count,
            )
        round_prefix = response_path.with_name(
            response_path.name.removesuffix(".response.json") + f".tool-{tool_rounds:02d}"
        )
        round_prefix.with_suffix(round_prefix.suffix + ".response.json").write_text(
            json.dumps(native_response.model_dump(), indent=2) + "\n"
        )
        request_digest = canonical_json_sha256(conversation_record(conversation))
        conversation.extend(native_response.continuation)
        transcript = []
        for tool_call in tool_calls:
            if tool_call.name not in allowed_tool_names:
                raise ValueError("Model requested a tool outside the sensor allowlist")
            try:
                result = snapshot.query(tool_call.arguments)
            except ValidationError as error:
                result = {
                    "error": "invalid_depth_query",
                    "details": error.errors(include_input=False, include_context=False),
                    "hint": "Use integer pixels x=0..639 and y=0..479 in the current onboard image",
                }
            try:
                parsed_arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
            transcript.append(
                {
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": parsed_arguments,
                    "arguments_raw": tool_call.arguments,
                    "result": result,
                }
            )
            conversation.append(
                ToolResultMessage(call_id=tool_call.call_id, output=json.dumps(result))
            )
        round_prefix.with_suffix(round_prefix.suffix + ".sensor-tools.json").write_text(
            json.dumps(
                {
                    "model": native_response.model,
                    "input_sha256": request_digest,
                    "tool_schema_sha256": canonical_json_sha256(
                        DEPTH_TOOL if enable_depth_queries else []
                    ),
                    "calls": transcript,
                    "read_only": True,
                    "motion_commands_from_tools": 0,
                },
                indent=2,
            )
            + "\n"
        )
        perception_path.write_text(
            json.dumps(
                {
                    "queries": snapshot.queries,
                    "read_only": True,
                    "request_group_usage": totals.model_dump(),
                },
                indent=2,
            )
            + "\n"
        )
        tool_rounds += 1
    raise SensorRequestStopped("Episode ended before another model tool round could begin")
