"""Price retained API attempts without changing frozen experiment records."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from cyclopts import App
from pydantic import Field, model_validator

from humanoid_agent.analysis.summarize_shutdown_round import group_label, latex_table
from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.recording import read_record_bytes
from humanoid_agent.validated_types import ValidatedModel

TokenCount = Annotated[int, Field(ge=0)]
DollarAmount = Annotated[Decimal, Field(ge=0)]


class ModelRate(ValidatedModel):
    model: str
    study_provider: str
    api: Literal["openai", "anthropic", "google"]
    input_per_million: DollarAmount
    cached_input_per_million: DollarAmount
    cache_write_short_per_million: DollarAmount
    cache_write_long_per_million: DollarAmount
    output_per_million: DollarAmount
    maximum_input_tokens: Annotated[int, Field(gt=0)]
    source_url: str
    cache_source_url: str


class PricingPolicy(ValidatedModel):
    schema_version: Literal["api-token-pricing-v1"]
    currency: Literal["USD"]
    checked_on: str
    models: list[ModelRate]


class TokenUsage(ValidatedModel):
    input_tokens: TokenCount
    cached_input_tokens: TokenCount = 0
    short_cache_write_tokens: TokenCount = 0
    long_cache_write_tokens: TokenCount = 0
    unspecified_cache_write_tokens: TokenCount = 0
    output_tokens: TokenCount

    @model_validator(mode="after")
    def require_disjoint_input_categories(self) -> Self:
        if self.uncached_input_tokens < 0:
            raise ValueError("Cache reads and writes exceed total input tokens")
        return self

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cached_input_tokens - self.cache_write_tokens

    @property
    def cache_write_tokens(self) -> int:
        return (
            self.short_cache_write_tokens
            + self.long_cache_write_tokens
            + self.unspecified_cache_write_tokens
        )


class CostBounds(ValidatedModel):
    minimum_usd: DollarAmount
    maximum_usd: DollarAmount


class EpisodeCost(ValidatedModel):
    episode: str
    group: str
    condition: str
    is_task_attempt: bool
    task_success: bool | None = None
    elapsed_seconds: Annotated[float, Field(ge=0)] | None = None
    median_decision_seconds: Annotated[float, Field(ge=0)] | None = None
    recorded_calls: TokenCount
    unreported_calls: TokenCount
    recovered_calls: TokenCount = 0
    input_tokens: TokenCount
    cached_input_tokens: TokenCount
    cache_write_tokens: TokenCount
    output_tokens: TokenCount
    cost: CostBounds


class AggregateCost(ValidatedModel):
    attempts: Annotated[int, Field(gt=0)]
    recorded_calls: TokenCount
    unreported_calls: TokenCount
    recovered_calls: TokenCount = 0
    total_cost: CostBounds
    mean_cost_per_attempt: CostBounds


class AttemptTiming(ValidatedModel):
    mean_elapsed_seconds: Annotated[float, Field(ge=0)] | None
    mean_successful_elapsed_seconds: Annotated[float, Field(ge=0)] | None
    median_decision_seconds: Annotated[float, Field(ge=0)] | None
    median_successful_decision_seconds: Annotated[float, Field(ge=0)] | None


def attempt_timing(episodes: list[EpisodeCost]) -> AttemptTiming:
    tasks = [episode for episode in episodes if episode.is_task_attempt]
    successful = [episode for episode in tasks if episode.task_success is True]
    elapsed = [episode.elapsed_seconds for episode in tasks if episode.elapsed_seconds is not None]
    successful_elapsed = [
        episode.elapsed_seconds for episode in successful if episode.elapsed_seconds is not None
    ]
    decisions = [
        episode.median_decision_seconds
        for episode in tasks
        if episode.median_decision_seconds is not None
    ]
    successful_decisions = [
        episode.median_decision_seconds
        for episode in successful
        if episode.median_decision_seconds is not None
    ]
    return AttemptTiming(
        mean_elapsed_seconds=statistics.mean(elapsed)
        if tasks and len(elapsed) == len(tasks)
        else None,
        mean_successful_elapsed_seconds=statistics.mean(successful_elapsed)
        if successful and len(successful_elapsed) == len(successful)
        else None,
        median_decision_seconds=statistics.median(decisions) if decisions else None,
        median_successful_decision_seconds=statistics.median(successful_decisions)
        if successful_decisions
        else None,
    )


class GroupCost(ValidatedModel):
    group: str
    label: str
    task_attempts: AggregateCost
    attempt_timing: AttemptTiming | None = None
    successful_task_attempts: AggregateCost | None = None
    control: AggregateCost


class OpenAIModelUsage(ValidatedModel):
    model: str
    calls: TokenCount
    usage: TokenUsage


class OpenAIUsageEvidence(ValidatedModel):
    schema_version: Literal["ahoppa-openai-usage-v1"] = "ahoppa-openai-usage-v1"
    source_url: str
    raw_report_sha256: str
    query_start_time: TokenCount
    query_end_time: TokenCount
    models: list[OpenAIModelUsage]


class RecoveredUsage(ValidatedModel):
    episode: str
    attempt: str
    model: str
    attempt_timing_sha256: str
    evidence_sha256: str
    source_url: str
    method: Literal["usage_api_residual", "usage_api_zero_residual", "provider_rejection_no_charge"]
    usage: TokenUsage

    @model_validator(mode="after")
    def require_zero_usage_for_no_charge_evidence(self) -> Self:
        if self.method != "usage_api_residual" and any(self.usage.model_dump().values()):
            raise ValueError("No-charge recovery must contain zero tokens")
        return self


class UsageRecoveries(ValidatedModel):
    schema_version: Literal["ahoppa-api-usage-recoveries-v1"] = "ahoppa-api-usage-recoveries-v1"
    calls: list[RecoveredUsage]

    @model_validator(mode="after")
    def require_unique_attempts(self) -> Self:
        identities = {(row.episode, row.attempt) for row in self.calls}
        if len(identities) != len(self.calls):
            raise ValueError("Recovery repeats an API attempt")
        return self


class CostReport(ValidatedModel):
    schema_version: Literal["ahoppa-api-costs-v1"] = "ahoppa-api-costs-v1"
    pricing: PricingPolicy
    openai_usage_evidence: OpenAIUsageEvidence | None = None
    usage_recoveries: UsageRecoveries | None = None
    summary_sha256: str
    record_manifest_sha256: str
    terminal_samples_sha256: str | None = None
    recorded_calls: TokenCount
    unreported_calls: TokenCount
    recovered_calls: TokenCount = 0
    groups: list[GroupCost]
    episodes: list[EpisodeCost]


def token_count(payload: dict[str, Any], field: str, *, default: int | None = None) -> int:
    value = payload.get(field, default)
    if type(value) is not int or value < 0:
        raise ValueError(f"Missing or invalid token count: {field}")
    return value


def recorded_usage(response: dict[str, Any], rate: ModelRate) -> TokenUsage | None:
    """Read native provider usage; normalized totals omit some billing categories."""
    native = response.get("raw_response", response)
    if native.get("model", response.get("model")) != rate.model:
        raise ValueError("Recorded model differs from the pricing model")
    usage = native.get("usage")
    if usage is None:
        return None
    tier = native.get("service_tier", usage.get("service_tier"))
    if tier not in {None, "default", "standard", "standard_only"}:
        raise ValueError(f"Unpriced service tier: {tier}")
    match rate.api:
        case "openai":
            details = usage.get("input_tokens_details") or {}
            writes = token_count(details, "cache_write_tokens", default=0)
            write_details = details.get("cache_write_tokens_details")
            short_writes = long_writes = 0
            if write_details is not None:
                short_writes = token_count(write_details, "ttl_30m_tokens")
                long_writes = token_count(write_details, "ttl_12h_tokens")
                if short_writes + long_writes != writes:
                    raise ValueError("Cache-write lifetime counts disagree with total writes")
            result = TokenUsage(
                input_tokens=token_count(usage, "input_tokens"),
                cached_input_tokens=token_count(details, "cached_tokens", default=0),
                short_cache_write_tokens=short_writes,
                long_cache_write_tokens=long_writes,
                unspecified_cache_write_tokens=writes - short_writes - long_writes,
                output_tokens=token_count(usage, "output_tokens"),
            )
        case "anthropic":
            cache_creation = usage.get("cache_creation") or {}
            writes = token_count(usage, "cache_creation_input_tokens", default=0)
            short_writes = token_count(cache_creation, "ephemeral_5m_input_tokens", default=0)
            long_writes = token_count(cache_creation, "ephemeral_1h_input_tokens", default=0)
            if short_writes + long_writes > writes:
                raise ValueError("Cache-write lifetime counts exceed total writes")
            cached_input = token_count(usage, "cache_read_input_tokens", default=0)
            result = TokenUsage(
                input_tokens=token_count(usage, "input_tokens") + cached_input + writes,
                cached_input_tokens=cached_input,
                short_cache_write_tokens=short_writes,
                long_cache_write_tokens=long_writes,
                unspecified_cache_write_tokens=writes - short_writes - long_writes,
                output_tokens=token_count(usage, "output_tokens"),
            )
        case "google":
            if token_count(usage, "total_tool_use_tokens", default=0):
                raise ValueError("Hosted tool usage needs separate pricing")
            result = TokenUsage(
                input_tokens=token_count(usage, "total_input_tokens"),
                cached_input_tokens=token_count(usage, "total_cached_tokens", default=0),
                output_tokens=token_count(usage, "total_output_tokens")
                + token_count(usage, "total_thought_tokens", default=0),
            )
    if result.input_tokens > rate.maximum_input_tokens:
        raise ValueError("Request exceeds the context length covered by the recorded rate")
    return result


def price_usage(usage: TokenUsage, rate: ModelRate) -> CostBounds:
    known_cost = (
        usage.uncached_input_tokens * rate.input_per_million
        + usage.cached_input_tokens * rate.cached_input_per_million
        + usage.short_cache_write_tokens * rate.cache_write_short_per_million
        + usage.long_cache_write_tokens * rate.cache_write_long_per_million
        + usage.output_tokens * rate.output_per_million
    )
    write_rates = (rate.cache_write_short_per_million, rate.cache_write_long_per_million)
    return CostBounds(
        minimum_usd=(known_cost + usage.unspecified_cache_write_tokens * min(write_rates))
        / 1_000_000,
        maximum_usd=(known_cost + usage.unspecified_cache_write_tokens * max(write_rates))
        / 1_000_000,
    )


def aggregate_costs(episodes: list[EpisodeCost]) -> AggregateCost:
    if not episodes:
        raise ValueError("Cannot report a mean cost without scheduled attempts")
    minimum = sum((episode.cost.minimum_usd for episode in episodes), Decimal(0))
    maximum = sum((episode.cost.maximum_usd for episode in episodes), Decimal(0))
    return AggregateCost(
        attempts=len(episodes),
        recorded_calls=sum(episode.recorded_calls for episode in episodes),
        unreported_calls=sum(episode.unreported_calls for episode in episodes),
        recovered_calls=sum(episode.recovered_calls for episode in episodes),
        total_cost=CostBounds(minimum_usd=minimum, maximum_usd=maximum),
        mean_cost_per_attempt=CostBounds(
            minimum_usd=minimum / len(episodes), maximum_usd=maximum / len(episodes)
        ),
    )


def apply_openai_usage_evidence(
    episodes: list[EpisodeCost], policy: PricingPolicy, evidence: OpenAIUsageEvidence
) -> list[EpisodeCost]:
    """Resolve a model's cache rate only when aggregate usage covers the retained calls."""
    if evidence.query_end_time <= evidence.query_start_time:
        raise ValueError("Usage evidence has an empty query window")
    evidence_by_model = {row.model: row for row in evidence.models}
    if len(evidence_by_model) != len(evidence.models):
        raise ValueError("Usage evidence repeats a model")
    confirmed_lifetimes: dict[str, Literal["30m", "12h"]] = {}
    rates_by_provider = {rate.study_provider: rate for rate in policy.models}
    for rate in policy.models:
        if rate.api != "openai":
            continue
        retained = [
            episode
            for episode in episodes
            if episode.group.partition("-")[0] == rate.study_provider
        ]
        if not retained:
            continue
        model_evidence = evidence_by_model[rate.model]
        recorded_calls = sum(episode.recorded_calls for episode in retained)
        missing_calls = sum(episode.unreported_calls for episode in retained)
        if model_evidence.calls != recorded_calls + missing_calls:
            raise ValueError(
                f"Usage API call count does not match recorded and interrupted calls: {rate.model}"
            )
        reported = model_evidence.usage
        match (reported.short_cache_write_tokens > 0, reported.long_cache_write_tokens > 0):
            case (True, False) if reported.unspecified_cache_write_tokens == 0:
                lifetime = "30m"
            case (False, True) if reported.unspecified_cache_write_tokens == 0:
                lifetime = "12h"
            case _:
                raise ValueError(
                    "Mixed or unknown cache lifetimes cannot resolve per-attempt pricing"
                )
        additional_usage = TokenUsage(
            input_tokens=reported.input_tokens - sum(episode.input_tokens for episode in retained),
            cached_input_tokens=reported.cached_input_tokens
            - sum(episode.cached_input_tokens for episode in retained),
            unspecified_cache_write_tokens=reported.cache_write_tokens
            - sum(episode.cache_write_tokens for episode in retained),
            output_tokens=reported.output_tokens
            - sum(episode.output_tokens for episode in retained),
        )
        unresolved_calls = missing_calls - sum(episode.recovered_calls for episode in retained)
        if unresolved_calls == 0 and any(additional_usage.model_dump().values()):
            raise ValueError("Usage API token totals differ despite complete retained responses")
        confirmed_lifetimes[rate.study_provider] = lifetime
    reconciled_episodes = []
    for episode in episodes:
        provider = episode.group.partition("-")[0]
        if provider not in confirmed_lifetimes:
            reconciled_episodes.append(episode)
            continue
        lifetime = confirmed_lifetimes[provider]
        usage = TokenUsage(
            input_tokens=episode.input_tokens,
            cached_input_tokens=episode.cached_input_tokens,
            short_cache_write_tokens=episode.cache_write_tokens if lifetime == "30m" else 0,
            long_cache_write_tokens=episode.cache_write_tokens if lifetime == "12h" else 0,
            output_tokens=episode.output_tokens,
        )
        reconciled_episodes.append(
            episode.model_copy(update={"cost": price_usage(usage, rates_by_provider[provider])})
        )
    return reconciled_episodes


def main(
    summary_path: Path,
    /,
    *,
    episodes_root: Path,
    pricing: Path,
    output: Path,
    openai_usage_evidence: Path | None = None,
    usage_recoveries: Path | None = None,
    include_episode_timing: bool = False,
) -> None:
    """Write JSON cost evidence and a LaTeX results table from retained API attempts."""
    summary = json.loads(summary_path.read_text())
    policy = PricingPolicy.model_validate_json(pricing.read_text())
    rates_by_model = {rate.model: rate for rate in policy.models}
    recoveries = (
        UsageRecoveries.model_validate_json(usage_recoveries.read_text())
        if usage_recoveries
        else None
    )
    recoveries_by_attempt = (
        {(row.episode, row.attempt): row for row in recoveries.calls} if recoveries else {}
    )
    consumed_recoveries: set[tuple[str, str]] = set()
    record_manifest = hashlib.sha256()
    timing_manifest = hashlib.sha256()
    episode_costs = []
    seen_episodes: set[str] = set()
    for row in summary["episodes"]:
        episode_name = Path(row["directory"]).name
        if episode_name in seen_episodes:
            raise ValueError(f"Repeated scheduled episode: {episode_name}")
        seen_episodes.add(episode_name)
        operator_directory = episodes_root / episode_name / "operator"
        configuration_path = operator_directory / "operator-configuration.json"
        configuration = json.loads(configuration_path.read_text())
        if configuration.get("requested_service_tier") != "standard":
            raise ValueError("Cost policy covers Standard service only")
        rate = rates_by_model[configuration["model"]]
        if row["provider"] != rate.study_provider:
            raise ValueError("Episode provider differs from the priced model")
        record_manifest.update(hashlib.sha256(configuration_path.read_bytes()).digest())
        usage_records = []
        bounds = []
        unreported_calls = 0
        recovered_calls = 0
        recorded_calls = 0
        for timing_path in sorted(operator_directory.glob("*/model.attempt-*.timing.json")):
            timing = json.loads(timing_path.read_text())
            record_manifest.update(hashlib.sha256(timing_path.read_bytes()).digest())
            response_path = timing_path.with_name(
                timing_path.name.removesuffix(".timing.json") + ".response.json"
            )
            attempt = str(timing_path.relative_to(operator_directory)).removesuffix(".timing.json")
            recovery_identity = (episode_name, attempt)
            recovery = recoveries_by_attempt.get(recovery_identity)
            if not response_path.exists():
                if timing["state"] == "received":
                    raise ValueError(f"Received response is missing: {response_path}")
                usage = None
            else:
                response_bytes = response_path.read_bytes()
                record_manifest.update(hashlib.sha256(response_bytes).digest())
                usage = recorded_usage(json.loads(response_bytes), rate)
            if usage is None:
                unreported_calls += 1
                if recovery is None:
                    continue
                if (
                    recovery.model != rate.model
                    or recovery.attempt_timing_sha256
                    != hashlib.sha256(timing_path.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        "Recovered usage differs from the recorded model or attempt timing"
                    )
                usage = recovery.usage
                recovered_calls += 1
                consumed_recoveries.add(recovery_identity)
            else:
                if recovery is not None:
                    raise ValueError("Cannot add recovered usage to a retained response")
                recorded_calls += 1
            usage_records.append(usage)
            bounds.append(price_usage(usage, rate))
        elapsed_seconds = None
        if include_episode_timing:
            terminal_bytes = read_record_bytes(
                operator_directory.parent / "physics/physical-trace.jsonl"
            )
            terminal = json.loads(terminal_bytes)
            elapsed_seconds = float(terminal["wall_seconds"])
            if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
                raise ValueError("Invalid terminal episode elapsed time")
            timing_manifest.update(hashlib.sha256(terminal_bytes).digest())
        episode_costs.append(
            EpisodeCost(
                episode=episode_name,
                group=row["group"],
                condition=row["condition"],
                is_task_attempt=row["requires_manipulation"],
                task_success=row.get("task_success"),
                elapsed_seconds=elapsed_seconds,
                median_decision_seconds=row.get("median_request_seconds"),
                recorded_calls=recorded_calls,
                unreported_calls=unreported_calls,
                recovered_calls=recovered_calls,
                input_tokens=sum(usage.input_tokens for usage in usage_records),
                cached_input_tokens=sum(usage.cached_input_tokens for usage in usage_records),
                cache_write_tokens=sum(usage.cache_write_tokens for usage in usage_records),
                output_tokens=sum(usage.output_tokens for usage in usage_records),
                cost=CostBounds(
                    minimum_usd=sum((bound.minimum_usd for bound in bounds), Decimal(0)),
                    maximum_usd=sum((bound.maximum_usd for bound in bounds), Decimal(0)),
                ),
            )
        )
    if consumed_recoveries != recoveries_by_attempt.keys():
        raise ValueError("Recovery references an API attempt outside the retained summary")
    usage_evidence = (
        OpenAIUsageEvidence.model_validate_json(openai_usage_evidence.read_text())
        if openai_usage_evidence is not None
        else None
    )
    if usage_evidence is not None:
        episode_costs = apply_openai_usage_evidence(episode_costs, policy, usage_evidence)
    grouped: dict[str, list[EpisodeCost]] = defaultdict(list)
    for episode in episode_costs:
        grouped[episode.group].append(episode)
    provider_order = {rate.study_provider: index for index, rate in enumerate(policy.models)}
    ordered_groups = sorted(
        grouped,
        key=lambda group: (provider_order[group.partition("-")[0]], group.endswith("high")),
    )
    report = CostReport(
        pricing=policy,
        openai_usage_evidence=usage_evidence,
        usage_recoveries=recoveries,
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        record_manifest_sha256=record_manifest.hexdigest(),
        terminal_samples_sha256=timing_manifest.hexdigest() if include_episode_timing else None,
        recorded_calls=sum(episode.recorded_calls for episode in episode_costs),
        unreported_calls=sum(episode.unreported_calls for episode in episode_costs),
        recovered_calls=sum(episode.recovered_calls for episode in episode_costs),
        groups=[
            GroupCost(
                group=group,
                label=group_label(group),
                attempt_timing=attempt_timing(grouped[group]),
                task_attempts=aggregate_costs(
                    [episode for episode in grouped[group] if episode.is_task_attempt]
                ),
                successful_task_attempts=aggregate_costs(
                    [
                        episode
                        for episode in grouped[group]
                        if episode.is_task_attempt and episode.task_success is True
                    ]
                )
                if any(
                    episode.is_task_attempt and episode.task_success is True
                    for episode in grouped[group]
                )
                else None,
                control=aggregate_costs(
                    [episode for episode in grouped[group] if not episode.is_task_attempt]
                ),
            )
            for group in ordered_groups
        ],
        episodes=episode_costs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report.model_dump(mode="json"))
    cost_columns = {
        group.group: (
            group.task_attempts.mean_cost_per_attempt.minimum_usd,
            group.task_attempts.mean_cost_per_attempt.maximum_usd,
            group.task_attempts.unreported_calls - group.task_attempts.recovered_calls,
        )
        for group in report.groups
    }
    output.with_suffix(".tex").write_text(
        latex_table(
            {group: summary["providers"][group] for group in ordered_groups},
            api_costs=cost_columns,
            attempt_times={
                group.group: (
                    group.attempt_timing.mean_elapsed_seconds,
                    group.attempt_timing.mean_successful_elapsed_seconds,
                )
                for group in report.groups
                if group.attempt_timing is not None
            }
            if include_episode_timing
            else None,
            decision_times={
                group.group: (
                    group.attempt_timing.median_decision_seconds,
                    group.attempt_timing.median_successful_decision_seconds,
                )
                for group in report.groups
                if group.attempt_timing is not None
            }
            if include_episode_timing
            else None,
            successful_api_costs={
                group.group: (
                    group.successful_task_attempts.mean_cost_per_attempt.minimum_usd,
                    group.successful_task_attempts.mean_cost_per_attempt.maximum_usd,
                )
                if group.successful_task_attempts is not None
                else None
                for group in report.groups
            },
        )
    )


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
