"""Billing estimates preserve provider token semantics and scheduled-attempt scope."""

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from humanoid_agent.analysis.summarize_api_costs import (
    CostBounds,
    EpisodeCost,
    ModelRate,
    OpenAIModelUsage,
    OpenAIUsageEvidence,
    PricingPolicy,
    TokenUsage,
    apply_openai_usage_evidence,
    main,
    price_usage,
    recorded_usage,
)

PRICING_PATH = Path(__file__).resolve().parents[1] / "configs/api-token-pricing.json"


class ApiCostTests(unittest.TestCase):
    def test_provider_billing_categories_are_counted_once(self) -> None:
        policy = PricingPolicy.model_validate_json(PRICING_PATH.read_text())
        rates: dict[str, ModelRate] = {rate.api: rate for rate in policy.models}
        examples = [
            (
                "openai",
                {
                    "input_tokens": 1200,
                    "input_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 300},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 60},
                },
                Decimal("0.00203"),
                Decimal("0.00278"),
            ),
            (
                "anthropic",
                {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 700,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 300,
                        "ephemeral_1h_input_tokens": 400,
                    },
                    "output_tokens": 50,
                },
                Decimal("0.00614"),
                Decimal("0.00614"),
            ),
            (
                "google",
                {
                    "total_input_tokens": 1000,
                    "total_cached_tokens": 800,
                    "total_output_tokens": 30,
                    "total_thought_tokens": 70,
                },
                Decimal("0.00148"),
                Decimal("0.00148"),
            ),
        ]
        for api, native_usage, minimum, maximum in examples:
            with self.subTest(api=api):
                rate = rates[api]
                usage = recorded_usage({"model": rate.model, "usage": native_usage}, rate)
                assert usage is not None
                estimate = price_usage(usage, rate)
                self.assertEqual(estimate.minimum_usd, minimum)
                self.assertEqual(estimate.maximum_usd, maximum)

    def test_known_cache_lifetime_removes_the_price_range(self) -> None:
        rate = PricingPolicy.model_validate_json(PRICING_PATH.read_text()).models[1]
        usage = recorded_usage(
            {
                "model": rate.model,
                "usage": {
                    "input_tokens": 1200,
                    "input_tokens_details": {
                        "cached_tokens": 800,
                        "cache_write_tokens": 300,
                        "cache_write_tokens_details": {
                            "ttl_30m_tokens": 200,
                            "ttl_12h_tokens": 100,
                        },
                    },
                    "output_tokens": 100,
                },
            },
            rate,
        )
        assert usage is not None
        estimate = price_usage(usage, rate)
        self.assertEqual(estimate.minimum_usd, Decimal("0.00228"))
        self.assertEqual(estimate.maximum_usd, Decimal("0.00228"))

    def test_missing_usage_is_unknown_and_invalid_cache_counts_are_rejected(self) -> None:
        rate = PricingPolicy.model_validate_json(PRICING_PATH.read_text()).models[1]
        self.assertIsNone(recorded_usage({"model": rate.model}, rate))
        with self.assertRaisesRegex(ValueError, "exceed total input"):
            recorded_usage(
                {
                    "model": rate.model,
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 30},
                        "output_tokens": 10,
                    },
                },
                rate,
            )

    def test_usage_evidence_resolves_cache_price_without_allocating_extra_calls(self) -> None:
        policy = PricingPolicy.model_validate_json(PRICING_PATH.read_text())
        episode = EpisodeCost(
            episode="retained-attempt",
            group="sol-low",
            condition="open-A",
            is_task_attempt=True,
            recorded_calls=2,
            unreported_calls=1,
            input_tokens=1000,
            cached_input_tokens=600,
            cache_write_tokens=300,
            output_tokens=100,
            cost=CostBounds(minimum_usd=Decimal("0.00201"), maximum_usd=Decimal("0.00276")),
        )
        model_usage = OpenAIModelUsage(
            model="gpt-6.1-sol",
            calls=3,
            usage=TokenUsage(
                input_tokens=1300,
                cached_input_tokens=650,
                short_cache_write_tokens=550,
                output_tokens=140,
            ),
        )
        evidence = OpenAIUsageEvidence(
            source_url="https://developers.openai.com/docs/api-reference/usage",
            raw_report_sha256="a" * 64,
            query_start_time=0,
            query_end_time=60,
            models=[model_usage],
        )
        result = apply_openai_usage_evidence([episode], policy, evidence)[0]
        self.assertEqual(result.cost.minimum_usd, Decimal("0.00201"))
        self.assertEqual(result.cost.maximum_usd, Decimal("0.00201"))
        self.assertEqual(result.unreported_calls, 1)
        self.assertEqual(result.output_tokens, 100)
        invalid_cases = [
            (episode, model_usage.model_copy(update={"calls": 4}), "call count"),
            (
                episode,
                model_usage.model_copy(
                    update={
                        "usage": model_usage.usage.model_copy(
                            update={"short_cache_write_tokens": 300, "long_cache_write_tokens": 250}
                        )
                    }
                ),
                "Mixed",
            ),
            (
                episode.model_copy(update={"unreported_calls": 0}),
                model_usage.model_copy(update={"calls": 2}),
                "token totals differ",
            ),
        ]
        for retained, reported, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                apply_openai_usage_evidence(
                    [retained], policy, evidence.model_copy(update={"models": [reported]})
                )

    def test_report_prices_native_calls_once_and_separates_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode_rows = []
            for name, task, tokens in [
                ("task-one", True, 100),
                ("task-two", True, 300),
                ("control", False, 900),
            ]:
                operator = root / name / "operator"
                decision = operator / "decision-000"
                decision.mkdir(parents=True)
                physics = root / name / "physics"
                physics.mkdir()
                (physics / "physical-trace.jsonl").write_text(
                    json.dumps(
                        {"wall_seconds": {"task-one": 100, "task-two": 20, "control": 500}[name]}
                    )
                )
                (operator / "operator-configuration.json").write_text(
                    json.dumps(
                        {
                            "requested_service_tier": "standard",
                            "model": "gpt-6.1-sol",
                        }
                    )
                )
                response = {
                    "model": "gpt-6.1-sol",
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                }
                for filename, content in [
                    ("model.attempt-000.timing.json", {"state": "received"}),
                    ("model.attempt-000.response.json", response),
                    ("model.response.json", response),
                    ("model.attempt-000.tool-00.response.json", response),
                ]:
                    (decision / filename).write_text(json.dumps(content))
                if name == "task-two":
                    (decision / "model.attempt-001.timing.json").write_text(
                        '{"state":"interrupted"}'
                    )
                episode_rows.append(
                    {
                        "directory": name,
                        "provider": "sol",
                        "group": "sol-low",
                        "condition": name,
                        "requires_manipulation": task,
                        "task_success": name == "task-one",
                        "median_request_seconds": {"task-one": 40, "task-two": 10, "control": 1}[
                            name
                        ],
                    }
                )
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps({"episodes": episode_rows, "providers": {"sol-low": {}}})
            )
            output_path = root / "costs.json"
            main(summary_path, episodes_root=root, pricing=PRICING_PATH, output=output_path)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["recorded_calls"], 3)
            self.assertEqual(report["unreported_calls"], 1)
            group = report["groups"][0]
            self.assertEqual(group["task_attempts"]["attempts"], 2)
            self.assertEqual(group["successful_task_attempts"]["attempts"], 1)
            self.assertEqual(
                Decimal(group["successful_task_attempts"]["mean_cost_per_attempt"]["minimum_usd"]),
                Decimal("0.0002"),
            )
            self.assertEqual(
                Decimal(group["task_attempts"]["mean_cost_per_attempt"]["minimum_usd"]),
                Decimal("0.0004"),
            )
            self.assertEqual(
                Decimal(group["control"]["total_cost"]["minimum_usd"]), Decimal("0.0018")
            )
            missing_timing = root / "task-two/operator/decision-000/model.attempt-001.timing.json"
            recovery_path = root / "recoveries.json"
            recovery = {
                "episode": "task-two",
                "attempt": "decision-000/model.attempt-001",
                "model": "gpt-6.1-sol",
                "attempt_timing_sha256": hashlib.sha256(missing_timing.read_bytes()).hexdigest(),
                "evidence_sha256": "b" * 64,
                "source_url": "https://developers.openai.com/docs/api-reference/usage",
                "method": "usage_api_residual",
                "usage": {"input_tokens": 200, "output_tokens": 100},
            }
            recovery_path.write_text(json.dumps({"calls": [recovery]}))
            main(
                summary_path,
                episodes_root=root,
                pricing=PRICING_PATH,
                output=output_path,
                usage_recoveries=recovery_path,
                include_episode_timing=True,
            )
            recovered = json.loads(output_path.read_text())
            self.assertEqual(recovered["recorded_calls"], 3)
            self.assertEqual(recovered["unreported_calls"], 1)
            self.assertEqual(recovered["recovered_calls"], 1)
            timing = recovered["groups"][0]["attempt_timing"]
            self.assertEqual(timing["mean_elapsed_seconds"], 60)
            self.assertEqual(timing["mean_successful_elapsed_seconds"], 100)
            self.assertEqual(timing["median_decision_seconds"], 25)
            self.assertEqual(timing["median_successful_decision_seconds"], 40)
            self.assertEqual(
                Decimal(
                    recovered["groups"][0]["task_attempts"]["mean_cost_per_attempt"]["minimum_usd"]
                ),
                Decimal("0.0011"),
            )
            self.assertEqual(
                Decimal(recovered["groups"][0]["control"]["total_cost"]["minimum_usd"]),
                Decimal("0.0018"),
            )
            recovery["attempt_timing_sha256"] = "c" * 64
            recovery_path.write_text(json.dumps({"calls": [recovery]}))
            with self.assertRaisesRegex(ValueError, "attempt timing"):
                main(
                    summary_path,
                    episodes_root=root,
                    pricing=PRICING_PATH,
                    output=output_path,
                    usage_recoveries=recovery_path,
                )


if __name__ == "__main__":
    unittest.main()
