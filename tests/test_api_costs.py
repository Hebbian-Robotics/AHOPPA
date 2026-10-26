"""Billing estimates preserve provider token semantics and scheduled-attempt scope."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from humanoid_agent.analysis.summarize_api_costs import (
    ModelRate,
    PricingPolicy,
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
            self.assertEqual(
                Decimal(group["task_attempts"]["mean_cost_per_attempt"]["minimum_usd"]),
                Decimal("0.0004"),
            )
            self.assertEqual(
                Decimal(group["control"]["total_cost"]["minimum_usd"]), Decimal("0.0018")
            )


if __name__ == "__main__":
    unittest.main()
