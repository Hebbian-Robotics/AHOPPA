# Estimated API costs

[Results](figures/storage-results.json) · [Cost evidence](figures/api-costs.json)

## Scope and coverage

The paper and project page report mean estimated API cost in USD over all seven
scheduled task attempts per model and effort, including unsuccessful and
API-error attempts. The all-closed control is separate. These are estimates
from recorded usage and published standard rates, not billed amounts. GPU
hosting, taxes, account discounts and calls without usage are excluded.

Across all 64 starts, 3,677 native responses have usage and 24 calls have no
recorded usage. The 56 task attempts have 21 calls without usage; the remaining
three belong to controls. Asterisks identify groups with missing usage. Those
charges are unknown, not zero, so the estimate may understate actual charges.
API-error episodes can still contain earlier billable responses.

## Published prices

Prices below are USD per million tokens, checked October 26, 2026. This is a
pricing verification date. The machine-readable policy is
[`configs/api-token-pricing.json`](../configs/api-token-pricing.json).

| Model | Uncached input | Cache read | Cache write, short / long | Output |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 10 | 1 | 12.5 / 25 | 50 |
| GPT-6.1 Sol | 2 | 0.1 | 2.5 / 5 | 10 |
| Claude Opus 5.5 | 4 | 0.2 | 5 / 8 | 20 |
| Gemini 4 Argon | 2 | 0.1 | Not applicable | 10 |

Sources: [OpenAI pricing][openai], [OpenAI prompt caching][cache],
[Anthropic Opus 5.5][anthropic], [Gemini pricing][google]. Gemini uses the
published promotional standard rates valid through November 30, 2026.
No recorded request exceeds the context length covered by these prices.

OpenAI's two write prices cover 30-minute and 12-hour caching. Saved responses
identify cache-write tokens but do not identify their billed lifetime. The
older `prompt_cache_retention: "24h"` request field does not resolve that
billing category. We therefore report a range using both published write
rates. It describes pricing uncertainty, not variation between attempts or a
confidence interval. Calls without usage remain excluded from both endpoints;
the range is not a bound on the actual invoice.

## Token accounting

Each native response is priced once, from `model.attempt-*.response.json` paired
with its attempt timing file. Final response wrappers and tool-response copies
are excluded. All native calls within a decision count, including depth-tool
and correction calls. Interrupted or failed calls without a response are
counted as missing usage. No extra provider-hosted tool charges were recorded.

- OpenAI input is partitioned into uncached input, cache reads and cache writes.
  Write prices replace the ordinary input price for those tokens. Output
  already includes reasoning tokens.
- Anthropic input excludes cache reads and writes, which are added separately.
  Output already includes thinking tokens. Recorded Opus cache counts are zero.
- Gemini input includes cached tokens, which are priced separately from the
  remainder. Its native visible-output and thought counts are added for billed
  output. The study uses implicit caching with no explicit cache-storage charge.

For each call, multiply each disjoint token category by its price and divide
by one million. Sum across all retained calls in an attempt, then divide group
totals by seven. The JSON preserves attempt-level counts, control estimates,
full-precision decimal costs, the pricing policy, the frozen-summary SHA-256
and a digest of the configuration, timing and response evidence consumed.
It contains no raw prompts, response text or account credentials.

## Reproduce

With the frozen summary and extracted episode records available locally:

```bash
mkdir -p artifacts/api-costs-v1
uv run python -m humanoid_agent.analysis.summarize_api_costs \
  artifacts/storage-comparison-v3/summary.json \
  --episodes-root artifacts/storage-comparison-v3/episodes \
  --pricing configs/api-token-pricing.json \
  --output artifacts/api-costs-v1/summary.json
cp artifacts/api-costs-v1/summary.json docs/figures/api-costs.json
cp artifacts/api-costs-v1/summary.tex paper/results-table.tex
```

This reads the frozen experiment evidence without changing it or making API
requests. The raw records for all 64 reported starts are available from
[hebbianrobotics/AHOPPA on Hugging Face][dataset]. The dataset card provides
download instructions, checksums and documented omissions.

[openai]: https://developers.openai.com/api/docs/pricing
[cache]: https://developers.openai.com/api/docs/guides/prompt-caching
[anthropic]: https://platform.claude.com/docs/en/models/opus-5-5/overview
[google]: https://ai.google.dev/gemini-api/docs/pricing#gemini-4-argon
[dataset]: https://huggingface.co/datasets/hebbianrobotics/AHOPPA
