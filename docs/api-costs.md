# Cost and timing estimates

[Results](figures/storage-results.json) · [Cost evidence](figures/api-costs.json)

## Reported quantities

The paper and project page report two means for each model and reasoning effort:

- **All attempts:** API cost and elapsed time averaged over all seven scheduled
  task attempts, including failures, early terminations and API-error starts.
- **Successful attempts:** the same quantities averaged only over attempts that
  met the full success criterion. These are the costs and durations of observed
  successful tasks, rather than total spending divided by the success count.

The all-closed control is separate. A dash and asterisk mean **no successful
attempt observed**, as with Gemini at both efforts. They do not mean zero cost.
Costs use recorded and provider-reconciled token usage at published standard
rates in USD. They are estimates, not invoice amounts. GPU hosting, taxes and
account discounts are excluded. Small remaining accounting gaps are documented
below and in the page's expandable pricing details.

## Timing

Elapsed episode time comes from the final retained physics trace sample's
`wall_seconds`, available for all 64 starts. It includes active episode time
through the observed termination, including inference while physics runs.
It excludes preflight and deployment. Failed attempts can terminate early;
their durations measure time until termination, not successful completion.
All-attempt means retain all seven starts. Successful means use the successful
subset. The cost evidence records a digest of all terminal samples consumed.

The completion chart shows each successful declaration time. Its separate
model-decision latency column reports all / successful task attempts: the median
of per-attempt median completed-decision times from the frozen summary.
Interrupted decisions have no completed-decision latency. Controls are excluded.
Physics lag is a separate simulator timing quantity.

In this sweep, Astra high was cheaper despite generating more output tokens.
Among the seven task attempts, cached reads covered about 50% of high-effort
input versus 29% of low-effort input. Both all-attempt and successful-attempt
cost means are lower at high effort. These observations do not establish a
general relationship between reasoning effort and cost.

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
OpenAI's organization report confirms only 30-minute cache writes in these runs,
resolving the earlier price ranges.

## Token accounting

Each retained native response is priced once, from
`model.attempt-*.response.json` paired with its attempt timing file. Final
response wrappers and tool-response copies are excluded. All native calls within
a decision count, including depth-tool and correction calls. No separate
provider-hosted tool charges were recorded.

- OpenAI input is partitioned into uncached input, cache reads and cache writes.
  Write prices replace the ordinary input price. Output includes reasoning.
- Anthropic input excludes cache reads and writes, which are added separately.
  Output includes thinking tokens. Recorded Opus cache counts are zero.
- Gemini input includes cached tokens, priced separately from the remainder.
  Visible-output and thought counts are added for billed output. Implicit
  caching has no explicit cache-storage charge.

Multiply each disjoint category by its rate and divide by one million. Add
reconciled usage to its uniquely matched attempt, then calculate group means.
Native records are unchanged: no replacement responses or physical outcomes
are invented. Recovery records bind to the original timing file's SHA-256 and
are rejected if they refer to a retained response, mismatch its model or timing,
repeat an attempt or reference an attempt outside the summary.

## Accounting coverage and recovery

There are 3,701 attempted native calls. Usage was saved for 3,677; 24 calls
(0.65%) lacked saved response usage. Eighteen were interrupted during shutdown;
four Gemini calls returned HTTP 400 and two timed out. Completed responses
record usage; stopping a request before its response arrives leaves no native
usage payload, even if the provider later charges it.

| Provider | Saved responses | Gaps | Resolved gaps | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| OpenAI Astra | 996 | 0 | 0 | 0 |
| OpenAI Sol | 1,023 | 6 | 6 | 0 |
| Anthropic Opus | 941 | 3 | 3 | 0 |
| Gemini | 717 | 15 | 4 | 11 |

Provider reports and rejection policy resolve 13 gaps. Eleven Gemini calls
remain unresolved, 0.30% of all attempted calls; ten belong to task attempts
and one to the control. These charges are excluded from cost estimates. This
small accounting limitation is separate from task success and timing validity.

### OpenAI

The [usage evidence](figures/openai-usage.json) records provider totals and the
query window. All 996 Astra responses match every provider token total exactly.
Sol reports 1,029 calls versus 1,023 saved responses, with six additional calls.
All writes use 30-minute caching, with zero 12-hour writes.

Use native `completed_at` timestamps to align saved responses with provider
minute buckets. An earlier pass used creation times, which shifted usage
between buckets and obscured this matching. Completion times reconcile every
saved call and leave exactly six one-call residual buckets. Each uniquely
matches one interrupted Sol attempt in its bounded request window. UTC bounds
come from retained same-host request clock pairs and the configured request
timeout. All six residual token totals are included in the appropriate attempt.
Their aggregate cost is USD 0.179259 at the standard rate.

### Anthropic

The [Messages Usage API][anthropic-usage] reports 17,054,933 uncached input and
1,359,397 output tokens for the model/key/workspace in the experiment window.
Match the 941 saved responses to timestamped HTTP 200 receipts in each original
`operator.log`, in request order. These receipts reconcile every saved token
and leave two positive residual minutes, each with one candidate interrupted
attempt. They add 32,613 input and 1,757 output tokens, USD 0.165592, to two
low-effort attempts. The third interrupted call has no additional reported
usage in the queried window and receives zero tokens from that reconciliation.

### Gemini

Four original error records explicitly report HTTP 400 rejections. Google's
[billing policy][google-billing] states that HTTP 400 requests are not charged
for tokens; those attempts have confirmed zero cost for their rejected call.
Other calls in the same episode are still priced normally.

After Google authentication was refreshed, Cloud Monitoring queries for the
experiment window returned 27,487,727 input-token quota units and 781,985 output
tokens. These exceed saved usage by 463,287 input and 40,526 output tokens.
Two request-quota series each count 732 requests; they are duplicate quota
views, not 1,464 separate calls. These monitoring metrics do not expose the
cached-input split needed for exact pricing or reliably match residual usage
to individual attempts. No billing export was present in the queried project's
BigQuery dataset list. Do not treat quota consumption as itemized invoice data.
The remaining nine interruptions and two timeouts stay unresolved.

## Reproduce

With the frozen summary and extracted episode records available locally:

```bash
mkdir -p artifacts/api-costs-v3
uv run python -m humanoid_agent.analysis.summarize_api_costs \
  artifacts/storage-comparison-v3/summary.json \
  --episodes-root artifacts/storage-comparison-v3/episodes \
  --pricing configs/api-token-pricing.json \
  --openai-usage-evidence docs/figures/openai-usage.json \
  --usage-recoveries docs/figures/api-usage-recoveries.json \
  --include-episode-timing \
  --output artifacts/api-costs-v3/summary.json
cp artifacts/api-costs-v3/summary.json docs/figures/api-costs.json
cp artifacts/api-costs-v3/summary.tex paper/results-table.tex
uv run python -m humanoid_agent.presentation.build_shutdown_results \
  artifacts/storage-comparison-v3/summary.json \
  --attempt-metrics docs/figures/api-costs.json \
  --output docs/figures/storage-results.json
```

This reads frozen evidence without changing it or making model requests.
The cost JSON preserves per-attempt counts, reconciled usage, conditional means,
source digests and the pricing policy. Recovery evidence is in
[api-usage-recoveries.json](figures/api-usage-recoveries.json). Account-level
reports stay outside Git; public evidence omits project, key and user IDs and
contains no credentials. The raw-record dataset is public and ungated at
[hebbianrobotics/AHOPPA][dataset], with all 64 archives and checksums.

[openai]: https://developers.openai.com/api/docs/pricing
[cache]: https://developers.openai.com/api/docs/guides/prompt-caching
[anthropic]: https://platform.claude.com/docs/en/models/opus-5-5/overview
[anthropic-usage]: https://platform.claude.com/docs/en/api/admin/usage_report/retrieve_messages
[google]: https://ai.google.dev/gemini-api/docs/pricing#gemini-4-argon
[google-billing]: https://ai.google.dev/gemini-api/docs/billing
[dataset]: https://huggingface.co/datasets/hebbianrobotics/AHOPPA
