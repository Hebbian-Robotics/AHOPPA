# Reproduction

## What can be reproduced from this release

The maintained package contains the storage task, observations, model adapters,
controller integration and evaluator. The released CSV and aggregate JSON
contain every scheduled start in the initial study. Selected videos and model
input images are a separate release asset. Full MCAP traces and raw provider
responses are not distributed here, so this is not a complete raw-data archive.

Rerunning the protocol requires paid model API access and a separately prepared
GPU simulator environment. API aliases and service behavior can change; a new
run is not expected to produce an identical trajectory or response.

## Frozen experiment

The reported 64 episodes used source commit
`b1f28c3956a4f6880ca67a633a26425e4afe080b`, with 136 captured Python source
files.
The source manifest and recorded settings are in
[`reproduction/frozen-study.json`](../reproduction/frozen-study.json) and
[`frozen-source-manifest.json`](../reproduction/frozen-source-manifest.json).
The frozen runtime source is distributed as the separately hashed release asset
listed in [`release-assets.json`](../release-assets.json).

That snapshot uses the original `webolt` package and `WEBOLT_` environment
prefix. The maintained package is `humanoid_agent` with `HUMANOID_AGENT_`.
Historical recording/schema identifiers are unchanged. Use the frozen snapshot
for an attempt under the original source identity; running today's package is
a new source revision, even though the retained records, decision schemas and
summary values were checked after the rename.

<!-- markdownlint-disable MD013 -->

| Setting | Recorded value |
| --- | --- |
| Task | Four-cabinet storage round, `drawer-shutdown-v2` |
| Conditions | none-open; open-A, B, C, D, A+D, B+C, A+B |
| Models | GPT-6.1 Sol, GPT-6 Astra, Claude Opus 5.5, Gemini 4 Argon |
| Effort | Low and high for each model |
| Attempts | One per model, effort and condition; 64 total |
| Episode budget | 1800 s, physics continuing during inference |
| Request timeout | 120 s, capped by the remaining episode time |
| Output ceiling | 12,000 tokens |
| API service tier | Standard |
| Interface | Four head/wrist keyframes, native walking/posture, binary hand presets |
| Facing | Scene `leashed_facing: true`; command stays within 15° of measured heading |
| Observations | Onboard RGB-D, fixed external RGB, robot feedback, static map and history |
| Completion | Agent declaration followed by independent physical verification |
| Timing policy | `wall-clock-lag-v1`, 100 ms limit, violations retained and reported |
| Schedule seed | 20261025; this shuffles conditions, not provider sampling |

<!-- markdownlint-enable MD013 -->

The original condition-block order was B, D, none-open, A+B, A+D, A, C, B+C.
All starts, API failures, falls, timeouts and rejected declarations are
retained.
The schedule's seed is not a shared model or physics random seed.

## Controller and asset requirements

The runtime uses NVIDIA's unchanged native SONIC v1.1 C++/TensorRT deployment,
revision `45c03facbf848e0c743091b7d9c17952096ff908`, with its fixed encoder,
decoder and walking planner. The controller executable and weights are
hash-checked by `runtime/deployment.py`. The project does not train a task
policy
or replace SONIC's learned controller.

The study environment used Isaac Lab 3.0 source revision
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Newton 1.6.1 and Warp 1.17.0.
The installed component versions were `isaaclab` 28.0.0,
`isaaclab-newton` 9.1.4 and `isaaclab-ov` 5.1.5. Physics ran at 200 Hz,
SONIC at 50 Hz. See [architecture](architecture.md) for sensor, gain and frame
conventions, and [task configuration](storage-round.md) for drawer contact,
friction, damping and floor support.

Obtain the G1/Dex3 description, SONIC weights and native dependencies from their
upstream releases. Download NVIDIA's Simple Room and Sektion cabinet assets and
the pinned SimReady lockers. `configs/sonic_constants.json` retains the recorded
controller constants; the robot URDF is supplied by upstream and hash-checked.
These downloads and their licenses are separate from this repository.

The CLI accepts explicit paths for `--isaac-workspace`, `--native-workspace`,
`--system-libraries` and `--sonic-constants`. Defaults in the deployment
examples
refer to prepared Linux workers; they are not a simulator installation tool.
`python -m humanoid_agent.tools.prepare_sonic_native_runtime WORKSPACE` prepares
the pinned native controller dependencies. Isaac Lab/Newton and licensed assets
must already be available separately.

## Maintained-source retry policy

New maintained-source jobs default to a 300 s request timeout and
`bounded-api-retry-v1`: at most one recorded retry of the unchanged request,
after one second, for connection/timeouts, HTTP 408/409/429/5xx, or an explicit
HTTP 400 `content_blocked` error. Other invalid requests and authentication
failures are not retried. A retry cannot extend the episode deadline, and
SDK-internal retries stay disabled so every attempt is counted in the record.
No prompt rewriting or service-tier fallback is performed.

This policy is for follow-up experiments. The initial 64 episodes used a 120 s
timeout and no retries; their records and success counts are unchanged. The
retry policy is included in new episode protocols to prevent accidental
aggregation with the initial sweep.

## Run the storage round

Install the maintained application with Python 3.12:

```bash
uv sync --locked
uv run humanoid-agent --help
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` securely outside
source control. For a new maintained-source attempt on a prepared host:

```bash
uv run humanoid-agent --role episode --episode-kind model \
  --provider astra --reasoning-effort low \
  --mode hybrid --wall-seconds 1800 \
  --api-request-timeout-seconds 300 --api-service-tier standard \
  --api-retry-policy bounded \
  --model-max-output-tokens 12000 --diagnostic-camera on \
  --task-scene configs/shutdown-storage-open-A-D.json \
  --isaac-workspace /path/to/isaac-workspace \
  --native-workspace /path/to/native-workspace \
  --system-libraries /path/to/system-libraries \
  --sonic-constants configs/sonic_constants.json \
  --output artifacts/astra-low-open-a-d-NEW
```

For the original frozen source, extract its release asset into a fresh
workspace and install its lockfile. Use `uv run webolt`, set the request timeout
to 120 s, and omit `--api-retry-policy` (the frozen snapshot has retries disabled
and does not accept that option). The other numerical settings above apply.
The frozen package supports `--mode hybrid`. Set `WEBOLT_SIMREADY_ROOT` rather
than the maintained package's `HUMANOID_AGENT_SIMREADY_ROOT` for the locker
asset root. Use a fresh output directory for every attempt.

Use a separate scripted `native-controller-check` before model calls; it is
privileged feasibility evidence, not a model result. The post-cleanup reference
reached a push limit at D and did not complete the round, so a fully successful
scripted reference is not claimed for that check.

## Hardware and scheduling

The original study used A6000 GPUs and fourteen CPU cores per active episode.
At most four episodes ran concurrently: one on a single-GPU host and three on
a four-GPU host. More concurrent load raised physics lag. Keep rendering and
other GPU workloads outside a timed comparison window. New hardware or changed
scheduling requires its own timing check.

The [SkyPilot examples](../configs/skypilot/lambda-a6000.yaml) use containerized
managed jobs on prepared SSH workers. Adapt their infrastructure and asset paths
to your own machines. Preserve one episode per GPU and non-overlapping CPU
masks. Deploy source and retrieve results with rsync; do not rely on an
interactive SSH session to own an experiment.

## Analyze and render

When complete records are available locally:

```bash
uv run humanoid-agent --role analyze \
  --episode-directory artifacts/EPISODE --output artifacts/analysis/EPISODE
uv run humanoid-agent --role render \
  --episode-directory artifacts/EPISODE --output artifacts/replay/EPISODE
```

Rendering restores recorded poses and does not run task physics or model calls.
It is a presentation derivative. See [records](records.md) for MCAP evidence,
protocol validation and independent execution/physical/timing outcomes.

The website reads versioned result summaries and a separate recording manifest.
New results can be added without replacing the initial sweep. Do not silently
mix a selectively repeated subset with the original condition coverage.
