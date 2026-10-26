# Records and timing

[Documentation guide](README.md) · [Operations](reproduction.md) ·
[Research protocol](reproduction.md)

## Episode evidence

The package is `humanoid_agent`; retained schema, MCAP and protocol names keep
their original identifiers. Renaming those evidence identifiers would change
the record contract rather than just the package.

Every started episode belongs in the experiment ledger. The parent writes a
terminal record for normal completion, cancellation or infrastructure failure.
Analysis, replay and publication are not prerequisites for a valid record.

<!-- markdownlint-disable MD013 -->

<!-- markdownlint-disable MD013 -->

| Location in an episode | Contents |
| --- | --- |
| `inputs/` | Scene, facility map, robot URDF, controller constants, project configuration, recording requirements and both lockfiles |
| `source/`, `source-manifest.json` | Frozen Python source and its hashes |
| `physics/records.mcap` | Indexed trace, native publication receipts, admission ledger, controller bases, observations and terminal evidence |
| `physics/` | Configuration, `task-outcome.json`, `timing.json`, `io-summary.json`, exported observations and live projections |
| `operator/` | Model configuration, exact request images and prompts, native responses, tool rounds and timing |
| `commands/` | Identity used to derive the private operator IPC socket; no command files are exchanged |
| `shared-physics/` | Owner timing, camera startup and controller and camera worker records |
| `episode.json`, component logs | Validated terminal status, component exits, errors and cleanup evidence |

<!-- markdownlint-enable MD013 -->

<!-- markdownlint-enable MD013 -->

Controller weights and installed runtimes are external dependencies identified
by configuration and asset checks. A lockfile describes declared dependencies;
it does not prove installed-package compatibility.

[`experiment_records.py`](../src/humanoid_agent/experiment_records.py) owns the
`webolt-episode-v3` terminal schema, the `webolt-cohort-v3` manifest and
physical record validation. It rejects unknown schema versions, unsupported
protocols, malformed fields, nonfinite measurements, contradictory success
flags and inconsistent lag classifications. Startup errors can legitimately
lack a protocol or physical outcome; completed executions need consistent
evidence. Before publishing completion, the owner checks the physical result,
independent outcome and recording drain together; missing or contradictory
evidence produces a durable infrastructure-error terminal that keeps each
readable outcome and timing record.

The protocol binds source, scene, appearance, robot, constants, instruction,
facility map, wall budget, execution topology, GPU scheduling, sensor and
action interfaces, completion and balance rules, clock policy, API tier and
timeout, and both lockfiles. Native response records identify the actual model
when the provider supplies it. Keep provider configuration with the shared
protocol when freezing a comparison.

`execution_topology` (`single-robot` or `shared-physics`) and `gpu_scheduling`
(`driver-default` or `cuda-mps`) are required. A group keeps one canonical
episode per robot under `episodes/env_<index>/`, linking to the group's single
copy of `inputs/` and `source/`, with `group.json` for the owner's component
states. One robot's completion does not end another's episode; a shared
infrastructure failure stops the group and keeps each physical outcome. A
completed MPS episode also requires a successful GPU-service exit.

Executed-input rows come from SDK-worker acknowledgments after successful
native planner publications, with source command identity, the exact numerical
sample, sequence number and host-monotonic publication time. The owner stops
all streams and confirms each final receipt count before workers exit;
`sdk-worker-shutdown.json` keeps that audit. This boundary is later than the
physics enqueue and earlier than controller consumption or physical response.
Per-episode physical traces use episode-local coordinates for scoring audits
and replay; private render snapshots keep shared-world poses.

## Three independent outcomes

Execution status records whether infrastructure completed, was cancelled or
failed. Simulation validity records whether the measured maximum physics lag
stayed within the `wall-clock-lag-v1` limit of 100 ms. The physical task
outcome records the model's completion declaration and the independent
physical criterion, per drawer, as described in
[Storage round](storage-round.md#independent-scoring).

A completed process can contain an unsuccessful task, and a physically
successful trajectory can have invalid timing. An API error is an operational
failure under the configured budget, not evidence of deficient reasoning. Keep
all three dimensions rather than collapsing them into one success flag.

## Recorded reasoning summaries

Every decision requests the provider's reasoning summaries: `summary` for
OpenAI Responses, `thinking` display `summarized` for Anthropic Messages and
`thinking_summaries` for Google Interactions. `model.response.json` keeps the
native blocks and adds a normalized `reasoning_summary` list.

No provider returns a raw chain of thought. These are the model's own summaries
of the blocks it emitted; they may omit or smooth over steps and do not
establish why an action was taken. An empty `reasoning_summary` is an expected
result: providers return none for short reasoning passes, and Google returns
none at its `low` thinking level. Signature and encrypted reasoning fields stay
unchanged because continuation requests replay them. Summaries change the
request, so `inference_configuration` records `reasoning_summaries_requested`.

## Atomic and durable writes

The `webolt-recording-v1` protocol uses Foxglove SDK channels for MCAP evidence
and live visualization. Each background writer owns one recording and one SDK
context, so robots cannot leak records into each other's sinks, and
`telemetry-endpoint.json` names its loopback WebSocket endpoint. Sensor PNGs are
both exact attachments and standard compressed-image messages; depth and
optical-point arrays are exact attachments. Observation files stay available
during execution as immutable model inputs.

Typed records embed their Pydantic JSON schemas; scientific adapter
dictionaries use JSON object channels whose consumers validate the fields they
read. MCAP timestamps are host-monotonic writer reception times; measurement
and publication clocks are explicit payload fields. The pinned MCAP reader owns
indexed reads and checksum verification. Historical JSON is read-only;
archived evidence is never rewritten.

The writer flushes at durable evidence boundaries, synchronizes the recording
and finalizes indexes at shutdown, then stores the finalized SHA-256 in
`physics/io-summary.json` (`webolt-episode-io-v3`) with submitted and processed
operation counts. Completed records require a successful drain with equal
counts, no pending camera captures and a matching digest; missing or truncated
recordings fail closed even when an old JSON projection exists. The drain runs
after timed physics and does not extend the deadline. Recovery must keep the
original recording and produce a separate artifact.

[`artifact_io.py`](../src/humanoid_agent/artifact_io.py) rejects nonfinite
JSON and
writes through unique sibling temporary files. Live handoffs use atomic
replacement without `fsync`; terminal records use `durable=True`, syncing the
file through boltons and then its parent directory. Synchronization stays
outside active physics. This does not make the whole episode power-loss-safe.

Command admissions have their own identity ledger, persisted by the I/O worker
after in-memory admission: asynchronous persistence, not synchronous durability
at actuation. A crash before the drain acknowledgment cannot count as a
complete execution. The ledger does not implement checkpoint/restart or
authorize replay after a crash.

## Schema validation

Model decisions, planner publications, SDK packets and operator
acknowledgments use strict Pydantic schemas that reject unknown fields and
coerced values such as string booleans. Domain models are frozen, with
immutable numerical vectors and nested sequences that still serialize as JSON
arrays; copied updates are revalidated. Native SDK adapters receive ordinary
lists. The operator checks that status acknowledgments agree with their
admissions. Invalid decisions get correction feedback within the original
deadline, and sensor allowlists discard private evaluator fields.

## Bounded timing instrumentation

[`timing.py`](../src/humanoid_agent/runtime/timing.py) runs inside the physics
loop
with no profiler process. `physics/timing.json` summarizes completed active
ticks for sensor and SDK handling, actuation, physics update, evaluation and
trace, and full tick work before pacing sleep. Means, maxima and counts above
the physics period are exact; p95 and p99 are nearest-rank histogram upper
bounds with 5% buckets above one microsecond.

The last 128 spike records keep stage durations, clocks, lag, command ID and
coincident camera, admission, planner, log-scan and trace-write events; the
total spike count includes evicted records. A spike is tick work or lag growth
above one physics period and does not change the 100 ms validity policy. Stage
measurements are host-observed latency without added CUDA synchronization, and
a coincident event does not prove causation.

Private operator sidecars bind command, episode and observation IDs and
response hashes. The first executed-input record of each command includes
observation age at the first native planner send, split into snapshot-to-
operator receipt, preparation, model request with tool rounds, publication
delay and admission-to-send delay, on the shared-host monotonic clock. Missing
or mismatched sidecars are labelled and do not alter admission. Planner send is
not motor receipt or proof of physical response.

## Validate a comparison

Freeze one cohort manifest per condition and effort before launching, listing
every intended episode directory, its condition and provider, and the complete
`EpisodeProtocol`. Paths are relative to the manifest unless absolute. Use a
prepared controller check's protocol to fix the intended source and input
identity.

```bash
uv run python -m humanoid_agent.analysis.analyze_episode_cohort /path/to/cohort.json
```

[`analyze_episode_cohort.py`](../src/humanoid_agent/analysis/analyze_episode_cohort.py)
validates terminal and physical records and verifies retained source, inputs,
appearance and lockfile hashes. Duplicate paths, wrong providers, controller
checks and protocol or topology mismatches abort the comparison, and each
condition names one provider. Counts keep every declared start;
infrastructure, cancellation and invalid or unknown timing counts can overlap.
Failed or cancelled starts stay visible without physical records that startup
never produced.
[`summarize_shutdown_round.py`](../src/humanoid_agent/analysis/summarize_shutdown_round.py)
reads the verified records of several manifests and reports all started
episodes and, separately, the timing-valid ones.

The analyzer cannot prove manifest completeness or statistical independence.
Unversioned historical records need their frozen analyzers and stay outside
current cohorts; see the [frozen-study
description](reproduction.md#frozen-experiment) for evidence scope.
