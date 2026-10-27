---
pretty_name: AHOPPA initial 64-episode comparison
language:
  - en
license: other
license_name: ahoppa-mixed
license_link: LICENSE.md
tags:
  - robotics
  - humanoid
  - simulation
  - multimodal
  - agents
  - evaluation
size_categories:
  - n<1K
configs:
  - config_name: episodes
    data_files:
      - split: test
        path: data/episodes.csv
---

# AHOPPA: initial 64-episode records

Agentic Humanoid Operation through Perception, Planning, and Action.

Kingston Kuan¹ · Samuel J Chew² · Brandon Ong¹

¹ Hebbian Robotics · ² Carnegie Mellon University

This project is not endorsed by Carnegie Mellon University.

[Code and paper](https://github.com/Hebbian-Robotics/AHOPPA) ·
[Project page](https://hebbian-robotics.github.io/AHOPPA/) ·
[Task and scoring](https://github.com/Hebbian-Robotics/AHOPPA/blob/main/docs/storage-round.md)

This dataset contains the retained records for exactly the **64 scheduled starts
reported in AHOPPA's initial comparison**: four models, two reasoning settings,
eight drawer conditions and one attempt per cell. It includes unsuccessful
episodes and all six API-failed starts. It contains no pilot, development,
controller-check or follow-up experiment records.

The task uses a simulated Unitree G1 to inspect four cabinet drawers, push open
drawers closed by hand, leave closed drawers alone and return home before
declaring completion. Models operate through visual feedback and numerical
head/wrist and locomotion commands to a fixed SONIC controller. These results
were obtained in the last few weeks following frontier model releases. No
hardware experiments are included.

## Downloads

<!-- markdownlint-disable MD013 -->

| Path | Contents |
| --- | --- |
| `data/episodes.csv` | The public ledger, one row per scheduled start |
| `data/summary.json` | Public aggregate and episode-level results |
| `data/provenance.json` | Provenance of the public results |
| `episode-summaries.tar.zst` | Small JSON records for all episodes, including outcomes, timing, configurations and source manifests where recorded |
| `episodes/<episode>.tar.zst` | One raw-record archive per episode |
| `metadata/archives.csv` | Episode archive paths, compressed sizes and SHA-256 hashes |
| `metadata/files.jsonl` | Original SHA-256 and size of every included regular episode file |
| `metadata/cohort-*.json` | The sixteen original frozen comparison manifests |
| `metadata/schedule.json` | Frozen condition schedule, with the host workspace path removed |
| `metadata/launches.jsonl` | Launch ledger, with node addresses removed |
| `metadata/release.json` | Release scope and frozen source identity |
| `metadata/omissions.json` | Explicit list of omitted files and metadata changes |
| `metadata/symlinks.json` | Original internal symbolic links retained in the archives |
| `SHA256SUMS` | Checksums for the release files |

<!-- markdownlint-enable MD013 -->

The archives use Zstandard compression. Install `zstd` before extracting them.
Download only the episodes needed for an analysis. For example:

```bash
hf download hebbianrobotics/AHOPPA --type dataset \
  --include 'data/*' --include 'metadata/*' --include 'SHA256SUMS' \
  --include 'episode-summaries.tar.zst' --local-dir ahoppa-records
cd ahoppa-records
tar --zstd -xf episode-summaries.tar.zst
```

To download and extract one complete episode:

```bash
hf download hebbianrobotics/AHOPPA --type dataset \
  --include 'episodes/storage-cmp3-astra-low-open-a-d-r1.tar.zst' \
  --local-dir .
tar --zstd -xf episodes/storage-cmp3-astra-low-open-a-d-r1.tar.zst
```

For an exact paper reference, pin the dataset commit or the `initial-64-v1` tag
using `hf download --revision initial-64-v1`. The checksum file covers all
release payload files other than itself. If the entire release is downloaded,
run `sha256sum -c SHA256SUMS`. For a partial download, verify the selected
archive against its row in `metadata/archives.csv`.

Each archive extracts to `episodes/<episode>/`. Keep that directory structure.
To use the frozen cohort manifests, copy `metadata/cohort-*.json` to the
extraction root; their `episodes/...` paths will then resolve. A full cohort
requires all four corresponding episode archives. Extracting summaries alone
does not provide the full evidence required by the cohort validator.

## Episode contents

- `episode.json`: terminal execution status, independent task outcome and timing
  classification.
- `inputs/`: frozen scene, facility map, controller constants, project settings
  and lockfiles, subject to the robot-description omission below.
- `source/` and `source-manifest.json`: the captured runtime source and hashes.
- `physics/records.mcap`: indexed physical trace, model observations, command
  ledger and terminal evidence.
- `shared-physics/`: owner timing, controller logs and native numerical state
  and command logs.
- `operator/`: exact recorded request payloads, prompts, input images and arrays,
  provider responses, tool rounds and timing, where execution reached them.

Some records are stored inside MCAP attachments rather than as loose files.
Use the repository's recording readers to access them. Failed starts may lack
records that startup never produced. The dataset preserves that absence rather
than inventing outcomes. Recorded provider reasoning summaries are provider
outputs, not raw chains of thought or causal explanations of model decisions.

Historical `webolt` identifiers and local file paths are retained inside episode
records to preserve their original bytes and hashes. Those paths describe the
original runtime and are not paths to use on a new machine. Internal symbolic
links stay within their episode directory. The artifacts contain JSON, MCAP,
PNG, NPY, CSV and source files; interpreting source is separate from running it.

## Evidence integrity and omissions

Every included regular episode file is checked against its original SHA-256
while packaging. Model interaction records and MCAP recordings are distributed
without content edits. The original local evidence is unchanged.

The release omits:

- Generated Python bytecode and `__pycache__` directories.
- Upstream robot-description `.urdf` files, including the generated articulated
  description. These remain identified by the original manifests and runtime
  configuration but are not relicensed by this dataset.
- Infrastructure launch/deployment YAML and unrelated local experiment folders.
- Generated offline replays, available separately through the public project's
  hash-pinned media release. Recorded model images remain in the raw archives.

The schedule's workspace path and launch ledger's node addresses are removed in
the separate public metadata copies. `metadata/omissions.json` lists every
omitted episode file and these metadata transformations. Other historical local
paths in episode records remain unchanged.

Full validation of retained input hashes requires restoring the original
`inputs/robot.urdf` from the separately obtained upstream robot assets. Its
expected SHA-256 is
`6e107391d015ab27438026bb953b21ef8fa6a02305e65393985908941d7fd98d`.
Do not substitute a newer robot description. Robot asset bundles, meshes,
controller weights, installed simulator environments and API credentials are
not supplied. See the public repository's
[reproduction guide](https://github.com/Hebbian-Robotics/AHOPPA/blob/main/docs/reproduction.md).

For analysis of the original experiment, use the captured source and lockfiles,
or the hash-pinned frozen-runtime source asset linked in
`metadata/release.json`. The maintained runtime can differ from the recorded
study. No simulator, GPU or model API access is needed simply to read retained
records; new experiments require those external dependencies.

## Interpretation and limitations

Success requires the model's completion declaration and the independent
physical criterion. Infrastructure completion, physical task outcome and timing
validity are distinct quantities. The reported success denominators retain all
seven scheduled task attempts per model and effort, including API errors;
the all-closed condition is a separate control.

Only 17 of 58 completed episodes met the recorded 100 ms physics-lag limit.
Timing-invalid episodes remain included and labelled. The experiment uses one
fixed layout and one attempt per model/effort/condition. Frames, model decisions,
controller ticks and drawers within an episode are not independent trials.
The records support analysis of these observed runs; repeated experiments are
needed to estimate reliability. Simulation does not establish hardware
performance or safety.

## Cost and timing

The [paper and project page](https://hebbian-robotics.github.io/AHOPPA/) report
estimated API cost, full episode duration and model-decision latency for all
seven scheduled task attempts and for successful attempts only. The all-closed
control is separate. A dash means no successful attempt was observed. The
website adds an asterisk to that dash; the paper instead uses a cost asterisk
for unresolved usage.
Costs use recorded and provider-reconciled token usage at published standard
rates, rather than invoice amounts. Astra high was cheaper in these runs because
cached reads covered about 50% of its input, compared with 29% at low effort.

The [cost and timing method](https://github.com/Hebbian-Robotics/AHOPPA/blob/main/docs/api-costs.md)
provides the definitions, pricing policy, recovery evidence and reproduction
commands. Provider reconciliation is published separately in the repository;
these raw archives preserve the original responses, including missing responses
from interrupted calls. Eleven Gemini calls remain unpriced. Full episode
duration includes inference while physics runs and ends at recorded termination.
Decision latency is the median of per-attempt medians for completed decisions.
The paper reports all task attempts; the page also shows successful attempts
separately. Physics lag is a separate simulator timing quantity. Cost per success
in the paper includes spending on failed task attempts and divides total task
spending by the success count, unlike mean cost conditional on success.

## Attribution and licenses

Original results, numerical records, prompts and author-created presentation
content are released under CC BY 4.0. Original project code retains the MIT
license. Third-party assets and their depictions retain applicable upstream
terms; no rights to upstream weights, geometry or software are granted by the
dataset license. See [LICENSE.md](LICENSE.md),
[CODE-LICENSE.txt](CODE-LICENSE.txt) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Credit AHOPPA, Kingston Kuan, Samuel J Chew and Brandon Ong. Cite the paper
linked from the public repository and identify the dataset revision or
`initial-64-v1` release when reporting analyses.
