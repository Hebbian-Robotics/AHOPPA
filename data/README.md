# Initial result data

`episodes.csv` contains all 64 scheduled starts, including six API failures.
`summary.json` contains their aggregate statistics and episode-level summaries.
These are derived from the retained frozen records; they are not substitutes
for raw physics traces. Machine-specific path prefixes are removed from the
public summary. The original source summary's SHA-256 is recorded in
`provenance.json`.

The paper's primary success rates use all seven scheduled task attempts per
model and effort, including API-error starts. The all-closed control is reported
separately. Physical outcome counts use episodes with recorded task outcomes.
The CSV retains every start. API rejections and client timeouts remain separate
from falls, time limits and incorrect completion claims.

Physical outcome cells are blank for API-failed starts where the summary does
not establish that quantity. The condition still identifies the intended
initial drawer state.

All quantities are episode-level observations on one fixed layout. Video
frames, controller ticks, decisions and drawers within an episode are not
independent repetitions. The initial sweep has one attempt per condition.

## Availability

- Included: maintained source, scene configurations, paper, all 64 summary rows
  and aggregate statistics.
- Separate media release: selected synchronized replay videos, original model
  input images and plan timelines.
- Separate source release: the exact captured study runtime and its lockfile.
- Separate [raw-record dataset](https://huggingface.co/datasets/hebbianrobotics/AHOPPA):
  MCAP recordings, exact recorded requests, input images, provider responses,
  numerical controller logs and frozen cohort manifests for exactly these
  64 starts, including unsuccessful and API-failed starts. No additional
  experiments are included. See its dataset card for downloads and omissions.
- Not included: credentials, installed environments, model weights or
  NVIDIA/Unitree asset bundles. The raw-record release also omits upstream
  robot-description files and generated Python caches.

We train no new weights. Obtain SONIC and the simulation assets from their
upstream releases under their own terms. Model API access is required for new
agent episodes.

These result tables are licensed under CC BY 4.0. Replays credit the upstream
scene and robot assets identified in the task specification.
