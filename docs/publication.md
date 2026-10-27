# Public material consistency

The repository, project page, paper and raw-record dataset describe the same
initial 64-start comparison. Editorial changes use normal commits. Preserve
`review-v1`, `initial-64-v1`, raw archives, recorded outcomes and source hashes.
The frozen runtime release reproduces the study; maintained code can differ.

## Shared sources

<!-- markdownlint-disable MD013 -->

| Material | Source |
| --- | --- |
| Paper download and manuscript | `paper/paper.tex`, `paper/paper.pdf`; detailed companion: `paper/results-table.tex` |
| Project page | `web/`, shared paper PDF and `docs/figures/` |
| Initial outcomes | `data/episodes.csv`, `data/summary.json`, `data/provenance.json` |
| Cost and timing | `docs/figures/api-costs.json`, `docs/api-costs.md` |
| Website outcome and latency charts | `docs/figures/storage-results.json` |
| Hugging Face dataset card | `data/dataset-card.md`, published as dataset `README.md` |
| Raw evidence | Hugging Face `episodes/`, `metadata/` and `SHA256SUMS` |
| Replay and frozen source releases | Hash-pinned assets in `release-assets.json` |

<!-- markdownlint-enable MD013 -->

## Verified comparison

The consistency review checked all 64 ledger rows against the summary, all
16 cohort manifests against the episode identifiers, and the 64 hosted archive
sizes and LFS SHA-256 values against the dataset archive manifest. The dataset's
CSV, summary and provenance are byte-identical to the repository copies.
The original summary SHA-256 is
`affe4a2076d72cd3bb059c289bda17b18f1b0d801b2d7dcc343fbf52678fa1fe`.

Each model and effort has seven scheduled task attempts and one separate
all-closed control. Success includes API-error starts in its denominator.
The paper table and website charts match the summary and cost report. API
cost and episode duration use means; decision latency uses the median of
per-attempt medians for completed decisions. Cost and episode time distinguish
all task attempts from successful attempts.
The paper keeps all-attempt decision latency; the website additionally reports
successful-attempt latency. The inline paper table is the compact presentation;
`paper/results-table.tex` remains a detailed generated companion. Costs are
standard-rate estimates,
including uniquely reconciled usage, rather than itemized invoice amounts.

All 43 drawer closures were by hand, with no initially closed drawer disturbed.
Of 58 completed episodes, 17 met the physics timing limit: seven successful
task attempts, two successful controls and eight unsuccessful episodes. Timing
validity remains separate from task success. The main paper text occupies two
pages, with references beginning on page three.

The overview, paper sequence, website observations and featured replay use
the evaluated Astra-low `open-A-D` episode. Figure source hashes match the
raw dataset's file manifest. The final selected observation is at 942 s;
the completion declaration follows at 955 s. Offline replay frames are
presentation derivatives, distinct from the original model input images.

## Updating and publishing

After a factual edit, review the README, results guide, paper, website and
dataset card together. Rebuild the paper, check its two-page body, then build
the website so its Paper button contains that PDF. Run `pnpm verify` and use
[the website publication procedure](website.md) to deploy. Fetch the live
page and check the served paper and figure hashes against the repository.

Publish `data/dataset-card.md` as the Hugging Face dataset's `README.md` and
update only that entry in `SHA256SUMS` when the card changes. Verify the
published card, checksum and shared data files after upload. Keep the dataset's
original release tag and payload hashes unchanged. Provider reconciliation and
pricing derivatives live in this repository; they do not replace raw responses
inside the frozen episode archives. Update GitHub release descriptions when
resource availability changes while retaining the pinned release assets.
