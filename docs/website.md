# Project page

The React source is in `web/`. The page uses the same versioned paper and result
figures as the repository. Videos, plans and original input images are supplied
by the separately hashed media asset in `release-assets.json`; they are not
stored in source history.

Build for GitHub Pages:

```bash
pnpm install --frozen-lockfile
pnpm check
HUMANOID_AGENT_SITE_BASE=/AHOPPA/ pnpm exec vite build
```

Extract the media asset into `web/dist/`, retaining its `transfer/` directory.
The release asset contains only the reviewed recording records and their media;
it does not include historical site directories, raw traces or provider logs.

The manual GitHub Actions workflow builds the page on a Blacksmith CPU runner,
verifies the media archive's SHA-256 and prepares a Pages artifact. Publishing
is a separate explicit workflow input. In repository settings, select GitHub
Actions as the Pages source.

For later results, update the versioned summary and paper, prepare a new media
asset and update its URL and digest. Preserve the initial CSV/summary or move it
under a named release rather than replacing evidence without a version. The
workflow never runs simulations or model calls.

## Presentation

The page retains the centered Nerfies paper header, with a sharp vector overview
immediately below the introduction, followed by the featured Astra replay.
Additional recordings expand on request. The plan, decision timeline and
original model inputs remain available in each recording's inspector. Recording
details contain the stable video ID and direct link; shared links open their
containing sections.

The task appears beside the facility map, followed by three original
observations of inspection, hand contact and return home. Their hashes and
timestamps are in `figures/storage-observations/provenance.json`. These are
unmodified inputs already selected for the paper episode strip. Results and
sampling limits precede the detailed charts; the system overview stays at the
top of the page. The featured replay's preview is the original external
observation near the return home; playback shows synchronized views.

The design review used the official [ReKep](https://rekep-robot.github.io/),
[SONIC](https://nvlabs.github.io/GEAR-SONIC/) and
[HomeBody](https://tml.stanford.edu/homebody/) pages as references for prominent
demonstrations and capability-oriented sections.

The estimated-cost table reads `figures/api-costs.json`, the same data used to
generate the paper's paired cost, episode duration and decision latency columns.
All-attempt means include all
seven scheduled task attempts; successful-attempt means include only completed
tasks. A dash and asterisk indicate no observed successful attempt. The
completion chart separately pairs model-decision latency over all and successful
task attempts. Physics lag is separate. Follow [the cost and timing method](api-costs.md)
for aggregation and recovery. Small accounting gaps stay in expandable details;
public evidence omits account identifiers. The author
order is Kingston Kuan, Samuel J Chew, then Brandon Ong. Kingston and Brandon
are at Hebbian Robotics; Samuel is at Carnegie Mellon University. His name links
to his supplied LinkedIn profile. The affiliation row includes CMU's primary
red square university logo from its
official homepage alongside the Hebbian Robotics logo;
the website license notice records its source and trademark status. A dagger
beside CMU links to the non-endorsement note at the end of the page content,
immediately before the footer. The footer sits outside the centered main
container, so its background spans the viewport.

## Paper and page consistency

The paper source and PDF, project-page components, figures and result summaries
share the same recent-results framing, counts and interpretation. A prose update
requires a rebuilt PDF and a new site build. Keep the hash-pinned media asset
when its recordings and metadata are unchanged. Review the README and results
guide for the same factual changes.

Describe these results as obtained in the last few weeks using frontier-model
releases. Keep exact experiment dates in the retained records.
Introduce the research problem before the task. Describe the observed agent
behavior directly: inspection routes, drawer pushes and return-home decisions
through numerical commands to a fixed controller. Reliability and gains over
older models require further experiments.

Follow [Google's developer documentation style guide][google-style] for active
verbs, short sentences, specific subjects and consistent terms. Replace
rhetorical questions, filler, em dashes and formulaic contrasts with concrete
claims. Keep the simulation scope, limited repetitions, API errors and timing
limits visible beside the results.

[google-style]: https://developers.google.com/style/highlights

The Raw data button uses an original bucket SVG. The paper prints the project
URL directly and pairs the shared vector overview with a five-frame episode
sequence after its abstract. Both figures use retained inputs from the evaluated
Astra-low `open-A-D` episode. Source hashes are recorded beside the figures;
the strip uses larger labels at paper width. Its last observation is at 942 s,
preceding the completion declaration at 955 s. The table retains the primary
numerical comparisons; the page includes fuller charts and episode observations.
