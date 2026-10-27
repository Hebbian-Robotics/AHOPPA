# Initial cabinet inspection results

This release contains the initial 64-episode comparison, with results obtained
in the last few weeks using frontier-model releases. Four models each
attempted eight conditions at low and high reasoning effort, once per cell, with
an 1800 s episode budget. Seven conditions require drawer closure; one condition
starts with all drawers closed. Every scheduled start is retained in [the
episode ledger](../data/episodes.csv).

## Outcomes and denominators

The primary success rate uses all seven scheduled task attempts per model and
reasoning setting, including API-error starts. Astra succeeded in 6/7 at each
setting, Sol in 4/7 and 3/7, Opus in 2/7 and 3/7, and Gemini in 0/7 at both
settings. These conditions begin with at least one open drawer; the condition
with all drawers closed is a separate control.

Gemini has recorded task outcomes for four low-effort and five high-effort
attempts. Three low-effort and two high-effort attempts ended in API errors;
one further API error affected the all-closed control. Provider filters rejected
four calls and two reached a client timeout. The chart shows API errors as their
own category. Physical completion and inspection are counts from episodes with
recorded task outcomes, without inferred measurements for API-error starts.

All 43 drawer closures were by hand, and no initially closed drawer was
disturbed. Among 51 completed episodes starting with open drawers there were 15
falls, including one before any model command, eight time limits and four
completion claims rejected by the physical evaluator. Success requires all
drawers closed and settled, an upright hold at home and the model's completion
declaration. Hand attribution and disturbance are measured separately.

The CSV and aggregate JSON retain both scheduled starts and completed episodes,
which provide the denominator for the reported end-to-end success rate. The
original observations and outcomes are unchanged by this reporting choice.

## Reasoning effort and falls

Eleven of Astra's twelve successful task attempts viewed all four drawer fronts.
Higher effort increased decision latency roughly two to four times without
consistent success gains. Sol high closed 9/10 initially open drawers versus
4/10 at low effort, reaching physical completion in 5/7 attempts versus 4/7.
Two physical completions went undeclared before a timeout or fall, leaving 3/7
successes at high effort versus 4/7 at low.

Of eight all-closed controls, three succeeded, four fell and one had an API
error. Across all conditions, 16/19 falls had no recorded drawer contact;
other collisions remain possible. Eleven of 18 post-command falls ended in
zero-speed endpoint holds with retained pose and facing targets. Prolonged
holds during inference are a possible contributor to instability; the records
do not establish a cause. [The fall audit](figures/fall-analysis.json) lists the
episodes and the fields used to reproduce these counts from the raw records.

## Cost and elapsed time

Astra high matched low's success at 18% lower mean API cost, with a larger
cache-read share despite more output tokens. Sol high's successful attempts
cost $1.59 on average versus $1.96 at low, but including failed-attempt spending,
Sol low spent less per success: $3.00 versus $4.06. Cost per success divides total
task-attempt spending by the number of successful attempts.

The paper and page separate means over all seven task attempts from means over
successful attempts only. Gemini has no successful-attempt cost or duration;
a dash and asterisk mean no observed success. Failed attempts still incur costs
and can terminate early. Per-decision latency is reported separately from full
episode duration and physics lag. See [the cost and timing method](api-costs.md)
for source evidence, provider reconciliation and minor accounting gaps.

## Timing and scope

Only 17/58 completed episodes, including seven task successes, met the recorded
100 ms physics-lag limit. Median peak lag was 192 ms. The results include episodes
that exceeded this timing limit. All episodes ran in simulation.

There is one attempt per model, setting and condition, on one layout. The
conditions test coverage of different drawer states; they do not provide
repeated measurements of the same state. Successful Astra episodes took 615 to
1492 seconds and combined route choice, visual inspection, hand contact and a
return home through online commands to a fixed controller. Repetitions,
older-model baselines and hardware trials are needed to measure reliability,
capability gains and transfer. Identify later repetitions separately from this
comparison.

The [raw-record dataset](https://huggingface.co/datasets/hebbianrobotics/AHOPPA)
provides the retained MCAP traces, model interaction records and numerical logs
for exactly these 64 starts. Its dataset card documents omitted files and
metadata redactions. Episode-level summaries remain versioned in this repository;
selected replay/request-image examples are also distributed as a media release.
