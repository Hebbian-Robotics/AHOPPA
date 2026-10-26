# Research context

AHOPPA studies online model decisions through a fixed humanoid controller. It
does not introduce drawer closing, a new gait policy or a new skill-learning
method. The [bibliography](references.bib) contains the manuscript's sources.

Relevant agentic-robotics approaches include
[SayCan](https://arxiv.org/abs/2204.01691), which grounds planning in skill
affordances; [Code as Policies](https://arxiv.org/abs/2209.07753), which
generates
robot programs; [Inner Monologue](https://arxiv.org/abs/2207.05608), which uses
embodied feedback; and [VoxPoser](https://arxiv.org/abs/2307.05973), which
builds
spatial value maps.
[PIVOT](https://proceedings.mlr.press/v235/nasiriany24a.html)
selects visual action proposals, while
[ReKep](https://proceedings.mlr.press/v270/huang25g.html) optimizes visual
keypoint constraints. These studies motivate feedback and spatial interfaces;
their published scores are not matched baselines for this task.

[HomeBody](https://tml.stanford.edu/homebody/) uses spatial memory and humanoid
skills. [SONIC](https://nvlabs.github.io/GEAR-SONIC/) supplies the fixed
tracking
and locomotion foundation here. AHOPPA's model directly updates numerical
head/wrist references and walking commands without a task skill library.

The task combines cabinet inspection, navigation and hand contact across a room.
The contribution is the observed behavior of online models under this common
interface, with independent physical scoring, retained failure categories and
reported timing limitations. Simulation results do not establish hardware
performance or safety.
