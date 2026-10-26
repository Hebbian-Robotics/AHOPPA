# Contributing

Use Python 3.12 with uv, and pnpm for the project page. Run `pnpm verify` before
sending a change; it checks Python and TypeScript, formatting, local behavior
tests and Markdown. It launches no GPU jobs or paid model requests.

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
pnpm check
pnpm verify
```

Use Biome for web files and markdownlint for Markdown. Do not add Prettier.
Test observable behavior; avoid tests that only reproduce a mock or a vendor
library. The optional `pnpm test:assets` suite requires the released robot
assets and fails when they are absent.

Keep the implementation in `src/humanoid_agent/`. The `humanoid-agent` episode
owner manages physics, observations, model calls and recording. Analysis,
rendering and publication are separate explicit commands and must not determine
whether an experiment has a valid terminal record.

Completed experiment records are immutable evidence. Changes to a scene,
prompt, evaluator, model setting or timing policy need a newly identified
protocol. Do not overwrite failed episodes or selectively retry outcomes. Keep
credentials, downloaded assets, checkpoints, raw recordings and environments out
of source control.
