"""Workspace paths are configuration, independent of the installed package location."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("HUMANOID_AGENT_PROJECT_ROOT", Path.cwd())).resolve()
