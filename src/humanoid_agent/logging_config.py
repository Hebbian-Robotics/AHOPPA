"""Consistent UTC diagnostics on stderr; machine-readable results stay on stdout."""

import logging
import os
import time


def configure_logging() -> None:
    level_name = os.environ.get("HUMANOID_AGENT_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if level is None:
        raise ValueError(f"Invalid HUMANOID_AGENT_LOG_LEVEL: {level_name}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for library in ("httpx", "httpcore", "openai", "anthropic", "google_genai"):
        logging.getLogger(library).setLevel(logging.WARNING)
