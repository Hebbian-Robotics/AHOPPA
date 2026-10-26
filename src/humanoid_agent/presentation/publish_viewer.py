"""Publish the compiled private viewer without touching experiment artifacts."""

from __future__ import annotations

import logging
import shutil

from cyclopts import App

from humanoid_agent.artifact_io import atomic_write_text
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

ROOT = PROJECT_ROOT


def main() -> None:
    distribution = ROOT / "web" / "dist"
    site = ROOT / "site"
    # Keep older hashed assets available to tabs opened before a rebuild.
    shutil.copytree(distribution / "assets", site / "assets", dirs_exist_ok=True)
    atomic_write_text(site / "index.html", (distribution / "index.html").read_text())
    archive = site / "archive"
    if archive.is_dir():
        atomic_write_text(
            archive / "index.html",
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="0;url=../">'
            "<title>AHOPPA</title></head><body>"
            '<a href="../">Continue to the AHOPPA project page</a></body></html>\n',
        )
    logger.info("Published viewer to the existing private Tailscale site.")


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
