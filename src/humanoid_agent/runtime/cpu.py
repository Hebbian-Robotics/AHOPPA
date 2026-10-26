"""Explicit per-owner CPU allocations propagated to every child process."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humanoid_agent.cli import EpisodeOptions


def parse_cpu_set(cpu_set: str) -> set[int]:
    """Accept taskset's comma-separated CPU indices and inclusive ranges."""
    if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", cpu_set):
        raise ValueError(f"Invalid CPU allocation: {cpu_set!r}")
    indices: set[int] = set()
    for section in cpu_set.split(","):
        endpoints = section.split("-")
        first = int(endpoints[0])
        last = int(endpoints[-1])
        if last < first or last > 65535:
            raise ValueError(f"Invalid CPU range: {section!r}")
        indices.update(range(first, last + 1))
    return indices


@dataclass(frozen=True)
class CpuAllocation:
    physics: str = "2-5"
    operator: str = "0-1"
    sdk: str = "0-1"
    io: str = "0-1"
    controller: str = "6-9"
    camera: str = "10-13"

    def __post_init__(self) -> None:
        for cpu_set in asdict(self).values():
            parse_cpu_set(cpu_set)

    def environment(self) -> dict[str, str]:
        return {
            f"HUMANOID_AGENT_{role.upper()}_CPU_SET": cpu_set
            for role, cpu_set in asdict(self).items()
        }

    @classmethod
    def from_options(cls, arguments: EpisodeOptions) -> CpuAllocation:
        return cls(
            physics=arguments.physics_cpu_set,
            operator=arguments.operator_cpu_set,
            sdk=arguments.sdk_cpu_set,
            io=arguments.io_cpu_set,
            controller=arguments.controller_cpu_set,
            camera=arguments.camera_cpu_set,
        )

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> CpuAllocation:
        environment = os.environ if environment is None else environment
        defaults = cls()
        return cls(
            **{
                role: environment.get(f"HUMANOID_AGENT_{role.upper()}_CPU_SET", default)
                for role, default in asdict(defaults).items()
            }
        )
