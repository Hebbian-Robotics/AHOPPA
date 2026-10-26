"""Identify owned local processes across sessions without confusing reused PIDs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["webolt-process-v1"] = "webolt-process-v1"
    pid: int = Field(gt=0, strict=True)
    created_time_seconds: FiniteFloat = Field(gt=0)
    boot_id: str = Field(min_length=1)

    def matching_process(self) -> psutil.Process | None:
        if self.boot_id != current_boot_id():
            return None
        try:
            process = psutil.Process(self.pid)
            if (
                process.create_time() == self.created_time_seconds
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                return process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        return None


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_identity(pid: int) -> ProcessIdentity | None:
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return ProcessIdentity(
            pid=pid, created_time_seconds=process.create_time(), boot_id=current_boot_id()
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def process_record(pid: int) -> dict[str, object]:
    identity = process_identity(pid)
    if identity is None:
        raise ProcessLookupError(f"Cannot identify live process {pid}")
    return identity.model_dump(mode="json")
