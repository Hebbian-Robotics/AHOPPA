"""Own NVIDIA's CUDA MPS process for a shared episode group."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import BinaryIO

import psutil

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.process_lifecycle import ProcessIdentity, process_identity, process_record
from humanoid_agent.runtime.episode import stop_process


class CudaMultiprocessService:
    """Keep upstream GPU scheduling inside the ordinary episode lifecycle."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.process: subprocess.Popen[bytes] | None = None
        self.log: BinaryIO | None = None
        self.pipe: tempfile.TemporaryDirectory[str] | None = None
        self.environment: dict[str, str] = {}
        self.children: dict[int, ProcessIdentity] = {}
        self.next_identity_check = 0.0

    def start(self, environment: dict[str, str]) -> dict[str, str]:
        if any(name.startswith("CUDA_MPS_") for name in environment):
            raise ValueError(
                "The group owns MPS settings; inherited CUDA_MPS_* settings are forbidden"
            )
        self.output.mkdir()
        self.pipe = tempfile.TemporaryDirectory(prefix="humanoid-agent-mps-")
        logs = self.output / "logs"
        logs.mkdir()
        self.environment = {
            **environment,
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe.name,
            "CUDA_MPS_LOG_DIRECTORY": str(logs),
        }
        self.log = (self.output / "control.log").open("wb")
        self.process = subprocess.Popen(
            ["nvidia-cuda-mps-control", "-f"],
            env=self.environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atomic_json(self.output / "process.json", process_record(self.process.pid), durable=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.check()
            if self.control("get_server_list").returncode == 0:
                atomic_json(
                    self.output / "configuration.json",
                    {
                        "scheduling": "cuda-mps",
                        "pipe_directory": self.pipe.name,
                        "command": ["nvidia-cuda-mps-control", "-f"],
                        "compute_mode_changed": False,
                        "thread_partition_changed": False,
                    },
                )
                return self.environment
            time.sleep(0.1)
        raise TimeoutError("NVIDIA MPS control did not become ready")

    def control(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            text=True,
            capture_output=True,
            env=self.environment,
            timeout=5,
            check=False,
        )

    def check(self) -> None:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Owned NVIDIA MPS control exited before group cleanup")
        if time.monotonic() < self.next_identity_check:
            return
        self.next_identity_check = time.monotonic() + 1
        owner = psutil.Process(self.process.pid)
        for child in owner.children(recursive=True):
            if child.pid not in self.children:
                identity = process_identity(child.pid)
                if identity is not None:
                    self.children[child.pid] = identity
                    atomic_json(
                        self.output / "children.json",
                        {
                            str(pid): child_identity.model_dump(mode="json")
                            for pid, child_identity in self.children.items()
                        },
                        durable=True,
                    )

    def close(self) -> None:
        """Call after all CUDA clients stop; retain failures and force owned cleanup."""
        errors = []
        try:
            if self.process is not None:
                try:
                    self.check()
                    result = self.control("quit")
                    result.check_returncode()
                    self.process.wait(timeout=10)
                    if self.process.returncode != 0:
                        raise RuntimeError(f"NVIDIA MPS control exited {self.process.returncode}")
                except Exception as exception:
                    errors.append(f"{type(exception).__name__}: {exception}")
                finally:
                    stop_process(self.process, extra_owned=list(self.children.values()))
                atomic_json(
                    self.output / "shutdown.json",
                    {
                        "status": "failed" if errors else "completed",
                        "control_exit_code": self.process.returncode,
                        "errors": errors,
                        "remaining_owned_processes": [
                            identity.pid
                            for identity in self.children.values()
                            if identity.matching_process() is not None
                        ],
                    },
                    durable=True,
                )
        finally:
            if self.log is not None:
                self.log.close()
            if self.pipe is not None:
                self.pipe.cleanup()
        if errors:
            raise RuntimeError("NVIDIA MPS shutdown failed: " + "; ".join(errors))
