"""Own the lifecycle of the pinned unchanged SONIC deployment."""

import subprocess
from pathlib import Path

from humanoid_agent.runtime.cpu import CpuAllocation
from humanoid_agent.runtime.deployment import (
    EXECUTABLE_SHA256,
    SONIC_REVISION,
    native_controller_environment,
)
from humanoid_agent.runtime.deployment import (
    native_controller_arguments as deployment_controller_arguments,
)


def native_controller_arguments(workspace: Path, output: Path) -> list[str]:
    return deployment_controller_arguments(
        workspace, output, cpu_set=CpuAllocation.from_environment().controller
    )


class NativeControllerSession:
    """Own an unchanged SONIC instance inside its SDK worker's private network."""

    def __init__(self, workspace: Path, output: Path) -> None:
        import os
        import pty

        from humanoid_agent.artifact_io import atomic_json
        from humanoid_agent.process_lifecycle import process_record

        self.workspace = workspace
        self.output = output
        self.process: subprocess.Popen[bytes] | None = None
        self.master: int | None = None
        self.log = (output / "controller.log").open("w")
        self._control_seen_at: float | None = None
        self._ready = False
        slave = None
        try:
            arguments = native_controller_arguments(workspace, output)
            self.arguments = arguments
            self.master, slave = pty.openpty()
            self.process = subprocess.Popen(
                arguments,
                stdin=slave,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                cwd=workspace / "SONIC/gear_sonic_deploy",
                start_new_session=True,
                env=native_controller_environment(workspace),
            )
            atomic_json(output / "controller-process.json", process_record(self.process.pid))
            atomic_json(
                output / "controller-configuration.json",
                {
                    "arguments": arguments,
                    "executable_sha256": EXECUTABLE_SHA256,
                    "source_revision": SONIC_REVISION,
                    "state_scope": "One independent native policy/planner/history for one robot",
                },
            )
        except BaseException:
            self.close()
            raise
        finally:
            if slave is not None:
                os.close(slave)

    def ready(self) -> bool:
        import time

        from humanoid_agent.artifact_io import atomic_json

        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Native controller exited; inspect its controller.log")
        if self._ready:
            return True
        contents = (self.output / "controller.log").read_text(errors="replace")
        now = time.monotonic()
        if "transitioning to CONTROL state" in contents and self._control_seen_at is None:
            self._control_seen_at = now
        if (
            self._control_seen_at is not None
            and "VR 3-point control enabled" in contents
            and "Switched to mode 'teleop'" in contents
        ):
            self._ready = True
            atomic_json(
                self.output / "controller-ready.json",
                {
                    "pid": self.process.pid,
                    "ready_monotonic_seconds": now,
                    "control_observed_monotonic_seconds": self._control_seen_at,
                    "readiness": "CONTROL and native VR teleop mode observed; no additional frozen-physics dwell",
                },
            )
        return self._ready

    def close(self) -> None:
        import os
        import signal

        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
                if self.process.returncode not in (0, -signal.SIGINT, 128 + signal.SIGINT):
                    raise RuntimeError(
                        f"Native SONIC exited {self.process.returncode}; inspect controller.log"
                    )
        finally:
            if self.master is not None:
                os.close(self.master)
                self.master = None
            self.log.close()
