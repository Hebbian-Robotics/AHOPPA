"""Launch arguments for the pinned unchanged SONIC deployment."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

SONIC_REVISION = "45c03facbf848e0c743091b7d9c17952096ff908"
EXECUTABLE_SHA256 = "1004e196c772846214f119604364cff6cc7cc945fc639997acb8f5923836cdf0"


def native_controller_environment(workspace: Path) -> dict[str, str]:
    directories = [
        workspace / "SONIC/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64",
        workspace / "TensorRT-10.13.3.9/lib",
        workspace / "onnxruntime-linux-x64-1.16.3/lib",
        workspace / "dependencies/usr/lib/x86_64-linux-gnu",
    ]
    return {
        **os.environ,
        "LD_LIBRARY_PATH": os.pathsep.join(map(str, directories))
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", ""),
    }


def native_controller_arguments(
    workspace: Path,
    output: Path,
    *,
    cpu_set: str,
) -> list[str]:
    source = workspace / "SONIC"
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    executable = source / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    if (
        revision != SONIC_REVISION
        or hashlib.sha256(executable.read_bytes()).hexdigest() != EXECUTABLE_SHA256
    ):
        raise ValueError(
            "Native executable or source revision differs from the validated deployment"
        )
    models = workspace / "deployment-models-v1"
    manifest = json.loads((workspace / "deployment-assets-v1.json").read_text())
    for relative in (
        "sonic_v1_1/model_decoder.onnx",
        "sonic_v1_1/model_encoder.onnx",
        "sonic_v1_1/observation_config.yaml",
        "planner_sonic.onnx",
    ):
        with (models / relative).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["files"][relative]["sha256"]:
            raise ValueError("Native deployment asset changed: " + relative)
    planner = models / "planner/target_vel/V2/planner_sonic.onnx"
    if planner.resolve() != (models / "planner_sonic.onnx").resolve():
        raise ValueError("V2 planner path must resolve to the verified asset")
    return [
        "taskset",
        "-c",
        cpu_set,
        str(executable),
        "lo",
        str(models / "sonic_v1_1/model_decoder.onnx"),
        str(source / "gear_sonic_deploy/reference/example"),
        "--encoder-file",
        str(models / "sonic_v1_1/model_encoder.onnx"),
        "--obs-config",
        str(models / "sonic_v1_1/observation_config.yaml"),
        "--planner-file",
        str(planner),
        "--planner-precision",
        "32",
        "--policy-precision",
        "32",
        "--input-type",
        "zmq_manager",
        "--zmq-host",
        "127.0.0.1",
        "--zmq-port",
        "17756",
        "--zmq-out-port",
        "17757",
        # Pinned SONIC README: tested v1.1 ankle-pitch tuning (hardware order).
        "--motor-kp-scale",
        "4,10=1.5",
        "--motor-kd-scale",
        "4,10=1.5",
        "--disable-crc-check",
        "--logs-dir",
        str(output / "native-state-logs"),
        "--enable-csv-logs",
        "--policy-input-logfile",
        str(output / "policy-input.csv"),
        "--target-motion-logfile",
        str(output / "target-motion.csv"),
    ]
