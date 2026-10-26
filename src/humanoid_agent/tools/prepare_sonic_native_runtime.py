"""Build the unchanged pinned SONIC deployment in an isolated Ubuntu 22.04 prefix.

No system package is installed and no existing Python/Isaac environment is
changed. Downloads, source revision, SDKs, package versions and the executable
are retained beside the build. This prepares tooling, never a robot episode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.runtime.deployment import SONIC_REVISION

CPPZMQ_REVISION = "d67b6352b87a238775cd17e4376b980d07fa7939"
TENSORRT_ARCHIVE = "TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz"
TENSORRT_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/"
    "tensorrt/10.13.3/tars/" + TENSORRT_ARCHIVE
)
ONNX_ARCHIVE = "onnxruntime-linux-x64-1.16.3.tgz"
ONNX_URL = "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/" + ONNX_ARCHIVE
DEVELOPMENT_PACKAGES = (
    "git-lfs",
    "libeigen3-dev",
    "libmsgpack-dev",
    "libmsgpackc2",
    "libzmq3-dev",
    "libzmq5",
    "nlohmann-json3-dev",
    "libgtest-dev",
    "googletest",
)


logger = logging.getLogger(__name__)


def run_command(arguments: list[str], directory: Path, environment: dict[str, str]) -> None:
    logger.info("Running dependency build command in %s", directory)
    subprocess.run(arguments, cwd=directory, env=environment, check=True)


def expose_tensor_rt_library_directory(tensor_rt: Path) -> None:
    """Expose the SDK's architecture-specific layout to upstream CMake."""
    target_directory = tensor_rt / "targets/x86_64-linux-gnu"
    for directory_name in ("lib", "bin"):
        destination = tensor_rt / directory_name
        if not destination.exists() and (target_directory / directory_name).is_dir():
            destination.symlink_to(Path("targets/x86_64-linux-gnu") / directory_name)


def main(cli_workspace: Annotated[Path, Parameter(name="workspace")], /) -> None:
    workspace = cli_workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    tensor_rt = workspace / "TensorRT-10.13.3.9"
    expose_tensor_rt_library_directory(tensor_rt)
    required_sdk_files = (
        "include/NvInfer.h",
        "include/NvOnnxParser.h",
        "lib/libnvinfer.so",
        "lib/libnvinfer_plugin.so",
        "lib/libnvonnxparser.so",
        "lib/libnvinfer_builder_resource.so.10.13.3",
    )
    sdk_ready = all((tensor_rt / relative).is_file() for relative in required_sdk_files)
    required_free_space = 2 * 1024**3
    if not (workspace / "downloads" / TENSORRT_ARCHIVE).exists():
        required_free_space += 8 * 1024**3
    if not sdk_ready:
        required_free_space += 8 * 1024**3
    if shutil.disk_usage(workspace).free < required_free_space:
        raise ValueError("Insufficient free disk space for the remaining isolated build stages")
    operating_system = Path("/etc/os-release").read_text()
    if 'VERSION_ID="22.04"' not in operating_system:
        raise ValueError("This isolated dependency recipe targets the inspected Ubuntu 22.04 host")
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "HAS_ROS2": "0"}

    def record_phase(phase: str) -> None:
        atomic_json(workspace / "status.json", {"phase": phase})

    downloads = workspace / "downloads"
    downloads.mkdir(exist_ok=True)
    record_phase("download_tensor_rt")
    for filename, url in ((TENSORRT_ARCHIVE, TENSORRT_URL), (ONNX_ARCHIVE, ONNX_URL)):
        destination = downloads / filename
        if not destination.exists():
            incoming = destination.with_suffix(destination.suffix + ".incoming")
            run_command(
                ["curl", "--fail", "--location", "--retry", "2", "--output", str(incoming), url],
                downloads,
                environment,
            )
            incoming.replace(destination)
    record_phase("extract_sdks")
    if not sdk_ready:
        with tempfile.TemporaryDirectory(
            prefix="tensor-rt-extraction-", dir=workspace
        ) as temporary:
            extraction_directory = Path(temporary)
            run_command(
                [
                    "tar",
                    "-xzf",
                    str(downloads / TENSORRT_ARCHIVE),
                    "--wildcards",
                    "*/include/*",
                    "*/lib/*.so*",
                    "*/bin/trtexec",
                ],
                extraction_directory,
                environment,
            )
            extracted_sdk = extraction_directory / tensor_rt.name
            expose_tensor_rt_library_directory(extracted_sdk)
            if not all((extracted_sdk / relative).is_file() for relative in required_sdk_files):
                raise ValueError("TensorRT extraction lacks required headers or shared libraries")
            if tensor_rt.exists():
                tensor_rt.rename(workspace / (tensor_rt.name + ".incomplete"))
            extracted_sdk.rename(tensor_rt)
    onnx_runtime = workspace / "onnxruntime-linux-x64-1.16.3"
    required_onnx_files = ("include/onnxruntime_cxx_api.h", "lib/libonnxruntime.so")
    if not all((onnx_runtime / relative).is_file() for relative in required_onnx_files):
        with tempfile.TemporaryDirectory(prefix="onnx-extraction-", dir=workspace) as temporary:
            extraction_directory = Path(temporary)
            run_command(
                ["tar", "-xzf", str(downloads / ONNX_ARCHIVE)], extraction_directory, environment
            )
            extracted_sdk = extraction_directory / onnx_runtime.name
            if not all((extracted_sdk / relative).is_file() for relative in required_onnx_files):
                raise ValueError("ONNX Runtime extraction lacks required headers or shared library")
            if onnx_runtime.exists():
                onnx_runtime.rename(workspace / (onnx_runtime.name + ".incomplete"))
            extracted_sdk.rename(onnx_runtime)
    record_phase("stage_debian_dependencies")
    dependencies = workspace / "dependencies"
    dependencies.mkdir(exist_ok=True)
    package_manifest = dependencies / "packages.json"
    recorded_packages = (
        json.loads(package_manifest.read_text()) if package_manifest.exists() else {}
    )
    if not all(
        any(filename.startswith(package + "_") for filename in recorded_packages)
        for package in DEVELOPMENT_PACKAGES
    ):
        run_command(["apt", "download", *DEVELOPMENT_PACKAGES], downloads, environment)
        package_versions = {}
        for package in sorted(downloads.glob("*.deb")):
            description = subprocess.run(
                ["dpkg-deb", "-f", str(package), "Package", "Version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            package_versions[package.name] = description.strip()
            run_command(["dpkg-deb", "-x", str(package), str(dependencies)], downloads, environment)
        (dependencies / "packages.json").write_text(json.dumps(package_versions, indent=2) + "\n")
    environment["PATH"] = str(dependencies / "usr/bin") + os.pathsep + environment["PATH"]
    cpp_zmq = workspace / "cppzmq"
    if not cpp_zmq.exists():
        run_command(["git", "init", str(cpp_zmq)], workspace, environment)
        run_command(
            ["git", "remote", "add", "origin", "https://github.com/zeromq/cppzmq.git"],
            cpp_zmq,
            environment,
        )
        run_command(
            ["git", "fetch", "--depth", "1", "origin", CPPZMQ_REVISION], cpp_zmq, environment
        )
        run_command(["git", "checkout", "--detach", "FETCH_HEAD"], cpp_zmq, environment)
    cpp_zmq_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cpp_zmq, check=True, capture_output=True, text=True
    ).stdout.strip()
    if cpp_zmq_revision != CPPZMQ_REVISION:
        raise ValueError("Existing cppzmq checkout differs from the pinned source")
    cpp_zmq_changes = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cpp_zmq,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if cpp_zmq_changes:
        raise ValueError("cppzmq headers must come from the unchanged pinned source")
    for header in ("zmq.hpp", "zmq_addon.hpp"):
        shutil.copyfile(cpp_zmq / header, dependencies / "usr/include" / header)
    record_phase("checkout_pinned_sonic")
    source = workspace / "SONIC"
    if not source.exists():
        run_command(["git", "init", str(source)], workspace, environment)
        run_command(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/NVlabs/GR00T-WholeBodyControl.git",
            ],
            source,
            environment,
        )
        run_command(["git", "fetch", "--depth", "1", "origin", SONIC_REVISION], source, environment)
        run_command(["git", "checkout", "--detach", "FETCH_HEAD"], source, environment)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != SONIC_REVISION:
        raise ValueError("Existing native source checkout differs from the pinned revision")
    source_changes = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_changes:
        raise ValueError("Native deployment must build from the unchanged upstream source")
    native_library_patterns = (
        "gear_sonic_deploy/thirdparty/unitree_sdk2/lib/x86_64/libunitree_sdk2.a,"
        "gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64/libddsc*.so*"
    )
    run_command(["git", "lfs", "install", "--local", "--skip-smudge"], source, environment)
    run_command(
        ["git", "lfs", "fetch", "--include=" + native_library_patterns, "origin", revision],
        source,
        environment,
    )
    run_command(["git", "lfs", "checkout"], source, environment)
    environment.update({"TensorRT_ROOT": str(tensor_rt), "onnxruntime_ROOT": str(onnx_runtime)})
    record_phase("configure_native_runtime")
    run_command(
        [
            "cmake",
            "-S",
            str(source / "gear_sonic_deploy"),
            "-B",
            str(workspace / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PREFIX_PATH=" + str(dependencies / "usr"),
            "-DCMAKE_CXX_FLAGS=-I" + str(dependencies / "usr/include"),
            "-DCUDA_INCLUDE_DIR=/usr/include",
            "-DCUDA_RUNTIME_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudart.so",
            "-DMSGPACK_INCLUDE_DIR=" + str(dependencies / "usr/include"),
            "-DZMQ_INCLUDE_DIR=" + str(dependencies / "usr/include"),
            "-DZMQ_LIBRARY=" + str(dependencies / "usr/lib/x86_64-linux-gnu/libzmq.so"),
        ],
        workspace,
        environment,
    )
    record_phase("compile_native_runtime")
    run_command(
        ["cmake", "--build", str(workspace / "build"), "--target", "g1_deploy_onnx_ref", "-j", "2"],
        workspace,
        environment,
    )
    executable = source / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    run_command(["git", "diff", "--exit-code", "HEAD"], source, environment)
    manifest = {
        "sonic_revision": revision,
        "cppzmq_revision": cpp_zmq_revision,
        "tensor_rt_url": TENSORRT_URL,
        "onnx_runtime_url": ONNX_URL,
        "runtime_library_directory": str(tensor_rt / "lib"),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "download_sha256": {
            path.name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            for path in sorted(downloads.iterdir())
            if path.is_file()
        },
        "model_requests": 0,
        "episodes_executed": 0,
    }
    (workspace / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    record_phase("build_complete")


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
