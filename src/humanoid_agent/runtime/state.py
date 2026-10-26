"""Translate Newton state conventions at the robot sensor and recording boundary."""

from __future__ import annotations

import importlib
import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class _TensorTransferStorage:
    device: Any
    host: Any


class TensorReadbackBuffers:
    """Reuse transfer storage while returning independent CPU-owned arrays."""

    def __init__(self) -> None:
        self.storage: OrderedDict[tuple[int, int, str, str], _TensorTransferStorage] = OrderedDict()

    def copy(self, fields: dict[str, Any], tensors: list[Any], widths: list[int]) -> dict[str, Any]:
        torch: Any = importlib.import_module("torch")
        reference = tensors[0]
        shape = (reference.shape[0], sum(widths))
        key = (*shape, str(reference.dtype), str(reference.device))
        if key not in self.storage:
            self.storage[key] = _TensorTransferStorage(
                device=torch.empty(shape, dtype=reference.dtype, device=reference.device),
                host=torch.empty(
                    shape,
                    dtype=reference.dtype,
                    device="cpu",
                    pin_memory=str(reference.device).startswith("cuda"),
                ),
            )
            if len(self.storage) > 16:
                self.storage.popitem(last=False)
        self.storage.move_to_end(key)
        storage = self.storage[key]
        torch.cat(
            [tensor.reshape(tensor.shape[0], -1) for tensor in tensors], dim=1, out=storage.device
        )
        storage.host.copy_(storage.device)
        packed = storage.host.numpy()
        result = {}
        offset = 0
        for (name, tensor), width in zip(fields.items(), widths, strict=True):
            # Evidence and scoring may retain these arrays after the next copy.
            # Ownership crosses here; reusable pinned staging never escapes.
            result[name] = packed[:, offset : offset + width].copy().reshape(tensor.shape)
            offset += width
        return result


def copy_tensor_fields_to_cpu(
    fields: dict[str, Any], *, buffers: TensorReadbackBuffers | None = None
) -> dict[str, Any]:
    """Copy named batched sensor fields through one device-to-host transfer.

    The pinned simulator owns tensor construction and field semantics. Preserve
    each field's shape, dtype and robot order rather than copying each small
    sensor tensor separately and synchronizing the GPU repeatedly.
    """
    if not fields:
        raise ValueError("A sensor snapshot requires at least one tensor field")
    tensors = list(fields.values())
    reference = tensors[0]
    if reference.ndim < 2 or reference.shape[0] < 1:
        raise ValueError("Sensor fields require a nonempty leading robot dimension")
    if any(
        tensor.ndim < 2
        or tensor.shape[0] != reference.shape[0]
        or tensor.dtype != reference.dtype
        or tensor.device != reference.device
        for tensor in tensors
    ):
        raise ValueError("Batched sensor fields must share robot count, dtype and device")
    torch: Any = importlib.import_module("torch")
    widths = [math.prod(tensor.shape[1:]) for tensor in tensors]
    if buffers is not None:
        return buffers.copy(fields, tensors, widths)
    packed = torch.cat([tensor.reshape(tensor.shape[0], -1) for tensor in tensors], dim=1).cpu()
    return {
        name: chunk.numpy().reshape(tensor.shape)
        for name, tensor, chunk in zip(
            fields, tensors, torch.split(packed, widths, dim=1), strict=True
        )
    }


class TickSnapshotCache:
    """Own CPU fields only until the next physics step changes the state."""

    def __init__(self) -> None:
        self.tick: int | None = None
        self.fields: dict[str, Any] = {}
        self.field_sample_times: dict[str, float] = {}
        self.readback_buffers = TensorReadbackBuffers()

    def invalidate(self) -> None:
        self.tick = None
        self.fields.clear()
        self.field_sample_times.clear()

    def read(self, tick: int, fields: dict[str, Any]) -> dict[str, Any]:
        if tick != self.tick:
            self.invalidate()
            self.tick = tick
        missing = {name: tensor for name, tensor in fields.items() if name not in self.fields}
        if missing:
            self.fields.update(copy_tensor_fields_to_cpu(missing, buffers=self.readback_buffers))
            sampled_at = time.monotonic()
            self.field_sample_times.update(dict.fromkeys(missing, sampled_at))
        return {name: self.fields[name] for name in fields}

    def sampled_monotonic_seconds(self, field_names: Sequence[str]) -> float:
        """Return when the final required field became an owned CPU snapshot."""
        return max(self.field_sample_times[name] for name in field_names)


def recorded_state_from_newton(
    state: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return position, wxyz orientation, linear velocity and angular velocity.

    Isaac Lab 3 exposes Newton state as position, xyzw orientation, linear
    velocity and angular velocity. SONIC sensor packets and our saved camera
    states use wxyz with the same velocity order. This changes representation
    only; it does not rotate either world-frame velocity vector.
    """
    source = np.asarray(state, dtype=np.float64)
    if source.shape != (13,) or not np.isfinite(source).all():
        raise ValueError("Newton state must contain 13 finite values")
    if not np.isclose(np.linalg.norm(source[3:7]), 1, atol=1e-3, rtol=0):
        raise ValueError("Newton state requires a unit xyzw quaternion")
    return source[[0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12]]


# Five 50 Hz controller periods; one quarter of the shortest permitted motion.
# This is a prespecified engineering tolerance, not an industry standard.
MAXIMUM_VALID_PHYSICS_LAG_SECONDS = 0.1


@dataclass(frozen=True)
class ClockValidity:
    status: Literal["valid", "invalid_physics_lag"]
    maximum_physics_lag_seconds: float
    limit_seconds: float = MAXIMUM_VALID_PHYSICS_LAG_SECONDS
    policy: str = "wall-clock-lag-v1"


def clock_validity(maximum_lag_seconds: float) -> dict[str, str | float]:
    """Keep simulator validity independent of robot task and execution outcomes."""
    if not math.isfinite(maximum_lag_seconds) or maximum_lag_seconds < 0:
        raise ValueError("Maximum physics lag must be finite and nonnegative")
    return asdict(
        ClockValidity(
            "valid"
            if maximum_lag_seconds <= MAXIMUM_VALID_PHYSICS_LAG_SECONDS
            else "invalid_physics_lag",
            maximum_lag_seconds,
        )
    )
