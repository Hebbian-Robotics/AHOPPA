"""Measure MuJoCo Warp contact and constraint usage against the fixed allocation.

Contacts beyond the allocation are dropped silently, so a scene that fills it can
lose foot contacts. This reads solver counters only; it never changes the state.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContactCapacityMonitor:
    samples: int = 0
    maximum_contacts: int = 0
    maximum_constraint_rows: int = 0
    overflow_samples: int = 0
    contact_capacity: int | None = None
    constraint_row_capacity: int | None = None
    unavailable_reason: str | None = None
    _data: Any = field(default=None, repr=False)

    def attach(self) -> None:
        try:
            manager: Any = importlib.import_module(
                "isaaclab_newton.physics.newton_manager"
            ).NewtonManager
            self._data = manager._solver.mjw_data
            self.contact_capacity = int(self._data.naconmax)
            self.constraint_row_capacity = int(self._data.njmax)
        except (AttributeError, ImportError) as error:
            self.unavailable_reason = f"{type(error).__name__}: {error}"
            self._data = None

    def sample(self) -> None:
        if self._data is None:
            return
        contacts = int(self._data.nacon.numpy()[0])
        constraint_rows = int(self._data.nefc.numpy().max())
        self.samples += 1
        self.maximum_contacts = max(self.maximum_contacts, contacts)
        self.maximum_constraint_rows = max(self.maximum_constraint_rows, constraint_rows)
        self.overflow_samples += int(self._data.overflow.numpy().any())

    def summary(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "maximum_contacts": self.maximum_contacts,
            "contact_capacity": self.contact_capacity,
            "maximum_constraint_rows": self.maximum_constraint_rows,
            "constraint_row_capacity": self.constraint_row_capacity,
            "overflow_samples": self.overflow_samples,
            "unavailable_reason": self.unavailable_reason,
            "model_requests": 0,
        }
