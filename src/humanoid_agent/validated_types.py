"""Strict boundary models and immutable numerical values with JSON-array encoding."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, FiniteFloat, PlainSerializer


class ValidatedModel(BaseModel):
    """Keep domain values valid after construction, including copied updates."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        # Pydantic's default copy trusts updates; executable values must validate them.
        values = {name: getattr(self, name) for name in type(self).model_fields}
        if deep:
            values = deepcopy(values)
        return type(self).model_validate({**values, **(update or {})})


type ImmutableSequence[Item] = Annotated[
    Sequence[Item], AfterValidator(tuple), PlainSerializer(list)
]
Vector3 = Annotated[ImmutableSequence[FiniteFloat], Field(min_length=3, max_length=3)]
Quaternion = Annotated[ImmutableSequence[FiniteFloat], Field(min_length=4, max_length=4)]


def require_unit_quaternion(quaternion: Sequence[float]) -> Sequence[float]:
    if not math.isclose(
        math.sqrt(sum(value * value for value in quaternion)), 1, rel_tol=0, abs_tol=0.001
    ):
        raise ValueError("Orientation must be a unit quaternion in w, x, y, z order")
    return quaternion


UnitQuaternion = Annotated[Quaternion, AfterValidator(require_unit_quaternion)]


def require_unit_tracking_orientations(orientations: Sequence[float]) -> Sequence[float]:
    for offset in range(0, 12, 4):
        require_unit_quaternion(orientations[offset : offset + 4])
    return orientations


TrackingQuaternions = Annotated[
    ImmutableSequence[FiniteFloat],
    Field(min_length=12, max_length=12),
    AfterValidator(require_unit_tracking_orientations),
]
