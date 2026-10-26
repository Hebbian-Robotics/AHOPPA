"""Physical balance checks for standing and deliberate ground-reaching tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RootBalanceCriterion:
    protocol: Literal["neutral-standing-v1", "ground-reaching-root-state-v1"]
    minimum_root_height_m: float
    minimum_upright_component: float

    def is_fallen(self, root_height_m: float, projected_gravity_z: float) -> bool:
        if not math.isfinite(root_height_m) or not math.isfinite(projected_gravity_z):
            raise ValueError("Balance measurements must be finite")
        return (
            root_height_m < self.minimum_root_height_m
            or -projected_gravity_z < self.minimum_upright_component
        )


STANDING_DIAGNOSTIC_BALANCE = RootBalanceCriterion("neutral-standing-v1", 0.35, 0.5)
# Root-state component shared by both simulators, allowing deliberate kneels.
# Isaac also applies the versioned filtered floor-contact rule below.
GROUND_REACHING_BALANCE = RootBalanceCriterion("ground-reaching-root-state-v1", 0.18, 0.4)
ISAAC_CONTACT_BALANCE_PROTOCOL = "ground-reaching-filtered-floor-contact-v2"


def forbidden_floor_contact(
    pelvis_peak_normal_force_n: float, torso_peak_normal_force_n: float
) -> bool:
    """Reject load-bearing pelvis/torso contact while permitting foot/knee support.

    Forces are private, filtered Isaac measurements for the task terrain only.
    Zero is the contact/no-force boundary, not a tuned force tolerance. The
    root-state criterion remains a separate guard against collapse or tipping.
    """
    forces = (pelvis_peak_normal_force_n, torso_peak_normal_force_n)
    if any(not math.isfinite(force) or force < 0 for force in forces):
        raise ValueError("Filtered floor-force magnitudes must be finite and nonnegative")
    return any(force > 0 for force in forces)
