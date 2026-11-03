# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-vehicle parameters of heterogeneous traffic.

By default, every agent has the parameters of the agent profile (`AGENTS`). A
log replay can give an agent the parameters of a recorded vehicle
(`ScenarioRoadTraffic.set_vehicle_params`): its footprint, its axle distances, and its
acceleration limits. Steering, speed, and steering-rate limits stay those of
the profile.
"""

from dataclasses import dataclass

import numpy as np

from sigmarl.constants import AGENTS

# Columns of `RoadTraffic.vehicle_params`
VEHICLE_PARAM_KEYS = ("length", "width", "l_f", "l_r", "l_wb", "min_acc", "max_acc")


@dataclass(frozen=True)
class VehicleParams:
    """Footprint [m], axle distances from the center [m], and acceleration
    limits [m/s^2] of one vehicle (kinematic bicycle model)."""

    length: float
    width: float
    l_f: float
    l_r: float
    min_acc: float
    max_acc: float

    def __post_init__(self):
        # The stopping horizon of the backup (BH-CBF Lemma 1) assumes that the
        # first control cannot raise the speed more than braking reduces it
        assert self.min_acc < 0.0 < self.max_acc <= -self.min_acc, self

    @property
    def l_wb(self) -> float:
        return self.l_f + self.l_r

    @classmethod
    def nominal(cls) -> "VehicleParams":
        """The parameters of the agent profile."""
        return cls(
            length=AGENTS["length"],
            width=AGENTS["width"],
            l_f=AGENTS["l_f"],
            l_r=AGENTS["l_r"],
            min_acc=AGENTS["min_acc"],
            max_acc=AGENTS["max_acc"],
        )

    @classmethod
    def scaled(
        cls, length: float, width: float, min_acc: float, max_acc: float
    ) -> "VehicleParams":
        """A vehicle of the given footprint whose axle distances scale with its
        length like those of the profile (the wheelbase of recorded vehicles is
        unknown)."""
        scale = length / AGENTS["length"]
        return cls(
            length=float(length),
            width=float(width),
            l_f=AGENTS["l_f"] * scale,
            l_r=AGENTS["l_r"] * scale,
            min_acc=float(min_acc),
            max_acc=float(max_acc),
        )

    def row(self) -> np.ndarray:
        """Values in the order of VEHICLE_PARAM_KEYS."""
        return np.array(
            [
                self.length,
                self.width,
                self.l_f,
                self.l_r,
                self.l_wb,
                self.min_acc,
                self.max_acc,
            ]
        )
