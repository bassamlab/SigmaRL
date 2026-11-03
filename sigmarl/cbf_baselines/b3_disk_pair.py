# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B3: disk barriers. Each vehicle is covered by its circumscribed disk of
radius R = sqrt(L^2 + W^2) / 2; the pair keeps the center distance
D_s = 2 R + buffer.

b3_wang: safety barrier certificates (Wang, Ames & Egerstedt, T-RO 2017,
Eq. 6). For double integrators with acceleration bound alpha, the relative
velocity must allow braking before the disks touch:
  h = sqrt(2 (alpha_i + alpha_j) (|dp| - D_s)) + dp . dv / |dp|,
dp = p_i - p_j, dv = v_i - v_j (vehicle centers). alpha is the braking limit
(the paper's derivation stops the closing speed). Since a car brakes only
along its heading, the braking argument is exact for aligned pairs and
optimistic for crossing ones. The square root is regularized,
sqrt(2A(e + e0)) - sqrt(2A e0) with e = |dp| - D_s, and continued linearly
for e < 0, which keeps the zero level and a finite slope. The barrier depends
on the velocities, so the discrete-time condition is one-step (as B4), and it
is multiplied by REF_TIME to be in meters. Decentralized: the paper's split
alpha_i / (alpha_i + alpha_j) = 1/2 (Eq. 12). The paper's class-K function
gamma h^3 is replaced by the common decay rate lambda.

b3_disk: the position-only disk distance h = |p_i - p_j| - D_s, a common
discrete-time disk CBF (no braking term), with the reciprocal split 1/2.
"""

import math

import numpy as np

from sigmarl.cbf_baselines.common import PairBaseline, predict_center_velocity
from sigmarl.constants import LENGTH_SCALE


class DiskPairBaseline(PairBaseline):
    """B3 (b3_disk) pair rows: one disk-disk distance barrier per pair."""

    CROSS_SHARE = 0.5

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.radius = 0.5 * math.hypot(float(ctrl.length), float(ctrl.width))
        self.clearance = 2.0 * self.radius + float(ctrl.pair_safety_buffer)

    def value(self, x_a, x_b) -> float:
        return float(np.linalg.norm(np.asarray(x_a[0:2]) - x_b[0:2])) - self.clearance

    def value_and_gradients(self, x_a, x_b) -> tuple:
        delta = np.asarray(x_a[0:2], dtype=np.float64) - x_b[0:2]
        distance = max(float(np.linalg.norm(delta)), 1e-9)
        grad_a = np.zeros(3)
        grad_a[0:2] = delta / distance
        grad_b = -grad_a
        return distance - self.clearance, grad_a, grad_b


class WangPairBaseline(PairBaseline):
    """B3 (b3_wang) pair rows: one safety barrier certificate per pair.

    The per-agent representation is (x, y, v_x, v_y) of the vehicle center.
    """

    CROSS_SHARE = 0.5
    # [s] h * REF_TIME is in meters
    REF_TIME = 1.0
    # [m] Regularization of the square root at the disk distance
    SQRT_OFFSET = 0.0025 * LENGTH_SCALE

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.clearance = math.hypot(float(ctrl.length), float(ctrl.width)) + float(
            ctrl.pair_safety_buffer
        )
        # alpha_i + alpha_j with the braking limit of identical agents
        self.acc_sum = 2.0 * abs(float(ctrl.a_min))

    def predict(self, state_np, u):
        return predict_center_velocity(self.ctrl, state_np, u)

    def braking_term(self, excess: float) -> tuple:
        """Regularized sqrt(2 A excess) and its derivative."""
        two_a, e0 = 2.0 * self.acc_sum, self.SQRT_OFFSET
        slope0 = self.acc_sum / math.sqrt(two_a * e0)
        if excess < 0.0:
            return slope0 * excess, slope0
        root = math.sqrt(two_a * (excess + e0))
        return root - math.sqrt(two_a * e0), self.acc_sum / root

    def value_and_gradients(self, x_a, x_b) -> tuple:
        dp = np.asarray(x_a[0:2], dtype=np.float64) - x_b[0:2]
        dv = np.asarray(x_a[2:4], dtype=np.float64) - x_b[2:4]
        distance = max(float(np.linalg.norm(dp)), 1e-9)
        n = dp / distance
        phi, dphi = self.braking_term(distance - self.clearance)
        closing = float(n @ dv)
        value = phi + closing
        grad_p = dphi * n + (dv - closing * n) / distance
        grad_a = np.concatenate([grad_p, n]) * self.REF_TIME
        return value * self.REF_TIME, grad_a, -grad_a

    def value(self, x_a, x_b) -> float:
        return self.value_and_gradients(x_a, x_b)[0]
