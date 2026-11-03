# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B4: collision-cone CBF for car-like robots, C3BF (Thontepu et al., 2022;
Goswami et al., ACC 2024).

With the relative position p = p_b - p_a and velocity v = v_b - v_a of the
vehicle centers (obstacle minus ego, Goswami Eqs. 12, 15-16),
  h = <p, v> + |p| |v| cos(phi),   cos(phi) = sqrt(|p|^2 - r^2) / |p|,
is nonnegative iff the relative velocity points outside the cone of
directions that lead into the disk of radius r. The radius is
r = max(c1, c2) + w/2 with the obstacle ellipse's semi-axes c1 = L/sqrt(2),
c2 = W/sqrt(2) and, per the paper's text, w the largest ego dimension (L):
r = L/sqrt(2) + L/2 (plus the pair buffer). The radius "width" uses the ego
width instead, as the paper's Fig. 3 caption suggests. The tuned choice per
block and mode is in tuned_defaults.json; SIGMARL_B4_RADIUS overrides it.

Deviations from the paper: the velocity points along psi + beta (the course),
since the steering rate (not the slip angle) is our input; the barrier depends
on the velocities, which the inputs change within one Euler step, so the
discrete-time condition is one-step, h(x_{k+1}(u)) >= (1 - lambda) h(x_k). The
barrier is divided by the speed limit so that it is in meters, like the other
barriers, for the common slack weight. In the decentralized mode, each ego
takes full responsibility and treats the other car as moving at constant
velocity, as the paper's ego does with its obstacles.
"""

import math

import numpy as np

from sigmarl.cbf_baselines import tuned_env_value
from sigmarl.cbf_baselines.common import PairBaseline, predict_center_velocity
from sigmarl.constants import SPEED_SCALE

# Ego dimension added to the obstacle ellipse's semi-axis: "length" (paper
# text) or "width" (Fig. 3 caption)
RADIUS_VARIABLE = "SIGMARL_B4_RADIUS"


class CollisionConePairBaseline(PairBaseline):
    """B4 pair rows: one collision-cone barrier per pair.

    The per-agent representation is (x, y, v_x, v_y) of the vehicle center.
    """

    # The ego treats the other car as an uncontrolled obstacle
    CROSS_SHARE = 1.0
    # Regularization of |v| at zero relative velocity [m/s]
    SPEED_EPS = 1e-3 * SPEED_SCALE

    def __init__(self, ctrl):
        super().__init__(ctrl)
        dimension = tuned_env_value(
            ctrl.parameters, "b4_collision_cone", RADIUS_VARIABLE
        )
        if dimension not in ("length", "width"):
            raise ValueError(f"Unknown {RADIUS_VARIABLE} '{dimension}'.")
        ego = ctrl.length if dimension == "length" else ctrl.width
        self.radius = (
            float(ctrl.length) / math.sqrt(2.0)
            + 0.5 * float(ego)
            + float(ctrl.pair_safety_buffer)
        )
        # [m/s] h / speed_scale is in meters
        self.speed_scale = float(ctrl.v_max)

    def predict(self, state_np, u):
        return predict_center_velocity(self.ctrl, state_np, u)

    def value_and_gradients(self, x_a, x_b) -> tuple:
        p = np.asarray(x_b[0:2], dtype=np.float64) - x_a[0:2]
        v = np.asarray(x_b[2:4], dtype=np.float64) - x_a[2:4]
        q = math.sqrt(max(float(p @ p) - self.radius**2, 0.0))
        speed = math.sqrt(float(v @ v) + self.SPEED_EPS**2)
        speed_reg = speed - self.SPEED_EPS  # 0 at v = 0
        value = float(p @ v) + speed_reg * q
        grad_p = v + (speed_reg * p / q if q > 0.0 else 0.0)
        grad_v = p + q * v / speed
        grad_b = np.concatenate([grad_p, grad_v]) / self.speed_scale
        return value / self.speed_scale, -grad_b, grad_b

    def value(self, x_a, x_b) -> float:
        return self.value_and_gradients(x_a, x_b)[0]
