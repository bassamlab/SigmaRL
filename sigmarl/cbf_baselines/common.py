# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Discrete-time rows shared by the SOTA CBF baselines (barrier swap).

A baseline barrier h enters the same discrete-time condition as the
rectangle_sat rows, h(x_next(u)) >= (1 - lambda) h(x_ref), where x_ref is the
state the previous control has already fixed and x_next the first state that
depends on the control (poses y at k+1 and z at k+2 of the exact two-step
Euler map for pose barriers, see CBFQP._lane_sat_predict). The row is
linearized once at the clipped nominal control u_lin:
  h(x_next(u_lin)) + dh/dx J (u - u_lin) + slack >= (1 - lambda) h(x_ref).

A decentralized (cross) pair knows only the ego control. The other agent is
predicted at zero input, and the ego takes its share s of the required change:
  h(x_ego(u), x0_other) >= (1 - s) h(x0) + s (1 - lambda) h(x_ref).
Each baseline uses the split of its paper (PairBaseline.CROSS_SHARE, e.g., 1/2
for reciprocal splits, 1 for an ego that treats the other as an obstacle).
"""

import math
from collections import defaultdict

import numpy as np

# Upper bound of one-sided QP rows (as in sigmarl/cbf_qp_proxsuite.py)
PROXQP_INF = 1e20


def append_linearized_row(
    ctrl, rows, lowers, uppers, n_vars, linear_terms, value_lin, rhs, slack_idx
):
    """Append value(u) >= rhs, with value linearized at the controls u_lin.

    Args:
        ctrl: the CBFQP controller (row writer and linearization margin).
        linear_terms: (QP control slice, d value / d u [2], u_lin [2]) per
            agent whose control is in the QP.
        value_lin: value at the linearization controls.
        rhs: required value.
        slack_idx: slack variable of the row (or None).
    """
    terms = []
    constant = float(value_lin)
    for ui_slice, grad_u, u_lin in linear_terms:
        grad_u = np.asarray(grad_u, dtype=np.float64)
        terms.append((ui_slice, grad_u))
        constant -= float(grad_u @ np.asarray(u_lin, dtype=np.float64))
    if slack_idx is not None:
        terms.append((slack_idx, 1.0))
    # The same tightening against linearization errors as the rectangle_sat rows
    lower = float(rhs) - constant + ctrl.lane_sat_linearization_margin
    ctrl._append_qp_row(rows, lowers, uppers, n_vars, terms, lower, PROXQP_INF)


class LaneBaseline:
    """Lane-boundary baseline of one CBFQP controller (one agent's rows).

    Subclasses implement ``side_constraints``; all rows of a side share the
    side's slack variable, as the rectangle_sat lane rows do.
    """

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.stats = defaultdict(int)

    def side_constraints(self, env_idx: int, agent_idx: int, pose_y, pose_z):
        """Constraints of both sides at the poses y (k+1) and z (k+2, under u_lin).

        Returns:
            One (side, h(y), [(value(z), d value / d pose_z [3]), ...]) per
            side with a boundary; each value(z) must be >= (1 - lambda) h(y).
        """
        raise NotImplementedError

    def append_rows(
        self,
        rows,
        lowers,
        uppers,
        n_vars: int,
        ui_slice: slice,
        state_np,
        u_lin,
        env_idx: int,
        agent_idx: int,
        slack_idx,
        lambda_idx,
    ):
        """Append the lane rows of one agent (see _append_lane_sat_rows)."""
        if lambda_idx is not None:
            raise NotImplementedError("Baselines support only a fixed lambda.")
        ctrl = self.ctrl
        pose_y, pose_z, jacobian = ctrl._lane_sat_predict(state_np, u_lin)
        decay = 1.0 - ctrl.lambda_ttcbf_predefined
        for side, h_y, constraints in self.side_constraints(
            env_idx, agent_idx, pose_y, pose_z
        ):
            # Diagnostics: evaluated sides and sides already violated at k+1
            self.stats["lane_sides"] += 1
            self.stats["lane_sides_negative"] += int(h_y < 0.0)
            for value_z, grad_pose in constraints:
                append_linearized_row(
                    ctrl,
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                    [(ui_slice, np.asarray(grad_pose) @ jacobian, u_lin)],
                    value_z,
                    decay * h_y,
                    slack_idx[side],
                )
                self.stats["lane_rows"] += 1


class PairBaseline:
    """Inter-vehicle baseline of one CBFQP controller.

    Subclasses define the barrier on a per-agent representation x of the
    state (e.g., the pose) and how the control moves it:
      ``predict(state, u)`` -> (x_ref, x_next(u), d x_next / d u),
      ``value(x_a, x_b)`` and ``value_and_gradients(x_a, x_b)``, or
      ``rows_at(x_a, x_b)`` for several rows (e.g., several witness points).
    """

    # Ego share of a decentralized pair in the baseline's paper (None: the
    # pipeline's share, Parameters.cross_cbf_responsibility)
    CROSS_SHARE = None

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.stats = defaultdict(int)

    def predict(self, state_np, u):
        """Pose-based default: poses y (k+1) and z (k+2) of the Euler map."""
        return self.ctrl._lane_sat_predict(state_np, u)

    def value(self, x_a, x_b) -> float:
        raise NotImplementedError

    def value_and_gradients(self, x_a, x_b) -> tuple:
        """(h, d h / d x_a, d h / d x_b)."""
        raise NotImplementedError

    def rows_at(self, x_a, x_b) -> list:
        """Constraint rows [(value, d value / d x_a, d value / d x_b)] at the
        linearization point; every value must reach the pair's right side."""
        return [self.value_and_gradients(x_a, x_b)]

    def append_rows(
        self, pairs, states_np, u_lin, fixed_agents, rows, lowers, uppers, n_vars
    ):
        """Append the rows of every pair (see CBFQP._append_pair_sat_rows for
        the arguments). An agent without a QP control is predicted at zero
        input."""
        agents = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
        zero = np.zeros(2)
        x_ref, x_zero, lin = {}, {}, {}
        for agent in agents:
            x_ref[agent], x_zero[agent], _ = self.predict(states_np[agent], zero)
            if agent in u_lin:
                _, x_lin, jacobian = self.predict(states_np[agent], u_lin[agent])
                lin[agent] = (x_lin, jacobian, np.asarray(u_lin[agent]))

        for pair in pairs:
            if pair["lambda_idx"] is not None:
                raise NotImplementedError("Baselines support only a fixed lambda.")
            a, b = pair["a"], pair["b"]
            x_next = []
            for agent, ui_slice in zip((a, b), pair["slices"]):
                x_next.append(lin[agent][0] if ui_slice is not None else x_zero[agent])
            h_ref = self.value(x_ref[a], x_ref[b])
            self.stats["pairs"] += 1
            self.stats["pairs_negative"] += int(h_ref < 0.0)
            decay_rhs = (1.0 - float(pair["lam"])) * h_ref
            share = pair["share"]
            # The published split, except for a fully responsible ego (share 1,
            # e.g., against a log-replayed vehicle)
            if share is not None and share < 1.0 and self.CROSS_SHARE is not None:
                share = self.CROSS_SHARE
            if share is None:
                rhs = decay_rhs
            else:
                h_zero = self.value(x_zero[a], x_zero[b])
                rhs = (1.0 - share) * h_zero + share * decay_rhs
            for h_lin, grad_a, grad_b in self.rows_at(*x_next):
                linear_terms = []
                for agent, ui_slice, grad in zip(
                    (a, b), pair["slices"], (grad_a, grad_b)
                ):
                    if ui_slice is None:
                        continue
                    _, jacobian, u = lin[agent]
                    linear_terms.append((ui_slice, np.asarray(grad) @ jacobian, u))
                append_linearized_row(
                    self.ctrl,
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                    linear_terms,
                    h_lin,
                    rhs,
                    pair["slack_idx"],
                )
                self.stats["pair_rows"] += 1


def predict_center_velocity(ctrl, state_np, u):
    """One-step prediction of the center position and velocity (x, y, v_x, v_y).

    The position at k+1 does not depend on u, the velocity
    v_{k+1} [cos(psi_{k+1} + beta_{k+1}), sin(...)] does (explicit Euler, as
    the simulator).

    Returns:
        x_k [4], x_{k+1}(u) [4], and d x_{k+1} / d u [4, 2].
    """
    dt = ctrl.dt
    k = ctrl.l_r / ctrl.l_wb
    x, y, psi, v, delta = (float(value) for value in state_np[:5])
    beta = math.atan(k * math.tan(delta))
    x_ref = np.array([x, y, v * math.cos(psi + beta), v * math.sin(psi + beta)])
    pose_y, _, _ = ctrl._lane_sat_predict(state_np, u)
    v1 = v + dt * float(u[0])
    delta1 = delta + dt * float(u[1])
    tan1 = math.tan(delta1)
    beta1 = math.atan(k * tan1)
    dbeta_domega = dt * k / (math.cos(delta1) ** 2 * (1.0 + (k * tan1) ** 2))
    course = pose_y[2] + beta1
    cos_c, sin_c = math.cos(course), math.sin(course)
    x_next = np.array([pose_y[0], pose_y[1], v1 * cos_c, v1 * sin_c])
    jacobian = np.zeros((4, 2))
    jacobian[2:4, 0] = dt * np.array([cos_c, sin_c])
    jacobian[2:4, 1] = v1 * dbeta_domega * np.array([-sin_c, cos_c])
    return x_ref, x_next, jacobian


class UnfilteredLaneBaseline(LaneBaseline):
    """No lane rows (unfiltered reference)."""

    def side_constraints(self, env_idx, agent_idx, pose_y, pose_z):
        return []


class UnfilteredPairBaseline(PairBaseline):
    """No pair rows (unfiltered reference)."""

    def append_rows(
        self, pairs, states_np, u_lin, fixed_agents, rows, lowers, uppers, n_vars
    ):
        return


def softmin(values: np.ndarray, kappa: float, axis: int = -1) -> tuple:
    """Log-sum-exp soft minimum (a lower bound of the min) and its weights.

    -inf/+inf entries are allowed (+inf entries get zero weight).
    """
    neg = -kappa * values
    peak = np.max(neg, axis=axis, keepdims=True)
    expo = np.exp(neg - peak)
    total = np.sum(expo, axis=axis, keepdims=True)
    value = -(np.squeeze(peak, axis) + np.log(np.squeeze(total, axis))) / kappa
    return value, expo / total


def softmax_lower(values: np.ndarray, kappa: float, count, axis: int = -1) -> tuple:
    """Log-sum-exp soft maximum minus ln(count)/kappa (a lower bound of the
    max) and its weights; count is the number of finite entries."""
    pos = kappa * values
    peak = np.max(pos, axis=axis, keepdims=True)
    expo = np.exp(pos - peak)
    total = np.sum(expo, axis=axis, keepdims=True)
    value = (
        np.squeeze(peak, axis) + np.log(np.squeeze(total, axis)) - np.log(count)
    ) / kappa
    return value, expo / total
