# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B6: backup CBF with one decay term per flow point (condition baseline).

Discrete-time counterpart of the backup CBFs of Gurriet et al. (IEEE Access
2020) and Chen, Jankovic, Santillo, Ames (CDC 2021), and of the decentralized
backup CBF of Chen, Singletary, Ames (IEEE L-CSS 2021), on the same exact
barriers, tails, linearization, exact check, repair, and fallback as the
backup-horizon rows of ours (sigmarl/backup_horizon_cbf.py). Unlike the other
baselines, B6 keeps our barriers and changes the condition:

  joint pair or lane:  m_e(Psi_t(u)) >= (1 - lambda) m_e(Psi^b_t),  t = 1..N
  cross pair:          sd(Psi_t(u_i), Psi^b_{j,t}) - eps_0
                       >= (1 - lambda / 2) (sd(Psi^b_{i,t}, Psi^b_{j,t}) - eps_0)

Psi^b_t is the backup rollout from the current state (the tail of the backup
control), so each flow point decays from its own value at the same absolute
time, instead of from the backup value min_t m_e(Psi^b_t) shared by all tail
steps (BH). The joint backup control satisfies the joint rows, so the
centralized condition keeps recursive feasibility; its right-hand sides are
never looser than those of BH for lambda < 1, and both reduce to the shield's
test for lambda = 1.

As published, the horizon is fixed: N = T for every QP instead of our adaptive
stopping horizon. T is the stopping horizon from the speed limit (the shortest
fixed horizon that does not cap the speed), so every tail has stopped by T and
the terminal condition of Chen et al. (reach the invariant set S_0 = standstill
by T) holds for every control and is omitted. A shorter T would need that
terminal condition, which would cap the speed.

As published, the QP holds only the flow-point rows: run B6 with
Parameters.is_two_step_rows = False (benchmark method b6_backup_cbf).

The cross-pair row is Chen et al.'s decentralized split: each vehicle takes
half of the decay of the pair's signed distance, with the other vehicle on its
backup tail. In continuous time, the chain rule splits the derivative of the
pair barrier exactly into the two vehicles' parts, which is the guarantee of
Chen et al. Over a finite step, however, the change of the signed distance when
both vehicles deviate from their tails is not the sum of the two one-sided
changes (a mixed term: the distance is nonlinear, and the separating axes of
rectangles turn with the vehicles' headings), so these discrete-time rows carry
no guarantee.
"""

import numpy as np

from sigmarl.backup_horizon_cbf import BackupHorizonCBF, signed_distance

# Share of the decay of a cross pair carried by each vehicle (Chen et al.)
CROSS_SHARE = 0.5


class BackupCBFPerPoint(BackupHorizonCBF):
    """Backup-horizon rows with per-flow-point decay (B6)."""

    KINDS = BackupHorizonCBF.KINDS + ("sdpair",)

    def horizon(self, speeds, params=None) -> int:
        """The fixed horizon T, whatever the speeds: the stopping horizon from
        the speed limits (nominal vehicles only)."""
        assert params is None, "B6 supports only the nominal vehicle parameters."
        ctrl = self.ctrl
        return ctrl._shield_horizon([float(ctrl.v_max), float(ctrl.v_min)])

    def _decay_rhs(self, ref_values: np.ndarray, lam) -> np.ndarray:
        """(1 - lambda) times the margin of each flow point [E, T]."""
        return ((1.0 - np.asarray(lam)) * ref_values).T

    def _build_cross_entries(self, block, ie, io, cross):
        """Rows of the ego on the signed distance to the other vehicle's
        backup tail, with half of the pair's per-point decay."""
        ref = block["ref"]
        n_t = ref.shape[0]
        lam = np.array([float(c[2]["lam"]) for c in cross])
        buffer = float(self.ctrl.pair_safety_buffer)
        distance, _ = signed_distance(
            ref[:, ie].reshape(-1, 3), ref[:, io].reshape(-1, 3), self.half
        )
        margin = (distance - buffer).reshape(n_t, -1)  # [T, K]
        rhs = ((1.0 - CROSS_SHARE * lam)[None, :] * margin).T
        self._add_entries(
            block,
            "sdpair",
            ie[:, None],
            [c[2]["slack_idx"] for c in cross],
            rhs,
            ego=ie,
            other_ref=ref[:, io],
            buffer=buffer,
        )
        self.stats["bh_pairs_unrecoverable"] += int((margin.min(axis=0) < 0.0).sum())

    def _kind_values(self, block, poses, kind: str, sub, variant_a=0, variant_b=0):
        """Values [len(sub), T]; "sdpair" is the ego's signed distance to the
        other vehicle's backup tail minus the buffer."""
        if kind != "sdpair":
            return super()._kind_values(block, poses, kind, sub, variant_a, variant_b)
        data = block[kind]
        ego = poses[:, data["ego"][sub], variant_a]
        distance, _ = signed_distance(
            ego.reshape(-1, 3), data["other_ref"][:, sub].reshape(-1, 3), self.half
        )
        return (distance.reshape(ego.shape[0], -1) - data["buffer"]).T
