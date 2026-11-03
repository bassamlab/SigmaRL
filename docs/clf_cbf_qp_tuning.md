# CLF-CBF-QP Tuning (October 2026)

This note records five changes to the CLF nominal controller and the CBF-QP in
`sigmarl/cbf_qp_proxsuite.py`, why they were made, and how to get the old
behavior back. They were found while comparing centralized and fully
decentralized CBF on `cpm_mixed`, `cpm_entire` (lab scale) and
`round_location0` (full scale). Before these changes, most lane collisions came
from the controller, not from the scenarios.

| Parameter (`Parameters`) | New default | Legacy value | CLI flag of `main_testing_clf_cbf_qp.py` |
|---|---|---|---|
| `lane_hessian_mode` | `"geometric"` | `"stencil"` | `--lane_hessian_mode` |
| `clf_nominal_mode` | `"pure_pursuit"` | `"heading_rate"` | `--clf_nominal_mode` |
| `is_cbf_limit_acc_to_nominal` | `True` | `False` | `--[no-]is_cbf_limit_acc_to_nominal` |
| `cross_cbf_responsibility` | `"follower"` | `"half"` | `--cross_cbf_responsibility` |

`clf_nominal_mode` only matters when `nom_controller_type="clf"`. The other
three also change CBF-filtered RL rollouts.

The tuning values are local attributes set in `CBFQP.initialize_params`, not
`Parameters` fields:

| Attribute | Value | Legacy value |
|---|---|---|
| `clf_steering_gain` | `5.0` 1/s | — (`"pure_pursuit"` only) |
| `clf_lookahead_index` | `1.0` | — (`"pure_pursuit"` only) |
| `clf_max_lat_acc` | `0.7 * ACCELERATION_SCALE` m/s² | — (`"pure_pursuit"` only) |
| `nom_weight` | `diag(1, 1)` | `diag(10, 1)` |

## 1. Lane CBF Hessian (bug fix)

The second-order lane TTCBF needs the Hessian `H` of the squared boundary
distance, because `ddot h` contains `v^T H v`. `H` used to be computed by finite
differences of the softmin surrogate (`SoftminDistance._softmin_distance_sq`).
The softmin blends neighboring segments, so this surrogate ripples along the
boundary. On a straight boundary sampled every 8 cm, the tangential entry of the
finite-difference `H` oscillated between -0.15 and +0.30 instead of 0 (the normal
entry is 2). Multiplied by the squared tangential speed, the ripple was about
10 times larger than the real approach term. The CBF therefore often predicted
that a vehicle drifting toward a boundary would turn away from it.

`SoftminDistance.geometric_hessians` now builds `H` from the gradient direction
`n` and the boundary curvature at the foot point:
`H = 2 n n^T + 2 d kappa_s / (1 + kappa_s d) t t^T`. Its error on a sampled
circle is about 3%, while the stencil Hessian was off by up to 300% and often had
the wrong sign.

## 2. Nominal CLF steering law

`"heading_rate"` sets the steering rate to `k * heading error`. The steering
angle then integrates the heading error without damping and overshoots in every
curve. That overshoot caused the earlier "weird left turns" in the rounD
roundabout and the cuts across tight right turns in `cpm_entire`.

`"pure_pursuit"` (`CBFQP._clf_nominal_for_agent`) works as follows:

- It computes a pure-pursuit steering target from the rear axle to the
  short-term reference point at index `clf_lookahead_index`.
- It tracks that target with `steering_rate = clf_steering_gain * (delta_des - delta)`.
- It slows down to `sqrt(clf_max_lat_acc / kappa)` for the sharpest curvature
  ahead.
- The CLF heading row of the QP uses the steering-angle error, so the row stays
  consistent with the nominal control.

## 3. No acceleration beyond the nominal control

With a vehicle approaching from behind, the min-deviation QP could satisfy the
pair CBF by speeding up the leader. Leaders went to full speed and could no
longer make the next curve. The QP's acceleration upper bound is now
`max(u_nom_acc, 0)`, so the filter may brake but not accelerate beyond the
nominal control.

## 4. Acceleration tracking weight

The QP cost was `||diag(10, 1)(u - u_nom)||^2`, which makes braking 100 times
more expensive than steering. The QP then used steering even for constraints
that steering barely affects. Examples:

- It steered vehicles into a lane boundary to clear a merging neighbor, instead
  of braking.
- At crawl speed it made the steering chatter between its limits.

An acceleration weight of 1 (`nom_weight = diag(1, 1)`) removes this. The
weight also fixes an inconsistency: the fully
decentralized QP used `diag(10, 1)` itself as the quadratic weight (that is,
`diag(sqrt(10), 1)^2`), while the centralized and grouped QPs used
`diag(10, 1)^2`. All modes now share `nom_weight^T nom_weight`.

## 5. Responsibility split of decentralized pair CBFs

The fully decentralized (and grouped cross-group) pair constraints split each
centralized pair constraint into two halves (`_cross_cbf_drift_scale`). With
`"follower"`, `_cross_cbf_responsibility_share` gives the larger share to the
agent that drives toward the other one:

- `f_i = max(0, (p_j - p_i) . heading_i)`
- `share_i = (f_i + L) / (f_i + f_j + 2 L)`, where `L` is the vehicle length.

The two directed constraints still sum to the centralized constraint.

## Results

These are 60 s runs, seeds 1-5, from `main_testing_clf_cbf_qp.py` with default
backup settings. Each cell sums the 5 seeds and shows legacy → new.
`cen` = `--max_group_size none`, `dec` = `--max_group_size 1`.

| Scenario | Mode | Lane collisions | Goals reached | Deadlocks | Mean speed |
|---|---|---|---|---|---|
| `cpm_mixed`, 6 agents | cen | 28 → 0 | 184 → 282 | 20 → 4 | 0.27 → 0.34 m/s |
| `cpm_mixed`, 6 agents | dec | 21 → 0 | 167 → 258 | 15 → 0 | 0.24 → 0.30 m/s |
| `cpm_entire`, 10 agents | cen | 111 → 0 | loop paths | 2 → 5 | 0.40 → 0.51 m/s |
| `cpm_entire`, 10 agents | dec | 92 → 13 | loop paths | 1 → 0 | 0.39 → 0.50 m/s |
| `round_location0`, 6 agents | cen | 7 → 0 | 74 → 108 | 20 → 4 | 4.66 → 4.86 m/s |
| `round_location0`, 6 agents | dec | 6 → 0 | 92 → 97 | 0 → 3 | 4.77 → 4.89 m/s |
| `round_location0`, 10 agents | cen | 10 → 0 | 95 → 165 | 25 → 7 | 4.39 → 4.77 m/s |
| `round_location0`, 10 agents | dec | 11 → 1 | 152 → 164 | 2 → 2 | 5.05 → 4.79 m/s |

There were no agent-agent collisions in either version. The remaining
decentralized lane contacts in `cpm_entire` happen at crawl speed in the crowded
central crossing and at lane splits, while the lightweight backup filter is in
recovery mode.
