# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from sigmarl.constants import LENGTH_SCALE
from sigmarl.map_manager import MapManager


class SoftminDistance:
    """
    Smooth squared-distance surrogate from a point to a polyline.
    """

    def __init__(self, scenario_type, map: MapManager, parameters=None):
        self.scenario_type: str = scenario_type
        self.map = map
        self.beta = float(getattr(parameters, "lane_softmin_beta", 20.0))
        self.eps = float(getattr(parameters, "lane_softmin_eps", 1e-3 * LENGTH_SCALE))
        self.tau_sq = float(
            getattr(parameters, "lane_softmin_tau_sq", 1e-4 * LENGTH_SCALE**2)
        )
        # "geometric": Hessian from the gradient and the boundary curvature;
        # "stencil": finite differences of the surrogate (legacy, see
        # geometric_hessians for why it is unreliable)
        self.hessian_mode = str(getattr(parameters, "lane_hessian_mode", "geometric"))

        if self.scenario_type == "cpm_intersection_extended":
            self.reference_paths = self.map.parser.reference_paths_intersection_extended
        elif self.scenario_type == "cpm_mixed":
            self.reference_paths = self.map.parser.reference_paths_intersection
        else:
            self.reference_paths = self.map.parser.reference_paths

        self._boundary_cache = []
        for ref_path in self.reference_paths:
            self._boundary_cache.append(
                {
                    "left": self._prepare_boundary(ref_path["left_boundary_shared"]),
                    "right": self._prepare_boundary(ref_path["right_boundary_shared"]),
                }
            )

    @staticmethod
    def _to_numpy(array_like):
        if isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().numpy()
        return np.asarray(array_like)

    @staticmethod
    def _sanitize_ref_id(ref_id) -> int:
        if isinstance(ref_id, torch.Tensor):
            return int(ref_id.detach().cpu().item())
        return int(ref_id)

    def _prepare_boundary(self, polyline):
        points = np.asarray(self._to_numpy(polyline), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Polyline must have shape [N, 2].")
        if points.shape[0] < 2:
            raise ValueError("Polyline must contain at least two points.")

        p1 = points[:-1]
        p2 = points[1:]
        a = p2 - p1
        denom = np.sum(a * a, axis=-1)
        valid = denom > 1e-12
        if not np.any(valid):
            raise ValueError("Polyline contains no valid non-degenerate segments.")

        return {
            "p1": p1[valid],
            "a": a[valid],
            "denom": denom[valid],
        }

    def _softplus(self, value):
        return np.logaddexp(0.0, self.beta * value) / self.beta

    def _smooth_clip(self, mu):
        return self._softplus(mu) - self._softplus(mu - 1.0)

    def _segment_surrogate_distance_sq(self, boundary, points):
        points = np.asarray(points, dtype=np.float64)
        diff = points[:, None, :] - boundary["p1"][None, :, :]
        mu = (
            np.sum(diff * boundary["a"][None, :, :], axis=-1)
            / boundary["denom"][None, :]
        )
        clip_mu = self._smooth_clip(mu)
        z = boundary["p1"][None, :, :] + clip_mu[..., None] * boundary["a"][None, :, :]
        residual = points[:, None, :] - z
        return np.sum(residual * residual, axis=-1) + self.eps * self.eps

    def _softmin_distance_sq(self, boundary, points):
        segment_distances_sq = self._segment_surrogate_distance_sq(boundary, points)
        min_distance_sq = np.min(segment_distances_sq, axis=-1, keepdims=True)
        stabilized = np.exp(-(segment_distances_sq - min_distance_sq) / self.tau_sq)
        return min_distance_sq[:, 0] - self.tau_sq * np.log(np.sum(stabilized, axis=-1))

    def _distances_to_boundaries(self, ref_id, points):
        boundary_cache = self._boundary_cache[self._sanitize_ref_id(ref_id)]
        points_np = np.asarray(self._to_numpy(points), dtype=np.float64).reshape(-1, 2)
        left = self._softmin_distance_sq(boundary_cache["left"], points_np)
        right = self._softmin_distance_sq(boundary_cache["right"], points_np)
        return left, right

    @staticmethod
    def _build_stencil_field(values):
        field = np.empty((3, 3), dtype=np.float64)
        field[1, 1] = values[0]
        field[1, 2] = values[1]
        field[2, 1] = values[2]
        field[1, 0] = values[3]
        field[0, 1] = values[4]
        field[2, 2] = values[5]
        field[0, 2] = values[6]
        field[2, 0] = values[7]
        field[0, 0] = values[8]
        return field

    def _distance_derivatives_from_stencil(self, values, dx, dy):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        center = values[0]
        x_plus = values[1]
        y_plus = values[2]
        x_minus = values[3]
        y_minus = values[4]
        x_plus_y_plus = values[5]
        x_plus_y_minus = values[6]
        x_minus_y_plus = values[7]
        x_minus_y_minus = values[8]

        mixed = (x_plus_y_plus - x_plus_y_minus - x_minus_y_plus + x_minus_y_minus) / (
            4.0 * dx * dy
        )
        grad = np.array(
            [
                (x_plus - x_minus) / (2.0 * dx),
                (y_plus - y_minus) / (2.0 * dy),
            ],
            dtype=np.float64,
        )
        hessian = np.array(
            [
                [(x_plus - 2.0 * center + x_minus) / (dx * dx), mixed],
                [mixed, (y_plus - 2.0 * center + y_minus) / (dy * dy)],
            ],
            dtype=np.float64,
        )
        return float(center), grad, hessian

    def _distance_derivatives_from_stencils(self, values, dx, dy):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 9:
            raise ValueError("values must have shape [N, 9].")

        center = values[:, 0]
        x_plus = values[:, 1]
        y_plus = values[:, 2]
        x_minus = values[:, 3]
        y_minus = values[:, 4]
        x_plus_y_plus = values[:, 5]
        x_plus_y_minus = values[:, 6]
        x_minus_y_plus = values[:, 7]
        x_minus_y_minus = values[:, 8]

        mixed = (x_plus_y_plus - x_plus_y_minus - x_minus_y_plus + x_minus_y_minus) / (
            4.0 * dx * dy
        )
        grad = np.stack(
            [
                (x_plus - x_minus) / (2.0 * dx),
                (y_plus - y_minus) / (2.0 * dy),
            ],
            axis=-1,
        )
        hessian = np.empty((values.shape[0], 2, 2), dtype=np.float64)
        hessian[:, 0, 0] = (x_plus - 2.0 * center + x_minus) / (dx * dx)
        hessian[:, 0, 1] = mixed
        hessian[:, 1, 0] = mixed
        hessian[:, 1, 1] = (y_plus - 2.0 * center + y_minus) / (dy * dy)
        return center.astype(np.float64, copy=False), grad, hessian

    def geometric_hessians(self, boundary, points, values, grads):
        """Hessians of the squared boundary distance from geometry.

        The finite-difference Hessian of the softmin surrogate is dominated by the
        blending between neighboring segments: on a straight boundary sampled
        every 4 cm, its tangential entry oscillates between about -0.15 and +0.3
        although it should be 0 (the normal entry is 2). Multiplied by the squared
        tangential speed, this ripple can exceed every other term of the
        second-order lane CBF. This method instead uses the exact Hessian of the
        squared distance to a curve with the local boundary curvature,

            H = 2 n n^T + 2 d kappa_s / (1 + kappa_s d) t t^T,

        where n is the unit gradient (pointing away from the boundary), t is
        perpendicular to it, d is the distance, and kappa_s is the boundary
        curvature at the foot point, positive where the boundary bulges toward the
        point and negative where it curves around it.

        Args:
            boundary: prepared boundary dict (see _prepare_boundary).
            points: [N, 2] stencil centers.
            values: [N] squared-distance surrogate at the centers.
            grads: [N, 2] gradients of the surrogate at the centers.

        Returns:
            [N, 2, 2] Hessians.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        grads = np.asarray(grads, dtype=np.float64).reshape(-1, 2)
        p1, a, denom = boundary["p1"], boundary["a"], boundary["denom"]
        seg_len = np.sqrt(denom)

        # Signed vertex curvature (positive for left turns), lightly smoothed
        kappa_vertex = np.zeros(a.shape[0] + 1, dtype=np.float64)
        if a.shape[0] > 1:
            cross = a[:-1, 0] * a[1:, 1] - a[:-1, 1] * a[1:, 0]
            dot = np.sum(a[:-1] * a[1:], axis=-1)
            kappa_vertex[1:-1] = np.arctan2(cross, dot) / (
                0.5 * (seg_len[:-1] + seg_len[1:])
            )
            padded = np.pad(kappa_vertex, 1, mode="edge")
            kappa_vertex = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

        diff = points[:, None, :] - p1[None, :, :]
        mu = np.clip(np.sum(diff * a[None, :, :], axis=-1) / denom[None, :], 0.0, 1.0)
        residual = diff - mu[..., None] * a[None, :, :]
        k = np.argmin(np.sum(residual * residual, axis=-1), axis=-1)
        mu_k = mu[np.arange(points.shape[0]), k]
        kappa_foot = (1.0 - mu_k) * kappa_vertex[k] + mu_k * kappa_vertex[k + 1]

        g_norm = np.linalg.norm(grads, axis=-1)
        n = grads / np.maximum(g_norm, 1e-12)[:, None]
        t = np.stack([-n[:, 1], n[:, 0]], axis=-1)
        a_k = a[k]
        side = np.sign(a_k[:, 0] * n[:, 1] - a_k[:, 1] * n[:, 0])
        # A left turn bends the boundary around the points on its left side
        kappa_s = -kappa_foot * side
        d = np.sqrt(np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0))
        tangential = np.clip(
            2.0 * d * kappa_s / np.maximum(1.0 + kappa_s * d, 0.1), -2.0, 2.0
        )
        return 2.0 * n[:, :, None] * n[:, None, :] + tangential[:, None, None] * (
            t[:, :, None] * t[:, None, :]
        )

    def derivatives_at_points(self, boundary, points, dx, dy):
        """Squared-distance surrogate, gradient, and Hessian at stencil centers.

        Args:
            boundary: prepared boundary dict.
            points: [N, 9, 2] stencil points ordered as
                [center, +x, +y, -x, -y, +x+y, +x-y, -x+y, -x-y].
            dx, dy: stencil steps.

        Returns:
            ([N] values, [N, 2] gradients, [N, 2, 2] Hessians).
        """
        points = np.asarray(points, dtype=np.float64)
        n_stencils = points.shape[0]
        values = self._softmin_distance_sq(
            boundary, points.reshape(n_stencils * 9, 2)
        ).reshape(n_stencils, 9)
        center, grad, hessian = self._distance_derivatives_from_stencils(values, dx, dy)
        if self.hessian_mode == "geometric":
            hessian = self.geometric_hessians(boundary, points[:, 0], center, grad)
        elif self.hessian_mode != "stencil":
            raise ValueError(f"Unknown lane_hessian_mode {self.hessian_mode!r}.")
        return center, grad, hessian

    def get_distance(self, ref_id, pos):
        """
        Keep the same interface as PseudoDistance.get_distance.
        """
        return self._distances_to_boundaries(ref_id, pos)

    def get_distance_with_derivatives(self, ref_id, query_points, dx, dy):
        """
        Compute the softmin squared surrogate and its first/second derivatives at the
        stencil center using recursive np.gradient evaluations on the 3x3 stencil.
        """
        points_np = np.asarray(self._to_numpy(query_points), dtype=np.float64)
        if points_np.shape != (9, 2):
            raise ValueError(
                "query_points must have shape [9, 2] ordered as "
                "[center, +x, +y, -x, -y, +x+y, +x-y, -x+y, -x-y]."
            )

        boundary_cache = self._boundary_cache[self._sanitize_ref_id(ref_id)]
        left = self.derivatives_at_points(
            boundary_cache["left"], points_np[None], dx, dy
        )
        right = self.derivatives_at_points(
            boundary_cache["right"], points_np[None], dx, dy
        )
        return tuple(x[0] for x in left), tuple(x[0] for x in right)
