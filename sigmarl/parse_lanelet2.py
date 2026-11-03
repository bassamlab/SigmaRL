# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import xml.etree.ElementTree as ET
from importlib import resources

import numpy as np
import torch

from sigmarl.constants import SCENARIOS
from sigmarl.parse_map_base import ParseMapBase
from sigmarl.parse_osm import ParseOSM
from sigmarl.parse_xml import ParseXML

# Lanelet subtypes that cars drive on (Lanelet2 traffic rules for vehicles). Bus and emergency lanes are reserved
# for other vehicles, and lanelets without subtype are not drivable.
DRIVABLE_SUBTYPES = {"road", "highway", "play_street"}
# Lanelet ends narrower than this many lane widths are lane tapers, where a lane begins or ends beside another lane
# (e.g., on- and off-ramps of exiD). Vehicles change lanes there, so routes neither start nor end at a taper.
TAPER_WIDTH_LANE_WIDTHS = 0.25
# Bound types that vehicles may cross to change to the neighboring lane
LANE_CHANGE_LINE_TYPES = {("line_thin", "dashed"), ("line_thick", "dashed")}
# Ways that vehicles cannot cross; (type, None) matches any subtype
PHYSICAL_BORDER_TYPES = {
    ("road_border", None),
    ("curbstone", None),
    ("line_thin", "solid"),
    ("line_thick", "solid"),
}
# How far (in lane widths) a virtual bound may be moved out to the next physical border
PHYSICAL_BORDER_SEARCH_LANE_WIDTHS = 2.5
# Side of the road edge in right-hand traffic. Only there a virtual bound is moved to the physical border; on the
# other side, virtual bounds separate a lane from crossing or merging traffic (e.g. the circle of a roundabout).
ROAD_EDGE_SIDE = "right"
# Largest change of a shared boundary's lateral offset per meter of center line. Steeper steps (e.g., where the
# physical-border search hits a curb at one station but not the next) become walls across the lane for the
# rectangle lane CBF and get large resampling buffers there
MAX_BOUNDARY_OFFSET_SLOPE = 1.0

# WGS84 ellipsoid and UTM constants
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_UTM_K0 = 0.9996
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)
_WGS84_EP2 = _WGS84_E2 / (1 - _WGS84_E2)


def latlon_to_utm(lat, lon):
    """Project WGS84 latitude/longitude [deg] to UTM easting/northing [m] (northern hemisphere).

    The UTM zone follows from the mean longitude. Transverse Mercator series from Snyder,
    "Map Projections - A Working Manual" (USGS 1987), pp. 61-64.
    """
    zone = int((np.mean(lon) + 180) // 6) + 1
    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)
    e2, ep2 = _WGS84_E2, _WGS84_EP2
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    n = _WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = ep2 * np.cos(lat) ** 2
    a = np.cos(lat) * (lon - central_meridian)
    meridian_arc = _WGS84_A * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat)
        - (35 * e2**3 / 3072) * np.sin(6 * lat)
    )
    easting = (
        _UTM_K0
        * n
        * (
            a
            + (1 - t + c) * a**3 / 6
            + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * a**5 / 120
        )
        + 500000.0
    )
    northing = _UTM_K0 * (
        meridian_arc
        + n
        * np.tan(lat)
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * a**6 / 720
        )
    )
    return easting, northing


class ParseLanelet2(ParseOSM):
    """
    Parse a Lanelet2 OSM map, e.g., one exported from a drone dataset (https://github.com/fzi-forschungszentrum-informatik/Lanelet2).

    Unlike the JOSM center-line maps read by `ParseOSM`, a Lanelet2 lanelet is a relation with a "left" and a
    "right" bound way, so lanes keep their real, varying width. The parser provides the same data as `ParseOSM`:
    - Coordinates: latitude/longitude are projected to UTM, shifted by `utm_origin`, scaled by `scale` of the
      scenario (e.g., 1/18 for CPM Lab vehicles) and by `scale_map`, and shifted into the positive quadrant.
      `utm_to_map` applies the same transform, e.g., to recorded trajectories.
    - Lanelets: drivable lanelets only (`DRIVABLE_SUBTYPES`), renumbered 1..N in the order of their absolute OSM
      IDs (`lanelet_osm_ids` maps index to OSM ID). Bounds are oriented in driving direction with Lanelet2's rule
      and resampled with the scenario's `point_spacing`, so left boundary, right boundary, and center line align
      by index.
    - Topology: a successor starts at both end nodes of a lanelet (Lanelet2's rule). Lanes that share a bound way
      in the same direction across a dashed line widen each other's shared boundaries.
    - Reference paths: `reference_paths_ids` of the scenario (lists of OSM lanelet IDs) if given; otherwise all
      routes from lanelets without predecessor to lanelets without successor that visit no lanelet twice and
      neither start nor end at a lane taper.
    """

    def __init__(self, scenario_type, device, **kwargs):
        self._scale_map = kwargs.pop("scale_map", 1.0)
        ParseMapBase.__init__(self, scenario_type, device, **kwargs)

        scenario = SCENARIOS[scenario_type]
        self._fig_title = scenario["name"]
        self._utm_origin = np.array(scenario["utm_origin"], dtype=float)
        self._point_spacing = scenario["point_spacing"] * self._scale_map
        self._reference_paths_osm_ids = scenario.get("reference_paths_ids")

        self._parse_map_file()
        self._process_map_data()
        self._find_topology()
        self._compute_shared_boundaries()
        self._get_reference_paths()
        self._compute_pseudo_tangent_vectors()
        self._determine_neighboring_lanelets()

        self._get_map_dimension()

        if self._is_visualize_map:
            self.visualize_map()

    def utm_to_map(self, xy_utm):
        """Transform UTM coordinates [m] (array [..., 2]) to map coordinates."""
        return (
            np.asarray(xy_utm) - self._utm_origin
        ) * self._map_factor + self.map_offset

    def _parse_map_file(self):
        """Read nodes, ways, and road lanelets, and project the nodes to map coordinates."""
        if not resources.is_resource("sigmarl.scenarios.assets.maps", self._map_path):
            # The maps of the drone datasets may not be redistributed, so the
            # repository does not include them (docs/lanelet2_maps.md)
            raise FileNotFoundError(
                f"Map '{self._map_path}' not found in sigmarl/scenarios/assets/maps/. "
                "Generate it from the downloaded dataset with "
                "scripts/lanelet2_maps/fix_lanelet_map.py (highD: "
                "make_highd_lanelet_maps.py); see docs/lanelet2_maps.md."
            )
        with resources.open_binary(
            "sigmarl.scenarios.assets.maps", self._map_path
        ) as map_file:
            root = ET.parse(map_file).getroot()

        node_ids = [int(node.get("id")) for node in root.findall("node")]
        lat = [float(node.get("lat")) for node in root.findall("node")]
        lon = [float(node.get("lon")) for node in root.findall("node")]
        easting, northing = latlon_to_utm(lat, lon)

        self._ways = {}
        for way in root.findall("way"):
            self._ways[int(way.get("id"))] = {
                "nodes": [int(nd.get("ref")) for nd in way.findall("nd")],
                "tags": {tag.get("k"): tag.get("v") for tag in way.findall("tag")},
            }

        self._lanelet_bound_ways = {}  # OSM lanelet ID -> (left way ID, right way ID)
        for relation in root.findall("relation"):
            tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
            if (
                tags.get("type") != "lanelet"
                or tags.get("subtype") not in DRIVABLE_SUBTYPES
            ):
                continue
            roles = {
                member.get("role"): int(member.get("ref"))
                for member in relation.findall("member")
                if member.get("type") == "way"
            }
            self._lanelet_bound_ways[int(relation.get("id"))] = (
                roles["left"],
                roles["right"],
            )

        # Scale and shift so that the road lies in the positive quadrant with a margin
        self._map_factor = self._scale * self._scale_map
        xy = (np.stack([easting, northing], axis=1) - self._utm_origin) * (
            self._map_factor
        )
        road_nodes = {
            n
            for ways in self._lanelet_bound_ways.values()
            for way_id in ways
            for n in self._ways[way_id]["nodes"]
        }
        is_road = np.array([n in road_nodes for n in node_ids])
        margin = 2 * SCENARIOS[self._scenario_type]["lane_width"] * self._scale_map
        self.map_offset = margin - xy[is_road].min(axis=0)
        self._nodes = dict(zip(node_ids, xy + self.map_offset))

    def _orient_bounds(self, left_nodes, right_nodes):
        """Orient both bounds in driving direction like Lanelet2: the left bound is inverted if the right bound
        is on its left, and the right bound is inverted if the left bound is on its right."""

        def side(line, point):
            # > 0 if the point is left of the closest segment of the line
            i = min(int(np.argmin(np.linalg.norm(line - point, axis=1))), len(line) - 2)
            a, b = line[i], line[i + 1]
            return (b - a)[0] * (point - a)[1] - (b - a)[1] * (point - a)[0]

        def middle(line):
            return line[len(line) // 2] if len(line) > 2 else line.mean(axis=0)

        left = np.array([self._nodes[n] for n in left_nodes])
        right = np.array([self._nodes[n] for n in right_nodes])
        if side(left, middle(right)) > 0:
            left_nodes, left = left_nodes[::-1], left[::-1]
        if side(right, middle(left)) < 0:
            right_nodes = right_nodes[::-1]
        return list(left_nodes), list(right_nodes)

    def _process_map_data(self):
        """Compute the oriented, resampled bounds and the center line of each lanelet."""
        self.lanelet_osm_ids = sorted(self._lanelet_bound_ways, key=abs)
        self._bound_nodes = []  # (left node IDs, right node IDs) in driving direction
        self.lanelets_all = []
        for osm_id in self.lanelet_osm_ids:
            left_way, right_way = self._lanelet_bound_ways[osm_id]
            left_nodes, right_nodes = self._orient_bounds(
                self._ways[left_way]["nodes"], self._ways[right_way]["nodes"]
            )
            self._bound_nodes.append((left_nodes, right_nodes))
            left = torch.tensor(
                np.array([self._nodes[n] for n in left_nodes]),
                device=self._device,
                dtype=torch.float32,
            )
            right = torch.tensor(
                np.array([self._nodes[n] for n in right_nodes]),
                device=self._device,
                dtype=torch.float32,
            )
            length = max(
                torch.diff(left, dim=0).norm(dim=1).sum(),
                torch.diff(right, dim=0).norm(dim=1).sum(),
            )
            num_points = max(2, math.ceil(length / self._point_spacing) + 1)
            left = ParseXML._resample_polyline_to_num_points(left, num_points)
            right = ParseXML._resample_polyline_to_num_points(right, num_points)
            center_line = (left + right) / 2
            (
                center_line_yaw,
                center_line_vec_normalized,
                center_line_vec_mean_length,
            ) = self._compute_center_line_info(center_line)
            self.lanelets_all.append(
                {
                    "osm_id": osm_id,
                    "center_line": center_line,
                    "center_line_yaw": center_line_yaw,
                    "center_line_vec_normalized": center_line_vec_normalized,
                    "center_line_vec_mean_length": center_line_vec_mean_length,
                    "left_boundary": left,
                    "right_boundary": right,
                    "left_boundary_shared": left,  # Widened in `_compute_shared_boundaries`
                    "right_boundary_shared": right,
                    "predecessor": [],  # Lanelet IDs (1..N)
                    "successor": [],
                    "left_neighbor": None,  # Lanelet ID of the same-direction lane to the left
                    "right_neighbor": None,
                }
            )

    def _find_topology(self):
        """Find successors/predecessors (shared end nodes) and same-direction lateral neighbors."""
        starts = {}
        for idx, (left_nodes, right_nodes) in enumerate(self._bound_nodes):
            starts.setdefault((left_nodes[0], right_nodes[0]), []).append(idx)
        right_ways = {}
        for idx, osm_id in enumerate(self.lanelet_osm_ids):
            right_ways.setdefault(self._lanelet_bound_ways[osm_id][1], []).append(idx)

        for idx, (left_nodes, right_nodes) in enumerate(self._bound_nodes):
            for succ_idx in starts.get((left_nodes[-1], right_nodes[-1]), []):
                self.lanelets_all[idx]["successor"].append(succ_idx + 1)
                self.lanelets_all[succ_idx]["predecessor"].append(idx + 1)

            left_way = self._lanelet_bound_ways[self.lanelet_osm_ids[idx]][0]
            for other_idx in right_ways.get(left_way, []):
                if self._bound_nodes[other_idx][1][0] == left_nodes[0]:
                    # The other lane's right bound is this lane's left bound in the same direction
                    self.lanelets_all[idx]["left_neighbor"] = other_idx + 1
                    self.lanelets_all[other_idx]["right_neighbor"] = idx + 1

    def _is_lane_change_allowed(self, idx, side):
        way_id = self._lanelet_bound_ways[self.lanelet_osm_ids[idx]][
            0 if side == "left" else 1
        ]
        tags = self._ways[way_id]["tags"]
        return (tags.get("type"), tags.get("subtype")) in LANE_CHANGE_LINE_TYPES

    def _compute_shared_boundaries(self):
        """Widen each lanelet's boundaries to the outer bounds of the lanes it may change to.

        The outer bound is projected onto the stations of the lanelet's center line, so the shared boundaries
        stay index-aligned with the center line. Where the outer bound on the road-edge side is a virtual line (no
        marking, e.g. where a lane splits off), the boundary is moved to the nearest physical border behind it.
        """
        self._physical_border_segments = self._get_physical_border_segments()
        for idx, lanelet in enumerate(self.lanelets_all):
            center_line = lanelet["center_line"]
            progress = ParseXML._get_polyline_progress_ratios(center_line)
            for side in ("left", "right"):
                outer_idx = idx
                while self.lanelets_all[outer_idx][
                    f"{side}_neighbor"
                ] is not None and self._is_lane_change_allowed(outer_idx, side):
                    outer_idx = self.lanelets_all[outer_idx][f"{side}_neighbor"] - 1
                if outer_idx != idx:
                    lanelet[
                        f"{side}_boundary_shared"
                    ] = ParseXML._project_points_to_polyline(
                        points=center_line,
                        polyline=self.lanelets_all[outer_idx][f"{side}_boundary"],
                        target_progress_ratios=progress,
                    )
                outer_way = self._lanelet_bound_ways[self.lanelet_osm_ids[outer_idx]][
                    0 if side == "left" else 1
                ]
                if (
                    side == ROAD_EDGE_SIDE
                    and self._ways[outer_way]["tags"].get("type") == "virtual"
                ):
                    lanelet[f"{side}_boundary_shared"] = self._move_to_physical_border(
                        center_line, lanelet[f"{side}_boundary_shared"], side
                    )

    def _get_physical_border_segments(self):
        """Segments [S, 2, 2] of all ways that vehicles cannot cross."""
        segments = []
        for way in self._ways.values():
            tags = way["tags"]
            if (tags.get("type"), tags.get("subtype")) in PHYSICAL_BORDER_TYPES or (
                tags.get("type"),
                None,
            ) in PHYSICAL_BORDER_TYPES:
                points = np.array([self._nodes[n] for n in way["nodes"]])
                segments.append(np.stack([points[:-1], points[1:]], axis=1))
        return np.concatenate(segments)

    def _move_to_physical_border(self, center_line, boundary, side):
        """Move boundary points laterally (perpendicular to the center line) to the first physical border behind them.

        Only borders at or beyond the current boundary count, so the boundary only moves outward. Physical lines
        inside the lane (a solid line or a low curb that the lane crosses) would otherwise make the corridor
        narrower than the lane. Points whose lateral ray finds no such border within
        `PHYSICAL_BORDER_SEARCH_LANE_WIDTHS` lane widths keep their position.
        """
        center = center_line.cpu().numpy().astype(float)
        tangent = np.gradient(center, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        if side == "right":
            normal = -normal

        def cross(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        # Ray center + t * normal against segment a + u * (b - a)
        a = self._physical_border_segments[None, :, 0]
        e = self._physical_border_segments[None, :, 1] - a
        offset = a - center[:, None]
        denom = cross(normal[:, None], e)
        current = np.linalg.norm(boundary.cpu().numpy() - center, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(offset, e) / denom
            u = cross(offset, normal[:, None]) / denom
        is_behind = t >= current[:, None] - 1e-6
        t = np.where(
            (np.abs(denom) > 1e-12) & is_behind & (u >= 0) & (u <= 1), t, np.inf
        )
        hit = t.min(axis=1)
        max_distance = (
            PHYSICAL_BORDER_SEARCH_LANE_WIDTHS
            * SCENARIOS[self._scenario_type]["lane_width"]
            * self._scale_map
        )
        distance = np.where(hit <= max_distance, hit, current)
        return torch.tensor(
            center + normal * distance[:, None],
            device=self._device,
            dtype=torch.float32,
        )

    def _enumerate_routes(self):
        """All routes (lists of lanelet IDs) from lanelets without predecessor to lanelets without successor."""
        routes = []

        def extend(route):
            successors = self.lanelets_all[route[-1] - 1]["successor"]
            if not successors:
                routes.append(route)
            for succ_id in successors:
                if succ_id not in route:
                    extend(route + [succ_id])

        for idx, lanelet in enumerate(self.lanelets_all):
            if not lanelet["predecessor"]:
                extend([idx + 1])
        return routes

    def _is_route_self_crossing(self, route):
        """Whether the route's center line returns to a place it passed much earlier (more than one round)."""
        center_line = torch.cat(
            [self.lanelets_all[route[0] - 1]["center_line"]]
            + [self.lanelets_all[i - 1]["center_line"][1:] for i in route[1:]]
        )
        lane_width = SCENARIOS[self._scenario_type]["lane_width"] * self._scale_map
        arc_length = torch.cat(
            [torch.zeros(1), torch.diff(center_line, dim=0).norm(dim=1).cumsum(0)]
        )
        is_close = torch.cdist(center_line, center_line) < lane_width / 2
        is_far_along = (
            arc_length[None, :] - arc_length[:, None]
        ).abs() > 10 * lane_width
        return bool((is_close & is_far_along).any())

    def _is_route_at_taper(self, route):
        """Whether the route starts or ends where its lanelet's bounds (almost) meet."""
        first_left, first_right = self._bound_nodes[route[0] - 1]
        last_left, last_right = self._bound_nodes[route[-1] - 1]
        lane_width = SCENARIOS[self._scenario_type]["lane_width"] * self._scale_map
        return any(
            np.linalg.norm(self._nodes[a] - self._nodes[b])
            < TAPER_WIDTH_LANE_WIDTHS * lane_width
            for a, b in (
                (first_left[0], first_right[0]),
                (last_left[-1], last_right[-1]),
            )
        )

    @staticmethod
    def _blend_jump(polyline, joint, n_blend=4):
        """Linearly blend a polyline over `n_blend` points on each side of index `joint` to remove a jump."""
        i0, i1 = max(joint - n_blend, 0), min(joint + n_blend, polyline.shape[0] - 1)
        weights = torch.linspace(
            0, 1, i1 - i0 + 1, device=polyline.device, dtype=polyline.dtype
        ).unsqueeze(1)
        polyline[i0 : i1 + 1] = (1 - weights) * polyline[i0] + weights * polyline[i1]

    @staticmethod
    def _limit_offset_slope(center_line, boundary, max_slope=MAX_BOUNDARY_OFFSET_SLOPE):
        """Limit how fast a boundary's lateral offset from the center line changes along the center line.

        The offset is replaced by its largest minorant with slope at most `max_slope` per meter of center line, so
        the boundary only moves inward: a widening starts at its step and ramps outward, a narrowing ramps inward
        before its step, and a one-station spike shrinks to a bump of height `max_slope` times the point spacing.
        Each point moves along its own offset vector.
        """
        vectors = boundary - center_line
        offset = vectors.norm(dim=1)
        limit = max_slope * torch.diff(center_line, dim=0).norm(dim=1)
        limited = offset.clone()
        for k in range(1, limited.shape[0]):
            limited[k] = torch.minimum(limited[k], limited[k - 1] + limit[k - 1])
        for k in range(limited.shape[0] - 2, -1, -1):
            limited[k] = torch.minimum(limited[k], limited[k + 1] + limit[k])
        scale = torch.where(offset > 1e-9, limited / offset.clamp_min(1e-9), 1.0)
        return center_line + vectors * scale.unsqueeze(1)

    def _get_reference_paths(self):
        """Concatenate the lanelets of each route into a reference path."""
        if self._reference_paths_osm_ids is not None:
            osm_to_id = {osm_id: i + 1 for i, osm_id in enumerate(self.lanelet_osm_ids)}
            routes = [
                [osm_to_id[osm_id] for osm_id in route]
                for route in self._reference_paths_osm_ids
            ]
        else:
            routes = self._enumerate_routes()
            n_routes = len(routes)
            routes = [r for r in routes if not self._is_route_self_crossing(r)]
            n_self_crossing = n_routes - len(routes)
            routes = [r for r in routes if not self._is_route_at_taper(r)]
            print(
                f"[INFO] {n_routes} routes found in the Lanelet2 map; {n_self_crossing} of them pass the same "
                f"place twice (more than one round) and {n_routes - n_self_crossing - len(routes)} start or end at "
                "a lane taper; these are dropped."
            )
            if not routes:
                raise ValueError(
                    f"No routes found in the Lanelet2 map of '{self._scenario_type}'."
                )

        keys = (
            "center_line",
            "left_boundary",
            "right_boundary",
            "left_boundary_shared",
            "right_boundary_shared",
        )
        for route in routes:
            for prev_id, next_id in zip(route[:-1], route[1:]):
                assert (
                    next_id in self.lanelets_all[prev_id - 1]["successor"]
                ), f"Lanelet {self.lanelet_osm_ids[next_id - 1]} does not succeed lanelet {self.lanelet_osm_ids[prev_id - 1]}."
            # Consecutive lanelets share their connecting points; keep them once
            pieces = {
                key: [self.lanelets_all[route[0] - 1][key]]
                + [self.lanelets_all[i - 1][key][1:] for i in route[1:]]
                for key in keys
            }
            path = {key: torch.cat(pieces[key], dim=0) for key in keys}
            # Shared boundaries step sideways where the set of neighboring lanes changes or where a boundary is
            # moved to a physical border
            lane_width = SCENARIOS[self._scenario_type]["lane_width"] * self._scale_map
            joint = 0
            for piece in pieces["center_line"][:-1]:
                joint += piece.shape[0]  # Index of the second point of the next lanelet
                for key in ("left_boundary_shared", "right_boundary_shared"):
                    offset = (path[key] - path["center_line"]).norm(dim=1)
                    if (offset[joint] - offset[joint - 1]).abs() > 0.1 * lane_width:
                        self._blend_jump(path[key], joint)
            # Also within lanelets, e.g., where the physical-border search hits a curb at one station only
            for key in ("left_boundary_shared", "right_boundary_shared"):
                path[key] = self._limit_offset_slope(path["center_line"], path[key])
            (
                path["center_line_yaw"],
                path["center_line_vec_normalized"],
                path["center_line_vec_mean_length"],
            ) = self._compute_center_line_info(path["center_line"])
            path["lanelet_IDs"] = route
            path["is_loop"] = bool(
                (path["center_line"][0] - path["center_line"][-1]).norm() <= 1e-4
            )
            self.reference_paths.append(path)

    def _determine_neighboring_lanelets(self):
        """Lanelets that share at least one node with a lanelet (including itself), as 0-based indices."""
        node_sets = [set(left) | set(right) for left, right in self._bound_nodes]
        self.neighboring_lanelets_idx = [
            [j for j, other in enumerate(node_sets) if nodes & other]
            for nodes in node_sets
        ]


if __name__ == "__main__":
    import argparse

    cli_parser = argparse.ArgumentParser(
        description="Parse and visualize Lanelet2 maps."
    )
    cli_parser.add_argument("--scenario_type", default="round_location0_1to18")
    cli_parser.add_argument("--is_visu_lane_ids", action="store_true")
    cli_args = cli_parser.parse_args()

    parser = ParseLanelet2(
        scenario_type=cli_args.scenario_type,
        device="cpu",
        is_visualize_map=True,
        is_visualize_random_agents=True,
        is_save_fig=True,
        is_plt_show=False,
        is_visu_lane_ids=cli_args.is_visu_lane_ids,
        seed=0,
    )
    print(
        f"{len(parser.lanelets_all)} lanelets, {len(parser.reference_paths)} reference paths, "
        f"map size {parser.bounds['max_x'] - parser.bounds['min_x']:.2f} x "
        f"{parser.bounds['max_y'] - parser.bounds['min_y']:.2f}"
    )
