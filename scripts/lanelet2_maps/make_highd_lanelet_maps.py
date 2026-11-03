# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate Lanelet2 OSM maps of the highD locations from the lane markings in the recording meta files.

highD comes without maps and without georeference. Each recording only gives the y positions of the lane markings
of both carriageways (`upperLaneMarkings`, `lowerLaneMarkings`, meters, image y pointing down) of a straight road
section. This script builds one map per location:

  1. Map frame: x as in the tracks, y relative to the middle of the median strip, i.e. the mean of the last upper
     and the first lower marking of each recording (c). The recordings of a location are shot from slightly
     different drone positions, so their markings differ by several meters in y but much less after subtracting c.
     Each marking of the map is the median over the recordings of the location.
  2. Lanes: every lane between two markings is a lanelet over the x range of the tracks (X_RANGE), with dashed
     bounds between lanes of the same carriageway and solid outer bounds. The upper carriageway (drivingDirection 1)
     drives towards -x, the lower one (drivingDirection 2) towards +x. Lanelet IDs encode highD's laneId:
     -(100 * laneId + segment), segments numbered in driving direction from 1.
  3. Lane changes along x: LAYOUT_OVERRIDES split a carriageway at x positions where markings change type, start or
     end, e.g. the entry lane of location 6, which merges into the main lanes with a taper (TAPERS).
  4. Georeference (synthetic): map (x, y) -> UTM (UTM_ORIGIN[0] + x, UTM_ORIGIN[1] - y) in UTM_ZONE, so north is
     up like in the images, then to WGS84 lat/lon. A recorded position (x, y) of recording r maps to UTM
     (UTM_ORIGIN[0] + x, UTM_ORIGIN[1] - (y - c_r)).
  5. Check: share of vehicle positions inside a lanelet of their own laneId, and write an overlay image.

The input is the downloaded highD dataset, which may not be redistributed (https://levelxdata.com). The maps are
written to sigmarl/scenarios/assets/maps/highd_location<N>.osm. Requires pandas.

Example (from the repository root):
    python scripts/lanelet2_maps/make_highd_lanelet_maps.py --dataset_dir <path>/highD_v1.0/data \
        [--overlay_dir <folder for the overlay images>]
"""
import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas
from matplotlib.path import Path

from fix_lanelet_map import write_osm
from fix_lanelet_map import DEFAULT_OUT_DIR, read_recording_meta
from lanelet_map_utils import latlon_to_utm, utm_to_latlon

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Synthetic georeference: highD does not tell where the locations are. Zone 32 covers western Germany; the origin
# lies on the zone's central meridian (9 deg E), where UTM grid north equals true north.
UTM_ZONE = 32
UTM_ORIGIN = (500000.0, 5640000.0)

# x range of the lanelets [m]: the tracks of all recordings span x = -23.6 ... 439.2 (box corners); vehicles are
# tracked beyond the image (0 ... about 409 m).
X_RANGE = (-30.0, 445.0)
NODE_SPACING = 5.0  # [m] Max. distance between nodes of a bound
PX_TO_METER = 0.4  # Meters per pixel of the *_highway.png images (0.1 m orthophoto, downscaled by 4)

# Markings that change along x, per location and carriageway ("upper"/"lower"): breakpoints are x positions in
# driving direction; for each marking index (0 = outermost marking on the right-hand side of the carriageway, i.e.
# the first upper / last lower marking) the subtype in each interval between breakpoints, None if it does not exist.
# A lane exists in an interval where both of its markings exist.
# Location 6: entry lane (laneId 2) on the upper carriageway, coming from the on-ramp east of the images. It is
# separated by a solid line up to x = 352 m, by a dashed line up to x = 185 m, and ends with a taper (TAPERS). The
# positions are medians of recordings 58-60, which are shifted by about 7 m in x against each other.
LAYOUT_OVERRIDES = {
    6: {"upper": {"breakpoints": [352.0, 185.0],
                  "markings": {0: ["solid", "solid", None], 1: ["solid", "dashed", "solid"]}}},
}
# Tapers: (location, carriageway, interval) -> (marking, x where it starts to bend to the next marking). Within the
# interval, the marking runs straight to this x and then diagonally to the end node of the next marking.
TAPERS = {(6, "upper", 1): (0, 212.0)}


def create_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, type=str,
                        help="Directory with the highD csv files and images (highD_v1.0/data).")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR, type=str,
                        help="Directory for highd_location<N>.osm (default: the maps of SigmaRL).")
    parser.add_argument("--overlay_dir", default=None, type=str,
                        help="Directory for overlay_location<N>.png (default: no overlay images).")
    return vars(parser.parse_args())


def parse_markings(value):
    return [float(v) for v in str(value).split(";")]


def load_recordings(dataset_dir):
    """Recording meta of all recordings, with the parsed markings and the median center c."""
    metas = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*_recordingMeta.csv"))):
        meta = read_recording_meta(path)
        meta["upper"] = parse_markings(meta["upperLaneMarkings"])
        meta["lower"] = parse_markings(meta["lowerLaneMarkings"])
        meta["center"] = (meta["upper"][-1] + meta["lower"][0]) / 2
        metas.append(meta)
    return metas


def location_markings(metas):
    """Median markings of a location in the map frame (y relative to the median center) and the max. deviation."""
    result, deviation = {}, 0.0
    for side in ("upper", "lower"):
        aligned = np.array([np.array(m[side]) - m["center"] for m in metas])
        result[side] = np.median(aligned, axis=0)
        deviation = max(deviation, float(np.abs(aligned - result[side]).max()))
    return result, deviation


class OsmBuilder:
    """Collects nodes, ways and lanelets with negative JOSM-style IDs and writes them in the layout of rounD's map."""

    def __init__(self):
        self.nodes, self.ways, self.relations = {}, {}, {}
        self._node_ids = {}  # rounded (x, y) -> node ID, so that touching ways share their nodes
        self._next_node, self._next_way = -10001, -1001

    def node(self, x, y):
        key = (round(x, 3), round(y, 3))
        if key not in self._node_ids:
            self._node_ids[key] = self._next_node
            self.nodes[self._next_node] = (x, y)
            self._next_node -= 1
        return self._node_ids[key]

    def way(self, points, subtype):
        """A line_thin way through the points, densified to NODE_SPACING."""
        dense = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            n = max(1, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / NODE_SPACING - 1e-9)))
            dense += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
        way_id = self._next_way
        self._next_way -= 1
        self.ways[way_id] = ([self.node(x, y) for x, y in dense], [("subtype", subtype), ("type", "line_thin")])
        return way_id

    def lanelet(self, relation_id, left_way, right_way):
        self.relations[relation_id] = (left_way, right_way)

    def to_osm(self):
        """The data structure of fix_lanelet_map.write_osm; node coordinates converted to lat/lon."""
        ids = list(self.nodes)
        xy = np.array([self.nodes[i] for i in ids])
        lat, lon = map_to_latlon(xy[:, 0], xy[:, 1])
        attrs = lambda i: [("id", str(i)), ("action", "modify"), ("visible", "true")]  # noqa: E731
        osm = {"attrs": [("version", "0.6"), ("generator", "JOSM")], "newline": "\n",
               "nodes": {}, "ways": {}, "relations": {},
               # JOSM layout (fix_lanelet_map.write_osm)
               "declaration": "<?xml version='1.0' encoding='UTF-8'?>", "quote": "'", "close": " />",
               "member_attrs": ["type", "ref", "role"], "final_newline": True}
        for i, la, lo in zip(ids, lat, lon):
            osm["nodes"][i] = {"attrs": attrs(i) + [("lat", "{:.11f}".format(la)), ("lon", "{:.11f}".format(lo))],
                               "tags": []}
        for i, (nds, tags) in self.ways.items():
            osm["ways"][i] = {"attrs": attrs(i), "nds": nds, "tags": tags}
        for i, (left, right) in sorted(self.relations.items(), key=lambda item: -item[0]):
            osm["relations"][i] = {
                "attrs": attrs(i), "members": [["way", left, "left"], ["way", right, "right"]],
                "tags": [("location", "nonurban"), ("one_way", "yes"), ("region", "de"), ("subtype", "road"),
                         ("type", "lanelet")]}
        return osm


def map_to_latlon(x, y):
    """Map frame (x, y down) -> WGS84 lat/lon, refined with Newton steps so that latlon_to_utm inverts it exactly."""
    easting, northing = UTM_ORIGIN[0] + np.asarray(x), UTM_ORIGIN[1] - np.asarray(y)
    lat, lon = utm_to_latlon(easting, northing, UTM_ZONE)
    for _ in range(3):
        e, n = latlon_to_utm(lat, lon, UTM_ZONE)
        d = 1e-6
        e_lat, n_lat = latlon_to_utm(lat + d, lon, UTM_ZONE)
        e_lon, n_lon = latlon_to_utm(lat, lon + d, UTM_ZONE)
        j = np.array([[(e_lat - e) / d, (e_lon - e) / d], [(n_lat - n) / d, (n_lon - n) / d]])  # [2, 2, N]
        r = np.array([easting - e, northing - n])
        step = np.linalg.solve(np.moveaxis(j, -1, 0), np.moveaxis(r, -1, 0)[..., None])[..., 0]
        lat, lon = lat + step[:, 0], lon + step[:, 1]
    return lat, lon


def build_carriageway(builder, location_id, side, markings, first_lane_id):
    """Add the lanelets of one carriageway. Returns {lanelet ID: laneId}."""
    upper = side == "upper"
    x_start, x_end = (X_RANGE[1], X_RANGE[0]) if upper else X_RANGE  # driving direction
    # Marking 0 is on the right-hand side of the driving direction (north for upper, south for lower)
    ys = list(markings) if upper else list(markings)[::-1]
    layout = LAYOUT_OVERRIDES.get(location_id, {}).get(side, {})
    xs = [x_start] + layout.get("breakpoints", []) + [x_end]
    n_intervals, n_lanes = len(xs) - 1, len(ys) - 1
    subtypes = {}
    for k in range(len(ys)):
        default = "solid" if k in (0, len(ys) - 1) else "dashed"
        subtypes[k] = layout.get("markings", {}).get(k, [default] * n_intervals)

    ways = {}  # (marking, interval) -> way ID
    for k, y in enumerate(ys):
        for i in range(n_intervals):
            if subtypes[k][i] is None:
                continue
            points = [(xs[i], y), (xs[i + 1], y)]
            taper = TAPERS.get((location_id, side, i))
            if taper is not None and taper[0] == k:
                points = [(xs[i], y), (taper[1], y), (xs[i + 1], ys[k + 1])]
            ways[k, i] = builder.way(points, subtypes[k][i])

    lane_ids = {}
    for j in range(n_lanes):
        # Lane j lies between markings j (right) and j + 1 (left); laneIds count from the top of the image
        lane_id = first_lane_id + (j if upper else n_lanes - 1 - j)
        segment = 0
        for i in range(n_intervals):
            if (j, i) in ways and (j + 1, i) in ways:
                segment += 1
                relation_id = -(100 * lane_id + segment)
                builder.lanelet(relation_id, ways[j + 1, i], ways[j, i])
                lane_ids[relation_id] = lane_id
    return lane_ids


def lanelet_polygons(builder):
    polygons = {}
    for relation_id, (left, right) in builder.relations.items():
        left_pts = np.array([builder.nodes[n] for n in builder.ways[left][0]])
        right_pts = np.array([builder.nodes[n] for n in builder.ways[right][0]])
        polygons[relation_id] = np.vstack([left_pts, right_pts[::-1]])
    return polygons


def check_tracks(dataset_dir, metas, polygons, lane_ids):
    """Share of vehicle center positions inside a lanelet with their own laneId, per recording."""
    paths = {rid: Path(poly) for rid, poly in polygons.items()}
    shares = {}
    for meta in metas:
        rec = "{:02d}".format(int(meta["id"]))
        tracks = pandas.read_csv(os.path.join(dataset_dir, rec + "_tracks.csv"),
                                 usecols=["x", "y", "width", "height", "laneId"])
        tracks = tracks.iloc[::25]  # One sample per second
        pts = np.stack([tracks.x + tracks.width / 2, tracks.y + tracks.height / 2 - meta["center"]], axis=1)
        ok = np.zeros(len(pts), dtype=bool)
        for rid, path in paths.items():
            ok |= path.contains_points(pts) & (tracks.laneId.values == lane_ids[rid])
        shares[rec] = ok.mean()
    return shares


def plot_overlay(dataset_dir, meta, polygons, builder, path):
    rec = "{:02d}".format(int(meta["id"]))
    img = mpimg.imread(os.path.join(dataset_dir, rec + "_highway.png"))
    c = meta["center"]
    fig, axes = plt.subplots(2, 1, figsize=(24, 6))
    for ax, xlim in zip(axes, [(X_RANGE[0] - 5, 225), (205, X_RANGE[1] + 5)]):
        ax.imshow(img, extent=[0, img.shape[1] * PX_TO_METER, img.shape[0] * PX_TO_METER - c, -c], aspect="auto")
        for poly in polygons.values():
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.15)
        for nds, tags in builder.ways.values():
            pts = np.array([builder.nodes[n] for n in nds])
            ax.plot(pts[:, 0], pts[:, 1], "r-" if dict(tags)["subtype"] == "solid" else "y--", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(img.shape[0] * PX_TO_METER - c + 2, -c - 2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y - c [m]")
    axes[0].set_title("highD location {}, recording {} (red: solid, yellow: dashed)".format(meta["locationId"], rec))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    config = create_args()
    os.makedirs(config["output_dir"], exist_ok=True)
    metas = load_recordings(config["dataset_dir"])
    for location_id in sorted({int(m["locationId"]) for m in metas}):
        loc_metas = [m for m in metas if int(m["locationId"]) == location_id]
        markings, deviation = location_markings(loc_metas)
        builder = OsmBuilder()
        lane_ids = build_carriageway(builder, location_id, "upper", markings["upper"], 2)
        lane_ids.update(build_carriageway(builder, location_id, "lower", markings["lower"],
                                          len(markings["upper"]) + 2))
        osm_path = os.path.join(config["output_dir"], "highd_location{}.osm".format(location_id))
        write_osm(builder.to_osm(), osm_path)
        polygons = lanelet_polygons(builder)
        if config["overlay_dir"] is not None:
            os.makedirs(config["overlay_dir"], exist_ok=True)
            plot_overlay(config["dataset_dir"], loc_metas[0], polygons, builder,
                         os.path.join(config["overlay_dir"], "overlay_location{}.png".format(location_id)))
        shares = check_tracks(config["dataset_dir"], loc_metas, polygons, lane_ids)
        print("Location {}: {} recordings, {} lanelets, {} ways, {} nodes -> {}".format(
            location_id, len(loc_metas), len(builder.relations), len(builder.ways), len(builder.nodes), osm_path))
        print("  markings (y - c) upper {}, lower {}; max. deviation of a recording {:.2f} m".format(
            np.round(markings["upper"], 2).tolist(), np.round(markings["lower"], 2).tolist(), deviation))
        print("  vehicle positions inside a lanelet of their laneId: min {:.2%} (recording {}), mean {:.2%}".format(
            min(shares.values()), min(shares, key=shares.get), np.mean(list(shares.values()))))


if __name__ == "__main__":
    main()
