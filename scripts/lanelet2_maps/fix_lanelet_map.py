# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Clean up the Lanelet2 OSM maps of the drone datasets so that they match a location's background image and
recorded tracks, and so that routing works on them.

The fixes of each map are listed in MAPS (keys in parentheses). Steps:
  1. Map-specific repairs of errors in the input:
     - merge duplicate nodes, so that lanelets that end at the same place are connected (merge_nodes);
     - flip the driving direction of lanelets by swapping their left/right roles (flip);
     - close multipolygon outlines that are not closed rings (close_multipolygons, remove_multipolygon_members);
     - connect lanelets whose end/start misses the lanelet that continues them (connect).
  2. Delete lanelets that lie entirely outside the useful area (lanelets that are not drivable, e.g. walkways: at
     most 25 % inside), and junk lanelets (delete). The
     useful area is the (non-black) background image plus a buffer around every track recorded at this location,
     since vehicles are also tracked beyond the visible image. Regulatory elements that referred to deleted
     lanelets and areas entirely outside the useful area go with them (and, after step 3, areas other than traffic
     islands that still reach outside it).
  3. At the open ends of the map (lanelets without predecessor/successor), cut bounds that run outside the
     useful area at its border, and extend bounds that end inside it up to the border, with node spacing
     similar to the existing nodes of the bound. Ends where that does not follow the road (rounD: bending roads, the
     bypasses merging with the north and south arms) are instead extended through hard-coded points, and the traffic
     islands there are extended to their real tips (HARDCODED_EXTENSIONS, ISLAND_OUTLINES).
  4. Delete ways and nodes that are no longer used (ways that no relation used in the input are kept if they
     touch the useful area).
  5. inD 2 and 3: make lanes a car does not fit in wider (widen), and replace the lanes through the intersections,
     drawn as chains of short lanelets that meet at sharp angles, by lanelets with smooth center lines that are wide
     enough for the area a car sweeps in turns (connectors, see rebuild_connectors). Ways and nodes no longer used
     are deleted again.
  6. Reverse the node order of bounds stored against their lanelet's driving direction (cosmetic, for JOSM).
  7. Write the map in the JOSM layout of the input and check it: references, multipolygon rings, duplicate nodes,
     every lanelet's direction against the tracks, dead ends of drivable lanelets inside the image, and routes that
     a car cannot follow (turns tighter than 5 m, lanelets narrower than the area a car sweeps).

The inputs are the downloaded datasets, which may not be redistributed (https://levelxdata.com). The outputs are
written to sigmarl/scenarios/assets/maps/<map>.osm, where SigmaRL loads them. Requires pandas and opencv-python.

Example (from the repository root), the two maps of the T-ITS 2026 evaluation:
    python scripts/lanelet2_maps/fix_lanelet_map.py --maps round_location0 ind_location4 \
        --round_dir <path>/rounD_v1.0 --ind_dir <path>/inD_v1.1
All maps (also the highD ones, see make_highd_lanelet_maps.py):
    python scripts/lanelet2_maps/fix_lanelet_map.py --round_dir <path>/rounD_v1.0 --ind_dir <path>/inD_v1.1 \
        --exid_dir <path>/exiD_v2.0
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict

import cv2
import numpy as np
import pandas
from matplotlib.path import Path

from lanelet_map_utils import align_bounds, latlon_to_utm, resample, utm_to_latlon, utm_zone


def read_recording_meta(recording_meta_file):
    """Recording meta of a drone dataset recording (<k>_recordingMeta.csv) as a dict."""
    return pandas.read_csv(recording_meta_file).to_dict(orient="records")[0]

# Lanelet subtypes that motor vehicles drive on (Lanelet2 traffic rules; lanelets without subtype are not
# drivable); used by the checks
DRIVABLE_SUBTYPES = {"road", "highway", "play_street", "bus_lane"}

# Output folder: the maps of SigmaRL
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sigmarl", "scenarios",
                               "assets", "maps")
# Downscaling of the background images per dataset (as in drone-dataset-tools, data/visualizer_params)
SCALE_DOWN_FACTOR = {"round": 10, "ind": 12, "exid": 6}

# ---------------------------------------------------------------------------------------------------------------
# rounD location 0
# ---------------------------------------------------------------------------------------------------------------
# Lanelets whose left/right roles are swapped so that they point in the direction vehicles actually drive.
# Found with the trajectory check of this script (all vehicle samples inside each of them drove backwards).
FLIP_LANELETS = [-162054, -162053, -162052, -162051, -162019, -162009, -162008]

# Lanelets that are deleted regardless of the useful area: 1.4 m long stubs at the far west end of the westbound
# lanes, bounded by untyped ways, pointing in opposite directions and never driven on by any vehicle.
DELETE_LANELETS = [-162055, -162056]

# Traffic island outlines (multipolygon -> missing ways) that are not closed rings in the input. The island
# between lanelets -162047 and -162049 lacks the 1.9 m road border at its north tip.
CLOSE_MULTIPOLYGONS = {-162073: [-152309]}

# Lanelets in the circle whose end/start does not meet the lanelet that continues the circle, so that routing finds
# no way around the roundabout (Lanelet2 successors share both end nodes). (lanelet, "end"/"start", lanelet to
# meet): the bound that misses is extended along the bound of the overlapping lanelet that spans the gap.
CONNECT_LANELETS = [(-162031, "end", -162044), (-162000, "start", -162044), (-162001, "end", -162005)]

# Open map ends where a straight extension does not follow the road: the roads bend, and beyond the traffic islands
# of the north and south arms the bypass merges with the lanes of the arm, separated by a solid double line. These
# ends are extended through the points below instead of by trim_and_extend. Points are in meters relative to the
# UTM origin of the location (x east, y north), digitized from the composite of all background images of the
# location (lane markings, curbs and grass edges) up to the image border. Points with the same name share a node.
HARDCODED_POINTS = {
    # South arm: island between the bypass -161965 and the entry lanes -162049/-162016, and the line beyond it
    "s_isl_w1": (73.7, -81.4), "s_isl_w2": (72.9, -83.3), "s_isl_w3": (72.6, -84.6),
    "s_isl_e1": (75.2, -83.0), "s_isl_e2": (74.15, -84.0), "s_isl_e3": (73.2, -85.0),
    "s_tip": (72.5, -85.7),
    "s_sep1": (71.6, -86.5), "s_sep2": (70.65, -88.0), "s_sep3": (69.5, -90.0), "s_sep4": (68.45, -92.0),
    "s_sep5": (67.55, -94.0), "s_sep6": (67.35, -94.5),
    # South arm: island between the exit lane -162047 and the entry lane -162049
    "w_mid": (64.1, -80.2), "w_tip": (62.8, -83.8),
    # North arm: island between the bypass -161987 and the entry lanes -162021/-162020, and the line beyond it
    "n_isl_w1": (89.3, -9.1), "n_isl_w2": (90.3, -8.1),
    "n_isl_e1": (90.6, -9.6), "n_isl_e2": (91.1, -8.4),
    "n_tip": (91.2, -7.4),
    "n_sep1": (92.2, -6.2), "n_sep2": (93.25, -4.7), "n_sep3": (94.35, -2.7), "n_sep4": (95.5, -0.7),
}
S_SEP = ["s_tip", "s_sep1", "s_sep2", "s_sep3", "s_sep4", "s_sep5", "s_sep6"]
N_SEP = ["n_tip", "n_sep1", "n_sep2", "n_sep3", "n_sep4"]
# way id -> (node at the way's open end in the input, points appended there in order)
HARDCODED_EXTENSIONS = {
    # South: bypass -161965 (left: island edge, then the line; right: curb)
    -152169: (-111513, ["s_isl_e1", "s_isl_e2", "s_isl_e3"] + S_SEP),
    -152198: (-111537, [(82.2, -84.0), (80.4, -85.5), (78.85, -87.0), (76.75, -89.0), (74.9, -91.0),
                        (73.35, -93.0), (71.85, -94.8)]),
    # South: entry lanes -162049/-162016 (right: island edge, then the line; dashed lane line; left lane line)
    -152171: (-111577, ["s_isl_w1", "s_isl_w2", "s_isl_w3"] + S_SEP),
    -152276: (-111589, [(69.5, -81.5), (67.95, -84.5), (66.95, -86.5), (65.35, -89.6)]),
    -161925: (-111603, [(65.65, -81.0), (64.6, -83.0), (63.55, -85.0), (63.15, -86.0)]),
    # South: exit lane -162047 (left: island edge to its tip; right: curb)
    -152311: (-111623, ["w_mid", "w_tip"]),
    -152176: (-111633, [(61.15, -77.5), (60.6, -79.0), (59.95, -80.5), (59.4, -82.4)]),
    # South: walkway -162035 along the bypass (grass edges)
    -152291: (-123360, [(83.0, -86.0), (80.5, -88.0), (78.5, -90.0), (76.8, -92.0), (75.0, -94.0), (74.5, -94.8)]),
    -152292: (-123466, [(86.6, -86.0), (84.0, -88.0), (81.8, -90.0), (79.5, -92.0), (77.8, -94.0), (77.3, -94.8)]),
    # North: bypass -161987 (left: island edge, then the line; right: curb)
    -152187: (-111853, ["n_isl_w1", "n_isl_w2"] + N_SEP),
    -152186: (-111819, [(83.8, -7.2), (85.4, -5.7), (86.9, -4.2), (88.2, -2.7), (89.5, -1.2), (90.3, -0.2)]),
    # North: entry lanes -162021/-162020 (right: island edge, then the line; dashed lane line; left: island edge)
    -152188: (-111889, ["n_isl_e1", "n_isl_e2"] + N_SEP),
    -152250: (-111901, [(95.5, -8.7), (96.55, -6.7)]),
    -161964: (-111925, [(98.3, -10.6)]),
}
# Traffic islands whose outline ended where the map ended, with a closing way across the island. They are extended
# to their real tips: the listed member ways are replaced by one way through the parts (way ids of the input with
# their input nodes, and point names), since the lanelet bounds among them now continue beyond the tip.
ISLAND_OUTLINES = {
    -162072: ([-152171, "s_isl_w1", "s_isl_w2", "s_isl_w3", "s_tip", "s_isl_e3", "s_isl_e2", "s_isl_e1", -152169],
              [-152171, -161942, -152169]),
    -162073: ([-152311, "w_mid", "w_tip", -161925], [-152311, -152277, -161925]),
    -162069: ([-152187, "n_isl_w1", "n_isl_w2", "n_tip", "n_isl_e2", "n_isl_e1", -152188],
              [-152187, -161647, -152188]),
}

# ---------------------------------------------------------------------------------------------------------------
# All maps (keys: the scenario types of SigmaRL, also the output file names). Per map: dataset (key of
# SCALE_DOWN_FACTOR and of the dataset folder arguments), dataset_dir, recordings (those recorded with this map;
# None: all recordings at the location of the first one, which must then be given as `recording`), input (the
# untouched Lanelet2 map shipped with the dataset), both relative to the dataset folder, and the map-specific
# repairs (all optional):
#   merge_nodes: {node to drop: node to keep}, nodes at the same position that should be one node
#   flip: lanelets to flip, see FLIP_LANELETS
#   delete: lanelets deleted regardless of the useful area
#   close_multipolygons: {multipolygon: ways to add}; remove_multipolygon_members: {multipolygon: ways to remove}
#   connect: see CONNECT_LANELETS
#   hardcoded_points/hardcoded_extensions/island_outlines: see HARDCODED_POINTS
#   widen: [(lanelets, "left"/"right", width)], see widen_lanelets
#   connectors: chains of lanelets replaced by lanelets with smooth center lines, see rebuild_connectors
# The inD and exiD maps had no wrong-way lanelets (checked against all their recordings).
# ---------------------------------------------------------------------------------------------------------------
MAPS = OrderedDict([
    ("round_location0", dict(
        dataset="round", dataset_dir="data/", recording=2, recordings=None,
        input="lanelets/location0.osm",
        flip=FLIP_LANELETS, delete=DELETE_LANELETS, close_multipolygons=CLOSE_MULTIPOLYGONS,
        connect=CONNECT_LANELETS, hardcoded_points=HARDCODED_POINTS, hardcoded_extensions=HARDCODED_EXTENSIONS,
        island_outlines=ISLAND_OUTLINES)),
    # inD location 1 (Bendplatz) without the construction site (recordings 7-10). The walkway area 1718 has an
    # extra virtual way across it (1509, 1184 -> 1028), so its outline is no ring.
    ("ind_location1", dict(
        dataset="ind", dataset_dir="data/", recordings=[7, 8, 9, 10],
        input="lanelets/location1.osm",
        remove_multipolygon_members={1718: [1509]})),
    # inD location 1 with the construction site (recordings 11-17). Same walkway area as above. Lanelet 2157 (left
    # turn from the south-east arm into the north-east arm) ends 0.9 m/1.5 m short of lanelet 2187, which starts the
    # lane past the construction site, so the left turn leads nowhere.
    ("ind_location1_construction", dict(
        dataset="ind", dataset_dir="data/", recordings=[11, 12, 13, 14, 15, 16, 17],
        input="lanelets/location1_construction.osm",
        remove_multipolygon_members={1718: [1509]}, connect=[(2157, "end", 2187)])),
    # inD location 2 (Neukoellner Strasse). The lanes through the intersection are single lanelets with straight
    # bounds; with the kinked ends of the lanes before and after them, their center lines turn with radii of
    # 1.2-2.9 m. Each approach (its kinked end, if any) and each exit (its kinked start) is rebuilt with the first
    # lane through the intersection that uses it; the other two lanes start (end) where they do. The lanes in the
    # west arm (towards the west) and in the south arm (towards the north) are only 1.8-2.4 m wide (parked cars on
    # their right side), and vehicles drive 0.6-0.8 m left of their center lines: they are widened to the right
    # (virtual line) and to the left (the unmarked center of the road, moving it to the middle of the curbs).
    ("ind_location2", dict(
        dataset="ind", dataset_dir="data/", recordings=list(range(18, 30)),
        input="lanelets/location2.osm",
        widen=[([1099, 1100, 1210, 1101, 1102], "right", 2.8), ([1163, 1164, 1165, 1168], "left", 2.9)],
        connectors=[
            dict(lanelets=[1093, 1094, 1188]),
            dict(lanelets=[1191, 1161], start_with=1093),
            dict(lanelets=[1196, 1127], start_with=1093),
            dict(lanelets=[1125, 1126, 1189], end_with=1161),
            dict(lanelets=[1193, 1095, 1096, 1097], start_with=1125),
            dict(lanelets=[1198], start_with=1125),
            dict(lanelets=[1163, 1162, 1190], end_with=1127),
            dict(lanelets=[1192], start_with=1163),
            dict(lanelets=[1195], start_with=1163, end_with=1097),
            dict(lanelets=[1181, 1180, 1179, 1178, 1177, 1187], end_with=1097),
            dict(lanelets=[1194], start_with=1181, end_with=1127),
            dict(lanelets=[1197], start_with=1181, end_with=1161),
        ])),
    # inD location 3 (Heckstrasse). Lanelet 1099 (turn from the north-east arm into the side street) starts at the
    # left node of short lanelet 1080 but at the right node of lanelet 1081, so it has no predecessor; it should
    # continue 1081 (0.7 m along the solid line). The lanes through the two junctions are chains of short lanelets
    # whose center lines turn by up to 60 degrees at single points (radii of 0.9-2.7 m) and narrow to 1.2 m. They
    # are rebuilt: in the north the fork from the west into the north-east arm, and the lanes from the north-east
    # arm (fork at the end of lanelet 1082); the turn into the side street; in the south the lanes out of the south
    # arm (fork at its map end, the arm being a narrow play street) and into it.
    ("ind_location3", dict(
        dataset="ind", dataset_dir="data/", recordings=[30, 31, 32],
        input="lanelets/location3.osm",
        connect=[(1099, "start", 1081)], connectors=[
            dict(lanelets=[1045, 1079, 1044, 1043]),
            dict(lanelets=[1081, 1099, 1100]),
            dict(lanelets=[1080, 1078], start_with=1081),
            dict(lanelets=[1069, 1112]),
            dict(lanelets=[1125, 1126, 1127, 1128, 1129]),
            dict(lanelets=[1130, 1131], start_with=1125),
            dict(lanelets=[1119, 1120, 1121, 1122, 1123]),
        ])),
    ("ind_location4", dict(
        dataset="ind", dataset_dir="data/", recordings=list(range(0, 7)),
        input="lanelets/location4.osm")),
    # exiD location 0. Lanelet 1991 ends at node 1204 and its successor 1987 starts at node 1203, a duplicate at the
    # same position, so the two are not connected. Lanelet 1953 is a 21 m sliver (0-1.5 m wide, starting at a point)
    # along the hard shoulder, tagged highway and never driven on.
    ("exid_location0", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(0, 19)),
        input="lanelets/location0.osm",
        merge_nodes={1204: 1203}, delete=[1953])),
    ("exid_location1", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(19, 39)),
        input="lanelets/location1.osm")),
    ("exid_location2", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(39, 53)),
        input="lanelets/location2.osm")),
    ("exid_location3", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(53, 61)),
        input="lanelets/location3.osm")),
    ("exid_location4", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(61, 73)),
        input="lanelets/location4.osm")),
    ("exid_location5", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(73, 78)),
        input="lanelets/location5.osm")),
    ("exid_location6", dict(
        dataset="exid", dataset_dir="data/", recordings=list(range(78, 93)),
        input="lanelets/location6.osm")),
])


def create_args():
    cs = argparse.ArgumentParser(description="Fix Lanelet2 OSM maps to match the background images and tracks")
    cs.add_argument("--maps", nargs="+", default=list(MAPS), choices=list(MAPS), help="Maps to fix (default: all).")
    cs.add_argument("--round_dir", default=None, help="Folder of rounD (holds data/ and lanelets/).")
    cs.add_argument("--ind_dir", default=None, help="Folder of inD (holds data/ and lanelets/).")
    cs.add_argument("--exid_dir", default=None, help="Folder of exiD (holds data/ and lanelets/).")
    cs.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output folder (default: the maps of SigmaRL).")
    cs.add_argument("--track_buffer", default=2.0, type=float,
                    help="Radius in meters around track center lines that counts as useful area.")
    cs.add_argument("--min_inside_fraction", default=0.0, type=float,
                    help="Lanelets with at most this fraction of their center line inside the useful area are "
                         "deleted. Lanelets partly inside are cut at its border instead (at open map ends).")
    cs.add_argument("--min_inside_fraction_other", default=0.25, type=float,
                    help="The same for lanelets that are not drivable (walkways, bicycle lanes, ...).")
    cs.add_argument("--max_extension", default=80.0, type=float,
                    help="Maximum length in meters a bound is extended. Ends that would need more are left alone.")
    cs.add_argument("--verbose", action="store_true", help="List every lanelet partly outside the useful area.")
    return vars(cs.parse_args())


# --------------------------------------------------------------------------------------------------------------
# OSM reading/writing that keeps the JOSM layout of the input, so that the diff only shows real changes
# --------------------------------------------------------------------------------------------------------------
def load_osm(path):
    root = ET.parse(path).getroot()
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw[:4096] else "\n"
    first_member = root.find("relation/member")
    osm = {"attrs": list(root.attrib.items()), "nodes": OrderedDict(), "ways": OrderedDict(),
           "relations": OrderedDict(), "newline": newline,
           # Layout of the input: XML declaration, quotes, "/>" or " />", member attribute order, final newline
           "declaration": raw.split(newline, 1)[0], "quote": "'" if "id='" in raw[:4096] else '"',
           "close": " />" if " />" in raw[:4096] else "/>",
           "member_attrs": list(first_member.attrib) if first_member is not None else ["type", "ref", "role"],
           "final_newline": raw.endswith(newline)}
    for e in root:
        tags = [(t.get("k"), t.get("v")) for t in e.iter("tag")]
        item = {"attrs": list(e.attrib.items()), "tags": tags}
        if e.tag == "node":
            osm["nodes"][int(e.get("id"))] = item
        elif e.tag == "way":
            item["nds"] = [int(nd.get("ref")) for nd in e.iter("nd")]
            osm["ways"][int(e.get("id"))] = item
        elif e.tag == "relation":
            item["members"] = [[m.get("type"), int(m.get("ref")), m.get("role")] for m in e.iter("member")]
            osm["relations"][int(e.get("id"))] = item
        else:
            raise ValueError("Unsupported OSM element <{}>".format(e.tag))
    return osm


def _esc(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("'", "&apos;").replace('"', "&quot;"))


def write_osm(osm, path):
    q, close = osm["quote"], osm["close"]

    def attrs(pairs):
        return " ".join("{}={}{}{}".format(k, q, _esc(v), q) for k, v in pairs)

    def tag_lines(tags):
        return ["    <tag {}{}".format(attrs([("k", k), ("v", v)]), close) for k, v in tags]

    lines = [osm["declaration"], "<osm {}>".format(attrs(osm["attrs"]))]
    for node in osm["nodes"].values():
        if node["tags"]:
            lines += ["  <node {}>".format(attrs(node["attrs"]))] + tag_lines(node["tags"]) + ["  </node>"]
        else:
            lines.append("  <node {}{}".format(attrs(node["attrs"]), close))
    for way in osm["ways"].values():
        lines.append("  <way {}>".format(attrs(way["attrs"])))
        lines += ["    <nd {}{}".format(attrs([("ref", ref)]), close) for ref in way["nds"]]
        lines += tag_lines(way["tags"]) + ["  </way>"]
    for rel in osm["relations"].values():
        lines.append("  <relation {}>".format(attrs(rel["attrs"])))
        for t, ref, role in rel["members"]:
            member = {"type": t, "ref": ref, "role": role}
            lines.append("    <member {}{}".format(attrs([(k, member[k]) for k in osm["member_attrs"]]), close))
        lines += tag_lines(rel["tags"]) + ["  </relation>"]
    lines.append("</osm>")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(osm["newline"].join(lines) + (osm["newline"] if osm["final_newline"] else ""))


# --------------------------------------------------------------------------------------------------------------
# Map model helpers
# --------------------------------------------------------------------------------------------------------------
class LaneletMap:
    def __init__(self, osm, recording_meta):
        self.osm = osm
        self.x0, self.y0 = recording_meta["xUtmOrigin"], recording_meta["yUtmOrigin"]
        self.zone = utm_zone(recording_meta["lonLocation"])
        self.xy = {}
        for nid, node in osm["nodes"].items():
            a = dict(node["attrs"])
            e, n = latlon_to_utm(float(a["lat"]), float(a["lon"]), self.zone)
            self.xy[nid] = np.array([e - self.x0, n - self.y0])
        all_ids = list(osm["nodes"]) + list(osm["ways"]) + list(osm["relations"])
        self.next_id = min(min(all_ids), 0) - 1

    def add_node(self, point):
        lat, lon = utm_to_latlon(point[0] + self.x0, point[1] + self.y0, self.zone)
        nid = self.next_id
        self.next_id -= 1
        self.osm["nodes"][nid] = {"attrs": [("id", str(nid)), ("action", "modify"), ("visible", "true"),
                                            ("lat", "{:.11f}".format(float(lat))),
                                            ("lon", "{:.11f}".format(float(lon)))], "tags": []}
        self.xy[nid] = np.asarray(point, dtype=float)
        return nid

    def add_way(self, node_ids, tags):
        wid = self.next_id
        self.next_id -= 1
        self.osm["ways"][wid] = {"attrs": [("id", str(wid)), ("action", "modify"), ("visible", "true")],
                                 "tags": list(tags), "nds": list(node_ids)}
        return wid

    def points(self, node_ids):
        return np.array([self.xy[n] for n in node_ids])

    def lanelets(self):
        return {rid: rel for rid, rel in self.osm["relations"].items() if dict(rel["tags"]).get("type") == "lanelet"}

    def bounds(self, rel):
        """Left/right way ids and their node ids in driving direction (Lanelet2 semantics)."""
        roles = {role: ref for t, ref, role in rel["members"] if t == "way"}
        left_nds, right_nds = self.osm["ways"][roles["left"]]["nds"], self.osm["ways"][roles["right"]]["nds"]
        _, _, left_inv, right_inv = align_bounds(self.points(left_nds), self.points(right_nds))
        return (roles["left"], roles["right"],
                left_nds[::-1] if left_inv else list(left_nds), right_nds[::-1] if right_inv else list(right_nds))

    def centerline(self, rel, num=50):
        _, _, left, right = self.bounds(rel)
        return (resample(self.points(left), num) + resample(self.points(right), num)) / 2


class UsefulArea:
    """Non-black pixels of the background images of all recordings at the location (their extents differ),
    plus a buffer around all tracks of the location."""

    def __init__(self, background_paths, track_frames, px_per_meter, track_buffer):
        content = None
        for path in background_paths:
            visible = (cv2.imread(path).astype(int).sum(axis=2) > 10).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(visible)
            visible = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            content = visible if content is None else content | visible
        content = cv2.erode(content.astype(np.uint8), np.ones((5, 5), np.uint8))  # ignore the blurry border
        tracks = np.zeros_like(content)
        thickness = max(1, int(round(2 * track_buffer * px_per_meter)))
        for _, track in track_frames.groupby(["recordingId", "trackId"]):
            pts = np.round(track[["xCenter", "yCenter"]].values * [px_per_meter, -px_per_meter]).astype(np.int32)
            cv2.polylines(tracks, [pts.reshape(-1, 1, 2)], False, 1, thickness)
        self.image_mask, self.track_mask = content.astype(bool), tracks.astype(bool)
        self.mask = self.image_mask | self.track_mask
        self.px_per_meter = px_per_meter

    def contains(self, point, image_only=False):
        mask = self.image_mask if image_only else self.mask
        u, v = int(round(point[0] * self.px_per_meter)), int(round(-point[1] * self.px_per_meter))
        return 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and bool(mask[v, u])

    def border_crossing(self, inside_pt, outside_pt, tol=0.02):
        a, b = np.asarray(inside_pt, float), np.asarray(outside_pt, float)
        while np.linalg.norm(b - a) > tol:
            m = (a + b) / 2
            if self.contains(m):
                a = m
            else:
                b = m
        return a


def load_location(dataset_dir, location_id, recordings=None):
    """Track samples (every 5th frame) and background image paths of the given recordings, or of all recordings at
    the location."""
    frames, backgrounds = [], []
    for meta_file in sorted(glob.glob(os.path.join(dataset_dir, "*_recordingMeta.csv"))):
        meta = read_recording_meta(meta_file)
        if meta["locationId"] != location_id or recordings is not None and meta["recordingId"] not in recordings:
            continue
        prefix = meta_file[:-len("recordingMeta.csv")]
        backgrounds.append(prefix + "background.png")
        tracks = pandas.read_csv(prefix + "tracks.csv", usecols=["recordingId", "trackId", "frame", "xCenter",
                                                                  "yCenter", "xVelocity", "yVelocity"])
        classes = pandas.read_csv(prefix + "tracksMeta.csv", usecols=["trackId", "class"])
        frames.append(tracks[tracks["frame"] % 5 == 0].merge(classes, on="trackId"))
    return pandas.concat(frames, ignore_index=True), backgrounds


# --------------------------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------------------------
def flip_lanelets(lmap, lanelet_ids):
    for rid in lanelet_ids:
        for member in lmap.osm["relations"][rid]["members"]:
            member[2] = {"left": "right", "right": "left"}.get(member[2], member[2])
    print("Flipped {} lanelets: {}".format(len(lanelet_ids), lanelet_ids))


def close_multipolygons(lmap, missing_ways):
    for rid, way_ids in missing_ways.items():
        for way_id in way_ids:
            lmap.osm["relations"][rid]["members"].append(["way", way_id, "outer"])
        print("Added ways {} to the outline of multipolygon {}".format(way_ids, rid))


def is_connected_at(lmap, node):
    """Whether a lanelet that has `node` at an end of a bound is connected to another lanelet there."""
    info = [lmap.bounds(rel) for rel in lmap.lanelets().values()]
    starts = {(l[0], r[0]) for _, _, l, r in info}
    ends = {(l[-1], r[-1]) for _, _, l, r in info}
    return any(node in pair and pair in ends for pair in ((l[0], r[0]) for _, _, l, r in info)) or \
        any(node in pair and pair in starts for pair in ((l[-1], r[-1]) for _, _, l, r in info))


def connect_lanelets(lmap, connections):
    ways = lmap.osm["ways"]
    for rid, which, other in connections:
        rel = lmap.osm["relations"][rid]
        left_id, right_id, left, right = lmap.bounds(rel)
        _, _, other_left, other_right = lmap.bounds(lmap.osm["relations"][other])
        k = -1 if which == "end" else 0
        for way_id, nds, target in ((left_id, left, other_left[-1 - k]), (right_id, right, other_right[-1 - k])):
            if nds[k] == target:
                continue
            donor_id, donor = next((w, way["nds"]) for w, way in ways.items()
                                   if w != way_id and nds[k] in way["nds"] and target in way["nds"])
            i, j = donor.index(nds[k]), donor.index(target)
            step = 1 if j > i else -1
            bridge = [donor[x] for x in range(i + step, j + step, step)]  # after nds[k] up to the target
            new = nds + bridge if which == "end" else bridge[::-1] + nds
            new = new if ways[way_id]["nds"] == nds else new[::-1]
            length = np.linalg.norm(np.diff(lmap.points([nds[k]] + bridge), axis=0), axis=1).sum()
            shared = any(r != rid and any(m[0] == "way" and m[1] == way_id for m in other_rel["members"])
                         for r, other_rel in lmap.lanelets().items())
            if shared and is_connected_at(lmap, nds[k]):
                # Extending the way would also move the end of another lanelet away from its neighbor: give this
                # lanelet its own copy of the bound
                new_id = lmap.add_way(new, ways[way_id]["tags"])
                for member in rel["members"]:
                    if member[0] == "way" and member[1] == way_id:
                        member[1] = new_id
                print("Replaced way {} of lanelet {} by way {}, which extends it at the lanelet's {} by {:.1f} m "
                      "along way {} to meet lanelet {}".format(way_id, rid, new_id, which, length, donor_id, other))
            else:
                ways[way_id]["nds"] = new
                print("Extended way {} of lanelet {} at its {} by {:.1f} m along way {} to meet lanelet {}".format(
                    way_id, rid, which, length, donor_id, other))


def delete_outside_lanelets(lmap, area, min_inside_fraction, min_inside_fraction_other, delete_ids, verbose=True):
    deleted = [rid for rid in delete_ids if rid in lmap.osm["relations"]]
    for rid, rel in lmap.lanelets().items():
        fraction = np.mean([area.contains(p) for p in lmap.centerline(rel)])
        if fraction < 1 and verbose:
            print("  lanelet {}: {:.0%} of its center line inside the useful area".format(rid, fraction))
        drivable = dict(rel["tags"]).get("subtype") in DRIVABLE_SUBTYPES
        if fraction <= (min_inside_fraction if drivable else min_inside_fraction_other) and rid not in deleted:
            deleted.append(rid)
    for rid in deleted:
        del lmap.osm["relations"][rid]
    print("Deleted {} lanelets (DELETE_LANELETS and outside the useful area): {}".format(len(deleted),
                                                                                       sorted(deleted)))
    return deleted


def delete_outside_areas(lmap, area, partly=False):
    """Delete multipolygon areas (buildings, vegetation, ...) without any outline node inside the useful area, or,
    with `partly`, areas other than traffic islands with an outline node outside it. The latter is done after the
    map ends have been moved: areas that shared a bound with a lanelet there follow the new end and would otherwise
    keep their parts far outside the image."""
    deleted = []
    for rid, rel in list(lmap.osm["relations"].items()):
        tags = dict(rel["tags"])
        if tags.get("type") != "multipolygon" or partly and tags.get("subtype") == "traffic_island":
            continue
        inside = [area.contains(lmap.xy[n]) for t, ref, _ in rel["members"] if t == "way" and ref in lmap.osm["ways"]
                  for n in lmap.osm["ways"][ref]["nds"]]
        if not all(inside) if partly else not any(inside):
            deleted.append(rid)
            del lmap.osm["relations"][rid]
    if deleted:
        print("Deleted {} areas {} the useful area: {}".format(
            len(deleted), "reaching outside" if partly else "outside", sorted(deleted)))


def remove_dangling_references(lmap):
    """Drop members that refer to deleted relations. Regulatory elements that lose a referenced lanelet are deleted
    if they no longer refer to any lanelet, or, for right_of_way, to a lanelet with right of way and one that
    yields; lanelets drop the references to deleted regulatory elements."""
    relations = lmap.osm["relations"]
    deleted = []
    changed = True
    while changed:
        changed = False
        for rid, rel in list(relations.items()):
            members = [m for m in rel["members"] if m[0] != "relation" or m[1] in relations]
            if len(members) == len(rel["members"]):
                continue
            rel["members"][:] = members
            tags = dict(rel["tags"])
            roles = {role for t, _, role in members if t == "relation"}
            if tags.get("type") == "regulatory_element" and (
                    not roles or tags.get("subtype") == "right_of_way" and not {"right_of_way", "yield"} <= roles):
                del relations[rid]
                deleted.append(rid)
            changed = True
    if deleted:
        print("Deleted {} regulatory elements that referred to deleted lanelets: {}".format(len(deleted),
                                                                                           sorted(deleted)))


def merge_nodes(lmap, mapping):
    """Replace nodes by other nodes (at the same position) in all ways and relations."""
    for way in lmap.osm["ways"].values():
        nds = [mapping.get(n, n) for n in way["nds"]]
        way["nds"] = [n for i, n in enumerate(nds) if i == 0 or n != nds[i - 1]]
    for rel in lmap.osm["relations"].values():
        for member in rel["members"]:
            if member[0] == "node":
                member[1] = mapping.get(member[1], member[1])
    for drop, keep in mapping.items():
        print("Merged node {} into node {} ({:.3f} m apart)".format(drop, keep,
                                                                  np.linalg.norm(lmap.xy[drop] - lmap.xy[keep])))
        del lmap.osm["nodes"][drop]


def remove_multipolygon_members(lmap, members):
    for rid, way_ids in members.items():
        lmap.osm["relations"][rid]["members"][:] = [m for m in lmap.osm["relations"][rid]["members"]
                                                    if not (m[0] == "way" and m[1] in way_ids)]
        print("Removed ways {} from the outline of multipolygon {}".format(way_ids, rid))


def open_ends(lmap):
    """Way ends at the open ends of the map: (way id, 0 or -1, terminal node) per lanelet end, grouped by road."""
    info = {rid: lmap.bounds(rel) for rid, rel in lmap.lanelets().items()}
    starts, ends = {}, {}
    for rid, (_, _, left, right) in info.items():
        starts.setdefault((left[0], right[0]), []).append(rid)
        ends.setdefault((left[-1], right[-1]), []).append(rid)
    terminal = []  # (lanelet, "start"/"end")
    for rid, (_, _, left, right) in info.items():
        if (left[0], right[0]) not in ends:
            terminal.append((rid, "start"))
        if (left[-1], right[-1]) not in starts:
            terminal.append((rid, "end"))
    terminal_nodes = {}
    for rid, which in terminal:
        _, _, left, right = info[rid]
        k = 0 if which == "start" else -1
        terminal_nodes.setdefault(left[k], set()).add(rid)
        terminal_nodes.setdefault(right[k], set()).add(rid)

    groups = []
    for rid, which in terminal:
        left_way, right_way, left, right = info[rid]
        k = 0 if which == "start" else -1
        way_ends = []
        for way_id, node in ((left_way, left[k]), (right_way, right[k])):
            # Only a real open end: every lanelet touching this node must end there without a neighbor
            touching = {r for r, (_, _, l, rr) in info.items() if node in l or node in rr}
            if touching - terminal_nodes[node]:
                way_ends = []
                break
            nds = lmap.osm["ways"][way_id]["nds"]
            if nds[0] == node:
                way_ends.append((way_id, 0, node))
            elif nds[-1] == node:
                way_ends.append((way_id, -1, node))
            else:
                way_ends = []
                break
        if way_ends:
            groups.append((rid, which, way_ends))
    # Merge lanelet ends that share a way end (parallel lanes of one road) so they extend in the same direction
    merged = []
    for rid, which, way_ends in groups:
        keys = {(w, k) for w, k, _ in way_ends}
        hits = [g for g in merged if g["keys"] & keys]
        new = {"lanelets": [(rid, which)], "keys": keys, "way_ends": list(way_ends)}
        for g in hits:
            merged.remove(g)
            new["lanelets"] += g["lanelets"]
            new["keys"] |= g["keys"]
            new["way_ends"] += g["way_ends"]
        merged.append(new)
    for g in merged:
        g["way_ends"] = list(OrderedDict(((w, k), n) for w, k, n in g["way_ends"]).items())
    return merged


def lanelet_end_centerline(lmap, rid, which, num=200):
    """Center line of a lanelet ordered so that the given open end is its last point."""
    center = lmap.centerline(lmap.osm["relations"][rid], num)
    return center if which == "end" else center[::-1]


def end_direction(center, length=5.0):
    """Unit vector pointing out of a polyline at its last point, averaged over the last `length` meters."""
    end = center[-1]
    for p in center[-2::-1]:
        if np.linalg.norm(end - p) >= length:
            break
    d = end - p
    return d / np.linalg.norm(d)


def trim_offset(center, direction, area):
    """For a center line ending outside the useful area: offset along `direction` (negative) from its end to
    where it re-enters the area. None if the end is inside or the center line never enters the area."""
    if area.contains(center[-1]):
        return None
    for p in center[::-1]:
        if area.contains(p):
            return float(np.dot(p - center[-1], direction))
    return None


def image_border_distance(point, direction, area, max_extension, step=0.1):
    """Distance along `direction` from a point inside the background image to the image border, or None."""
    for s in np.arange(step, max_extension + step, step):
        if not area.contains(point + s * direction, image_only=True):
            return s - step
    return None


def ray_hit_distance(start, direction, length, segments):
    """Distance along the ray to the first of the segments (array K x 2 x 2) it crosses, or None."""
    if len(segments) == 0:
        return None
    a, b = segments[:, 0], segments[:, 1]
    e = b - a
    denom = direction[0] * e[:, 1] - direction[1] * e[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a[:, 0] - start[0]) * e[:, 1] - (a[:, 1] - start[1]) * e[:, 0]) / denom
        u = ((a[:, 0] - start[0]) * direction[1] - (a[:, 1] - start[1]) * direction[0]) / denom
    hit = (np.abs(denom) > 1e-9) & (t > 1e-6) & (t <= length) & (u >= 0) & (u <= 1)
    return float(t[hit].min()) if hit.any() else None


def cut_polyline(pts, origin, direction, offset):
    """Index i and point where the polyline (ordered towards the open end) crosses the cut line, or None."""
    proj = (pts - origin) @ direction - offset
    for i in range(len(pts) - 1, 0, -1):
        if proj[i - 1] <= 0 < proj[i]:
            t = proj[i - 1] / (proj[i - 1] - proj[i])
            return i - 1, pts[i - 1] + t * (pts[i] - pts[i - 1])
    return None


def follow_moved_end(lmap, end_node, new_nds, moved_ways):
    """Connected ways that end at a moved way end (e.g. traffic island outlines) follow its new end."""
    for other_id, other in lmap.osm["ways"].items():
        if other_id in moved_ways or end_node not in other["nds"]:
            continue
        if other["nds"][0] == end_node:
            other["nds"][0] = new_nds[-1]
        elif other["nds"][-1] == end_node:
            other["nds"][-1] = new_nds[-1]
        elif end_node not in new_nds:
            print("  WARNING: way {} uses removed node {} in its interior".format(other_id, end_node))


def apply_hardcoded_geometry(lmap, extensions, island_outlines, points):
    """Rebuild the outlines of ISLAND_OUTLINES, then extend the ways of HARDCODED_EXTENSIONS through their points."""
    ways, relations = lmap.osm["ways"], lmap.osm["relations"]
    named = {}

    def node(point):
        if not isinstance(point, str):
            return lmap.add_node(point)
        if point not in named:
            named[point] = lmap.add_node(points[point])
        return named[point]

    for rid, (parts, replaced) in island_outlines.items():
        outline = []
        for i, part in enumerate(parts):
            if isinstance(part, str):
                outline.append(node(part))
                continue
            nds = list(ways[part]["nds"])
            # Orient the way so that it chains with the previous part (or, as first part, with the next one)
            if outline:
                anchor, first = lmap.xy[outline[-1]], True
            else:
                nxt = parts[i + 1]
                anchor = np.asarray(points[nxt]) if isinstance(nxt, str) else lmap.xy[ways[nxt]["nds"][0]]
                first = False
            d0, d1 = (np.linalg.norm(lmap.xy[n] - anchor) for n in (nds[0], nds[-1]))
            if (d1 < d0) == first:
                nds.reverse()
            outline += nds
        members = relations[rid]["members"]
        members[:] = [m for m in members if m[1] not in replaced]
        members.append(["way", lmap.add_way(outline, ways[replaced[0]]["tags"]), "outer"])
        print("Replaced ways {} in the outline of traffic island {} by one way through {}".format(
            replaced, rid, [p for p in parts if isinstance(p, str)]))

    for way_id, (end_node, new_points) in extensions.items():
        nds = ways[way_id]["nds"]
        if end_node not in (nds[0], nds[-1]):
            raise ValueError("Node {} is not an end of way {}".format(end_node, way_id))
        new_nds = (nds if nds[-1] == end_node else nds[::-1]) + [node(p) for p in new_points]
        ways[way_id]["nds"] = new_nds if nds[-1] == end_node else new_nds[::-1]
        follow_moved_end(lmap, end_node, new_nds, {way_id})
        length = np.linalg.norm(np.diff(lmap.points([end_node] + new_nds[-len(new_points):]), axis=0), axis=1).sum()
        print("  way {} extended by {:.1f} m (+{} nodes)".format(way_id, length, len(new_points)))
    print("Extended {} ways through hard-coded points".format(len(extensions)))


def trim_and_extend(lmap, area, max_extension, skip_ways=()):
    """Move the open ends of the map to the border of the useful area.

    All lanes of a road end (lanelets sharing bounds) end at one line perpendicular to the road:
    - if a lane's center line ends inside the background image, the bounds are extended straight up to where the
      center line leaves the image, but not across other ways (e.g. where two roads separated by an island meet);
    - bounds ending outside the useful area are cut back to where the lanes' center lines enter the area;
    - other ends (outside the image but on tracks) stay.
    """
    ways = lmap.osm["ways"]
    report = []
    for group in open_ends(lmap):
        if any(w in skip_ways for w, _ in group["keys"]):
            continue
        centers =[lanelet_end_centerline(lmap, rid, which) for rid, which in group["lanelets"]]
        direction = np.mean([end_direction(c) for c in centers], axis=0)
        direction /= np.linalg.norm(direction)
        origin = np.mean([c[-1] for c in centers], axis=0)
        lanelet_ids = sorted(r for r, _ in group["lanelets"])
        proj = {key: float(np.dot(lmap.xy[n] - origin, direction)) for key, n in group["way_ends"]}
        trims = [float(np.dot(c[-1] - origin, direction)) + o for c in centers
                 for o in [trim_offset(c, direction, area)] if o is not None]
        trim_cut = max(trims) if trims else None
        extensions = [float(np.dot(c[-1] - origin, direction)) + o for c in centers
                      if area.contains(c[-1], image_only=True)
                      for o in [image_border_distance(c[-1], direction, area, max_extension)] if o is not None]
        extend_cut = max(extensions) if extensions else None
        if extend_cut is not None:
            for key, end_node in group["way_ends"]:
                end = lmap.xy[end_node]
                gap = extend_cut - proj[key]
                if gap <= 0 or not area.contains(end):
                    continue
                segments = [(lmap.xy[a], lmap.xy[b]) for other in ways.values() if end_node not in other["nds"]
                            for a, b in zip(other["nds"][:-1], other["nds"][1:])]
                hit = ray_hit_distance(end, direction, gap, np.array(segments))
                if hit is not None:
                    extend_cut = min(extend_cut, proj[key] + max(hit - 0.5, 0.0))
        gaps = {}
        for key, end_node in group["way_ends"]:
            if not area.contains(lmap.xy[end_node]):
                gaps[key] = trim_cut - proj[key] if trim_cut is not None else 0.0
            elif extend_cut is not None:
                gaps[key] = max(0.0, extend_cut - proj[key])
            else:
                gaps[key] = 0.0
        for (way_id, k), end_node in group["way_ends"]:
            nds = ways[way_id]["nds"] if k == -1 else ways[way_id]["nds"][::-1]  # end_node last
            moved_ways = {w for (w, _), n in group["way_ends"] if n == end_node}
            end_node = nds[-1]  # differs if the end followed a way moved at another open end before
            pts = lmap.points(nds)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            gap = gaps[(way_id, k)]
            if abs(gap) < 0.3:
                continue
            if gap > 0:
                # Extend straight with about the usual node spacing of the bound
                spacing = float(np.clip(np.median(seg), 1.0, 15.0))
                n_new = max(1, int(round(gap / spacing)))
                new_pts = [pts[-1] + direction * gap * (j + 1) / n_new for j in range(n_new)]
                new_nds = list(nds) + [lmap.add_node(q) for q in new_pts]
                action = "extended by {:.1f} m (+{} nodes)".format(gap, n_new)
            else:
                crossing = cut_polyline(pts, origin, direction, trim_cut)
                if crossing is None:
                    print("  WARNING: way {} does not cross the cut line of lanelets {}, left alone".format(
                        way_id, lanelet_ids))
                    continue
                i, point = crossing
                removed = nds[i + 1:]
                new_nds = list(nds[:i + 1])
                if np.linalg.norm(point - pts[i]) > 0.3:
                    new_nds.append(lmap.add_node(point))
                action = "trimmed by {:.1f} m (-{} nodes)".format(-gap, len(removed))
            ways[way_id]["nds"] = new_nds if k == -1 else new_nds[::-1]
            follow_moved_end(lmap, end_node, new_nds, moved_ways)
            report.append((way_id, dict(ways[way_id]["tags"]).get("type"), action, lanelet_ids))
    print("Moved {} way ends at open map ends:".format(len(report)))
    for way_id, way_type, action, lanelets in report:
        print("  way {} ({}) {}, lanelets {}".format(way_id, way_type, action, lanelets))


PHYSICAL_TYPES = {"road_border", "curbstone"}  # Way types that are no lane markings and stay in the map
# Passenger car whose swept path sets the width of the rebuilt connectors and is checked on every route: length,
# width and distance from its center to its rear axle [m] (the full-size car of the CBF simulation), and the clearance
# kept to both sides of its swept path
CAR_LENGTH, CAR_WIDTH, CAR_REAR_AXLE, CLEARANCE = 4.6, 2.0, 1.35, 0.25


def swept_extents(curvature):
    """Extents (left, right) [m] of the area that a car sweeps whose center follows a path with the given signed
    curvature (1/m, > 0: left turn), measured from the path: the inner end of the rear axle swings in, the outer
    front corner out."""
    curvature = np.asarray(curvature, dtype=float)
    radius = 1 / np.maximum(np.abs(curvature), 1e-6)
    axle = np.sqrt(np.maximum(radius ** 2 - CAR_REAR_AXLE ** 2, 0))  # Turn radius of the center of the rear axle
    inner = radius - (axle - CAR_WIDTH / 2)
    outer = np.hypot(axle + CAR_WIDTH / 2, CAR_REAR_AXLE + CAR_LENGTH / 2) - radius
    return np.where(curvature > 0, inner, outer), np.where(curvature > 0, outer, inner)


def lanelet_topology(lmap):
    """Predecessors and successors of every lanelet (Lanelet2: a successor starts at both end nodes)."""
    info = {rid: lmap.bounds(rel) for rid, rel in lmap.lanelets().items()}
    starts = {}
    for rid, (_, _, left, right) in info.items():
        starts.setdefault((left[0], right[0]), []).append(rid)
    preds, succs = {rid: [] for rid in info}, {rid: [] for rid in info}
    for rid, (_, _, left, right) in info.items():
        for other in starts.get((left[-1], right[-1]), []):
            succs[rid].append(other)
            preds[other].append(rid)
    return preds, succs


def lanelet_direction(lmap, rid, which, length=3.0):
    """Unit driving direction at the start or end of a lanelet, averaged over `length` meters."""
    center = lmap.centerline(lmap.osm["relations"][rid], 200)
    return end_direction(center, length) if which == "end" else -end_direction(center[::-1], length)


def mean_direction(directions):
    d = np.sum(directions, axis=0)
    return d / np.linalg.norm(d)


def heading_vector(degrees):
    return np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])


def turn_path(p0, h0, p1, h1, on_road=None, max_extra_turns=np.radians([30, 90, 150]),
              leads=(0, 1, 2, 3, 4, 6, 8, 10, 12, 15), spacing=1.0, end_pieces=0.0):
    """Center line from p0 (heading h0) to p1 (heading h1, radians) that turns as gently as possible.

    The path is a straight line, an arc, a straight line, an arc and a straight line (an arc-line-arc path as in
    Dubins' shortest paths, with straight lines of the lengths in `leads` before and after it). Among the paths whose
    turns add up to at most the first of `max_extra_turns` (more than the heading change) for which there are any and
    on which, if `on_road` (see `road_area`) is given, a car stays on the road with the area it sweeps (see
    `swept_extents`), or if there is none, with its width, it is the one with the largest radius (the shortest one if
    several have it). Returns its straight lines and arcs, and pieces of `end_pieces` meters at both ends (points
    about `spacing` meters apart, consecutive pieces share their end point), the curvature at their points, the
    radius and whether the car stays on the road."""
    def normal(h):
        return np.stack([-np.sin(h), np.cos(h)], axis=-1)

    def direction(h):
        return np.array([np.cos(h), np.sin(h)])

    def points(lead0, radius, s1, a1, straight, s2, a2, lead1, step=0.1):
        """Points, headings and curvatures of the path, and the arc length at the end of each part."""
        pts, heads, curvatures, h = [p0[None]], [np.array([h0])], [np.zeros(1)], h0
        for kind, value in (("line", lead0), ("arc", s1 * a1), ("line", straight), ("arc", s2 * a2),
                            ("line", lead1)):
            n = max(1, int(np.ceil((radius * abs(value) if kind == "arc" else value) / step)))
            t = np.linspace(0, 1, n + 1)[1:]
            if kind == "arc":
                center = pts[-1][-1] + np.sign(value) * radius * normal(h)
                hs = h + value * t
                pts.append(center - np.sign(value) * radius * normal(hs))
                curvatures.append(np.full(n, np.sign(value) / radius if value else 0.0))
            else:
                hs = np.full(n, h)
                pts.append(pts[-1][-1] + np.outer(value * t, direction(h)))
                curvatures.append(np.zeros(n))
            heads.append(hs)
            h = hs[-1]
        return np.vstack(pts), np.concatenate(heads), np.concatenate(curvatures), np.cumsum(
            [lead0, radius * a1, straight, radius * a2, lead1])

    def candidates(q0, q1, radius, max_turn):
        """Arc-line-arc paths from q0 (heading h0) to q1 (heading h1) with the given radius."""
        for s1, s2 in ((1, 1), (-1, -1), (1, -1), (-1, 1)):  # +1: left turn, -1: right turn
            c0, c1 = q0 + s1 * radius * normal(h0), q1 + s2 * radius * normal(h1)
            dist, psi = np.linalg.norm(c1 - c0), np.arctan2(*(c1 - c0)[::-1])
            if s1 == s2:
                straight, phi = dist, psi
            elif dist >= 2 * radius:
                straight = np.sqrt(dist ** 2 - 4 * radius ** 2)
                phi = psi - np.arctan2((s2 - s1) * radius, straight)
            else:
                continue
            a1, a2 = (s1 * (phi - h0)) % (2 * np.pi), (s2 * (h1 - phi)) % (2 * np.pi)
            a1, a2 = (0.0 if a > 2 * np.pi - 1e-6 else a for a in (a1, a2))
            if max(a1, a2) <= np.pi and a1 + a2 <= max_turn:
                yield radius * (a1 + a2) + straight, s1, a1, straight, s2, a2

    net = abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)
    chord = np.linalg.norm(p1 - p0)
    best, fallback = None, None
    for swept, max_extra_turn in ((swept, e) for swept in (True, False) for e in max_extra_turns):
        for lead0 in leads:
            for lead1 in leads:
                if lead0 + lead1 >= chord:
                    continue
                q0, q1 = p0 + lead0 * direction(h0), p1 - lead1 * direction(h1)
                for radius in np.arange(30.0, 0.95 if best is None else best[0] + 0.01, -0.1):
                    found = False
                    for length, *params in sorted(candidates(q0, q1, radius, net + max_extra_turn)):
                        pts, heads, curvatures, ends = points(lead0, radius, *params, lead1)
                        fallback = fallback or (radius, pts, curvatures, ends)
                        left, right = swept_extents(curvatures) if swept else (np.full(len(pts), CAR_WIDTH / 2),) * 2
                        if on_road is None or on_road(pts[::5], [pts + left[:, None] * normal(heads),
                                                                 pts - right[:, None] * normal(heads)]):
                            length += lead0 + lead1
                            if best is None or radius > best[0] + 1e-9 or length < best[1]:
                                best = (radius, length, pts, curvatures, ends)
                            found = True
                            break
                    if found:
                        break
        if best is not None:
            break
    if fallback is None:
        raise ValueError("No path from {} to {}".format(p0, p1))
    radius, pts, curvatures, ends, ok = (*best[:1], *best[2:], True) if best is not None else (*fallback, False)
    pts = pts + np.outer(np.linspace(0, 1, len(pts)), p1 - pts[-1])  # Numerical drift
    # Resample each straight line and arc on its own, so that the pieces start and end at points
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    ends = [e * s[-1] / ends[-1] for e in ends]
    cuts = [0.0] + sorted(e for e in list(ends[:-1]) + [end_pieces, s[-1] - end_pieces] if 0.5 < e < s[-1] - 0.5) + \
        [s[-1]]
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] > 0.5 or c == s[-1]]
    pieces, piece_curvatures = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        t = np.linspace(a, b, max(2, int(np.ceil((b - a) / spacing)) + 1))
        pieces.append(np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1))
        piece_curvatures.append(np.interp(t, s, curvatures))
    return pieces, piece_curvatures, radius, ok


def offset_bound(center, offset, start_point, end_point):
    """Offset a center line by `offset` (per point, > 0: to the left) and blend the result into the fixed end
    points: the residual at each end fades out along the line (smoothstep of the arc length)."""
    tangent = np.gradient(center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))])
    sigma = (s / s[-1]) ** 2 * (3 - 2 * s / s[-1])
    bound = center + normal * offset[:, None]
    return bound + np.outer(1 - sigma, start_point - bound[0]) + np.outer(sigma, end_point - bound[-1])


def road_area(lmap, tolerance=1.5, max_angle=np.radians(135)):
    """Function that tells whether a path is on the road: every point of its center line is on a drivable lanelet of
    the map (in its current state) that does not point against the path (more than `max_angle`, oncoming traffic), or
    on no lanelet but at most `tolerance` meters away from one (the lanelets do not cover all of the road, e.g. not
    paved medians that vehicles cross), and the given lines to its sides cross no road border or high curb."""
    polygons, centers, edges = [], [], []
    for rel in lmap.lanelets().values():
        if dict(rel["tags"]).get("subtype") in DRIVABLE_SUBTYPES:
            _, _, left, right = lmap.bounds(rel)
            outline = np.vstack([lmap.points(left), lmap.points(right)[::-1]])
            polygons.append(Path(outline))
            centers.append(lmap.centerline(rel))
            edges.append(np.stack([outline, np.roll(outline, -1, axis=0)], axis=1))
    edges = np.concatenate(edges)
    borders = [lmap.points(w["nds"]) for w in lmap.osm["ways"].values()
               if dict(w["tags"]).get("type") == "road_border" or
               dict(w["tags"]).get("type") == "curbstone" and dict(w["tags"]).get("subtype") != "low"]
    borders = np.concatenate([np.stack([b[:-1], b[1:]], axis=1) for b in borders if len(b) > 1])

    def crosses(line):
        a, e = line[:-1, None], (line[1:] - line[:-1])[:, None]
        c, f = borders[None, :, 0], borders[None, :, 1] - borders[None, :, 0]
        denom = e[..., 0] * f[..., 1] - e[..., 1] * f[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((c - a)[..., 0] * f[..., 1] - (c - a)[..., 1] * f[..., 0]) / denom
            u = ((c - a)[..., 0] * e[..., 1] - (c - a)[..., 1] * e[..., 0]) / denom
        return bool(((t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)).any())

    def on_road(center, sides):
        heading = np.gradient(center, axis=0)
        heading /= np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-12)
        inside, along = np.zeros(len(center), dtype=bool), np.zeros(len(center), dtype=bool)
        for polygon, lanelet_center in zip(polygons, centers):
            hit = polygon.contains_points(center)
            if hit.any():
                tangent = np.gradient(lanelet_center, axis=0)
                nearest = np.argmin(((center[hit, None] - lanelet_center[None]) ** 2).sum(-1), axis=1)
                cosine = (tangent[nearest] * heading[hit]).sum(1) / np.linalg.norm(tangent[nearest], axis=1)
                inside |= hit
                along[np.where(hit)[0][cosine > np.cos(max_angle)]] = True
        a, e = edges[None, :, 0], edges[None, :, 1] - edges[None, :, 0]
        t = np.clip(((center[:, None] - a) * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-12), 0, 1)
        near = np.linalg.norm(a + t[..., None] * e - center[:, None], axis=-1).min(axis=1) <= tolerance
        return (along | ~inside & near).all() and not any(crosses(side) for side in sides)
    return on_road


def widen_lanelets(lmap, specs):
    """Move the nodes of a bound of lanelets outward where the lanelet is narrower than a given width.

    Each spec is (lanelets, side, width): the nodes of the "left" or "right" bound of the lanelets move away from
    the other bound, perpendicular to it, until they are `width` meters from it. Other ways that use these nodes
    (e.g. the bound of the lane in the opposite direction, or an area outline) move with them."""
    for lanelet_ids, side, width in specs:
        moved = {}
        for rid in lanelet_ids:
            _, _, left, right = lmap.bounds(lmap.osm["relations"][rid])
            nodes, other = (left, right) if side == "left" else (right, left)
            line = lmap.points(other)
            for n in nodes:
                a, e = line[:-1], np.diff(line, axis=0)
                t = np.clip(((lmap.xy[n] - a) * e).sum(1) / np.maximum((e * e).sum(1), 1e-12), 0, 1)
                foot = a + t[:, None] * e
                k = int(np.argmin(np.linalg.norm(foot - lmap.xy[n], axis=1)))
                normal = np.array([-e[k][1], e[k][0]]) / np.linalg.norm(e[k]) * (1 if side == "left" else -1)
                distance = float(np.dot(lmap.xy[n] - foot[k], normal))
                if distance < width and (n not in moved or moved[n][1] < width - distance):
                    moved[n] = (foot[k] + normal * width, width - distance)
        for n, (point, shift) in moved.items():
            lat, lon = utm_to_latlon(point[0] + lmap.x0, point[1] + lmap.y0, lmap.zone)
            attrs = dict(lmap.osm["nodes"][n]["attrs"])
            attrs["lat"], attrs["lon"] = "{:.11f}".format(float(lat)), "{:.11f}".format(float(lon))
            attrs.setdefault("action", "modify")
            lmap.osm["nodes"][n]["attrs"] = list(attrs.items())
            lmap.xy[n] = point
        print("Widened lanelets {} to {:.1f} m: moved {} nodes of their {} bound by up to {:.2f} m".format(
            lanelet_ids, width, len(moved), side, max([m[1] for m in moved.values()] or [0])))


def rebuild_connectors(lmap, connectors, spacing=1.0, flare=1.0):
    """Replace chains of lanelets (in intersections) by lanelets with smooth center lines that a car fits in.

    The inputs draw the lanes through intersections as chains of short lanelets with straight bounds that meet at
    an angle, so that the center lines turn by up to 60 degrees at a single point, which no vehicle can follow.
    Each chain of consecutive lanelets is replaced by a new center line from the start of the chain to its end,
    tangent to the lanelets before and after the chain: straight lines and arcs with the largest radius for which
    a car stays on the road (see `turn_path`, `road_area`). The width changes smoothly from the width at the start of
    the chain to the width at its end, but in turns the lanelet is wide enough for the area that the car sweeps plus
    CLEARANCE to both sides (see `swept_extents`), reached within `flare` meters from the ends. Every straight line
    and arc, and the first and last `flare` meters, become a lanelet of its own (the first one keeps the id and tags
    of the first lanelet of the chain, the others get new ids), since the center line of a lanelet with bounds of
    different shapes, computed from points at the same fraction of the length of both bounds, is not the line midway
    between them. The end nodes of the chain stay, so its connections to other lanelets stay. The new bounds are
    virtual lines; curbs and road borders that bounded the chain stay in the map unchanged.

    Each connector is a dict with
      lanelets: the chain of lanelets in driving direction,
      start_with/end_with: (optional) a lanelet of the input (possibly rebuilt before) where the chain starts/ends
        instead (where lanes fork or merge),
      start_heading/end_heading: (optional) driving direction in degrees (x east, y north) at the start/end of the
        chain. Default: the direction of the lanelets before/after the chain (mean if there are several).
    Returns the ways that bounded only the replaced lanelets and are kept (curbs, road borders).
    """
    relations, ways = lmap.osm["relations"], lmap.osm["ways"]
    on_road = road_area(lmap)
    # End nodes of the lanelets of the input (the end nodes of each chain stay, see start_with/end_with)
    starts, ends = {}, {}
    for rid, rel in lmap.lanelets().items():
        _, _, left, right = lmap.bounds(rel)
        starts[rid], ends[rid] = (left[0], right[0]), (left[-1], right[-1])
    keep_ways = set()
    for connector in connectors:
        chain = connector["lanelets"]
        _, succs = lanelet_topology(lmap)
        for a, b in zip(chain[:-1], chain[1:]):
            if b not in succs[a]:
                raise ValueError("Lanelet {} does not succeed lanelet {}".format(b, a))
        start_nodes = starts[connector.get("start_with", chain[0])]
        end_nodes = ends[connector.get("end_with", chain[-1])]
        others = {rid: lmap.bounds(rel)[2:] for rid, rel in lmap.lanelets().items() if rid not in chain}
        before = [lanelet_direction(lmap, r, "end") for r, (left, right) in others.items()
                  if (left[-1], right[-1]) == start_nodes]
        after = [lanelet_direction(lmap, r, "start") for r, (left, right) in others.items()
                 if (left[0], right[0]) == end_nodes]
        # Without lanelets before/after the chain: the direction of the lanelet where it starts/ends
        first = next(r for r in (connector.get("start_with"), chain[0]) if r in relations)
        last = next(r for r in (connector.get("end_with"), chain[-1]) if r in relations)
        d0 = heading_vector(connector["start_heading"]) if "start_heading" in connector else mean_direction(
            before or [lanelet_direction(lmap, first, "start")])
        d1 = heading_vector(connector["end_heading"]) if "end_heading" in connector else mean_direction(
            after or [lanelet_direction(lmap, last, "end")])
        (start_left, start_right), (end_left, end_right) = start_nodes, end_nodes
        p0, p1 = (lmap.xy[start_left] + lmap.xy[start_right]) / 2, (lmap.xy[end_left] + lmap.xy[end_right]) / 2
        pieces, curvatures, radius, ok = turn_path(p0, np.arctan2(*d0[::-1]), p1, np.arctan2(*d1[::-1]), on_road,
                                                   spacing=spacing, end_pieces=flare)
        center = np.vstack([pieces[0]] + [piece[1:] for piece in pieces[1:]])
        curvature = np.concatenate([curvatures[0]] + [c[1:] for c in curvatures[1:]])
        joints = np.cumsum([len(piece) - 1 for piece in pieces])  # Index of the last point of each piece

        # Half widths: from the width at the start to the width at the end, but at least the swept path of a car
        # (plus clearance) whose body around the point may be in a turn
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))])
        w0, w1 = (np.linalg.norm(lmap.xy[a] - lmap.xy[b]) for a, b in (start_nodes, end_nodes))
        base = (w0 + (w1 - w0) * (s / s[-1]) ** 2 * (3 - 2 * s / s[-1])) / 2
        ramp = np.clip(np.minimum(s, s[-1] - s) / flare, 0, 1)
        # Both sides as wide as the outer side needs, since the center line is computed as the middle of the bounds.
        # The car's body extends CAR_LENGTH / 2 to both sides of the point of the path; beyond, the bounds flare at
        # 45 degrees. The pieces at the ends, where the bounds flare from the width of the chain's ends, are
        # lanelets of their own (see above).
        extent = np.maximum(*swept_extents(curvature)) + CLEARANCE
        needed = (extent[None] - np.maximum(np.abs(s[:, None] - s[None]) - CAR_LENGTH / 2, 0)).max(axis=1)
        half_width = base + np.maximum(needed - base, 0) * ramp
        left = offset_bound(center, half_width, lmap.xy[start_left], lmap.xy[end_left])
        right = offset_bound(center, -half_width, lmap.xy[start_right], lmap.xy[end_right])

        # Nodes of the bounds: the end nodes of the chain stay. A new node that would almost coincide with another
        # node (where bounds cross) is moved along its bound, so that the nodes are not taken for duplicates.
        existing = list(lmap.xy.values())

        def add_nodes(bound):
            nodes = []
            for i in range(1, len(bound) - 1):
                step = (bound[i + 1] - bound[i]) / np.linalg.norm(bound[i + 1] - bound[i])
                shifts = [d for d in (0.0, 0.2, -0.2, 0.4, -0.4)
                          if np.linalg.norm(np.array(existing) - (bound[i] + d * step), axis=1).min() >= 0.1]
                point = bound[i] + (shifts[0] if shifts else 0.0) * step
                nodes.append(lmap.add_node(point))
                existing.append(point)
            return nodes

        left_nodes = [start_left] + add_nodes(left) + [end_left]
        right_nodes = [start_right] + add_nodes(right) + [end_right]
        replaced_ways = {ref for rid in chain for t, ref, _ in relations[rid]["members"] if t == "way"}
        tags = relations[chain[0]]["tags"]
        new_ids = []
        for k, (i0, i1) in enumerate(zip(np.concatenate([[0], joints[:-1]]), joints)):
            left_way = lmap.add_way(left_nodes[i0:i1 + 1], [("type", "virtual")])
            right_way = lmap.add_way(right_nodes[i0:i1 + 1], [("type", "virtual")])
            if k == 0:
                rid = chain[0]
                for member in relations[rid]["members"]:
                    if member[0] == "way" and member[2] in ("left", "right"):
                        member[1] = left_way if member[2] == "left" else right_way
            else:
                rid = lmap.next_id
                lmap.next_id -= 1
                relations[rid] = {"attrs": [("id", str(rid)), ("action", "modify"), ("visible", "true")],
                                  "tags": list(tags),
                                  "members": [["way", left_way, "left"], ["way", right_way, "right"]]}
            new_ids.append(rid)
        # Other lanelets of the chain are deleted; references to them refer to the first new lanelet instead
        for rid in chain[1:]:
            del relations[rid]
        for rel in relations.values():
            members = []
            for t, ref, role in rel["members"]:
                member = [t, chain[0] if t == "relation" and ref in chain else ref, role]
                if member not in members:
                    members.append(member)
            rel["members"][:] = members
        keep_ways |= {w for w in replaced_ways if dict(ways[w]["tags"]).get("type") in PHYSICAL_TYPES}
        print("Rebuilt lanelets {} as {}: radius {:.1f} m{}, length {:.1f} m, width {:.1f}-{:.1f} m".format(
            chain, new_ids, radius, "" if ok else " (WARNING: a car leaves the road)", s[-1],
            2 * half_width.min(), 2 * half_width.max()))
    return keep_ways


def route_geometry(lmap, spacing=1.44):
    """Per route (sequence of drivable lanelets from one without predecessor to one without successor): the
    smallest radius of its center line and where it is, and the smallest margin of the width of its lanelets over
    the width of the area that a car following the center line sweeps (see `swept_extents`) and where it is. The
    center line is computed like in the simulation: both bounds of each lanelet are resampled to the same number of
    points, about `spacing` meters apart, and averaged."""
    preds, succs = lanelet_topology(lmap)
    drivable = {rid for rid, rel in lmap.lanelets().items() if dict(rel["tags"]).get("subtype") in DRIVABLE_SUBTYPES}
    routes = []

    def extend(route):
        nexts = [s for s in succs[route[-1]] if s in drivable and s not in route]
        if not nexts:
            routes.append(route)
        for s in nexts:
            extend(route + [s])

    for rid in sorted(drivable, key=abs):
        if not [p for p in preds[rid] if p in drivable]:
            extend([rid])
    result = []
    for route in routes:
        centers, widths = [], []
        for rid in route:
            _, _, left, right = lmap.bounds(lmap.osm["relations"][rid])
            left_pts, right_pts = lmap.points(left), lmap.points(right)
            length = max(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in (left_pts, right_pts))
            num = max(2, int(np.ceil(length / spacing)) + 1)
            left_pts, right_pts = resample(left_pts, num), resample(right_pts, num)
            first = 0 if not centers else 1
            centers.append(((left_pts + right_pts) / 2)[first:])
            widths.append(np.linalg.norm(left_pts - right_pts, axis=1)[first:])
        center, width = np.concatenate(centers), np.concatenate(widths)
        step = np.diff(center, axis=0)
        seg = np.linalg.norm(step, axis=1)
        heading = np.arctan2(step[:, 1], step[:, 0])
        turn = (np.diff(heading) + np.pi) % (2 * np.pi) - np.pi
        curvature = np.concatenate([[0], turn / np.maximum((seg[:-1] + seg[1:]) / 2, 1e-9), [0]])
        radius = 1 / np.maximum(np.abs(curvature), 1e-9)
        left, right = swept_extents(curvature)
        margin = width - left - right
        i, j = int(np.argmin(radius)), int(np.argmin(margin))
        result.append((route, float(radius[i]), center[i], float(margin[j]), center[j]))
    return result


def check_route_geometry(lmap, min_radius=5.0):
    """Routes whose center line turns tighter than `min_radius` (about the smallest turn radius of a car) or whose
    lanelets are narrower than the area that a car following the center line sweeps."""
    bad = []
    for route, radius, radius_point, margin, margin_point in route_geometry(lmap):
        if radius < min_radius:
            print("  WARNING: route {} -> {} turns with a radius of {:.1f} m at ({:.1f}, {:.1f}) (lanelets {})".format(
                route[0], route[-1], radius, *radius_point, route))
        if margin < 0:
            print("  WARNING: route {} -> {} is {:.2f} m narrower than a car sweeps at ({:.1f}, {:.1f}) (lanelets "
                  "{})".format(route[0], route[-1], -margin, *margin_point, route))
        if radius < min_radius or margin < 0:
            bad.append(route)
    return bad


def unused_elements(osm):
    """Ways that no relation uses and nodes that neither a way nor a relation uses."""
    members = [(t, ref) for rel in osm["relations"].values() for t, ref, _ in rel["members"]]
    used_ways = {ref for t, ref in members if t == "way"}
    used_nodes = {n for way in osm["ways"].values() for n in way["nds"]} | {ref for t, ref in members if t == "node"}
    return ({w for w in osm["ways"] if w not in used_ways}, {n for n in osm["nodes"] if n not in used_nodes})


def remove_unused(lmap, keep_ways=(), keep_nodes=()):
    unused_ways, _ = unused_elements(lmap.osm)
    unused_ways = [w for w in lmap.osm["ways"] if w in unused_ways and w not in keep_ways]
    for w in unused_ways:
        del lmap.osm["ways"][w]
    _, unused_nodes = unused_elements(lmap.osm)
    unused_nodes = [n for n in lmap.osm["nodes"] if n in unused_nodes and n not in keep_nodes]
    for n in unused_nodes:
        del lmap.osm["nodes"][n]
    print("Removed {} unused ways and {} unused nodes".format(len(unused_ways), len(unused_nodes)))


def orient_bounds(lmap):
    """Reverse the node order of lanelet bounds stored against their lanelet's driving direction.

    Cosmetic for Lanelet2, which derives the direction from the left/right roles, but it makes the way arrows in
    JOSM show the driving direction. Ways shared by lanelets that need opposite orders are left alone.
    """
    needs = {}
    for rel in lmap.lanelets().values():
        left_id, right_id, left, right = lmap.bounds(rel)
        for way_id, nds in ((left_id, left), (right_id, right)):
            needs.setdefault(way_id, set()).add(nds != lmap.osm["ways"][way_id]["nds"])
    reversed_ways = [w for w, n in needs.items() if n == {True}]
    for w in reversed_ways:
        lmap.osm["ways"][w]["nds"].reverse()
    conflicts = [w for w, n in needs.items() if len(n) > 1]
    print("Reversed {} ways to driving direction{}".format(
        len(reversed_ways), "; left {} shared ways with conflicting directions: {}".format(len(conflicts), conflicts)
        if conflicts else ""))


def check_map(lmap):
    """Members exist, lanelets have both bounds and multipolygon outlines are closed rings."""
    ok = True
    elements = {"node": lmap.osm["nodes"], "way": lmap.osm["ways"], "relation": lmap.osm["relations"]}
    for rid, rel in lmap.osm["relations"].items():
        missing = [(t, ref) for t, ref, _ in rel["members"] if ref not in elements[t]]
        if missing:
            print("  ERROR: relation {} references missing members {}".format(rid, missing))
            ok = False
        tags = dict(rel["tags"])
        if tags.get("type") == "multipolygon":
            ends = {}
            for ref in [ref for t, ref, _ in rel["members"] if t == "way"]:
                for n in (lmap.osm["ways"][ref]["nds"][0], lmap.osm["ways"][ref]["nds"][-1]):
                    ends[n] = ends.get(n, 0) + 1
            if any(c % 2 for c in ends.values()):
                print("  ERROR: multipolygon {} is not a closed ring".format(rid))
                ok = False
    return ok


def check_duplicate_nodes(lmap, tol=0.05):
    """Distinct nodes used by ways at (almost) the same position, which keeps the ways from being connected."""
    used = sorted({n for way in lmap.osm["ways"].values() for n in way["nds"]})
    xy = lmap.points(used)
    i, j = np.nonzero(np.triu(np.linalg.norm(xy[:, None] - xy[None], axis=2) < tol, k=1))
    for a, b in zip(i, j):
        print("  WARNING: nodes {} and {} are {:.3f} m apart".format(used[a], used[b], np.linalg.norm(xy[a] - xy[b])))
    return len(i) == 0


def check_directions(lmap, track_frames):
    """Vehicle samples inside each one-way drivable lanelet that drive against it. Since lanelets overlap in
    intersections, a lanelet is reported if more samples drive against it and against all other lanelets containing
    them than along it, and, if no sample drives along it, most of the samples against it do so (otherwise it is a
    lane across other lanes that no recorded vehicle took)."""
    vehicles = track_frames[~track_frames["class"].isin(["pedestrian", "bicycle"])]
    vehicles = vehicles[np.hypot(vehicles["xVelocity"], vehicles["yVelocity"]) > 1.0]
    pts, vel = vehicles[["xCenter", "yCenter"]].values, vehicles[["xVelocity", "yVelocity"]].values
    vel = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    explained = np.zeros(len(pts), dtype=bool)
    samples = {}
    for rid, rel in lmap.lanelets().items():
        tags = dict(rel["tags"])
        if tags.get("one_way") == "no" or tags.get("subtype") not in DRIVABLE_SUBTYPES:
            continue
        _, _, left, right = lmap.bounds(rel)
        left_pts, right_pts = resample(lmap.points(left)), resample(lmap.points(right))
        inside = np.nonzero(Path(np.vstack([left_pts, right_pts[::-1]])).contains_points(pts))[0]
        if not len(inside):
            continue
        center = (left_pts + right_pts) / 2
        tangent = np.gradient(center, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        nearest = np.argmin(((pts[inside][:, None] - center[None]) ** 2).sum(-1), axis=1)
        along = (tangent[nearest] * vel[inside]).sum(axis=1)  # cosine
        samples[rid] = (inside, along)
        explained[inside[along > 0.5]] = True
    wrong = []
    for rid, (inside, along) in samples.items():
        fwd, bwd = int((along > 0.5).sum()), int((along < -0.5).sum())
        bwd_unexplained = int(((along < -0.5) & ~explained[inside]).sum())
        # Without samples along it, a lanelet is not reported if other lanelets explain most samples against it
        # (a lanelet across other lanes that no recorded vehicle drove)
        if bwd_unexplained > max(fwd, 5) and (fwd > 0 or bwd_unexplained > bwd / 2):
            wrong.append((rid, fwd, bwd, bwd_unexplained))
    if wrong:
        for rid, fwd, bwd, bwd_unexplained in wrong:
            print("  lanelet {}: {} samples along, {} against its direction ({} of them against all lanelets "
                  "there)".format(rid, fwd, bwd, bwd_unexplained))
    else:
        print("  all one-way lanelets agree with the driving direction of the tracks")
    print("  {:.2%} of the moving vehicle samples are in no lanelet that points their way".format(
        1 - explained.mean()))
    return wrong


def check_open_ends(lmap, area, min_border_distance=3.0):
    """Drivable lanelets without predecessor/successor whose open end is inside the background image, more than
    `min_border_distance` from its border, i.e. dead ends in the middle of the map. Lane ends of zero width (lanes
    that begin or end beside another lane) are expected."""
    info = {rid: lmap.bounds(rel) for rid, rel in lmap.lanelets().items()
            if dict(rel["tags"]).get("subtype") in DRIVABLE_SUBTYPES}
    starts = {(l[0], r[0]) for _, _, l, r in info.values()}
    ends = {(l[-1], r[-1]) for _, _, l, r in info.values()}
    found = []
    for rid, (_, _, left, right) in info.items():
        for which, k, others in (("start", 0, ends), ("end", -1, starts)):
            if (left[k], right[k]) in others:
                continue
            center = lanelet_end_centerline(lmap, rid, which)
            if not area.contains(center[-1], image_only=True):
                continue
            distance = image_border_distance(center[-1], end_direction(center), area, 1000.0)
            if distance is not None and distance <= min_border_distance:
                continue
            width = np.linalg.norm(lmap.xy[left[k]] - lmap.xy[right[k]])
            found.append(rid)
            print("  {}: lanelet {} {} has no {} {:.0f} m inside the image, width there {:.1f} m{}".format(
                "note" if width < 1.0 else "WARNING", rid, which, "predecessor" if which == "start" else "successor",
                distance if distance is not None else 1000, width, " (lane taper)" if width < 1.0 else ""))
    return found


def fix_map(name, cfg, config):
    root = config["{}_dir".format(cfg["dataset"])]
    if root is None:
        raise SystemExit("{} needs --{}_dir (the folder of the downloaded dataset).".format(name, cfg["dataset"]))
    cfg = dict(cfg, dataset_dir=os.path.join(root, cfg["dataset_dir"]), input=os.path.join(root, cfg["input"]),
               output=os.path.join(config["out_dir"], name + ".osm"))
    print("=" * 100 + "\n{}: {} -> {}".format(name, cfg["input"], cfg["output"]))
    recordings = cfg["recordings"] or [cfg["recording"]]
    recording_meta = read_recording_meta(os.path.join(cfg["dataset_dir"],
                                                      "{:02d}_recordingMeta.csv".format(recordings[0])))
    scale_down_factor = SCALE_DOWN_FACTOR[cfg["dataset"]]
    px_per_meter = 1 / (recording_meta["orthoPxToMeter"] * scale_down_factor)

    print("Loading tracks of location {}".format(recording_meta["locationId"]))
    track_frames, backgrounds = load_location(cfg["dataset_dir"], recording_meta["locationId"], cfg["recordings"])
    area = UsefulArea(backgrounds, track_frames, px_per_meter, config["track_buffer"])
    lmap = LaneletMap(load_osm(cfg["input"]), recording_meta)
    n_nodes, n_ways, n_rels = len(lmap.osm["nodes"]), len(lmap.osm["ways"]), len(lmap.osm["relations"])
    # Ways and nodes that no relation uses in the input (e.g. markings) are kept if they touch the useful area
    standalone_ways, standalone_nodes = unused_elements(lmap.osm)
    keep_ways = {w for w in standalone_ways if any(area.contains(lmap.xy[n]) for n in lmap.osm["ways"][w]["nds"])}
    keep_nodes = {n for n in standalone_nodes if area.contains(lmap.xy[n])}

    print("Checking the input:")
    if check_map(lmap) & check_duplicate_nodes(lmap):
        print("  references, multipolygon rings and node positions OK")
    print("Lanelets driving against the tracks in the input:")
    check_directions(lmap, track_frames)
    print("Dead ends in the input:")
    check_open_ends(lmap, area)
    merge_nodes(lmap, cfg.get("merge_nodes", {}))
    flip_lanelets(lmap, cfg.get("flip", []))
    close_multipolygons(lmap, cfg.get("close_multipolygons", {}))
    remove_multipolygon_members(lmap, cfg.get("remove_multipolygon_members", {}))
    connect_lanelets(lmap, cfg.get("connect", []))
    delete_outside_lanelets(lmap, area, config["min_inside_fraction"], config["min_inside_fraction_other"],
                            cfg.get("delete", []), config["verbose"])
    delete_outside_areas(lmap, area)
    remove_dangling_references(lmap)
    remove_unused(lmap, keep_ways, keep_nodes)
    apply_hardcoded_geometry(lmap, cfg.get("hardcoded_extensions", {}), cfg.get("island_outlines", {}),
                             cfg.get("hardcoded_points", {}))
    trim_and_extend(lmap, area, config["max_extension"], cfg.get("hardcoded_extensions", {}))
    delete_outside_areas(lmap, area, partly=True)
    remove_dangling_references(lmap)
    remove_unused(lmap, keep_ways, keep_nodes)
    if cfg.get("connectors") or cfg.get("widen"):
        print("Routes that a car cannot follow, before rebuilding the connectors:")
        check_route_geometry(lmap)
        widen_lanelets(lmap, cfg.get("widen", []))
        keep_ways |= rebuild_connectors(lmap, cfg.get("connectors", []))
        remove_unused(lmap, keep_ways, keep_nodes)
    orient_bounds(lmap)

    print("Checking the result:")
    if check_map(lmap) & check_duplicate_nodes(lmap):
        print("  references, multipolygon rings and node positions OK")
    check_directions(lmap, track_frames)
    print("Dead ends:")
    if not check_open_ends(lmap, area):
        print("  none")
    print("Routes that a car cannot follow:")
    if not check_route_geometry(lmap):
        print("  none")
    write_osm(lmap.osm, cfg["output"])
    print("Wrote {}: nodes {} -> {}, ways {} -> {}, relations {} -> {}".format(
        cfg["output"], n_nodes, len(lmap.osm["nodes"]), n_ways, len(lmap.osm["ways"]), n_rels,
        len(lmap.osm["relations"])))


def main():
    config = create_args()
    for name in config["maps"]:
        fix_map(name, MAPS[name], config)


if __name__ == "__main__":
    main()
