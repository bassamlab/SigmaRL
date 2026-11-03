# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared by the Lanelet2 map scripts (fix_lanelet_map.py, make_highd_lanelet_maps.py)."""
import math
import xml.etree.ElementTree as ET

import numpy as np

# WGS84 ellipsoid and UTM constants
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def utm_zone(lon):
    return int((lon + 180) // 6) + 1


def _central_meridian(zone):
    return math.radians((zone - 1) * 6 - 180 + 3)


def _meridian_arc(lat):
    e2 = _E2
    return _A * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
                 - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
                 + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
                 - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))


def latlon_to_utm(lat, lon, zone):
    """Project WGS84 lat/lon (degrees) to UTM easting/northing (meters) in the given zone, northern hemisphere.

    Transverse Mercator series from Snyder, "Map Projections - A Working Manual" (USGS 1987), pp. 61-64.
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = _EP2 * np.cos(lat) ** 2
    a = np.cos(lat) * (lon - _central_meridian(zone))
    easting = _K0 * n * (a + (1 - t + c) * a ** 3 / 6
                         + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120) + 500000.0
    northing = _K0 * (_meridian_arc(lat) + n * np.tan(lat) * (
        a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
        + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720))
    return easting, northing


def utm_to_latlon(easting, northing, zone):
    """Inverse of latlon_to_utm (Snyder pp. 63-64). Returns lat, lon in degrees."""
    x = np.asarray(easting, dtype=float) - 500000.0
    m = np.asarray(northing, dtype=float) / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    c1 = _EP2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = _A / np.sqrt(1 - _E2 * np.sin(phi1) ** 2)
    r1 = _A * (1 - _E2) / (1 - _E2 * np.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = _central_meridian(zone) + (
        d - (1 + 2 * t1 + c1) * d ** 3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(phi1)
    return np.degrees(lat), np.degrees(lon)


def read_osm(path):
    """Return nodes {id: (lat, lon)}, ways {id: (node ids, tags)} and relations [(id, members, tags)]."""
    root = ET.parse(path).getroot()
    nodes = {int(n.get("id")): (float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}
    ways = {}
    for w in root.iter("way"):
        ways[int(w.get("id"))] = ([int(nd.get("ref")) for nd in w.iter("nd")],
                                  {t.get("k"): t.get("v") for t in w.iter("tag")})
    relations = []
    for r in root.iter("relation"):
        members = [(m.get("type"), int(m.get("ref")), m.get("role")) for m in r.iter("member")]
        relations.append((int(r.get("id")), members, {t.get("k"): t.get("v") for t in r.iter("tag")}))
    return nodes, ways, relations


def resample(polyline, num=50):
    """Resample a polyline to num points equally spaced along its arc length."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(polyline[:1], num, axis=0)
    t = np.linspace(0, s[-1], num)
    return np.stack([np.interp(t, s, polyline[:, 0]), np.interp(t, s, polyline[:, 1])], axis=1)


def side_of(line, point):
    """Cross product sign of point relative to the closest segment of line: > 0 left, < 0 right (x right, y up)."""
    d = np.linalg.norm(line - point, axis=1)
    i = min(int(np.argmin(d)), len(line) - 2)
    a, b = line[i], line[i + 1]
    return (b - a)[0] * (point - a)[1] - (b - a)[1] * (point - a)[0]


def align_bounds(left, right, y_down=False):
    """Orient the bounds of a lanelet in its driving direction, like lanelet2::geometry::align on map load.

    In Lanelet2 the driving direction follows from which way is the "left" and which the "right" member,
    not from the order of the way's nodes: the left bound is inverted if the right bound is not on its right
    and vice versa. Returns (left, right, left_inverted, right_inverted). Points are N x 2 arrays; set y_down
    for image coordinates.
    """
    sign = -1 if y_down else 1

    def middle(ls):
        return ls[len(ls) // 2] if len(ls) > 2 else ls.mean(axis=0)

    left_inverted = sign * side_of(left, middle(right)) > 0  # right bound is left of the left bound
    if left_inverted:
        left = left[::-1]
    right_inverted = sign * side_of(right, middle(left)) < 0
    if right_inverted:
        right = right[::-1]
    return left, right, left_inverted, right_inverted
