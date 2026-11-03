#!/usr/bin/env python3
"""Scale the geometry in an OpenStreetMap XML file around its map center.

The scale percentage is relative to the original map size: ``100`` leaves the
geometry unchanged, ``120`` makes it 20% larger, and ``80`` makes it 20%
smaller.  The center is the center of the latitude/longitude bounding box of
all OSM ``node`` elements.

Coordinates are converted to a local east/north plane before scaling.  This
keeps the scale physically uniform for local maps instead of treating one
degree of latitude and one degree of longitude as the same distance.  The
script does not change node IDs, way references, tags, or any other OSM
metadata.  Existing ``bounds`` elements are transformed with the same
operation, preserving their declared extent and other attributes.

Example::

    python scripts/scale_osm_map.py \
        sigmarl/scenarios/assets/maps/interchange_3.osm \
        /tmp/interchange_3_120.osm \
        --percentage 120

This utility is intended for local road maps.  For very large maps or maps
crossing the antimeridian, use a GIS workflow with an explicitly selected
projected coordinate reference system instead.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# WGS84's semi-major axis.  The exact radius is not important for a uniform
# scale around one local center, but using meters makes the intent explicit.
EARTH_RADIUS_M = 6_378_137.0
OUTPUT_DECIMAL_PLACES = 12


@dataclass(frozen=True)
class ScaleSummary:
    """Summary of a completed OSM geometry transformation."""

    node_count: int
    percentage: float
    center_latitude: float
    center_longitude: float


def _local_name(tag: object) -> str | None:
    """Return an XML tag's local name, including for namespaced XML."""

    if not isinstance(tag, str):
        return None
    return tag.rsplit("}", maxsplit=1)[-1]


def _register_namespaces(input_path: Path) -> None:
    """Preserve namespace prefixes when ElementTree serializes the XML."""

    seen: set[tuple[str, str]] = set()
    for _, namespace in ET.iterparse(input_path, events=("start-ns",)):
        prefix, uri = namespace
        prefix = prefix or ""
        if prefix == "xml" or (prefix, uri) in seen:
            continue
        ET.register_namespace(prefix, uri)
        seen.add((prefix, uri))


def _parse_tree(input_path: Path) -> ET.ElementTree:
    """Parse an OSM XML file while retaining XML comments."""

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(input_path, parser=parser)


def _node_records(root: ET.Element) -> list[tuple[ET.Element, float, float]]:
    """Read every coordinate-bearing OSM node from ``root``."""

    records: list[tuple[ET.Element, float, float]] = []
    for element in root.iter():
        if _local_name(element.tag) != "node":
            continue

        node_id = element.get("id", "<unknown>")
        lat_text = element.get("lat")
        lon_text = element.get("lon")
        if lat_text is None or lon_text is None:
            raise ValueError(
                f"Node {node_id!r} is missing a required 'lat' or 'lon' attribute."
            )

        try:
            latitude = float(lat_text)
            longitude = float(lon_text)
        except ValueError as error:
            raise ValueError(
                f"Node {node_id!r} has a non-numeric latitude or longitude."
            ) from error

        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError(f"Node {node_id!r} has a non-finite coordinate.")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Node {node_id!r} has an invalid latitude: {latitude}.")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"Node {node_id!r} has an invalid longitude: {longitude}.")

        records.append((element, latitude, longitude))

    if not records:
        raise ValueError("The input XML does not contain any OSM node elements.")
    return records


def _format_coordinate(value: float) -> str:
    """Format a coordinate without unnecessary trailing zeroes."""

    formatted = f"{value:.{OUTPUT_DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    if formatted in {"", "-0"}:
        return "0"
    return formatted


def _normalize_longitude(longitude: float) -> float:
    """Keep a longitude in the conventional inclusive ``[-180, 180]`` range."""

    if -180.0 <= longitude <= 180.0:
        return longitude
    return (longitude + 180.0) % 360.0 - 180.0


def _to_local_meters(
    latitude: float,
    longitude: float,
    center_latitude: float,
    center_longitude: float,
) -> tuple[float, float]:
    """Convert a nearby latitude/longitude pair to local east/north meters."""

    center_latitude_radians = math.radians(center_latitude)
    cosine_latitude = math.cos(center_latitude_radians)
    if abs(cosine_latitude) < 1e-12:
        raise ValueError("The map center is too close to a geographic pole.")

    east = EARTH_RADIUS_M * math.radians(longitude - center_longitude) * cosine_latitude
    north = EARTH_RADIUS_M * math.radians(latitude - center_latitude)
    return east, north


def _from_local_meters(
    east: float,
    north: float,
    center_latitude: float,
    center_longitude: float,
) -> tuple[float, float]:
    """Convert local east/north meters back to latitude/longitude."""

    center_latitude_radians = math.radians(center_latitude)
    cosine_latitude = math.cos(center_latitude_radians)
    latitude = center_latitude + math.degrees(north / EARTH_RADIUS_M)
    longitude = center_longitude + math.degrees(
        east / (EARTH_RADIUS_M * cosine_latitude)
    )
    return latitude, _normalize_longitude(longitude)


def _scale_coordinate(
    latitude: float,
    longitude: float,
    center_latitude: float,
    center_longitude: float,
    scale_factor: float,
) -> tuple[float, float]:
    """Scale one latitude/longitude pair around the selected map center."""

    if scale_factor == 1.0:
        return latitude, longitude

    east, north = _to_local_meters(
        latitude, longitude, center_latitude, center_longitude
    )
    transformed_latitude, transformed_longitude = _from_local_meters(
        east * scale_factor,
        north * scale_factor,
        center_latitude,
        center_longitude,
    )
    if not -90.0 <= transformed_latitude <= 90.0:
        raise ValueError(
            "Scaling moved a coordinate outside the valid latitude range. "
            "Use a smaller scale or a local map near the equator."
        )
    return transformed_latitude, transformed_longitude


def _update_bounds(
    root: ET.Element,
    center_latitude: float,
    center_longitude: float,
    scale_factor: float,
) -> None:
    """Transform all existing OSM ``bounds`` elements around the map center."""

    if scale_factor == 1.0:
        return

    for element in root.iter():
        if _local_name(element.tag) != "bounds":
            continue

        try:
            min_latitude = float(element.get("minlat"))
            min_longitude = float(element.get("minlon"))
            max_latitude = float(element.get("maxlat"))
            max_longitude = float(element.get("maxlon"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Every bounds element must contain numeric minlat, minlon, "
                "maxlat, and maxlon attributes."
            ) from error
        if (
            not all(
                math.isfinite(value)
                for value in (
                    min_latitude,
                    min_longitude,
                    max_latitude,
                    max_longitude,
                )
            )
            or not -90.0 <= min_latitude <= 90.0
            or not -90.0 <= max_latitude <= 90.0
            or not -180.0 <= min_longitude <= 180.0
            or not -180.0 <= max_longitude <= 180.0
            or min_latitude > max_latitude
            or min_longitude > max_longitude
        ):
            raise ValueError(
                "Every bounds element must contain a valid coordinate range."
            )

        corners = (
            (min_latitude, min_longitude),
            (min_latitude, max_longitude),
            (max_latitude, min_longitude),
            (max_latitude, max_longitude),
        )
        transformed_corners = [
            _scale_coordinate(
                latitude,
                longitude,
                center_latitude,
                center_longitude,
                scale_factor,
            )
            for latitude, longitude in corners
        ]
        element.set(
            "minlat",
            _format_coordinate(min(latitude for latitude, _ in transformed_corners)),
        )
        element.set(
            "minlon",
            _format_coordinate(min(longitude for _, longitude in transformed_corners)),
        )
        element.set(
            "maxlat",
            _format_coordinate(max(latitude for latitude, _ in transformed_corners)),
        )
        element.set(
            "maxlon",
            _format_coordinate(max(longitude for _, longitude in transformed_corners)),
        )


def scale_osm_tree(tree: ET.ElementTree, percentage: float) -> ScaleSummary:
    """Scale all OSM node coordinates in ``tree`` around the map center.

    Args:
        tree: Parsed OSM XML tree.
        percentage: Desired map size as a percentage of the original size.

    Returns:
        A summary containing the node count and center used for the transform.

    Raises:
        ValueError: If the scale or input coordinates are invalid.
    """

    if not math.isfinite(percentage) or percentage <= 0.0:
        raise ValueError("The scale percentage must be a finite number greater than 0.")

    records = _node_records(tree.getroot())
    latitudes = [latitude for _, latitude, _ in records]
    longitudes = [longitude for _, _, longitude in records]
    center_latitude = (min(latitudes) + max(latitudes)) / 2.0
    center_longitude = (min(longitudes) + max(longitudes)) / 2.0
    scale_factor = percentage / 100.0

    summary = ScaleSummary(
        node_count=len(records),
        percentage=percentage,
        center_latitude=center_latitude,
        center_longitude=center_longitude,
    )
    if scale_factor == 1.0:
        return summary

    for element, latitude, longitude in records:
        transformed_latitude, transformed_longitude = _scale_coordinate(
            latitude,
            longitude,
            center_latitude,
            center_longitude,
            scale_factor,
        )

        element.set("lat", _format_coordinate(transformed_latitude))
        element.set("lon", _format_coordinate(transformed_longitude))

    _update_bounds(tree.getroot(), center_latitude, center_longitude, scale_factor)
    return summary


def _write_tree_atomically(tree: ET.ElementTree, output_path: Path) -> None:
    """Serialize ``tree`` and atomically replace ``output_path``."""

    output_directory = output_path.parent
    if not output_directory.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_directory}")

    output_mode = output_path.stat().st_mode if output_path.exists() else None
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_directory
    )
    os.close(temporary_fd)
    try:
        tree.write(
            temporary_name,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=True,
        )
        if output_mode is not None:
            os.chmod(temporary_name, output_mode)
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def scale_osm_file(
    input_path: str | Path,
    output_path: str | Path,
    percentage: float,
) -> ScaleSummary:
    """Read, scale, and write an OSM file."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    _register_namespaces(input_path)
    tree = _parse_tree(input_path)
    summary = scale_osm_tree(tree, percentage)
    _write_tree_atomically(tree, output_path)
    return summary


def _positive_percentage(value: str) -> float:
    """Parse a positive CLI percentage."""

    try:
        percentage = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("percentage must be numeric") from error
    if not math.isfinite(percentage) or percentage <= 0.0:
        raise argparse.ArgumentTypeError("percentage must be greater than 0")
    return percentage


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input OSM XML file")
    parser.add_argument("output", type=Path, help="Output OSM XML file")
    parser.add_argument(
        "-p",
        "--percentage",
        "--scale-percent",
        required=True,
        type=_positive_percentage,
        help="Output size relative to input, e.g. 120 for 120%% (default: none)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line OSM scaling utility."""

    args = parse_args()
    try:
        summary = scale_osm_file(args.input, args.output, args.percentage)
    except (ET.ParseError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(
        f"Scaled {summary.node_count} nodes to {summary.percentage:g}% "
        f"around center ({summary.center_latitude:.12f}, "
        f"{summary.center_longitude:.12f}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
