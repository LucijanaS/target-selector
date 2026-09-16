"""
Converts telescope pairs given as absolute (lat, lon, height) coordinates into the local
East/North/Up baseline vector that uv_track.compute_uv_track expects.

Two coordinate string formats are handled, matching what shows up in practice:
  - DMS with a cardinal suffix, e.g. "31 40 29.6N", "110 57 03.5W"
  - plain signed decimal degrees, e.g. "-24.68361537"
"""

import re
import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u


def parse_lat_lon(value):
    """Parse a lat/lon string in either 'DD MM SS.SX' or plain decimal-degree form."""
    value = value.strip()

    dms_match = re.match(r"^(\d+)\s+(\d+)\s+([\d.]+)\s*([NSEW])$", value)
    if dms_match:
        deg, minutes, seconds, hemisphere = dms_match.groups()
        decimal = float(deg) + float(minutes) / 60 + float(seconds) / 3600
        if hemisphere in ("S", "W"):
            decimal = -decimal
        return decimal

    return float(value)  # plain decimal degrees


def to_earth_location(point):
    """Convert a {'lat': ..., 'lon': ..., 'height': ...} dict to an EarthLocation."""
    lat_deg = parse_lat_lon(point["lat"])
    lon_deg = parse_lat_lon(point["lon"])
    height = point["height"]
    return EarthLocation.from_geodetic(lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=height)


def enu_baseline(point_a, point_b):
    """
    East/North/Up baseline vector [meters] from telescope A to telescope B,
    computed via the geocentric (ECEF) difference rotated into the local tangent
    plane at A's latitude/longitude.
    """
    loc_a = to_earth_location(point_a)
    loc_b = to_earth_location(point_b)

    dx = (loc_b.x - loc_a.x).to(u.m).value
    dy = (loc_b.y - loc_a.y).to(u.m).value
    dz = (loc_b.z - loc_a.z).to(u.m).value

    lat = np.radians(parse_lat_lon(point_a["lat"]))
    lon = np.radians(parse_lat_lon(point_a["lon"]))

    east = -np.sin(lon) * dx + np.cos(lon) * dy
    north = (-np.sin(lat) * np.cos(lon) * dx
             - np.sin(lat) * np.sin(lon) * dy
             + np.cos(lat) * dz)
    up = (np.cos(lat) * np.cos(lon) * dx
          + np.cos(lat) * np.sin(lon) * dy
          + np.sin(lat) * dz)

    return east, north, up


def all_pairwise_baselines(points):
    """
    East/North/Up baseline vectors for every telescope pair in an array.

    Parameters
    ----------
    points : list of {'lat': ..., 'lon': ..., 'height': ...} dicts

    Returns
    -------
    list of (i, j, x_E, x_N, x_up)
        i, j are indices into `points` (i < j); the baseline points from
        telescope i to telescope j.
    """
    baselines = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            east, north, up = enu_baseline(points[i], points[j])
            baselines.append((i, j, east, north, up))
    return baselines
