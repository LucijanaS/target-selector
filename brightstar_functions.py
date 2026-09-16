"""
Created on: 20.03.2023
Created by: Lucijana Stanic

All the functions needed in this repository are defined in this

"""
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


import json
import csv
import os
from datetime import date, datetime
import pytz
from astropy.units import brightness_temperature
from tqdm import tqdm
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from itertools import zip_longest

import numpy as np
import pandas as pd
from scipy.constants import c, h, k, pi
from scipy.special import j0, j1, jn_zeros
import scipy.signal as signal

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

lambda_U = 364 * 10 ** (-9)
lambda_V = 540 * 10 ** (-9)
lambda_B = 442 * 10 ** (-9)
# Gaia G band pivot wavelength (Gaia DR3 nominal passband, ~622 nm).
# Gaia has no Johnson U/B/V; G is the closest available broadband magnitude.
lambda_G = 622 * 10 ** (-9)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def dms_to_decimal(dms_str):
    """Converts DMS string of coordinates to a decimal float in degrees"""
    # Split the string into degrees, minutes, seconds, and direction
    degrees, minutes, seconds = map(float, dms_str[:-1].split(' '))

    # Extract direction
    direction = dms_str[-1]

    # Calculate the decimal degrees
    decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)

    # Adjust for negative values if direction is South or West
    if direction in ['S', 'W']:
        decimal_degrees *= -1

    return decimal_degrees


def calculate_covered_area(U, V):
    """Used to evaluate covered area by the track in the UV-plane"""
    # Combine U and V coordinates into a single array
    points = np.column_stack((U, V))

    # Calculate the convex hull of the points
    hull = ConvexHull(points)

    # Calculate the area of the convex hull
    area = hull.volume if len(U) > 2 else 0

    return area


def R_x(a):
    """Part of a rotation matrix, split into R_x and R_y for better overview of what is happening"""
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


def R_y(b):
    return np.array([[np.cos(b), 0, np.sin(b)],
                     [0, 1, 0],
                     [-np.sin(b), 0, np.cos(b)]])


def RA_2_HA(right_ascension, lat, lon, time):
    """Converts right ascension (in decimal degrees) to hour angle (in decimal degrees) given the observation time (
    Julian date)."""
    observing_location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    observing_time = Time(time, format='jd', scale='utc', location=observing_location)
    LST = observing_time.sidereal_time('mean').hour
    LST = (LST / 24.0) * 360
    right_ascension = (right_ascension / 24.0) * 360
    HA = (LST - right_ascension) % 360

    # Convert to h:m:s
    h = int(HA)
    m = int((HA - h) * 60)
    s = ((((HA - h) * 60) - m) * 60)

    # Adjust rounding overflow
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        h += 1
    h %= 24
    return float(HA)


def compute_uvw_track(ra_hours, dec_deg, lat_deg, lon_deg, baseline_enu, times_jd):
    """
    Project a fixed East/North/Up baseline onto the UVW plane at an array of times.

    Same physics as calling RA_2_HA()/R_x()/R_y() once per time-step and applying
    R_x(dec).R_y(HA).R_x(-lat) to the baseline vector -- but RA_2_HA rebuilds an
    EarthLocation and Time object on every call, which is fine for one star but
    becomes the dominant cost when looping over many stars and/or many time-steps
    (e.g. screening a star catalog). This builds one array-valued Time object for all
    of times_jd at once, and, since the baseline vector itself doesn't change with
    time, applies R_y(HA)/R_x(dec) as closed-form elementwise array operations
    instead of constructing/multiplying a separate 3x3 matrix per time-step.
    Verified to agree with the original per-time-step approach to ~1e-14.

    Parameters
    ----------
    ra_hours : float
        Right ascension of the star [decimal hours].
    dec_deg : float
        Declination of the star [decimal degrees].
    lat_deg, lon_deg : float
        Observer's latitude/longitude [decimal degrees].
    baseline_enu : tuple of float
        (x_E, x_N, x_up) baseline vector between the two telescopes [meters].
    times_jd : array-like of float
        Observation times [Julian Date].

    Returns
    -------
    U, V, W : ndarray
        UVW-plane coordinates at each time [meters]. W is the light-travel/delay
        axis; U, V span the plane of the sky as seen from the star.
    """
    x_E, x_N, x_up = baseline_enu

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    time_arr = Time(np.asarray(times_jd), format='jd', scale='utc', location=location)
    lst_deg = time_arr.sidereal_time('mean').deg
    ha_rad = np.radians((lst_deg - ra_hours * 15.0) % 360)

    lat_rad = np.radians(lat_deg)
    dec_rad = np.radians(dec_deg)

    # v0 = R_x(-lat) @ [x_E, x_N, x_up]
    v0x = x_E
    v0y = x_N * np.cos(-lat_rad) - x_up * np.sin(-lat_rad)
    v0z = x_N * np.sin(-lat_rad) + x_up * np.cos(-lat_rad)

    # intermediate = R_y(HA) @ v0, elementwise over the HA array
    ix = v0x * np.cos(ha_rad) + v0z * np.sin(ha_rad)
    iy = np.full_like(ha_rad, v0y)
    iz = -v0x * np.sin(ha_rad) + v0z * np.cos(ha_rad)

    # uvw = R_x(dec) @ intermediate
    U = ix
    V = iy * np.cos(dec_rad) - iz * np.sin(dec_rad)
    W = iy * np.sin(dec_rad) + iz * np.cos(dec_rad)

    return U, V, W


def find_observable_times(ra_hours, dec_deg, lat_deg, lon_deg, date_str, height_m=0,
                           min_altitude_deg=10, sun_altitude_threshold_deg=0,
                           time_resolution_minutes=5):
    """
    Find the actual times a star is observable from a given site on a given night,
    rather than assuming a fixed clock window (e.g. "8pm to 8am"). "Observable" means
    both of the following, at the same moment:
      - the sun is below sun_altitude_threshold_deg (default 0 degrees, i.e. below
        the horizon -- sunset/sunrise). Pass a stricter value (e.g. -18, the
        standard definition of astronomical twilight/full darkness) if this
        observing program needs more than just the sun being down.
      - the star's own altitude is at least min_altitude_deg (default 10, matching
        the convention already used in 2find_stars.py).

    To avoid needing to look up the site's time zone, this searches a 24-hour window
    from mean solar noon of date_str to mean solar noon of the following day, using
    the longitude-only approximation UT(local noon) = 12h - lon_deg/15 (exact to a
    few minutes -- the equation of time is the only thing left out). That window
    comfortably contains exactly one full local night (evening through morning) for
    any site, so the sun/star altitude conditions pick out that single night's
    observable stretch without pulling in a slice of the next one. (An earlier
    version used UTC noon directly, which is only close to local noon near the
    Greenwich meridian -- anywhere with several hours of UTC offset, e.g. Narrabri
    at UTC+10 or VERITAS at UTC-7, the window missed the start of the correct night
    and instead picked up the following night's early evening, showing up as a
    multi-night gap and jump in the sky-track plot.)

    Parameters
    ----------
    ra_hours : float
        Right ascension of the star [decimal hours].
    dec_deg : float
        Declination of the star [decimal degrees].
    lat_deg, lon_deg : float
        Observer's latitude/longitude [decimal degrees].
    date_str : str
        The night to search, e.g. "2026-08-17" meaning the night starting on the
        evening of that date and ending the following morning.
    height_m : float
        Observer's height above sea level [meters].
    min_altitude_deg : float
        Minimum star altitude to count as observable [degrees].
    sun_altitude_threshold_deg : float
        Sun altitude below which it counts as "night" [degrees].
    time_resolution_minutes : float
        Spacing between candidate time samples [minutes]. The returned times are
        exactly this grid, filtered down to the observable ones -- not further
        refined -- so this also sets the time resolution of whatever uses the
        result (e.g. compute_uvw_track).

    Returns
    -------
    ndarray
        Julian Dates of the observable time samples, ready to pass as times_jd to
        compute_uvw_track. Empty if the star is never observable during the night
        (e.g. always below min_altitude_deg from this site, or never dark).
    """
    # height_m may already be an astropy Quantity (several brightstar_input.py blocks
    # define height1 = ... * u.m) or a plain float; u.Quantity(x, u.m) handles both
    # instead of every caller having to remember to strip units first.
    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=u.Quantity(height_m, u.m))

    search_start = Time(date_str) + (12 - lon_deg / 15.0) * u.hour
    n_steps = int(24 * 60 / time_resolution_minutes)
    search_times = search_start + np.linspace(0, 24, n_steps) * u.hour

    altaz_frame = AltAz(obstime=search_times, location=location)

    sun_alt_deg = get_sun(search_times).transform_to(altaz_frame).alt.deg

    star_coord = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_deg * u.deg, frame='icrs')
    star_alt_deg = star_coord.transform_to(altaz_frame).alt.deg

    observable = (sun_alt_deg < sun_altitude_threshold_deg) & (star_alt_deg >= min_altitude_deg)

    return search_times.jd[observable]


def convert_ra_dec(ra_str, dec_str):
    """Converts the right ascenscion and declination string into decimal floats"""
    ra_parts = ra_str.split(' ')
    ra_h = int(ra_parts[0][:-1])
    ra_m = int(ra_parts[1][:-1])
    ra_s = float(ra_parts[2][:-1])
    ra_decimal = ra_h + ra_m / 60 + ra_s / 3600

    dec_parts = dec_str.split(' ')
    dec_d = int(dec_parts[0][:-1])
    dec_m = int(dec_parts[1][:-1])
    dec_s = float(dec_parts[2][:-1])
    if dec_d >=0:
        dec_decimal = dec_d + dec_m / 60 + dec_s / 3600
    else:
        dec_decimal = dec_d - dec_m / 60 - dec_s / 3600
    return ra_decimal, dec_decimal



def Phi(mag, wavelength):
    """Determine Phi (spectral photon flux density) as a function of magnitude and wavelength"""
    if mag is not None:
        nu = c / wavelength
        return 10 ** (-22.44 - mag / 2.5) / (2 * nu * h)
    else:
        return None

def calculate_diameter(mag, wavelength, temp):
    """Estimate the diameter of a star based on a magnitude, wavelength and effective temperature"""
    if temp is not None and mag is not None:
        nu = c / wavelength
        S = (nu ** 2 / c ** 2) / np.exp((h * nu) / (k * temp))
        area_steradian = Phi(mag, wavelength) / S

        radius_radians = np.sqrt(area_steradian / (pi))
        diameter_ = (6 / pi) * 60 ** 3 * radius_radians
        diameter = np.round(diameter_ * 10 ** 3, 2)
    else:
        diameter = None
    return diameter


def mag_from_phi(phi, wavelength=lambda_V):
    """Inverse of Phi(): recover an apparent magnitude from a spectral photon flux density."""
    nu = c / wavelength
    return -2.5 * (22.44 + np.log10(2 * nu * h * phi))


def mas_to_rad(theta_mas):
    """Convert an angle in milliarcseconds to radians."""
    return theta_mas / 1000 * pi / (3600 * 180)


def baseline_needed(theta_mas, wavelength=lambda_V):
    """Minimum interferometric baseline (m) to resolve the first visibility null for a uniform disk of angular
    diameter theta_mas (in milliarcseconds) at the given wavelength."""
    theta_rad = mas_to_rad(theta_mas)
    j1_root = jn_zeros(1, 1)[0]
    return float(j1_root / (pi * theta_rad / wavelength))


def phi_theoretical(wavelength, T, theta):
    """Theoretical blackbody spectral photon flux density Phi as a function of angular diameter theta
    (in arcseconds), temperature T (K) and wavelength (m). Inverse relation of calculate_diameter/Phi."""
    top = theta ** 2 * pi ** 3
    bottom = 36 * 60 ** 6 * wavelength ** 2 * np.exp(h * c / (wavelength * k * T))
    return top / bottom


def relmag_to_absmag(rel_magnitude, distance):
    """Converts an apparent magnitude to absolute magnitude given a distance in parsec."""
    return rel_magnitude + 5 - 5 * np.log10(distance)


def luminosity_from_absmag(absmag):
    """Bolometric-ish luminosity (L_sun) from an absolute V magnitude, calibrated against the Sun (M_V=4.74)."""
    return 10 ** (0.4 * (4.74 - absmag))


def diameter_in_solar_radii(diameter_mas, distance_pc):
    """Converts an angular diameter (mas) at a given distance (pc) into a physical diameter in solar radii."""
    solar_radius_km = 696340
    diameter_radians = (diameter_mas * 1e-3 * pi) / 648000
    physical_diameter_km = diameter_radians * distance_pc * 3.0857e13  # pc -> km
    return physical_diameter_km / solar_radius_km


def load_bsc_catalog(path):
    """Loads a Yale Bright Star Catalogue-style CSV (BayerF, Common, Parallax, Distance, Umag, Vmag, Bmag, Temp,
    RA_decimal, Dec_decimal, RA, Dec, Diameter_U, Diameter_V, Diameter_B, Phi_U, Phi_V, Phi_B) into the
    standardized schema shared with load_gaia_catalog(). RA_decimal in this catalogue is in hours; it is
    converted to degrees for consistency with the Gaia loader."""
    raw = pd.read_csv(path)

    df = pd.DataFrame({
        'name': raw['Common'].fillna(raw['BayerF']),
        'bayer_flamsteed': raw['BayerF'],
        'common_name': raw['Common'],
        'source': 'bsc',
        'band': 'V',
        'wavelength_nm': 540.0,
        'ra_deg': raw['RA_decimal'].astype(float) * 15.0,
        'dec_deg': raw['Dec_decimal'].astype(float),
        'parallax_mas': pd.to_numeric(raw['Parallax'], errors='coerce') * 1000.0,
        'distance_pc': pd.to_numeric(raw['Distance'], errors='coerce'),
        'temp_K': pd.to_numeric(raw['Temp'], errors='coerce'),
        'mag': pd.to_numeric(raw['Vmag'], errors='coerce'),
        'theta_mas': pd.to_numeric(raw['Diameter_V'], errors='coerce'),
        'theta_mas_err': np.nan,
        'phi': pd.to_numeric(raw['Phi_V'], errors='coerce'),
        'diameter_u_mas': pd.to_numeric(raw['Diameter_U'], errors='coerce'),
        'diameter_b_mas': pd.to_numeric(raw['Diameter_B'], errors='coerce'),
        'phi_u': pd.to_numeric(raw['Phi_U'], errors='coerce'),
        'phi_b': pd.to_numeric(raw['Phi_B'], errors='coerce'),
        'is_multiple': np.nan,  # not available in this catalogue
        'ruwe': np.nan,
    })
    return df


def load_gaia_catalog(path):
    """Loads a Gaia-derived CSV (as produced from a gaiadr3.gaia_source / astrophysical_parameters ADQL query,
    e.g. columns source_id, common_name, bayer_flamsteed, ra_decimal, dec_decimal, parallax, distance_gspphot,
    phot_g_mean_mag, teff_gspphot, radius_gspphot, angular_diameter_mas, ...) into the standardized schema shared
    with load_bsc_catalog().

    Gaia has no Johnson U/B/V photometry, so theta_mas here comes directly from Gaia's own radius_gspphot +
    distance_gspphot fit (more direct than a flux-reconstructed diameter), and Phi is computed from the G
    magnitude at G's ~622nm pivot wavelength using the same zero point Phi() applies to U/B/V -- Gaia's true
    G-band zero point differs slightly, so treat phi/mag round-trips for Gaia rows as approximate.
    """
    raw = pd.read_csv(path, low_memory=False)

    name = raw['common_name'].fillna(raw['bayer_flamsteed'])
    name = name.fillna('Gaia DR3 ' + raw['source_id'].astype(str))

    df = pd.DataFrame({
        'name': name,
        'bayer_flamsteed': raw['bayer_flamsteed'],
        'common_name': raw['common_name'],
        'source': 'gaia',
        'band': 'G',
        'wavelength_nm': 622.0,
        'ra_deg': pd.to_numeric(raw['ra_decimal'], errors='coerce'),
        'dec_deg': pd.to_numeric(raw['dec_decimal'], errors='coerce'),
        'parallax_mas': pd.to_numeric(raw['parallax'], errors='coerce'),
        'distance_pc': pd.to_numeric(raw['distance_gspphot'], errors='coerce'),
        'temp_K': pd.to_numeric(raw['teff_gspphot'], errors='coerce'),
        'mag': pd.to_numeric(raw['phot_g_mean_mag'], errors='coerce'),
        'theta_mas': pd.to_numeric(raw['angular_diameter_mas'], errors='coerce'),
        'theta_mas_err': pd.to_numeric(raw['angular_diameter_mas_err'], errors='coerce'),
        'phi': Phi(pd.to_numeric(raw['phot_g_mean_mag'], errors='coerce'), lambda_G),
        'diameter_u_mas': np.nan,
        'diameter_b_mas': np.nan,
        'phi_u': np.nan,
        'phi_b': np.nan,
        'is_multiple': pd.to_numeric(raw['non_single_star'], errors='coerce') != 0,
        'ruwe': pd.to_numeric(raw['ruwe'], errors='coerce'),
    })
    return df


def load_star_catalog(path):
    """Auto-detects whether `path` is a BSC-style, Gaia-style, or already-standardized (e.g. saved by
    combine_catalogs) star CSV by header, and loads it into the standardized schema (see load_bsc_catalog /
    load_gaia_catalog). This is the single entry point notebooks/scripts should use so plotting code doesn't
    care which catalogue the data came from."""
    header = pd.read_csv(path, nrows=0).columns
    if 'source_id' in header:
        return load_gaia_catalog(path)
    elif 'BayerF' in header:
        return load_bsc_catalog(path)
    elif 'theta_mas' in header and 'source' in header:
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(
            f"Could not auto-detect catalogue schema for {path} "
            f"(no 'source_id', 'BayerF', or standardized 'theta_mas'/'source' columns found)"
        )


def _normalize_name(s):
    """Lowercase, whitespace-collapsed key for fuzzy-ish name matching."""
    if pd.isna(s):
        return None
    return ' '.join(str(s).split()).casefold()


def load_observation_log(catalog, path='observation_log.csv'):
    """Merges a manually-maintained SII observation log onto a standardized catalog (from load_star_catalog).

    This tracks a fundamentally different thing than the `is_multiple` column: `is_multiple` is Gaia's own
    astrometric/spectroscopic multiplicity flag (is the star physically a binary/triple), while this tracks
    observation history (has *this* star already been measured via intensity interferometry, and with which
    instrument). Neither dataset carries the latter -- it has to be maintained separately.

    The log file is a simple long-format CSV: one row per (star, instrument) observation, e.g.:
        name,instrument,date,notes
        Sirius,HBT,1956,Narrabri -- the original Hanbury Brown Twiss measurement

    `name` is matched case/whitespace-insensitively against the catalog's `name`, `common_name`, and
    `bayer_flamsteed` columns. Unmatched log rows are reported, not silently dropped, so typos surface.

    Adds one boolean column per distinct `instrument` value found (e.g. `observed_hbt`, `observed_magic`,
    `observed_veritas`), plus `sii_observed` (True if observed by any instrument). If `path` doesn't exist yet,
    returns the catalog unchanged with all those columns set to False, so the rest of a pipeline can run before
    the log has been filled in.
    """
    instrument_cols = []
    catalog = catalog.copy()

    if not os.path.exists(path):
        print(f"load_observation_log: '{path}' not found -- returning catalog with no SII flags set. "
              f"Create it (columns: name,instrument,date,notes) to start tracking observations.")
        catalog['sii_observed'] = False
        return catalog

    log = pd.read_csv(path)
    log['_key'] = log['name'].map(_normalize_name)

    name_to_idx = {}
    for col in ('name', 'common_name', 'bayer_flamsteed'):
        if col in catalog.columns:
            for idx, val in catalog[col].map(_normalize_name).items():
                if val is not None and val not in name_to_idx:
                    name_to_idx[val] = idx

    unmatched = []
    for instrument, group in log.groupby(log['instrument'].str.strip().str.upper()):
        col = f'observed_{instrument.lower()}'
        instrument_cols.append(col)
        catalog[col] = False
        for key in group['_key']:
            idx = name_to_idx.get(key)
            if idx is None:
                unmatched.append((key, instrument))
            else:
                catalog.loc[idx, col] = True

    if unmatched:
        print(f"load_observation_log: {len(unmatched)} log entries had no matching star in the catalog:")
        for key, instrument in unmatched:
            print(f"  - {key!r} ({instrument})")

    catalog['sii_observed'] = catalog[instrument_cols].any(axis=1) if instrument_cols else False
    return catalog


def combine_catalogs(bsc_df, gaia_df, tolerance_arcsec=10.0):
    """Combines a BSC-style and a Gaia-style standardized catalog (from load_bsc_catalog/load_gaia_catalog)
    into one, keeping every BSC row and adding only the Gaia rows that don't already correspond to a BSC star.

    Duplicates are found by sky position (RA/Dec), not name matching -- name strings differ too much in
    formatting between the two catalogues (e.g. BSC's "alf01 Centauri" vs Gaia/SIMBAD's "alf01 Cen") to match
    reliably, whereas position is unambiguous. Any Gaia row within `tolerance_arcsec` of a BSC row is treated as
    the same physical star and dropped in favor of the BSC row, since BSC carries real Johnson U/B/V photometry
    that Gaia doesn't have (see load_gaia_catalog's docstring on the G-band approximation).

    Returns
    -------
    combined : pd.DataFrame
        BSC rows unchanged, plus non-duplicate Gaia rows, in the standardized schema.
    n_duplicates : int
        Number of Gaia rows that were dropped as duplicates of a BSC row.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    bsc_valid = bsc_df.dropna(subset=['ra_deg', 'dec_deg'])
    gaia_valid = gaia_df.dropna(subset=['ra_deg', 'dec_deg'])

    if len(bsc_valid) == 0 or len(gaia_valid) == 0:
        return pd.concat([bsc_df, gaia_df], ignore_index=True), 0

    bsc_coords = SkyCoord(ra=bsc_valid['ra_deg'].to_numpy() * u.deg,
                           dec=bsc_valid['dec_deg'].to_numpy() * u.deg)
    gaia_coords = SkyCoord(ra=gaia_valid['ra_deg'].to_numpy() * u.deg,
                            dec=gaia_valid['dec_deg'].to_numpy() * u.deg)

    _, sep2d, _ = gaia_coords.match_to_catalog_sky(bsc_coords)
    is_duplicate = sep2d < (tolerance_arcsec * u.arcsec)

    gaia_unique = gaia_valid.loc[~is_duplicate]
    combined = pd.concat([bsc_df, gaia_unique], ignore_index=True)

    return combined, int(is_duplicate.sum())


def process_star(star):
    """Extracts the values and parameters needed from the catalogue"""
    BayerF = star.get("BayerF")
    common = star.get("Common")

    parallax_value = star.get("Parallax")
    parallax = float(parallax_value) if parallax_value is not None else None
    distance = round(1 / parallax, 3) if parallax_value and abs(parallax) > 0 else None
    Vmag = round(float(star.get("Vmag")), 3)

    BV_value = star.get("B-V")
    BV = float(BV_value) if BV_value is not None else None
    Bmag = round(BV + Vmag, 3) if BV is not None else None

    UB_value = star.get("U-B")
    UB = float(UB_value) if UB_value is not None and Bmag is not None else None
    Umag = round(UB + Bmag, 3) if UB is not None else None

    temp = round(float(star.get("K")), 3) if star.get("K") is not None else None

    star_ra_decimal, star_dec_decimal = convert_ra_dec(star["RA"], star["Dec"])
    star_ra_decimal = round(star_ra_decimal, 3)
    star_dec_decimal = round(star_dec_decimal, 3)

    diameter_U = calculate_diameter(Umag, lambda_U, temp)
    diameter_V = calculate_diameter(Vmag, lambda_V, temp)
    diameter_B = calculate_diameter(Bmag, lambda_B, temp)

    Phi_V = Phi(Vmag, lambda_V)
    Phi_B = Phi(Bmag, lambda_B)
    Phi_U = Phi(Umag, lambda_U)

    return {
        "BayerF": BayerF,
        "Common": common,
        "Parallax": parallax,
        "Distance": distance,
        "Umag": Umag,
        "Vmag": Vmag,
        "Bmag": Bmag,
        "Temp": temp,
        "RA_decimal": star_ra_decimal,
        "Dec_decimal": star_dec_decimal,
        "RA": star.get("RA"),
        "Dec": star.get("Dec"),
        "Diameter_U": diameter_U,
        "Diameter_V": diameter_V,
        "Diameter_B": diameter_B,
        "Phi_U":  Phi_U,
        "Phi_V":  Phi_V,
        "Phi_B": Phi_B
    }


def visibility(b, theta, lambda_):
    """The squared visibility, often denoted in papers as |V_12|^2 and equals g**(2)-1"""
    # b may be a scalar or an array (e.g. a whole UVW-plane grid); np.where handles
    # both, unlike a plain "if b == 0" which is ambiguous for arrays with more than
    # one element.
    b = np.asarray(b, dtype=float)
    safe_b = np.where(b == 0, 1e-20, b)
    input = np.pi * safe_b * theta / lambda_
    I = (2 * j1(input) / input) ** 2
    return np.where(b == 0, 1.0, I)

def visibility_binaries(u, v, theta_1, theta_2, x_1, y_1, x_2, y_2, lambda_, brightness_ratio = 1):
    """The squared visibility, often denoted in papers as |V_12|^2 and equals g**(2)-1"""
    input_1 = np.pi * np.sqrt(u**2+v**2) * theta_1 / lambda_
    V_ud_1 = (2 * j1(input_1) / input_1)

    input_2 = np.pi * np.sqrt(u**2+v**2) * theta_2 / lambda_
    V_ud_2 = (2 * j1(input_2) / input_2)
    #print("input 2:", input_2)


    exp_1 = brightness_ratio*np.exp(((2*np.pi*1j)/lambda_)*(u*x_1+v*y_1))
    exp_2 = np.exp(((2*np.pi*1j)/lambda_)*(u*x_2+v*y_2))

    V_binaries = V_ud_1*exp_1 + V_ud_2*exp_2

    V_2 = abs(V_binaries)**2

    V_2 = (1+brightness_ratio)**(-2) * V_2

    return V_2


def visibility_stars(u, v, thetas, xs, ys, lambda_, brightnesses=None):
    """
    Squared visibility for an arbitrary number of stars.

    Parameters
    ----------
    u, v : 2D arrays
        UV-plane coordinates [meters].
    thetas : array-like of float
        Angular diameters [radians] for each star.
    xs, ys : array-like of float
        Sky positions [radians] for each star (east, south).
    lambda_ : float
        Wavelength [meters].
    brightnesses : array-like of float, optional
        Relative fluxes for each star (needn't be normalized).
        If None, all ones.

    Returns
    -------
    V2 : 2D array
        Squared visibility |V|^2 on the provided UV grid.
    """
    thetas = np.asarray(thetas, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    N = len(thetas)
    if brightnesses is None:
        brightnesses = np.ones(N, dtype=float)
    else:
        brightnesses = np.asarray(brightnesses, dtype=float)

    # Complex coherent sum
    V_tot = 0.0 + 0.0j

    # Precompute baseline length on UV plane (meters)
    rho = np.sqrt(u**2 + v**2)

    for i in range(N):
        x_i, y_i = xs[i], ys[i]
        th_i = thetas[i]
        b_i = brightnesses[i]

        # Uniform-disk visibility, robust at input=0
        arg = np.pi * rho * th_i / lambda_
        V_ud = np.ones_like(arg)
        nz = arg != 0
        V_ud[nz] = (2.0 * j1(arg[nz]) / arg[nz])

        phase = np.exp((2j * np.pi / lambda_) * (u * x_i + v * y_i))
        V_tot += b_i * V_ud * phase

    V_tot /= np.sum(brightnesses)
    return np.abs(V_tot)**2


def triple_positions_radians(jd,
                             orbit_inner_AaAb, a_inner_rad,
                             orbit_outer_A_B,  a_outer_rad):
    """
    Positions of Aa, Ab, B on the sky (east, south) in radians at one JD.

    Parameters
    ----------
    jd : float
        Julian Date.
    orbit_inner_AaAb : Orbit
        Orbit object for the inner Aa-Ab pair (use q = M_Ab / M_Aa).
    a_inner_rad : float
        Angular semimajor axis of the *relative* inner orbit [radians].
        (i.e., Aa-Ab separation's a on the sky)
    orbit_outer_A_B : Orbit
        Orbit object for the outer A_barycenter - B orbit (use q = M_B / M_A_total).
    a_outer_rad : float
        Angular semimajor axis of the *relative* outer orbit [radians].
        (i.e., A_bary - B separation's a on the sky)

    Returns
    -------
    xs, ys : list of floats
        [x_Aa, x_Ab, x_B], [y_Aa, y_Ab, y_B] in radians.
    """
    # Outer: A_bary (index 0) and B (index 1)
    x_out, y_out = orbit_outer_A_B.binarypos(jd)   # dimensionless
    xA_bary = x_out[0] * a_outer_rad
    yA_bary = y_out[0] * a_outer_rad
    xB      = x_out[1] * a_outer_rad
    yB      = y_out[1] * a_outer_rad

    # Inner: Aa (index 0) and Ab (index 1), relative to A_bary
    x_in, y_in = orbit_inner_AaAb.binarypos(jd)    # dimensionless
    xAa_rel = x_in[0] * a_inner_rad
    yAa_rel = y_in[0] * a_inner_rad
    xAb_rel = x_in[1] * a_inner_rad
    yAb_rel = y_in[1] * a_inner_rad

    # Absolute sky positions (east, south) in radians
    xAa = xA_bary + xAa_rel
    yAa = yA_bary + yAa_rel
    xAb = xA_bary + xAb_rel
    yAb = yA_bary + yAb_rel

    return [xAa, xAb, xB], [yAa, yAb, yB]


def quadruple_positions_radians(jd,
                                orbit_inner_AaAb, a_innerA_rad,
                                orbit_inner_BaBb, a_innerB_rad,
                                orbit_outer_A_B,  a_outer_rad):
    """
    Positions of Aa, Ab, Ba, Bb on the sky (east, south) in radians at one JD.

    Returns
    -------
    xs, ys : list of floats
        [x_Aa, x_Ab, x_Ba, x_Bb], [y_Aa, y_Ab, y_Ba, y_Bb]
    """
    # Outer: A_bary (index 0), B_bary (index 1)
    x_out, y_out = orbit_outer_A_B.binarypos(jd)
    xA_bary = x_out[0] * a_outer_rad
    yA_bary = y_out[0] * a_outer_rad
    xB_bary = x_out[1] * a_outer_rad
    yB_bary = y_out[1] * a_outer_rad

    # Inner A: Aa (0), Ab (1) relative to A_bary
    xA_in, yA_in = orbit_inner_AaAb.binarypos(jd)
    xAa = xA_bary + xA_in[0] * a_innerA_rad
    yAa = yA_bary + yA_in[0] * a_innerA_rad
    xAb = xA_bary + xA_in[1] * a_innerA_rad
    yAb = yA_bary + yA_in[1] * a_innerA_rad

    # Inner B: Ba (0), Bb (1) relative to B_bary
    xB_in, yB_in = orbit_inner_BaBb.binarypos(jd)
    xBa = xB_bary + xB_in[0] * a_innerB_rad
    yBa = yB_bary + yB_in[0] * a_innerB_rad
    xBb = xB_bary + xB_in[1] * a_innerB_rad
    yBb = yB_bary + yB_in[1] * a_innerB_rad

    return [xAa, xAb, xBa, xBb], [yAa, yAb, yBa, yBb]

def apparent_size_in_mas(size, distance):
    """
    Calculates the apparent angular size of an object in milliarcseconds (mas).

    Args:
        size: Physical size of the object in solar radii.
        distance: Distance to the object in parsecs.

    Returns:
        Apparent angular size in milliarcseconds (mas).
    """
    # Convert size to meters (1 solar radius = 695700 km = 6.957e8 meters)
    size_meters = size * 6.957e8

    # Convert distance to meters (1 parsec = 3.085677581e16 meters)
    distance_meters = distance * 3.085677581e16

    # Angular size in radians (using small-angle approximation)
    angular_size_rad = size_meters / distance_meters

    # Convert radians to milliarcseconds (1 rad = 206265 mas)
    angular_size_mas = angular_size_rad * 206265e3  # or angular_size_rad * 206264.806

    return angular_size_mas

def mag_to_flux(mags):
    """Return *relative* fluxes normalized to sum=1 from V magnitudes (F ~ 10^-0.4m)."""
    F = np.array([10.0**(-0.4*m) for m in mags], dtype=float)
    return F / F.sum()
