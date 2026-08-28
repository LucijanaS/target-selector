import json
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import streamlit as st
from timezonefinder import TimezoneFinder

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

# Explicit imports (not `from brightstar_functions import *`) so a stale copy of
# brightstar_functions.py fails loudly here at startup instead of as a NameError deep
# in a callback. These must be the versions from the brightstar repo -- see README.
from brightstar_functions import (
    process_star, find_observable_times, compute_uvw_track, visibility,
    dms_to_decimal, lambda_U, lambda_V, lambda_B,
)
from telescopes import (
    TELESCOPE_PRESETS, CUSTOM, preset_has_dishes, preset_site,
    pairwise_baselines, latlon_to_enu,
)
from telescope_coords import parse_lat_lon

# - * - coding: utf - 8 - * -

# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

# lambda_U / lambda_V / lambda_B come from brightstar_functions.
BANDS = {"V": lambda_V, "B": lambda_B, "U": lambda_U}

n_brightest_stars = 10000

# High-contrast colours for the UV tracks -- the visibility map is near-black away from
# the centre, so tab10's darker entries vanish on it.
TRACK_COLOURS = ["#ffd400", "#00e5ff", "#ff4dd2", "#7CFC00", "#ff8c00",
                 "#ffffff", "#00ff9c", "#ff5555"]


@st.cache_resource
def _timezone_finder():
    return TimezoneFinder()


def _fmt_offset(hours):
    sign = "+" if hours >= 0 else "-"
    hours = abs(hours)
    return f"{sign}{int(hours):02d}:{int(round((hours - int(hours)) * 60)):02d}"


@st.cache_data(show_spinner=False)
def site_utc_offset(lat, lon, date):
    """(offset_hours, label) for the site on `date`, from its IANA time zone (DST-aware);
    falls back to lon/15 if the zone can't be resolved (e.g. mid-ocean)."""
    try:
        name = _timezone_finder().timezone_at(lat=float(lat), lng=float(lon))
        if name:
            off = pytz.timezone(name).utcoffset(datetime.datetime.combine(date, datetime.time(0, 0)))
            hours = off.total_seconds() / 3600.0
            return hours, f"{name}, UTC{_fmt_offset(hours)}"
    except Exception:
        pass
    hours = float(round(float(lon) / 15.0))
    return hours, f"UTC{_fmt_offset(hours)} (estimated from longitude)"


# ---------------------------------------------------------------------------------------------------------------------------------------
# UV-coverage quality -- being above the horizon is necessary but not sufficient. To measure a
# star's angular diameter you want the night's UV track to sit on the *steep* part of the
# visibility curve: the descent of the first lobe, roughly x = pi*rho*theta/lambda in [1, 3.83].
# There the slope d|V|^2/d(rho) is large, so |V|^2 pins theta tightly. The flat top (x < ~1)
# barely constrains it, and the low side-lobes past the first null add little for a diameter.
# This scores that with geometry only -- no photon-noise model, not a limb-darkening question.
# ---------------------------------------------------------------------------------------------------------------------------------------

X_RESOLVED = 1.0            # x below this -> |V|^2 within a few % of 1, star barely resolved
X_FIRST_NULL = 3.8317059    # first zero of J1 -> |V|^2 = 0; right edge of the useful descent

# Curve slope |d|V|^2 / d ln x| on a fixed x grid, normalised so the first-lobe descent peaks
# near 1. This single weight already encodes "flat top ~ 0, first-lobe descent ~ 1, side-lobes
# ~ 0.03", so a point's weight is just how much diameter leverage sampling there gives.
_X_GRID = np.linspace(1e-3, 9.0, 4000)
_SLOPE_GRID = np.abs(np.gradient(visibility(_X_GRID, 1.0, np.pi), np.log(_X_GRID)))
_SLOPE_WEIGHT = np.clip(_SLOPE_GRID / _SLOPE_GRID[_X_GRID < X_FIRST_NULL].max(), 0.0, 1.0)


def coverage_score(rho_m, theta_rad, lambda_m):
    """0-100 score (+ diagnostics) for how well traced projected-baseline lengths rho [m] sit on
    the steep first-lobe descent of a star's visibility curve, given theta [rad], lambda [m].
    Score = 100 * (mean curve-slope weight over the traced points)."""
    rho_m = np.asarray(rho_m, dtype=float)
    rho_m = rho_m[np.isfinite(rho_m) & (rho_m > 0)]
    if rho_m.size == 0 or not np.isfinite(theta_rad) or theta_rad <= 0:
        return dict(score=0.0, x_min=0.0, x_max=0.0, rho_max=0.0, frac_on_slope=0.0)

    x = np.pi * rho_m * theta_rad / lambda_m
    w = np.interp(x, _X_GRID, _SLOPE_WEIGHT)
    return dict(
        score=100.0 * float(np.clip(w.mean(), 0.0, 1.0)),
        x_min=float(x.min()), x_max=float(x.max()), rho_max=float(rho_m.max()),
        frac_on_slope=float(np.mean(w >= 0.33)),   # fraction of the night usefully on the descent
    )


def score_verdict(d):
    if d["x_max"] < X_RESOLVED:
        return "still on the flat top of the curve — star barely resolved by this array tonight"
    if d["x_min"] > X_FIRST_NULL:
        return "over-resolved — track sits past the first null, in the low side-lobes"
    if d["score"] >= 55:
        return "track sits on the steep first-lobe descent — good diameter leverage"
    if d["score"] >= 25:
        return "track partly samples the first-lobe descent"
    return "track mostly on the flat part of the curve — weak diameter leverage"


# ---------------------------------------------------------------------------------------------------------------------------------------
# Catalogue loading -- built once and cached, so changing a sidebar widget does not
# re-parse bsc5-all.json and re-run process_star for every star on every rerun. Nothing
# is written to disk (works the same on Streamlit Community Cloud and locally).
# ---------------------------------------------------------------------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading the bright-star catalogue…")
def load_bright_stars(n):
    with open('bsc5-all.json', 'r') as file:
        stars_data = json.load(file)

    # Keep only stars with a Bayer/Flamsteed designation or a common name.
    stars_data = [
        star for star in stars_data
        if (star.get('BayerF') not in ('', None)) or (star.get('Common') not in ('', None))
    ]
    brightest = sorted(stars_data, key=lambda x: float(x['Vmag']))[:n]

    df = pd.DataFrame(process_star(star) for star in brightest)
    df['Identifier'] = df.apply(
        lambda x: f"{x['BayerF']} ({x['Common']})" if pd.notna(x['Common']) else x['BayerF'],
        axis=1,
    )
    return df


df_all_stars = load_bright_stars(n_brightest_stars)
# Columns shown in the "stars to run through" preview (drop the diameter/Phi columns).
display_cols = ["BayerF", "Common", "Parallax", "Distance", "Umag", "Vmag", "Bmag",
                "Temp", "RA_decimal", "Dec_decimal", "RA", "Dec"]
df_display = df_all_stars[display_cols]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Visibility search -- vectorised over time (one AltAz transform per star, plus one for
# the Sun), instead of the old per-star / per-timestep Python loop.
# ---------------------------------------------------------------------------------------------------------------------------------------


@st.cache_data(show_spinner="Searching for stars visible that night…")
def search_visible_stars(ra_hours, dec_deg, lat, lon, height_m, date_str,
                         min_altitude_deg, sun_altitude_deg=0.0,
                         time_resolution_minutes=5):
    """
    Indices of the stars (into the ra_hours/dec_deg tuples) that are observable -- Sun
    below sun_altitude_deg AND star above min_altitude_deg -- for at least 3/4 of that
    night's real dark window at the given site/date. Single-star equivalent:
    brightstar_functions.find_observable_times().

    Returns (kept_indices, dark_window_hours).
    """
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height_m * u.m)
    start = Time(date_str) + 12 * u.hour
    n_steps = int(24 * 60 / time_resolution_minutes)
    times = start + np.linspace(0, 24, n_steps) * u.hour
    frame = AltAz(obstime=times, location=location)

    sun_down = get_sun(times).transform_to(frame).alt.deg < sun_altitude_deg
    n_dark = int(np.count_nonzero(sun_down))
    dark_hours = n_dark * time_resolution_minutes / 60.0
    if n_dark == 0:
        return [], 0.0

    kept = []
    for i, (ra_h, dec_d) in enumerate(zip(ra_hours, dec_deg)):
        star = SkyCoord(ra=float(ra_h) * u.hourangle, dec=float(dec_d) * u.deg, frame='icrs')
        star_alt = star.transform_to(frame).alt.deg
        observable = int(np.count_nonzero(sun_down & (star_alt >= min_altitude_deg)))
        if observable >= 0.75 * n_dark:
            kept.append(i)
    return kept, dark_hours


@st.cache_data(show_spinner="Scoring UV coverage…")
def score_candidates(ra_hours, dec_deg, theta_mas, lat, lon, height_m, date_str,
                     min_altitude_deg, baselines_t, lambda_m):
    """coverage_score() dict per star, using each star's own observable window and the UV
    tracks of every array baseline. `baselines_t` is a tuple of (label, (E, N, Up))."""
    out = []
    for ra_h, dec_d, th_mas in zip(ra_hours, dec_deg, theta_mas):
        tj = find_observable_times(ra_h, dec_d, lat, lon, date_str,
                                   height_m=height_m, min_altitude_deg=min_altitude_deg)
        if len(tj) < 3 or not baselines_t:
            out.append(None)
            continue
        rho = np.concatenate([
            np.hypot(*compute_uvw_track(ra_h, dec_d, lat, lon, enu, tj)[:2])
            for _, enu in baselines_t
        ])
        theta_rad = float(th_mas) / 1000 * np.pi / (3600 * 180)
        out.append(coverage_score(rho, theta_rad, lambda_m))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


st.markdown(
    """
    ## Identifying SII Candidates by Location and Date
    On this webpage, you can find stars that are ideal candidates for stellar intensity interferometry. \n

    By entering your location and date (or picking a telescope coupling from the presets), you can either
    view a list of stars observable from that site on that night, or select a specific star from the
    dropdown menu to track its path across the sky and explore its visibility map."""
)

st.markdown(
    """
    Similarly to the Target Stars WebApp (https://target-stars-sii.streamlit.app/), the stars used are from the
    Yale Bright Star Catalog, which contains 9110 of the brightest stars (http://tdc-www.harvard.edu/catalogs/bsc5.html).
    The catalogue is in ASCII format and was converted to JSON in https://github.com/brettonw/YaleBrightStarCatalog
    (the file used here is `bsc5-all.json`). \n
    You choose how many of the brightest stars the visibility search runs through, so you can trade completeness
    for speed.
    """
)

st.markdown(
    """
    ### Enter parameters
    It is best to enter all the parameters you have first -- the site coordinates (or a telescope preset),
    the baseline, the date and the observing band -- in the sidebar on the left. Every change in the sidebar
    reloads the plots and tables.
    """
)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.markdown("## Observation setup")

date = st.sidebar.date_input("Date of observation", datetime.date.today())
date_str = date.isoformat()

preset_name = st.sidebar.selectbox(
    "Telescope coupling",
    list(TELESCOPE_PRESETS),
    help="Pick a known site + dish pair to auto-fill the coordinates and baseline, or "
         "'Custom' to enter everything by hand.",
)
preset = TELESCOPE_PRESETS[preset_name]
is_custom = preset_name == CUSTOM

if not is_custom:
    st.sidebar.caption(f"ℹ️ {preset['note']}")
    if preset['approx']:
        st.sidebar.caption("⚠️ Site coordinates are approximate — verify before precise UV work.")

band = st.sidebar.radio(
    "Observing band",
    list(BANDS),
    help="Wavelength band used for the angular diameter, Φ and the visibility map. "
         "U/B are only available for stars that have the corresponding colour index.",
)
lambda_sel = BANDS[band]

# --- Site + baselines ---------------------------------------------------------------
# Produces:  lat_dec1, lon_dec1, height1  -- observer site, for the observability window
#            baselines : list of (label, (x_E, x_N, x_up))  -- one entry per dish pair
#            utc_offset / tz_label       -- derived from the site, for local-time axes
lat_dec2 = lon_dec2 = None
height1 = 0.0
height2 = 0.0
two_telescopes = "Yes"
baselines = []

site_lat, site_lon, site_height = preset_site(preset)

if preset_has_dishes(preset):
    # Fully determined by the dish coordinates -- nothing to enter here.
    lat_dec1, lon_dec1, height1 = float(site_lat), float(site_lon), float(site_height)
    baselines = pairwise_baselines(preset["dishes"])
    st.sidebar.caption(
        f"Site: {lat_dec1:.4f}°, {lon_dec1:.4f}°, {height1:.0f} m — "
        f"{len(baselines)} baseline{'s' if len(baselines) != 1 else ''} "
        f"({', '.join(lbl for lbl, _ in baselines)})."
    )
else:
    two_telescopes = st.sidebar.radio(
        "Two telescopes?",
        ["Yes", "No"],
        help="Two telescopes (with at least a baseline) are needed for the visibility map / "
             "UV trace. With one, you can still search for stars visible from your site.",
    )
    loc_mode = st.sidebar.radio("Enter second telescope as", ["baseline", "coordinates"]) \
        if two_telescopes == "Yes" else None

    if is_custom:
        coordinates_form = st.sidebar.radio(
            "Enter coordinates in:",
            ["decimal degrees (DD)", "degrees, minute, second (DMS)"],
        )
        if coordinates_form == "degrees, minute, second (DMS)":
            d1 = st.sidebar.number_input("Latitude degree:", value=23.0, format="%.0f")
            m1 = st.sidebar.number_input("Latitude minute:", value=20.0, format="%.0f")
            s1 = st.sidebar.number_input("Latitude second:", value=31.9, format="%.1f")
            ns1 = st.sidebar.selectbox("North or South:", ("N", "S"))
            dl1 = st.sidebar.number_input("Longitude degree:", value=16.0, format="%.0f")
            ml1 = st.sidebar.number_input("Longitude minute:", value=13.0, format="%.0f")
            sl1 = st.sidebar.number_input("Longitude second:", value=29.7, format="%.1f")
            we1 = st.sidebar.selectbox("West or East:", ("W", "E"))
            lat_dec1 = dms_to_decimal(f"{d1} {m1} {s1}{ns1}")
            lon_dec1 = dms_to_decimal(f"{dl1} {ml1} {sl1}{we1}")

            if two_telescopes == "Yes" and loc_mode == "coordinates":
                d2 = st.sidebar.number_input("2nd telescope latitude degree:", value=23.0, format="%.0f")
                m2 = st.sidebar.number_input("2nd telescope latitude minute:", value=20.0, format="%.0f")
                s2 = st.sidebar.number_input("2nd telescope latitude second:", value=29.7, format="%.1f")
                ns2 = st.sidebar.selectbox("North or South: ", ("N", "S"))
                dl2 = st.sidebar.number_input("2nd telescope longitude degree:", value=16.0, format="%.0f")
                ml2 = st.sidebar.number_input("2nd telescope longitude minute:", value=13.0, format="%.0f")
                sl2 = st.sidebar.number_input("2nd telescope longitude second:", value=28.1, format="%.1f")
                we2 = st.sidebar.selectbox("West or East ", ("W", "E"))
                lat_dec2 = dms_to_decimal(f"{d2} {m2} {s2}{ns2}")
                lon_dec2 = dms_to_decimal(f"{dl2} {ml2} {sl2}{we2}")
        else:
            lat_dec1 = st.sidebar.number_input("Latitude [deg]:", value=-23.34220, format="%.5f")
            lon_dec1 = st.sidebar.number_input("Longitude [deg]:", value=16.22494, format="%.5f")
            if two_telescopes == "Yes" and loc_mode == "coordinates":
                lat_dec2 = st.sidebar.number_input("2nd telescope latitude [deg]:", value=-23.34157, format="%.5f")
                lon_dec2 = st.sidebar.number_input("2nd telescope longitude [deg]:", value=16.22447, format="%.5f")
    else:
        # Placeholder-site preset (H.E.S.S. / CTAO-North): site fixed, 2nd telescope manual.
        lat_dec1 = st.sidebar.number_input("Site latitude [deg]:", value=float(site_lat), format="%.5f")
        lon_dec1 = st.sidebar.number_input("Site longitude [deg]:", value=float(site_lon), format="%.5f")
        if two_telescopes == "Yes" and loc_mode == "coordinates":
            lat_dec2 = st.sidebar.number_input("2nd telescope latitude [deg]:", value=float(site_lat), format="%.5f")
            lon_dec2 = st.sidebar.number_input("2nd telescope longitude [deg]:", value=float(site_lon), format="%.5f")

    if two_telescopes == "Yes":
        if st.sidebar.checkbox(
            "Enter heights above sea level",
            help="Matters when the height difference between telescopes is comparable to the baseline.",
        ):
            height1 = st.sidebar.number_input("Height of first telescope [m]:", value=0.0, format="%.1f")
            height2 = st.sidebar.number_input("Height of second telescope [m]:", value=0.0, format="%.1f")

    if two_telescopes == "Yes" and loc_mode == "baseline":
        b_len = st.sidebar.number_input(
            "Baseline length [m]:", value=100.0, format="%.1f",
            help="N–S / E–W separation of the two telescopes (excluding the height difference).",
        )
        b_ang = st.sidebar.number_input(
            "Baseline orientation [deg]:", min_value=0, max_value=359, value=90,
            help="0° → x_N = baseline, x_E = 0;  90° → x_N = 0, x_E = baseline.",
        )
        if b_len > 0:
            baselines = [("baseline", (float(np.sin(np.radians(b_ang)) * b_len),
                                       float(np.cos(np.radians(b_ang)) * b_len),
                                       float(height2 - height1)))]
    elif two_telescopes == "Yes" and loc_mode == "coordinates" and lat_dec2 is not None:
        baselines = [("T1–T2", latlon_to_enu(lat_dec1, lon_dec1, height1,
                                             lat_dec2, lon_dec2, height2))]

# Local-time offset for the plot axes, derived from the site (no manual entry).
utc_offset, tz_label = site_utc_offset(lat_dec1, lon_dec1, date)
st.sidebar.caption(f"🕑 Local time on the plots: {tz_label}")

min_altitude_deg = st.sidebar.slider(
    "Minimum star altitude [deg]:", min_value=0, max_value=60, value=10,
    help="A star counts as observable only while it is above this altitude and the Sun is down.",
)

number_of_stars = st.sidebar.number_input(
    "Number of brightest stars to check:", min_value=1, max_value=n_brightest_stars, value=50,
    help="The visibility search runs through this many of the brightest stars. 50–100 is a good "
         "starting point.",
)

show_map = st.sidebar.checkbox(
    "Show map of telescope location(s)",
    help="Has no effect on the calculation; only a cross-check of the coordinates you entered.",
)

if show_map:
    st.write("Map of the telescope location(s):")
    if preset_has_dishes(preset):
        map_data = pd.DataFrame([
            {'lat': parse_lat_lon(d['lat']), 'lon': parse_lat_lon(d['lon'])}
            for d in preset['dishes']
        ])
    else:
        map_data = pd.DataFrame({'lat': [lat_dec1], 'lon': [lon_dec1]})
        if lat_dec2 is not None:
            map_data = pd.concat(
                [map_data, pd.DataFrame({'lat': [lat_dec2], 'lon': [lon_dec2]})], ignore_index=True
            )
    st.map(map_data, size=1, zoom=13)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Stars to run through
# ---------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## Candidate stars for the night of " + date_str)
st.write(
    "The table below lists the brightest " + str(int(number_of_stars)) + " stars the search runs "
    "through. Press **Search** to keep those that stay above " + str(min_altitude_deg) +
    "° while the Sun is down for at least 3/4 of the night" +
    (", ranked by how well tonight's UV track covers the informative part of each star's "
     "visibility curve (**coverage** 0–100)." if baselines else ".")
)
st.dataframe(df_display.head(int(number_of_stars)))

run = df_all_stars.head(int(number_of_stars)).reset_index(drop=True)

with st.form("search_visible"):
    st.write("Search for stars observable on the night of ", date_str, " from this site.")
    do_search = st.form_submit_button("Search")

if do_search:
    kept_idx, dark_hours = search_visible_stars(
        tuple(run['RA_decimal'].astype(float)),
        tuple(run['Dec_decimal'].astype(float)),
        float(lat_dec1), float(lon_dec1), float(height1),
        date_str, float(min_altitude_deg),
    )
    if dark_hours == 0:
        st.warning("The Sun never sets at this site on this date — no dark window.")
    else:
        visible = run.iloc[kept_idx].reset_index(drop=True)
        st.write(
            f"Dark window: {dark_hours:.1f} h.  {len(visible)} of the {len(run)} checked stars are "
            f"observable for at least 3/4 of it."
        )

        if baselines and len(visible):
            theta_col = visible[f"Diameter_{band}"].where(
                visible[f"Diameter_{band}"].notna(), visible["Diameter_V"])
            scores = score_candidates(
                tuple(visible['RA_decimal'].astype(float)),
                tuple(visible['Dec_decimal'].astype(float)),
                tuple(theta_col.astype(float)),
                float(lat_dec1), float(lon_dec1), float(height1),
                date_str, float(min_altitude_deg),
                tuple((lbl, tuple(enu)) for lbl, enu in baselines), float(lambda_sel),
            )
            visible['coverage'] = [round(s['score']) if s else np.nan for s in scores]
            visible['x_range'] = [f"{s['x_min']:.1f}–{s['x_max']:.1f}" if s else "" for s in scores]
            visible['slope%'] = [round(s['frac_on_slope'] * 100) if s else np.nan for s in scores]
            visible = visible.sort_values('coverage', ascending=False, na_position='last') \
                             .reset_index(drop=True)
            diam_col = f"Diameter_{band}"
            cols = ['BayerF', 'Common', 'Vmag', diam_col, 'coverage', 'x_range', 'slope%', 'RA', 'Dec']
            cols = list(dict.fromkeys(cols))  # de-dup if band == V
            st.dataframe(visible[cols])
            st.caption("coverage: 0–100, the mean steepness of each star's visibility curve where "
                       "tonight's UV track samples it.  x = π·ρ·θ/λ (|V|²=1 at 0, first null at "
                       "3.83); the useful descent is x ≈ 1–3.83.  slope% = fraction of the night "
                       "the track is on that descent.")
        else:
            st.dataframe(visible[display_cols])

        st.download_button(
            "Download list as CSV",
            data=visible.to_csv(index=False),
            file_name=f"stars_visible_{date_str}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------------------------------------------------------------------
# Single star: sky track + visibility map + UV trace
# ---------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## Track a single star")

selected_star = st.selectbox(
    'Select a star:', options=df_all_stars['Identifier'],
    help="Sorted by apparent (V) magnitude, brightest first.",
)
star = df_all_stars[df_all_stars['Identifier'] == selected_star].iloc[0]

BayerF = star['BayerF']
given_ra_decimal = float(star['RA_decimal'])
given_dec_decimal = float(star['Dec_decimal'])

diameter_band = star.get(f"Diameter_{band}")
phi_band = star.get(f"Phi_{band}")
if diameter_band is None or pd.isna(diameter_band):
    st.warning(f"No {band}-band diameter for {selected_star} (missing colour index) — using V band instead.")
    diameter_band = star['Diameter_V']
    phi_band = star['Phi_V']
    lambda_star = lambda_V
else:
    lambda_star = lambda_sel
diameter_band = float(diameter_band)
phi_band = float(phi_band)
diameter_in_rad = diameter_band / 1000 * np.pi / (3600 * 180)

lat = float(lat_dec1)
lon = float(lon_dec1)

# Real observable window for this star this night (Sun down + above the altitude limit).
times_jd = find_observable_times(
    given_ra_decimal, given_dec_decimal, lat, lon, date_str,
    height_m=float(height1), min_altitude_deg=float(min_altitude_deg),
)

if len(times_jd) < 3:
    st.warning(f"{selected_star} is not observable from this site on the night of {date_str} "
               f"(never above {min_altitude_deg}° while the Sun is down).")
else:
    obs_times = Time(times_jd, format='jd')
    star_coord = SkyCoord(given_ra_decimal, given_dec_decimal,
                          unit=(u.hourangle, u.deg), frame='icrs')
    altaz = star_coord.transform_to(
        AltAz(obstime=obs_times, location=EarthLocation(lat=lat, lon=lon, height=height1))
    )
    altitudes = altaz.alt.deg
    azimuths = altaz.az.deg
    local_dt = (obs_times + utc_offset * u.hour).to_datetime()
    time_labels = [dt.strftime('%H:%M') for dt in local_dt]
    xtick_step = max(1, len(time_labels) // 8)

    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    sc = ax1.scatter(time_labels, altitudes, c=azimuths)
    plt.colorbar(sc, label='Azimuth [°]', ax=ax1)
    ax1.set_xticks(time_labels[::xtick_step])
    ax1.set_title("Celestial path of " + str(BayerF))
    ax1.set_xlabel(f'Local time ({tz_label})')
    ax1.set_ylabel('Altitude [°]')
    ax1.set_ylim(0, 90)
    ax1.grid(True)
    st.pyplot(fig1)
    plt.close(fig1)

    if not baselines:
        st.info("Choose a telescope-array preset, or set **Two telescopes → Yes** with a "
                "non-zero baseline, to see the visibility map and UV coverage.")
    else:
        # One UVW track per baseline over the observable window (both this point and its
        # conjugate -U,-V are measured by an intensity interferometer).
        tracks = [(lbl, compute_uvw_track(given_ra_decimal, given_dec_decimal, lat, lon, enu, times_jd))
                  for lbl, enu in baselines]
        st.caption("Baselines: " +
                   ", ".join(f"{lbl} ({np.linalg.norm(enu):.0f} m)" for lbl, enu in baselines))

        # Frame the plot to the actual UV coverage, so the tracks fill it rather than
        # sitting in a corner of a box sized by the nominal baseline length.
        track_reach = max(np.sqrt(U ** 2 + V ** 2).max() for _, (U, V, W) in tracks)
        size_to_plot = max(track_reach * 1.15, 1.0)
        title = ("Visibility map of " + str(BayerF) + ", diameter: " + str(diameter_band) +
                 " mas\n Φ = " + str(np.round(phi_band, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")

        resolution = 400
        grid = np.linspace(-size_to_plot, size_to_plot, resolution)
        X, Y = np.meshgrid(grid, grid)
        intensity_values = visibility(np.sqrt(X ** 2 + Y ** 2), diameter_in_rad, lambda_star)

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        cax = ax2.imshow(intensity_values,
                         extent=(-size_to_plot, size_to_plot, -size_to_plot, size_to_plot),
                         origin='lower', cmap='gray', zorder=0)
        for (lbl, (U, V, W)), c in zip(tracks, TRACK_COLOURS):
            ax2.plot(U, V, '-', color=c, lw=1.0, alpha=0.9, zorder=3)
            ax2.plot(U, V, 'o', color=c, ms=4, markeredgecolor='black', markeredgewidth=0.4,
                     label=lbl, zorder=4)
            ax2.plot(-U, -V, '-', color=c, lw=1.0, alpha=0.9, zorder=3)
            ax2.plot(-U, -V, 'o', color=c, ms=4, markeredgecolor='black', markeredgewidth=0.4,
                     zorder=4)
        ax2.set_xlim(-size_to_plot, size_to_plot)
        ax2.set_ylim(-size_to_plot, size_to_plot)
        ax2.set_title(title)
        ax2.set_xlabel('U [m]')
        ax2.set_ylabel('V [m]')
        ax2.set_aspect('equal')
        ax2.legend(fontsize=7, loc='upper right', framealpha=0.85)
        plt.colorbar(cax, label="Squared visibility", ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)

        # 1-D visibility curve with the actually-traced points marked on it.
        if diameter_in_rad > 0:
            rho_all = np.concatenate([np.hypot(U, V) for _, (U, V, W) in tracks])
            sc = coverage_score(rho_all, diameter_in_rad, lambda_star)
            k = lambda_star / (np.pi * diameter_in_rad)          # rho = k * x
            r_null = X_FIRST_NULL * k
            r_hi = max(rho_all.max() * 1.1, r_null * 1.6)
            rr = np.linspace(r_hi * 1e-3, r_hi, 600)

            fig4, axc = plt.subplots(figsize=(9, 4))
            axc.axvspan(X_RESOLVED * k, r_null, color="#d9c8a0", alpha=0.30, zorder=0,
                        label="steep descent (best for θ)")
            axc.plot(rr, visibility(rr, diameter_in_rad, lambda_star), "k-", lw=1.6, zorder=2)
            axc.axvline(r_null, ls="--", color="0.45", lw=1, zorder=1)
            for (lbl, (U, V, W)), c in zip(tracks, TRACK_COLOURS):
                r_i = np.hypot(U, V)
                axc.plot(r_i, visibility(r_i, diameter_in_rad, lambda_star), "o", color=c, ms=4,
                         markeredgecolor="black", markeredgewidth=0.4, label=lbl, zorder=4)
            axc.set_yscale("log")
            axc.set_ylim(1e-4, 1.4)
            axc.set_xlim(0, r_hi)
            axc.set_xlabel(r"projected baseline  $\rho=\sqrt{U^2+V^2}$  [m]")
            axc.set_ylabel(r"squared visibility  $|V|^2$")
            axc.set_title(f"Visibility curve — coverage {sc['score']:.0f}/100: {score_verdict(sc)}")
            secx = axc.secondary_xaxis("top", functions=(lambda r: r / k, lambda x: x * k))
            secx.set_xlabel(r"$x=\pi\rho\theta/\lambda$   ($|V|^2=1$ at 0, first null at 3.83)")
            axc.legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.85)
            axc.grid(True, which="both", alpha=0.3)
            st.pyplot(fig4)
            plt.close(fig4)
