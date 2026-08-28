import json
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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

utc_offset = st.sidebar.number_input(
    "UTC offset (for displaying local time on the altitude plot only):",
    min_value=-12, max_value=12, step=1, value=+2,
)

# --- Site + baselines ---------------------------------------------------------------
# Produces:  lat_dec1, lon_dec1, height1  -- observer site, for the observability window
#            baselines : list of (label, (x_E, x_N, x_up))  -- one entry per dish pair
lat_dec2 = lon_dec2 = None
height1 = 0.0
height2 = 0.0
two_telescopes = "Yes"
baselines = []

site_lat, site_lon, site_height = preset_site(preset)

if preset_has_dishes(preset):
    lat_dec1 = st.sidebar.number_input("Site latitude [deg]:", value=float(site_lat), format="%.5f")
    lon_dec1 = st.sidebar.number_input("Site longitude [deg]:", value=float(site_lon), format="%.5f")
    height1 = st.sidebar.number_input("Site elevation [m]:", value=float(site_height), format="%.1f")
    st.sidebar.caption("Site fields drive the observability window only; the baselines come "
                       "from the fixed array geometry.")
    all_baselines = pairwise_baselines(preset["dishes"])
    labels = [lbl for lbl, _ in all_baselines]
    chosen = st.sidebar.multiselect(
        f"Baselines to plot ({len(labels)} available)", labels, default=labels,
    )
    baselines = [(lbl, enu) for lbl, enu in all_baselines if lbl in chosen]
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

baseline_lengths = [float(np.linalg.norm(enu)) for _, enu in baselines]
max_baseline = max(baseline_lengths) if baseline_lengths else 1.0

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

st.markdown("## Stars visible on the night of " + date_str)
st.write(
    "The table below lists the brightest " + str(int(number_of_stars)) + " stars the search will run "
    "through. Press **Search** to keep only those that stay above " + str(min_altitude_deg) +
    "° while the Sun is down for at least 3/4 of the night."
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
    ax1.set_xlabel(f'Local time (UTC{utc_offset:+d})')
    ax1.set_ylabel('Altitude [°]')
    ax1.set_ylim(0, 90)
    ax1.grid(True)
    st.pyplot(fig1)
    plt.close(fig1)

    if not baselines:
        st.info("Pick a telescope preset with dishes, or set **Two telescopes → Yes** with a "
                "non-zero baseline, to see the visibility map and UV coverage.")
    else:
        st.caption("Baselines plotted: " +
                   ", ".join(f"{lbl} ({np.linalg.norm(enu):.0f} m)" for lbl, enu in baselines))

        # One UVW track per baseline over the observable window.
        tracks = [(lbl, compute_uvw_track(given_ra_decimal, given_dec_decimal, lat, lon, enu, times_jd))
                  for lbl, enu in baselines]
        colours = plt.cm.tab10(np.linspace(0, 1, 10))[:len(tracks)]
        multi = len(tracks) > 1
        size_to_plot = max(max_baseline, 1.0)
        title = ("Visibility map of " + str(BayerF) + ", diameter: " + str(diameter_band) +
                 " mas\n Φ = " + str(np.round(phi_band, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")

        resolution = 300
        grid = np.linspace(-size_to_plot, size_to_plot, resolution)
        X, Y = np.meshgrid(grid, grid)
        Rgrid = np.sqrt(X ** 2 + Y ** 2)
        intensity_values = visibility(Rgrid, diameter_in_rad, lambda_star)

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        cax = ax2.imshow(intensity_values,
                         extent=(-size_to_plot, size_to_plot, -size_to_plot, size_to_plot),
                         origin='lower', cmap='gray')
        for (lbl, (U, V, W)), c in zip(tracks, colours):
            ax2.plot(U, V, '.', color=c, markeredgecolor='black', label=lbl)
            ax2.plot(-U, -V, '.', color=c, markeredgecolor='black')
        ax2.set_title(title)
        ax2.set_xlabel('U [m]')
        ax2.set_ylabel('V [m]')
        ax2.set_aspect('equal')
        if multi:
            ax2.legend(fontsize=7, loc='upper right')
        plt.colorbar(cax, label="Intensity")
        st.pyplot(fig2)
        plt.close(fig2)

        # W (delay) over the night, one line per baseline.
        fig4, ax4 = plt.subplots(figsize=(9, 3.5))
        for (lbl, (U, V, W)), c in zip(tracks, colours):
            ax4.plot(time_labels, W, '.', color=c, label=lbl)
        ax4.set_xticks(time_labels[::xtick_step])
        ax4.set_xlabel(f'Local time (UTC{utc_offset:+d})')
        ax4.set_ylabel('W [m]')
        ax4.set_title("Delay W over the night")
        if multi:
            ax4.legend(fontsize=7)
        ax4.grid(True)
        st.pyplot(fig4)
        plt.close(fig4)
