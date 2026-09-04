"""
Telescope-coupling presets for the target-selector app.

A preset is a site plus a list of dish positions. An N-dish array gives N(N-1)/2
simultaneous baselines and the app draws the UV track of every pair (with a multiselect
to focus on a subset); two-dish presets (MAGIC, C2PU) give a single baseline.

The baseline geometry is *not* re-implemented here -- `pairwise_baselines()` and
`latlon_to_enu()` delegate to `telescope_coords.all_pairwise_baselines` /
`telescope_coords.enu_baseline`, vendored verbatim from `brightstar/LimbO/`, the same code
`brightstar/LimbO/candidate_ranking.py` uses to rank stars against a real array's coverage.
The UV projection itself is `brightstar_functions.compute_uvw_track`.

Dish coordinates for MAGIC, VERITAS and CTAO-South are copied from
`brightstar/LimbO/telescope_arrays.py`; C2PU from `brightstar/brightstar_input.py`.
Presets flagged `approx=True` (H.E.S.S., CTAO-North) have only a placeholder site centre
and no dish list yet -- the app keeps its manual baseline controls for them. Add the real
dishes to `dishes=[...]` (and drop `site_*` + `approx`) once coordinates are available.
"""

from telescope_coords import parse_lat_lon, enu_baseline, all_pairwise_baselines


def latlon_to_enu(lat1_deg, lon1_deg, height1_m, lat2_deg, lon2_deg, height2_m):
    """East / North / Up baseline [m] from telescope 1 to telescope 2, via
    telescope_coords.enu_baseline (ECEF difference rotated into the local tangent plane)."""
    a = {"lat": str(lat1_deg), "lon": str(lon1_deg), "height": height1_m}
    b = {"lat": str(lat2_deg), "lon": str(lon2_deg), "height": height2_m}
    e, n, up = enu_baseline(a, b)
    return round(float(e), 3), round(float(n), 3), round(float(up), 3)


def pairwise_baselines(dishes):
    """[(label, (x_E, x_N, x_up)), ...] for every dish pair (i < j). `dishes` is a list of
    {'name','lat','lon','height'} dicts; the ENU vectors come from
    telescope_coords.all_pairwise_baselines."""
    out = []
    for i, j, x_e, x_n, x_up in all_pairwise_baselines(dishes):
        label = f'{dishes[i]["name"]}–{dishes[j]["name"]}'
        out.append((label, (round(float(x_e), 3), round(float(x_n), 3), round(float(x_up), 3))))
    return out


def preset_site(preset):
    """(lat_deg, lon_deg, height_m) for the observability/altitude calc: first dish, else
    the placeholder `site_*` fields, else (None, None, None). Mirrors how
    candidate_ranking.rank_candidates takes the array's site from its first telescope."""
    if preset.get("dishes"):
        d = preset["dishes"][0]
        h = d["height"]
        return parse_lat_lon(d["lat"]), parse_lat_lon(d["lon"]), (h.value if hasattr(h, "value") else float(h))
    return preset.get("site_lat"), preset.get("site_lon"), preset.get("site_height_m")


def preset_has_dishes(preset):
    return bool(preset.get("dishes"))


# --- Real dish coordinates ---------------------------------------------------------------
# MAGIC / VERITAS / CTAO-South: copied from brightstar/LimbO/telescope_arrays.py
_magic = [
    dict(name="MAGIC-1", lat="28 45 40.8N", lon="17 53 26.0W", height=2200.0),
    dict(name="MAGIC-2", lat="28 45 43.1N", lon="17 53 24.2W", height=2200.0),
]
_veritas = [
    dict(name="T1", lat="31 40 29.6N", lon="110 57 03.5W", height=1275.893),
    dict(name="T2", lat="31 40 28.3N", lon="110 57 06.9W", height=1271.016),
    dict(name="T3", lat="31 40 32.0N", lon="110 57 07.5W", height=1267.358),
    dict(name="T4", lat="31 40 30.3N", lon="110 57 10.0W", height=1268.273),
]
_cta_south_lsts = [
    dict(name="LST-01", lat="-24.68361537", lon="-70.31570456", height=2165.0),
    dict(name="LST-02", lat="-24.68270683", lon="-70.31633741", height=2160.0),
    dict(name="LST-03", lat="-24.68360410", lon="-70.31698922", height=2162.0),
    dict(name="LST-04", lat="-24.68451264", lon="-70.31635637", height=2164.0),
]
# C2PU / Calern Épsilon + Omicron 1 m telescopes (from brightstar/brightstar_input.py).
_c2pu = [
    dict(name="Épsilon", lat="43.75370", lon="6.92294", height=1776.507),
    dict(name="Omicron", lat="43.75370", lon="6.92312", height=1783.504),
]

custom = "Custom (enter manually)"

# Fields used only by the rough SII integration-time estimate:
#   dish_m      light-collector diameter [m]; collecting area ~ pi (dish_m/2)^2. Every dish
#               in a given array is assumed identical (true for all real presets below --
#               the one exception is H.E.S.S., where CT5 is 28 m vs CT1-4 ~12 m).
#   delta_t_ns  detector time resolution ~ 1/(electronic bandwidth) [ns]. Set only where a
#               value is published; otherwise None -> the sidebar default is used.
telescope_presets = {
    custom: dict(
        dishes=None, dish_m=None, delta_t_ns=None, approx=False,
        note="Enter site coordinates and baseline by hand, as before.",
    ),

    "MAGIC (La Palma)": dict(
        dishes=_magic, dish_m=17.0, delta_t_ns=2.2, approx=False,
        note="MAGIC-1 ↔ MAGIC-2, Roque de los Muchachos. δt from Acciari et al. 2024 "
             "(measured correlation-peak width).",
    ),

    "VERITAS (Whipple, Arizona)": dict(
        dishes=_veritas, dish_m=12.0, delta_t_ns=4.0, approx=False,
        note="4 telescopes → 6 baselines, Fred Lawrence Whipple Observatory. δt from "
             "Abeysekara et al. 2020 (system time response).",
    ),

    "CTAO-South LSTs (Paranal, Chile)": dict(
        dishes=_cta_south_lsts, dish_m=23.0, delta_t_ns=None, approx=False,
        note="4 LSTs → 6 baselines, near Cerro Paranal. No published SII δt yet.",
    ),

    "C2PU – Épsilon/Omicron (Calern)": dict(
        dishes=_c2pu, dish_m=1.0, delta_t_ns=None, approx=False,
        note="Épsilon ↔ Omicron 1 m telescopes on the Plateau de Calern.",
    ),

    "H.E.S.S. (Khomas, Namibia)": dict(
        dishes=None, dish_m=12.0, delta_t_ns=None,
        site_lat=-23.2717, site_lon=16.5028, site_height_m=1800.0, approx=True,
        note="Placeholder site centre — enter a CT pair (~120 m) manually. dish_m is the "
             "CT1-4 value; CT5 is 28 m (heterogeneous — not modelled).",
    ),

    "CTAO-North LSTs (La Palma)": dict(
        dishes=None, dish_m=23.0, delta_t_ns=None,
        site_lat=28.7616, site_lon=-17.8906, site_height_m=2200.0, approx=True,
        note="Placeholder site centre — enter an LST pair baseline manually for now.",
    ),

    "Narrabri NSII (Paul Wild Obs., Australia)": dict(
        dishes=None, dish_m=6.5, delta_t_ns=10.0,
        site_lat=-30.3128, site_lon=149.5501, site_height_m=217.0, approx=True,
        note="Hanbury Brown & Twiss 1963–74. Two 6.5 m reflectors on a 188 m circular rail — "
             "baseline 0–188 m; enter the value you want. δt ~10 ns (1970s ~60 MHz correlator).",
    ),

    "StarBase Utah (Grantsville)": dict(
        dishes=None, dish_m=3.0, delta_t_ns=None,
        site_lat=40.6939, site_lon=-112.4611, site_height_m=1310.0, approx=True,
        note="Placeholder site centre — University of Utah SII testbed, two 3 m dishes ~23 m "
             "apart. Enter the baseline manually.",
    ),
}
