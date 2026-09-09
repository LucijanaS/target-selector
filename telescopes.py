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
`brightstar/LimbO/telescope_arrays.py`; C2PU from `brightstar/brightstar_input.py`; H.E.S.S.
CT3/CT4 and the CTAO-North LSTs were supplied directly. A preset with `movable=True`
(Narrabri) has a fixed site but a user-set baseline length (`baseline_range`) and
orientation rather than a dish list. Any remaining `approx=True` preset has only a
placeholder site centre.
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


def pairwise_dish_diams(dishes, fallback_dish_m):
    """[(dm_i, dm_j), ...] light-collector diameters [m] for every dish pair (i < j), in the
    same order as pairwise_baselines(). Uses each dish's own 'dish_m' when present (mixed
    arrays like MAGIC + LST-1), else `fallback_dish_m`."""
    dm = [float(d.get("dish_m") or fallback_dish_m) for d in dishes]
    return [(dm[i], dm[j]) for i in range(len(dishes)) for j in range(i + 1, len(dishes))]


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
# H.E.S.S. CT3 + CT4 (12 m dishes), Khomas Highland.
_hess = [
    dict(name="CT3", lat="-23.271541", lon="16.499250", height=1800.0),
    dict(name="CT4", lat="-23.272336", lon="16.500116", height=1800.0),
]
# CTAO-North LST-1..4 (23 m dishes), Roque de los Muchachos, La Palma.
_lst_north = [
    dict(name="LST-1", lat="28.761538", lon="-17.891495", height=2200.0),
    dict(name="LST-2", lat="28.761853", lon="-17.892707", height=2200.0),
    dict(name="LST-3", lat="28.762863", lon="-17.892546", height=2200.0),
    dict(name="LST-4", lat="28.762458", lon="-17.891386", height=2200.0),
]
# MAGIC-1, MAGIC-2 (17 m) + LST-1 (23 m) -- a mixed array, all at Roque de los Muchachos.
_magic_lst1 = [
    dict(name="MAGIC-1", lat=_magic[0]["lat"], lon=_magic[0]["lon"], height=2200.0, dish_m=17.0),
    dict(name="MAGIC-2", lat=_magic[1]["lat"], lon=_magic[1]["lon"], height=2200.0, dish_m=17.0),
    dict(name="LST-1", lat=_lst_north[0]["lat"], lon=_lst_north[0]["lon"], height=2200.0, dish_m=23.0),
]

custom = "Custom (enter manually)"

# Fields used only by the SII integration-time estimate (Rai/Basak/Saha 2021 noise model):
#   dish_m      light-collector diameter [m]; area ~ pi (dish_m/2)^2. A dish dict may carry
#               its own dish_m to override this (mixed arrays, e.g. MAGIC x2 + LST-1).
#   delta_t_ns  detector time resolution [ns]; 1/sqrt(dt_i dt_j) is the effective bandwidth
#               in the SII SNR formula. MAGIC's b_v ~ 110 MHz -> dt ~ 9 ns.
#   efficiency  detector QE x optical throughput.
# Set only where a value is published (MAGIC: b_v ~ 110 MHz, eff ~ 0.09; VERITAS eff ~ 0.15);
# otherwise None -> the sidebar defaults are used.
telescope_presets = {
    custom: dict(
        dishes=None, dish_m=None, delta_t_ns=None, approx=False,
        note="Enter site coordinates and baseline by hand, as before.",
    ),

    "MAGIC (La Palma)": dict(
        dishes=_magic, dish_m=17.0, delta_t_ns=9.09, efficiency=0.09, approx=False,
        note="MAGIC-1 ↔ MAGIC-2, Roque de los Muchachos. Δt ≈ 9 ns (b_v ≈ 110 MHz) and ε ≈ 0.09 from the "
             "siicheduler MAGIC config (Acciari et al. 2020/2024).",
    ),

    "VERITAS (Whipple, Arizona)": dict(
        dishes=_veritas, dish_m=12.0, delta_t_ns=None, efficiency=0.15, approx=False,
        note="4 telescopes → 6 baselines, Fred Lawrence Whipple Observatory. ε ≈ 0.15 (Abeysekara et al. 2020); "
             "no published effective bandwidth.",
    ),

    "CTAO-South LSTs (Paranal, Chile)": dict(
        dishes=_cta_south_lsts, dish_m=23.0, delta_t_ns=None, approx=False,
        note="4 LSTs → 6 baselines, near Cerro Paranal. No published SII bandwidth/efficiency.",
    ),

    "C2PU – Épsilon/Omicron (Calern)": dict(
        dishes=_c2pu, dish_m=1.0, delta_t_ns=None, approx=False,
        note="Épsilon ↔ Omicron 1 m telescopes on the Plateau de Calern.",
    ),

    "H.E.S.S. CT3 + CT4 (Khomas, Namibia)": dict(
        dishes=_hess, dish_m=12.0, delta_t_ns=None, approx=False,
        note="CT3 ↔ CT4, 12 m dishes. (CT5, 28 m, not included.)",
    ),

    "CTAO-North LSTs (La Palma)": dict(
        dishes=_lst_north, dish_m=23.0, delta_t_ns=None, approx=False,
        note="LST-1..4 → 6 baselines, Roque de los Muchachos. No published SII bandwidth/efficiency.",
    ),

    "MAGIC ×2 + LST-1 (La Palma)": dict(
        dishes=_magic_lst1, dish_m=17.0, delta_t_ns=9.09, efficiency=0.09, approx=False,
        note="MAGIC-1, MAGIC-2 (17 m) and LST-1 (23 m) → 3 baselines. Mixed dish sizes are "
             "handled per pair.",
    ),

    "Narrabri NSII (Paul Wild Obs., Australia)": dict(
        dishes=None, dish_m=6.5, delta_t_ns=16.7,
        site_lat=-30.209167, site_lon=149.751111, site_height_m=217.0,
        movable=True, baseline_range=(10.0, 188.0), approx=False,
        note="Hanbury Brown & Twiss 1963–74. Two 6.5 m reflectors moved around a central "
             "point; set the baseline length (10–188 m) and orientation. Δt ≈ 17 ns (b_v ≈ 60 MHz, 1970s correlator).",
    ),
}
