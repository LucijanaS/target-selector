# target-selector

A Streamlit app for picking stellar intensity interferometry (SII) targets for a given
site and night.

Given a location (typed in, or chosen from a telescope-coupling preset) and a date, it:

- lists the brightest stars that stay above a minimum altitude while the Sun is down for
  most of that night's real dark window (sunset/sunrise computed for the site, not a fixed
  clock window). With a telescope array selected, the list is **ranked by a UV-coverage
  score** (0–100): how much of the SNR-weighted first lobe each star's UV track spans —
  ideally from a short baseline (bright, |V|² ≈ 1) out toward the first null (x = π·ρ·θ/λ =
  3.83), with points weighted by |V|² and a floor at 0.2. A star being up is necessary but
  not sufficient; baseline past the first null is wasted.
- for a single selected star, plots its sky track, its uniform-disk visibility map with the
  UV tracks of every array baseline, and the **1-D visibility curve** |V|²(ρ) with the
  traced points marked on it and the coverage score / verdict in the title.

The star data is the Yale Bright Star Catalogue
(<http://tdc-www.harvard.edu/catalogs/bsc5.html>), converted to JSON by
<https://github.com/brettonw/YaleBrightStarCatalog> (`bsc5-all.json`). Angular diameters and
photon flux densities Φ are reconstructed from the U/B/V magnitudes and effective
temperature; you choose the band in the sidebar.

Companion projects:

- [`brightstar`](https://github.com/LucijanaS/brightstar) — the research repo. This app's
  `brightstar_functions.py` (UVW projection `compute_uvw_track`, observability window
  `find_observable_times`, uniform-disk `visibility`) and `telescope_coords.py` (pairwise
  ENU baselines, as used by `brightstar/LimbO/candidate_ranking.py`) are copied from it
  verbatim; the app only adds the Streamlit UI and the per-baseline plotting loop on top.
- [`target-stars`](https://github.com/LucijanaS/target-stars) /
  <https://target-stars-sii.streamlit.app/> — the H-R / Φ-vs-θ catalogue explorer.

## Telescope presets

The sidebar "Telescope coupling" menu fills in a site and its dish positions (see
`telescopes.py`). An N-dish array is plotted with the UV track of **every** dish pair
(N(N-1)/2 baselines); a multiselect lets you focus on a subset. **MAGIC, VERITAS,
CTAO-South LSTs and C2PU (Calern)** have real dish coordinates (copied from
`brightstar/LimbO/telescope_arrays.py` and `brightstar/brightstar_input.py`); the baseline
maths is `telescope_coords.all_pairwise_baselines`. **H.E.S.S.** and **CTAO-North LSTs**
still have placeholder site centres and no dish list — enter a baseline by hand for those,
or add their dishes to `telescopes.py`. Pick **Custom (enter manually)** to type everything
by hand.

## Run it

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
