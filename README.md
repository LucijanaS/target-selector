# target-selector

A Streamlit app for picking stellar intensity interferometry (SII) targets for a given
site and night.

Given a location (typed in, or chosen from a telescope-coupling preset) and a date, it:

- lists the brightest stars that stay above a minimum altitude while the Sun is down for
  most of that night's real dark window (sunset/sunrise computed for the site, not a fixed
  clock window). With a telescope array selected, each star gets a **UV-coverage score**
  (0–100) — how much of the first lobe of its visibility curve (|V|² from 1 at zero baseline
  to 0 at the first null, ρ = 1.22 λ/θ) tonight's UV track sweeps — and an **integration
  time** for a 5σ detection from the standard SII SNR
  (`SNR = 2^{-1/2}·|V₁₂|²·ε·Φ·√(A₁A₂)·√(T/√(Δt₁Δt₂))`), evaluated with the real |V|² along
  each baseline's traced track (so an over-resolved star — baselines past the null — is slow
  even at high coverage) and Fisher-combined over the array. Inputs (sidebar): dish size,
  detector time resolution Δt (MAGIC ≈ 9 ns ↔ b_v ≈ 110 MHz), efficiency ε (MAGIC ≈ 0.09,
  VERITAS ≈ 0.15). Sky background, spectral channels and the sub-2× instrumental-noise terms
  (PMT excess noise, electronic noise, filter shape) are omitted, so it stays ~1.5×
  optimistic vs a full model like `siicheduler`.
- for a single selected star, plots its sky track, its uniform-disk visibility map with the
  UV tracks of every array baseline, and the **1-D visibility curve** |V|²(ρ) with the
  traced points marked on it, the coverage score and the estimated integration time.

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
`telescopes.py`), and an N-dish array is plotted with the UV track of **every** dish pair
(N(N−1)/2 baselines). Presets: MAGIC, VERITAS, CTAO-South LSTs, CTAO-North LSTs, C2PU
(Calern), H.E.S.S. CT3+CT4, and **MAGIC ×2 + LST-1** (a mixed 17 m / 23 m array — dish areas
combined per baseline). **Narrabri NSII** is a movable pair: a fixed site with a
user-set baseline length (10–188 m) and orientation. The baseline maths is
`telescope_coords.all_pairwise_baselines`. Pick **Custom (enter manually)** to type
everything by hand.

## Run it

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
