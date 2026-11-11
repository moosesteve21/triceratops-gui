#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical validation of a TESS planet candidate with TRICERATOPS.

Pipeline
--------
1.  Build a multi-sector light curve from TESScut FFI cutouts, with a
    per-sector aperture and a background/scattered-light correction.
2.  Two-pass detrending: a first wide biweight pass -> TLS search ->
    re-detrend with the transits masked, so the transit depth is not
    suppressed by the filter.
3.  TLS transit search, with a period grid limited by the actual baseline.
4.  Per-epoch transit depths, measured by fitting the *TLS transit shape*
    with only the depth and a local linear baseline free.  Gives odd/even
    and depth-consistency diagnostics.
5.  Difference-image centroid per sector (in-transit minus out-of-transit).
6.  TRICERATOPS FPP/NFPP, run n_iter times, with the correct per-sector
    apertures.
7.  Optional: nearby-star vetting from their own light curves, which lets
    you call target.remove_star() and drive NFPP to ~0.

Notes
-----
* Robust FPP (v2): calc_probs is given a *realistic* flux uncertainty (OOT
  scatter of the binned LC incl. red noise, optional floor, optionally
  inflated until a smooth reference model has chi2_red = 1).  Every
  calc_probs run is checked for NaN / degenerate lnZ and retried with a larger
  sigma; invalid runs are never averaged in silently, and the final verdict is
  'indeterminate' when too few runs are valid.  triceratops >= 1.1.0 is
  recommended (logsumexp normalisation, target.lnZ, target.FPP_degenerate).
* Optional limb-darkened fit of the folded LC (--ld-fit, --ld-free,
  --ld-coeffs U1 U2): quadratic limb darkening, Rp/Rs, b, a/Rs, T0 shift and
  baseline free; exposure smoothing included.  It replaces the trapezoid as
  the reference model for the sigma inflation and writes ld_fit_<TIC>.pdf.
* Individual transits (v3): the per-transit analysis now also runs in
  known-ephemeris mode (before, `results is None` skipped it entirely), fits
  the mid-time of every transit (+- per_transit_tc_range x T14), lists every
  expected epoch that is unusable with the reason, refines P and T0 from the
  measured mid-times (--no-refine-ephemeris to disable) and tests the depth at
  phase 0.5 (P/2 alias / secondary eclipse).
* FPP/NFPP are derived from the *pooled* scenario evidences of all valid runs
  (log-mean-exp per scenario, bootstrap error over runs) and the log warns
  when one run dominates the evidence (Monte Carlo not converged: raise
  --n-samples).  --no-pool gives the old mean of per-run FPPs.
* Stars requested for removal are verified to be gone after remove_star.
* Detection (v3.1): --detrend-windows W1 W2 ... runs the blind TLS search once
  per first-pass biweight window and keeps the highest-SDE candidate; the log
  lists all windows (P, T14, depth, SDE, n_transits, fraction of in-transit
  cadences near data-segment edges, whether the period agrees).  A period
  found with a single window only is flagged as probable red-noise alias.
* v3.2: a known ephemeris (--ephemeris) is checked against the data before
  the transit-masked detrend: every expected transit is searched within
  +-max(2 T14, 12 h) and P/T0 are refined from a robust fit of the found
  mid-times.  --ephem-search grid uses a folded box search on a small
  (P, T0) grid instead (stacks all transits: works for transits that are
  individually below the noise; S/N map in ephem_grid_<TIC>.pdf); the
  default 'auto' falls back to it when the scan is inconclusive.  TRICERATOPS cutouts from
  astrocut >= 1.4 (no 1CRV4P/2CRV4P keywords) are put in the same pixel
  frame as the apertures; sectors TRICERATOPS drops are removed from
  target.sectors; TESScut downloads are retried; TESScut photometry is
  passed to TRICERATOPS as not crowding-corrected (--dilution-corrected to
  override).  Outlier clipping no longer cuts the bottom off deep eclipses
  (first pass: really positive-only; in transit: only fluxes < 10 % of the
  continuum are removed, instead of everything below continuum - 15 sigma).
  Secondary-eclipse search at all orbital phases and several widths
  (eccentric orbits), with a single-event check, 3-sigma depth limits and a
  companion brightness-temperature limit (secondary_search_<TIC>.pdf/.txt).
  Per-transit depths corrected for the target's flux fraction in each
  sector's aperture (odd/even overall and per cadence); grazing warning of
  the limb-darkened fit.
* v3.2.1: secondary-eclipse candidates are re-measured event by event with a
  local quadratic baseline and need consistent event depths (S/N with
  scatter-inflated errors) and a dip in >= 60 % of the events.
* v4.0 (October 2026, "definitive" version):
  - TESScut aperture chosen to maximise the S/N of the target's own flux,
    from a PSF model of the TIC field (--aperture optimal|percentile);
  - every sector's light curve is put on the target's flux scale with its
    own target fraction g_s before folding (--dilution-mode sector|none);
    TRICERATOPS then receives dilution_corrected=True;
  - TRICERATOPS models are smoothed with the effective exposure of the
    folded, binned light curve (exptime/nsamples; the package default of
    2 min is wrong for binned FFI data);
  - calc_probs runs are added until no single run dominates the pooled
    evidence (or tri_max_iter / tri_time_budget); probabilities are also
    reported per class of explanation;
  - difference imaging: systematic floor on the centroid error, direct-
    image fit held on the target, differential offset as the main metric;
  - per-transit depths refitted with the limb-darkened shape smoothed with
    each sector's cadence; per-sector depth table with odd/even per cadence;
  - eclipse-timing / depth-variation analysis (odd-even timing, period
    change, light-travel-time limits, depth trend, ground-based times via
    --extra-times), etv_<TIC>.pdf/.txt;
  - SUMMARY block at the end of the log and summary_<TIC>.json;
  - regression tests: tests/test_triceratops_validation.py (pytest).
* v4.1 (November 2026): ground-based follow-up photometry
  (--ground-lc / --ground-list): AstroImageJ, AAVSO/EXOTIC and plain
  tables; transit fit with free depth and mid-time; flux fraction of every
  nearby star in the ground aperture (seeing PSF, band magnitudes); stars
  whose expected depth is inconsistent with the data are cleared before
  calc_probs; chromaticity test; ground mid-times enter the ETV analysis;
  with TRICERATOPS+ installed, up to seven light curves (one per band) enter calc_probs
  (external_lc_files / filt_lcs).  --exclude-stars for stars cleared
  elsewhere.
* v4.1.1: the command line is built by build_parser() (read by the GUI
  triceratops_gui.py); TRICERATOPS+ compatibility (TESScut requests via
  Tesscut.get_cutouts, evidences in probs["lnZ"]); up to 7 light curves
  for TRICERATOPS+.
* v4.2: Kepler mode (--mission kepler --kic N): long-cadence Kepler SAP /
  PDCSAP light curves and target pixel files per quarter (the quarters take
  the place of the sectors), times in BKJD (BJD - 2454833), Kepler-band
  limb darkening, TRICERATOPS with mission="Kepler" (4" pixels).  The TIC
  number is looked up from the KIC when --tic is not given; the default
  output folder is KIC<kic>/.
* Known ephemeris for a specific target: --ephemeris P T0 DEPTH_PPM T14_H
  (Config holds one global default, which is easily the wrong object).
* triceratops.calc_depths(tdepth=...) takes a *fractional* depth
  (0.005 for 5000 ppm), despite what the docstring says.
* all_ap_pixels must be one aperture per sector, in the same order as
  target.sectors.  Pixel coordinates are [column, row] in full-FFI frame.

Author: rewritten version of triceratops_script.py
"""

from __future__ import annotations

import os
import inspect
import warnings
import argparse
from types import SimpleNamespace
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import lightkurve as lk
from lightkurve import TessLightCurve

from wotan import flatten, t14
from transitleastsquares import transitleastsquares, transit_mask
from astroquery.mast import Catalogs
from scipy.special import erf
from scipy.optimize import least_squares
from astropy.coordinates import SkyCoord
import astropy.units as u

# Compatibility shims for packages that use names removed from NumPy 2 /
# SciPy >= 1.14 (pytransit <= 2.5, TRICERATOPS+, triceratops < 1.1).  Each
# shim is a no-op when the name still exists.
import scipy.integrate as _sci_int
for _name, _val in (("int", int), ("float", float), ("bool", bool),
                    ("NaN", np.nan), ("Inf", np.inf),
                    ("trapz", getattr(np, "trapezoid", None)),
                    ("in1d", getattr(np, "isin", None))):
    if _val is not None and not hasattr(np, _name):
        setattr(np, _name, _val)
if not hasattr(_sci_int, "trapz") and hasattr(_sci_int, "trapezoid"):
    _sci_int.trapz = _sci_int.trapezoid

import triceratops.triceratops as tr


def _patch_triceratops_compat():
    """Fix triceratops' Gauss2D for NumPy 2 in versions that lack the fix.

    calc_depths integrates Gauss2D with scipy's dblquad, which passes scalar
    x, y; np.meshgrid then returns 1x1 arrays and NumPy 2 refuses to turn
    them into the float quad needs ("only 0-dimensional arrays can be
    converted to Python scalars").  triceratops 1.1 has a scalar fast path;
    TRICERATOPS+ (1.0.17) does not.
    """
    g = getattr(tr, "Gauss2D", None)
    if g is None or getattr(g, "_tv_patched", False):
        return

    def Gauss2D(x, y, mu_x, mu_y, sigma, A, _orig=g):
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            ex = ((float(x) - mu_x) ** 2 + (float(y) - mu_y) ** 2) / (2 * sigma ** 2)
            return float(A / (2 * np.pi * sigma ** 2) * np.exp(-ex))
        return _orig(x, y, mu_x, mu_y, sigma, A)

    Gauss2D._tv_patched = True
    tr.Gauss2D = Gauss2D


_patch_triceratops_compat()

__version__ = "4.2"

# Mission and time system (set by set_mission()): TESS times are BTJD
# (BJD_TDB - 2457000), Kepler times BKJD (BJD_TDB - 2454833).
MISSION = "tess"
TIME_ZERO = 2457000.0
TIME_LABEL = "BTJD"


def set_mission(mission: str) -> None:
    global MISSION, TIME_ZERO, TIME_LABEL
    MISSION = (mission or "tess").lower()
    if MISSION == "kepler":
        TIME_ZERO, TIME_LABEL = 2454833.0, "BKJD"
    else:
        MISSION, TIME_ZERO, TIME_LABEL = "tess", 2457000.0, "BTJD"


def is_kepler() -> bool:
    return MISSION == "kepler"


def _mission_id(tic, cfg=None) -> str:
    """Search string for lightkurve: 'TIC n' or 'KIC n'."""
    if is_kepler():
        kic = getattr(cfg, "kic", None) if cfg is not None else None
        if kic is None:
            raise RuntimeError("Kepler mode needs the KIC number (--kic)")
        return f"KIC {int(kic)}"
    return f"TIC {tic}"


def _tic_row_ok(row) -> bool:
    """False for TIC entries flagged ARTIFACT / DUPLICATE / SPLIT."""
    try:
        disp = str(row["disposition"]).strip().upper()
    except Exception:
        return True
    return disp not in ("ARTIFACT", "DUPLICATE", "SPLIT")


def _masked_int(v):
    try:
        if np.ma.is_masked(v):
            return None
        return int(v)
    except Exception:
        return None


def resolve_tic_from_kic(kic: int) -> int:
    """TIC number of a Kepler Input Catalog star.

    1. TIC entries with KIC = kic that are not flagged ARTIFACT / DUPLICATE
       / SPLIT (a DUPLICATE points to its replacement via duplicate_id);
    2. otherwise the nearest unflagged TIC star within 4 arcsec of the KIC
       position (old 2MASS-based TIC entries of close pairs are often SPLIT
       into Gaia-based stars).  The candidates are printed.
    """
    ra = dec = None
    try:
        tab = Catalogs.query_criteria(catalog="Tic", KIC=str(int(kic)))
    except Exception as exc:
        print(f"  ! TIC query by KIC failed ({exc}); trying the KIC position")
        tab = []
    for row in tab:
        if _tic_row_ok(row):
            print(f"  KIC {kic}: TIC {int(row['ID'])} (TIC column KIC)")
            return int(row["ID"])
        disp = str(row["disposition"]).strip().upper()
        dup = _masked_int(row["duplicate_id"]) if "duplicate_id" in tab.colnames else None
        print(f"  KIC {kic}: TIC {int(row['ID'])} is flagged {disp}"
              + (f" (replaced by TIC {dup})" if dup else ""))
        if disp == "DUPLICATE" and dup:
            return dup
        try:
            ra, dec = float(row["ra"]), float(row["dec"])
        except Exception:
            pass
    if ra is None:
        from astroquery.vizier import Vizier
        res = Vizier(columns=["_RA", "_DE"]).query_constraints(
            KIC=str(int(kic)), catalog="J/ApJS/229/30/catalog")
        if not len(res):
            raise RuntimeError(f"KIC {kic} not found (TIC and Vizier)")
        ra, dec = float(res[0]["_RA"][0]), float(res[0]["_DE"][0])
    tab = Catalogs.query_region(SkyCoord(ra, dec, unit="deg"),
                                radius=4 * u.arcsec, catalog="TIC")
    return _pick_tic_at_position(tab, kic)


def _pick_tic_at_position(tab, kic) -> int:
    """Choose the TIC star that IS the KIC star from a 4" cone (sorted by
    distance).  Within 1": an unflagged star; else the replacement of a
    DUPLICATE when that one is unflagged; else the DUPLICATE (a Gaia star at
    this position) before a SPLIT entry (old blend of several stars).  Only
    when nothing lies within 1" the nearest unflagged star is taken (that may
    be a companion: a warning is printed)."""
    rows = list(tab)
    if not rows:
        raise RuntimeError(f"no TIC star within 4 arcsec of KIC {kic}; give --tic")
    for r in rows:
        print(f"    TIC {int(r['ID']):>12d}  Tmag {float(r['Tmag']):6.2f}  "
              f"d = {float(r['dstArcSec']):4.2f}\"  "
              f"{'' if _tic_row_ok(r) else str(r['disposition'])}")
    by_id = {int(r["ID"]): r for r in rows}
    near = [r for r in rows if float(r["dstArcSec"]) <= 1.0]
    disp = lambda r: str(r["disposition"]).strip().upper()   # noqa: E731
    for r in near:
        if _tic_row_ok(r):
            print(f"  KIC {kic}: TIC {int(r['ID'])} (unflagged, at the KIC position)")
            return int(r["ID"])
    for r in near:
        if disp(r) == "DUPLICATE":
            try:
                dup = _masked_int(r["duplicate_id"])
            except (KeyError, ValueError, IndexError):
                dup = None
            if dup and dup in by_id and _tic_row_ok(by_id[dup]):
                print(f"  KIC {kic}: TIC {dup} (replacement of DUPLICATE "
                      f"TIC {int(r['ID'])})")
                return int(dup)
    for flag in ("DUPLICATE", "SPLIT"):
        for r in near:
            if disp(r) == flag:
                print(f"  ! KIC {kic}: only flagged TIC entries at the KIC "
                      f"position; using TIC {int(r['ID'])} ({flag}).  Other "
                      f"flagged entries are removed from the TRICERATOPS star "
                      f"table; check its parameters or give --stellar")
                return int(r["ID"])
    good = [r for r in rows if _tic_row_ok(r)]
    if not good:
        raise RuntimeError(f"no usable TIC star within 4 arcsec of KIC {kic}; "
                           f"give --tic")
    best = good[0]
    print(f"  ! KIC {kic}: no TIC star within 1\"; nearest unflagged TIC "
          f"{int(best['ID'])} at {float(best['dstArcSec']):.2f}\" may be a "
          f"companion - check, or give --tic")
    return int(best["ID"])


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    tic: int = 165501544
    # v4.2: "tess" or "kepler" (then kic is the Kepler Input Catalog number)
    mission: str = "tess"
    kic: Optional[int] = None
    # v4.2: host star for TRICERATOPS and stellar-parameter source
    host_tic: Optional[int] = None     # TRICERATOPS target if not the TIC
    star_source: str = "tic"           # "tic" or "kepler" (Berger 2020 / DR25)
    stellar_manual: Optional[tuple] = None   # (R, M, Teff[, logg])
    # None -> auto "TIC{tic}" under the current working directory.
    # Pass an explicit path (absolute or relative) to override.
    outdir: Optional[str] = None

    # --- cutout / aperture / light-curve source ---
    cutout_size: int = 10
    # Aperture: 'optimal' (v4: greedy S/N-maximising aperture built from a
    # catalogue PSF model of the field; less light of neighbours) or
    # 'percentile' (pixels above `percentile` of the median image).
    aperture_mode: str = "optimal"
    aperture_min_pix: int = 4         # never fewer pixels (pointing jitter)
    percentile: float = 85.0          # aperture = pixels above this percentile
    quality_bitmask: str = "default"  # lightkurve quality masking
    # Shared cache for extraction, neighbour vetting and TRICERATOPS cutouts.
    # Matching FITS bypass search/download; catalogue/coverage queries can
    # still contact MAST. None -> outdir/tesscut_cache.
    download_dir: Optional[str] = None
    # Photometry product: "tesscut" (FFI custom aperture), "sap", or "pdcsap".
    # For sap/pdcsap the matching SPOC TPF is still loaded when available
    # (apertures, centroids, difference imaging).
    lc_source: str = "tesscut"

    # --- detrending ---
    # First (blind) pass biweight window [d].  Must be LONG compared to the
    # transit duration so the filter cannot fit into the dip, but also NOT
    # comparable to the orbital period (window ≈ P lets the filter remove a
    # periodic transit).  Rule of thumb:
    #   * long-period candidates (P ≳ 10 d): 1–2 d is fine
    #   * short-period / hot Jupiters (P ≲ 3 d): use 5–10 d
    # For a blind search over a wide P range, 5 d is a safe compromise.
    window_length_search: float = 1.5   # [d] first-pass biweight window
    # Several first-pass windows: the blind search is run for each one and
    # the highest-SDE candidate is kept (all are listed in the log, with a
    # consistency check).  A shorter window removes red noise better but also
    # suppresses a long transit; the best compromise depends on the data and
    # is not known before the search.  None: use window_length_search only.
    # Cost: one TLS run per window.
    window_length_grid: Optional[tuple] = None
    window_factor: float = 3.0          # second pass: window = factor * T14
    window_fallback: float = 0.5        # [d] lower bound on any window
    detrend_method: str = "biweight"
    # Reject extreme outliers after detrending, in units of the robust RMS.
    # Positive spikes (flares / residual scattered light) are always clipped.
    # Negative outliers are clipped only *outside* a transit mask when an
    # ephemeris is available; in-transit negatives are kept so real dips are
    # not deleted.  Set to 0 / None to disable.
    outlier_sigma: float = 5.0
    # After each detrend pass, divide the sector LC by its out-of-transit
    # median so the continuum sits at 1.0 before stitching / folding.
    renorm_oot: bool = True
    # Reject cadences whose formal flux error exceeds this multiple of the
    # sector median (scattered light / bad-pixel blow-ups: errors of 10-15 %
    # on a 0.2 % LC).  They carry no information but distort plots and
    # error-based estimates.  0 / None disables.
    err_clip_factor: float = 5.0
    # On the phase-folded LC, before binning: two-sided MAD clip on OOT
    # points; in-transit only positive spikes (and fluxes < 10 % of continuum).
    fold_clip_sigma: float = 4.0
    # Use the per-bin *median* (not the mean) when building the binned fold.
    # Much more robust to residual ramps and single bad cadences.
    fold_robust_bin: bool = True
    # Sectors to drop before extraction (e.g. catastrophic normalisation).
    # Example: (56, 79).  None / empty = use all available sectors.
    exclude_sectors: Optional[tuple] = None
    # After detrending, automatically drop sectors whose normalised LC fails
    # basic quality cuts (failed normalisation / extreme scatter).
    auto_exclude_bad_sectors: bool = True
    # Robust RMS of norm_flux above this (fractional) => bad (0.02 = 2%).
    sector_max_rms: float = 0.02
    # |median(norm_flux) - 1| above this => continuum failed.
    sector_max_median_offset: float = 0.05
    # Fraction of cadences with |f-1| > sector_outlier_level => bad.
    sector_max_outlier_frac: float = 0.05
    sector_outlier_level: float = 0.05   # 5% from continuum
    # Scramble-based red-noise false-alarm test on the stacked depth.
    # 0 disables the test.  Typical values: 500–5000.
    red_noise_scrambles: int = 0
    red_noise_seed: int = 42
    # Secondary-eclipse search at *all* phases (eccentric orbits shift the
    # secondary away from phase 0.5 and change its duration).  Box widths
    # are secondary_dur_factors x T14; a detection must be significant
    # (stacked S/N >= secondary_min_snr), come from >= 2 events and survive
    # removal of the strongest event (S/N >= 0.5 x threshold).
    secondary_search: bool = True
    secondary_dur_factors: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)
    secondary_min_snr: float = 7.0
    # Eclipse-timing / depth-variation analysis (v4)
    etv_analysis: bool = True
    extra_times_file: Optional[str] = None   # ground-based mid-times
    # Ground-based follow-up photometry (v4.1)
    ground_specs: Optional[list] = None      # list of GroundSpec
    ground_tc_window_min: float = 60.0       # min half-width of the Tc search
    ground_min_snr: float = 5.0              # detection threshold
    ground_depth_tol: float = 0.25           # fractional tolerance on the
                                             # expected depth (dilution model,
                                             # band, limb darkening)
    ground_frac_sys: float = 0.03            # systematic on flux fractions
    ground_fwhm_default: float = 3.0         # [arcsec] if not given
    ground_clip_sigma: float = 5.0
    ground_clear: bool = True                # remove cleared stars
    tri_plus: str = "auto"                   # TRICERATOPS+: auto | on | off
    exclude_stars: Optional[tuple] = None    # TIC IDs cleared elsewhere

    # --- TLS ---
    # Broad defaults suitable for most TESS planets.  Narrow with --p-min /
    # --p-max when you already know the approximate period (much faster).
    # At runtime P_max is also clipped to baseline / n_transits_min.
    P_min: float = 0.5
    P_max: float = 100.0             # clipped to baseline/2 at runtime
    duration_grid_step: float = 1.1
    n_transits_min: int = 2
    SDE_threshold: float = 3.0
    tls_threads: Optional[int] = None
    tls_oversampling_factor: int = 2   # TLS default is 3; lower = faster, less precise T0/P

    # Bin the light curve before the TLS *search* only (speed). None disables
    # binning. Keep well below the shortest transit duration you expect to
    # find: a few points per hour of T14 is enough for detection; the
    # per-transit depth fit and difference imaging always use full-resolution
    # data regardless of this setting.
    tls_search_bin_minutes: Optional[float] = 10.0

    # --- per-transit depths ---
    n_dur_window: float = 3.0         # window around each Tc, in transit durations
    min_pts_per_transit: int = 6
    min_coverage: float = 0.8         # require the transit bottom to be sampled
    # Fit the mid-time of every individual transit (+- range x T14).  Needed
    # when P/T0 are uncertain: with a fixed Tc a shifted transit is fitted
    # too shallow or rejected.
    per_transit_fit_tc: bool = True
    per_transit_tc_range: float = 0.5
    per_transit_min_snr: float = 3.0     # transits used to refine P and T0
    # Refine P and T0 from the measured mid-times before folding.  Skipped
    # when the linear ephemeris fits poorly (TTVs / false detections).
    # With only a few transits (long-period TESS) chi2_red is noisy (dof=n-2).
    # 4.0 was too strict: a real 55 d planet with O-C rms ~5 min was rejected,
    # the fold stayed smeared, and TRICERATOPS FPP jumped from ~0.001 to ~0.1.
    # 8.0 still blocks gross TTVs / false alarms while allowing modest scatter.
    refine_from_transits: bool = True
    refine_chi2_max: float = 8.0
    # A refined ephemeris is adopted only if it predicts the mid-time of every
    # epoch with data to within this fraction of T14 (1 sigma).  A fit to a
    # cluster of transits (e.g. two adjacent sectors) is otherwise
    # extrapolated over the whole baseline.
    ephem_adopt_max_sigma: float = 0.25
    # Transits closer than this [d] to a data-segment edge are flagged and the
    # depth and odd/even tests are also reported without them.
    edge_transit_d: float = 0.5
    # The LD fit may overwrite the adopted depth / T14 only if its T0 shift
    # is below this fraction of T14 (a larger shift means the fold is smeared).
    ld_update_max_t0shift: float = 0.15
    # triceratops.remove_star drops rows of target.stars but NOT the matching
    # rows of target.pix_coords.  A calc_depths() call after remove_star then
    # pairs every remaining star with the pixel position of another star
    # (wrong flux ratios, spurious tdepth > 0, wrong NFPP).  The documented
    # sequence is calc_depths -> remove_star -> calc_probs, so the depths are
    # NOT recomputed by default; if requested they are recomputed with
    # pix_coords kept aligned.
    recalc_depths_after_removal: bool = False

    # --- TRICERATOPS ---
    n_iter: int = 20
    bin_minutes: float = 10.0         # binning of the folded LC fed to calc_probs
    fold_n_dur: float = 5.0           # phase window, in transit durations
    contrast_curve_file: Optional[str] = None
    contrast_filt: str = "TESS"
    parallel: bool = True
    search_radius: int = 10

    # --- robust flux uncertainty / FPP numerics ---
    # calc_probs gets ONE scalar sigma.  It is the largest of: white noise of
    # a bin (OOT point scatter / sqrt(n)), the empirical OOT scatter of the
    # binned LC (includes red noise) and err_floor_ppm.
    err_floor_ppm: float = 0.0
    # Scale sigma up so a smooth reference model (trapezoid) has
    # chi2_red = 1 on the binned LC.  Never scales down.  Protects against
    # phase-curve / ellipsoidal / detrending residuals at very high S/N.
    tri_inflate_to_model: bool = True
    tri_max_inflate: float = 50.0
    # If a calc_probs run is invalid (NaN / degenerate), retry with sigma
    # multiplied by tri_retry_factor, at most tri_max_retries times.
    tri_max_retries: int = 3
    tri_retry_factor: float = 2.0
    # Fraction of the n_iter runs that must be valid for a verdict.
    tri_min_valid_frac: float = 0.8
    # Extra calc_probs runs at sigma * scale (reported in the log).  Costs
    # len(scales) extra calc_probs calls.  A result that changes strongly with
    # the scale is set by the noise model, not by the data.
    tri_sensitivity: bool = False
    tri_sens_scales: tuple = (0.5, 1.0, 2.0, 4.0)
    tri_sens_runs: int = 3             # calc_probs runs per scale (pooled)
    # Number of Monte-Carlo draws per scenario in calc_probs (None: the
    # triceratops default, 1e6).  The evidences are Monte-Carlo estimates; when
    # the likelihood is narrow, single runs are noisy.
    tri_N: Optional[int] = None
    # Combine the evidences of all valid runs (log-mean-exp per scenario,
    # equivalent to one run with n_iter x N draws) and derive FPP/NFPP from
    # the pooled evidences instead of averaging per-run FPPs.
    pool_evidences: bool = True
    tri_boot: int = 300                # bootstrap resamples over runs
    # Convergence (v4): after n_iter runs, keep adding calc_probs runs while
    # one run holds more than tri_conv_max_share of the leading scenario's
    # pooled evidence, up to tri_max_iter runs or tri_time_budget seconds.
    tri_conv_max_share: float = 0.5
    tri_max_iter: Optional[int] = None    # None: 3 x n_iter
    tri_time_budget: float = 3600.0
    # Exposure smoothing of the TRICERATOPS models (v4).  None: the
    # effective exposure of the folded, binned light curve (bin width and
    # native cadences, as in the limb-darkened fit); 0: package default
    # (2 min, wrong for binned FFI data).
    tri_exptime_d: Optional[float] = None
    tri_nsamples: int = 20
    # Limb-darkened fit of the folded LC (replaces the trapezoid as the
    # reference model of the sigma inflation; adopted ephemeris unchanged).
    ld_fit: bool = False
    # 'claret': u1,u2 fixed to the Claret values from the TIC/TLS catalogue;
    # 'free'  : fit Kipping q1,q2;  'fixed': use ld_coeffs = (u1, u2).
    ld_mode: str = "claret"
    ld_coeffs: Optional[tuple] = None
    # None: keep triceratops' default (True).  False: the folded flux is an
    # un-corrected aperture LC (as built here from FFI cutouts), so
    # triceratops must apply the dilution itself.  Needs triceratops >= 1.1.
    dilution_corrected: Optional[bool] = None
    # v4: 'sector' converts every sector's detrended light curve to the
    # on-target scale, f -> 1 + (f - 1) / g_s, with g_s the target's flux
    # fraction in that sector's aperture (catalogue + Gaussian PSF, the
    # model triceratops uses); TRICERATOPS then gets dilution_corrected=True.
    # 'none' keeps the raw aperture flux (dilution_corrected=False).
    dilution_mode: str = "sector"
    field_radius_arcsec: float = 220.0   # catalogue radius for the PSF model
    # Optional pre-downloaded TRILEGAL table.  If None, triceratops queries
    # stev.oapd.inaf.it (needs working SSL).  Save the table once and reuse it
    # to avoid the network call on every run.
    trilegal_fname: Optional[str] = None

    # --- nearby star vetting (optional, see vet_nearby_stars) ---
    vet_neighbours: bool = False
    neighbour_max_sep: float = 200.0  # [arcsec] only test stars within this
    neighbour_min_sep: float = 42.0   # [arcsec] ~2 px; closer is not separable
    neighbour_ap_radius: float = 1.0  # [px] radius of the aperture on the neighbour
    neighbour_sigma: float = 3.0      # exclusion threshold
    psf_sigma: float = 0.75           # [px], same PSF model triceratops uses

    # --- difference imaging ---
    difference_imaging: bool = True
    di_in_frac: float = 0.7           # in-transit window, in units of T14
    di_out_gap: float = 1.2           # out-of-transit starts at gap * T14/2
    di_out_span: float = 3.0          # ... and ends at span * T14/2
    di_min_frames: int = 3            # min in-transit frames per epoch
    di_prf_sigma_init: float = 1.0    # [px] Gaussian approximation of the PRF
    di_prf_sigma_bounds: tuple = (0.4, 3.0)
    di_mc: int = 300                  # Monte Carlo draws for the centroid error
    di_sigma: float = 3.0             # exclusion threshold for neighbours
    di_min_snr: float = 3.0           # min SNR of the difference image to trust it
    # Systematic floor added in quadrature to the difference-image centroid
    # error (Gaussian approximation of the TESS PRF, pointing): without it a
    # 0.007 px statistical error turns a 1-2" PRF-model offset into 5-13 sigma.
    di_sys_floor_px: float = 0.1

    # --- known ephemeris (skip the TLS search) ---
    # Default is False: run a blind TLS search.  Set True only via --ephemeris
    # (or by editing these fields for a single-target notebook session).
    use_known_ephemeris: bool = False
    P_orb_known: float = 10.0
    T0_known: float = 1500.0         # BTJD
    depth_known: float = 1000e-6
    dur_known: float = 0.1           # [d]
    # Check / refine a known ephemeris *before* the transit-masked detrend.
    # A period that is off by only 0.01 d drifts by hours over a 5-year TESS
    # baseline; the transit mask, the per-transit fits (+- 0.5 T14) and the
    # fold then all miss the real transits.  Every expected transit is
    # searched with a box of width T14 within +- ephem_scan_range_d
    # (None: max(2 x T14, 0.5 d)) and a robust (pair-consensus) linear fit
    # of the mid-times gives P and T0.  Adopted when >= 3 transits agree.
    # Method: 'scan' (per-transit box search + robust linear fit), 'grid'
    # (folded box search on a small (P, T0) grid: stacks all transits, for
    # signals too shallow to see one by one), 'auto' (scan; grid when the
    # scan is inconclusive) or 'off'.
    ephem_search: str = "auto"
    ephem_scan: bool = True              # False == ephem_search 'off'
    ephem_scan_range_d: Optional[float] = None   # also the grid T0 half-range
    ephem_scan_min_snr: float = 5.0
    # Grid: P half-range [d]; None -> same drift budget as the T0 range
    # (dP = dT0 / epochs from the reference epoch).  Steps: T14/20 in T0 and
    # T14/20/n_epochs in P, so a run of ~100 x 100 points takes seconds.
    ephem_grid_dP: Optional[float] = None
    ephem_grid_min_snr: float = 7.0
    # MAST TESScut downloads fail now and then (502 / "Request failed").
    tesscut_retries: int = 3

    def __post_init__(self) -> None:
        # Default output directory is TIC{tic}/ so every target lands in its
        # own folder without the user having to pass --outdir each time.
        if not self.outdir:
            self.outdir = (f"KIC{self.kic}" if str(self.mission).lower() == "kepler"
                           and self.kic else f"TIC{self.tic}")
        self.outdir = os.path.abspath(self.outdir)
        os.makedirs(self.outdir, exist_ok=True)

    def path(self, name: str) -> str:
        os.makedirs(self.outdir, exist_ok=True)
        return os.path.join(self.outdir, name)

    def cache_dir(self) -> str:
        """Directory for TESScut FITS files (shared across runs of the same TIC).

        Cutouts are matched by target, sector and exact pixel dimensions.
        Different targets can safely share this directory.
        """
        d = self.download_dir if self.download_dir else os.path.join(
            self.outdir, "tesscut_cache")
        d = os.path.abspath(d)
        os.makedirs(d, exist_ok=True)
        return d


@dataclass
class SectorData:
    """Everything extracted from one TESS sector."""
    sector: int
    time: np.ndarray
    flux: np.ndarray            # background-corrected, un-normalised
    flux_err: np.ndarray
    ap_pixels: np.ndarray       # [[col, row], ...] in full-FFI coordinates
    cen_x: np.ndarray           # absolute centroid column
    cen_y: np.ndarray           # absolute centroid row
    norm_flux: np.ndarray = field(default=None)
    norm_err: np.ndarray = field(default=None)
    trend: np.ndarray = field(default=None)
    # kept for difference imaging (unfiltered, in the original frame order)
    cube: np.ndarray = field(default=None, repr=False)
    cube_time: np.ndarray = field(default=None, repr=False)
    wcs: object = field(default=None, repr=False)
    col0: int = 0
    row0: int = 0
    # v4: target's flux fraction in this sector's aperture (catalogue PSF
    # model) and how the aperture was chosen
    g_target: float = float("nan")
    aperture_kind: str = ""
    cadence_d: float = float("nan")


# --------------------------------------------------------------------------
# Field catalogue, PSF model, apertures and dilution (v4)
# --------------------------------------------------------------------------
def field_catalog(tic: int, ra: float, dec: float, cfg: "Config", log=None):
    """TIC stars within cfg.field_radius_arcsec, target in row 0.

    Positions are moved to epoch 2022 with the TIC proper motions.  The
    table is cached as field_TIC<tic>.csv in the output directory and reused
    (also offline).  Returns a DataFrame (ID, ra, dec, Tmag) or None.
    """
    path = cfg.path(f"field_TIC{tic}.csv")
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            if len(df) and int(df["ID"].iloc[0]) == int(tic):
                return df
        except Exception:
            pass
    try:
        tab = Catalogs.query_region(SkyCoord(ra, dec, unit="deg"),
                                    radius=cfg.field_radius_arcsec * u.arcsec,
                                    catalog="TIC")
    except Exception as exc:
        _log(log, f"  ! field catalogue query failed ({exc}); "
                  f"no PSF model of the field")
        return None
    df = tab.to_pandas()
    keep = np.isfinite(pd.to_numeric(df["Tmag"], errors="coerce"))
    if "disposition" in df.columns:
        disp = df["disposition"].astype(str).str.upper()
        keep &= ~disp.isin(["ARTIFACT", "DUPLICATE", "SPLIT"])
    if "objType" in df.columns:
        keep &= df["objType"].astype(str).str.upper().isin(["STAR", "NAN", "NONE", ""])
    df = df[keep].copy()
    dt = 22.0                                     # 2000 -> 2022 [yr]
    pmra = pd.to_numeric(df.get("pmRA"), errors="coerce").fillna(0.0).values
    pmde = pd.to_numeric(df.get("pmDEC"), errors="coerce").fillna(0.0).values
    dec_ = df["dec"].values.astype(float) + pmde * dt / 3.6e6
    ra_ = (df["ra"].values.astype(float)
           + pmra * dt / 3.6e6 / np.cos(np.radians(df["dec"].values.astype(float))))
    out = pd.DataFrame(dict(ID=df["ID"].astype(np.int64).values, ra=ra_, dec=dec_,
                            Tmag=df["Tmag"].astype(float).values))
    is_t = out["ID"].values == int(tic)
    if not is_t.any():
        _log(log, "  ! target not in the field catalogue; no PSF model")
        return None
    out = pd.concat([out[is_t], out[~is_t]]).reset_index(drop=True)
    try:
        out.to_csv(path, index=False)
    except Exception:
        pass
    return out


def star_pixels(field_df, wcs, col0, row0):
    """Catalogue positions in the frame of the apertures ([col, row])."""
    x, y = wcs.world_to_pixel(SkyCoord(field_df["ra"].values * u.deg,
                                       field_df["dec"].values * u.deg))
    return np.asarray(x, float) + col0, np.asarray(y, float) + row0


def psf_model_images(xs, ys, tmags, cols, rows, sigma):
    """(target image, total image) of the catalogue PSF model on a pixel grid."""
    flux = 10.0 ** (-0.4 * np.asarray(tmags, float))
    tot = np.zeros(cols.shape, float)
    tgt = None
    for k, (x, y, f) in enumerate(zip(xs, ys, flux)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if abs(x - cols.mean()) > np.ptp(cols) + 6 or abs(y - rows.mean()) > np.ptp(rows) + 6:
            continue
        img = f * _gauss_pixel_fraction(x, y, cols, rows, sigma)
        tot += img
        if k == 0:
            tgt = img
    return tgt, tot


def optimal_aperture(median_image, wcs, col0, row0, field_df, cfg: "Config"):
    """Greedy aperture that maximises the S/N of the *target's* flux.

    Each pixel's target flux = (median image - background) x model fraction
    of the target in that pixel; its variance ~ the median image (photon
    noise incl. sky).  Pixels are added in order of target S/N; the
    aperture with the highest cumulative S/N wins, never fewer than
    cfg.aperture_min_pix pixels.  Returns a boolean mask or None.
    """
    ny, nx = median_image.shape
    rows, cols = np.mgrid[row0:row0 + ny, col0:col0 + nx].astype(float)
    xs, ys = star_pixels(field_df, wcs, col0, row0)
    tgt, tot = psf_model_images(xs, ys, field_df["Tmag"].values, cols, rows,
                                cfg.psf_sigma)
    if tgt is None or not np.isfinite(median_image).any():
        return None
    frac = np.where(tot > 0, tgt / np.where(tot > 0, tot, 1.0), 0.0)
    img = np.where(np.isfinite(median_image), median_image, np.nan)
    bkg = float(np.nanpercentile(img, 10))
    sig = np.clip(img - bkg, 0.0, None) * frac
    var = np.clip(np.where(np.isfinite(img), img, np.nan), 1e-6, None)
    score = np.where(np.isfinite(var), sig / np.sqrt(var), -np.inf).ravel()
    order = np.argsort(score)[::-1]
    order = order[np.isfinite(score[order]) & (score[order] > 0)]
    if order.size == 0:
        return None
    csig = np.cumsum(sig.ravel()[order])
    cvar = np.cumsum(var.ravel()[order])
    snr = csig / np.sqrt(cvar)
    n = max(int(np.argmax(snr)) + 1, int(cfg.aperture_min_pix))
    n = min(n, order.size)
    mask = np.zeros(ny * nx, bool)
    mask[order[:n]] = True
    return mask.reshape(ny, nx)


def target_fraction(ap_pixels, wcs, col0, row0, field_df, sigma) -> float:
    """Target's share of the catalogue-model flux in an aperture."""
    if field_df is None or ap_pixels is None or len(ap_pixels) == 0:
        return float("nan")
    xs, ys = star_pixels(field_df, wcs, col0, row0)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok[0]:
        return float("nan")
    g = aperture_flux_fractions(xs[ok], ys[ok], field_df["Tmag"].values[ok],
                                np.asarray(ap_pixels), sigma=sigma)
    return float(g[0])


def apply_dilution_correction(sd: "SectorData", cfg: "Config") -> None:
    """Detrended sector LC -> on-target scale: f -> 1 + (f - 1) / g_s."""
    if (getattr(cfg, "dilution_mode", "none") != "sector"
            or sd.norm_flux is None):
        return
    if getattr(sd, "_dilution_done", False):
        return                          # never twice on the same detrend
    g = float(getattr(sd, "g_target", np.nan))
    if not (np.isfinite(g) and 0.02 < g <= 1.0):
        return
    sd.norm_flux = 1.0 + (np.asarray(sd.norm_flux, float) - 1.0) / g
    if sd.norm_err is not None:
        sd.norm_err = np.asarray(sd.norm_err, float) / g
    sd._dilution_done = True


# --------------------------------------------------------------------------
# Stellar parameters
# --------------------------------------------------------------------------
def stellar_params(tic: int, kic: Optional[int] = None) -> dict:
    """Query the TIC by *column name* rather than by positional index."""
    tab = Catalogs.query_criteria(catalog="Tic", ID=tic)
    if len(tab) == 0:
        raise ValueError(f"TIC {tic} not found")
    row = tab[0]

    def val(key, default=np.nan):
        try:
            v = float(row[key])
            return v if np.isfinite(v) else default
        except (KeyError, TypeError, ValueError):
            return default

    rad, e_rad = val("rad"), val("e_rad", 0.0)
    mass, e_mass = val("mass"), val("e_mass", 0.0)

    p = dict(
        Teff=val("Teff"),
        logg=val("logg"),
        metal=val("MH", 0.0),
        rad=rad,
        rad_min=rad - e_rad,
        rad_max=rad + e_rad,
        mass=mass,
        mass_min=mass - e_mass,
        mass_max=mass + e_mass,
        Tmag=val("Tmag"),
        ra=val("ra"),
        dec=val("dec"),
    )

    # Limb darkening from the TLS catalogue helper (Claret tables).
    try:
        from transitleastsquares import catalog_info
        if is_kepler() and kic:
            ab, *_ = catalog_info(KIC_ID=int(kic))      # Kepler band
        else:
            ab, *_ = catalog_info(TIC_ID=tic)
        p["u"] = list(ab)
    except Exception as exc:                                  # pragma: no cover
        print(f"  ! catalog_info failed ({exc}); using u=[0.4, 0.3]")
        p["u"] = [0.4, 0.3]

    return p


def _vizier_pick(row, names):
    """First finite value among candidate column names (case-insensitive)."""
    names_all = list(row.colnames) if hasattr(row, "colnames") else list(row.keys())
    lower = {c.lower(): c for c in names_all}
    for n in names:
        # exact name first: E_Rad (upper error) and e_Rad (lower) differ in case
        c = n if n in names_all else lower.get(n.lower())
        if c is None:
            continue
        try:
            v = row[c]
            if np.ma.is_masked(v):
                continue
            v = float(v)
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return np.nan


def _sym_err(row, base):
    up = _vizier_pick(row, [f"E_{base}", f"e_{base}_up", f"{base}_err1"])
    lo = _vizier_pick(row, [f"e_{base}", f"e_{base}_lo", f"{base}_err2"])
    vals = [abs(x) for x in (up, lo) if np.isfinite(x)]
    return float(np.mean(vals)) if vals else 0.0


KEPLER_STAR_CATALOGS = (
    # (Vizier catalogue, description)
    ("J/AJ/159/280/table2", "Berger et al. 2020 (Gaia-Kepler stellar properties)"),
    ("J/ApJS/229/30/catalog", "Mathur et al. 2017 (Kepler DR25 stellar table)"),
)


VIZIER_MIRRORS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu",
                  "vizier.iucaa.in", "vizier.nao.ac.jp")
_VIZIER_HOST = None          # chosen mirror; "" = none reachable
_VIZIER_PATCHED = False


def setup_vizier(log=None) -> Optional[str]:
    """Pick the first reachable VizieR mirror and make every astroquery
    Vizier query use it (TLS limb darkening, Kepler catalogues, TRICERATOPS).
    Returns the host or None."""
    global _VIZIER_HOST, _VIZIER_PATCHED
    if _VIZIER_HOST is not None:
        return _VIZIER_HOST or None
    import requests
    host = ""
    for h in VIZIER_MIRRORS:
        try:
            r = requests.get(f"https://{h}/viz-bin/votable",
                             params={"-source": "I/239/hip_main", "-out.max": 1,
                                     "HIP": 1}, timeout=8)
            if r.status_code == 200:
                host = h
                break
        except Exception:
            continue
    _VIZIER_HOST = host
    try:
        from astroquery.vizier import conf as _vconf
        from astroquery.vizier.core import VizierClass
        if host:
            _vconf.server = host
            if not _VIZIER_PATCHED:
                _orig = VizierClass._server_to_url

                def _server_to_url(self, *a, **kw):
                    if _VIZIER_HOST:
                        self.VIZIER_SERVER = _VIZIER_HOST
                    return _orig(self, *a, **kw)
                VizierClass._server_to_url = _server_to_url
                _VIZIER_PATCHED = True
        _vconf.timeout = 20
    except Exception:
        pass
    if host:
        if host != VIZIER_MIRRORS[0]:
            _log(log, f"VizieR: {VIZIER_MIRRORS[0]} not reachable, using the "
                      f"mirror {host}")
    else:
        _log(log, "! VizieR not reachable (all mirrors): Kepler catalogue "
                  "parameters and Kepler-band limb darkening are skipped")
    return host or None


@contextmanager
def _kic_position_offline(kic, ra, dec):
    """TRICERATOPS (mission='Kepler', no ra/dec argument) asks VizieR for the
    KIC position; answer that one query locally."""
    try:
        from astroquery.vizier.core import VizierClass
    except Exception:
        yield
        return
    orig = VizierClass.query_constraints

    class _Res:
        def __init__(self):
            self._a = np.array([(float(ra), float(dec))],
                               dtype=[("_RA", float), ("_DE", float)])

        def as_array(self):
            return self._a

    def query_constraints(self, *a, **kw):
        if (str(kw.get("catalog", "")) == "J/ApJS/229/30/catalog"
                and str(kw.get("KIC", "")) == str(int(kic))):
            return [_Res()]
        return orig(self, *a, **kw)
    VizierClass.query_constraints = query_constraints
    try:
        yield
    finally:
        VizierClass.query_constraints = orig


def kepler_stellar_params(kic: int, log=None) -> Optional[dict]:
    """Radius, mass, Teff, logg, [Fe/H] of a KIC star from Berger et al. 2020,
    else the DR25 stellar table (Mathur et al. 2017).  None if both fail."""
    try:
        from astroquery.vizier import Vizier
    except Exception as exc:
        _log(log, f"  ! astroquery.vizier unavailable ({exc})")
        return None
    if setup_vizier(log) is None:
        return None
    for cat, desc in KEPLER_STAR_CATALOGS:
        try:
            res = Vizier(columns=["**"], row_limit=5).query_constraints(
                KIC=str(int(kic)), catalog=cat)
        except Exception as exc:
            _log(log, f"  ! {desc}: query failed ({exc})")
            continue
        if not len(res) or not len(res[0]):
            _log(log, f"  ! {desc}: KIC {kic} not found")
            continue
        row = res[0][0]
        out = dict(
            rad=_vizier_pick(row, ["Rad", "R", "iso_rad", "Radius"]),
            mass=_vizier_pick(row, ["Mass", "M", "iso_mass"]),
            Teff=_vizier_pick(row, ["Teff", "iso_teff"]),
            logg=_vizier_pick(row, ["logg", "iso_logg"]),
            metal=_vizier_pick(row, ["__Fe_H_", "[Fe/H]", "FeH", "iso_feh"]),
        )
        if not (np.isfinite(out["rad"]) and np.isfinite(out["mass"])):
            _log(log, f"  ! {desc}: no radius / mass for KIC {kic}")
            continue
        e_r = _sym_err(row, "Rad") or _sym_err(row, "R")
        e_m = _sym_err(row, "Mass") or _sym_err(row, "M")
        out.update(rad_min=out["rad"] - e_r, rad_max=out["rad"] + e_r,
                   mass_min=out["mass"] - e_m, mass_max=out["mass"] + e_m,
                   source=f"{desc}, Vizier {cat}")
        return out
    return None


def apply_star_overrides(star: dict, cfg, tic: int, log=None) -> dict:
    """Host-star parameters: --host-tic, then --star-source, then --stellar."""
    star = dict(star)
    star["source"] = "TIC"
    host = getattr(cfg, "host_tic", None)
    if host and int(host) != int(tic):
        hs = stellar_params(int(host))
        _log(log, f"Host star for TRICERATOPS: TIC {host} (the light curve "
                  f"stays that of TIC {tic}); its TIC parameters are used")
        star.update(hs)
        star["source"] = f"TIC {host}"
    src = (getattr(cfg, "star_source", "tic") or "tic").lower()
    if src == "kepler":
        if host and int(host) != int(tic):
            _log(log, "  ! --star-source kepler ignored: the Kepler catalogues "
                      "describe the KIC star, not the --host-tic star")
        elif not getattr(cfg, "kic", None):
            _log(log, "  ! --star-source kepler needs --kic: TIC parameters kept")
        else:
            kp = kepler_stellar_params(cfg.kic, log=log)
            if kp is None:
                _log(log, "  ! no Kepler catalogue parameters: TIC values kept")
            else:
                src_txt = kp.pop("source")
                star.update({k: v for k, v in kp.items() if np.isfinite(v)})
                star["source"] = src_txt
    man = getattr(cfg, "stellar_manual", None)
    if not man and not (np.isfinite(star.get("rad", np.nan))
                        and np.isfinite(star.get("mass", np.nan))):
        teff = star.get("Teff", np.nan)
        if np.isfinite(teff) and 2800 < teff < 7500:
            # main-sequence scaling (L ~ M^4, R ~ M^0.8 -> M ~ Teff^1.67)
            M = float((teff / 5772.0) ** 1.67)
            R = float(M ** 0.8)
            star.update(mass=M, mass_min=0.8 * M, mass_max=1.2 * M,
                        rad=R, rad_min=0.8 * R, rad_max=1.2 * R)
            star["source"] = (f"{star.get('source', 'TIC')}; R, M missing - "
                              f"estimated from Teff = {teff:.0f} K "
                              f"(main sequence, +-20 %)")
            _log(log, f"  ! no radius / mass in the catalogue: main-sequence "
                      f"estimate R = {R:.2f} Rsun, M = {M:.2f} Msun from Teff "
                      f"{teff:.0f} K (better: --stellar or --star-source kepler)")
        else:
            raise RuntimeError("no stellar radius / mass / Teff for the host "
                               "star: give --stellar R M Teff")
    if man:
        R, M, Teff = (float(x) for x in man[:3])
        star.update(rad=R, rad_min=R, rad_max=R, mass=M, mass_min=M,
                    mass_max=M, Teff=Teff)
        if len(man) > 3:
            star["logg"] = float(man[3])
        star["source"] = "manual (--stellar)"
    return star


def find_tess_sectors(tic: int, cfg: Optional[Config] = None) -> np.ndarray:
    """Sectors with photometry for this TIC (FFI cutouts or SPOC LC).

    When ``cfg.lc_source`` is ``sap`` / ``pdcsap``, returns sectors that have
    a SPOC light curve. Otherwise returns TESScut FFI coverage.
    """
    if is_kepler():
        return _find_kepler_quarters(cfg)
    src = (getattr(cfg, "lc_source", "tesscut") or "tesscut").lower() if cfg else "tesscut"
    if src in ("sap", "pdcsap"):
        return _find_spoc_lc_sectors(tic, cfg)

    if cfg is not None:
        found = set()
        for cache in _tesscut_cache_dirs(cfg):
            for sec, paths in _build_tesscut_index(cache).items():
                if any(_cached_tesscut_matches(p, tic, sec, None, cfg)
                       for p in paths):
                    found.add(int(sec))
    else:
        found = set()
    try:
        search = lk.search_tesscut(f"TIC {tic}")
        if len(search) == 0:
            raise RuntimeError(f"No TESScut data for TIC {tic}")
        return np.array(sorted({int(s) for s in search.table["sequence_number"]}))
    except Exception:
        if not found:
            raise
        print(f"  ! coverage query failed; using cached sectors: {sorted(found)}")
        return np.array(sorted(found))


def _kepler_search(kind: str, cfg, quarter=None):
    """lightkurve search for Kepler long-cadence light curves / TPFs."""
    func = lk.search_lightcurve if kind == "lc" else lk.search_targetpixelfile
    kw = dict(mission="Kepler", author="Kepler", exptime="long")
    if quarter is not None:
        kw["quarter"] = int(quarter)
    try:
        return func(_mission_id(None, cfg), **kw)
    except TypeError:                        # older lightkurve: cadence=
        kw.pop("exptime")
        kw["cadence"] = "long"
        return func(_mission_id(None, cfg), **kw)


def _quarters_of(search) -> list:
    import re
    out = []
    tab = search.table
    for i in range(len(search)):
        q = None
        for key in ("sequence_number", "quarter"):
            if key in tab.colnames:
                q = _masked_int(tab[key][i])
                if q is not None:
                    break
        if q is None:
            if "mission" in tab.colnames:
                mis = str(tab["mission"][i])
            else:
                try:
                    mis = str(list(search.mission)[i])
                except Exception:
                    mis = ""
            m = re.search(r"Quarter\s*(\d+)", mis)
            q = int(m.group(1)) if m else None
        out.append(q)
    return out


def _find_kepler_quarters(cfg) -> np.ndarray:
    """Kepler quarters with a long-cadence light curve for this KIC."""
    search = _kepler_search("lc", cfg)
    if len(search) == 0:
        raise RuntimeError(f"No Kepler long-cadence light curve for "
                           f"{_mission_id(None, cfg)}")
    qs = sorted({q for q in _quarters_of(search) if q is not None})
    if not qs:
        raise RuntimeError("Kepler search returned products but no quarters")
    return np.array(qs)


def _find_spoc_lc_sectors(tic: int, cfg: Optional[Config] = None) -> np.ndarray:
    """Sectors with a SPOC (SAP/PDCSAP) light curve for this TIC."""
    search = lk.search_lightcurve(f"TIC {tic}", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(f"TIC {tic}")
    if len(search) == 0:
        raise RuntimeError(f"No SPOC light curve for TIC {tic}")
    secs = set()
    tab = search.table
    for row in tab:
        for key in ("sequence_number", "sector", "SECTOR"):
            if key in tab.colnames:
                try:
                    val = row[key]
                    if val is not np.ma.masked and val is not None:
                        secs.add(int(val))
                        break
                except (TypeError, ValueError):
                    pass
    if not secs:
        try:
            secs = {int(s) for s in np.atleast_1d(search.sector)}
        except Exception as exc:
            raise RuntimeError(
                f"SPOC search returned products but no sector numbers for TIC {tic}"
            ) from exc
    return np.array(sorted(secs))


# --------------------------------------------------------------------------
# Light curve extraction
# --------------------------------------------------------------------------
# cache_dir -> {sector: [paths]}; built once per process
_TESSCUT_INDEX: dict = {}


def _tesscut_cache_dirs(cfg):
    """Configured cache first; also reuse existing default/legacy LK caches.

    New downloads always go to cfg.cache_dir(); old files are read in place.
    """
    roots = [cfg.cache_dir()]
    config = getattr(lk, "config", None)
    get_cache_dir = getattr(config, "get_cache_dir", None)
    if get_cache_dir is not None:
        roots.append(get_cache_dir())
    roots.extend((os.path.expanduser("~/.lightkurve/cache"),
                  os.path.expanduser("~/.lightkurve-cache")))
    result = []
    for root in roots:
        root = os.path.abspath(root)
        if root not in result and os.path.isdir(root):
            result.append(root)
    return result


def _tesscut_fits_paths(cache_dir: str):
    """All FITS paths under the cache directory (any depth)."""
    if not os.path.isdir(cache_dir):
        return []
    roots = []
    sub = os.path.join(cache_dir, "tesscut")
    if os.path.isdir(sub):
        roots.append(sub)
    roots.append(cache_dir)
    out, seen = [], set()
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith((".fits", ".fits.gz")):
                    p = os.path.join(dirpath, name)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
    return out


def _sector_from_fits_header(path: str) -> Optional[int]:
    """Read SECTOR from a FITS header (primary or first extension)."""
    try:
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                for key in ("SECTOR", "SECTORID", "SECTR"):
                    if key in hdu.header:
                        try:
                            return int(hdu.header[key])
                        except (TypeError, ValueError):
                            pass
    except Exception:
        return None
    return None


def _sector_from_filename(path: str) -> Optional[int]:
    """Best-effort sector number from a TESScut / lightkurve filename."""
    import re
    low = os.path.basename(path).lower()
    for pat in (r"(?:^|[_\-])s(\d{1,4})(?:[_\-]|$)",
                r"sector[_-]?(\d{1,3})",
                r"-s(\d{4})-"):
        m = re.search(pat, low)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _build_tesscut_index(cache_dir: str) -> dict:
    """Map sector -> list of FITS paths. Built once; uses headers if needed."""
    cache_dir = os.path.abspath(cache_dir)
    if cache_dir in _TESSCUT_INDEX:
        return _TESSCUT_INDEX[cache_dir]

    paths = _tesscut_fits_paths(cache_dir)
    index: dict = {}
    n_hdr = 0
    for path in paths:
        sec = _sector_from_fits_header(path)
        if sec is not None:
            n_hdr += 1
        if sec is None:
            sec = _sector_from_filename(path)
        if sec is None:
            continue
        index.setdefault(int(sec), []).append(path)

    _TESSCUT_INDEX[cache_dir] = index
    n_files = sum(len(v) for v in index.values())
    print(f"  TESScut cache index: {n_files} FITS in {cache_dir}")
    print(f"    sectors: {sorted(index.keys()) or '(none)'}"
          f"{f'  ({n_hdr} matched via header)' if n_hdr else ''}")
    if paths and not index:
        sample = [os.path.basename(p) for p in paths[:8]]
        print(f"  ! {len(paths)} FITS present but sector could not be parsed.")
        print(f"    sample: {sample}")
    return index


def _cached_tesscut_matches(path, tic, sector, cutout_size, cfg):
    """Require a readable FITS with the right target, sector and FLUX shape.

    TESScut filenames normally identify coordinates rather than a TIC ID.
    Legacy cutouts are checked via their FITS headers or coordinate filename;
    no match is inferred merely from sharing a cache directory.
    """
    import re
    from astropy.io import fits

    try:
        with fits.open(path, memmap=True) as hdul:
            # A target pixel file must have a FLUX table, not just an image.
            flux_hdu = next(h for h in hdul
                            if getattr(h, "columns", None) is not None
                            and "FLUX" in h.columns.names)
            dim = flux_hdu.columns["FLUX"].dim
            dims = tuple(int(x) for x in re.findall(r"\d+", dim or ""))
            if len(dims) != 2:
                return False
            if cutout_size is not None and dims != (int(cutout_size),) * 2:
                return False
            header_sector = None
            target_ids, coord_pairs = [], []
            for hdu in hdul:
                hdr = hdu.header
                for key in ("SECTOR", "SECTORID", "SECTR"):
                    if key in hdr:
                        header_sector = int(hdr[key])
                        break
                for key in ("TICID", "TIC_ID", "TARGETID", "OBJECT", "TARGNAME"):
                    value = str(hdr.get(key, "")).strip()
                    match = re.fullmatch(r"(?:TIC[\s_-]*)?(\d+)", value,
                                         flags=re.IGNORECASE)
                    if match:
                        target_ids.append(int(match.group(1)))
                for ra_key, dec_key in (("RA_OBJ", "DEC_OBJ"),
                                        ("RA_TARG", "DEC_TARG")):
                    if ra_key in hdr and dec_key in hdr:
                        coord_pairs.append((float(hdr[ra_key]),
                                            float(hdr[dec_key])))
            if header_sector is not None and header_sector != int(sector):
                return False
            if target_ids:
                return all(value == int(tic) for value in target_ids)

            # Astrocut names: <sectorName>_<ra>_<dec>_<NxM>_astrocut.fits.
            match = re.search(
                r"_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_"
                r"\d+x\d+_astrocut\.fits(?:\.gz)?$",
                os.path.basename(path), flags=re.IGNORECASE)
            if not coord_pairs and match:
                coord_pairs.append((float(match.group(1)), float(match.group(2))))
            if not coord_pairs:
                return False
            coords = _tesscut_target_coord(tic, cfg)
            # Sub-pixel tolerance, not a whole TESS pixel: do not reuse a
            # neighbouring target's cutout in a shared cache.
            return all(coords.separation(SkyCoord(ra, dec, unit="deg")).arcsec
                       <= 1.0 for ra, dec in coord_pairs)
    except (OSError, ValueError, TypeError, StopIteration, KeyError):
        return False


def _tesscut_target_coord(tic, cfg):
    """Memoize coordinates; main() normally already obtained them from TIC."""
    known = getattr(cfg, "_tesscut_coords", None)
    if known is None or known[0] != int(tic):
        tab = Catalogs.query_criteria(catalog="Tic", ID=tic)
        if len(tab) == 0:
            raise ValueError(f"TIC {tic} not found")
        coord = SkyCoord(float(tab[0]["ra"]), float(tab[0]["dec"]), unit="deg")
        cfg._tesscut_coords = (int(tic), coord)
    return cfg._tesscut_coords[1]


def _find_cached_tesscut(cache_dir: str, sector: int,
                         cutout_size: Optional[int] = None, *,
                         tic: int, cfg: Config) -> Optional[str]:
    """Return an exact target/sector/size match, never an arbitrary sector hit."""
    index = _build_tesscut_index(cache_dir)
    hits = list(index.get(int(sector), []))
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for path in hits:
        if _cached_tesscut_matches(path, tic, sector, cutout_size, cfg):
            return path
    return None


def _invalidate_tesscut_index(cache_dir: str) -> None:
    _TESSCUT_INDEX.pop(os.path.abspath(cache_dir), None)


def _load_tesscut_tpf(tic: int, sector: int, cutout_size: int, cfg: Config,
                      *, search_func=None):
    """Open TESScut from cache if present; otherwise search+download from MAST.

    A cache hit never calls ``search_tesscut``, so lightkurve will not print
    ``Getting TessCut for sector N`` and there is no network wait.
    Kepler mode: the quarter's target pixel file (there are no FFI cutouts).
    """
    if is_kepler():
        tpf = _load_spoc_tpf(tic, sector, cfg)
        if tpf is None:
            raise RuntimeError(f"no Kepler TPF for quarter {sector}")
        return tpf
    cache = cfg.cache_dir()
    cached = None
    for root in _tesscut_cache_dirs(cfg):
        cached = _find_cached_tesscut(root, sector, cutout_size, tic=tic, cfg=cfg)
        if cached is not None:
            break
    if cached is not None:
        print(f"  TESScut sector {sector}  (cache hit: {os.path.basename(cached)})")
        try:
            return lk.TessTargetPixelFile(cached,
                                          quality_bitmask=cfg.quality_bitmask)
        except TypeError:
            return lk.TessTargetPixelFile(cached)

    print(f"  TESScut sector {sector}  (not in cache → download to {cache})")
    import time as _time
    n_try = max(int(getattr(cfg, "tesscut_retries", 1) or 1), 1)
    last_exc = None
    for attempt in range(1, n_try + 1):
        try:
            search = (search_func or lk.search_tesscut)(f"TIC {tic}",
                                                        sector=sector)
            if len(search) == 0:
                raise RuntimeError(
                    f"No TESScut product for TIC {tic} sector {sector}")
            tpf = search.download(cutout_size=cutout_size,
                                  quality_bitmask=cfg.quality_bitmask,
                                  download_dir=cache)
            if tpf is None:
                raise RuntimeError("download returned None")
            _invalidate_tesscut_index(cache)
            return tpf
        except Exception as exc:
            last_exc = exc
            if "No TESScut product" in str(exc) or attempt == n_try:
                break
            wait = 15 * attempt
            print(f"    download attempt {attempt}/{n_try} failed "
                  f"({str(exc)[:80]}); retrying in {wait} s")
            _time.sleep(wait)
    _invalidate_tesscut_index(cache)
    raise last_exc


@contextmanager
def _triceratops_tesscut_cache(tic, cfg):
    """Route TRICERATOPS' internal TESS requests through the same cache.

    Some versions ignore lightkurve_cache_dir for TESS. The temporary adapter
    supports their search_tesscut(...).download_all(...) call without changing
    the installed package. This scope is for serial target construction only.
    """
    from unittest.mock import patch
    from astropy.wcs import WCS

    original_search = lk.search_tesscut
    cfg._tri_loaded_sectors = []

    def _fix_pixel_frame(tpf, sector):
        """Give TRICERATOPS' cutout the same pixel frame as our apertures.

        Astrocut >= 1.4 no longer writes the physical-pixel keywords
        1CRV4P/2CRV4P (TRICERATOPS reads them -> KeyError, sector dropped)
        and lightkurve's tpf.column/row fall back to 0.  The apertures are
        built in [tpf.column + i, tpf.row + j] of the *extraction* cutout
        (10x10), TRICERATOPS uses its own (larger) cutout; the offset
        between both frames follows from their WCS.  Setting 1CRV4P/2CRV4P
        to that offset makes pix_coords and all_ap_pixels consistent,
        whatever astrocut version produced either file.
        """
        hdr = tpf.hdu[1].header
        frame = getattr(cfg, "_sector_frames", {}).get(int(sector))
        if frame is not None and frame[0] is not None:
            wcs_s, c0, r0 = frame
            coord = _tesscut_target_coord(tic, cfg)
            xs, ys = (float(v) for v in wcs_s.world_to_pixel(coord))
            xt, yt = (float(v) for v in
                      WCS(tpf.hdu[2].header).world_to_pixel(coord))
            dc, dr = c0 + xs - xt, r0 + ys - yt
            if max(abs(dc - round(dc)), abs(dr - round(dr))) > 0.25:
                print(f"  ! sector {sector}: cutout frames differ by a "
                      f"non-integer offset ({dc:.2f}, {dr:.2f}) px")
            hdr["1CRV4P"] = int(round(dc))
            hdr["2CRV4P"] = int(round(dr))
        elif "1CRV4P" not in hdr or "2CRV4P" not in hdr:
            hdr["1CRV4P"] = int(tpf.column)
            hdr["2CRV4P"] = int(tpf.row)
        return tpf

    class CachedSearch:
        def download_all(self, cutout_size=5, **kwargs):
            size = cutout_size
            if isinstance(size, (tuple, list)):
                if len(size) != 2 or size[0] != size[1]:
                    raise ValueError("TRICERATOPS cache requires square cutouts")
                size = size[0]
            if kwargs.get("quality_bitmask", cfg.quality_bitmask) != cfg.quality_bitmask:
                raise ValueError("Unexpected TRICERATOPS quality_bitmask")
            tpf = _load_tesscut_tpf(tic, self.sector, int(size), cfg,
                                    search_func=original_search)
            tpf = _fix_pixel_frame(tpf, self.sector)
            cfg._tri_loaded_sectors.append(int(self.sector))
            return lk.TargetPixelFileCollection([tpf])

        def download(self, **kwargs):
            return self.download_all(**kwargs)[0]

    def cached_search(target, sector=None, **kwargs):
        if sector is None:
            return original_search(target, sector=sector, **kwargs)
        if isinstance(target, SkyCoord):
            same_target = target.separation(_tesscut_target_coord(tic, cfg)).arcsec <= 1.0
        else:
            same_target = str(target).replace(" ", "").upper() == f"TIC{tic}"
        if not same_target or kwargs:
            raise ValueError("Unexpected TRICERATOPS TESScut search request")
        result = CachedSearch()
        result.sector = int(sector)
        return result

    # TRICERATOPS+ (and triceratops 1.0.x) call astroquery's
    # Tesscut.get_cutouts directly and read 1CRV4P/2CRV4P from the result:
    # serve those requests from the same cache, with the same frame fix.
    _tc = getattr(tr, "Tesscut", None)
    original_get = getattr(_tc, "get_cutouts", None) if _tc is not None else None

    def cached_get_cutouts(*args, coordinates=None, size=5, sector=None, **kw):
        coord = coordinates if coordinates is not None else (args[0] if args else None)
        try:
            same = (coord is not None and sector is not None and
                    SkyCoord(coord).separation(
                        _tesscut_target_coord(tic, cfg)).arcsec <= 1.0)
        except Exception:
            same = False
        if same:
            sz = size[0] if isinstance(size, (tuple, list)) else size
            try:
                tpf = _load_tesscut_tpf(tic, int(sector), int(sz), cfg,
                                        search_func=original_search)
                tpf = _fix_pixel_frame(tpf, int(sector))
                cfg._tri_loaded_sectors.append(int(sector))
                return [tpf.hdu]
            except Exception as exc:
                print(f"  TESScut cache not used for sector {sector} ({exc})")
        res = original_get(*args, coordinates=coordinates, size=size,
                           sector=sector, **kw)
        out = []
        for h in res:
            if sector is not None:
                _fix_pixel_frame(SimpleNamespace(hdu=h, column=0, row=0),
                                 int(sector))
            out.append(h)
        return out

    with patch.object(lk, "search_tesscut", cached_search):
        if original_get is not None:
            with patch.object(_tc, "get_cutouts", cached_get_cutouts):
                yield
        else:
            yield


def extract_sector(tic: int, sector: int, cfg: Config) -> SectorData:
    """Dispatch to TESScut or SPOC (SAP/PDCSAP) extraction."""
    src = (getattr(cfg, "lc_source", "tesscut") or "tesscut").lower()
    if is_kepler() and src not in ("sap", "pdcsap"):
        src = "pdcsap"
    if src in ("sap", "pdcsap"):
        return extract_sector_spoc(tic, sector, cfg, flux_column=src)
    return extract_sector_tesscut(tic, sector, cfg)


def _load_spoc_lightcurve(tic: int, sector: int, cfg: Config):
    """Download / open a SPOC light curve for one sector."""
    cache = cfg.cache_dir()
    if is_kepler():
        search = _kepler_search("lc", cfg, quarter=sector)
        if len(search) == 0:
            raise RuntimeError(f"No Kepler light curve for "
                               f"{_mission_id(tic, cfg)} quarter {sector}")
        print(f"  Kepler LC quarter {sector}  -> download/cache in {cache}")
        lc = search[0].download(download_dir=cache,
                                quality_bitmask=cfg.quality_bitmask)
        try:
            lc = lc.remove_nans()
        except Exception:
            pass
        return lc
    search = lk.search_lightcurve(f"TIC {tic}", sector=sector, author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(f"TIC {tic}", sector=sector)
    if len(search) == 0:
        raise RuntimeError(f"No SPOC light curve for TIC {tic} sector {sector}")
    try:
        exptime = np.asarray(
            search.exptime.value if hasattr(search.exptime, "value")
            else search.exptime, dtype=float)
        search = search[np.argsort(exptime)]
    except Exception:
        pass
    print(f"  SPOC LC sector {sector}  -> download/cache in {cache}")
    lc = search[0].download(download_dir=cache,
                            quality_bitmask=cfg.quality_bitmask)
    try:
        lc = lc.remove_nans()
    except Exception:
        pass
    return lc


def _load_spoc_tpf(tic: int, sector: int, cfg: Config):
    """Optional SPOC Target Pixel File for aperture / difference imaging."""
    cache = cfg.cache_dir()
    if is_kepler():
        try:
            search = _kepler_search("tpf", cfg, quarter=sector)
            if len(search) == 0:
                return None
            print(f"  Kepler TPF quarter {sector}  -> download/cache in {cache}")
            return search[0].download(download_dir=cache,
                                      quality_bitmask=cfg.quality_bitmask)
        except Exception as exc:
            print(f"  ! no Kepler TPF for quarter {sector}: {exc}")
            return None
    try:
        search = lk.search_targetpixelfile(
            f"TIC {tic}", sector=sector, author="SPOC")
        if len(search) == 0:
            search = lk.search_targetpixelfile(f"TIC {tic}", sector=sector)
        if len(search) == 0:
            return None
        print(f"  SPOC TPF sector {sector}  -> download/cache in {cache}")
        return search[0].download(download_dir=cache,
                                  quality_bitmask=cfg.quality_bitmask)
    except Exception as exc:
        print(f"  ! no SPOC TPF for sector {sector}: {exc}")
        return None


def _ap_pixels_from_tpf(tpf, mask=None) -> np.ndarray:
    """Full-FFI [col, row] coordinates of aperture pixels on a TPF."""
    if mask is None:
        mask = getattr(tpf, "pipeline_mask", None)
        if mask is None:
            median_image = np.nanmedian(tpf.flux.value, axis=0)
            thresh = np.nanpercentile(median_image, 85.0)
            mask = median_image > thresh
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        raise RuntimeError("Empty aperture mask on TPF")
    ny, nx = mask.shape
    rows, cols = np.mgrid[tpf.row:tpf.row + ny, tpf.column:tpf.column + nx]
    return np.column_stack([cols[mask], rows[mask]]).astype(int)


def _col_to_array(lc, names):
    for name in names:
        if hasattr(lc, name) and getattr(lc, name) is not None:
            v = getattr(lc, name)
            try:
                return np.asarray(v.value if hasattr(v, "value") else v, dtype=float)
            except Exception:
                continue
    return None


def extract_sector_spoc(tic: int, sector: int, cfg: Config,
                        flux_column: str = "pdcsap") -> SectorData:
    """Extract one sector from a SPOC SAP or PDCSAP light curve.

    Attaches the matching SPOC TPF when available so ap_pixels, centroids and
    the flux cube remain usable for TRICERATOPS and difference imaging.
    """
    lc = _load_spoc_lightcurve(tic, sector, cfg)
    flux_column = flux_column.lower()
    if flux_column == "pdcsap":
        flux = _col_to_array(lc, ("pdcsap_flux", "flux"))
        err = _col_to_array(lc, ("pdcsap_flux_err", "flux_err"))
    else:
        flux = _col_to_array(lc, ("sap_flux", "flux"))
        err = _col_to_array(lc, ("sap_flux_err", "flux_err"))
    if flux is None:
        raise RuntimeError(f"SPOC LC sector {sector}: no {flux_column} flux")
    if err is None:
        err = np.full_like(flux, np.nanmedian(np.abs(flux)) * 1e-3)
    time = np.asarray(lc.time.value if hasattr(lc.time, "value") else lc.time,
                      dtype=float)

    cen_x = cen_y = None
    for xa, ya in (("mom_centr1", "mom_centr2"),
                   ("centroid_col", "centroid_row")):
        cx = _col_to_array(lc, (xa,))
        cy = _col_to_array(lc, (ya,))
        if cx is not None and cy is not None:
            cen_x, cen_y = cx, cy
            break

    tpf = _load_spoc_tpf(tic, sector, cfg)
    cube = cube_time = wcs = None
    col0 = row0 = 0
    if tpf is not None:
        try:
            ap_pixels = _ap_pixels_from_tpf(tpf)
        except Exception as exc:
            print(f"  ! pipeline aperture failed ({exc}); percentile mask")
            med = np.nanmedian(tpf.flux.value, axis=0)
            thr = np.nanpercentile(med, cfg.percentile)
            ap_pixels = _ap_pixels_from_tpf(tpf, med > thr)
        if cen_x is None or cen_y is None:
            try:
                mask = getattr(tpf, "pipeline_mask", None)
                cx, cy = tpf.estimate_centroids(aperture_mask=mask)
                cen_x = np.asarray(cx.value, dtype=float)
                cen_y = np.asarray(cy.value, dtype=float)
            except Exception:
                pass
        cube = np.asarray(tpf.flux.value, dtype=float)
        cube_time = np.asarray(tpf.time.value, dtype=float)
        wcs = tpf.wcs
        col0, row0 = int(tpf.column), int(tpf.row)
        try:
            with PdfPages(cfg.path(f"spoc_tpf_{tic}_s{sector:02d}.pdf")) as pdf:
                ax = tpf.plot(
                    aperture_mask=getattr(tpf, "pipeline_mask", None),
                    mask_color="k")
                pdf.savefig(ax.figure)
                plt.close(ax.figure)
        except Exception as exc:
            print(f"  ! TPF plot failed: {exc}")
    else:
        ap_pixels = np.array([[0, 0]], dtype=int)
        print(f"  ! sector {sector}: no TPF — placeholder aperture; "
              f"difference imaging / neighbour dilution limited")

    if cen_x is None or cen_x.size != time.size:
        cen_x = np.full(time.size, np.nan)
        cen_y = np.full(time.size, np.nan)

    good = (np.isfinite(time) & np.isfinite(flux) & np.isfinite(err) & (flux > 0))
    if cfg.err_clip_factor and good.any():
        rel = np.where(good, err / np.where(flux > 0, flux, np.nan), np.nan)
        med_rel = np.nanmedian(rel)
        bad_err = good & (rel > cfg.err_clip_factor * med_rel)
        if bad_err.any():
            print(f"  sector {sector}: rejected {int(bad_err.sum())}/"
                  f"{int(good.sum())} cadences with flux error > "
                  f"{cfg.err_clip_factor:.0f} x median "
                  f"({med_rel * 1e6:.0f} ppm)")
        good &= ~bad_err

    time = np.ascontiguousarray(time[good], dtype=float)
    flux = np.ascontiguousarray(flux[good], dtype=float)
    err = np.ascontiguousarray(err[good], dtype=float)
    cen_x = np.ascontiguousarray(cen_x[good], dtype=float)
    cen_y = np.ascontiguousarray(cen_y[good], dtype=float)

    with PdfPages(cfg.path(f"spoc_{flux_column}_{tic}_s{sector:02d}.pdf")) as pdf:
        fig = plt.figure(figsize=(9, 4))
        med = np.nanmedian(flux)
        plt.plot(time, flux / med if med and med > 0 else flux, "k.", ms=2)
        plt.xlabel(f"{TIME_LABEL} (d)")
        plt.ylabel("Normalised flux")
        plt.title(f"{_mission_id(tic, cfg)} - "
                  f"{'quarter' if is_kepler() else 'sector'} {sector} - "
                  f"{'Kepler' if is_kepler() else 'SPOC'} {flux_column.upper()}")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # PDCSAP is already crowding-corrected (CROWDSAP); SAP is not
    g_t = float("nan")
    fdf = getattr(cfg, "_field", None)
    if flux_column == "sap" and wcs is not None and fdf is not None:
        try:
            g_t = target_fraction(ap_pixels, wcs, col0, row0, fdf, cfg.psf_sigma)
        except Exception:
            g_t = float("nan")
    return SectorData(sector=sector, time=time, flux=flux, flux_err=err,
                      ap_pixels=ap_pixels, cen_x=cen_x, cen_y=cen_y,
                      cube=cube, cube_time=cube_time, wcs=wcs,
                      col0=col0, row0=row0, g_target=g_t,
                      aperture_kind=f"{'Kepler' if is_kepler() else 'SPOC'} {flux_column}",
                      cadence_d=float(np.nanmedian(np.diff(time)))
                      if time.size > 2 else float("nan"))


def extract_sector_tesscut(tic: int, sector: int, cfg: Config) -> SectorData:
    """Load a TESScut cutout (cache-first) and extract a background-corrected LC.

    If a matching FITS already exists under ``cfg.cache_dir()``, it is opened
    directly — no MAST query.
    """
    tpf = _load_tesscut_tpf(tic, sector, cfg.cutout_size, cfg)

    # --- aperture: optimal (catalogue PSF model) or percentile ---
    median_image = np.nanmedian(tpf.flux.value, axis=0)
    target_mask, ap_kind = None, "percentile"
    fdf = getattr(cfg, "_field", None)
    if cfg.aperture_mode == "optimal" and fdf is not None:
        try:
            target_mask = optimal_aperture(median_image, tpf.wcs,
                                           int(tpf.column), int(tpf.row),
                                           fdf, cfg)
            ap_kind = "optimal"
        except Exception as exc:
            print(f"  ! optimal aperture failed in sector {sector} ({exc}); "
                  f"percentile aperture used")
            target_mask = None
    if target_mask is None:
        ap_kind = "percentile"
        thresh = np.nanpercentile(median_image, cfg.percentile)
        target_mask = median_image > thresh

    if target_mask.sum() == 0:
        raise RuntimeError(f"Empty aperture in sector {sector}")

    # --- pixel coordinates, [col, row], full-FFI frame ---
    # NOTE: the original script transposed rows and columns here.  mgrid
    # returns (row, col) in the same [y, x] order as the image array, which
    # is what target_mask indexes, so the mapping is now consistent.
    rows, cols = np.mgrid[tpf.row:tpf.row + cfg.cutout_size,
                          tpf.column:tpf.column + cfg.cutout_size]
    ap_pixels = np.column_stack([cols[target_mask], rows[target_mask]]).astype(int)

    # Centroids and flux share the same cadence axis of the TPF.  Build
    # everything at full TPF length first, then apply one common quality mask
    # so the boolean index can never go out of bounds (this used to break on
    # newer sectors where remove_nans() and estimate_centroids() disagreed).
    cen_x_all, cen_y_all = tpf.estimate_centroids(aperture_mask=target_mask)
    cen_x_all = np.asarray(cen_x_all.value, dtype=float)
    cen_y_all = np.asarray(cen_y_all.value, dtype=float)

    target_lc = tpf.to_lightcurve(aperture_mask=target_mask)

    # --- background from the faintest pixels ---
    background_mask = ~tpf.create_threshold_mask(threshold=0.001,
                                                 reference_pixel=None)
    n_bkg = int(background_mask.sum())
    n_tgt = int(target_mask.sum())
    if n_bkg < 4:
        print(f"  ! only {n_bkg} background pixels in sector {sector}; "
              "skipping background correction")
        flux_raw = np.asarray(target_lc.flux.value, dtype=float)
        ferr_raw = np.asarray(target_lc.flux_err.value, dtype=float)
    else:
        bkg_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_bkg
        bkg_estimate = bkg_per_pixel * n_tgt
        flux_raw = (np.asarray(target_lc.flux.value, dtype=float)
                    - np.asarray(bkg_estimate.flux.value, dtype=float))
        ferr_raw = np.sqrt(
            np.asarray(target_lc.flux_err.value, dtype=float) ** 2
            + np.asarray(bkg_estimate.flux_err.value, dtype=float) ** 2
        )

    time_raw = np.asarray(tpf.time.value, dtype=float)

    good = (np.isfinite(time_raw) & np.isfinite(flux_raw)
            & np.isfinite(ferr_raw) & (flux_raw > 0)
            & np.isfinite(cen_x_all) & np.isfinite(cen_y_all))
    if cfg.err_clip_factor and good.any():
        # relative error, so a sector-wide flux trend does not matter
        rel = np.where(good, ferr_raw / np.where(flux_raw > 0, flux_raw, np.nan),
                       np.nan)
        med_rel = np.nanmedian(rel)
        bad_err = good & (rel > cfg.err_clip_factor * med_rel)
        if bad_err.any():
            print(f"  sector {sector}: rejected {int(bad_err.sum())}/"
                  f"{int(good.sum())} cadences with flux error > "
                  f"{cfg.err_clip_factor:.0f} x median "
                  f"({med_rel * 1e6:.0f} ppm)")
        good &= ~bad_err
    time = np.ascontiguousarray(time_raw[good], dtype=float)
    flux = np.ascontiguousarray(flux_raw[good], dtype=float)
    ferr = np.ascontiguousarray(ferr_raw[good], dtype=float)
    cen_x = np.ascontiguousarray(cen_x_all[good], dtype=float)
    cen_y = np.ascontiguousarray(cen_y_all[good], dtype=float)

    # --- diagnostic plot ---
    with PdfPages(cfg.path(f"tesscut_{tic}_s{sector:02d}.pdf")) as pdf:
        # tpf.plot() creates its own Axes/Figure internally; capture the one
        # it actually returns rather than a separate, empty plt.figure().
        ax = tpf.plot(aperture_mask=target_mask, mask_color="k")
        pdf.savefig(ax.figure)
        plt.close(ax.figure)

        fig = plt.figure(figsize=(9, 4))
        plt.plot(time, flux / np.median(flux), "k.", ms=2)
        plt.xlabel(f"{TIME_LABEL} (d)")
        plt.ylabel("Normalised flux")
        plt.title(f"TIC {tic} - sector {sector} - corrected")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    g_t = target_fraction(ap_pixels, tpf.wcs, int(tpf.column), int(tpf.row),
                          fdf, cfg.psf_sigma) if fdf is not None else float("nan")
    print(f"  sector {sector}: {ap_kind} aperture, {int(target_mask.sum())} px, "
          f"target flux fraction g = {g_t:.3f}")
    return SectorData(sector=sector, time=time, flux=flux, flux_err=ferr,
                      ap_pixels=ap_pixels, cen_x=cen_x, cen_y=cen_y,
                      cube=np.asarray(tpf.flux.value, dtype=float),
                      cube_time=np.asarray(tpf.time.value, dtype=float),
                      wcs=tpf.wcs, col0=int(tpf.column), row0=int(tpf.row),
                      g_target=g_t, aperture_kind=ap_kind,
                      cadence_d=float(np.nanmedian(np.diff(time)))
                      if time.size > 2 else float("nan"))


def detrend_sector(sd: SectorData, window_length: float, cfg: Config,
                   mask_transits: Optional[np.ndarray] = None) -> None:
    """Biweight detrend one sector, optionally with the transits masked out.

    wotan excludes points flagged True in `mask` from the trend estimate but
    still divides them out, which is exactly what we want: the transit is
    preserved instead of being absorbed by the filter.
    """
    window_length = max(float(window_length), cfg.window_fallback)
    flat, trend = flatten(
        sd.time,
        sd.flux,
        window_length=window_length,
        method=cfg.detrend_method,
        return_trend=True,
        edge_cutoff=0.0,
        mask=mask_transits,
    )
    sd.norm_flux = flat
    sd.norm_err = sd.flux_err / np.where(trend > 0, trend, np.nan)
    sd.trend = trend
    sd._dilution_done = False          # a fresh detrend: not yet corrected


def renormalize_sector_oot(sd: SectorData, cfg: Config,
                           P: Optional[float] = None,
                           T0: Optional[float] = None,
                           dur: Optional[float] = None) -> None:
    """Divide the detrended sector LC by its out-of-transit median → continuum at 1.

    When an ephemeris is given, in-transit points (1.5 × T14) are excluded from
    the median so a deep transit cannot pull the normalisation down.
    """
    _renormalize_core(sd, cfg, P, T0, dur)
    apply_dilution_correction(sd, cfg)


def _renormalize_core(sd, cfg, P=None, T0=None, dur=None) -> None:
    if not cfg.renorm_oot or sd.norm_flux is None:
        return
    f = np.asarray(sd.norm_flux, float)
    ok = np.isfinite(f) & (f > 0)
    if P is not None and T0 is not None and dur is not None and dur > 0:
        in_tr = transit_mask(sd.time, float(P), float(1.5 * dur), float(T0))
        sample = ok & ~in_tr
    else:
        sample = ok
    if int(sample.sum()) < 10:
        sample = ok
    if int(sample.sum()) < 5:
        return
    med = float(np.nanmedian(f[sample]))
    if not np.isfinite(med) or med <= 0:
        return
    sd.norm_flux = f / med
    if sd.norm_err is not None:
        sd.norm_err = np.asarray(sd.norm_err, float) / med



def sector_norm_quality(sd: "SectorData", cfg: Config) -> tuple:
    """Return (ok, reason) for a detrended sector LC.

    Flags catastrophic normalisation failures (e.g. sector 56/79-style
    continuum blow-ups) that would dominate the fold and TRICERATOPS sigma.
    """
    f = getattr(sd, "norm_flux", None)
    if f is None:
        return True, ""
    f = np.asarray(f, float)
    fin = np.isfinite(f)
    if int(fin.sum()) < 30:
        return False, f"too few finite points ({int(fin.sum())})"
    x = f[fin]
    med = float(np.median(x))
    rms = float(1.4826 * np.median(np.abs(x - med)))
    if not np.isfinite(rms):
        return False, "non-finite robust RMS"
    if rms > float(cfg.sector_max_rms):
        return False, (f"robust RMS = {rms * 1e2:.2f}% > "
                       f"{cfg.sector_max_rms * 1e2:.2f}%")
    if abs(med - 1.0) > float(cfg.sector_max_median_offset):
        return False, (f"|median-1| = {abs(med - 1.0) * 1e2:.2f}% > "
                       f"{cfg.sector_max_median_offset * 1e2:.2f}%")
    level = float(cfg.sector_outlier_level)
    frac = float(np.mean(np.abs(x - 1.0) > level))
    if frac > float(cfg.sector_max_outlier_frac):
        return False, (f"fraction |f-1|>{level * 1e2:.1f}% = {frac * 1e2:.1f}% "
                       f"> {cfg.sector_max_outlier_frac * 1e2:.1f}%")
    return True, ""


def filter_bad_sectors(sectors_data, cfg: Config, log=None, tag: str = ""):
    """Drop sectors that fail sector_norm_quality; return the filtered list."""
    if not getattr(cfg, "auto_exclude_bad_sectors", True):
        return list(sectors_data)
    kept, dropped = [], []
    for sd in sectors_data:
        ok, reason = sector_norm_quality(sd, cfg)
        if ok:
            kept.append(sd)
        else:
            dropped.append((int(sd.sector), reason))
    if dropped:
        msg = (f"  auto-excluded {len(dropped)} sector(s)"
               + (f" after {tag}" if tag else "") + ":")
        _log(log, msg)                  # _log prints as well
        for sec, reason in dropped:
            _log(log, f"    sector {sec}: {reason}")
    if not kept:
        raise RuntimeError(
            "All sectors failed the normalisation quality cuts; "
            "relax sector_max_rms / sector_max_outlier_frac or disable "
            "auto_exclude_bad_sectors")
    return kept


def clip_outliers(time, flux, flux_err, sector_ids=None, sigma: float = 5.0,
                  P: Optional[float] = None, T0: Optional[float] = None,
                  dur: Optional[float] = None):
    """Drop extreme outliers with transit-aware negative clipping.

    * Always reject upward spikes (flares / residual scattered-light spikes).
    * Without an ephemeris: nothing else (positive-only).  Any downward cut
      would delete the cores of transits deeper than sigma x RMS.
    * Outside a transit window (when P, T0, dur are given): also reject
      downward spikes, which are residual ramps / bad cadences, not transits.
    * Inside the transit window: keep every negative excursion; only
      non-physical fluxes (below 10 % of the continuum) are removed.
      (v3.2 fix: the old floor of continuum - 15 sigma cut the bottom off
      every eclipse deeper than ~15 x RMS, e.g. 2.7 % at RMS = 1800 ppm.)
    """
    if sigma is None or sigma <= 0:
        if sector_ids is None:
            return time, flux, flux_err
        return time, flux, flux_err, sector_ids

    flux = np.asarray(flux, float)
    med = np.nanmedian(flux[np.isfinite(flux)]) if np.any(np.isfinite(flux)) else np.nan
    rms = 1.4826 * np.nanmedian(np.abs(flux[np.isfinite(flux)] - med)) if np.isfinite(med) else np.nan
    if not np.isfinite(rms) or rms <= 0 or not np.isfinite(med):
        if sector_ids is None:
            return time, flux, flux_err
        return time, flux, flux_err, sector_ids

    finite = np.isfinite(flux)
    # upward spikes always out
    ok = finite & (flux < med + sigma * rms)

    in_tr = np.zeros(flux.size, dtype=bool)
    if P is not None and T0 is not None and dur is not None and dur > 0:
        in_tr = transit_mask(np.asarray(time, float), float(P),
                             float(1.5 * dur), float(T0))

    have_eph = P is not None and T0 is not None and dur is not None and dur > 0
    if have_eph:
        # OOT: also clip downward outliers
        oot = ~in_tr
        ok &= ~oot | (flux > med - sigma * rms)
    # in-transit (or no ephemeris): only non-physical fluxes
    ok &= flux > 0.1 * med

    n_rej = int((finite & ~ok).sum())
    if n_rej:
        kind = "transit-aware" if have_eph else "positive-only"
        print(f"  outlier clip ({kind}, {sigma:.1f}σ): rejected "
              f"{n_rej}/{int(finite.sum())} points  "
              f"(robust RMS = {rms * 1e6:.0f} ppm)")
    if sector_ids is None:
        return time[ok], flux[ok], flux_err[ok]
    return time[ok], flux[ok], flux_err[ok], sector_ids[ok]


def stitch(sectors_data: Sequence[SectorData]):
    """Concatenate the detrended sectors; drop any residual NaNs."""
    t = np.concatenate([sd.time for sd in sectors_data])
    f = np.concatenate([sd.norm_flux for sd in sectors_data])
    e = np.concatenate([sd.norm_err for sd in sectors_data])
    s = np.concatenate([np.full(sd.time.size, sd.sector, dtype=int)
                        for sd in sectors_data])

    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    order = np.argsort(t[ok])
    return t[ok][order], f[ok][order], e[ok][order], s[ok][order]


def save_sector_lightcurves(sectors_data: Sequence[SectorData], cfg: Config,
                            tic: int, tag: str = "detrended") -> None:
    """Write per-sector light curves to text files.

    Columns: time (BTJD), flux (normalised if detrended), flux_err,
    and for the detrended product also the trend that was divided out.
    """
    for sd in sectors_data:
        if tag == "detrended" and sd.norm_flux is not None:
            cols = np.column_stack([
                sd.time, sd.norm_flux, sd.norm_err, sd.trend
            ])
            header = "time_BTJD  norm_flux  norm_err  trend"
        else:
            # background-corrected, not yet detrended
            med = np.nanmedian(sd.flux)
            cols = np.column_stack([
                sd.time, sd.flux / med, sd.flux_err / med
            ])
            header = "time_BTJD  flux_norm  flux_err"
        fname = cfg.path(f"lc_{tag}_{tic}_s{sd.sector:02d}.txt")
        np.savetxt(fname, cols, header=header)
        print(f"  saved {fname} ({cols.shape[0]} points)")


def save_stitched_lightcurve(time, flux, ferr, sector_ids, cfg: Config,
                             tic: int) -> None:
    """Write the full multi-sector detrended light curve."""
    cols = np.column_stack([time, flux, ferr, sector_ids])
    fname = cfg.path(f"lc_stitched_{tic}.txt")
    np.savetxt(fname, cols, header="time_BTJD  norm_flux  norm_err  sector")
    print(f"  saved {fname} ({cols.shape[0]} points)")


def plot_sector_panels(sectors_data: Sequence[SectorData], cfg: Config,
                       tic: int, tag: str = "detrended") -> None:
    """One panel per sector: light curve (and trend if available)."""
    n = len(sectors_data)
    if n == 0:
        return
    # arrange in a column that stays readable for a typical TESS sector count
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.2 * n, 4)),
                             sharex=False, squeeze=False)
    axes = axes.ravel()

    for ax, sd in zip(axes, sectors_data):
        if tag == "detrended" and sd.norm_flux is not None:
            y = sd.norm_flux
            ax.plot(sd.time, y, "k.", ms=1.5, alpha=0.7, label="detrended")
            if sd.trend is not None:
                med = np.nanmedian(sd.flux)
                ax.plot(sd.time, sd.trend / med, "C1-", lw=0.8, alpha=0.8,
                        label="trend / median")
        else:
            med = np.nanmedian(sd.flux)
            y = sd.flux / med
            ax.plot(sd.time, y, "k.", ms=1.5, alpha=0.7)
        lo, hi = np.nanpercentile(y, [1.0, 99.0])
        pad = 0.5 * (hi - lo) if hi > lo else 0.01
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("Norm. flux")
        ax.set_title(f"Sector {sd.sector}", fontsize=10, loc="left")
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel(f"{TIME_LABEL} (d)")
    if tag == "detrended":
        axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(f"TIC {tic} – light curves per sector ({tag})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = cfg.path(f"lc_panels_{tag}_{tic}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved multi-panel figure: {out}")


def plot_transits_on_sectors(sectors_data: Sequence[SectorData],
                             P: float, T0: float, dur: float,
                             cfg: Config, tic: int) -> None:
    """Detrended LC per sector with TLS transit windows marked.

    Vertical shaded bands show every predicted transit (T0 + nP ± T14/2)
    that falls inside the sector baseline.  Useful to check that the
    ephemeris lands on real dips and that the second-pass detrend did not
    swallow them.
    """
    n = len(sectors_data)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.4 * n, 4)),
                             sharex=False, squeeze=False)
    axes = axes.ravel()

    for ax, sd in zip(axes, sectors_data):
        if sd.norm_flux is None:
            continue
        ax.plot(sd.time, sd.norm_flux, "k.", ms=1.5, alpha=0.7)

        tmin, tmax = float(sd.time.min()), float(sd.time.max())
        # all epochs that could touch this sector
        n0 = int(np.floor((tmin - T0) / P)) - 1
        n1 = int(np.ceil((tmax - T0) / P)) + 1
        labeled = False
        for k in range(n0, n1 + 1):
            tc = T0 + k * P
            t1, t2 = tc - 0.5 * dur, tc + 0.5 * dur
            if t2 < tmin or t1 > tmax:
                continue
            ax.axvspan(t1, t2, color="C3", alpha=0.25,
                       label="TLS transit" if not labeled else None)
            ax.axvline(tc, color="C3", ls="--", lw=0.7, alpha=0.8)
            labeled = True

        # robust y-limits
        lo, hi = np.nanpercentile(sd.norm_flux, [1.0, 99.0])
        pad = 0.5 * (hi - lo) if hi > lo else 0.01
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("Norm. flux")
        ax.set_title(f"Sector {sd.sector}", fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        if labeled:
            ax.legend(loc="upper right", fontsize=7, frameon=False)

    axes[-1].set_xlabel(f"{TIME_LABEL} (d)")
    fig.suptitle(
        f"TIC {tic} – detrended LCs with TLS ephemeris "
        f"(P = {P:.5f} d, T0 = {T0:.5f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = cfg.path(f"lc_transits_sectors_{tic}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved transit-marked sector figure: {out}")


# --------------------------------------------------------------------------
# TLS search
# --------------------------------------------------------------------------
def bin_for_search(time, flux, flux_err, bin_minutes: float):
    """Error-weighted binning of an irregularly-sampled light curve.

    Used only to speed up the TLS period search: TLS's runtime scales with
    the number of input points. For a multi-sector TESS baseline the gain
    from binning is real but modest (roughly a factor of ~2 for a mix of
    200s/10-30min FFI cadences at 10-minute bins), since most of the
    baseline is already gaps between sectors rather than dense in-sector
    sampling; combine it with a lower oversampling_factor and, if you have
    any prior on the period, a narrower P_max for a bigger effect. Bins of
    10 minutes still fully resolve transits with T14 of an hour or more.
    Nothing downstream of the search (per-transit depths, difference
    imaging, centroids) uses this binned version.
    """
    if bin_minutes is None or bin_minutes <= 0:
        return time, flux, flux_err

    binsize = bin_minutes / (60.0 * 24.0)
    edges = np.arange(time.min(), time.max() + binsize, binsize)
    idx = np.digitize(time, edges) - 1

    w = 1.0 / np.clip(flux_err, 1e-12, None) ** 2
    n_bins = edges.size - 1
    sum_w = np.bincount(idx, weights=w, minlength=n_bins)
    sum_wf = np.bincount(idx, weights=w * flux, minlength=n_bins)

    good = sum_w > 0
    t_bin = 0.5 * (edges[:-1] + edges[1:])[good]
    f_bin = (sum_wf[good] / sum_w[good])
    e_bin = np.sqrt(1.0 / sum_w[good])

    print(f"  binned for search: {time.size} pts -> {t_bin.size} pts "
         f"({bin_minutes:.0f} min bins)")
    return t_bin, f_bin, e_bin



def segment_edge_distance(time, gap: float = 0.5):
    """Distance [d] of every cadence to the nearest edge of its contiguous
    data segment (a segment ends at any gap longer than *gap* days)."""
    t = np.asarray(time, float)
    order = np.argsort(t)
    ts = t[order]
    br = np.where(np.diff(ts) > gap)[0]
    starts = np.concatenate([[0], br + 1])
    ends = np.concatenate([br, [ts.size - 1]])
    seg = np.repeat(np.arange(starts.size), ends - starts + 1)
    d_sorted = np.minimum(ts - ts[starts][seg], ts[ends][seg] - ts)
    out = np.empty_like(d_sorted)
    out[order] = d_sorted
    return out


def transit_edge_distance(time, tcs, gap: float = 0.5):
    """Distance [d] of each mid-time to the nearest edge of the data segment
    that contains it (a segment ends at any gap longer than *gap* days);
    0 for a mid-time outside the data."""
    t = np.asarray(time, float)
    ts = np.sort(t[np.isfinite(t)])
    if ts.size == 0:
        return np.zeros(len(tcs))
    br = np.where(np.diff(ts) > gap)[0]
    starts = ts[np.concatenate([[0], br + 1])]
    ends = ts[np.concatenate([br, [ts.size - 1]])]
    out = []
    for tc in np.asarray(tcs, float):
        k = int(np.searchsorted(starts, tc, side="right")) - 1
        out.append(0.0 if k < 0 or tc > ends[k]
                   else float(min(tc - starts[k], ends[k] - tc)))
    return np.array(out)


def candidate_edge_fraction(time, P, T0, dur, within: float = 0.5,
                            gap: float = 0.5):
    """Fraction of the in-transit cadences of a candidate that lie within
    *within* days of a data-segment edge (orbit start/end, gaps).  Ramps and
    scattered light at the edges mimic transits; a candidate that lives
    mostly there is suspect."""
    inm = transit_mask(np.asarray(time, float), float(P), float(dur), float(T0))
    if int(inm.sum()) == 0:
        return float("nan"), 0
    d = segment_edge_distance(time, gap=gap)
    return float(np.mean(d[inm] < within)), int(inm.sum())


def search_multi_window(sectors_data, star, cfg: Config, log=None):
    """Blind TLS search for several first-pass biweight windows.

    For every window the sectors are detrended, stitched, clipped and
    searched; the highest-SDE result is returned together with the stitched
    light curve of that window (the sectors are left detrended with it).
    Returns (results, window, (time, flux, ferr, sector_ids), rows).
    """
    rows, best = [], None
    for w in cfg.window_length_grid:
        w = max(float(w), cfg.window_fallback)
        for sd in sectors_data:
            detrend_sector(sd, w, cfg)
            renormalize_sector_oot(sd, cfg)
        t, f, e, sid = stitch(sectors_data)
        t, f, e, sid = clip_outliers(t, f, e, sid, sigma=cfg.outlier_sigma)
        bt, bf, be = bin_for_search(t, f, e, cfg.tls_search_bin_minutes)
        try:
            res = run_tls(bt, bf, be, star, cfg,
                          tag=f" (first pass, window {w:g} d)")
        except Exception as exc:
            _log(log, f"  ! TLS failed for window {w:g} d: {exc}")
            continue
        frac, n_in = candidate_edge_fraction(t, res.period, res.T0,
                                             res.duration)
        rows.append(dict(window=w, P=float(res.period), T0=float(res.T0),
                         dur_h=float(res.duration) * 24,
                         depth_ppm=float(1 - res.depth) * 1e6,
                         SDE=float(res.SDE),
                         n_tr=int(getattr(res, "distinct_transit_count", -1)),
                         edge_frac=frac))
        if best is None or res.SDE > best[0].SDE:
            best = (res, w)
    if best is None:
        raise RuntimeError("TLS failed for every detrending window")
    res, w_best = best
    for r in rows:
        r["agrees"] = abs(r["P"] - res.period) / res.period < 0.01
    _log(log, "\n--- Blind search for several first-pass windows ---")
    _log(log, "  window[d]   P[d]        T14[h]  depth[ppm]  SDE   n_tr  "
              "edge-frac  same-P")
    for r in rows:
        mark = "  <-- best" if r["window"] == w_best else ""
        _log(log, f"  {r['window']:7.2f}  {r['P']:10.4f}  {r['dur_h']:6.2f}  "
                  f"{r['depth_ppm']:9.0f}  {r['SDE']:5.1f}  {r['n_tr']:4d}  "
                  f"{r['edge_frac']:8.2f}   {'yes' if r['agrees'] else 'no'}"
                  f"{mark}")
    n_agree = sum(r["agrees"] for r in rows)
    if len(rows) > 1 and n_agree < 2:
        _log(log, "  ! the best period is found with only one window: "
                  "probably residual red noise / sampling alias, not a "
                  "signal.  Do not trust it without the per-transit table.")
    # leave the sectors detrended with the winning window
    for sd in sectors_data:
        detrend_sector(sd, w_best, cfg)
        renormalize_sector_oot(sd, cfg)
    t, f, e, sid = stitch(sectors_data)
    t, f, e, sid = clip_outliers(t, f, e, sid, sigma=cfg.outlier_sigma)
    return res, w_best, (t, f, e, sid), rows


def run_tls(time, flux, flux_err, star: dict, cfg: Config, tag: str = ""):
    """Run TLS with a period grid that the baseline can actually support."""
    baseline = float(time.max() - time.min())
    P_max = min(cfg.P_max, baseline / max(cfg.n_transits_min, 1))
    if P_max <= cfg.P_min:
        raise RuntimeError(f"Baseline of {baseline:.1f} d is too short for "
                           f"P_min = {cfg.P_min} d")

    kwargs = dict(
        u=star["u"],
        period_min=cfg.P_min,
        period_max=P_max,
        duration_grid_step=cfg.duration_grid_step,
        n_transits_min=cfg.n_transits_min,
        transit_template="default",
        oversampling_factor=cfg.tls_oversampling_factor,
        show_progress_bar=True,   # tqdm progress during the period search
        verbose=True,             # status lines: grid size, periods, etc.
    )
    if np.isfinite(star["rad"]):
        kwargs.update(R_star=star["rad"],
                      R_star_min=max(star["rad_min"], 0.1),
                      R_star_max=star["rad_max"])
    if np.isfinite(star["mass"]):
        kwargs.update(M_star=star["mass"],
                      M_star_min=max(star["mass_min"], 0.05),
                      M_star_max=star["mass_max"])
    if cfg.tls_threads:
        kwargs["use_threads"] = cfg.tls_threads

    # verbose=True on the constructor prints the TLS banner / data summary
    model = transitleastsquares(time, flux, flux_err, verbose=True)
    results = model.power(**kwargs)

    print(f"  TLS{tag}: P = {results.period:.6f} d, T0 = {results.T0:.5f}, "
          f"depth = {(1 - results.depth) * 1e6:.0f} ppm, "
          f"T14 = {results.duration * 24:.2f} h, SDE = {results.SDE:.1f}")
    return results


def _trapezoid_model(phase, depth, t14, ingress):
    """Trapezoid transit; phase is time from mid-transit [d]."""
    depth = max(float(depth), 0.0)
    t14 = max(float(t14), 1e-4)
    ingress = float(np.clip(ingress, 1e-5, 0.9 * t14))
    half = 0.5 * t14
    flat = max(half - 0.5 * ingress, 0.0)
    a = np.abs(phase)
    y = np.ones_like(phase, dtype=float)
    y[a <= flat] = 1.0 - depth
    wing = (a > flat) & (a < half + 0.5 * ingress)
    if wing.any():
        span = half + 0.5 * ingress - flat
        frac = (a[wing] - flat) / max(span, 1e-12)
        y[wing] = 1.0 - depth * (1.0 - frac)
    return y


def fit_transit_shape(time, flux, flux_err, P, T0, dur_guess=None,
                      depth_guess=None):
    """Least-squares trapezoid fit on the phase-folded LC (independent of TLS).

    Free parameters: depth, T14, ingress duration, small T0 offset.
    Returns (dur, depth, t0_shift, model_callable).
    """
    P, T0 = float(P), float(T0)
    d_g = float(dur_guess) if (dur_guess and dur_guess > 0) else 0.1
    ph = (time - T0 + 0.5 * P) % P - 0.5 * P
    if depth_guess is None or depth_guess <= 0:
        core = np.abs(ph) < max(0.25 * d_g, 0.01)
        out = (np.abs(ph) > 1.5 * d_g) & (np.abs(ph) < max(3 * d_g, 0.25))
        f_out = np.nanmedian(flux[out]) if out.sum() > 10 else 1.0
        f_in = np.nanmedian(flux[core]) if core.sum() > 5 else f_out
        depth_guess = max(f_out - f_in, 1e-4)
    depth_guess = float(depth_guess)

    win = max(3.0 * d_g, 0.15)
    use = np.abs(ph) < win
    if use.sum() < 30:
        use = np.ones(ph.size, dtype=bool)
    x = ph[use]
    y = flux[use]
    e = np.clip(flux_err[use], 1e-6, None)

    p0 = np.array([depth_guess, d_g, 0.2 * d_g, 0.0])
    lo = np.array([0.1 * depth_guess, 0.3 * d_g, 1e-4, -0.3 * d_g])
    hi = np.array([3.0 * depth_guess, min(0.4 * P, 3.0 * d_g),
                   0.8 * min(0.4 * P, 3.0 * d_g), 0.3 * d_g])
    lo = np.minimum(lo, hi - 1e-8)

    def residual(p):
        depth, t14, ingress, t0s = p
        return (y - _trapezoid_model(x - t0s, depth, t14, ingress)) / e

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=1.0, max_nfev=200)
        depth, t14, ingress, t0s = sol.x
    except Exception as exc:
        print(f"  ! trapezoid fit failed ({exc}); using guesses")
        depth, t14, ingress, t0s = depth_guess, d_g, 0.2 * d_g, 0.0

    depth = float(max(depth, 1e-6))
    t14 = float(np.clip(t14, 0.02, 0.4 * P))
    ingress = float(np.clip(ingress, 1e-4, 0.9 * t14))
    t0s = float(t0s)

    def model(phase_days):
        return _trapezoid_model(np.asarray(phase_days) - t0s, depth, t14, ingress)

    return t14, depth, t0s, model


def estimate_duration_depth(time, flux, flux_err, P, T0, dur_guess=None):
    """Quick duration/depth via trapezoid fit (wrapper for mask sizing)."""
    d_g = float(dur_guess) if (dur_guess and dur_guess > 0) else 0.1
    t14_fit, depth_fit, _, _ = fit_transit_shape(
        time, flux, flux_err, P, T0, dur_guess=d_g)
    return t14_fit, depth_fit


def refine_ephemeris(time, flux, flux_err, P, T0, dur):
    """Fine-tune P and T0; fit depth+T14 with a trapezoid (not TLS).

    Returns (P, T0, T14, depth).
    """
    P, T0, dur = float(P), float(T0), float(dur)

    baseline = float(time.max() - time.min())
    dP = max(3.0 * max(dur, 0.05) * P / max(baseline, P), 1e-6)
    dT0 = max(0.5 * max(dur, 0.05), 0.01)
    P_grid = np.linspace(P - dP, P + dP, 41)
    T0_grid = np.linspace(T0 - dT0, T0 + dT0, 41)
    half = 0.5 * max(dur, 0.05)

    def score(p, t0):
        phase = (time - t0 + 0.5 * p) % p - 0.5 * p
        ins = np.abs(phase) < half
        outs = (np.abs(phase) > 1.5 * half) & (np.abs(phase) < 4.0 * half)
        if ins.sum() < 5 or outs.sum() < 10:
            return -np.inf
        w_in = 1.0 / np.clip(flux_err[ins], 1e-12, None) ** 2
        w_out = 1.0 / np.clip(flux_err[outs], 1e-12, None) ** 2
        mean_in = np.sum(w_in * flux[ins]) / np.sum(w_in)
        mean_out = np.sum(w_out * flux[outs]) / np.sum(w_out)
        var_in = np.sum(w_in * (flux[ins] - mean_in) ** 2) / np.sum(w_in)
        return (mean_out - mean_in) / np.sqrt(max(var_in, 1e-16))

    best, best_sc = (P, T0), -np.inf
    for p in P_grid:
        for t0 in T0_grid:
            sc = score(p, t0)
            if sc > best_sc:
                best_sc, best = sc, (p, t0)
    P_ref, T0_ref = best
    if abs(P_ref - P) / P > 1e-8 or abs(T0_ref - T0) > 1e-5:
        print(f"  refined ephemeris: P = {P:.6f} -> {P_ref:.6f} d, "
              f"T0 = {T0:.5f} -> {T0_ref:.5f}")

    dur_start = max(dur, 0.05)
    t14_fit, depth_fit, t0s, _ = fit_transit_shape(
        time, flux, flux_err, P_ref, T0_ref,
        dur_guess=dur_start, depth_guess=None)
    T0_ref = T0_ref + t0s
    print(f"  trapezoid fit: T14 = {t14_fit * 24:.2f} h, "
          f"depth = {depth_fit * 1e6:.0f} ppm")
    return P_ref, T0_ref, t14_fit, depth_fit


def plot_tls_diagnostics(results, time, flux, flux_err, cfg: Config, tic: int,
                         tag: str = "",
                         P: float = None, T0: float = None, dur: float = None,
                         depth: float = None) -> None:
    """TLS periodogram + phase-folded transit after the final detrend.

    Two-panel figure used to judge detection quality:
      * top  – SDE spectrum (power vs period) with the best period marked
      * bottom – phase-folded light curve around mid-transit

    Optional *P*, *T0*, *dur*, *depth* override the TLS ephemeris (use the
    refined values from ``refine_ephemeris`` so the fold is not phase-smeared).
    """
    P = float(P if P is not None else results.period)
    T0 = float(T0 if T0 is not None else results.T0)
    dur = float(dur if dur is not None else results.duration)
    if depth is not None and depth > 0:
        depth_ppm = float(depth) * 1e6
    else:
        depth_ppm = (1.0 - float(results.depth)) * 1e6
    sde = float(results.SDE)

    fig, (ax_pg, ax_fold) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )

    # ---- periodogram -------------------------------------------------------
    periods = np.asarray(results.periods, dtype=float)
    power = np.asarray(results.power, dtype=float)
    ax_pg.plot(periods, power, "k-", lw=0.8)
    ax_pg.axvline(float(results.period), color="C3", ls="--", lw=1.2,
                  label=f"P = {float(results.period):.6f} d")
    ax_pg.axhline(cfg.SDE_threshold, color="0.5", ls=":", lw=1.0,
                  label=f"SDE threshold = {cfg.SDE_threshold:.1f}")
    ax_pg.set_xlabel("Period (d)")
    ax_pg.set_ylabel("SDE (TLS power)")
    ax_pg.set_title(f"TIC {tic} – TLS periodogram{tag}   SDE = {sde:.1f}")
    ax_pg.legend(loc="best", fontsize=8, frameon=False)

    # ---- phase-folded transit (days from mid-transit) ----------------------
    ph = (time - T0 + 0.5 * P) % P - 0.5 * P
    win = max(4.0 * dur, 0.15)
    sel = np.abs(ph) < 0.5 * win
    order = np.argsort(ph[sel])
    tp, fp = ph[sel][order], flux[sel][order]
    ep = flux_err[sel][order] if flux_err is not None else None

    ax_fold.plot(tp, fp, ".", color="0.75", ms=2, zorder=1, label="data")

    # error-weighted bins so the transit shape is readable despite smear
    if tp.size > 20:
        binsize = max(dur / 30.0, 1.0 / (60.0 * 24.0))  # ~T14/30 or 1 min
        edges = np.arange(tp.min(), tp.max() + binsize, binsize)
        dig = np.digitize(tp, edges) - 1
        n_bins = edges.size - 1
        w = 1.0 / np.clip(ep if ep is not None else np.ones_like(fp),
                          1e-12, None) ** 2
        sw = np.bincount(dig, weights=w, minlength=n_bins)
        swf = np.bincount(dig, weights=w * fp, minlength=n_bins)
        okb = sw > 0
        tb = 0.5 * (edges[:-1] + edges[1:])[okb]
        fb = (swf[okb] / sw[okb])
        eb = np.sqrt(1.0 / sw[okb])
        ax_fold.errorbar(tb, fb, yerr=eb, fmt="o", color="k", ms=4,
                         zorder=3, label="binned")

    # Trapezoid model from the independent fit (same params as adopted depth/T14)
    depth_frac = depth_ppm * 1e-6
    ingress = max(0.2 * dur, 2.0 / (60.0 * 24.0))
    xx = np.linspace(-0.5 * win, 0.5 * win, 800)
    yy = _trapezoid_model(xx, depth_frac, dur, ingress)
    ax_fold.plot(xx, yy, "r-", lw=1.8, zorder=2, label="trapezoid fit")

    ax_fold.axvline(0.0, color="C3", ls="--", lw=0.8, alpha=0.7)
    if fp.size:
        lo, hi = np.nanpercentile(fp, [1.0, 99.0])
        pad = 0.5 * (hi - lo) if hi > lo else 0.01
        ax_fold.set_ylim(lo - pad, hi + pad)
    ax_fold.set_xlabel("Time from mid-transit (d)")
    ax_fold.set_ylabel("Normalised flux")
    ax_fold.set_title(
        f"Folded transit   depth = {depth_ppm:.0f} ppm   "
        f"T14 = {dur * 24:.2f} h   P = {P:.6f} d"
    )
    ax_fold.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    suffix = tag.strip().replace(" ", "_").replace(",", "") if tag else ""
    fname = cfg.path(f"tls_diagnostics_{tic}{suffix}.pdf")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  saved TLS diagnostics: {fname}")


# --------------------------------------------------------------------------
# Per-transit depths
# --------------------------------------------------------------------------
_PT_DTYPE = [("epoch", int), ("Tc", float), ("depth", float),
             ("e_depth", float), ("npts", int), ("coverage", float),
             ("dTc", float), ("e_dTc", float), ("snr", float),
             ("dchi2", float)]


def _unit_shape_func(results, shape_model, time, P, T0, dur, ref_depth=None):
    """Return (callable unit_shape(phase_days), description).

    unit_shape is 0 out of transit and 1 at the bottom.  Priority:
      1. ``shape_model(phase)`` -> normalised flux (baseline 1), centred;
      2. the TLS model, re-expressed as a function of phase;
      3. a soft-edged box of width *dur*.
    """
    if shape_model is not None:
        grid = np.linspace(-1.5 * dur, 1.5 * dur, 2001)
        # ref_depth: normalise to the *intrinsic* depth, so that a model
        # smoothed with a long cadence returns the intrinsic depth too
        d_ref = (float(ref_depth) if ref_depth is not None
                 else 1.0 - float(np.min(shape_model(grid))))
        if np.isfinite(d_ref) and d_ref > 0:
            return (lambda ph, _m=shape_model, _d=d_ref:
                    np.clip((1.0 - _m(np.asarray(ph, float))) / _d, 0.0, 1.5),
                    "supplied shape model")

    if (results is not None
            and hasattr(results, "model_lightcurve_time")
            and hasattr(results, "model_lightcurve_model")
            and abs(float(results.duration) - dur) / max(dur, 1e-6) < 0.3):
        d_ref = 1.0 - float(results.depth)
        tm = np.asarray(results.model_lightcurve_time, float)
        fm = np.asarray(results.model_lightcurve_model, float)
        ph_m = (tm - T0 + 0.5 * P) % P - 0.5 * P
        near = np.abs(ph_m) < 2.0 * dur
        if d_ref > 0 and near.sum() >= 10:
            order = np.argsort(ph_m[near])
            x, y = ph_m[near][order], ((1.0 - fm[near]) / d_ref)[order]
            return (lambda ph, _x=x, _y=y: np.interp(
                np.asarray(ph, float), _x, _y, left=0.0, right=0.0),
                "TLS model")

    edge = max(0.05 * dur, 1.0 / (60.0 * 24.0))
    return (lambda ph, _e=edge, _d=dur: np.clip(
        (0.5 * _d - np.abs(np.asarray(ph, float))) / _e, 0.0, 1.0),
        "soft-edged box")


def per_transit_depths(time, flux, flux_err, results, cfg: Config,
                       P: float = None, T0: float = None,
                       dur: float = None, shape_model=None,
                       return_rejected: bool = False, fit_tc=None,
                       ref_depth=None):
    """Depth (and mid-time) of each individual transit.

    For every epoch and for each trial shift of the mid-time within
    +-``per_transit_tc_range`` x T14, the linear model

        f(t) = a + b (t - Tc) - depth * s(t - Tc - shift)

    is solved; the shift with the lowest chi2 is kept.  A free mid-time is
    essential when the ephemeris is uncertain (long periods, few transits):
    with a fixed Tc, a transit that is a fraction of T14 off is fitted too
    shallow, or not at all.

    Works with or without a TLS result: *results* may be None when
    ``use_known_ephemeris`` is on; then *shape_model* (or a soft box) gives
    the unit-depth profile.

    Returns the structured array ``dt`` (fields in ``_PT_DTYPE``) or, with
    ``return_rejected=True``, ``(dt, rejected)`` where *rejected* lists every
    expected epoch that could not be used, with the reason.
    """
    P = float(P if P is not None else results.period)
    T0 = float(T0 if T0 is not None else results.T0)
    dur = float(dur if dur is not None else results.duration)
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    flux_err = np.clip(np.asarray(flux_err, float), 1e-9, None)

    unit_shape, shape_src = _unit_shape_func(results, shape_model, time, P, T0,
                                             dur, ref_depth=ref_depth)
    # coverage is judged against the highest value the shape can reach
    s_peak = float(np.nanmax(unit_shape(np.linspace(-dur, dur, 2001))))
    if not np.isfinite(s_peak) or s_peak <= 0:
        s_peak = 1.0
    use_tc = cfg.per_transit_fit_tc if fit_tc is None else bool(fit_tc)
    R = cfg.per_transit_tc_range * dur if use_tc else 0.0
    step = max(dur / 40.0, 2.0 / 1440.0)
    n_half = int(np.ceil(R / step)) if R > 0 else 0
    shifts = np.arange(-n_half, n_half + 1) * step
    half = dur * max(0.5 * cfg.n_dur_window, 0.5 + cfg.per_transit_tc_range
                     * (1 if use_tc else 0) + 0.75)

    ok_t = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    e_all = np.round((time - T0) / P).astype(int)
    e_lo = int(np.ceil((np.nanmin(time) - half - T0) / P))
    e_hi = int(np.floor((np.nanmax(time) + half - T0) / P))

    rows, rejected = [], []
    for e in range(e_lo, e_hi + 1):
        Tc = T0 + e * P
        sel = ok_t & (e_all == e) & (np.abs(time - Tc) < half)
        n = int(sel.sum())
        if n == 0:
            rejected.append((e, Tc, "no data"))
            continue
        if n < cfg.min_pts_per_transit:
            rejected.append((e, Tc, f"too few points ({n} < "
                                    f"{cfg.min_pts_per_transit})"))
            continue
        t, f = time[sel], flux[sel]
        w = 1.0 / flux_err[sel] ** 2
        dtc = t - Tc
        n_anchor = int((np.abs(dtc) > 0.5 * dur + R).sum())
        n_win = int((np.abs(dtc) < 0.5 * dur + R).sum())
        if n_anchor < 3:
            rejected.append((e, Tc, f"no baseline anchor ({n_anchor} pts "
                                    f"outside the transit window)"))
            continue
        if n_win < 3:
            rejected.append((e, Tc, f"no data in the transit window "
                                    f"({n_win} pts)"))
            continue

        chi2s, fits = [], []
        for sh in shifts:
            s = unit_shape(dtc - sh)
            A = np.column_stack([np.ones(n), dtc, -s])
            ATW = A.T * w
            try:
                cov = np.linalg.inv(ATW @ A)
            except np.linalg.LinAlgError:
                chi2s.append(np.inf)
                fits.append(None)
                continue
            p = cov @ (ATW @ f)
            chi2s.append(float(np.sum(w * (f - A @ p) ** 2)))
            fits.append((p, cov, s))
        chi2s = np.asarray(chi2s)
        if not np.isfinite(chi2s).any():
            rejected.append((e, Tc, "singular fit"))
            continue
        k = int(np.nanargmin(chi2s))
        p, cov, s = fits[k]
        best_shift = float(shifts[k])
        curv = np.nan
        if 0 < k < len(shifts) - 1 and np.all(np.isfinite(chi2s[k - 1:k + 2])):
            c0, c1, c2 = chi2s[k - 1], chi2s[k], chi2s[k + 1]
            curv = c0 - 2.0 * c1 + c2            # second difference
            if curv > 0:
                best_shift += float(np.clip(0.5 * (c0 - c2) / curv, -1, 1)) * step
        if s.max() / s_peak < cfg.min_coverage:
            rejected.append((e, Tc, f"bottom not sampled at the best "
                                    f"shift (coverage {s.max():.2f})"))
            continue

        dof = max(n - 3, 1)
        chi2r = float(chi2s[k] / dof)
        infl = max(chi2r, 1.0)
        e_depth = float(np.sqrt(cov[2, 2] * infl))
        # null model (no transit) for the detection significance
        A0 = np.column_stack([np.ones(n), dtc])
        ATW0 = A0.T * w
        p0 = np.linalg.solve(ATW0 @ A0, ATW0 @ f)
        chi2_null = float(np.sum(w * (f - A0 @ p0) ** 2))
        dchi2 = (chi2_null - chi2s[k]) / infl
        # 1-sigma interval of the mid-time from delta chi2 = infl
        good = np.where(chi2s <= chi2s[k] + infl)[0]
        edge = (k == 0 or k == len(shifts) - 1) and len(shifts) > 1
        if len(shifts) > 1 and not edge:
            if np.isfinite(curv) and curv > 0:
                # parabola through the three lowest points: delta chi2 = infl
                e_dtc = float(step * np.sqrt(2.0 * infl / curv))
            else:
                e_dtc = float(max(0.5 * (shifts[good.max()] - shifts[good.min()]),
                                  0.5 * step))
        else:
            e_dtc = float("nan")
        rows.append((int(e), float(Tc), float(p[2]), e_depth, n,
                     float(s.max()), best_shift, e_dtc,
                     float(p[2] / e_depth) if e_depth > 0 else float("nan"),
                     float(dchi2)))

    dt = np.array(rows, dtype=_PT_DTYPE) if rows else np.empty(0, dtype=_PT_DTYPE)
    if return_rejected:
        return dt, rejected
    return dt


def _ephem_pred_sigma(cov, epochs):
    """1-sigma uncertainty [d] of T0 + E * P at each epoch E, from the 2x2
    covariance of (T0, P) with T0 at epoch 0."""
    c = np.asarray(cov, float)
    E = np.asarray(epochs, float)
    return np.sqrt(np.maximum(c[0, 0] + 2.0 * E * c[0, 1] + E ** 2 * c[1, 1], 0.0))


def _ephem_constrained(cov, epochs_data, dur, cfg: Config):
    """(ok, worst_epoch, worst_sigma_d): does the fitted ephemeris predict
    every epoch with data to within cfg.ephem_adopt_max_sigma x T14?"""
    ep = np.asarray(sorted(set(int(e) for e in epochs_data)), float)
    if ep.size == 0:
        return True, None, 0.0
    sig = _ephem_pred_sigma(cov, ep)
    k = int(np.argmax(sig))
    return (bool(sig[k] <= cfg.ephem_adopt_max_sigma * dur),
            int(ep[k]), float(sig[k]))


def refine_ephemeris_from_transits(dt, P, T0, dur, cfg: Config,
                                   epochs_data=None):
    """Linear ephemeris from the measured individual mid-times.

    Uses transits with snr >= per_transit_min_snr and a measured timing
    error.  Returns a dict (``updated`` tells whether the new P, T0 should
    be adopted) or None when there are too few transits.  A high chi2_red
    of the linear fit points to TTVs or false detections; the ephemeris is
    then left alone.  So is a fit that does not predict every epoch in
    *epochs_data* (default: the epochs of *dt*) to within
    cfg.ephem_adopt_max_sigma x T14.
    """
    if dt is None or dt.size == 0:
        return None
    m = ((dt["snr"] >= cfg.per_transit_min_snr) & np.isfinite(dt["e_dTc"])
         & np.isfinite(dt["dTc"]))
    if m.sum() < 3 or np.ptp(dt["epoch"][m]) < 2:
        return dict(updated=False, n=int(m.sum()),
                    reason="fewer than 3 well-measured transits")
    ep = dt["epoch"][m].astype(float)
    y = dt["dTc"][m]
    sig = np.maximum(dt["e_dTc"][m], 0.01 * dur)
    w = 1.0 / sig ** 2
    A = np.column_stack([np.ones_like(ep), ep])
    cov = np.linalg.inv((A.T * w) @ A)
    a, b = cov @ ((A.T * w) @ y)
    resid = y - (a + b * ep)
    dof = max(int(m.sum()) - 2, 1)
    chi2r = float(np.sum(w * resid ** 2) / dof)
    scale = np.sqrt(max(chi2r, 1.0))
    out = dict(n=int(m.sum()), P=float(P + b), T0=float(T0 + a),
               eP=float(np.sqrt(cov[1, 1]) * scale),
               eT0=float(np.sqrt(cov[0, 0]) * scale),
               chi2_red=chi2r, oc_rms_min=float(np.std(resid) * 1440.0),
               epochs=ep.astype(int), oc=resid, e_oc=sig)
    # cov is for (a, b) = (T0 shift, P shift), the same as (T0, P)
    ok_c, e_w, s_w = _ephem_constrained(
        cov * scale ** 2, dt["epoch"] if epochs_data is None else epochs_data,
        dur, cfg)
    out.update(worst_epoch=e_w, worst_sigma_d=s_w)
    if chi2r > cfg.refine_chi2_max:
        out.update(updated=False,
                   reason=f"linear ephemeris fits poorly (chi2_red = "
                          f"{chi2r:.1f}): possible TTVs or false detections")
    elif not ok_c:
        out.update(updated=False,
                   reason=f"the fit predicts epoch {e_w} only to +-"
                          f"{s_w * 24:.2f} h (> {cfg.ephem_adopt_max_sigma:g} "
                          f"T14): the fitted transits do not constrain the "
                          f"whole baseline")
    else:
        out.update(updated=True, reason="ok")
    return out


def scan_transit_times(time, flux, flux_err, P, T0, dur, search_range,
                       min_snr=5.0, step=None):
    """Box search for the mid-time of every expected transit.

    For each epoch a box of width ``dur`` is slid over Tc +- search_range;
    the depth is (median baseline) - (mean in the box core) and the S/N uses
    the robust OOT scatter.  Returns a structured list of dicts.
    """
    time = np.asarray(time, float); flux = np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    step = step or max(dur / 40.0, 2.0 / 1440.0)
    core = 0.5 * dur                        # half-width of the box
    out = []
    e_lo = int(np.floor((time.min() - T0) / P)) - 1
    e_hi = int(np.ceil((time.max() - T0) / P)) + 1
    for e in range(e_lo, e_hi + 1):
        Tc = T0 + e * P
        win = np.abs(time - Tc) < search_range + 2.5 * dur
        if win.sum() < 10:
            continue
        t, f = time[win], flux[win]
        centres = Tc + np.arange(-search_range, search_range + step, step)
        best = None
        snrs = np.full(centres.size, np.nan)
        for j, c in enumerate(centres):
            d = np.abs(t - c)
            inn = d < core
            oot = (d > 0.75 * dur) & (d < 2.5 * dur)
            if inn.sum() < 3 or oot.sum() < 6 or \
                    (oot & (t < c)).sum() < 2 or (oot & (t > c)).sum() < 2:
                continue
            base = np.median(f[oot])
            sig = 1.4826 * np.median(np.abs(f[oot] - base))
            if not np.isfinite(sig) or sig <= 0:
                continue
            depth = base - np.mean(f[inn])
            snrs[j] = depth / (sig / np.sqrt(inn.sum()))
            if best is None or snrs[j] > best[0]:
                best = (snrs[j], j, depth, int(inn.sum()))
        if best is None:
            continue
        snr, j, depth, nin = best
        tc = centres[j]
        if 0 < j < centres.size - 1 and np.all(np.isfinite(snrs[j - 1:j + 2])):
            a, b, c = snrs[j - 1], snrs[j], snrs[j + 1]
            den = a - 2 * b + c
            if den < 0:
                tc += float(np.clip(0.5 * (a - c) / den, -1, 1)) * step
        at_edge = j in (0, centres.size - 1)
        out.append(dict(epoch=e, Tc_pred=Tc, Tc=tc, snr=float(snr),
                        depth=float(depth), n_in=nin, at_edge=at_edge))
    return out


def fit_linear_ephemeris(rows, P, T0, dur, min_snr=5.0, tol_frac=0.15,
                         max_rms_frac=0.25):
    """Robust (pair-consensus) linear fit of the scanned mid-times."""
    use = [r for r in rows if r["snr"] >= min_snr and not r["at_edge"]]
    if len(use) < 3:
        return None, use
    ep = np.array([r["epoch"] for r in use], float)
    tc = np.array([r["Tc"] for r in use])
    snr = np.array([r["snr"] for r in use])
    tol = tol_frac * dur
    best = None
    for i in range(len(ep)):
        for j in range(i + 1, len(ep)):
            if ep[i] == ep[j]:
                continue
            Pij = (tc[j] - tc[i]) / (ep[j] - ep[i])
            if abs(Pij - P) > 0.05 * P:
                continue
            res = tc - (tc[i] + Pij * (ep - ep[i]))
            inl = np.abs(res) < tol
            score = (inl.sum(), snr[inl].sum())
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None or best[0][0] < 3:
        return None, use
    inl = best[1]
    A = np.column_stack([np.ones(inl.sum()), ep[inl]])
    (T0n, Pn), *_ = np.linalg.lstsq(A, tc[inl], rcond=None)
    res = tc[inl] - (T0n + Pn * ep[inl])
    dof = max(int(inl.sum()) - 2, 1)
    s2 = float(np.sum(res ** 2) / dof)
    cov = np.linalg.inv(A.T @ A) * max(s2, (2.0 / 1440.0) ** 2)
    rms = float(np.sqrt(np.mean(res ** 2)))
    return dict(P=float(Pn), T0=float(T0n), eP=float(np.sqrt(cov[1, 1])),
                eT0=float(np.sqrt(cov[0, 0])), cov=cov.tolist(), n=int(inl.sum()),
                rms_min=rms * 1440.0, ok=rms < max_rms_frac * dur,
                epochs=[int(e) for e in ep[inl]],
                rejected=[int(e) for e in ep[~inl]],
                res_min=(res * 1440.0).tolist()), use


def fold_grid_search(time, flux, P, T0, dur, dT0, dP=None, step_frac=20.0,
                     core_frac=0.5):
    """Folded box search on a small (P, T0) grid around an ephemeris.

    All expected transits are stacked for every grid point, so transits
    that are individually below the noise still add up.  The reference
    epoch is placed in the middle of the data (decorrelates P and T0).
    For every trial period the data near the transits are binned in phase
    once; box sums for all trial mid-times then follow from cumulative sums.

    Returns a dict with the best P, T0 (epoch 0 of the input ephemeris),
    S/N, depth, the S/N map and bookkeeping.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]

    ep_all = np.round((time - T0) / P)
    e_ref = float(np.round(np.median(ep_all)))
    n_max = float(max(np.max(np.abs(ep_all - e_ref)), 1.0))
    if dP is None:                       # same drift budget as dT0
        dP = dT0 / n_max
    dT0_step = dur / step_frac
    dP_step = dT0_step / n_max
    T0_ref = T0 + e_ref * P
    p_grid = P + np.arange(-np.ceil(dP / dP_step), np.ceil(dP / dP_step) + 1) * dP_step
    t_grid = np.arange(-np.ceil(dT0 / dT0_step), np.ceil(dT0 / dT0_step) + 1) * dT0_step

    # only data that can be near a transit for any grid point
    reach = dT0 + n_max * dP + 2.6 * dur
    near = np.abs((time - T0_ref + 0.5 * P) % P - 0.5 * P) < reach + \
        np.abs(ep_all - e_ref) * dP
    t, f = time[near], flux[near]
    if t.size < 20:
        raise RuntimeError("too few data points near the expected transits")
    sig_pt = 1.4826 * np.median(np.abs(f - np.median(f)))

    bw = dT0_step / 4.0                  # fine phase bins
    half_span = dT0 + 2.6 * dur
    edges = np.arange(-half_span, half_span + bw, bw)
    core = core_frac * dur
    snr = np.full((p_grid.size, t_grid.size), np.nan)
    depth = np.full_like(snr, np.nan)
    n_ep = np.zeros_like(snr)
    for i, Pi in enumerate(p_grid):
        ph = (t - T0_ref + 0.5 * Pi) % Pi - 0.5 * Pi
        ep = np.round((t - T0_ref) / Pi).astype(int)
        m = np.abs(ph) < half_span
        if m.sum() < 20:
            continue
        cnt, _ = np.histogram(ph[m], edges)
        sm, _ = np.histogram(ph[m], edges, weights=f[m])
        C = np.concatenate([[0], np.cumsum(cnt)])
        S = np.concatenate([[0], np.cumsum(sm)])

        def _sum(lo, hi):
            a = np.clip(np.searchsorted(edges, lo), 0, edges.size - 1)
            b = np.clip(np.searchsorted(edges, hi), 0, edges.size - 1)
            return C[b] - C[a], S[b] - S[a]

        c = t_grid
        n_in, s_in = _sum(c - core, c + core)
        nl, sl = _sum(c - 2.5 * dur, c - 0.75 * dur)
        nr, sr = _sum(c + 0.75 * dur, c + 2.5 * dur)
        n_out, s_out = nl + nr, sl + sr
        good = (n_in >= 3) & (nl >= 3) & (nr >= 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s_out / n_out - s_in / n_in
            e = sig_pt * np.sqrt(1.0 / n_in + 1.0 / n_out)
            snr[i] = np.where(good, d / e, np.nan)
            depth[i] = np.where(good, d, np.nan)
        # number of distinct epochs inside the box core at each trial T0
        for j in np.where(good)[0]:
            sel = m & (np.abs(ph - c[j]) < core)
            n_ep[i, j] = np.unique(ep[sel]).size

    # S/N at the input ephemeris (grid point closest to (P, T0_ref))
    i0 = int(np.argmin(np.abs(p_grid - P)))
    j0 = int(np.argmin(np.abs(t_grid)))
    if not np.isfinite(snr).any():
        raise RuntimeError("no grid point with enough in- and out-of-transit data")
    ib, jb = np.unravel_index(np.nanargmax(snr), snr.shape)
    P_best = float(p_grid[ib])
    # a box core shorter than T14 gives a flat-topped S/N plateau in T0:
    # take the middle of the contiguous plateau (S/N within 1 of the peak)
    row = snr[ib]
    lo = hi = jb
    while lo > 0 and np.isfinite(row[lo - 1]) and row[lo - 1] >= row[jb] - 1.0:
        lo -= 1
    while hi < row.size - 1 and np.isfinite(row[hi + 1]) and row[hi + 1] >= row[jb] - 1.0:
        hi += 1
    T0ref_best = float(T0_ref + 0.5 * (t_grid[lo] + t_grid[hi]))
    at_edge = ib in (0, p_grid.size - 1) or jb in (0, t_grid.size - 1)
    # 1-sigma-like extent: grid points with S/N >= peak - 1
    reg = np.isfinite(snr) & (snr >= snr[ib, jb] - 1.0)
    ii, jj = np.where(reg)
    eP = max(0.5 * float(np.ptp(p_grid[ii])), 0.5 * dP_step)
    eT0 = max(0.5 * float(np.ptp(t_grid[jj])), 0.5 * dT0_step)
    return dict(P=P_best, T0=T0ref_best - e_ref * P_best, T0_ref=T0ref_best,
                e_ref=int(e_ref), snr=float(snr[ib, jb]),
                depth=float(depth[ib, jb]), n_epochs=int(n_ep[ib, jb]),
                snr_input=float(snr[i0, j0]), eP=eP, eT0_ref=eT0,
                at_edge=bool(at_edge), p_grid=p_grid, t_grid=t_grid,
                snr_map=snr, dP=float(dP), dT0=float(dT0),
                P_step=float(dP_step), T0_step=float(dT0_step))


def plot_ephem_grid(res, P_in, T0_in, cfg: Config, tic: int) -> None:
    """S/N map of the (P, T0) grid search with input and best ephemeris."""
    try:
        m = res["snr_map"]
        x = (res["t_grid"]) * 24.0                      # h, at the ref epoch
        y = (res["p_grid"] - P_in) * 1440.0             # min
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(x, y, m, shading="nearest", cmap="viridis",
                           edgecolors="none", linewidth=0, rasterized=True)
        fig.colorbar(im, ax=ax, label="box S/N (all transits stacked)")
        ax.plot(0, 0, "w+", ms=12, mew=2, label="input ephemeris")
        T0ref_in = T0_in + res["e_ref"] * P_in
        ax.plot((res["T0_ref"] - T0ref_in) * 24.0, (res["P"] - P_in) * 1440.0,
                "rx", ms=10, mew=2, label="best")
        ax.set_xlabel(f"T0 offset at reference epoch {res['e_ref']} [h]")
        ax.set_ylabel("P offset [min]")
        ax.set_title(f"TIC {tic} - folded (P, T0) grid search")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(cfg.path(f"ephem_grid_{tic}.pdf"))
        plt.close(fig)
    except Exception as exc:
        print(f"  ! ephemeris grid plot failed: {exc}")


def grid_known_ephemeris(time, flux, P, T0, dur, cfg: Config, log=None):
    """Folded (P, T0) grid search; returns (P, T0, result)."""
    dT0 = (cfg.ephem_scan_range_d if cfg.ephem_scan_range_d
           else max(2.0 * dur, 0.5))
    _log(log, f"\n--- Ephemeris check: folded (P, T0) grid search "
              f"(T0 +-{dT0 * 24:.1f} h"
              + (f", P +-{cfg.ephem_grid_dP:.4f} d" if cfg.ephem_grid_dP else "")
              + ") ---")
    try:
        res = fold_grid_search(time, flux, P, T0, dur, dT0,
                               dP=cfg.ephem_grid_dP)
    except Exception as exc:
        _log(log, f"  grid search failed ({exc}); the ephemeris is kept")
        return P, T0, None
    plot_ephem_grid(res, P, T0, cfg, cfg.tic)
    _log(log, f"  grid: {res['p_grid'].size} periods (step "
              f"{res['P_step'] * 1440:.2f} min) x {res['t_grid'].size} "
              f"mid-times (step {res['T0_step'] * 1440:.1f} min), reference "
              f"epoch {res['e_ref']}")
    _log(log, f"  S/N at the input ephemeris = {res['snr_input']:.1f}; "
              f"best S/N = {res['snr']:.1f} (depth {res['depth'] * 1e6:.0f} ppm, "
              f"{res['n_epochs']} transits in the box) at "
              f"P = {res['P']:.6f} +/- {res['eP']:.6f} d, "
              f"T0 = {res['T0_ref']:.5f} +/- {res['eT0_ref']:.5f} "
              f"(epoch {res['e_ref']}) = {res['T0']:.5f} (epoch 0)")
    _log(log, f"  S/N map -> {cfg.path(f'ephem_grid_{cfg.tic}.pdf')}")
    if res["at_edge"]:
        _log(log, "  ! best point lies on the edge of the grid: the real "
                  "ephemeris may be outside it (raise --ephem-scan-hours / "
                  "--ephem-grid-dp); the ephemeris is kept")
        return P, T0, res
    if res["snr"] < cfg.ephem_grid_min_snr or res["n_epochs"] < 3:
        _log(log, f"  grid peak not significant (S/N < "
                  f"{cfg.ephem_grid_min_snr:g} or < 3 transits); the "
                  f"ephemeris is kept")
        return P, T0, res
    ep = np.unique(np.round((np.asarray(time) - T0) / P))
    drift = float(np.max(np.abs((T0 + ep * P) - (res["T0"] + ep * res["P"]))))
    if drift < 0.1 * dur or res["snr"] < res["snr_input"] + 2.0:
        _log(log, "  input ephemeris confirmed by the grid search; kept")
        return P, T0, res
    _log(log, f"  ! the input ephemeris is off by up to {drift * 24:.2f} h "
              f"(= {drift / dur:.2f} T14) within the data; ADOPTING the grid "
              f"result (P {P:.6f} -> {res['P']:.6f} d, T0 -> {res['T0']:.5f}). "
              f"Use --ephem-search off to keep the input values.")
    return res["P"], res["T0"], res


def refine_known_ephemeris(time, flux, flux_err, P, T0, dur, cfg: Config,
                           log=None):
    """Dispatch on cfg.ephem_search: 'auto' | 'scan' | 'grid' | 'off'."""
    mode = (cfg.ephem_search or "auto").lower()
    if not cfg.ephem_scan:
        mode = "off"
    if mode == "off":
        return P, T0
    if mode in ("scan", "auto"):
        P1, T01, fit = check_known_ephemeris(time, flux, flux_err, P, T0,
                                             dur, cfg, log=log)
        if mode == "scan" or (fit is not None and fit.get("ok")):
            return P1, T01
        _log(log, "  -> falling back to the folded grid search")
    P1, T01, _ = grid_known_ephemeris(time, flux, P, T0, dur, cfg, log=log)
    return P1, T01


def check_known_ephemeris(time, flux, flux_err, P, T0, dur, cfg: Config,
                          log=None):
    """Scan for the real transits around a known ephemeris; maybe refine it.

    Returns (P, T0, info).  P/T0 are unchanged when fewer than three
    transits agree on a linear ephemeris.
    """
    R = (cfg.ephem_scan_range_d if cfg.ephem_scan_range_d
         else max(2.0 * dur, 0.5))
    _log(log, f"\n--- Ephemeris check: box search of every expected transit "
              f"within +-{R * 24:.1f} h ---")
    P1, T01, fit = P, T0, None
    for it in range(2):
        rows = scan_transit_times(time, flux, flux_err, P1, T01, dur, R,
                                  min_snr=cfg.ephem_scan_min_snr)
        fit_i, _ = fit_linear_ephemeris(rows, P1, T01, dur,
                                        min_snr=cfg.ephem_scan_min_snr)
        if it == 0:
            ep_data = [r["epoch"] for r in rows]       # epochs with data
            for r in rows:
                _log(log, f"  epoch {r['epoch']:4d}  predicted {r['Tc_pred']:.4f}"
                          f"  found {r['Tc']:.4f}  ({(r['Tc'] - r['Tc_pred']) * 24:+6.2f} h)"
                          f"  S/N = {r['snr']:6.1f}  box depth = "
                          f"{r['depth'] * 1e6:7.0f} ppm"
                          + ("  [at search limit]" if r["at_edge"] else ""))
        if fit_i is None:
            break
        fit = fit_i
        P1, T01 = fit["P"], fit["T0"]
        R = max(0.5 * dur, 4.0 * fit["rms_min"] / 1440.0)
    if fit is None or not fit["ok"]:
        _log(log, "  ephemeris check inconclusive (fewer than 3 consistent "
                  "transits); the known ephemeris is kept")
        return P, T0, fit
    # move T0 to the epoch nearest the middle of the used transits
    e_mid = int(np.round(np.median(fit["epochs"])))
    e_mid = min(fit["epochs"], key=lambda e: abs(e - e_mid))
    P_new = fit["P"]
    # fit["T0"] refers to epoch 0 of the input ephemeris; keep that epoch
    # numbering and also report the (better constrained) mid-baseline value
    T0_new = fit["T0"]
    T0_mid = fit["T0"] + e_mid * P_new
    ep_used = np.asarray(fit["epochs"], float)
    drift = float(np.max(np.abs((T0 + ep_used * P) - (T0_new + ep_used * P_new))))
    ep_all = np.asarray(sorted(set(ep_data) | set(fit["epochs"])), float)
    drift_all = float(np.max(np.abs((T0 + ep_all * P) - (T0_new + ep_all * P_new))))
    _log(log, f"  linear fit of {fit['n']} transits (epochs {fit['epochs']}"
              + (f", rejected {fit['rejected']}" if fit.get("rejected") else "")
              + f"): P = {P_new:.6f} +/- {fit['eP']:.6f} d, "
              f"T0 = {T0_new:.5f} {TIME_LABEL} (epoch 0) = {T0_mid:.5f} (epoch {e_mid}), "
              f"O-C rms = {fit['rms_min']:.1f} min")
    if drift_all < 0.1 * dur:
        _log(log, "  known ephemeris confirmed (drift < 0.1 T14 over the "
                  "baseline); kept")
        return P, T0, fit
    ok_c, e_w, s_w = _ephem_constrained(fit["cov"], ep_all, dur, cfg)
    if not ok_c:
        fit["ok"] = False
        _log(log, f"  ! refined ephemeris NOT adopted: fitted on epochs "
                  f"{min(fit['epochs'])}-{max(fit['epochs'])} only, it predicts "
                  f"epoch {e_w} (with data) to +-{s_w * 24:.2f} h "
                  f"(> {cfg.ephem_adopt_max_sigma:g} T14). It differs from the "
                  f"known ephemeris by up to {drift * 24:.2f} h at the fitted "
                  f"epochs and {drift_all * 24:.2f} h over all epochs with data; "
                  f"the known ephemeris is kept")
        return P, T0, fit
    _log(log, f"  ! the known ephemeris misses the transits by up to "
              f"{drift * 24:.2f} h (= {drift / dur:.2f} T14) at the fitted "
              f"epochs, {drift_all * 24:.2f} h over all epochs with data; "
              f"ADOPTING the refined ephemeris "
              f"(P {P:.6f} -> {P_new:.6f} d, T0 -> {T0_new:.5f}). "
              f"Use --ephem-search off to keep the input values.")
    return P_new, T0_new, fit


def sector_target_fractions(target, all_ap, sigma=0.75) -> dict:
    """Flux fraction of the target in every sector's aperture.

    Same Gaussian-PSF model as triceratops.calc_depths, evaluated with
    target.pix_coords (same frame as all_ap, see _fix_pixel_frame) and all
    stars in target.stars.  Returns {sector: fraction}.
    """
    st = target.stars
    tmag = np.asarray(st["Tmag"].values, float)
    ok = np.isfinite(tmag)
    out = {}
    for k, sec in enumerate(target.sectors):
        if k >= len(target.pix_coords) or k >= len(all_ap):
            break
        pc = np.asarray(target.pix_coords[k], float)
        ap = np.asarray(all_ap[k], float)
        if pc.shape[0] != len(st) or ap.size == 0:
            continue
        g = aperture_flux_fractions(pc[ok, 0], pc[ok, 1], tmag[ok], ap,
                                    sigma=sigma)
        # target is row 0 of the star table
        out[int(sec)] = float(g[0]) if ok[0] else float("nan")
    return out


def _sector_of_time(sectors_data, t):
    for sd in sectors_data:
        if len(sd.time) and sd.time.min() - 0.5 <= t <= sd.time.max() + 0.5:
            return int(sd.sector), float(np.median(np.diff(sd.time))) * 1440.0
    return None, float("nan")


def dilution_corrected_depths(dt, fractions, sectors_data, log=None):
    """Per-transit depths divided by the target's flux fraction per sector.

    The pipeline's aperture (percentile mask) differs from sector to sector,
    and so does the light of neighbours in it; raw aperture depths of the
    same eclipse then differ by tens of per cent.  Dividing by the target
    fraction g_s gives on-target depths that can be compared between
    sectors: constant-depth test, odd/even test (overall and per cadence).
    """
    if dt is None or dt.size == 0 or not fractions:
        return None
    rows = []
    for r in dt:
        sec, cad = _sector_of_time(sectors_data, float(r["Tc"]))
        g = fractions.get(sec, np.nan) if sec is not None else np.nan
        rows.append((int(r["epoch"]), float(r["Tc"]), sec, cad, g,
                     float(r["depth"]), float(r["e_depth"])))
    good = [x for x in rows if np.isfinite(x[4]) and x[4] > 0]
    if not good:
        return None
    dtc = np.array([(x[0], x[1], x[5] / x[4], x[6] / x[4], 0, 1.0, 0.0,
                     np.nan, x[5] / x[6] if x[6] > 0 else np.nan, 0.0)
                    for x in good], dtype=_PT_DTYPE)
    diag = depth_diagnostics(dtc)
    _log(log, "\n--- Per-transit depths corrected for the dilution in each "
              "sector's aperture ---")
    _log(log, "  epoch  sector  cadence   g_target   raw depth        on-target depth")
    for x in rows:
        if np.isfinite(x[4]) and x[4] > 0:
            _log(log, f"  {x[0]:5d}  {x[2] if x[2] is not None else -1:6d}  "
                      f"{x[3]:5.1f} min  {x[4]:7.4f}   {x[5] * 1e6:7.0f} +/- "
                      f"{x[6] * 1e6:5.0f}   {x[5] / x[4] * 1e6:7.0f} +/- "
                      f"{x[6] / x[4] * 1e6:5.0f} ppm")
        else:
            _log(log, f"  {x[0]:5d}  sector {x[2]}: no target fraction")
    gs = np.array([x[4] for x in good])
    _log(log, f"  target fraction per sector: {gs.min():.3f} - {gs.max():.3f} "
              f"(median {np.median(gs):.3f})")
    if "depth" in diag:
        _log(log, f"  weighted mean on-target depth = {diag['depth'] * 1e6:.0f} "
                  f"+/- {diag['depth_err'] * 1e6:.0f} ppm")
    if "chi2_red" in diag:
        raw = depth_diagnostics(dt)
        _log(log, f"  constant-depth chi2/dof = {diag['chi2_red']:.2f} "
                  f"(raw: {raw.get('chi2_red', float('nan')):.2f})")
    if "odd_even_sigma" in diag:
        _log(log, f"  odd/even difference (on-target) = "
                  f"{diag['odd_even_sigma']:.2f} sigma")
    # odd/even within one cadence (exposure smearing of a V shape differs)
    cads = {}
    for x, rec in zip(good, dtc):
        key = int(round(x[3])) if np.isfinite(x[3]) else -1
        cads.setdefault(key, []).append(rec)
    for key, recs in sorted(cads.items()):
        arr = np.array(recs, dtype=_PT_DTYPE)
        odd = arr[arr["epoch"] % 2 == 1]
        even = arr[arr["epoch"] % 2 == 0]
        if odd.size and even.size:
            def _wm(a):
                w = 1.0 / a["e_depth"] ** 2
                return np.sum(w * a["depth"]) / np.sum(w), 1.0 / np.sqrt(np.sum(w))
            mo, eo = _wm(odd)
            me, ee = _wm(even)
            _log(log, f"  cadence {key:4d} min: odd {mo * 1e6:7.0f} +/- "
                      f"{eo * 1e6:5.0f} ppm ({odd.size}), even {me * 1e6:7.0f} +/- "
                      f"{ee * 1e6:5.0f} ppm ({even.size}): "
                      f"{(mo - me) / np.hypot(eo, ee):+.2f} sigma")
    if "chi2_red" in diag and diag["chi2_red"] > 5:
        _log(log, "  ! depths still inconsistent after the dilution "
                  "correction: residual systematics, a varying eclipse, or an "
                  "imperfect PSF/flux model of the neighbours")
    return dict(rows=rows, diag=diag, fractions=fractions)


def report_depths_per_sector(dt, sectors_data, cfg: "Config", log=None):
    """Per-transit depths per sector: target fraction, on-target depth,
    constant-depth test and odd/even tests (overall and per cadence).

    With cfg.dilution_mode == 'sector' the light curve is already on the
    target's scale, so the measured depth IS the on-target depth; otherwise
    the depth is divided by the sector's target fraction g_s here.
    """
    if dt is None or dt.size == 0:
        return None
    applied = bool(getattr(cfg, "_dilution_applied", False))
    by_sec = {int(sd.sector): sd for sd in sectors_data}
    rows = []
    for r in dt:
        sec, cad = _sector_of_time(sectors_data, float(r["Tc"]))
        sd = by_sec.get(sec)
        g = float(getattr(sd, "g_target", np.nan)) if sd is not None else np.nan
        kind = getattr(sd, "aperture_kind", "") if sd is not None else ""
        d, e = float(r["depth"]), float(r["e_depth"])
        if applied:
            don, eon = d, e
        elif np.isfinite(g) and g > 0:
            don, eon = d / g, e / g
        else:
            don, eon = np.nan, np.nan
        rows.append((int(r["epoch"]), float(r["Tc"]), sec, cad, kind, g,
                     d, e, don, eon))
    good = [x for x in rows if np.isfinite(x[8]) and x[9] > 0]
    _log(log, "\n--- Per-transit depths per sector "
              + ("(light curve already on the target's scale)" if applied
                 else "(raw aperture depths divided by g_s)") + " ---")
    _log(log, "  epoch  sector  cadence    aperture        g_s    "
              "on-target depth")
    for x in rows:
        _log(log, f"  {x[0]:5d}  {x[2] if x[2] is not None else -1:6d}  "
                  f"{x[3]:5.1f} min  {x[4]:12s}  {x[5]:6.3f}   "
                  + (f"{x[8] * 1e6:7.0f} +/- {x[9] * 1e6:5.0f} ppm"
                     if np.isfinite(x[8]) else "n/a"))
    if not good:
        return None
    dtc = np.array([(x[0], x[1], x[8], x[9], 0, 1.0, 0.0, np.nan,
                     x[8] / x[9], 0.0) for x in good], dtype=_PT_DTYPE)
    diag = depth_diagnostics(dtc)
    out = dict(rows=rows, diag=diag, per_cadence=[])
    if "depth" in diag:
        _log(log, f"  weighted mean on-target depth = {diag['depth'] * 1e6:.0f} "
                  f"+/- {diag['depth_err'] * 1e6:.0f} ppm")
    if "chi2_red" in diag:
        _log(log, f"  constant-depth chi2/dof = {diag['chi2_red']:.2f}")
    if "odd_even_sigma" in diag:
        _log(log, f"  odd/even difference (all cadences) = "
                  f"{diag['odd_even_sigma']:.2f} sigma")
    cads = {}
    for x, rec in zip(good, dtc):
        key = round(float(x[3]), 1) if np.isfinite(x[3]) else -1.0
        cads.setdefault(key, []).append(rec)
    for key, recs in sorted(cads.items()):
        arr = np.array(recs, dtype=_PT_DTYPE)
        odd, even = arr[arr["epoch"] % 2 == 1], arr[arr["epoch"] % 2 == 0]
        if odd.size and even.size:
            def _wm(a):
                w = 1.0 / a["e_depth"] ** 2
                return np.sum(w * a["depth"]) / np.sum(w), 1.0 / np.sqrt(np.sum(w))
            mo, eo = _wm(odd)
            me, ee = _wm(even)
            zs = (mo - me) / np.hypot(eo, ee)
            out["per_cadence"].append((key, float(mo), float(eo), float(me),
                                       float(ee), float(zs)))
            _log(log, f"  cadence {key:5.1f} min: odd {mo * 1e6:7.0f} +/- "
                      f"{eo * 1e6:5.0f} ppm ({odd.size}), even {me * 1e6:7.0f} "
                      f"+/- {ee * 1e6:5.0f} ppm ({even.size}): {zs:+.2f} sigma")
    if "chi2_red" in diag and diag["chi2_red"] > 5:
        _log(log, "  ! depths inconsistent beyond their errors: residual "
                  "systematics, an imperfect PSF model of the neighbours, or "
                  "a real change of the eclipse (see the depth trend below)")
    # transits next to a data-segment edge (orbit gaps, sector ends): the
    # detrending is least reliable there, and with few transits a single
    # one can decide the depth and odd/even tests
    t_all = np.concatenate([np.asarray(sd.time, float) for sd in sectors_data
                            if len(sd.time)])
    d_edge = transit_edge_distance(t_all, dtc["Tc"])
    near = d_edge < cfg.edge_transit_d
    if near.any():
        _log(log, f"  transits within {cfg.edge_transit_d:g} d of a data-segment "
                  f"edge: " + ", ".join(f"epoch {int(e)} ({d * 24:.1f} h)"
                                        for e, d in zip(dtc["epoch"][near],
                                                        d_edge[near])))
        if (~near).sum() >= 2:
            dx = depth_diagnostics(dtc[~near])
            _log(log, f"  without them ({int((~near).sum())} transits): "
                      + (f"constant-depth chi2/dof = {dx['chi2_red']:.2f}"
                         if "chi2_red" in dx else "")
                      + (f", odd/even difference = {dx['odd_even_sigma']:.2f} sigma"
                         if "odd_even_sigma" in dx else ""))
            _summary(cfg).update(
                edge_transit_epochs=[int(e) for e in dtc["epoch"][near]],
                depth_chi2_red_excl_edge=float(dx.get("chi2_red", np.nan)),
                odd_even_sigma_excl_edge=float(dx.get("odd_even_sigma", np.nan)))
    _summary(cfg).update(
        depth_on_target_ppm=float(diag.get("depth", np.nan) * 1e6),
        depth_chi2_red=float(diag.get("chi2_red", np.nan)),
        odd_even_sigma=float(diag.get("odd_even_sigma", np.nan)),
        g_target_range=[float(np.nanmin([x[5] for x in rows])),
                        float(np.nanmax([x[5] for x in rows]))])
    return out


def _cadence_smoothed(model, cad_d, n=7):
    """Average a model over an exposure of cad_d days (n sub-samples)."""
    if not (np.isfinite(cad_d) and cad_d > 0):
        return model
    offs = (np.arange(n) / (n - 1) - 0.5) * cad_d
    return lambda ph, _m=model, _o=offs: np.mean(
        [_m(np.asarray(ph, float) + o) for o in _o], axis=0)


def per_transit_depths_ld(time, flux, ferr, sectors_data, P, T0, dur, ld_fit,
                          cfg: "Config", log=None):
    """Per-transit depths with the limb-darkened shape, smoothed per cadence.

    The first per-transit pass uses a trapezoid fitted to the mixed-cadence
    fold; for a V-shaped eclipse that shape is too narrow, and a 30-min
    sector smears the minimum more than a 200-s sector.  Here every transit
    is fitted with the intrinsic limb-darkened model averaged over its own
    sector's cadence.
    """
    mi = ld_fit.get("model_intrinsic") if ld_fit else None
    if mi is None:
        return None
    time = np.asarray(time, float)
    groups = {}
    for sd in sectors_data:
        cad = float(getattr(sd, "cadence_d", np.nan))
        if not np.isfinite(cad) and len(sd.time) > 2:
            cad = float(np.nanmedian(np.diff(sd.time)))
        key = round(cad * 1440.0, 1)
        groups.setdefault(key, []).append((float(np.min(sd.time)) - 0.5,
                                           float(np.max(sd.time)) + 0.5))
    parts = []
    for key, spans in groups.items():
        m = np.zeros(time.size, bool)
        for a, b in spans:
            m |= (time >= a) & (time <= b)
        if m.sum() < 20:
            continue
        shape = _cadence_smoothed(mi, key / 1440.0)
        grid = np.linspace(-1.5 * dur, 1.5 * dur, 3001)
        d_int = 1.0 - float(np.min(mi(grid)))       # intrinsic depth
        try:
            d, _rej = per_transit_depths(time[m], np.asarray(flux)[m],
                                         np.asarray(ferr)[m], None, cfg,
                                         P=P, T0=T0, dur=dur,
                                         shape_model=shape,
                                         return_rejected=True,
                                         ref_depth=d_int)
        except Exception as exc:
            _log(log, f"  ! per-transit LD fit failed for cadence {key} min ({exc})")
            continue
        if d.size:
            parts.append(d)
    if not parts:
        return None
    dt = np.concatenate(parts)
    dt = dt[np.argsort(dt["epoch"])]
    _log(log, f"\n--- Per-transit depths with the limb-darkened shape "
              f"(smoothed with each sector's cadence; {dt.size} transits) ---")
    for r in dt:
        _log(log, f"  epoch {r['epoch']:4d}  Tc = {r['Tc']:.5f}  "
                  f"dTc = {r['dTc'] * 1440:+6.1f} +/- {r['e_dTc'] * 1440:4.1f} min  "
                  f"depth = {r['depth'] * 1e6:8.0f} +/- {r['e_depth'] * 1e6:6.0f} ppm")
    return dt


def _read_extra_times(path):
    """Ground-based mid-times: 'BJD_TDB  err_min  [label]' per line.

    Times above 2.4e6 are full BJD and converted to BTJD (BJD - 2457000).
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                t = float(parts[0])
                e = float(parts[1]) / 1440.0 if len(parts) > 1 else np.nan
            except ValueError:
                continue
            if t > 2.4e6:
                t -= TIME_ZERO
            rows.append((t, e, parts[2] if len(parts) > 2 else "ground"))
    return rows


def etv_analysis(dt, P, T0, cfg: "Config", tic: int, log=None):
    """Eclipse timing (and depth) variations from the per-transit fits.

    * linear ephemeris, O-C scatter, chi2;
    * odd vs even mid-times (for a binary at 2P: e cos w of its orbit);
    * quadratic ephemeris (period change);
    * light-travel-time effect: sinusoid + linear ephemeris on a grid of
      outer periods; best fit and the 3-sigma amplitude limit, expressed as
      the projected size of the binary's orbit around a third body;
    * eclipse-depth trend (orbital precession changes the impact parameter;
      grazing eclipses are very sensitive to it);
    * optional ground-based mid-times (cfg.extra_times_file).
    """
    if dt is None or dt.size < 3:
        return None
    m = np.isfinite(dt["dTc"]) & np.isfinite(dt["e_dTc"]) & (dt["e_dTc"] > 0)
    if m.sum() < 3:
        return None
    E = dt["epoch"][m].astype(float)
    t = (dt["Tc"] + dt["dTc"])[m].astype(float)
    e = np.maximum(dt["e_dTc"][m].astype(float), 0.2 / 1440.0)
    src = np.array(["TESS"] * E.size, dtype=object)
    if cfg.extra_times_file:
        try:
            ext = _read_extra_times(cfg.extra_times_file)
            for tt, ee, lab in ext:
                E = np.append(E, float(np.round((tt - T0) / P)))
                t = np.append(t, tt)
                e = np.append(e, ee if np.isfinite(ee) and ee > 0 else 2.0 / 1440.0)
                src = np.append(src, lab)
            _log(log, f"  {len(ext)} extra mid-time(s) read from {cfg.extra_times_file}")
        except Exception as exc:
            _log(log, f"  ! could not read {cfg.extra_times_file}: {exc}")
    for tt, ee, lab in (getattr(cfg, "_ground_times", None) or []):
        E = np.append(E, float(np.round((tt - T0) / P)))
        t = np.append(t, tt)
        e = np.append(e, ee if np.isfinite(ee) and ee > 0 else 2.0 / 1440.0)
        src = np.append(src, lab)
    if getattr(cfg, "_ground_times", None):
        _log(log, f"  {len(cfg._ground_times)} mid-time(s) from the ground "
                  f"light curves added")
    w = 1.0 / e ** 2

    def wls(A):
        Aw = A * np.sqrt(w)[:, None]
        cov = np.linalg.pinv(Aw.T @ Aw)
        p = cov @ (Aw.T @ (t * np.sqrt(w)))
        r = t - A @ p
        return p, cov, r, float(np.sum(w * r ** 2))

    A1 = np.column_stack([np.ones_like(E), E])
    p1, c1, r1, chi1 = wls(A1)
    n = E.size
    out = dict(n=n, P=float(p1[1]), T0=float(p1[0]),
               eP=float(np.sqrt(c1[1, 1] * max(chi1 / max(n - 2, 1), 1))),
               rms_min=float(np.sqrt(np.mean(r1 ** 2)) * 1440),
               chi2_red=float(chi1 / max(n - 2, 1)))
    _log(log, f"\n--- Eclipse timing variations ({n} mid-times) ---")
    _log(log, f"  linear ephemeris: P = {out['P']:.6f} +/- {out['eP']:.6f} d; "
              f"O-C rms = {out['rms_min']:.1f} min, chi2_red = {out['chi2_red']:.2f}")
    if out["chi2_red"] > 2.0:
        _log(log, f"  ! the O-C scatter is {np.sqrt(out['chi2_red']):.1f} x the "
                  f"per-transit timing errors (errors underestimated, or real "
                  f"timing variations): every test below scales its errors by "
                  f"the chi2_red of its own model")

    def _scale(chi, k):
        """Error inflation from a model with k parameters (>= 1)."""
        return max(chi / max(n - k, 1), 1.0)
    # odd / even
    odd = (np.mod(E, 2) == 1)
    if odd.any() and (~odd).any():
        def wm(sel):
            ww = w[sel]
            return np.sum(ww * r1[sel]) / np.sum(ww), 1 / np.sqrt(np.sum(ww))
        mo, eo = wm(odd)
        me, ee = wm(~odd)
        _, _, _, chi_oe = wls(np.column_stack([np.ones_like(E), E, odd.astype(float)]))
        dlt, edl = mo - me, np.hypot(eo, ee) * np.sqrt(_scale(chi_oe, 3))
        ecosw = np.pi * dlt / (4.0 * P)            # binary at 2P: half-period offset
        eecw = np.pi * edl / (4.0 * P)
        out.update(odd_even_dt_min=float(dlt * 1440), odd_even_dt_err_min=float(edl * 1440),
                   ecosw_2P=float(ecosw), ecosw_2P_err=float(eecw))
        _log(log, f"  odd - even mid-times: {dlt * 1440:+.2f} +/- {edl * 1440:.2f} min "
                  f"({dlt / edl:+.1f} sigma); if the true period is 2P "
                  f"({2 * P:.4f} d) this gives e cos(w) = {ecosw:+.1e} +/- {eecw:.1e}")
    # quadratic
    if n >= 4:
        A2 = np.column_stack([np.ones_like(E), E, E ** 2])
        p2, c2, r2, chi2q = wls(A2)
        q, eq = p2[2], np.sqrt(c2[2, 2] * max(chi2q / max(n - 3, 1), 1))
        pdot = 2 * q / P                          # dP/dt [d/d]
        dchi2_q = (chi1 - chi2q) / _scale(chi2q, 3)
        out.update(dPdE_d=float(2 * q), dPdE_err_d=float(2 * eq),
                   Pdot=float(pdot), dchi2_quad=float(dchi2_q))
        _log(log, f"  quadratic ephemeris: dP/dE = {2 * q * 86400:+.3f} +/- "
                  f"{2 * eq * 86400:.3f} s per cycle ({q / eq:+.1f} sigma); "
                  f"delta chi2 vs linear = {dchi2_q:.1f}")
    # light-travel time effect
    span = float(t.max() - t.min())
    if n >= 6 and span > 0:
        Pg = np.exp(np.linspace(np.log(3.0 * P), np.log(3.0 * span), 400))
        best = None
        lims = []
        for Po in Pg:
            ph = 2 * np.pi * (t - t.min()) / Po
            A3 = np.column_stack([np.ones_like(E), E, np.sin(ph), np.cos(ph)])
            p3, c3, r3, chi3 = wls(A3)
            K = float(np.hypot(p3[2], p3[3]))
            sK = float(np.sqrt(0.5 * (c3[2, 2] + c3[3, 3])
                               * max(chi3 / max(n - 4, 1), 1)))
            lims.append((Po, K + 3 * sK))
            if best is None or chi3 < best[1]:
                best = (Po, chi3, K, sK)
        lims = np.array(lims)
        inside = lims[:, 0] <= span
        lim_in = float(np.median(lims[inside, 1])) if inside.any() else np.nan
        c_au_per_day = 173.1446
        dchi2_l = (chi1 - best[1]) / _scale(best[1], 4)
        out.update(ltte_best_P=float(best[0]), ltte_best_K_min=float(best[2] * 1440),
                   ltte_dchi2=float(dchi2_l),
                   ltte_lim3_min=float(lim_in * 1440),
                   ltte_lim3_au=float(lim_in * c_au_per_day))
        _log(log, f"  light-travel-time search (outer period {3 * P:.0f}-"
                  f"{3 * span:.0f} d): best P_out = {best[0]:.0f} d, amplitude "
                  f"{best[2] * 1440:.1f} +/- {best[3] * 1440:.1f} min, delta chi2 = "
                  f"{dchi2_l:.1f} (4 extra parameters: not significant "
                  f"below ~15)")
        if np.isfinite(lim_in):
            _log(log, f"  3-sigma limit on an LTTE amplitude for P_out <= "
                      f"{span:.0f} d (median over periods): {lim_in * 1440:.1f} "
                      f"min, i.e. a12 sin(i) < {lim_in * c_au_per_day:.2f} AU")
    # depth trend
    md = np.isfinite(dt["depth"]) & np.isfinite(dt["e_depth"]) & (dt["e_depth"] > 0)
    if md.sum() >= 4:
        tt = (dt["Tc"][md] - np.mean(dt["Tc"][md])) / 365.25
        dd, ed = dt["depth"][md], dt["e_depth"][md]
        ww = 1 / ed ** 2
        A = np.column_stack([np.ones_like(tt), tt])
        Aw = A * np.sqrt(ww)[:, None]
        cov = np.linalg.pinv(Aw.T @ Aw)
        pp = cov @ (Aw.T @ (dd * np.sqrt(ww)))
        rr = dd - A @ pp
        chi = float(np.sum(ww * rr ** 2) / max(md.sum() - 2, 1))
        es = float(np.sqrt(cov[1, 1] * max(chi, 1)))
        out.update(depth_slope_ppm_yr=float(pp[1] * 1e6),
                   depth_slope_err_ppm_yr=float(es * 1e6))
        _log(log, f"  eclipse-depth trend: {pp[1] * 1e6:+.0f} +/- {es * 1e6:.0f} "
                  f"ppm/yr ({pp[1] / es:+.1f} sigma, errors scaled by the "
                  f"scatter); a significant trend in a grazing eclipse points "
                  f"to a changing inclination (third body) - or to a "
                  f"changing aperture dilution")
    # plot
    try:
        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for sel, mk, lab in ((odd & (src == "TESS"), "o", "odd (TESS)"),
                             (~odd & (src == "TESS"), "s", "even (TESS)"),
                             (src != "TESS", "^", "ground")):
            if sel.any():
                axs[0].errorbar(t[sel], r1[sel] * 1440, e[sel] * 1440, fmt=mk,
                                ms=5, label=lab)
        axs[0].axhline(0, color="0.5", lw=0.8)
        axs[0].set_ylabel("O - C [min]")
        axs[0].legend(fontsize=8)
        axs[0].set_title(f"TIC {tic} - eclipse timing and depth "
                         f"(P = {out['P']:.6f} d)")
        if md.sum():
            axs[1].errorbar(dt["Tc"][md], dt["depth"][md] * 1e6,
                            dt["e_depth"][md] * 1e6, fmt="o", color="k", ms=4)
            if "depth_slope_ppm_yr" in out:
                xx = np.linspace(dt["Tc"][md].min(), dt["Tc"][md].max(), 50)
                axs[1].plot(xx, (pp[0] + pp[1] * (xx - np.mean(dt["Tc"][md]))
                                 / 365.25) * 1e6, "r-", lw=1)
        axs[1].set_ylabel("depth [ppm]")
        axs[1].set_xlabel(TIME_LABEL)
        fig.tight_layout()
        fig.savefig(cfg.path(f"etv_{tic}.pdf"))
        plt.close(fig)
        np.savetxt(cfg.path(f"etv_{tic}.txt"),
                   np.column_stack([E, t, r1 * 1440, e * 1440]),
                   header="epoch t_mid_BTJD O-C_min err_min")
    except Exception as exc:
        _log(log, f"  ! ETV plot failed: {exc}")
    _summary(cfg)["etv"] = {k: v for k, v in out.items()
                            if isinstance(v, (int, float))}
    return out


def write_run_summary(cfg: "Config", tic: int, log=None) -> None:
    """End-of-run summary block in the log and summary_<TIC>.json."""
    import json
    sm = _summary(cfg)
    _log(log, "\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    for k, v in sm.items():
        if isinstance(v, dict):
            _log(log, f"  {k}:")
            for kk, vv in v.items():
                _log(log, f"      {kk}: {vv:.6g}" if isinstance(vv, float)
                     else f"      {kk}: {vv}")
        else:
            _log(log, f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    try:
        with open(cfg.path(f"summary_{tic}.json"), "w") as fh:
            json.dump(sm, fh, indent=2, default=lambda o: float(o)
                      if isinstance(o, (np.floating, np.integer)) else str(o))
    except Exception as exc:
        _log(log, f"  ! could not write the JSON summary: {exc}")


def depth_diagnostics(dt: np.ndarray) -> dict:
    """Weighted mean depth, constancy test and odd/even test."""
    out = dict(n_transits=int(dt.size))
    if dt.size == 0:
        return out

    d, e = dt["depth"], dt["e_depth"]
    w = 1.0 / e ** 2
    mean = float(np.sum(w * d) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    out.update(depth=mean, depth_err=err)

    if dt.size > 1:
        chi2 = float(np.sum((d - mean) ** 2 / e ** 2))
        dof = dt.size - 1
        out.update(chi2=chi2, dof=dof, chi2_red=chi2 / dof)

    odd = dt["epoch"] % 2 == 1
    even = ~odd
    if odd.sum() >= 1 and even.sum() >= 1:
        d_odd = float(np.sum(w[odd] * d[odd]) / np.sum(w[odd]))
        d_even = float(np.sum(w[even] * d[even]) / np.sum(w[even]))
        sig = float(np.sqrt(1.0 / np.sum(w[odd]) + 1.0 / np.sum(w[even])))
        out.update(depth_odd=d_odd, depth_even=d_even,
                   odd_even_sigma=abs(d_odd - d_even) / sig)
    return out


def _planck_ratio_temperature(ratio, teff_star, lam_um=0.80):
    """Companion temperature whose surface brightness at lam is ratio x star's."""
    if not (np.isfinite(ratio) and ratio > 0 and np.isfinite(teff_star)):
        return float("nan")
    c2 = 14387.77                                    # um K
    x_star = c2 / (lam_um * teff_star)
    return float(c2 / (lam_um * np.log1p(np.expm1(x_star) / ratio)))


def _secondary_epoch_depths(time, flux, P, T0, phi, w_days, sig_pt):
    """Box depth of every individual event at phase phi.

    Each event gets its own quadratic baseline, fitted to the data within
    0.75 w ... 3 w of the event on both sides, and its own noise estimate
    from the residuals of that baseline.  A box depth measured against a
    flat baseline is biased by residual trends on the scale of a wide box
    (the detrending window is only a few box widths), which can turn a
    ramp in a single sector into an apparent 'secondary eclipse'.
    """
    e_lo = int(np.floor((time.min() - T0) / P - phi)) - 1
    e_hi = int(np.ceil((time.max() - T0) / P - phi)) + 1
    rows = []
    for n in range(e_lo, e_hi + 1):
        tc = T0 + (n + phi) * P
        win = np.abs(time - tc) < 3.0 * w_days
        if win.sum() < 12:
            continue
        x = time[win] - tc
        y = flux[win]
        inn = np.abs(x) < 0.5 * w_days
        oot = np.abs(x) > 0.75 * w_days
        if inn.sum() < 3 or (oot & (x < 0)).sum() < 5 or (oot & (x > 0)).sum() < 5:
            continue
        try:
            c = np.polyfit(x[oot], y[oot], 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        res = y[oot] - np.polyval(c, x[oot])
        sig = 1.4826 * np.median(np.abs(res - np.median(res)))
        if not np.isfinite(sig) or sig <= 0:
            sig = sig_pt
        d = float(np.mean(np.polyval(c, x[inn]) - y[inn]))
        # +3 parameters of the baseline: inflate the baseline term
        e = float(sig * np.sqrt(1.0 / inn.sum() + 3.0 / oot.sum()))
        rows.append((n, tc, d, e, int(inn.sum())))
    return rows


def _stack_stats(rows):
    """Weighted mean depth, its S/N, chi2_red and leave-strongest-out S/N."""
    if not rows:
        return dict(n=0, depth=np.nan, snr=np.nan, chi2r=np.nan, snr_loo=np.nan,
                    drop=None)
    d = np.array([r[2] for r in rows]); e = np.array([r[3] for r in rows])
    w = 1.0 / e ** 2
    m = float(np.sum(w * d) / np.sum(w)); em = float(1.0 / np.sqrt(np.sum(w)))
    chi2r = float(np.sum(w * (d - m) ** 2) / max(len(d) - 1, 1))
    if len(d) >= 2:
        k = int(np.argmax(w * d))                  # largest contribution
        keep = np.arange(len(d)) != k
        m2 = float(np.sum(w[keep] * d[keep]) / np.sum(w[keep]))
        snr_loo = m2 * np.sqrt(np.sum(w[keep]))
        drop = rows[k][0]
    else:
        snr_loo, drop = float("nan"), None
    return dict(n=len(d), depth=m, snr=m / em, chi2r=chi2r,
                snr_loo=float(snr_loo), drop=drop)


def secondary_eclipse_search(time, flux, P, T0, dur, dur_factors=(0.5, 0.75, 1.0, 1.5, 2.0),
                             step_frac=20.0, gap_frac=0.25,
                             min_snr=7.0, n_peaks=40):
    """Box search for a secondary eclipse at *any* orbital phase.

    For an eccentric orbit the secondary is displaced from phase 0.5
    (e cos w ~ pi/2 (phi - 0.5)) and its duration differs from the primary
    (T_sec / T_pri ~ (1 + e sin w) / (1 - e sin w)).  The light curve is
    folded on P; for every trial phase centre and every trial width
    w = factor x T14 the stacked box depth is compared with local baselines
    on both sides (0.75 w ... 2 w), using cumulative sums over fine phase
    bins (cost O(N_points + N_bins x N_widths)).

    Returns a dict with the S/N curve (max over widths) versus phase, the
    best candidate, per-phase 3-sigma depth limits and the phase coverage.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    # the primary eclipse itself (+- 1 T14) is removed from the data: it
    # must enter neither the box baselines nor the per-event baseline fits
    # (a deep primary at the edge of an event window fakes a 'secondary')
    ph0 = (time - T0 + 0.5 * P) % P - 0.5 * P
    ok &= np.abs(ph0) > dur
    time, flux = time[ok], flux[ok]
    phase = ((time - T0) / P) % 1.0                  # 0 = primary, 0.5 = classic secondary
    sig_pt = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    if not np.isfinite(sig_pt) or sig_pt <= 0:
        raise RuntimeError("cannot estimate the point scatter")

    dphi_bin = dur / P / 80.0
    nb = int(np.ceil(1.0 / dphi_bin))
    edges = np.linspace(0.0, 1.0, nb + 1)
    idx = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    sm = np.bincount(idx, weights=flux, minlength=nb)
    # periodic cumulative sums (3 copies so windows may wrap around phase 0/1)
    C = np.concatenate([[0.0], np.cumsum(np.tile(cnt, 3))])
    S = np.concatenate([[0.0], np.cumsum(np.tile(sm, 3))])

    def _win(lo, hi):                                 # lo, hi in phase, may be <0 or >1
        a = np.round((lo + 1.0) / dphi_bin).astype(int)
        b = np.round((hi + 1.0) / dphi_bin).astype(int)
        a = np.clip(a, 0, 3 * nb)
        b = np.clip(b, 0, 3 * nb)
        return C[b] - C[a], S[b] - S[a]

    step = dur / P / step_frac
    centres = np.arange(0.0, 1.0, step)
    best_snr = np.full(centres.size, np.nan)
    best_depth = np.full(centres.size, np.nan)
    best_w = np.full(centres.size, np.nan)
    lim3 = np.full(centres.size, np.nan)              # 3-sigma limit at w = T14
    for fac in dur_factors:
        w = fac * dur / P
        n_in, s_in = _win(centres - 0.5 * w, centres + 0.5 * w)
        nl, sl = _win(centres - 2.0 * w, centres - 0.75 * w)
        nr, sr = _win(centres + 0.75 * w, centres + 2.0 * w)
        good = (n_in >= 3) & (nl >= 3) & (nr >= 3)
        # keep away from the primary eclipse
        dist0 = np.minimum(centres, 1.0 - centres)
        good &= dist0 > (0.5 * (dur / P + w) + 2.0 * w + gap_frac * dur / P)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (sl + sr) / (nl + nr) - s_in / n_in
            e = sig_pt * np.sqrt(1.0 / n_in + 1.0 / (nl + nr))
            snr = np.where(good, d / e, np.nan)
        upd = np.isfinite(snr) & ~(snr <= np.nan_to_num(best_snr, nan=-np.inf))
        best_snr = np.where(upd, snr, best_snr)
        best_depth = np.where(upd, d, best_depth)
        best_w = np.where(upd, fac, best_w)
        if abs(fac - 1.0) < 1e-9:
            lim3 = np.where(good, np.maximum(d, 0.0) + 3.0 * e, np.nan)

    covered = np.isfinite(best_snr)
    if not covered.any():
        raise RuntimeError("no phase with enough in- and out-of-eclipse data")
    # robust S/N statistics over phases (the noise floor of the search)
    vals = best_snr[covered]
    med = float(np.median(vals))
    mad = float(1.4826 * np.median(np.abs(vals - med)))
    # candidates: the strongest local maxima, separated by > T14
    order = np.argsort(np.nan_to_num(best_snr, nan=-np.inf))[::-1]
    order = order[np.nan_to_num(best_snr[order], nan=-np.inf) > 0]
    peaks = []
    sep = dur / P
    for j in order:
        if not np.isfinite(best_snr[j]) or len(peaks) >= n_peaks:
            break
        if all(min(abs(centres[j] - centres[k]), 1 - abs(centres[j] - centres[k])) > sep
               for k in peaks):
            peaks.append(int(j))
    cands = []
    for j in peaks:
        w_d = float(best_w[j]) * dur
        rows = _secondary_epoch_depths(time, flux, P, T0, float(centres[j]), w_d, sig_pt)
        st = _stack_stats(rows)
        # coherent = significant, from >= 2 events, and not carried by one
        dpos = np.mean([r[2] > 0 for r in rows]) if rows else 0.0
        # a real eclipse has (nearly) the same depth in every event:
        # significant overall, not carried by one event, consistent depths
        # (chi2_red <= 4) and a dip in most of the events
        reasons = []
        if st["n"] < 2:
            reasons.append("fewer than 2 events")
        if not (st["snr"] >= min_snr):
            reasons.append("not significant")
        if not (st["snr_loo"] >= 0.5 * min_snr):
            reasons.append("carried by one event")
        # errors inflated by the scatter of the event depths (red noise,
        # residual trends): a 'detection' whose events disagree loses it
        infl = float(np.sqrt(max(st["chi2r"], 1.0))) if np.isfinite(st["chi2r"]) else np.inf
        st["snr_infl"] = float(st["snr"] / infl) if np.isfinite(infl) else float("nan")
        if not (st["snr_infl"] >= 0.7 * min_snr):
            reasons.append(f"event depths inconsistent (chi2_red {st['chi2r']:.1f}; "
                           f"S/N with scatter-inflated errors {st['snr_infl']:.1f})")
        if not (dpos >= 0.6):
            reasons.append(f"dip in only {dpos * 100:.0f} % of the events")
        st.update(j=j, phase=float(centres[j]), width_factor=float(best_w[j]),
                  snr_box=float(best_snr[j]), rows=rows, frac_dip=float(dpos),
                  reasons=reasons, coherent=not reasons)
        cands.append(st)
    good = [c for c in cands if c["coherent"]]
    best = good[0] if good else max(
        cands, key=lambda c: c["snr"] if np.isfinite(c["snr"]) else -np.inf)
    j = best["j"]
    phi_b, fac_b = best["phase"], best["width_factor"]
    n_ep = best["n"]
    phi_rel = (phi_b + 0.5) % 1.0 - 0.5               # in (-0.5, 0.5]
    return dict(phase=phi_b, snr=float(best["snr"]), depth=float(best["depth"]),
                snr_box=float(best_snr[j]), snr_loo=float(best["snr_loo"]),
                chi2r=float(best["chi2r"]), dropped_epoch=best["drop"],
                detected=bool(best["coherent"]), candidates=cands,
                epoch_rows=best["rows"],
                width_factor=fac_b, n_epochs=n_ep,
                sde=float((best_snr[j] - med) / mad) if mad > 0 else float("nan"),
                snr_median=med, snr_mad=mad,
                ecosw=float(0.5 * np.pi * (phi_b - 0.5)),
                esinw=float((fac_b - 1.0) / (fac_b + 1.0)),
                lim3_median=float(np.nanmedian(lim3)),
                lim3_max=float(np.nanmax(lim3)) if np.isfinite(lim3).any() else float("nan"),
                lim3_at_half=float(lim3[int(np.argmin(np.abs(centres - 0.5)))]),
                coverage=float(covered.mean()), centres=centres,
                snr_curve=best_snr, depth_curve=best_depth, lim3_curve=lim3,
                phase_rel=phi_rel, sig_pt=float(sig_pt))


def plot_secondary_search(res, time, flux, P, T0, dur, cfg: Config, tic: int):
    """S/N and 3-sigma limit versus phase, and the fold at the best phase."""
    try:
        fig, axs = plt.subplots(3, 1, figsize=(9, 10))
        c = res["centres"]
        axs[0].plot(c, res["snr_curve"], "k-", lw=0.7)
        axs[0].axhline(cfg.secondary_min_snr, color="r", ls="--", lw=0.8,
                       label=f"threshold {cfg.secondary_min_snr:g}")
        axs[0].axvline(0.5, color="0.6", ls=":", lw=0.8, label="phase 0.5")
        for cd in res["candidates"]:
            axs[0].plot(cd["phase"], cd["snr_box"], "o", ms=5,
                        mfc="g" if cd["coherent"] else "none", mec="g")
        axs[0].set_xlabel("orbital phase (0 = primary)")
        axs[0].set_ylabel("box S/N (best width)")
        axs[0].set_title(f"TIC {tic} - secondary-eclipse search "
                         f"({'detected' if res['detected'] else 'no coherent detection'})")
        axs[0].legend(fontsize=8)
        axs[1].plot(c, res["lim3_curve"] * 1e6, "b-", lw=0.7)
        axs[1].set_xlabel("orbital phase")
        axs[1].set_ylabel("3-sigma depth limit [ppm]\n(width = T14)")
        ph = ((np.asarray(time) - T0) / P - res["phase"] + 0.5) % 1.0 - 0.5
        x = ph * P * 24.0
        w = res["width_factor"] * dur * 24.0
        m = np.abs(x) < 3.0 * w
        axs[2].plot(x[m], np.asarray(flux)[m], ".", color="0.7", ms=2)
        if m.sum() > 10:
            edges = np.linspace(-3 * w, 3 * w, 61)
            idx = np.digitize(x[m], edges) - 1
            fb = [np.nanmedian(np.asarray(flux)[m][idx == k]) if np.any(idx == k)
                  else np.nan for k in range(60)]
            axs[2].plot(0.5 * (edges[1:] + edges[:-1]), fb, "ko", ms=3)
            lo, hi = np.nanpercentile(np.asarray(flux)[m], [1, 99])
            axs[2].set_ylim(lo, hi)
        axs[2].axvspan(-0.5 * w, 0.5 * w, color="g", alpha=0.15)
        axs[2].set_xlabel(f"hours from phase {res['phase']:.4f}")
        axs[2].set_ylabel("flux")
        fig.tight_layout()
        fig.savefig(cfg.path(f"secondary_search_{tic}.pdf"))
        plt.close(fig)
    except Exception as exc:
        print(f"  ! secondary-search plot failed: {exc}")


def plot_per_transit_depths(dt, diag, cfg: Config, tic: int) -> None:
    if dt.size == 0:
        return
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 2]))
    odd = dt["epoch"] % 2 == 1
    ax.errorbar(dt["Tc"][odd], dt["depth"][odd] * 1e6,
                yerr=dt["e_depth"][odd] * 1e6, fmt="o", color="crimson",
                label="odd epochs")
    ax.errorbar(dt["Tc"][~odd], dt["depth"][~odd] * 1e6,
                yerr=dt["e_depth"][~odd] * 1e6, fmt="s", color="navy",
                label="even epochs")
    ax.axhline(diag["depth"] * 1e6, color="k", ls="--",
               label=f"weighted mean = {diag['depth'] * 1e6:.0f} ppm")
    ax.fill_between([dt["Tc"].min(), dt["Tc"].max()],
                    (diag["depth"] - diag["depth_err"]) * 1e6,
                    (diag["depth"] + diag["depth_err"]) * 1e6,
                    color="k", alpha=0.15)
    ax.set_ylabel("Depth (ppm)")
    title = f"TIC {tic} - per-transit depths"
    if "chi2_red" in diag:
        title += f"   chi2/dof = {diag['chi2_red']:.2f}"
    if "odd_even_sigma" in diag:
        title += f"   odd/even = {diag['odd_even_sigma']:.1f} sigma"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    ok = np.isfinite(dt["e_dTc"])
    ax2.errorbar(dt["Tc"][ok], dt["dTc"][ok] * 1440.0,
                 yerr=dt["e_dTc"][ok] * 1440.0, fmt="o", color="0.2")
    if (~ok).any():
        ax2.plot(dt["Tc"][~ok], dt["dTc"][~ok] * 1440.0, "x", color="r",
                 label="at edge of search range")
        ax2.legend(loc="best", fontsize=8)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel(f"Nominal transit mid-time ({TIME_LABEL})")
    ax2.set_ylabel("Measured - nominal Tc (min)")
    fig.tight_layout()
    fig.savefig(cfg.path(f"per_transit_depths_{tic}.pdf"))
    plt.close(fig)



# --------------------------------------------------------------------------
# Centroid diagnostics
# --------------------------------------------------------------------------
def centroid_offsets(sectors_data, P, T0, dur):
    """In-transit centroid shift relative to the out-of-transit median.

    Computed per sector, because the absolute centroid position differs
    between sectors; concatenating raw positions produces a step function
    rather than a diagnostic.
    """
    rows = []
    for sd in sectors_data:
        in_tr = transit_mask(sd.time, P, dur, T0)
        if in_tr.sum() < 3 or (~in_tr).sum() < 10:
            continue
        x0 = np.nanmedian(sd.cen_x[~in_tr])
        y0 = np.nanmedian(sd.cen_y[~in_tr])
        dx = np.nanmean(sd.cen_x[in_tr]) - x0
        dy = np.nanmean(sd.cen_y[in_tr]) - y0
        sx = np.nanstd(sd.cen_x[~in_tr]) / np.sqrt(in_tr.sum())
        sy = np.nanstd(sd.cen_y[~in_tr]) / np.sqrt(in_tr.sum())
        rows.append(dict(sector=sd.sector, dx=dx, dy=dy,
                         offset=float(np.hypot(dx, dy)),
                         sigma=float(np.hypot(dx / max(sx, 1e-9),
                                              dy / max(sy, 1e-9)))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# PSF model used to convert between apertures (same as TRICERATOPS)
# --------------------------------------------------------------------------
def _gauss_pixel_fraction(xc, yc, cols, rows, sigma):
    """Fraction of a circular Gaussian PSF falling inside each unit pixel."""
    def cdf(a, c):
        return 0.5 * (1.0 + erf((a - c) / (sigma * np.sqrt(2.0))))
    fx = cdf(cols + 0.5, xc) - cdf(cols - 0.5, xc)
    fy = cdf(rows + 0.5, yc) - cdf(rows - 0.5, yc)
    return fx * fy


def aperture_flux_fractions(star_x, star_y, star_tmag, ap_pixels, sigma=0.75):
    """Relative flux contribution of each star to a given aperture.

    Parameters
    ----------
    star_x, star_y : pixel coordinates of every star, in the same frame as
        ap_pixels (i.e. full-FFI column / row).
    star_tmag : TESS magnitudes.
    ap_pixels : [[col, row], ...]

    Returns
    -------
    fractions : array, sums to 1 over the stars.
    """
    cols = ap_pixels[:, 0].astype(float)
    rows = ap_pixels[:, 1].astype(float)
    flux = 10.0 ** (-0.4 * np.asarray(star_tmag, dtype=float))
    contrib = np.array([
        f * _gauss_pixel_fraction(x, y, cols, rows, sigma).sum()
        for x, y, f in zip(star_x, star_y, flux)
    ])
    total = contrib.sum()
    return contrib / total if total > 0 else contrib


# --------------------------------------------------------------------------
# Nearby-star vetting from their own light curves
# --------------------------------------------------------------------------
def vet_nearby_stars(tic, sector, stars, P, T0, dur, depth_target, cfg: Config):
    """Test each resolved neighbour for the transit, in its own aperture.

    For every neighbour that is far enough away to be separable, we build a
    small aperture centred on it, measure the depth there with the transit
    shape fixed, and compare with the two competing predictions:

      * signal on the neighbour  ->  depth_nb = g_nb * tdepth_nb
      * signal on the target     ->  depth_nb = g_tgt * depth_target

    where g_x is the flux fraction star x contributes to that aperture.
    A neighbour is excluded when the measured depth is inconsistent with the
    first prediction at > cfg.neighbour_sigma.

    Returns a DataFrame with one row per tested neighbour.
    """
    size = int(max(cfg.cutout_size, 2 * cfg.neighbour_max_sep / 21.0 + 4))
    print(f"  TESScut neighbour cutout sector {sector} size={size}")
    tpf = _load_tesscut_tpf(tic, sector, size, cfg)

    wcs = tpf.wcs
    x_all, y_all = wcs.all_world2pix(stars["ra"].values, stars["dec"].values, 0)
    x_all = np.asarray(x_all) + tpf.column          # to full-FFI frame
    y_all = np.asarray(y_all) + tpf.row
    tmag_all = stars["Tmag"].values

    cols = np.arange(tpf.column, tpf.column + tpf.shape[2])
    rows = np.arange(tpf.row, tpf.row + tpf.shape[1])
    cgrid, rgrid = np.meshgrid(cols, rows)

    t = np.asarray(tpf.time.value, dtype=float)
    cube = np.asarray(tpf.flux.value, dtype=float)

    out = []
    for i, star in stars.iterrows():
        sep = float(star["sep (arcsec)"])
        if i == 0 or not (cfg.neighbour_min_sep <= sep <= cfg.neighbour_max_sep):
            continue
        # tdepth == 0 (set by calc_depths) means the required depth on this
        # star would exceed 100 %: it cannot produce the signal, so there is
        # nothing to test and triceratops ignores it in its NTP/NEB scenarios.
        if not float(star.get("tdepth", np.nan)) > 0:
            continue

        # circular aperture centred on the neighbour
        dist = np.hypot(cgrid - x_all[i], rgrid - y_all[i])
        amask = dist <= cfg.neighbour_ap_radius
        if amask.sum() == 0:
            continue
        ap_nb = np.column_stack([cgrid[amask], rgrid[amask]]).astype(int)

        f = np.nansum(cube[:, amask], axis=1)
        good = np.isfinite(t) & np.isfinite(f) & (f > 0)
        if good.sum() < 100:
            continue
        tn, fn = t[good], f[good]
        en = np.sqrt(np.abs(fn)) / np.median(fn)     # crude photon-noise scaling
        flat, trend = flatten(tn, fn, window_length=cfg.window_factor * dur,
                              method=cfg.detrend_method, return_trend=True,
                              mask=transit_mask(tn, P, dur, T0))
        ok = np.isfinite(flat)
        if ok.sum() < 50:
            continue

        d_meas, d_err = _fixed_shape_depth(tn[ok], flat[ok], en[ok], P, T0, dur)
        if not np.isfinite(d_meas):
            continue

        # flux fractions in the neighbour aperture
        g = aperture_flux_fractions(x_all, y_all, tmag_all, ap_nb,
                                    sigma=cfg.psf_sigma)
        g_nb, g_tgt = float(g[i]), float(g[0])
        d_required = float(star.get("tdepth", np.nan))    # from calc_depths

        pred_on_nb = g_nb * d_required if np.isfinite(d_required) else np.nan
        pred_on_target = g_tgt * depth_target

        excluded = (np.isfinite(pred_on_nb)
                    and abs(d_meas - pred_on_nb) > cfg.neighbour_sigma * d_err)

        out.append(dict(ID=int(star["ID"]), sep=sep, Tmag=float(star["Tmag"]),
                        g_nb=g_nb, g_target=g_tgt,
                        depth_measured=d_meas, depth_err=d_err,
                        depth_if_on_neighbour=pred_on_nb,
                        depth_if_on_target=pred_on_target,
                        excluded=bool(excluded)))

    return pd.DataFrame(out)


def _fixed_shape_depth(time, flux, flux_err, P, T0, dur):
    """Phase-folded depth with a trapezoid of fixed duration; returns (d, e)."""
    ph = (time - T0 + 0.5 * P) % P - 0.5 * P
    s = np.clip((0.5 * dur - np.abs(ph)) / (0.1 * dur), 0.0, 1.0)  # soft edges
    sel = np.abs(ph) < 2.0 * dur
    if sel.sum() < 20 or s[sel].max() < 0.5:
        return np.nan, np.nan
    n = int(sel.sum())
    w = 1.0 / np.clip(flux_err[sel], 1e-12, None) ** 2
    A = np.column_stack([np.ones(n), -s[sel]])
    ATW = A.T * w
    try:
        cov = np.linalg.inv(ATW @ A)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    p = cov @ (ATW @ flux[sel])
    chi2r = float(np.sum(w * (flux[sel] - A @ p) ** 2) / max(n - 2, 1))
    return float(p[1]), float(np.sqrt(cov[1, 1] * max(chi2r, 1.0)))


# --------------------------------------------------------------------------
# Difference imaging
# --------------------------------------------------------------------------
def _prf_model(p, cols, rows):
    """Circular Gaussian approximation of the TESS PRF + constant background."""
    amp, xc, yc, sig, bkg = p
    r2 = (cols - xc) ** 2 + (rows - yc) ** 2
    return amp * np.exp(-0.5 * r2 / sig ** 2) + bkg


def fit_source_position(image, err=None, col0=0, row0=0, cfg: Config = None,
                        guess=None, box: float = 1.5):
    """Fit a PRF to an image and return the source position in FFI pixels.

    guess = (x, y): start there and keep the position within +-box pixels
    (v4: the fit to the direct image used to jump to a brighter neighbour).
    """
    cfg = cfg or Config()
    ny, nx = image.shape
    cols, rows = np.meshgrid(np.arange(nx) + col0, np.arange(ny) + row0)
    finite = np.isfinite(image)
    if finite.sum() < 8:
        return None

    img = np.where(finite, image, 0.0)
    w = np.ones_like(img)
    if err is not None:
        e = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err))
        w = 1.0 / np.clip(e, 1e-12, None)

    bkg0 = float(np.nanmedian(img))
    if guess is not None and all(np.isfinite(guess)):
        gx, gy = float(guess[0]), float(guess[1])
        ix = int(np.clip(round(gx - col0), 0, nx - 1))
        iy = int(np.clip(round(gy - row0), 0, ny - 1))
        k = (iy, ix)
        x0, y0_ = gx, gy
        xlo, xhi = max(col0 - 0.5, gx - box), min(col0 + nx - 0.5, gx + box)
        ylo, yhi = max(row0 - 0.5, gy - box), min(row0 + ny - 0.5, gy + box)
    else:
        k = np.unravel_index(np.argmax(np.where(finite, image, -np.inf)),
                             image.shape)
        x0, y0_ = float(cols[k]), float(rows[k])
        xlo, xhi = col0 - 0.5, col0 + nx - 0.5
        ylo, yhi = row0 - 0.5, row0 + ny - 0.5
    amp0 = float(img[k] - bkg0)
    if not np.isfinite(amp0) or amp0 <= 0:
        amp0 = float(np.nanmax(img) - bkg0)
    if not np.isfinite(amp0) or amp0 <= 0:
        return None

    p0 = [amp0, x0, y0_, cfg.di_prf_sigma_init, bkg0]
    lo = [0.0, xlo, ylo, cfg.di_prf_sigma_bounds[0], -np.inf]
    hi = [np.inf, xhi, yhi, cfg.di_prf_sigma_bounds[1], np.inf]
    p0[1] = float(np.clip(p0[1], xlo + 1e-6, xhi - 1e-6))
    p0[2] = float(np.clip(p0[2], ylo + 1e-6, yhi - 1e-6))

    def resid(p):
        return ((_prf_model(p, cols, rows) - img) * w)[finite]

    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), method="trf")
    except Exception:
        return None
    if not sol.success:
        return None

    amp, xc, yc, sig, bkg = sol.x
    noise = np.nanmedian(err) if err is not None else np.nanstd(img[finite])
    snr = amp / noise if (noise and np.isfinite(noise) and noise > 0) else np.nan
    return dict(amp=float(amp), x=float(xc), y=float(yc),
                sigma=float(sig), bkg=float(bkg), snr=float(snr))


def build_difference_image(sd: SectorData, P, T0, dur, cfg: Config):
    """Out-of-transit minus in-transit image, averaged over the epochs.

    The out-of-transit reference is taken from a window immediately around
    each transit rather than from the whole sector, so that slow scattered
    light and pointing drift largely cancel in the difference.
    """
    t, cube = sd.cube_time, sd.cube
    if t is None or cube is None:
        return None

    ph = (t - T0 + 0.5 * P) % P - 0.5 * P
    in_w = cfg.di_in_frac * 0.5 * dur
    in_mask = np.abs(ph) < in_w
    out_mask = ((np.abs(ph) > cfg.di_out_gap * 0.5 * dur)
                & (np.abs(ph) < cfg.di_out_span * 0.5 * dur))

    epochs = np.round((t - T0) / P).astype(int)
    diffs, weights, n_used = [], [], 0
    for e in np.unique(epochs):
        sel_in = in_mask & (epochs == e)
        sel_out = out_mask & (epochs == e)
        if sel_in.sum() < cfg.di_min_frames or sel_out.sum() < cfg.di_min_frames:
            continue
        img_in = np.nanmean(cube[sel_in], axis=0)
        img_out = np.nanmean(cube[sel_out], axis=0)
        if not np.isfinite(img_in).any() or not np.isfinite(img_out).any():
            continue
        diffs.append(img_out - img_in)
        weights.append(float(sel_in.sum()))
        n_used += 1

    if n_used == 0:
        return None

    diffs = np.array(diffs)
    w = np.array(weights)
    diff = np.average(diffs, axis=0, weights=w)

    if n_used >= 3:
        diff_err = np.std(diffs, axis=0, ddof=1) / np.sqrt(n_used)
    else:
        # single/double epoch: propagate the frame-to-frame scatter instead
        scatter = np.nanstd(cube[out_mask], axis=0)
        n_in = max(int(in_mask.sum()), 1)
        n_out = max(int(out_mask.sum()), 1)
        diff_err = scatter * np.sqrt(1.0 / n_in + 1.0 / n_out)
    diff_err = np.where(np.isfinite(diff_err) & (diff_err > 0),
                        diff_err, np.nanmedian(diff_err))

    direct = np.nanmean(cube[out_mask], axis=0)
    return dict(diff=diff, diff_err=diff_err, direct=direct,
                n_epochs=n_used, per_epoch=diffs)


def difference_image_analysis(sectors_data, stars, P, T0, dur, cfg: Config,
                              tic: int):
    """Localise the transit source with difference imaging.

    Returns
    -------
    summary : DataFrame, one row per sector, with the offset of the
        difference-image centroid from the target position.
    star_table : DataFrame, one row per (sector, nearby star), giving the
        distance of each star from the difference-image centroid in units
        of the centroid uncertainty, and whether it is excluded.
    """
    summaries, star_rows, no_epochs = [], [], []
    c_target = SkyCoord(stars.iloc[0]["ra"] * u.deg,
                        stars.iloc[0]["dec"] * u.deg)

    for sd in sectors_data:
        di = build_difference_image(sd, P, T0, dur, cfg)
        if di is None:
            no_epochs.append(int(sd.sector))
            continue
        xt, yt = sd.wcs.world_to_pixel(c_target)
        xt, yt = float(xt) + sd.col0, float(yt) + sd.row0

        fit_diff = fit_source_position(di["diff"], di["diff_err"],
                                       col0=sd.col0, row0=sd.row0, cfg=cfg)
        # direct image: the target's own PRF (start at, and stay near, it)
        fit_dir = fit_source_position(di["direct"], None,
                                      col0=sd.col0, row0=sd.row0, cfg=cfg,
                                      guess=(xt, yt), box=1.0)
        if fit_diff is None:
            print(f"  ! sector {sd.sector}: PRF fit to the difference "
                  "image failed")
            continue

        # --- Monte Carlo uncertainty on the difference-image centroid ---
        xs, ys = [], []
        rng = np.random.default_rng(12345 + sd.sector)
        for _ in range(cfg.di_mc):
            noisy = di["diff"] + rng.normal(0.0, di["diff_err"])
            f = fit_source_position(noisy, di["diff_err"],
                                    col0=sd.col0, row0=sd.row0, cfg=cfg)
            if f is not None:
                xs.append(f["x"])
                ys.append(f["y"])
        if len(xs) < 10:
            sig_pix = np.nan
        else:
            # per-axis RMS of the two coordinate scatters.  Note that the MC
            # adds noise on top of an image that already contains noise, so
            # this overestimates the true positional error by ~sqrt(2).  That
            # margin is kept deliberately: it makes the exclusion of nearby
            # stars below conservative rather than optimistic.
            sig_pix = float(np.hypot(np.std(xs), np.std(ys)) / np.sqrt(2.0))

        # statistical error + systematic floor (PRF model, pointing)
        sig_stat = sig_pix
        if np.isfinite(sig_pix):
            sig_pix = float(np.hypot(sig_pix, cfg.di_sys_floor_px))

        # --- positions on the sky ---
        c_diff = sd.wcs.pixel_to_world(fit_diff["x"] - sd.col0,
                                       fit_diff["y"] - sd.row0)
        sep_cat = float(c_diff.separation(c_target).arcsec)

        # differential offset: cancels most of the PRF-model error
        if fit_dir is not None:
            dx = fit_diff["x"] - fit_dir["x"]
            dy = fit_diff["y"] - fit_dir["y"]
            sep_dir = float(np.hypot(dx, dy))
        else:
            sep_dir = np.nan

        scale = float(np.abs(sd.wcs.proj_plane_pixel_scales()[0]
                             .to(u.arcsec).value))
        summaries.append(dict(
            sector=sd.sector, n_epochs=di["n_epochs"],
            snr=fit_diff["snr"],
            x_diff=fit_diff["x"], y_diff=fit_diff["y"],
            sigma_stat_pix=sig_stat,
            sigma_pix=sig_pix, sigma_arcsec=sig_pix * scale,
            offset_cat_arcsec=sep_cat,
            offset_cat_sigma=sep_cat / (sig_pix * scale)
            if np.isfinite(sig_pix) and sig_pix > 0 else np.nan,
            offset_direct_pix=sep_dir,
            offset_direct_arcsec=sep_dir * scale,
            offset_direct_sigma=sep_dir / sig_pix
            if np.isfinite(sig_pix) and sig_pix > 0 else np.nan,
        ))

        # --- which stars are consistent with the centroid? ---
        trustworthy = (np.isfinite(sig_pix) and sig_pix > 0
                       and np.isfinite(fit_diff["snr"])
                       and fit_diff["snr"] >= cfg.di_min_snr)
        if trustworthy:
            xs_star, ys_star = sd.wcs.world_to_pixel(
                SkyCoord(stars["ra"].values * u.deg,
                         stars["dec"].values * u.deg))
            xs_star = np.asarray(xs_star) + sd.col0
            ys_star = np.asarray(ys_star) + sd.row0
            for i, row in stars.reset_index(drop=True).iterrows():
                d_pix = float(np.hypot(xs_star[i] - fit_diff["x"],
                                       ys_star[i] - fit_diff["y"]))
                nsig = d_pix / sig_pix
                star_rows.append(dict(
                    sector=sd.sector, ID=int(row["ID"]),
                    is_target=(i == 0),
                    sep_target_arcsec=float(row["sep (arcsec)"]),
                    dist_from_centroid_pix=d_pix,
                    dist_from_centroid_arcsec=d_pix * scale,
                    n_sigma=nsig,
                    excluded=bool(nsig > cfg.di_sigma and i != 0),
                ))

        # --- plot ---
        _plot_difference_image(sd, di, fit_diff, fit_dir, stars, cfg, tic)

    if no_epochs:
        print(f"  difference imaging: no transit in sector(s) {no_epochs}")
    summary = pd.DataFrame(summaries)
    star_table = pd.DataFrame(star_rows)
    return summary, star_table


def _plot_difference_image(sd, di, fit_diff, fit_dir, stars, cfg, tic):
    ext = (sd.col0 - 0.5, sd.col0 + di["diff"].shape[1] - 0.5,
           sd.row0 - 0.5, sd.row0 + di["diff"].shape[0] - 0.5)
    snr_map = di["diff"] / di["diff_err"]

    xs_star, ys_star = sd.wcs.world_to_pixel(
        SkyCoord(stars["ra"].values * u.deg, stars["dec"].values * u.deg))
    xs_star = np.asarray(xs_star) + sd.col0
    ys_star = np.asarray(ys_star) + sd.row0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, img, title in zip(
            axes,
            [di["direct"], di["diff"], snr_map],
            ["Direct (out of transit)", "Difference (out - in)",
             "Difference / sigma"]):
        im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("column")
        inside = ((xs_star > ext[0]) & (xs_star < ext[1])
                  & (ys_star > ext[2]) & (ys_star < ext[3]))
        ax.scatter(xs_star[inside], ys_star[inside], s=30,
                   facecolors="none", edgecolors="w", lw=0.8)
        ax.scatter(xs_star[0], ys_star[0], marker="*", s=180,
                   facecolors="none", edgecolors="orange", lw=1.2,
                   label="target")
        ax.scatter([fit_diff["x"]], [fit_diff["y"]], marker="x", s=80,
                   color="red", lw=2, label="diff. centroid")
        if fit_dir is not None:
            ax.scatter([fit_dir["x"]], [fit_dir["y"]], marker="+", s=80,
                       color="cyan", lw=1.5, label="direct centroid")
    axes[0].set_ylabel("row")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"TIC {tic} - sector {sd.sector} - "
                 f"{di['n_epochs']} transit epochs", fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg.path(f"difference_image_{tic}_s{sd.sector:02d}.pdf"))
    plt.close(fig)


def combine_difference_imaging(star_table, cfg: Config):
    """Stars excluded in *every* sector where the centroid was measurable."""
    if star_table is None or len(star_table) == 0:
        return []
    drop = []
    for sid, grp in star_table[~star_table["is_target"]].groupby("ID"):
        if len(grp) and bool(grp["excluded"].all()):
            drop.append(int(sid))
    return drop


# --------------------------------------------------------------------------
# Limb-darkened transit model (optional refinement of the trapezoid)
# --------------------------------------------------------------------------
def _lens_area(R, z, p):
    """Overlap area of a disk of radius R (at the origin) and a disk of
    radius p whose centre is at distance z.  Broadcasts over R and z."""
    R = np.asarray(R, float)
    z = np.asarray(z, float)
    full = z <= np.abs(R - p)
    part = (z > np.abs(R - p)) & (z < R + p)
    Rm = np.where(part, R, 1.0)
    zm = np.where(part, z, 1.0)
    c1 = np.clip((zm ** 2 + Rm ** 2 - p ** 2) / (2 * zm * Rm), -1, 1)
    c2 = np.clip((zm ** 2 + p ** 2 - Rm ** 2) / (2 * zm * p), -1, 1)
    rad = ((-zm + Rm + p) * (zm + Rm - p) * (zm - Rm + p) * (zm + Rm + p))
    lens = (Rm ** 2 * np.arccos(c1) + p ** 2 * np.arccos(c2)
            - 0.5 * np.sqrt(np.clip(rad, 0, None)))
    return np.where(full, np.pi * np.minimum(R, p) ** 2,
                    np.where(part, lens, 0.0))


def quad_ld_transit(z, p, u1, u2, n_annuli=150):
    """Normalised flux of a quadratic-limb-darkened star occulted by a dark
    disk of radius p (stellar radii) at sky separation z (stellar radii).

    The star is cut into annuli (equally spaced in mu = cos(theta), i.e.
    dense near the limb) and the exact circle-circle overlap area is used in
    each.  No external dependency; agrees with batman to < 1 ppm for
    n_annuli >= 150 (checked for b = 0 ... 1.05, p up to 0.3).
    """
    z = np.maximum(np.atleast_1d(np.asarray(z, float)), 1e-9)
    mu_e = np.linspace(1.0, 0.0, n_annuli + 1)
    r_e = np.sqrt(1.0 - mu_e ** 2)
    mu_m = 0.5 * (mu_e[:-1] + mu_e[1:])
    inten = 1.0 - u1 * (1 - mu_m) - u2 * (1 - mu_m) ** 2
    A = _lens_area(r_e[None, :], z[:, None], p)
    blocked = (np.diff(A, axis=1) * inten).sum(axis=1)
    total = (np.pi * np.diff(r_e ** 2) * inten).sum()
    return 1.0 - blocked / total


def limb_darkened_lc(t, P, p, aRs, b, u1, u2, t0s=0.0, f0=1.0,
                     exp_time=0.0, n_ss=5, n_annuli=150):
    """Circular-orbit transit light curve vs time from mid-transit [d].

    b = a/Rs * cos(i).  The model is averaged over `exp_time` [d] (n_ss
    sub-samples) so it matches the binned, phase-folded data.
    """
    t = np.asarray(t, float)
    if exp_time > 0 and n_ss > 1:
        off = (np.arange(n_ss) + 0.5) / n_ss - 0.5
        tt = (t[:, None] - t0s + off[None, :] * exp_time).ravel()
    else:
        tt = t - t0s
        n_ss = 1
    ph = 2.0 * np.pi * tt / P
    cosi = b / aRs
    z = aRs * np.sqrt(np.sin(ph) ** 2 + (cosi * np.cos(ph)) ** 2)
    z = np.where(np.cos(ph) > 0, z, 1.0e3)        # no transit on the far side
    flux = quad_ld_transit(z, p, u1, u2, n_annuli=n_annuli) * f0
    return flux.reshape(-1, n_ss).mean(axis=1) if n_ss > 1 else flux


def _u_from_q(q1, q2):
    """Kipping (2013) parametrisation -> quadratic coefficients."""
    s = np.sqrt(q1)
    return 2.0 * s * q2, s * (1.0 - 2.0 * q2)


def _q_from_u(u1, u2):
    q1 = (u1 + u2) ** 2
    q2 = u1 / (2.0 * (u1 + u2)) if (u1 + u2) > 0 else 0.3
    return float(np.clip(q1, 0.01, 0.99)), float(np.clip(q2, 0.01, 0.99))


def fit_limb_darkened(t, f, P, dur, depth, cfg: Config, u_claret=None,
                      exp_time=0.0):
    """Least-squares fit of a limb-darkened transit to the folded (binned) LC.

    Free: Rp/Rs, b, a/Rs, small T0 shift, baseline f0 (+ Kipping q1,q2 when
    cfg.ld_mode == 'free').  P is fixed (the LC is folded on it).  Limb
    darkening is otherwise fixed to cfg.ld_coeffs ('fixed') or to the Claret
    values from the TIC / TLS catalogue ('claret').  Circular orbit assumed.

    Returns dict with parameters, 1-sigma errors (covariance scaled by the
    reduced chi2 of this fit), derived quantities and model callables.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if t.size < 15:
        raise RuntimeError("too few points for a limb-darkened fit")

    mode = cfg.ld_mode
    if mode == "fixed" and cfg.ld_coeffs is not None:
        u_fix = tuple(float(x) for x in cfg.ld_coeffs)
    elif u_claret is not None and len(u_claret) >= 2:
        u_fix = (float(u_claret[0]), float(u_claret[1]))
    else:
        u_fix = (0.4, 0.3)
    free_ld = (mode == "free")

    oot = np.abs(t) > 0.75 * dur
    sig0 = _mad_sigma(f[oot]) if oot.sum() >= 5 else float("nan")
    if not np.isfinite(sig0) or sig0 <= 0:
        sig0 = max(depth * 0.01, 1e-5)

    p0 = float(np.clip(np.sqrt(max(depth, 1e-6)) * 0.97, 0.01, 0.6))
    names = ["rp", "b", "aRs", "t0s", "f0"] + (["q1", "q2"] if free_ld else [])
    lo = [0.005, 0.0, 1.2, -0.3 * dur, 0.99] + ([0.01, 0.01] if free_ld else [])
    hi = [0.9, 1.3, 300.0, 0.3 * dur, 1.01] + ([0.99, 0.99] if free_ld else [])

    def unpack(x):
        rp, b, aRs, t0s, f0 = x[:5]
        u1, u2 = _u_from_q(x[5], x[6]) if free_ld else u_fix
        return rp, b, aRs, t0s, f0, u1, u2

    def model_x(x, tt, et=exp_time):
        rp, b, aRs, t0s, f0, u1, u2 = unpack(x)
        return limb_darkened_lc(tt, P, rp, aRs, b, u1, u2, t0s=t0s, f0=f0,
                                exp_time=et)

    def resid(x):
        return (f - model_x(x, t)) / sig0

    best = None
    for b0 in (0.2, 0.55, 0.85):
        # a/Rs from the duration: T14 ~ P/pi * sqrt((1+p)^2 - b^2) / aRs
        arg = max((1 + p0) ** 2 - b0 ** 2, 0.05)
        a0 = float(np.clip(P / (np.pi * dur) * np.sqrt(arg), 1.5, 250.0))
        x0 = [p0, b0, a0, 0.0, 1.0] + ([0.35, 0.3] if free_ld else [])
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                                max_nfev=120)
        except Exception as exc:
            print(f"  ! limb-darkened start b0={b0} failed: {exc}")
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("limb-darkened fit failed for all starting points")

    x = best.x
    n_free = len(x)
    dof = max(t.size - n_free, 1)
    rss = float(np.sum((f - model_x(x, t)) ** 2))
    chi2_red_sig0 = rss / (sig0 ** 2 * dof)
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * chi2_red_sig0
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        perr = np.full(n_free, np.nan)

    rp, b, aRs, t0s, f0, u1, u2 = unpack(x)
    cosi = b / aRs
    inc = float(np.degrees(np.arccos(np.clip(cosi, -1, 1))))
    sini = np.sqrt(max(1 - cosi ** 2, 1e-12))

    def _dur(k):             # k = 1 + p (T14) or 1 - p (T23); b must be < k
        if b >= k:
            return float("nan")
        return float(P / np.pi * np.arcsin(
            np.clip(np.sqrt(k ** 2 - b ** 2) / (aRs * sini), 0, 1)))

    tg = np.linspace(-1.5 * dur, 1.5 * dur, 1501)
    depth_ld = float(f0 - np.min(model_x(x, tg, et=0.0)))
    rho = 3.0 * np.pi * aRs ** 3 / (6.674e-8 * (P * 86400.0) ** 2)  # g/cm^3

    out = dict(
        mode=mode, n_free=n_free, dof=dof, sigma0_ppm=sig0 * 1e6,
        rms_ppm=float(np.sqrt(rss / t.size)) * 1e6,
        chi2_red_vs_oot_scatter=chi2_red_sig0,
        params={n: (float(v), float(e)) for n, v, e in zip(names, x, perr)},
        rp=rp, b=b, aRs=aRs, inc_deg=inc, t0_shift_d=t0s, f0=f0,
        u1=float(u1), u2=float(u2),
        depth_ppm=depth_ld * 1e6, T14_h=_dur(1 + rp) * 24, T23_h=_dur(1 - rp) * 24,
        rho_star_cgs=float(rho),
        model=lambda tt, _x=x: model_x(_x, np.asarray(tt, float)),
        model_intrinsic=lambda tt, _x=x: model_x(_x, np.asarray(tt, float), et=0.0),
    )
    return out


def fit_limb_darkened_lc(tp, lc, P, dur, depth, star, sectors_data,
                         cfg: Config, log=None):
    """Fit the limb-darkened model to the binned folded LC and log the result.

    The exposure smoothing is estimated from the bin width and the native
    cadence of the sectors that went into the fold.
    """
    t_b = np.asarray(lc.time.value, float)
    f_b = np.asarray(lc.flux.value, float)
    binsize = cfg.bin_minutes / 1440.0
    n_b = np.array([np.count_nonzero(np.abs(np.asarray(tp, float) - t0)
                                     <= 0.5 * binsize) for t0 in t_b])
    good = (n_b >= 3) & np.isfinite(f_b)

    exp_time = effective_exposure_d(sectors_data, cfg.bin_minutes)
    u_claret = star.get("u") if isinstance(star, dict) else None

    fit = fit_limb_darkened(t_b[good], f_b[good], P, dur, depth, cfg,
                            u_claret=u_claret, exp_time=exp_time)
    fit["t_bins"], fit["f_bins"], fit["exp_time_d"] = t_b[good], f_b[good], exp_time

    _log(log, "\n--- Limb-darkened transit fit of the folded LC "
              f"(LD mode: {fit['mode']}) ---")
    _log(log, f"  smoothing box (bin + native cadence): {exp_time * 1440:.1f} min,"
              f"  {good.sum()} bins,  {fit['n_free']} free parameters")
    for k, (v, e) in fit["params"].items():
        _log(log, f"  {k:4s} = {v:.5f} +/- {e:.5f}")
    _log(log, f"  u1, u2 = {fit['u1']:.3f}, {fit['u2']:.3f}"
              + ("  (fitted)" if fit["mode"] == "free" else "  (fixed)"))
    _log(log, f"  Rp/Rs = {fit['rp']:.4f},  b = {fit['b']:.3f},  "
              f"a/Rs = {fit['aRs']:.3f},  i = {fit['inc_deg']:.2f} deg")
    _log(log, f"  depth = {fit['depth_ppm']:.0f} ppm,  "
              f"T14 = {fit['T14_h']:.3f} h,  T23 = {fit['T23_h']:.3f} h")
    _log(log, f"  rho_star (circular orbit) = {fit['rho_star_cgs']:.3f} g/cm^3")
    try:
        _rp, _b = float(fit["rp"]), float(fit["b"])
        fit["grazing"] = bool(_b > 1.0 - _rp)
        _summary(cfg)["ld_fit"] = dict(rp=_rp, b=_b, grazing=fit["grazing"],
                                       depth_ppm=float(fit["depth_ppm"]),
                                       T14_h=float(fit["T14_h"]))
        if fit["grazing"]:
            _log(log, f"  ! GRAZING geometry: b + Rp/Rs = {_b + _rp:.2f} > 1 "
                      f"(no flat bottom, T23 undefined). Rp/Rs and b are "
                      f"strongly degenerate: the radius ratio (and a planet "
                      f"radius derived from it) is NOT constrained; depth and "
                      f"T14 are. A V-shaped, grazing eclipse is typical of "
                      f"eclipsing binaries.")
    except (KeyError, TypeError, ValueError):
        pass
    mass, rad = star.get("mass", np.nan), star.get("rad", np.nan)
    if np.isfinite(mass) and np.isfinite(rad) and rad > 0:
        rho_tic = 1.41 * mass / rad ** 3          # rho_sun = 1.41 g/cm^3
        _log(log, f"  rho_star from TIC mass/radius       = {rho_tic:.3f} g/cm^3"
                  "   (consistency check)")
    _log(log, f"  residual rms = {fit['rms_ppm']:.0f} ppm  "
              f"(OOT bin scatter {fit['sigma0_ppm']:.0f} ppm, "
              f"chi2_red vs that = {fit['chi2_red_vs_oot_scatter']:.2f})")
    return fit


def plot_ld_fit(tp, fp, fit, cfg: Config, tic: int) -> None:
    tg = np.linspace(min(np.min(tp), -1e-3), max(np.max(tp), 1e-3), 800)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    a1.plot(tp, fp, ".", color="0.8", ms=1.5, zorder=0, rasterized=True)
    a1.plot(fit["t_bins"], fit["f_bins"], "ko", ms=4, zorder=2, label="binned")
    a1.plot(tg, fit["model"](tg), "r-", lw=1.8, zorder=3,
            label="limb-darkened fit (smoothed)")
    a1.plot(tg, fit["model_intrinsic"](tg), "b--", lw=1.0, zorder=3,
            label="intrinsic")
    lo = np.nanmin(fit["f_bins"]) - 0.002
    a1.set_ylim(lo, 1.0 + 0.002)
    a1.set_ylabel("Normalised flux")
    a1.set_title(f"TIC {tic}: Rp/Rs = {fit['rp']:.4f}, b = {fit['b']:.2f}, "
                 f"a/Rs = {fit['aRs']:.2f}, depth = {fit['depth_ppm']:.0f} ppm")
    a1.legend(loc="lower right", fontsize=8)
    res = (fit["f_bins"] - fit["model"](fit["t_bins"])) * 1e6
    a2.axhline(0, color="r", lw=0.8)
    a2.plot(fit["t_bins"], res, "ko", ms=3)
    a2.set_xlabel("Time from mid-transit (d)")
    a2.set_ylabel("Residual (ppm)")
    fig.tight_layout()
    fig.savefig(cfg.path(f"ld_fit_{tic}.pdf"))
    plt.close(fig)
    np.savetxt(cfg.path(f"ld_fit_{tic}_bins.txt"),
               np.column_stack([fit["t_bins"], fit["f_bins"],
                                fit["model"](fit["t_bins"])]),
               header="phase_days flux_binned flux_model")



# --------------------------------------------------------------------------
# Folding / binning for TRICERATOPS
# --------------------------------------------------------------------------
def measure_stacked_depth(time, flux, P, T0, dur):
    """Simple stacked depth: median(OOT) - median(in-transit).

    Returns NaN when either sample is too small.  Used for the red-noise
    scramble test (fast, no shape fit).
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    ph = (time - T0 + 0.5 * P) % P - 0.5 * P
    in_tr = np.abs(ph) < 0.5 * float(dur)
    oot = np.abs(ph) > 0.75 * float(dur)
    fin = np.isfinite(flux)
    n_in = int((in_tr & fin).sum())
    n_oot = int((oot & fin).sum())
    if n_in < 5 or n_oot < 20:
        return float("nan")
    d = float(np.nanmedian(flux[oot & fin]) - np.nanmedian(flux[in_tr & fin]))
    return d if np.isfinite(d) else float("nan")


def red_noise_scramble_test(time, flux, P, T0, dur, depth_obs,
                            n_scrambles: int = 1000, seed: int = 42,
                            log=None):
    """Empirical p-value for the stacked depth under a red-noise null.

    The light curve is held fixed (preserving correlated / non-Gaussian
    noise and sector structure).  Only the trial ephemeris epoch is
    randomised: for each scramble, T0' is drawn uniformly in ``[T0, T0+P)``
    and the stacked depth is remeasured.  Under pure noise the observed
    depth is just one draw from that distribution.

    Returns
    -------
    dict with keys:
      depth_obs, n_scrambles, n_valid, n_ge, p_value, depth_null_ppm,
      depth_null_p90_ppm, depth_null_p99_ppm

    Notes
    -----
    This is a *phase-randomisation* test at fixed P and T14.  It does not
    replace TRICERATOPS (astrophysical FPs) and does not model every look-
    elsewhere effect of a full blind search.  See Ivashtenko & Zackay (2025)
    for a more complete matched-filter + permutation framework.
    """
    if n_scrambles is None or int(n_scrambles) <= 0:
        return None
    n_scrambles = int(n_scrambles)
    rng = np.random.default_rng(int(seed))

    if not (np.isfinite(depth_obs) and depth_obs > 0):
        # fall back to measuring from the data
        depth_obs = measure_stacked_depth(time, flux, P, T0, dur)
    if not (np.isfinite(depth_obs) and depth_obs > 0):
        msg = "  red-noise scramble: observed depth is non-positive; skipped"
        print(msg)
        if log is not None:
            _log(log, msg)
        return None

    null = np.full(n_scrambles, np.nan)
    for i in range(n_scrambles):
        t0s = float(T0) + float(rng.uniform(0.0, float(P)))
        null[i] = measure_stacked_depth(time, flux, P, t0s, dur)

    valid = null[np.isfinite(null)]
    n_valid = int(valid.size)
    if n_valid < 10:
        msg = f"  red-noise scramble: only {n_valid} valid trials; skipped"
        print(msg)
        if log is not None:
            _log(log, msg)
        return None

    n_ge = int(np.sum(valid >= depth_obs))
    # conservative Laplace estimator
    p_value = (1.0 + n_ge) / (1.0 + n_valid)
    out = dict(
        depth_obs=float(depth_obs),
        n_scrambles=n_scrambles,
        n_valid=n_valid,
        n_ge=n_ge,
        p_value=float(p_value),
        depth_null_ppm=float(np.median(valid) * 1e6),
        depth_null_p90_ppm=float(np.percentile(valid, 90) * 1e6),
        depth_null_p99_ppm=float(np.percentile(valid, 99) * 1e6),
    )
    lines = [
        "\n--- Red-noise scramble test (phase randomisation at fixed P) ---",
        f"  observed stacked depth = {depth_obs * 1e6:.0f} ppm",
        f"  scrambles: {n_valid}/{n_scrambles} valid",
        f"  null median / p90 / p99 = {out['depth_null_ppm']:.0f} / "
        f"{out['depth_null_p90_ppm']:.0f} / {out['depth_null_p99_ppm']:.0f} ppm",
        f"  n(null >= observed) = {n_ge}",
        f"  empirical p-value = {p_value:.4g}   "
        f"[ (1+n_ge)/(1+n_valid) ]",
    ]
    if p_value > 0.01:
        lines.append(
            "  ! p > 0.01: the stacked depth is consistent with a red-noise "
            "draw at this P; do not treat TRICERATOPS FPP as a validation.")
    else:
        lines.append(
            "  p <= 0.01: stacked depth is unlikely under pure phase-random "
            "noise at this P (does not rule out astrophysical EBs).")
    for ln in lines:
        print(ln)
        if log is not None:
            _log(log, ln)
    return out


def fold_and_bin(time, flux, flux_err, P, T0, dur, cfg: Config):
    """Phase-fold to days-from-mid-transit, clip outliers, and bin.

    Steps
    -----
    1. Select the phase window ``|φ| < 0.5 * fold_n_dur * T14``.
    2. MAD-clip: two-sided on out-of-transit points; in-transit only upward
       spikes and pathological depths (``fold_clip_sigma``).
    3. Re-normalise the clipped fold so the OOT median is exactly 1.0.
    4. Bin with the per-bin *median* (robust) or the lightkurve mean.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    flux_err = np.asarray(flux_err, float)
    ph = (time - T0 + 0.5 * P) % P - 0.5 * P
    sel = np.abs(ph) < 0.5 * cfg.fold_n_dur * dur
    if int(sel.sum()) < 20:
        raise RuntimeError("Too few points in the phase window")

    tp = ph[sel].copy()
    fp = flux[sel].copy()
    ep = flux_err[sel].copy()

    # --- transit-aware MAD clip on the fold --------------------------------
    sig = cfg.fold_clip_sigma
    if sig is not None and sig > 0:
        in_tr = np.abs(tp) < 0.75 * dur
        oot = ~in_tr & np.isfinite(fp)
        if int(oot.sum()) >= 10:
            med = float(np.nanmedian(fp[oot]))
            rms = _mad_sigma(fp[oot])
            if np.isfinite(rms) and rms > 0 and np.isfinite(med):
                ok = np.isfinite(fp)
                ok &= ~oot | ((fp > med - sig * rms) & (fp < med + sig * rms))
                # in transit: upward spikes only (plus non-physical
                # fluxes); a deep eclipse must keep its bottom
                ok &= ~in_tr | ((fp < med + sig * rms) & (fp > 0.1 * med))
                n_rej = int((~ok & np.isfinite(fp)).sum())
                if n_rej:
                    print(f"  fold clip ({sig:.1f}σ): rejected {n_rej}/{fp.size} "
                          f"points  (OOT RMS = {rms * 1e6:.0f} ppm)")
                tp, fp, ep = tp[ok], fp[ok], ep[ok]

    # --- OOT continuum → 1.0 -----------------------------------------------
    oot = (np.abs(tp) > 0.75 * dur) & np.isfinite(fp)
    if int(oot.sum()) >= 5:
        med = float(np.nanmedian(fp[oot]))
        if np.isfinite(med) and med > 0:
            fp = fp / med
            ep = ep / med

    order = np.argsort(tp)
    tp, fp, ep = tp[order], fp[order], ep[order]

    binsize = cfg.bin_minutes / (60.0 * 24.0)
    if cfg.fold_robust_bin:
        lc = _median_bin_lc(tp, fp, ep, binsize)
    else:
        lc = TessLightCurve(time=tp, flux=fp, flux_err=ep).bin(
            time_bin_size=binsize)
        keep = (np.isfinite(lc.time.value) & np.isfinite(lc.flux.value)
                & np.isfinite(lc.flux_err.value))
        lc = lc[keep]
    print(f"  folded LC: {sel.sum()} raw → {tp.size} after clip → "
          f"{len(lc)} bins of {cfg.bin_minutes:.0f} min"
          f"{' (median)' if cfg.fold_robust_bin else ''}")
    return tp, fp, ep, lc



def plot_folded_lc(tp, fp, lc, cfg: Config, tic: int, dur: float,
                   model=None, model_label="model") -> None:
    """Folded light curve with sensible axes: the y-range follows the *binned*
    data, so a few cadences with huge errors cannot flatten the transit; the
    unbinned points are drawn without error bars."""
    t_b = np.asarray(lc.time.value, float)
    f_b = np.asarray(lc.flux.value, float)
    e_b = np.asarray(lc.flux_err.value, float)
    ok = np.isfinite(f_b)
    oot = ok & (np.abs(t_b) > 0.75 * dur)
    base = np.nanmedian(f_b[oot]) if oot.sum() >= 3 else 1.0
    rms = _mad_sigma(f_b[oot]) if oot.sum() >= 5 else np.nanstd(f_b[ok])
    lo = np.nanmin(f_b[ok]) - 4 * rms
    hi = max(np.nanmax(f_b[ok]), base) + 4 * rms
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    a1.plot(tp, fp, ".", color="0.8", ms=2, zorder=0, rasterized=True,
            label="unbinned")
    a1.errorbar(t_b, f_b, yerr=e_b, fmt="o", color="k", ms=3.5, lw=0.8,
                zorder=2, label=f"binned ({cfg.bin_minutes:.0f} min)")
    tg = np.linspace(tp.min(), tp.max(), 1000)
    if model is not None:
        a1.plot(tg, model(tg), "r-", lw=1.6, zorder=3, label=model_label)
    for x in (-0.5 * dur, 0.5 * dur):
        a1.axvline(x, color="b", ls=":", lw=0.8)
    a1.axvline(0.0, color="0.5", ls="--", lw=0.6)
    a1.set_ylim(lo, hi)
    a1.set_ylabel("Normalised flux")
    n_out = int(np.sum((fp < lo) | (fp > hi)))
    a1.set_title(f"TIC {tic}   (dotted: +-T14/2 = {0.5 * dur * 24:.2f} h; "
                 f"{n_out} unbinned points outside the y-range)")
    a1.legend(loc="lower right", fontsize=8)
    res = (f_b - (model(t_b) if model is not None else base)) * 1e6
    a2.axhline(0, color="r", lw=0.8)
    a2.errorbar(t_b, res, yerr=e_b * 1e6, fmt="o", color="k", ms=3, lw=0.7)
    a2.set_xlabel("Time from mid-transit (d)")
    a2.set_ylabel("Residual (ppm)" if model is not None else "Flux - baseline (ppm)")
    fig.tight_layout()
    fig.savefig(cfg.path(f"phased_transit_{tic}.pdf"))
    plt.close(fig)


def _median_bin_lc(tp, fp, ep, binsize):
    """Phase-fold binning with per-bin median flux and MAD error.

    More robust than the arithmetic mean when residual ramps or single bad
    cadences would otherwise pull a bin far from the continuum.
    """
    tp = np.asarray(tp, float)
    fp = np.asarray(fp, float)
    ep = np.asarray(ep, float)
    if tp.size == 0:
        return TessLightCurve(time=[], flux=[], flux_err=[])

    t0 = np.floor(tp.min() / binsize) * binsize
    idx = np.floor((tp - t0) / binsize).astype(int)
    idx -= idx.min()
    nbin = int(idx.max()) + 1

    t_b = np.full(nbin, np.nan)
    f_b = np.full(nbin, np.nan)
    e_b = np.full(nbin, np.nan)
    for i in range(nbin):
        m = idx == i
        n = int(m.sum())
        if n < 1:
            continue
        vals = fp[m]
        finite = np.isfinite(vals)
        if finite.sum() < 1:
            continue
        vals = vals[finite]
        t_b[i] = np.median(tp[m][finite])
        f_b[i] = np.median(vals)
        # MAD / sqrt(n), floored by the median formal error in the bin
        mad = 1.4826 * np.median(np.abs(vals - f_b[i])) if vals.size > 1 else 0.0
        formal = np.nanmedian(ep[m][finite]) if np.any(np.isfinite(ep[m])) else 0.0
        e_b[i] = max(mad / np.sqrt(vals.size),
                     formal / np.sqrt(vals.size) if formal > 0 else 0.0,
                     1e-6)

    keep = np.isfinite(t_b) & np.isfinite(f_b) & np.isfinite(e_b)
    return TessLightCurve(time=t_b[keep], flux=f_b[keep], flux_err=e_b[keep])


# --------------------------------------------------------------------------
# Robust flux uncertainty and checked calc_probs
# --------------------------------------------------------------------------
def _mad_sigma(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def triceratops_flux_error(tp, fp, lc, dur, P, cfg: Config,
                           ref_model=None, ref_npar=4):
    """One realistic flux uncertainty for calc_probs.

    The error of a binned point from lightkurve (propagated pipeline errors,
    divided by sqrt(n)) is typically tens of ppm and ignores red noise,
    phase-curve and detrending residuals.  At high S/N this makes the
    likelihood of *every* scenario astronomically small (lnL ~ -1e3..-1e5), so
    exp(lnZ) underflows to 0 and the FPP becomes NaN (0/0).

    ref_model : optional callable(t) -> flux (e.g. the limb-darkened fit) used
        as the smooth reference for the chi2_red = 1 inflation; the trapezoid
        is used when it is None.  ref_npar = number of free parameters.

    Returns (sigma, info).
    """
    tp = np.asarray(tp, float)
    fp = np.asarray(fp, float)
    t_b = np.asarray(lc.time.value, float)
    f_b = np.asarray(lc.flux.value, float)
    binsize = cfg.bin_minutes / (60.0 * 24.0)
    n_b = np.array([np.count_nonzero(np.abs(tp - t0) <= 0.5 * binsize)
                    for t0 in t_b])
    good = (n_b >= 3) & np.isfinite(f_b)
    if good.sum() < 10:
       raise RuntimeError("Too few well-populated bins for an error estimate")
    t_b, f_b, n_b = t_b[good], f_b[good], n_b[good]

    oot_pt = np.abs(tp) > 0.75 * dur            # clear of ingress/egress
    oot_b = np.abs(t_b) > 0.75 * dur
    sig_pt = _mad_sigma(fp[oot_pt])
    sig_white = sig_pt / np.sqrt(np.median(n_b))
    sig_emp = _mad_sigma(f_b[oot_b]) if oot_b.sum() >= 5 else 0.0
    sig_emp = 0.0 if not np.isfinite(sig_emp) else sig_emp
    floor = cfg.err_floor_ppm * 1e-6
    sigma = max(sig_white, sig_emp, floor)

    info = dict(n_bins_used=int(good.sum()),
                sigma_point_ppm=sig_pt * 1e6,
                sigma_white_bin_ppm=sig_white * 1e6,
                sigma_empirical_bin_ppm=sig_emp * 1e6,
                floor_ppm=floor * 1e6,
                sigma_lightkurve_mean_ppm=float(
                    np.nanmean(lc.flux_err.value)) * 1e6,
                chi2_red_model=np.nan, inflation=1.0)

    if cfg.tri_inflate_to_model:
        try:
            if ref_model is not None:
                model_b = np.asarray(ref_model(t_b), float)
                npar, ref_name = int(ref_npar), "limb-darkened fit"
            else:
                # P only needs to exceed the window so the phase does not wrap
                P_big = 4.0 * max(float(np.abs(t_b).max()), 0.1) + 1.0
                _, _, _, model = fit_transit_shape(
                    t_b, f_b, np.full(t_b.size, sigma), P_big, 0.0,
                    dur_guess=dur)
                model_b, npar, ref_name = model(t_b), 4, "trapezoid"
            resid = f_b - model_b
            dof = max(t_b.size - npar, 1)
            chi2_red = float(np.sum((resid / sigma) ** 2) / dof)
            scale = float(np.clip(np.sqrt(chi2_red), 1.0, cfg.tri_max_inflate))
            sigma *= scale
            info.update(chi2_red_model=chi2_red, inflation=scale,
                        reference_model=ref_name)
        except Exception as exc:
            print(f"  ! model-based error inflation skipped: {exc}")

    info["sigma_used_ppm"] = sigma * 1e6
    return float(sigma), info


def _tri_version() -> str:
    try:
        return importlib_metadata.version("triceratops")
    except Exception:
        return "unknown"


def _version_tuple(v: str):
    out = []
    for part in str(v).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def environment_problems() -> list:
    """Package versions known to break triceratops / TRICERATOPS+, checked
    before the (long) run starts.  Returns a list of messages with the fix."""
    out = []
    plus = triceratops_plus_available(SimpleNamespace(
        calc_probs=getattr(tr.target, "calc_probs", None)))
    if plus and _version_tuple(np.__version__) >= (2, 4):
        out.append(
            f"NumPy {np.__version__}: TRICERATOPS+ converts one-element arrays "
            f"with float(), an error since NumPy 2.4.  Fix: "
            f"conda install -c conda-forge \"numpy<2.4\"  (or pip install "
            f"\"numpy<2.4\")")
    try:
        from pytransit import QuadraticModel as _QM
        if not hasattr(_QM, "evaluate_pv") or not hasattr(_QM, "evaluate_ps"):
            try:
                pv = importlib_metadata.version("pytransit")
            except Exception:
                pv = "?"
            out.append(
                f"pytransit {pv}: QuadraticModel has no evaluate_pv/evaluate_ps, "
                f"which triceratops calls (triceratops 1.1 requires pytransit "
                f"2.2; TRICERATOPS+ does not pin it).  Fix: "
                f"pip install \"pytransit==2.2.0\"")
    except Exception as exc:
        out.append(f"pytransit cannot be imported ({type(exc).__name__}: {exc})")
    return out


def _dilution_kwargs(func, cfg: Config) -> dict:
    """{'dilution_corrected': ...} only if requested and supported."""
    if cfg.dilution_corrected is None:
        return {}
    try:
        if "dilution_corrected" in inspect.signature(func).parameters:
            return {"dilution_corrected": bool(cfg.dilution_corrected)}
    except (TypeError, ValueError):
        pass
    if cfg.dilution_corrected:
        print("  ! installed triceratops has no 'dilution_corrected' argument; "
              "option ignored (needs triceratops >= 1.1)")
    return {}



def _probs_from_lnZ(lnZ):
    """Scenario probabilities, FPP and NFPP from log-evidences (same formulas
    as triceratops 1.1: softmax of lnZ; FPP = 1 - (TP + PTP + DTP); NFPP =
    sum over the nearby-star scenarios)."""
    lnZ = np.asarray(lnZ, float)
    if (np.any(np.isnan(lnZ)) or np.any(np.isposinf(lnZ))
            or np.all(np.isneginf(lnZ)) or lnZ.size < 10):
        return None, np.nan, np.nan
    m = np.max(lnZ[np.isfinite(lnZ)])
    p = np.exp(lnZ - m)
    p = p / p.sum()
    fpp = float(1.0 - (p[0] + p[3] + p[9]))
    nfpp = float(np.sum(p[15:])) if p.size > 15 else 0.0
    return p, fpp, nfpp


def _normalise_tri_result(target):
    """Make the result of calc_probs look like triceratops 1.1.

    TRICERATOPS+ (based on 1.0.17) has no target.lnZ attribute (the
    evidences are only in target.probs["lnZ"]) and normalises the
    probabilities with exp(lnZ)/sum(exp(lnZ)), which underflows to NaN for
    very negative lnZ.  Copy lnZ to target.lnZ and, if the probabilities are
    not finite, recompute them (and FPP, NFPP) with the stable softmax."""
    probs = getattr(target, "probs", None)
    # Copy the evidences of THIS run.  The copy must be refreshed after every
    # calc_probs call: TRICERATOPS+ never sets target.lnZ itself, so a copy
    # left from the first run would otherwise be pooled n_iter times.
    if (getattr(target, "lnZ", None) is None
            or getattr(target, "_tv_lnz_from_probs", False)) \
            and probs is not None and "lnZ" in getattr(probs, "columns", []):
        try:
            target.lnZ = np.asarray(probs["lnZ"], float)
            target._tv_lnz_from_probs = True
        except Exception:
            return
    lnZ = getattr(target, "lnZ", None)
    if lnZ is None or probs is None or "prob" not in getattr(probs, "columns", []):
        return
    pr = np.asarray(probs["prob"], float)
    fpp = float(getattr(target, "FPP", np.nan))
    if np.all(np.isfinite(pr)) and np.isfinite(fpp):
        return
    p, fpp2, nfpp2 = _probs_from_lnZ(lnZ)
    if p is None or len(p) != len(pr):
        return
    try:
        target.probs = probs.assign(prob=p)
    except Exception:
        pass
    target.FPP, target.NFPP = fpp2, nfpp2
    print("      probabilities recomputed from lnZ with a stable softmax "
          "(the package result was not finite)")


def _pool_lnZ(lnZ_runs):
    """Log of the mean of exp(lnZ) over runs, per scenario.

    Each run gives an unbiased Monte-Carlo estimate of the same evidence Z;
    their mean is the estimate of one run with n_runs x N draws.  Only runs
    made with the same flux uncertainty may be pooled.
    """
    L = np.vstack([np.asarray(x, float) for x in lnZ_runs])
    mx = np.max(L, axis=0)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        return mx + np.log(np.mean(np.exp(L - mx), axis=0))


def _pooled_summary(lnZ_runs, n_boot=300, seed=1):
    """Pooled FPP/NFPP, bootstrap scatter over runs and a dominance check.

    Returns dict(p, lnZ, FPP, NFPP, FPP_std, NFPP_std, dominance) where
    dominance is the largest share one single run has in the pooled evidence
    of the most probable scenario (-> 1/n_runs when well converged, -> 1 when
    one run decides everything, i.e. the Monte Carlo is not converged).
    """
    n = len(lnZ_runs)
    pooled = _pool_lnZ(lnZ_runs)
    p, fpp, nfpp = _probs_from_lnZ(pooled)
    out = dict(p=p, lnZ=pooled, FPP=fpp, NFPP=nfpp, FPP_std=np.nan,
               NFPP_std=np.nan, dominance=np.nan, n_runs=n)
    if p is None:
        return out
    L = np.vstack([np.asarray(x, float) for x in lnZ_runs])
    k = int(np.argmax(p))
    with np.errstate(over="ignore"):
        share = np.exp(L[:, k] - pooled[k]) / n
    out["dominance"] = float(np.nanmax(share))
    if n >= 3 and n_boot > 0:
        rng = np.random.default_rng(seed)
        f, g = [], []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            _, a, b = _probs_from_lnZ(_pool_lnZ([lnZ_runs[i] for i in idx]))
            if np.isfinite(a):
                f.append(a)
                g.append(b)
        if len(f) > 5:
            out["FPP_std"], out["NFPP_std"] = float(np.std(f)), float(np.std(g))
    return out


def _check_run(target):
    """(valid, reason) for a finished calc_probs call."""
    if getattr(target, "FPP_degenerate", False):
        return False, "FPP_degenerate (NaN/+inf or all lnZ = -inf)"
    lnZ = getattr(target, "lnZ", None)
    if lnZ is not None:
        lnZ = np.asarray(lnZ, float)
        if np.any(np.isnan(lnZ)) or np.any(np.isposinf(lnZ)):
            return False, "NaN or +inf in lnZ"
        if np.all(np.isneginf(lnZ)):
            return False, "all lnZ = -inf"
    fpp = getattr(target, "FPP", np.nan)
    nfpp = getattr(target, "NFPP", np.nan)
    if not (np.isfinite(fpp) and np.isfinite(nfpp)):
        return False, "FPP/NFPP not finite"
    probs = getattr(target, "probs", None)
    if probs is not None and "prob" in getattr(probs, "columns", []):
        p = np.asarray(probs["prob"], float)
        if np.any(~np.isfinite(p)) or not np.isclose(p.sum(), 1.0, atol=1e-6):
            return False, "scenario probabilities do not sum to 1"
    return True, "ok"


def effective_exposure_d(sectors_data, bin_minutes: float) -> float:
    """Smoothing of a folded, binned light curve [d].

    sqrt(bin^2 + <cadence^2>), the cadences weighted by the number of
    points of each sector (the same value the limb-darkened fit uses).
    """
    binsize = float(bin_minutes) / 1440.0
    cad2, npts = 0.0, 0
    for sd in sectors_data:
        tt = np.asarray(sd.time, float)
        if tt.size > 2:
            cad = float(np.nanmedian(np.diff(tt)))
            cad2 += tt.size * cad ** 2
            npts += tt.size
    cad_rms = float(np.sqrt(cad2 / npts)) if npts else 0.0
    return float(np.sqrt(binsize ** 2 + cad_rms ** 2))


def _exptime_kwargs(func, cfg: Config) -> dict:
    """{'exptime', 'nsamples'} for calc_probs when supported and wanted."""
    et = getattr(cfg, "_tri_exptime", None)
    if not et or et <= 0:
        return {}
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    out = {}
    if "exptime" in params:
        out["exptime"] = float(et)
    if "nsamples" in params:
        out["nsamples"] = int(cfg.tri_nsamples)
    return out


SCENARIO_CLASSES = (
    ("planet on the target or an unresolved companion (TP, PTP, DTP)",
     ("TP", "PTP", "DTP")),
    ("eclipsing binary on the target / bound companion "
     "(EB, EBx2P, PEB, PEBx2P, SEB, SEBx2P, STP)",
     ("EB", "EBx2P", "PEB", "PEBx2P", "SEB", "SEBx2P", "STP")),
    ("unresolved background / foreground star (DEB, DEBx2P, BTP, BEB, BEBx2P)",
     ("DEB", "DEBx2P", "BTP", "BEB", "BEBx2P")),
    ("resolved nearby star (NTP, NEB, NEBx2P)", ("NTP", "NEB", "NEBx2P")),
)


def scenario_class_probs(p, names) -> list:
    """Sum the pooled scenario probabilities per class of explanation."""
    out = []
    if p is None or names is None or len(names) != len(p):
        return out
    names = [str(n).strip() for n in names]
    for label, members in SCENARIO_CLASSES:
        out.append((label, float(sum(pi for pi, n in zip(p, names)
                                     if n in members))))
    return out


def _calc_probs_checked(target, lc, P_orb, sigma, cfg: Config,
                        max_retries=None):
    """Run calc_probs; on an invalid result retry with a larger sigma.

    Returns (valid, FPP, NFPP, sigma_used, reason).
    """
    n_retry = cfg.tri_max_retries if max_retries is None else max_retries
    s = float(sigma)
    kw = _dilution_kwargs(target.calc_probs, cfg)
    extra = {"N": int(cfg.tri_N)} if cfg.tri_N else {}
    extra.update(_exptime_kwargs(target.calc_probs, cfg))
    try:
        extra.update(_plus_prepare(target, cfg))
    except Exception as exc:
        print(f"      ! TRICERATOPS+ inputs not prepared ({exc}); "
              f"running without ground light curves")
    reason = "not run"
    for attempt in range(n_retry + 1):
        try:
            with warnings.catch_warnings():
                # the RuntimeWarnings are diagnosed explicitly by _check_run
                warnings.simplefilter("ignore", RuntimeWarning)
                target.calc_probs(time=lc.time.value,
                                  flux_0=lc.flux.value,
                                  flux_err_0=s,
                                  P_orb=P_orb,
                                  contrast_curve_file=cfg.contrast_curve_file,
                                  filt=cfg.contrast_filt,
                                  parallel=cfg.parallel,
                                  verbose=0, **extra, **kw)
        except Exception as exc:
            txt = f"{type(exc).__name__}: {exc}"
            net = any(k in txt.lower() for k in (
                "url", "ssl", "http", "connection", "timed out", "trilegal",
                "certificate", "403", "404"))
            hint = ("This looks like a failed TRILEGAL download: fill in the "
                    "form at https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6 "
                    "in a browser and re-run with --trilegal-fname FILE."
                    if net else
                    "This is an error inside the triceratops package or its "
                    "dependencies (see the traceback above); check the "
                    "package versions (GUI: Tools -> Check interpreter).")
            raise RuntimeError(f"triceratops.calc_probs failed. {hint}  "
                               f"Original error: {txt}") from exc
        _normalise_tri_result(target)
        valid, reason = _check_run(target)
        fpp = float(getattr(target, "FPP", np.nan))
        nfpp = float(getattr(target, "NFPP", np.nan))
        if valid:
            return True, fpp, nfpp, s, reason
        print(f"      invalid run at sigma = {s * 1e6:.0f} ppm ({reason})"
              + ("; retrying with larger sigma"
                 if attempt < n_retry else ""))
        s *= cfg.tri_retry_factor
    return False, fpp, nfpp, s / cfg.tri_retry_factor, reason


# --------------------------------------------------------------------------
# TRICERATOPS
# --------------------------------------------------------------------------
def _ensure_trilegal_ssl():
    """Make TRILEGAL HTTPS work despite broken CA chains / bot blocking.

    triceratops uses both *requests* (mechanicalsoup form submit) and
    *urllib* (pandas.read_csv of the result URL).  The Padova server also
    often returns HTTP 403 unless a browser-like User-Agent is sent.
    """
    try:
        import certifi
        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
        print(f"  SSL: using certifi CA bundle ({ca})")
    except Exception as exc:
        print(f"  SSL: certifi not available ({exc})")

    # Last-resort: do not verify certs at all (Padova chain is frequently
    # incomplete on conda/Windows setups).
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    _UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    if not getattr(requests.Session.request, "_trilegal_patched", False):
        _orig = requests.Session.request

        def _request(self, method, url, *args, **kwargs):
            headers = dict(kwargs.get("headers") or {})
            headers.setdefault("User-Agent", _UA)
            kwargs["headers"] = headers
            if url and "stev.oapd.inaf.it" in str(url):
                kwargs.setdefault("verify", False)
            return _orig(self, method, url, *args, **kwargs)

        _request._trilegal_patched = True
        requests.Session.request = _request

    # pandas.read_csv / urllib used by triceratops.funcs.save_trilegal
    if not getattr(_ensure_trilegal_ssl, "_urllib_patched", False):
        import urllib.request
        _opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler())
        _opener.addheaders = [("User-Agent", _UA)]
        urllib.request.install_opener(_opener)
        _ensure_trilegal_ssl._urllib_patched = True


def _gaia_field_star_table(ra, dec, tic, cfg: Config, field_deg2=0.1, mag_limit=21.0):
    """Build a TRILEGAL-compatible CSV from Gaia DR3 (fallback when Padova is down).

    Columns expected by triceratops.trilegal_results: Mact, logg, logTe, [M/H],
    TESS (or G as proxy), J, H, Ks.
    """
    try:
        from astroquery.gaia import Gaia
    except Exception as exc:
        print(f"  ! astroquery.gaia not available ({exc})")
        return None

    radius = float(np.sqrt(field_deg2 / np.pi))
    adql = (
        "SELECT phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, "
        "parallax, parallax_over_error, teff_gspphot, logg_gspphot, mh_gspphot "
        "FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius})) "
        f"AND phot_g_mean_mag < {mag_limit}"
    )
    try:
        Gaia.ROW_LIMIT = -1
        tbl = Gaia.launch_job(adql, verbose=False).get_results()
    except Exception as exc:
        print(f"  ! Gaia DR3 query failed ({exc})")
        return None
    if len(tbl) == 0:
        print("  ! Gaia DR3 returned no field stars")
        return None

    G = np.asarray(tbl["phot_g_mean_mag"], dtype=float)
    BP = np.asarray(tbl["phot_bp_mean_mag"], dtype=float)
    RP = np.asarray(tbl["phot_rp_mean_mag"], dtype=float)
    teff = np.asarray(tbl["teff_gspphot"], dtype=float)
    logg = np.asarray(tbl["logg_gspphot"], dtype=float)
    mh = np.asarray(tbl["mh_gspphot"], dtype=float)
    plx = np.asarray(tbl["parallax"], dtype=float)

    # rough mass from Teff (main-sequence-ish); fallback 0.8 Msun
    mass = np.full(G.size, 0.8)
    ok_t = np.isfinite(teff) & (teff > 3000) & (teff < 10000)
    # M ~ (Teff/5772)^1.5 crude
    mass[ok_t] = np.clip((teff[ok_t] / 5772.0) ** 1.5, 0.1, 3.0)

    # TESS mag approx from Gaia G and BP-RP (Stassun-like rough)
    bprp = BP - RP
    tess = G - 0.00522555 * bprp ** 3 - 0.0894 * bprp ** 2 + 0.0411 * bprp
    tess = np.where(np.isfinite(tess), tess, G)

    # placeholder NIR mags (not critical for TESS scenarios)
    J = tess + 0.5
    H = tess + 0.7
    Ks = tess + 0.8

    logTe = np.where(np.isfinite(teff) & (teff > 0), np.log10(teff), 3.7)
    logg = np.where(np.isfinite(logg), logg, 4.5)
    mh = np.where(np.isfinite(mh), mh, 0.0)

    # Sloan-like g r i z (TRICERATOPS+ reads them for every field star):
    # Riello et al. (2021) relations; z from i and r - i (rough)
    def _band(b):
        return np.array([_gaia_to_band(g_, x_, b) if np.isfinite(x_) else g_
                         for g_, x_ in zip(G, bprp)])
    gmag, rmag, imag = _band("g"), _band("r"), _band("i")
    zmag = imag - 0.5 * (rmag - imag)

    df = pd.DataFrame({
        "Mact": mass,
        "logg": logg,
        "logTe": logTe,
        "[M/H]": mh,
        "TESS": tess,
        "J": J,
        "H": H,
        "Ks": Ks,
        "g": gmag,
        "r": rmag,
        "i": imag,
        "z": zmag,
    })
    out = cfg.path(f"trilegal_TIC{tic}_gaia.csv")
    df.to_csv(out, index=False)
    print(f"  Gaia DR3 field-star table ({len(df)} stars) -> {out}")
    return out


def _ensure_griz_columns(path, log=None):
    """TRICERATOPS+ reads g, r, i, z of every field star from the
    TRILEGAL table; a table made for triceratops 1.1 (TESS + 2MASS only)
    gets approximate columns from TESS - J (a copy *_griz.csv is written)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return path
    if all(c in df.columns for c in ("g", "r", "i", "z")):
        return path
    if not {"TESS", "J"} <= set(df.columns):
        return path
    T = pd.to_numeric(df["TESS"], errors="coerce")
    c = (T - pd.to_numeric(df["J"], errors="coerce")).clip(0.0, 3.0)
    df["g"] = T + 1.2 * c
    df["r"] = T + 0.6 * c
    df["i"] = T + 0.2 * c
    df["z"] = T - 0.1 * c
    out = os.path.splitext(path)[0] + "_griz.csv"
    df.to_csv(out, index=False)
    _log(log, f"  field-star table has no g r i z (needed by TRICERATOPS+): "
              f"approximate columns from TESS - J -> {out}")
    return out


@contextmanager
def _nullcontext():
    yield


def _drop_flagged_tic_rows(target, tic) -> None:
    """Remove TIC entries flagged ARTIFACT / DUPLICATE / SPLIT from the
    TRICERATOPS star table (and the matching pix_coords rows).

    A SPLIT entry is an old blended 2MASS source whose light is also in the
    Gaia-based stars that replace it; keeping it counts that light twice
    (e.g. the close pair KIC 3755522 A + B).  The target row is never
    removed, but it must not be a flagged entry itself."""
    st = getattr(target, "stars", None)
    if st is None or "disposition" not in st.columns or len(st) < 2:
        return
    disp = st["disposition"].astype(str).str.strip().str.upper()
    bad = disp.isin(["ARTIFACT", "DUPLICATE", "SPLIT"]).values
    if bad[0]:
        if int(st["ID"].iloc[0]) != int(tic):
            raise RuntimeError(
                f"TRICERATOPS chose TIC {int(st['ID'].iloc[0])} as the target, "
                f"a TIC entry flagged {disp.iloc[0]}; expected TIC {tic}")
        print(f"  ! the target TIC {tic} itself is flagged {disp.iloc[0]}; "
              f"kept as the target")
        bad = bad.copy()
        bad[0] = False
    if int(st["ID"].iloc[0]) != int(tic):
        print(f"  ! TRICERATOPS target is TIC {int(st['ID'].iloc[0])}, not TIC {tic}")
    if not bad.any():
        return
    keep = ~bad
    print(f"  removed {int(bad.sum())} flagged TIC entr(y/ies) from the "
          f"TRICERATOPS star table: "
          + ", ".join(f"TIC {int(i)} ({d})" for i, d in
                      zip(st["ID"].values[bad], disp.values[bad])))
    target.stars = st[keep].reset_index(drop=True)
    target.pix_coords = [np.asarray(pc)[keep] for pc in target.pix_coords]


def _set_host_params(target, cfg) -> None:
    """Write non-TIC host parameters (--star-source kepler / --stellar) into
    row 0 of the TRICERATOPS star table (its priors use M, R, Teff)."""
    st = getattr(cfg, "_star", None) or {}
    srcs = str(st.get("source", "TIC"))
    if srcs.startswith("TIC") and "estimated" not in srcs:
        return
    tab = target.stars
    for col, key in (("mass", "mass"), ("rad", "rad"), ("Teff", "Teff"),
                     ("logg", "logg")):
        v = st.get(key, np.nan)
        if col in tab.columns and v is not None and np.isfinite(v):
            tab.loc[tab.index[0], col] = float(v)
    print(f"  TRICERATOPS host parameters from {st['source']}: "
          f"R = {st.get('rad')}, M = {st.get('mass')}, Teff = {st.get('Teff')}")


def make_target(tic, sectors, apertures, depth, cfg: Config):
    """Build the TRICERATOPS target and compute the required depth per star.

    Field-star population sources, in order:
      1. --trilegal-fname / cached local file
      2. TRILEGAL web form (often blocked: SSL / 403)
      3. Gaia DR3 cone search written in TRILEGAL-compatible CSV form
    """
    _ensure_trilegal_ssl()

    host = int(getattr(cfg, "host_tic", None) or tic)
    kwargs = dict(ID=host, sectors=np.asarray(sectors),
                  search_radius=cfg.search_radius)
    if is_kepler():
        if "mission" not in inspect.signature(tr.target).parameters:
            raise RuntimeError("the installed triceratops has no 'mission' "
                               "argument: Kepler mode needs triceratops >= 1.0")
        # TRICERATOPS wants the KIC number and quarters; the position is
        # passed so it does not need the Vizier KIC query
        kwargs.update(ID=int(cfg.kic), mission="Kepler")
        sp = inspect.signature(tr.target).parameters
        st = getattr(cfg, "_star", None) or {}
        if "ra" in sp and np.isfinite(st.get("ra", np.nan)):
            kwargs.update(ra=float(st["ra"]), dec=float(st["dec"]))
    if "lightkurve_cache_dir" in inspect.signature(tr.target).parameters:
        kwargs["lightkurve_cache_dir"] = cfg.cache_dir()
    trilegal = cfg.trilegal_fname
    if trilegal is None:
        for cand in (cfg.path(f"trilegal_TIC{tic}.dat"),
                     cfg.path(f"trilegal_TIC{tic}.csv"),
                     cfg.path(f"trilegal_TIC{tic}_gaia.csv")):
            if os.path.isfile(cand):
                trilegal = cand
                print(f"  reusing cached field-star table: {trilegal}")
                break
    if trilegal is not None:
        if triceratops_plus_available(SimpleNamespace(
                calc_probs=getattr(tr.target, "calc_probs", None))):
            trilegal = _ensure_griz_columns(trilegal)
        kwargs["trilegal_fname"] = trilegal

    st0 = getattr(cfg, "_target_star", None) or getattr(cfg, "_star", None) or {}
    kic_pos = (_kic_position_offline(cfg.kic, st0.get("ra"), st0.get("dec"))
               if is_kepler() and np.isfinite(st0.get("ra", np.nan))
               and not (getattr(cfg, "host_tic", None)
                        and int(cfg.host_tic) != int(tic))
               else _nullcontext())
    if is_kepler() and getattr(cfg, "host_tic", None) and int(cfg.host_tic) != int(tic):
        hs = getattr(cfg, "_star", {}) or {}
        kic_pos = _kic_position_offline(cfg.kic, hs.get("ra"), hs.get("dec"))
    try:
        with _triceratops_tesscut_cache(tic, cfg), kic_pos:
            target = tr.target(**kwargs)
    except Exception as exc1:
        print(f"  TRICERATOPS target initialisation failed ({exc1}); this is "
              f"usually the online TRILEGAL or catalogue query")
        print("  falling back to Gaia DR3 field-star population ...")
        star = stellar_params(tic)
        gaia_file = _gaia_field_star_table(
            star["ra"], star["dec"], tic, cfg)
        if gaia_file is None:
            msg = (
                "Could not obtain a field-star population. "
                "TRILEGAL (https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6) is an "
                "online form that often returns SSL/403 for scripts. "
                "Fill the form in a browser (RA/Dec equatorial, area 0.1 deg^2, "
                "maglim 21, TESS filters), download the result, and re-run with "
                "--trilegal-fname /path/to/file. "
                "Or install astroquery for the Gaia DR3 fallback. "
                f"Original error: {exc1}"
            )
            raise RuntimeError(msg) from exc1
        kwargs["trilegal_fname"] = gaia_file
        with _triceratops_tesscut_cache(tic, cfg), kic_pos:
            target = tr.target(**kwargs)

    # Cache whatever table was used
    cache_path = cfg.path(f"trilegal_TIC{tic}.csv")
    for attr in ("trilegal_fname", "trilegal", "trilegal_file"):
        src = getattr(target, attr, None)
        if isinstance(src, str) and os.path.isfile(src):
            if os.path.abspath(src) != os.path.abspath(cache_path):
                try:
                    import shutil
                    shutil.copy2(src, cache_path)
                    print(f"  cached field-star table -> {cache_path}")
                except Exception:
                    pass
            break

    _drop_flagged_tic_rows(target, host)
    _set_host_params(target, cfg)

    # TRICERATOPS skips a sector whose cutout fails ("Ignoring for
    # validation") but keeps it in target.sectors, so pix_coords and
    # all_ap_pixels would be misaligned (IndexError in calc_depths).
    n_pc = len(getattr(target, "pix_coords", []))
    if n_pc != len(target.sectors):
        loaded = list(getattr(cfg, "_tri_loaded_sectors", []))
        # TRICERATOPS may also fail *after* our adapter returned the cutout
        if len(loaded) == n_pc:
            dropped = [int(s) for s in target.sectors if int(s) not in loaded]
            print(f"  ! TRICERATOPS could not use sector(s) {dropped}; "
                  f"continuing with {loaded}")
            target.sectors = np.asarray(loaded)
        else:
            raise RuntimeError(
                f"TRICERATOPS kept {n_pc} of {len(target.sectors)} sectors and "
                f"the failed ones cannot be identified; see the tracebacks "
                f"above (re-run, or use --exclude-sectors)")
        if n_pc == 0:
            raise RuntimeError("TRICERATOPS could not load any TESS cutout")
    ap_by_sector = {int(s): a for s, a in zip(sectors, apertures)}
    all_ap = [ap_by_sector[int(s)] for s in target.sectors]
    target.calc_depths(tdepth=depth, all_ap_pixels=all_ap,
                       **_dilution_kwargs(target.calc_depths, cfg))
    return target, all_ap


def nfpp_input_summary(target, depth, cfg: Config, log=None, n_removed=0):
    """Explain which resolved stars can enter the NTP/NEB/NEBx2P scenarios.

    calc_depths sets tdepth_i = (depth seen in the aperture) / fluxratio_i and
    replaces it by 0 when that exceeds 1.  Stars with tdepth = 0 cannot produce
    the observed dip even with a 100 % eclipse and are skipped by calc_probs,
    so they add nothing to NFPP.  This is a flux argument, not a statement
    about whether the star is an eclipsing binary.
    """
    st = target.stars
    if "tdepth" not in st.columns or "fluxratio" not in st.columns:
        _log(log, "\n(NFPP summary skipped: no fluxratio/tdepth columns)")
        return
    fr0 = float(st["fluxratio"].iloc[0])
    ap_depth = float(depth) if cfg.dilution_corrected is False \
        else float(depth) * fr0
    nb = st.iloc[1:]
    able = nb[nb["tdepth"] > 0]
    _log(log, "\n--- NFPP input: which neighbours can produce the signal ---")
    _log(log, f"  depth adopted for calc_depths      : {depth * 1e6:.0f} ppm"
              f"  (target flux ratio {fr0:.4f})")
    _log(log, f"  dip in the aperture light curve    : {ap_depth * 1e6:.0f} ppm")
    _log(log, f"  min. neighbour flux ratio needed   : {ap_depth:.2e}  "
              f"(a 100 % eclipse of a star with less flux cannot do it)")
    _log(log, f"  neighbours in table / still used   : {len(nb)} / {len(able)}"
              + (f"   ({n_removed} removed by vetting)" if n_removed else ""))
    if len(nb):
        j = nb["fluxratio"].astype(float).idxmax()
        b = nb.loc[j]
        fr = float(b["fluxratio"])
        need = ap_depth / fr if fr > 0 else np.inf
        _log(log, f"  brightest neighbour                : TIC {int(b['ID'])}, "
                  f"Tmag {float(b['Tmag']):.2f}, sep {float(b['sep (arcsec)']):.1f} arcsec, "
                  f"flux ratio {fr:.2e}")
        _log(log, f"    -> would need an eclipse of {need * 100:.0f} % of its "
                  f"own flux ({'possible' if need <= 1 else 'impossible'})")
    if len(able):
        cols = [c for c in ("ID", "Tmag", "sep (arcsec)", "fluxratio",
                            "tdepth") if c in able.columns]
        _log(log, "  neighbours that can host the signal:")
        _log(log, able[cols].to_string(index=False))
    else:
        _log(log, "  => no neighbour contributes enough flux: NFPP = 0 follows "
                  "from the flux budget (not from missing follow-up).")


def run_triceratops(target, all_ap, tic, depth, P_orb, lc, cfg: Config,
                    drop_stars=None, log=None, sigma=None):
    """Returns (target, FPPs, NFPPs, sigma_used).

    FPPs / NFPPs hold NaN for invalid iterations (never averaged in).
    """
    ver = _tri_version()
    _log(log, f"triceratops version: {ver}")
    if ver != "unknown" and _version_tuple(ver) < (1, 1, 0):
        _log(log, "  ! triceratops < 1.1.0 normalises the scenario evidences "
                  "with exp(lnZ)/sum(exp(lnZ)) and underflows to NaN for very "
                  "poor fits; consider upgrading.  Invalid runs are detected "
                  "and retried below.")
    if sigma is None:
        sigma = float(np.nanmean(lc.flux_err.value))
        _log(log, "  ! no sigma supplied; using the mean lightkurve bin error")
    # Field plots *before* remove_star: triceratops.plot_field breaks after
    # stars are dropped (matplotlib scatter 'c' length no longer matches).
    for s, ap in zip(target.sectors, all_ap):
        try:
            target.plot_field(sector=int(s), ap_pixels=ap, save=True,
                              fname=cfg.path(f"field_{tic}_s{int(s):02d}"))
        except Exception as exc:
            print(f"  ! plot_field sector {s} failed: {exc}")
        finally:
            plt.close("all")

    n_removed = 0
    if drop_stars is not None and len(drop_stars) > 0:
        requested = {str(int(x)) for x in drop_stars}
        print(f"  removing {len(drop_stars)} vetted nearby stars: {drop_stars}")
        keep_before = np.arange(len(target.stars))
        keep_after = np.where(~target.stars["ID"].astype(str).isin(requested)
                              .values)[0]
        try:
            target.remove_star(np.asarray(drop_stars, dtype=int))
        except Exception as exc:
            _log(log, f"  ! remove_star failed ({exc})")
        # verify: triceratops matches the IDs as strings; make sure that
        # every requested star is really gone, and drop it by hand if not
        ids = target.stars["ID"].astype(str)
        left = sorted(requested & set(ids))
        if left:
            _log(log, f"  ! remove_star left {len(left)} requested star(s) in "
                      f"the table ({left[:8]}{'...' if len(left) > 8 else ''}); "
                      f"dropping them directly")
            target.stars = target.stars[~ids.isin(requested)]
        n_removed = len(requested - set(target.stars["ID"].astype(str)))
        # remove_star leaves target.pix_coords untouched: keep it aligned
        # with the stars table (row i of pix_coords <-> row i of stars)
        try:
            if len(target.pix_coords[0]) == len(keep_before):
                target.pix_coords = [pc[keep_after] for pc in target.pix_coords]
            else:
                _log(log, "  ! pix_coords length does not match the star "
                          "table; left unchanged")
        except Exception as exc:
            _log(log, f"  ! could not realign pix_coords ({exc})")
        if cfg.recalc_depths_after_removal:
            try:
                target.calc_depths(tdepth=depth, all_ap_pixels=all_ap,
                                   **_dilution_kwargs(target.calc_depths, cfg))
            except Exception as exc:
                _log(log, f"  ! calc_depths after star removal failed ({exc})")
        else:
            _log(log, "  flux ratios and tdepth keep the values computed with "
                      "the full field (calc_depths -> remove_star -> "
                      "calc_probs); removed stars still dilute the aperture")
        _log(log, f"  removed {n_removed} of {len(requested)} requested "
                  f"star(s) from the TRICERATOPS field")

    _log(log, "\nStars in the field used for the calculation:")
    _log(log, target.stars.to_string())
    nfpp_input_summary(target, depth, cfg, log=log, n_removed=n_removed)

    # triceratops.marginal_likelihoods checks `type(P_orb) not in [float, int]`
    # to decide whether to treat P_orb as a fixed period or as [Pmin, Pmax].
    # A numpy scalar (np.float64 from TLS / refine_ephemeris) is neither float
    # nor int, so it takes the range branch and then fails on P_orb[0].
    # Always pass a plain Python float (or a length-2 sequence for a range).
    if np.ndim(P_orb) == 0:
        P_orb = float(P_orb)
    else:
        P_orb = np.asarray(P_orb, dtype=float)

    import time as _time
    max_iter = int(cfg.tri_max_iter or 3 * cfg.n_iter)
    t_start = _time.time()
    FPPs, NFPPs, valid_l, run_sigma_l, lnZ_runs, reasons = [], [], [], [], [], []
    sigma_cur = float(sigma)
    _log(log, f"\nflux uncertainty passed to calc_probs: "
              f"{sigma_cur * 1e6:.0f} ppm"
              + (f";  N = {cfg.tri_N:,} draws per scenario" if cfg.tri_N
                 else ";  N = triceratops default")
              + (f";  model exposure {cfg._tri_exptime * 1440:.1f} min "
                 f"({cfg.tri_nsamples} sub-samples)"
                 if getattr(cfg, "_tri_exptime", None) else ""))

    def _dominance_now():
        runs = [lnZ_runs[k] for k in range(len(lnZ_runs))
                if valid_l[k] and lnZ_runs[k] is not None
                and np.isclose(run_sigma_l[k], sigma_cur, rtol=1e-9)]
        if len(runs) < 2:
            return np.nan
        return _pooled_summary(runs, n_boot=0)["dominance"]

    i = 0
    while True:
        ok, fpp, nfpp, sigma_cur, why = _calc_probs_checked(
            target, lc, P_orb, sigma_cur, cfg)
        valid_l.append(ok)
        reasons.append(why)
        run_sigma_l.append(sigma_cur)
        FPPs.append(fpp if ok else np.nan)
        NFPPs.append(nfpp if ok else np.nan)
        lz = getattr(target, "lnZ", None) if ok else None
        lnZ_runs.append(np.array(lz, float) if lz is not None else None)
        i += 1
        tag = "" if ok else f"   INVALID ({why})"
        print(f"    iteration {i}/{cfg.n_iter if i <= cfg.n_iter else max_iter}: "
              f"sigma = {sigma_cur * 1e6:.0f} ppm, "
              f"FPP = {min(max(fpp, 0.0), 1.0) if np.isfinite(fpp) else fpp:.2e}, "
              f"NFPP = {min(max(nfpp, 0.0), 1.0) if np.isfinite(nfpp) else nfpp:.2e}{tag}")
        if i < cfg.n_iter or not cfg.pool_evidences:
            if i >= cfg.n_iter:
                break
            continue
        if (i - cfg.n_iter) % 5 != 0:
            continue
        dom = _dominance_now()
        if not np.isfinite(dom) or dom <= cfg.tri_conv_max_share:
            if i > cfg.n_iter:
                _log(log, f"  converged after {i} runs (largest single-run "
                          f"share {dom:.2f})")
            break
        if i >= max_iter or _time.time() - t_start > cfg.tri_time_budget:
            _log(log, f"  ! not converged after {i} runs / "
                      f"{_time.time() - t_start:.0f} s (largest single-run "
                      f"share {dom:.2f}); stopping (tri_max_iter / "
                      f"tri_time_budget)")
            break
        if i == cfg.n_iter:
            _log(log, f"  one run holds {dom:.2f} of the leading scenario's "
                      f"evidence: adding runs (max {max_iter}, "
                      f"{cfg.tri_time_budget:.0f} s)")
    n_total = i
    FPPs = np.asarray(FPPs, float)
    NFPPs = np.asarray(NFPPs, float)
    valid = np.asarray(valid_l, bool)
    run_sigma = np.asarray(run_sigma_l, float)

    n_valid = int(valid.sum())
    _log(log, "\nScenario probabilities (last iteration):")
    if getattr(target, "probs", None) is not None:
        _log(log, target.probs.to_string())
    lnZ = getattr(target, "lnZ", None)
    if lnZ is not None:
        _log(log, "\nlnZ per scenario (last iteration): "
                  + np.array2string(np.asarray(lnZ, float), precision=1,
                                    max_line_width=100))
    _log(log, f"\nvalid runs: {n_valid}/{n_total}; final sigma = "
              f"{sigma_cur * 1e6:.0f} ppm")

    mean_fpp = mean_nfpp = np.nan
    if n_valid:
        sd = lambda a: float(np.std(a[valid]))
        mean_fpp, mean_nfpp = np.mean(FPPs[valid]), np.mean(NFPPs[valid])
        _log(log, f"FPP  (mean of per-run values) = {mean_fpp:.5f} "
                  f"+/- {sd(FPPs):.5f}   [run-to-run scatter]")
        _log(log, f"NFPP (mean of per-run values) = {mean_nfpp:.5f} "
                  f"+/- {sd(NFPPs):.5f}")
    else:
        _log(log, "FPP  = n/a (no valid run)")
        _log(log, "NFPP = n/a (no valid run)")

    # --- pooled evidences ---------------------------------------------------
    fpp_use, nfpp_use, basis = mean_fpp, mean_nfpp, "mean of per-run values"
    pooled = None
    if cfg.pool_evidences and n_valid >= 2:
        same = valid & np.isclose(run_sigma, sigma_cur, rtol=1e-9)
        runs = [lnZ_runs[i] for i in np.where(same)[0]
                if lnZ_runs[i] is not None]
        if len(runs) >= 2:
            pooled = _pooled_summary(runs, n_boot=cfg.tri_boot)
        else:
            _log(log, "  (evidence pooling skipped: fewer than 2 valid runs "
                      "with the final sigma, or no target.lnZ)")
    if pooled is not None and pooled["p"] is not None:
        _log(log, f"\nPooled evidences of {pooled['n_runs']} runs at "
                  f"sigma = {sigma_cur * 1e6:.0f} ppm "
                  f"(= one run with {pooled['n_runs']} x N draws):")
        _log(log, f"FPP  = {pooled['FPP']:.5f} +/- {pooled['FPP_std']:.5f}"
                  f"   [bootstrap over runs]")
        _log(log, f"NFPP = {pooled['NFPP']:.5f} +/- {pooled['NFPP_std']:.5f}")
        top = np.argsort(pooled["p"])[::-1][:4]
        _log(log, "  most probable scenarios (index: prob, lnZ): "
                  + ", ".join(f"{int(k)}: {pooled['p'][k]:.3f}, "
                              f"{pooled['lnZ'][k]:.1f}" for k in top))
        dom = pooled["dominance"]
        _log(log, f"  largest share of a single run in the leading "
                  f"scenario's evidence: {dom:.2f} "
                  f"(1/n_runs = {1.0 / pooled['n_runs']:.2f} when converged)")
        if np.isfinite(dom) and dom > 0.5:
            _log(log, "  ! the evidence is dominated by a single run: the "
                      "Monte Carlo has not converged.  Increase --n-samples "
                      "(and/or --n-iter) before interpreting FPP/NFPP.")
        fpp_use, nfpp_use, basis = (pooled["FPP"], pooled["NFPP"],
                                    "pooled evidences")
        names = None
        if getattr(target, "probs", None) is not None and "scenario" in target.probs:
            names = list(target.probs["scenario"].values)
        classes = scenario_class_probs(pooled["p"], names)
        if classes:
            _log(log, "  probability per class of explanation (pooled):")
            for label, pc in classes:
                _log(log, f"    {pc:8.4f}  {label}")
            _summary(cfg)["scenario_classes"] = {lab: pc for lab, pc in classes}

    if cfg.tri_sensitivity and n_valid:
        _log(log, f"\nSensitivity of FPP/NFPP to the flux uncertainty "
                  f"({cfg.tri_sens_runs} run(s) per scale, pooled; no retries):")
        scales = [x for x in cfg.tri_sens_scales if x != 1.0] + [1.0]
        for sc in scales:       # 1.0 last: target keeps the adopted solution
            lz_runs, ok_all, s_used = [], True, sigma_cur * sc
            fpp_l, nfpp_l, why_l = [], [], ""
            for _ in range(max(int(cfg.tri_sens_runs), 1)):
                ok, fpp, nfpp, s_used, why = _calc_probs_checked(
                    target, lc, P_orb, sigma_cur * sc, cfg, max_retries=0)
                ok_all &= ok
                why_l = why
                fpp_l.append(fpp)
                nfpp_l.append(nfpp)
                if ok and getattr(target, "lnZ", None) is not None:
                    lz_runs.append(np.array(target.lnZ, float))
            if len(lz_runs) >= 2:
                ps = _pooled_summary(lz_runs, n_boot=0)
                f_, n_ = ps["FPP"], ps["NFPP"]
                tag_ = f"pooled of {len(lz_runs)}"
            else:
                f_, n_ = float(np.nanmean(fpp_l)), float(np.nanmean(nfpp_l))
                tag_ = f"{len(fpp_l)} run"
            _log(log, f"  scale {sc:4.1f}  sigma = {s_used * 1e6:8.0f} ppm  "
                      f"FPP = {f_:.3e}  NFPP = {n_:.3e}  ({tag_}) "
                      f"{'ok' if ok_all else 'INVALID: ' + why_l}")

    if n_valid / max(n_total, 1) < cfg.tri_min_valid_frac:
        verdict = (f"INDETERMINATE (only {n_valid}/{n_total} valid runs)")
    elif not (np.isfinite(fpp_use) and np.isfinite(nfpp_use)):
        verdict = "INDETERMINATE (FPP/NFPP not finite)"
    else:
        passed = fpp_use < 0.015 and nfpp_use < 0.001
        verdict = "PASSED" if passed else "NOT passed"
        if (pooled is not None and np.isfinite(pooled["dominance"])
                and pooled["dominance"] > 0.5):
            verdict += " (Monte Carlo not converged: treat as provisional)"
    _log(log, f"Giacalone+2021 validation criteria "
              f"(FPP < 0.015 and NFPP < 0.001; based on {basis}): {verdict}")
    _summary(cfg).update(
        FPP=float(fpp_use) if np.isfinite(fpp_use) else None,
        NFPP=float(nfpp_use) if np.isfinite(nfpp_use) else None,
        verdict=verdict, n_runs=int(n_total), n_valid=int(n_valid),
        dominance=(float(pooled["dominance"]) if pooled is not None
                   and np.isfinite(pooled["dominance"]) else None),
        sigma_ppm=float(sigma_cur * 1e6))
    return target, FPPs, NFPPs, sigma_cur


# --------------------------------------------------------------------------
# Ground-based photometry (v4.1)
# --------------------------------------------------------------------------
# Follow-up light curves from the ground (AAVSO campaigns, SG1, amateurs)
# resolve stars that are blended in the 21" TESS pixels.  For every light
# curve the pipeline
#   1. reads the file (AstroImageJ table, AAVSO/EXOTIC exoplanet format or a
#      plain time/flux/err table; fluxes or magnitudes; BJD_TDB, JD_UTC, ...),
#   2. fits the TESS transit shape with a free depth and mid-time plus a
#      baseline (polynomial in time and optional covariates such as airmass),
#   3. computes, for every star near the target, the fraction of the ground
#      aperture flux it contributes (Gaussian seeing PSF, circular aperture,
#      band magnitudes) and so the depth the ground aperture must show if that
#      star hosts the TESS signal,
#   4. clears (excludes) stars whose expected depth is inconsistent with the
#      measured one, before TRICERATOPS is run (target.remove_star),
#   5. tests whether the intrinsic depth depends on the band (chromaticity),
#   6. adds the ground mid-times to the timing (ETV) analysis, and
#   7. if the installed triceratops is TRICERATOPS+ (calc_probs accepts
#      external_lc_files), passes up to seven light curves to calc_probs.

GROUND_BAND_ALIASES = {
    "b": "B", "johnsonb": "B", "v": "V", "johnsonv": "V",
    "r": "R", "rc": "R", "cousinsr": "R", "johnsonr": "R",
    "i": "I", "ic": "I", "cousinsi": "I",
    "g'": "g", "gp": "g", "sg": "g", "sdssg": "g", "sloang": "g", "g": "g",
    "r'": "r", "rp": "r", "sr": "r", "sdssr": "r", "sloanr": "r",
    "i'": "i", "ip": "i", "si": "i", "sdssi": "i", "sloani": "i",
    "z'": "z", "zp": "z", "sz": "z", "sdssz": "z", "sloanz": "z", "zs": "z",
    "j": "J", "h": "H", "k": "K", "ks": "K",
    "tess": "TESS", "cbb": "CBB", "clear": "CV", "cv": "CV", "c": "CV",
    "lum": "CV", "l": "CV",
}
# The single-letter lower-case names g, r, i, z are Sloan; upper-case
# B, V, R, I are Johnson-Cousins.  'r' given by the user therefore means r'.
GROUND_LAMBDA_UM = {"B": 0.44, "V": 0.55, "g": 0.48, "R": 0.64, "r": 0.62,
                    "I": 0.79, "i": 0.76, "z": 0.91, "J": 1.25, "H": 1.65,
                    "K": 2.2, "TESS": 0.80, "CBB": 0.70, "CV": 0.60}
# TRICERATOPS+ accepts g r i z J H K; Rc and Ic are close enough to r' and i'
GROUND_PLUS_BAND = {"g": "g", "r": "r", "i": "i", "z": "z", "J": "J",
                    "H": "H", "K": "K", "R": "r", "I": "i"}


def canonical_band(name) -> str:
    s = str(name or "").strip()
    if s in ("g", "r", "i", "z", "B", "V", "R", "I", "J", "H", "K"):
        return s
    s = s.replace("\u2019", "'").replace("`", "'")
    key = s.lower().replace(" ", "").replace("_", "").replace("-", "")
    if key.endswith("'") and key[-2:-1] in ("g", "r", "i", "z"):
        return key[-2]
    return GROUND_BAND_ALIASES.get(key, s or "R")


@dataclass
class GroundSpec:
    """One ground-based light curve and how it was measured."""
    path: str
    band: Optional[str] = None        # None: FILTER from the file header, else R
    star: Optional[int] = None        # TIC at the aperture centre (None: target)
    ra: float = float("nan")          # or an explicit aperture centre [deg]
    dec: float = float("nan")
    aperture_arcsec: float = float("nan")   # aperture RADIUS
    fwhm_arcsec: float = float("nan")       # seeing FWHM
    pixscale: float = float("nan")          # arcsec/px (AIJ Source_Radius)
    observer: str = ""
    time_col: Optional[str] = None
    flux_col: Optional[str] = None
    err_col: Optional[str] = None
    time_system: Optional[str] = None       # BJD_TDB (default), JD_UTC, HJD_UTC
    mag: Optional[bool] = None
    detrend: str = "auto"                   # poly0/1/2[,col,col...] or auto
    plus_band: Optional[str] = None         # override the TRICERATOPS+ band
    label: str = ""


_GROUND_KEYS = {
    "band": "band", "filter": "band", "filt": "band",
    "star": "star", "tic": "star", "center": "star",
    "ra": "ra", "dec": "dec",
    "ap": "aperture_arcsec", "aperture": "aperture_arcsec",
    "r_ap": "aperture_arcsec", "radius": "aperture_arcsec",
    "fwhm": "fwhm_arcsec", "seeing": "fwhm_arcsec",
    "pixscale": "pixscale", "scale": "pixscale",
    "obs": "observer", "observer": "observer", "obscode": "observer",
    "tcol": "time_col", "time_col": "time_col",
    "fcol": "flux_col", "flux_col": "flux_col", "column": "flux_col",
    "ecol": "err_col", "err_col": "err_col",
    "time": "time_system", "timesys": "time_system",
    "time_system": "time_system",
    "mag": "mag", "detrend": "detrend", "plus": "plus_band",
    "plus_band": "plus_band", "label": "label",
}


def parse_ground_spec(text: str, base_dir: Optional[str] = None) -> GroundSpec:
    """'path,band=r,star=TIC,ap=4,fwhm=2.5,obs=ABC,detrend=poly1,AIRMASS'.

    Separators are commas and/or blanks; quote a path containing blanks.
    A bare word after detrend= (e.g. AIRMASS) is added to the detrend list.
    """
    import re
    text = re.sub(r"(?i)\b(star|tic|center)\s*=\s*TIC\s*(\d+)", r"\1=\2", text)
    toks = []
    # quoted strings (paths with blanks) or runs of non-blanks; an
    # apostrophe as in band=r' is an ordinary character
    for t in [m.group(1) if m.group(1) is not None else m.group(2)
              for m in re.finditer(r'"([^"]*)"|(\S+)', text)]:
        toks.extend([x for x in t.split(",") if x != ""] if "=" in t or "," in t
                    else [t])
    if not toks:
        raise ValueError("empty ground light-curve specification")
    path = toks[0]
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    sp = GroundSpec(path=path)
    last = None
    for t in toks[1:]:
        if "=" not in t:
            if last == "detrend":
                sp.detrend = sp.detrend + "," + t
                continue
            raise ValueError(f"cannot parse '{t}' in ground spec '{text}'")
        k, v = t.split("=", 1)
        key = _GROUND_KEYS.get(k.strip().lower())
        if key is None:
            raise ValueError(f"unknown key '{k}' in ground spec '{text}'")
        v = v.strip()
        last = key
        if key == "band":
            sp.band = canonical_band(v)
        elif key == "star":
            sp.star = int(float(v.upper().replace("TIC", "").strip()))
        elif key in ("ra", "dec", "aperture_arcsec", "fwhm_arcsec", "pixscale"):
            setattr(sp, key, float(v))
        elif key == "mag":
            sp.mag = v.lower() in ("1", "true", "yes", "y", "mag")
        elif key == "plus_band":
            sp.plus_band = v
        else:
            setattr(sp, key, v)
    if not sp.label:
        sp.label = os.path.splitext(os.path.basename(sp.path))[0]
    return sp


def read_ground_list(path: str) -> list:
    """One parse_ground_spec() line per light curve; '#' starts a comment."""
    out = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip() if not line.lstrip().startswith('"') \
                else line.strip()
            if line:
                out.append(parse_ground_spec(line, base_dir=base))
    return out


def _is_number(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _split_row(line, delim):
    if delim == "ws":
        return line.split()
    return [x.strip() for x in line.split(delim)]


def _read_table_any(path):
    """Header dict (#KEY=VALUE lines), column names and float columns.

    Handles AstroImageJ measurement tables (tab separated, first header cell
    empty), AAVSO/EXOTIC exoplanet reports (#KEY=VALUE header, '#DATE,DIFF,
    ERR,...' column line) and plain CSV / whitespace tables with or without
    a header line.
    """
    meta, names, rows = {}, None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n\r") for ln in fh]
    delim = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if "=" in body and not ("," in body.split("=", 1)[0]):
                k, v = body.split("=", 1)
                meta[k.strip().upper()] = v.strip()
                continue
            # a commented column line ('#DATE,DIFF,ERR' / '# time flux err')
            if names is None and body and not _is_number(body.split()[0].strip(",")):
                d = ("\t" if "\t" in body else "," if "," in body else "ws")
                names = _split_row(body, d)
                delim = d
            continue
        if delim is None:
            delim = "\t" if "\t" in s else "," if "," in s else (
                ";" if ";" in s else "ws")
            md = meta.get("DELIM", "").lower()
            if md in (",", "comma"):
                delim = ","
            elif md in ("tab", "\\t"):
                delim = "\t"
        cells = _split_row(ln if delim == "\t" else s, delim)
        if names is None and not all(_is_number(c) for c in cells if c != ""):
            names = cells
            continue
        rows.append(cells)
    if not rows:
        raise ValueError(f"no data rows in {path}")
    ncol = max(len(r) for r in rows)
    if names is None:
        names = [f"col{i}" for i in range(ncol)]
    # AIJ: the header has an empty first cell for the row-index column
    names = [n.strip() if n.strip() else f"col{i}" for i, n in enumerate(names)]
    if len(names) < ncol:
        names = names + [f"col{i}" for i in range(len(names), ncol)]
    data = {}
    for i, n in enumerate(names[:ncol]):
        col = []
        for r in rows:
            v = r[i] if i < len(r) else ""
            try:
                col.append(float(v))
            except ValueError:
                col.append(np.nan)
        data[n] = np.asarray(col, float)
    return meta, list(data.keys()), data


def _pick_col(names, wanted, explicit=None, default_index=None):
    low = {n.lower(): n for n in names}
    if explicit:
        if explicit in names:
            return explicit
        if explicit.lower() in low:
            return low[explicit.lower()]
        if explicit.isdigit() and int(explicit) < len(names):
            return names[int(explicit)]
        raise ValueError(f"column '{explicit}' not found (have {names[:12]}...)")
    for w in wanted:
        if w.lower() in low:
            return low[w.lower()]
    if default_index is not None and default_index < len(names):
        return names[default_index]
    return None


def _to_btjd(t, system, ra=None, dec=None):
    """Convert a time column to BTJD (BJD_TDB - 2457000).

    BJD_TDB in full JD, reduced (BJD-2450000), MJD or BTJD form is
    recognised by its size.  JD_UTC is converted with astropy (barycentric
    correction for the target direction, geocentre); HJD_UTC gets the
    TDB - UTC offset only (heliocentric vs barycentric < ~4 s).
    """
    t = np.asarray(t, float)
    med = float(np.nanmedian(t))
    if med > 2.4e6:
        jd = t
    elif med > 5.0e4 and med < 1.0e5:       # MJD
        jd = t + 2400000.5
    elif med > 1.0e4 and med < 5.0e4:       # reduced JD (JD - 2450000)?
        jd = t + 2450000.0 if med > 1.5e4 else t + 2400000.0
    else:                                   # already BTJD / BKJD
        jd = t + TIME_ZERO
    sysname = (system or "BJD_TDB").upper().replace("-", "_")
    note = ""
    if sysname in ("JD_UTC", "JD", "UTC", "JDUTC"):
        from astropy.time import Time
        from astropy.coordinates import EarthLocation
        if ra is None or dec is None or not np.isfinite(ra):
            raise ValueError("JD_UTC needs the target coordinates")
        tt = Time(jd, format="jd", scale="utc",
                  location=EarthLocation.from_geocentric(0, 0, 0, unit="m"))
        ltt = tt.light_travel_time(SkyCoord(ra, dec, unit="deg"), "barycentric")
        jd = (tt.tdb + ltt).jd
        note = "JD_UTC converted to BJD_TDB (geocentric barycentric correction)"
    elif sysname in ("HJD_UTC", "HJD"):
        from astropy.time import Time
        tt = Time(jd, format="jd", scale="utc")
        jd = jd + (tt.tdb.jd - tt.utc.jd)
        note = "HJD_UTC: TDB-UTC added; HJD vs BJD differ by < ~4 s"
    return jd - TIME_ZERO, note


def read_ground_lc(spec: GroundSpec, ra=None, dec=None, clip_sigma=5.0):
    """Read one ground light curve.  Returns a dict with t [BTJD], flux
    (normalised to its median), err, covariates {name: array}, meta, notes."""
    meta, names, data = _read_table_any(spec.path)
    notes = []
    tcol = _pick_col(names, ["BJD_TDB", "BJD_TDB_MOBS", "BJD", "BJD_TDB_obs",
                             "DATE", "TIME", "BTJD", "JD_UTC", "JD", "HJD"],
                     spec.time_col, default_index=0)
    fcol = _pick_col(names, ["rel_flux_T1", "DIFF", "FLUX", "REL_FLUX",
                             "NORM_FLUX", "rel_flux", "flux_norm", "MAG",
                             "DMAG", "Source-Sky_T1"],
                     spec.flux_col,
                     default_index=1 if names[0] == tcol else None)
    if fcol is None:
        raise ValueError(f"{spec.path}: no flux column found in {names[:12]}")
    if spec.err_col:
        ecol = _pick_col(names, [], spec.err_col)
    elif fcol.lower().startswith("rel_flux_t"):
        ecol = _pick_col(names, ["rel_flux_err_" + fcol.split("_")[-1]])
    else:
        ecol = _pick_col(names, ["ERR", "FLUX_ERR", "ERROR", "MAG_ERR",
                                 "E_FLUX", "FLUXERR", "rel_flux_err",
                                 "DIFF_ERR", "ERR_DIFF", "MERR", "e_mag"])
        if ecol is None and len(names) > 2 and names[0] == tcol and \
                names[1] == fcol and names[2].startswith("col"):
            ecol = names[2]
    system = spec.time_system or meta.get("DATE_TYPE") or (
        "JD_UTC" if tcol.upper() in ("JD_UTC", "JD") else
        "HJD_UTC" if tcol.upper().startswith("HJD") else "BJD_TDB")
    t, tnote = _to_btjd(data[tcol], system, ra, dec)
    if tnote:
        notes.append(tnote)
    f = np.asarray(data[fcol], float)
    e = (np.asarray(data[ecol], float) if ecol is not None
         else np.full_like(f, np.nan))
    mtype = (meta.get("MEASUREMENT_TYPE", "") + " " + fcol).lower()
    is_mag = spec.mag if spec.mag is not None else (
        "mag" in mtype or fcol.upper() in ("MAG", "DMAG"))
    if is_mag:
        m0 = np.nanmedian(f)
        flux = 10 ** (-0.4 * (f - m0))
        e = 0.4 * np.log(10) * e * flux
        f = flux
        notes.append("magnitudes converted to relative flux")
    ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
    covs = {}
    for n in names:
        if n in (tcol, fcol, ecol):
            continue
        a = np.asarray(data[n], float)
        if np.isfinite(a).mean() > 0.95:
            covs[n] = a
    t, f, e = t[ok], f[ok], e[ok]
    covs = {k: v[ok] for k, v in covs.items()}
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]
    covs = {k: v[order] for k, v in covs.items()}
    med = np.nanmedian(f)
    f, e = f / med, e / med
    if not np.isfinite(e).any() or np.nanmedian(e) <= 0:
        e = np.full_like(f, _mad_sigma(np.diff(f)) / np.sqrt(2))
        notes.append("no usable error column: errors from point-to-point scatter")
    else:
        bad = ~np.isfinite(e) | (e <= 0)
        e[bad] = np.nanmedian(e[~bad])
    # outliers against an 11-point running median (keeps an hours-long dip)
    if f.size > 15 and clip_sigma:
        from scipy.ndimage import median_filter
        rm = median_filter(f, size=11, mode="nearest")
        r = f - rm
        s = _mad_sigma(r)
        keep = np.abs(r) < clip_sigma * s if s > 0 else np.ones_like(f, bool)
        if (~keep).sum():
            notes.append(f"{int((~keep).sum())} outlier(s) clipped at "
                         f"{clip_sigma:.0f} sigma")
        t, f, e = t[keep], f[keep], e[keep]
        covs = {k: v[keep] for k, v in covs.items()}
    # aperture / seeing from the file when not given
    if not np.isfinite(spec.fwhm_arcsec):
        for k in ("FWHM_ARCSEC", "SEEING", "FWHM"):
            if k in meta and _is_number(meta[k]):
                spec.fwhm_arcsec = float(meta[k])
                break
    if not np.isfinite(spec.aperture_arcsec):
        for k in ("APERTURE_ARCSEC", "APERTURE_RADIUS", "APER_RADIUS"):
            if k in meta and _is_number(meta[k]):
                spec.aperture_arcsec = float(meta[k])
                break
    if np.isfinite(spec.pixscale):
        for k in list(covs):
            if not np.isfinite(spec.aperture_arcsec) and k.lower() == "source_radius":
                spec.aperture_arcsec = float(np.nanmedian(covs[k])) * spec.pixscale
            if not np.isfinite(spec.fwhm_arcsec) and k.lower() in ("fwhm_mean", "fwhm_t1"):
                spec.fwhm_arcsec = float(np.nanmedian(covs[k])) * spec.pixscale
    if not spec.observer:
        spec.observer = meta.get("OBSCODE", "") or meta.get("OBSERVER", "")
    if spec.band is None:
        spec.band = canonical_band(meta.get("FILTER") or "R")
        if not meta.get("FILTER"):
            notes.append("band not given: R assumed")
    cad = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    return dict(t=t, flux=f, err=e, covs=covs, meta=meta, notes=notes,
                tcol=tcol, fcol=fcol, ecol=ecol, cadence_d=cad,
                time_system=system)


def _ground_template(dur, ld_fit=None, shape_model=None):
    """Unit-depth transit shape u(dt) (0 outside, 1 at the deepest point),
    dt = time from mid-transit [d]: the TESS limb-darkened fit if available,
    else the trapezoid of the per-transit fits, else a trapezoid."""
    tg = np.linspace(-1.5 * dur, 1.5 * dur, 3001)
    if ld_fit is not None and "model_intrinsic" in ld_fit:
        s = float(ld_fit.get("t0_shift_d", 0.0))
        m = np.asarray(ld_fit["model_intrinsic"](tg + s), float)
        u = float(ld_fit.get("f0", 1.0)) - m
        kind = "TESS limb-darkened fit"
    elif shape_model is not None:
        u = 1.0 - np.asarray(shape_model(tg), float)
        kind = "TESS trapezoid fit"
    else:
        tau = 0.25 * dur
        a = np.abs(tg)
        u = np.clip((0.5 * dur - a) / tau, 0, 1)
        kind = "trapezoid (T12 = 0.25 T14)"
    u = np.clip(u, 0, None)
    pk = float(np.max(u)) if np.max(u) > 0 else 1.0
    u = u / pk

    def tmpl(dt, _tg=tg, _u=u):
        return np.interp(np.asarray(dt, float), _tg, _u, left=0.0, right=0.0)
    return tmpl, kind


def _ephemeris_cov(dt, P, T0):
    """Weighted linear fit of the TESS mid-times: (T0, P, cov[2x2])."""
    try:
        m = np.isfinite(dt["dTc"]) & np.isfinite(dt["e_dTc"]) & (dt["e_dTc"] > 0)
        if m.sum() < 2:
            return None
        E = dt["epoch"][m].astype(float)
        t = (dt["Tc"] + dt["dTc"])[m].astype(float)
        e = np.maximum(dt["e_dTc"][m].astype(float), 0.2 / 1440.0)
        A = np.column_stack([np.ones_like(E), E]) / e[:, None]
        cov = np.linalg.pinv(A.T @ A)
        p = cov @ (A.T @ (t / e))
        r = (t - (p[0] + p[1] * E)) / e
        chi2r = float(np.sum(r ** 2) / max(E.size - 2, 1))
        cov = cov * max(chi2r, 1.0)
        return float(p[0]), float(p[1]), cov
    except Exception:
        return None


def _beta_red_noise(res, err, t, bins_min=(5, 10, 20, 30)):
    """Time-averaging red-noise factor beta (>= 1) of the residuals."""
    betas = []
    s1 = float(np.std(res))
    if t.size < 20 or s1 <= 0:
        return 1.0
    cad = float(np.median(np.diff(t)))
    for bm in bins_min:
        n = int(round(bm / 1440.0 / max(cad, 1e-6)))
        if n < 2:
            continue
        m = res.size // n
        if m < 4:
            continue
        rb = res[: m * n].reshape(m, n).mean(axis=1)
        exp = s1 / np.sqrt(n) * np.sqrt(m / (m - 1.0))
        if exp > 0:
            betas.append(float(np.std(rb)) / exp)
    return float(max(1.0, np.median(betas))) if betas else 1.0


def _coverage(t, Tc, dur, cad):
    """Fraction of the in-transit window with data, and baseline flags."""
    lo, hi = Tc - 0.5 * dur, Tc + 0.5 * dur
    gap = max(3.0 * cad, 3.0 / 1440.0)
    edges = np.linspace(lo, hi, 101)
    mids = 0.5 * (edges[1:] + edges[:-1])
    if t.size == 0:
        return 0.0, False, False, 0.0, 0.0
    idx = np.clip(np.searchsorted(t, mids), 1, t.size - 1)
    near = np.minimum(np.abs(t[idx] - mids), np.abs(t[idx - 1] - mids))
    cov = float(np.mean(near < 0.5 * gap + 0.5 * (edges[1] - edges[0])))
    pre = float(np.clip(lo - t[0], 0, None))
    post = float(np.clip(t[-1] - hi, 0, None))
    has_pre = pre >= min(0.25 * dur, 30 / 1440.0) and np.any(t < lo)
    has_post = post >= min(0.25 * dur, 30 / 1440.0) and np.any(t > hi)
    return cov, bool(has_pre), bool(has_post), pre, post


def fit_ground_lc(lc, spec: GroundSpec, P, T0, dur, tmpl, eph=None,
                  cfg: "Config" = None):
    """Template fit with free depth and mid-time (grid), linear baseline.

    Returns a dict with the best fit, the fit at the predicted mid-time, the
    window upper limit on the depth and coverage diagnostics.
    """
    t, f, e = lc["t"], lc["flux"], lc["err"]
    tm = float(np.median(t))
    E = int(np.round((tm - T0) / P))
    Tc_pred = T0 + E * P
    s_pred = float("nan")
    if eph is not None:
        T0e, Pe, cov = eph
        # epoch numbering of the covariance fit is the same as (P, T0)
        s_pred = float(np.sqrt(max(cov[0, 0] + 2 * E * cov[0, 1]
                                   + E * E * cov[1, 1], 0.0)))
    win_min = float(getattr(cfg, "ground_tc_window_min", 60.0) or 60.0)
    W = max(3.0 * s_pred if np.isfinite(s_pred) else 0.0, win_min / 1440.0)
    W = min(W, 1.0 * dur) if dur > 0 else W
    step = min(1.0 / 1440.0, dur / 200.0)
    grid = Tc_pred + np.arange(-W, W + 0.5 * step, step)
    # baseline design
    span = max(float(np.ptp(t)), 1e-3)
    x = (t - tm) / span
    det = (spec.detrend or "auto").replace(";", ",")
    parts = [p.strip() for p in det.split(",") if p.strip()]
    order = 1
    cov_names = []
    for p in parts:
        pl = p.lower()
        if pl == "auto":
            order = 1
            for k in lc["covs"]:
                if k.lower() in ("airmass", "detrend_1") and k not in cov_names:
                    cov_names.append(k)
                    break
        elif pl.startswith("poly"):
            order = int(pl[4:] or 1)
        elif pl in ("none", "flat"):
            order = 0
        else:
            match = [k for k in lc["covs"] if k.lower() == pl]
            if match:
                cov_names.append(match[0])
    # a quadratic baseline is only safe with baseline on both sides
    cols = [np.ones_like(t)] + [x ** k for k in range(1, order + 1)]
    for k in cov_names:
        c = np.asarray(lc["covs"][k], float)
        sd = float(np.std(c))
        if sd > 0:
            cols.append((c - np.mean(c)) / sd)
    B = np.column_stack(cols)
    w = 1.0 / e
    Bw = B * w[:, None]
    fw = f * w

    def solve(Tc):
        u = tmpl(t - Tc)
        M = np.column_stack([B, -u]) * w[:, None]
        cov = np.linalg.pinv(M.T @ M)
        p = cov @ (M.T @ fw)
        r = fw - M @ p
        return float(r @ r), float(p[-1]), float(np.sqrt(cov[-1, -1])), p, u

    p0, *_ = np.linalg.lstsq(Bw, fw, rcond=None)
    chi0 = float(np.sum((fw - Bw @ p0) ** 2))
    chis = np.empty(grid.size)
    As = np.empty(grid.size)
    sAs = np.empty(grid.size)
    covs_ = np.empty(grid.size)
    for i, Tc in enumerate(grid):
        chis[i], As[i], sAs[i], _, u = solve(Tc)
        covs_[i] = _coverage(t, Tc, dur, lc["cadence_d"])[0]
    ib = int(np.argmin(chis))
    chi_b, A_b, sA_b, p_b, u_b = solve(grid[ib])
    model = (B @ p_b[:-1]) - p_b[-1] * u_b
    base = B @ p_b[:-1]
    res = f - model
    dof = max(t.size - B.shape[1] - 2, 1)
    chi2r = chi_b / dof
    beta = _beta_red_noise(res, e, t)
    infl = float(np.sqrt(max(chi2r, 1.0)) * beta)
    sA_b *= infl
    sAs = sAs * infl
    # fit at the predicted mid-time
    _, A_p, sA_p, _, _ = solve(Tc_pred)
    sA_p *= infl
    dchi = (chi0 - chi_b) / max(infl ** 2, 1e-12)
    snr = A_b / sA_b if sA_b > 0 else 0.0
    # mid-time error: Delta chi2 = 1 on the inflated chi2 curve
    cc = (chis - chi_b) / max(infl ** 2, 1e-12)
    eTc = float("nan")
    try:
        inside = np.where(cc <= 1.0)[0]
        if inside.size:
            lo_i, hi_i = inside.min(), inside.max()
            if lo_i > 0 and hi_i < grid.size - 1:
                eTc = max(0.5 * (grid[hi_i] - grid[lo_i]), 0.5 * step)
    except Exception:
        pass
    cov_b, has_pre, has_post, pre, post = _coverage(t, grid[ib], dur,
                                                    lc["cadence_d"])
    cov_p, pre_p, post_p, _, _ = _coverage(t, Tc_pred, dur, lc["cadence_d"])
    T12 = float(getattr(cfg, "_ground_T12", 0.25 * dur) if cfg else 0.25 * dur)
    ing = _coverage(t, grid[ib] - 0.5 * dur + 0.5 * T12, T12, lc["cadence_d"])[0]
    egr = _coverage(t, grid[ib] + 0.5 * dur - 0.5 * T12, T12, lc["cadence_d"])[0]
    min_snr = float(getattr(cfg, "ground_min_snr", 5.0) or 5.0)
    detected = bool(snr >= min_snr and dchi >= min_snr ** 2 and A_b > 0
                    and cov_b >= 0.3 and (has_pre or has_post))
    timing_ok = bool(detected and np.isfinite(eTc) and (ing >= 0.5 or egr >= 0.5)
                     and (has_pre or has_post))
    # window upper limit on the depth (grid points with >= 50 % coverage)
    okc = covs_ >= 0.5
    A_up = float(np.max(As[okc] + 3.0 * sAs[okc])) if okc.any() else float("nan")
    usable = bool(okc.any() and (pre_p or post_p or has_pre or has_post))
    return dict(
        label=spec.label, path=spec.path, band=spec.band, observer=spec.observer,
        star=spec.star, n=int(t.size), cadence_min=lc["cadence_d"] * 1440,
        epoch=E, Tc_pred=Tc_pred, eTc_pred=s_pred, window_d=W,
        Tc=float(grid[ib]), eTc=eTc, dTc_min=(grid[ib] - Tc_pred) * 1440,
        depth=A_b, e_depth=sA_b, snr=float(snr), dchi2=float(dchi),
        depth_pred=A_p, e_depth_pred=sA_p, depth_up3=A_up,
        chi2_red=float(chi2r), beta=beta, rms_ppm=float(np.std(res) * 1e6),
        coverage=cov_b, coverage_pred=cov_p, has_pre=has_pre, has_post=has_post,
        ingress_cov=ing, egress_cov=egr, detected=detected, timing_ok=timing_ok,
        usable=usable, baseline=("poly%d" % order) + (
            "+" + "+".join(cov_names) if cov_names else ""),
        notes=list(lc["notes"]), time_system=lc["time_system"],
        _t=t, _f=f, _e=e, _model=model, _base=base, _grid=grid, _chi=cc,
        _A=As, _sA=sAs,
    )


# ---- band magnitudes and aperture fractions ------------------------------
def _gaia_to_band(G, x, band):
    """Johnson-Cousins / Sloan magnitude from Gaia G and BP-RP = x
    (Riello et al. 2021, Gaia EDR3 photometric relations, polynomials in x)."""
    x = float(np.clip(x, -0.5, 3.5))
    poly = {
        "V": (-0.02704, 0.01424, -0.2156, 0.01426),
        "R": (-0.02275, 0.3961, -0.1243, -0.01396, 0.003775),
        "I": (0.01753, 0.76, -0.0991),
        "g": (0.2199, -0.6365, -0.1548, 0.0064),
        "r": (-0.09837, 0.08592, 0.1907, -0.1701, 0.02263),
        "i": (-0.293, 0.6404, -0.09609, -0.002104),
    }.get(band)
    if poly is None:
        return float("nan")
    g_minus = sum(c * x ** k for k, c in enumerate(poly))
    return float(G - g_minus)


def band_magnitude(row, band) -> tuple:
    """(magnitude, source) of a star in a ground band.  Uses the catalogue
    column of that band when present, otherwise Gaia G + BP-RP relations,
    otherwise the closest available magnitude (TESS ~ I, V for B)."""
    def val(*keys):
        for k in keys:
            if k in row and row[k] is not None:
                try:
                    v = float(row[k])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    return v, k
        return None
    band = canonical_band(band)
    direct = {"B": ("Bmag",), "V": ("Vmag",), "g": ("gmag",), "r": ("rmag",),
              "i": ("imag",), "z": ("zmag",), "J": ("Jmag",), "H": ("Hmag",),
              "K": ("Kmag",), "TESS": ("Tmag",)}.get(band, ())
    d = val(*direct) if direct else None
    if d:
        return d
    G = val("GAIAmag", "Gmag", "phot_g_mean_mag")
    bp = val("gaiabp", "BPmag", "phot_bp_mean_mag")
    rp = val("gaiarp", "RPmag", "phot_rp_mean_mag")
    gb = {"R": "R", "I": "I", "z": "i", "CBB": "R", "CV": "V"}.get(band, band)
    if G and bp and rp and gb in ("V", "R", "I", "g", "r", "i"):
        m = _gaia_to_band(G[0], bp[0] - rp[0], gb)
        if np.isfinite(m):
            return m, f"Gaia G,BP-RP -> {gb}"
    fb = {"B": ("Vmag", "GAIAmag", "Tmag"), "V": ("Vmag", "GAIAmag", "Tmag"),
          "g": ("Vmag", "GAIAmag", "Tmag"), "R": ("GAIAmag", "Vmag", "Tmag"),
          "r": ("GAIAmag", "Vmag", "Tmag"), "CBB": ("GAIAmag", "Tmag"),
          "CV": ("GAIAmag", "Vmag", "Tmag"), "I": ("Tmag", "GAIAmag"),
          "i": ("Tmag", "GAIAmag"), "z": ("Tmag",), "J": ("Jmag", "Tmag"),
          "H": ("Hmag", "Tmag"), "K": ("Kmag", "Tmag")}.get(band, ("Tmag",))
    d = val(*fb)
    if d:
        return d[0], d[1] + " (approximation)"
    return float("nan"), "none"


def ground_aperture_fractions(stars, spec: GroundSpec, target_tic, cfg=None):
    """Fraction of the ground aperture flux contributed by every star.

    Gaussian seeing PSF (sigma = FWHM / 2.3548), circular aperture of radius
    r centred on the aperture star; the encircled fraction of a star at
    distance d is the non-central chi^2 CDF  P(chi'^2_2(d^2/s^2) < r^2/s^2).
    Returns a DataFrame [ID, sep_arcsec, mag, mag_src, enc, fluxratio].
    """
    import pandas as pd
    from scipy.stats import ncx2, chi2
    df = stars.copy()
    df["ID"] = df["ID"].astype(str)
    notes = []
    fwhm = spec.fwhm_arcsec
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = float(getattr(cfg, "ground_fwhm_default", 3.0) or 3.0)
        notes.append(f"seeing FWHM not given: {fwhm:.1f}\" assumed")
    rap = spec.aperture_arcsec
    if not np.isfinite(rap) or rap <= 0:
        rap = 1.7 * fwhm
        notes.append(f"aperture radius not given: 1.7 x FWHM = {rap:.1f}\" assumed")
    if np.isfinite(spec.ra) and np.isfinite(spec.dec):
        ra0, de0 = spec.ra, spec.dec
        cen = "RA/Dec given"
    else:
        cid = str(int(spec.star)) if spec.star else str(int(target_tic))
        sel = df[df["ID"] == cid]
        if sel.empty:
            raise ValueError(f"aperture star TIC {cid} is not in the field table")
        ra0, de0 = float(sel["ra"].iloc[0]), float(sel["dec"].iloc[0])
        cen = f"TIC {cid}"
    dra = (df["ra"].values.astype(float) - ra0) * np.cos(np.radians(de0)) * 3600
    dde = (df["dec"].values.astype(float) - de0) * 3600
    sep = np.hypot(dra, dde)
    s = fwhm / 2.3548
    lim = (rap / s) ** 2
    nc = (sep / s) ** 2
    enc = np.where(nc > 0, ncx2.cdf(lim, 2, np.maximum(nc, 1e-12)),
                   chi2.cdf(lim, 2))
    enc = np.where(sep > rap + 6 * s, 0.0, enc)
    mags, srcs = [], []
    for _, row in df.iterrows():
        m, src_ = band_magnitude(row, spec.band)
        mags.append(m)
        srcs.append(src_)
    mags = np.asarray(mags, float)
    fl = np.where(np.isfinite(mags), 10 ** (-0.4 * (mags - 20.0)), 0.0)
    c = fl * enc
    tot = float(np.sum(c))
    fr = c / tot if tot > 0 else np.zeros_like(c)
    out = pd.DataFrame(dict(ID=df["ID"].values, sep_arcsec=sep, mag=mags,
                            mag_src=srcs, enc=enc, fluxratio=fr))
    if "tdepth" in df.columns:
        out["tdepth"] = df["tdepth"].values.astype(float)
    return out, dict(fwhm=fwhm, r_ap=rap, centre=cen, notes=notes)


def ground_star_clearance(fits, fracs, target_tic, cfg=None):
    """Per light curve and star: expected vs measured depth, and a status.

    expected depth if star j hosts the signal = tdepth_j x fluxratio_j
    (tdepth: intrinsic depth on j required by the TESS depth, from
    triceratops.calc_depths).  Status:
      excluded   the ground aperture would show a dip at least
                 (1 - tol) x expected but the data allow at most A_up
                 (best depth + 3 sigma, anywhere in the mid-time window), or a
                 dip is detected that is deeper than (1 + tol) x expected
                 + 3 sigma, i.e. j cannot produce it;
      consistent a dip is detected and agrees with the expectation;
      allowed    no conflict, no detection (or too little signal to test);
      unconstrained  the light curve does not cover the transit window.
    """
    import pandas as pd
    tol = float(getattr(cfg, "ground_depth_tol", 0.25) if cfg else 0.25)
    rows = []
    for k, fit in enumerate(fits):
        fr = fracs[k]
        if fr is None:
            continue
        for _, s in fr.iterrows():
            td = float(s.get("tdepth", np.nan))
            if not np.isfinite(td) or td <= 0:
                continue
            exp = td * float(s["fluxratio"])
            st, why = "allowed", "no dip and too faint here to test"
            if not fit["usable"]:
                st, why = "unconstrained", "transit window not covered"
            else:
                A_up = fit["depth_up3"]
                if np.isfinite(A_up) and exp * (1 - tol) > A_up and exp > 0:
                    st, why = "excluded", "its dip would be seen, but is not"
                elif fit["detected"]:
                    sig = np.hypot(fit["e_depth"], tol * exp)
                    why = "detected dip differs from expectation"
                    if fit["depth"] - 3 * fit["e_depth"] > exp * (1 + tol):
                        st, why = "excluded", "too faint here to cause the dip seen"
                    elif abs(fit["depth"] - exp) <= 3 * sig:
                        st, why = "consistent", "expected dip is seen"
            rows.append(dict(lc=fit["label"], band=fit["band"], ID=str(s["ID"]),
                             sep_arcsec=float(s["sep_arcsec"]),
                             fluxratio=float(s["fluxratio"]), tdepth=td,
                             expected_ppm=exp * 1e6,
                             measured_ppm=fit["depth"] * 1e6,
                             e_measured_ppm=fit["e_depth"] * 1e6,
                             upper3_ppm=fit["depth_up3"] * 1e6,
                             detected=fit["detected"], status=st, reason=why,
                             is_target=str(s["ID"]) == str(int(target_tic))))
    tab = pd.DataFrame(rows)
    verdict = {}
    if not tab.empty:
        for sid, g in tab.groupby("ID"):
            stv = set(g["status"])
            if "excluded" in stv and "consistent" in stv:
                verdict[sid] = "conflict"
            elif "excluded" in stv:
                verdict[sid] = "excluded"
            elif "consistent" in stv:
                verdict[sid] = "consistent"
            elif "allowed" in stv:
                verdict[sid] = "allowed"
            else:
                verdict[sid] = "unconstrained"
    return tab, verdict


def ground_chromaticity(fits, fracs, host_ids, tdepth_tess, cfg=None):
    """Intrinsic (dilution-corrected) depth per band for each candidate host,
    compared with the TESS intrinsic depth: chi^2 test of a band-independent
    depth.  Returns {host: dict(per_band=[...], chi2, dof, p, ratio...)}."""
    from scipy.stats import chi2 as _chi2
    sys_frac = float(getattr(cfg, "ground_frac_sys", 0.03) if cfg else 0.03)
    out = {}
    for hid in host_ids:
        pts = []
        for k, fit in enumerate(fits):
            fr = fracs[k]
            if fr is None or not fit["detected"]:
                continue
            row = fr[fr["ID"] == str(hid)]
            if row.empty:
                continue
            g = float(row["fluxratio"].iloc[0])
            if g < 0.05:
                continue
            d = fit["depth"] / g
            e = np.hypot(fit["e_depth"] / g, sys_frac * d)
            pts.append((fit["band"], d, e, fit["label"]))
        if not pts:
            continue
        bands = {}
        for b, d, e, _ in pts:
            bands.setdefault(b, []).append((d, e))
        per_band = []
        for b, v in bands.items():
            d = np.array([x[0] for x in v])
            e = np.array([x[1] for x in v])
            w = 1 / e ** 2
            per_band.append((b, float(np.sum(w * d) / np.sum(w)),
                             float(1 / np.sqrt(np.sum(w))), len(v)))
        td = tdepth_tess.get(str(hid), float("nan"))
        allp = [(b, d, e) for b, d, e, _ in per_band]
        if np.isfinite(td) and td > 0:
            allp.append(("TESS", td, sys_frac * td + 1e-12))
        if len(allp) >= 2:
            d = np.array([x[1] for x in allp])
            e = np.array([x[2] for x in allp])
            w = 1 / e ** 2
            mu = float(np.sum(w * d) / np.sum(w))
            c2 = float(np.sum(w * (d - mu) ** 2))
            dof = len(allp) - 1
            p = float(_chi2.sf(c2, dof))
        else:
            mu, c2, dof, p = float("nan"), float("nan"), 0, float("nan")
        # bluest vs reddest ground band
        rb = float("nan")
        if len(per_band) >= 2:
            sb = sorted(per_band, key=lambda x: GROUND_LAMBDA_UM.get(x[0], 0.6))
            rb = sb[-1][1] / sb[0][1] if sb[0][1] > 0 else float("nan")
        out[str(hid)] = dict(per_band=per_band, tess_intrinsic=td, mean=mu,
                             chi2=c2, dof=dof, p=p, red_over_blue=rb,
                             chromatic=bool(np.isfinite(p) and p < 0.0027))
    return out


# ---- TRICERATOPS+ --------------------------------------------------------
def triceratops_plus_available(target) -> bool:
    try:
        return "external_lc_files" in inspect.signature(
            target.calc_probs).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def select_plus_lcs(fits, specs, max_n=7):
    """One light curve per TRICERATOPS+ band (the most precise usable one,
    target-centred apertures first), at most max_n (TRICERATOPS+ accepts 7)."""
    best = {}
    for k, fit in enumerate(fits):
        if not fit["usable"]:
            continue
        sp = specs[k]
        pb = sp.plus_band or GROUND_PLUS_BAND.get(fit["band"])
        if pb not in ("g", "r", "i", "z", "J", "H", "K"):
            continue
        key = (0 if sp.star in (None,) else 1, fit["e_depth"])
        if pb not in best or key < best[pb][0]:
            best[pb] = (key, k)
    chosen = sorted(best.items(), key=lambda kv: kv[1][0])[:max_n]
    return [(pb, k) for pb, (_, k) in chosen]


def _plus_prepare(target, cfg, log=None):
    """Write the TRICERATOPS+ input files and the fluxratio_<band> columns
    of target.stars; returns the calc_probs keyword arguments."""
    gs = getattr(cfg, "_ground", None)
    if not gs or getattr(cfg, "tri_plus", "auto") == "off":
        return {}
    if not triceratops_plus_available(target):
        if getattr(cfg, "tri_plus", "auto") == "on" and not gs.get("_plus_warned"):
            _log(log, "  ! --tri-plus on, but the installed triceratops is not "
                      "TRICERATOPS+ (calc_probs has no external_lc_files)")
            gs["_plus_warned"] = True
        return {}
    sel = gs.get("plus_selection") or []
    if not sel:
        return {}
    fits, fracs = gs["fits"], gs["fracs"]
    stars = target.stars
    ids = stars["ID"].astype(str).values
    hosts = stars[stars["tdepth"] > 0]["ID"].astype(str).values \
        if "tdepth" in stars.columns else ids
    only_target = len(hosts) <= 1
    files, filts = [], []
    tic = str(int(cfg.tic))
    for pb, k in sel:
        fit = fits[k]
        fr = fracs[k]
        Tc = fit["Tc"] if fit["timing_ok"] else fit["Tc_pred"]
        t = fit["_t"] - Tc
        flux = fit["_f"] / fit["_base"]
        err = fit["_e"] / fit["_base"] * np.sqrt(max(fit["chi2_red"], 1.0)) \
            * fit["beta"]
        if fr is not None:
            fmap = dict(zip(fr["ID"].astype(str), fr["fluxratio"].astype(float)))
        else:
            fmap = {}
        if only_target and fmap:
            g = max(fmap.get(tic, 1.0), 1e-3)
            flux = (flux - (1 - g)) / g     # TRICERATOPS+ does not renormalise
            err = err / g                   # when only the target is left
        col = f"fluxratio_{pb}"
        target.stars[col] = [max(fmap.get(i, 0.0), 1e-4) for i in ids]
        fn = cfg.path(f"ground_plus_{tic}_{pb}_{k + 1}.txt")
        np.savetxt(fn, np.column_stack([t, flux, err]), fmt="%.8f")
        files.append(fn)
        filts.append(pb)
    if not gs.get("_plus_logged"):
        _log(log, f"  TRICERATOPS+: {len(files)} ground light curve(s) passed "
                  f"to calc_probs, bands {filts}"
                  + (" (renormalised to the target: it is the only possible "
                     "host left)" if only_target else
                     " (fluxratio_<band> columns set per star)"))
        gs["_plus_logged"] = True
    return dict(external_lc_files=files, filt_lcs=filts)


# ---- plots ---------------------------------------------------------------
def plot_ground_lcs(fits, cfg, tic):
    n = len(fits)
    if n == 0:
        return
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax, fit in zip(axes.ravel(), fits):
        t0 = fit["Tc_pred"]
        x = (fit["_t"] - t0) * 24
        fd = fit["_f"] / fit["_base"]
        md = fit["_model"] / fit["_base"]
        ax.plot(x, fd, ".", ms=2, color="0.6", label="data / baseline")
        bm = 10.0 / 60.0
        if x.size > 20:
            edges = np.arange(x.min(), x.max() + bm, bm)
            idx = np.digitize(x, edges)
            xb = [x[idx == i].mean() for i in np.unique(idx) if (idx == i).sum() >= 3]
            yb = [fd[idx == i].mean() for i in np.unique(idx) if (idx == i).sum() >= 3]
            ax.plot(xb, yb, "o", ms=3.5, color="k", label="10-min bins")
        ax.plot(x, md, "-", color="C3", lw=1.5, label="fit")
        w = fit["window_d"] * 24
        ax.axvspan(-w, w, color="C0", alpha=0.08, label="mid-time window")
        ax.axvline(0, color="C0", lw=0.8, ls="--")
        ax.set_xlabel(f"hours from predicted mid-time (epoch {fit['epoch']})")
        ax.set_ylabel("relative flux")
        star = f"TIC {fit['star']}" if fit["star"] else "target"
        ax.set_title(f"{fit['label']}  [{fit['band']}, aperture on {star}]",
                     fontsize=9)
        txt = (f"depth {fit['depth'] * 1e6:.0f} ± {fit['e_depth'] * 1e6:.0f} ppm, "
               f"S/N {fit['snr']:.1f}\n"
               + (f"Tc - pred = {fit['dTc_min']:+.1f} ± {fit['eTc'] * 1440:.1f} min"
                  if fit["timing_ok"] else "mid-time not constrained")
               + f"\ncoverage {fit['coverage'] * 100:.0f} %, "
                 f"beta {fit['beta']:.2f}")
        ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=7.5,
                va="bottom", bbox=dict(fc="w", ec="0.8", alpha=0.85))
        ax.legend(fontsize=6.5, loc="upper right")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(cfg.path(f"ground_lightcurves_{tic}.pdf"))
    plt.close(fig)


def plot_ground_clearance(tab, cfg, tic, max_stars=8):
    """One panel per ground light curve: for each star that could host the
    TESS signal, the depth this aperture would show if that star were the
    host (points), against what was measured (band or upper limit)."""
    if tab is None or tab.empty:
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lcs = list(dict.fromkeys(tab["lc"]))
    fig, axes = plt.subplots(len(lcs), 1, figsize=(8.0, 3.3 * len(lcs)),
                             squeeze=False)
    colors = {"excluded": "C3", "consistent": "C2", "allowed": "C0",
              "unconstrained": "0.6"}
    tid = str(int(tic))
    for ax, lab in zip(axes[:, 0], lcs):
        g = tab[tab["lc"] == lab].sort_values("sep_arcsec")
        g = g[g["expected_ppm"] > 1].head(max_stars)
        if g.empty:
            ax.axis("off")
            continue
        xs = np.arange(len(g))
        r0 = g.iloc[0]
        ymin = max(0.3 * float(g["expected_ppm"].min()), 1.0)
        # what was measured
        if r0["detected"]:
            m, e = r0["measured_ppm"], r0["e_measured_ppm"]
            ax.axhspan(max(m - 3 * e, ymin), m + 3 * e, color="C1", alpha=0.25)
            ax.axhline(m, color="C1", lw=1.2)
            meas = f"dip detected: {m:.0f} ± {e:.0f} ppm"
        else:
            meas = "no dip detected"
        if np.isfinite(r0["upper3_ppm"]) and r0["upper3_ppm"] > 0:
            ax.axhline(r0["upper3_ppm"], color="k", ls=":", lw=1.2)
            ax.text(len(g) - 0.55, r0["upper3_ppm"] * 1.12,
                    f"largest dip allowed by the data (3σ): "
                    f"{r0['upper3_ppm']:.0f} ppm", ha="right", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none",
                              alpha=0.8))
        # what each star would cause
        for xi, (_, r) in zip(xs, g.iterrows()):
            ax.plot(xi, r["expected_ppm"], "o", color=colors[r["status"]],
                    ms=10 if r["is_target"] else 8, mec="k", mew=0.8)
            ax.annotate(f"{r['expected_ppm']:.0f} ppm\n{r['status']}:\n"
                        f"{r.get('reason', '')}",
                        (xi, r["expected_ppm"]), textcoords="offset points",
                        xytext=(12, -4), fontsize=6.8, va="center",
                        color=colors[r["status"]],
                        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.8",
                                  alpha=0.9))
        ax.set_yscale("log")
        ax.set_ylim(ymin, max(float(g["expected_ppm"].max()),
                              float(r0["measured_ppm"] + 3 * r0["e_measured_ppm"])
                              if r0["detected"] else 0,
                              float(r0["upper3_ppm"]) if np.isfinite(r0["upper3_ppm"])
                              else 0) * 3)
        ax.set_xlim(-0.5, len(g) - 0.3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"TIC {r['ID']}" + (" (target)" if str(r["ID"]) == tid
                                                else "") +
                            f"\n{r['sep_arcsec']:.1f}\" from aperture centre,"
                            f"\n{100 * r['fluxratio']:.1f} % of the aperture light"
                            for _, r in g.iterrows()], fontsize=7)
        ax.set_ylabel("dip in this aperture [ppm]")
        cen = g[g["sep_arcsec"] < 0.5]
        cen_lab = (f"TIC {cen['ID'].iloc[0]}" + (" (target)" if cen["ID"].iloc[0] == tid
                                                  else "")) if len(cen) else "?"
        ax.set_title(f"Light curve {lab}  -  band {r0['band']}, aperture on "
                     f"{cen_lab}  -  {meas}", fontsize=8.5)
    handles = [Line2D([], [], marker="o", ls="", color=colors[k], mec="k",
                      label=f"{k}") for k in ("consistent", "excluded", "allowed")]
    handles += [Patch(color="C1", alpha=0.35, label="measured dip ± 3σ"),
                Line2D([], [], color="k", ls=":", label="3σ upper limit")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Points: dip this aperture would show if that star hosted the "
                 "TESS signal", fontsize=9, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(cfg.path(f"ground_clearance_{tic}.pdf"))
    plt.close(fig)


def plot_ground_chromaticity(chrom, cfg, tic):
    if not chrom:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, (hid, c) in enumerate(chrom.items()):
        for b, d, e, n in c["per_band"]:
            ax.errorbar(GROUND_LAMBDA_UM.get(b, 0.6) + 0.004 * j, d * 1e6,
                        yerr=e * 1e6, fmt="o", color=f"C{j}", capsize=3)
            ax.annotate(b, (GROUND_LAMBDA_UM.get(b, 0.6), d * 1e6),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        if np.isfinite(c["tess_intrinsic"]):
            ax.errorbar(0.80 + 0.004 * j, c["tess_intrinsic"] * 1e6,
                        yerr=0.03 * c["tess_intrinsic"] * 1e6, fmt="s",
                        color=f"C{j}", mfc="w", capsize=3)
        ax.plot([], [], "o", color=f"C{j}",
                label=f"host TIC {hid}: p(achromatic) = {c['p']:.3g}")
    ax.set_xlabel("effective wavelength [µm]  (open square: TESS)")
    ax.set_ylabel("intrinsic depth [ppm]")
    ax.set_title("Depth vs band (corrected for the aperture dilution)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.path(f"ground_chromaticity_{tic}.pdf"))
    plt.close(fig)


# ---- driver --------------------------------------------------------------
def ground_photometry_fits(cfg, tic, P, T0, dur, dt, ld_fit, shape_model,
                           star, log=None):
    """Step 7d: read and fit all ground light curves (no field model needed).
    Stores the results in cfg._ground and the mid-times in cfg._ground_times.
    """
    specs = list(getattr(cfg, "ground_specs", None) or [])
    if not specs:
        return None
    _log(log, f"\n--- Ground-based photometry ({len(specs)} light curve(s)) ---")
    tmpl, kind = _ground_template(dur, ld_fit, shape_model)
    if ld_fit is not None and np.isfinite(ld_fit.get("T14_h", np.nan)) and \
            np.isfinite(ld_fit.get("T23_h", np.nan)):
        cfg._ground_T12 = max(0.5 * (ld_fit["T14_h"] - ld_fit["T23_h"]) / 24, 0.05 * dur)
    else:
        cfg._ground_T12 = 0.5 * dur if (ld_fit or {}).get("grazing") else 0.25 * dur
    eph = _ephemeris_cov(dt, P, T0) if dt is not None else None
    _log(log, f"  transit template: {kind}; T14 = {dur * 24:.2f} h; "
              f"mid-time window >= {cfg.ground_tc_window_min:.0f} min"
              + ("" if eph is None else " or 3 sigma of the TESS ephemeris"))
    fits, used = [], []
    ra, dec = star.get("ra"), star.get("dec")
    for sp in specs:
        try:
            lc = read_ground_lc(sp, ra=ra, dec=dec,
                                clip_sigma=cfg.ground_clip_sigma)
            fit = fit_ground_lc(lc, sp, P, T0, dur, tmpl, eph=eph, cfg=cfg)
        except Exception as exc:
            _log(log, f"  ! {sp.path}: {exc}")
            continue
        fits.append(fit)
        used.append(sp)
        ctr = f"TIC {sp.star}" if sp.star else "target"
        _log(log, f"  {fit['label']}: band {fit['band']}, aperture on {ctr}, "
                  f"{fit['n']} points, cadence {fit['cadence_min']:.1f} min, "
                  f"baseline {fit['baseline']}, time {fit['time_system']}")
        for nt in fit["notes"]:
            _log(log, f"      note: {nt}")
        _log(log, f"      epoch {fit['epoch']}: predicted Tc = "
                  f"{fit['Tc_pred'] + TIME_ZERO:.5f} BJD_TDB"
                  + (f" ± {fit['eTc_pred'] * 1440:.1f} min"
                     if np.isfinite(fit['eTc_pred']) else ""))
        _log(log, f"      coverage of the transit {fit['coverage'] * 100:.0f} % "
                  f"(baseline before: {'yes' if fit['has_pre'] else 'no'}, "
                  f"after: {'yes' if fit['has_post'] else 'no'}); "
                  f"rms {fit['rms_ppm']:.0f} ppm, chi2_red "
                  f"{fit['chi2_red']:.2f}, beta {fit['beta']:.2f}")
        _log(log, f"      depth = {fit['depth'] * 1e6:.0f} ± "
                  f"{fit['e_depth'] * 1e6:.0f} ppm (S/N {fit['snr']:.1f}, "
                  f"Delta chi2 {fit['dchi2']:.1f}); at the predicted time "
                  f"{fit['depth_pred'] * 1e6:.0f} ± {fit['e_depth_pred'] * 1e6:.0f}"
                  f" ppm; 3-sigma upper limit in the window "
                  f"{fit['depth_up3'] * 1e6:.0f} ppm")
        if fit["timing_ok"]:
            _log(log, f"      mid-time Tc = {fit['Tc'] + TIME_ZERO:.5f} ± "
                      f"{fit['eTc'] * 1440:.1f} min BJD_TDB "
                      f"(O - C = {fit['dTc_min']:+.1f} min)")
        _log(log, "      " + ("DETECTED" if fit["detected"] else
                              "no significant dip" if fit["usable"] else
                              "NOT USABLE (transit window not covered)"))
    cfg._ground = dict(fits=fits, specs=used, fracs=[None] * len(fits))
    rows = [(f["Tc"], f["eTc"], f"ground:{f['label']}") for f in fits
            if f["timing_ok"]]
    cfg._ground_times = rows
    if rows:
        with open(cfg.path(f"ground_times_{tic}.txt"), "w") as fh:
            fh.write("# BJD_TDB  err_min  label  (ground mid-times, "
                     "--extra-times format)\n")
            for t, e, lab in rows:
                fh.write(f"{t + TIME_ZERO:.6f}  {e * 1440:.2f}  {lab.split(':', 1)[1]}\n")
    try:
        plot_ground_lcs(fits, cfg, tic)
    except Exception as exc:
        _log(log, f"  ! ground light-curve plot failed: {exc}")
    _summary(cfg)["ground"] = dict(
        n_lc=len(fits), n_detected=int(sum(f["detected"] for f in fits)),
        n_timing=len(rows),
        lcs={f["label"]: dict(band=f["band"], depth_ppm=f["depth"] * 1e6,
                              e_depth_ppm=f["e_depth"] * 1e6, snr=f["snr"],
                              detected=f["detected"], usable=f["usable"],
                              Tc_bjd=(f["Tc"] + TIME_ZERO if f["timing_ok"]
                                      else float("nan")),
                              eTc_min=(f["eTc"] * 1440 if f["timing_ok"]
                                       else float("nan")))
             for f in fits})
    return cfg._ground


def ground_photometry_field(target, cfg, tic, log=None) -> set:
    """Step 8c: aperture fractions, star clearance, chromaticity, the
    TRICERATOPS+ selection.  Returns the TIC IDs to remove (never the
    target)."""
    gs = getattr(cfg, "_ground", None)
    if not gs or not gs.get("fits"):
        return set()
    fits, specs = gs["fits"], gs["specs"]
    stars = target.stars
    fracs = []
    _log(log, "\n--- Ground-based photometry: apertures and star clearance ---")
    for fit, sp in zip(fits, specs):
        try:
            fr, info = ground_aperture_fractions(stars, sp, tic, cfg)
        except Exception as exc:
            _log(log, f"  ! {fit['label']}: aperture model failed ({exc})")
            fracs.append(None)
            continue
        fracs.append(fr)
        top = fr.sort_values("fluxratio", ascending=False).head(4)
        _log(log, f"  {fit['label']} [{fit['band']}]: aperture r = "
                  f"{info['r_ap']:.1f}\", FWHM {info['fwhm']:.1f}\", centre "
                  f"{info['centre']}; flux fractions: " + ", ".join(
                      f"{r.ID} {r.fluxratio:.3f}" for r in top.itertuples()))
        for nt in info["notes"]:
            _log(log, f"      note: {nt}")
        srcs = set(fr.loc[fr["fluxratio"] > 0.01, "mag_src"])
        if any("approximation" in s for s in srcs):
            _log(log, f"      note: some band magnitudes are approximations "
                      f"({sorted(srcs)})")
    gs["fracs"] = fracs
    tab, verdict = ground_star_clearance(fits, fracs, tic, cfg)
    gs["clearance"], gs["verdict"] = tab, verdict
    drop = set()
    tid = str(int(tic))
    if not tab.empty:
        tab.to_csv(cfg.path(f"ground_clearance_{tic}.csv"), index=False)
        show = tab[(tab["expected_ppm"] > 100.0) | tab["is_target"]].head(40)
        cols = ["lc", "band", "ID", "sep_arcsec", "fluxratio", "expected_ppm",
                "measured_ppm", "e_measured_ppm", "upper3_ppm", "status"]
        _log(log, show[cols].round(3).to_string(index=False))
        _log(log, "  combined verdict per star (stars that can host the TESS "
                  "signal):")
        for sid, v in sorted(verdict.items(), key=lambda kv: kv[0] != tid):
            _log(log, f"    TIC {sid}{' (target)' if sid == tid else ''}: {v}")
        excl = {s for s, v in verdict.items() if v == "excluded"}
        conflict = {s for s, v in verdict.items() if v == "conflict"}
        if conflict:
            _log(log, f"  ! conflicting ground results for {sorted(conflict)}: "
                      f"these stars are NOT removed")
        if tid in excl:
            _log(log, "  !! the ground data are INCONSISTENT with the target "
                      "hosting the signal (wrong depth or no dip where one is "
                      "expected). The target is kept in TRICERATOPS, but treat "
                      "the candidate as a likely false positive (check "
                      "the aperture, filter and comparison stars first).")
        on_nb = [s for s, v in verdict.items() if v == "consistent" and s != tid]
        if on_nb and tid in excl:
            _log(log, f"  !! signal consistent with neighbour(s) {on_nb}: "
                      f"probable nearby eclipsing binary (NEB)")
        drop = {int(s) for s in excl if s != tid}
        if drop and getattr(cfg, "ground_clear", True):
            _log(log, f"  {len(drop)} star(s) cleared by the ground photometry "
                      f"(removed before calc_probs): {sorted(drop)}")
        elif drop:
            _log(log, f"  {len(drop)} star(s) would be cleared; NOT removed "
                      f"(--ground-no-clear)")
            drop = set()
        _summary(cfg).setdefault("ground", {}).update(
            verdict=verdict, cleared=sorted(int(s) for s in excl if s != tid),
            target_status=verdict.get(tid, "untested"))
    try:
        plot_ground_clearance(tab, cfg, tic)
    except Exception as exc:
        _log(log, f"  ! clearance plot failed: {exc}")
    # chromaticity for the target and for every host not excluded
    td = {}
    if "tdepth" in stars.columns:
        td = dict(zip(stars["ID"].astype(str), stars["tdepth"].astype(float)))
    hosts = [tid] + [s for s, v in verdict.items()
                     if v == "consistent" and s != tid]
    chrom = ground_chromaticity(fits, fracs, hosts, td, cfg)
    gs["chromaticity"] = chrom
    for hid, c in chrom.items():
        pb = ", ".join(f"{b}: {d * 1e6:.0f} ± {e * 1e6:.0f}" for b, d, e, n in
                       c["per_band"])
        _log(log, f"  intrinsic depth if TIC {hid} is the host [ppm]: {pb}; "
                  f"TESS {c['tess_intrinsic'] * 1e6:.0f}")
        if c["dof"] > 0:
            _log(log, f"      band-independent depth: chi2 = {c['chi2']:.1f} "
                      f"for {c['dof']} dof, p = {c['p']:.3g}"
                      + (f"; red/blue = {c['red_over_blue']:.3f}"
                         if np.isfinite(c["red_over_blue"]) else "")
                      + ("  -> CHROMATIC (eclipsing binary / blend?)"
                         if c["chromatic"] else "  -> achromatic"))
    if chrom:
        _summary(cfg).setdefault("ground", {})["chromaticity"] = {
            h: dict(p=c["p"], chromatic=c["chromatic"],
                    red_over_blue=c["red_over_blue"]) for h, c in chrom.items()}
        try:
            plot_ground_chromaticity(chrom, cfg, tic)
        except Exception as exc:
            _log(log, f"  ! chromaticity plot failed: {exc}")
    # TRICERATOPS+
    if getattr(cfg, "tri_plus", "auto") != "off":
        gs["plus_selection"] = select_plus_lcs(fits, specs)
        if triceratops_plus_available(target) and gs["plus_selection"]:
            _log(log, "  TRICERATOPS+ detected: ground light curves "
                      + ", ".join(f"{fits[k]['label']} ({pb})"
                                  for pb, k in gs["plus_selection"])
                      + " will enter the likelihood")
        elif gs["plus_selection"]:
            _log(log, "  (triceratops is not TRICERATOPS+: the ground data "
                      "enter only through the star clearance)")
    return drop


def _summary(cfg) -> dict:
    """Key results collected for the end-of-run summary (summary_<TIC>.json)."""
    if not isinstance(getattr(cfg, "_summary", None), dict):
        cfg._summary = {}
    return cfg._summary


def _log(fh, text):
    print(text)
    if fh is not None:
        print(text, file=fh)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(cfg: Config):
    
    tic = cfg.tic
    log = open(cfg.path(f"triceratops_output_TIC{tic}.txt"), "w")

    _log(log, f"=== Statistical validation of TIC {tic} ===\n")
    _log(log, f"Output directory:       {cfg.outdir}")
    _log(log, f"TESScut cache directory: {cfg.cache_dir()}")
    _log(log, "(existing cutouts for the same TIC/sector/size are reused)\n")
    cmd = getattr(cfg, "_command_line", None)
    if cmd:
        _log(log, f"Command line: {cmd}\n")

    cfg._summary = {}
    if cmd:
        _summary(cfg)["command_line"] = cmd
    # TRICERATOPS+ (and triceratops < 1.1) have no dilution_corrected
    # argument: a light curve on the target's flux scale would then be
    # corrected for the dilution a second time.  Keep the aperture scale.
    try:
        _has_dc = "dilution_corrected" in inspect.signature(
            tr.target.calc_probs).parameters
    except (TypeError, ValueError, AttributeError):
        _has_dc = True
    if not _has_dc and cfg.dilution_mode == "sector":
        cfg.dilution_mode = "none"
        _log(log, "Installed triceratops has no 'dilution_corrected' argument "
                  "(TRICERATOPS+ or < 1.1): --dilution-mode none is used, so "
                  "TRICERATOPS receives the light curve on the aperture flux "
                  "scale and does the dilution correction itself.\n")
    problems = environment_problems()
    if problems:
        msg = ("The Python environment cannot run TRICERATOPS:\n  - "
               + "\n  - ".join(problems))
        _log(log, msg)
        log.close()
        raise SystemExit(msg)
    _summary(cfg)["triceratops_plus"] = bool(
        triceratops_plus_available(SimpleNamespace(
            calc_probs=getattr(tr.target, "calc_probs", None))))
    setup_vizier(log)
    tstar = stellar_params(tic, kic=getattr(cfg, "kic", None))
    # host-star parameters (TLS duration, LD, density test, TRICERATOPS):
    # the TIC star, or --host-tic / --star-source kepler / --stellar
    star = apply_star_overrides(tstar, cfg, tic, log=log)
    cfg._star = star
    cfg._target_star = tstar
    if is_kepler():
        _log(log, f"Mission: Kepler, KIC {cfg.kic} (= TIC {tic}); quarters take "
                  f"the place of sectors; times in BKJD = BJD_TDB - 2454833; "
                  f"long-cadence {cfg.lc_source.upper()} light curves\n")
        _summary(cfg).update(mission="Kepler", kic=int(cfg.kic),
                             time_system="BKJD = BJD_TDB - 2454833")
    else:
        _summary(cfg).update(mission="TESS", time_system="BTJD = BJD_TDB - 2457000")
    cfg._tesscut_coords = (int(tic), SkyCoord(tstar["ra"], tstar["dec"], unit="deg"))
    _log(log, f"Stellar parameters ({star.get('source', 'TIC')}):")
    for k, v in star.items():
        if k != "source":
            _log(log, f"  {k:10s} = {v}")
    _summary(cfg).update(tic=int(tic), script_version=__version__,
                         Teff=star.get("Teff"), R_star=star.get("rad"),
                         M_star=star.get("mass"), Tmag=star.get("Tmag"),
                         star_source=star.get("source", "TIC"))
    if getattr(cfg, "host_tic", None) and int(cfg.host_tic) != int(tic):
        _summary(cfg)["host_tic"] = int(cfg.host_tic)

    # field catalogue: PSF model for the optimal aperture and the dilution
    cfg._field = None
    cfg._dilution_applied = False
    if cfg.aperture_mode == "optimal" or cfg.dilution_mode == "sector":
        cfg._field = field_catalog(tic, tstar["ra"], tstar["dec"], cfg, log=log)
        if cfg._field is not None:
            _log(log, f"Field catalogue: {len(cfg._field)} TIC stars within "
                      f"{cfg.field_radius_arcsec:.0f} arcsec (PSF model for "
                      f"apertures and dilution)")

    sectors = find_tess_sectors(tic, cfg=cfg)
    src = (getattr(cfg, "lc_source", "tesscut") or "tesscut").lower()
    if is_kepler():
        _log(log, f"\nLight-curve source: Kepler long cadence {src.upper()}")
        _log(log, f"Quarters with data: {[int(x) for x in sectors]}")
    elif src in ("sap", "pdcsap"):
        _log(log, f"\nLight-curve source: SPOC {src.upper()}")
        _log(log, f"Sectors with SPOC LC coverage: {[int(x) for x in sectors]}")
    else:
        _log(log, "\nLight-curve source: TESScut FFI cutouts")
        _log(log, f"Sectors with FFI coverage: {[int(x) for x in sectors]}")
    if cfg.exclude_sectors:
        excl = {int(s) for s in cfg.exclude_sectors}
        before = list(sectors)
        sectors = np.array([int(s) for s in sectors if int(s) not in excl])
        dropped = [s for s in before if int(s) in excl]
        _log(log, f"Excluding sectors: {dropped}")
        _log(log, f"Remaining sectors: {[int(x) for x in sectors]}")
        if sectors.size == 0:
            raise RuntimeError("All sectors excluded")

    # ---- 1. extract every sector -----------------------------------------
    sectors_data = []
    for s in sectors:
        print(f"\nExtracting sector {s} ...")
        try:
            sectors_data.append(extract_sector(tic, int(s), cfg))
        except Exception as exc:
            print(f"  ! sector {s} failed: {exc}")
    if not sectors_data:
        raise RuntimeError("No usable sectors")
    sectors = np.array([sd.sector for sd in sectors_data])

    # apertures and the target's flux fraction per sector
    _log(log, "\n--- Apertures (target flux fraction g from the catalogue "
              "PSF model) ---")
    for sd in sectors_data:
        _log(log, f"  sector {sd.sector:3d}: {sd.aperture_kind:12s} "
                  f"{len(sd.ap_pixels):3d} px  g = {sd.g_target:6.3f}  "
                  f"cadence {sd.cadence_d * 1440:5.1f} min")
    want = cfg.dilution_mode == "sector" and src != "pdcsap"
    if want:
        missing = [sd.sector for sd in sectors_data
                   if not (np.isfinite(sd.g_target) and sd.g_target > 0.02)]
        if missing and len(missing) < len(sectors_data):
            _log(log, f"  ! no target fraction for sector(s) {missing}: "
                      f"dropped (their depth scale would differ)")
            sectors_data = [sd for sd in sectors_data if sd.sector not in missing]
            sectors = np.array([sd.sector for sd in sectors_data])
        elif missing:
            _log(log, "  ! no target fractions at all (field catalogue "
                      "unavailable): dilution correction switched off")
            want = False
    cfg._dilution_applied = bool(want)
    if not want:
        cfg.dilution_mode = "none"
    else:
        gs = np.array([sd.g_target for sd in sectors_data])
        _log(log, f"  every detrended sector is put on the target's scale, "
                  f"f -> 1 + (f - 1)/g (g = {gs.min():.3f} - {gs.max():.3f}); "
                  f"all depths below are on-target depths")

    # corrected (background-subtracted) LCs before any detrending
    print("\nSaving background-corrected light curves ...")
    save_sector_lightcurves(sectors_data, cfg, tic, tag="corrected")
    plot_sector_panels(sectors_data, cfg, tic, tag="corrected")

    # ---- 2. first detrending pass ----------------------------------------
    # Short fixed window: removes fast stellar/instrumental variability for
    # the blind search without assuming a transit duration.  (The old scheme
    # used window_factor * t14(P=10 d), which could be several days for large
    # stars and left residual activity that inflated the folded scatter.)
    w1 = max(cfg.window_length_search, cfg.window_fallback)
    print(f"\nFirst detrending pass, window = {w1 * 24:.1f} h "
          f"(window_length_search = {cfg.window_length_search} d)")
    for sd in sectors_data:
        detrend_sector(sd, w1, cfg)
        renormalize_sector_oot(sd, cfg)          # continuum → 1 (no ephemeris yet)
    sectors_data = filter_bad_sectors(sectors_data, cfg, log=log, tag="first detrend")
    sectors = np.array([sd.sector for sd in sectors_data])
    time, flux, ferr, sector_ids = stitch(sectors_data)
    time, flux, ferr, sector_ids = clip_outliers(
        time, flux, ferr, sector_ids, sigma=cfg.outlier_sigma)

    # ---- 3. TLS search ----------------------------------------------------
    if cfg.use_known_ephemeris:
        P_orb, T0, dur = cfg.P_orb_known, cfg.T0_known, cfg.dur_known
        lc_pass1 = (time.copy(), flux.copy(), ferr.copy())
        depth_tls = cfg.depth_known
        results = None
        _log(log, f"\nUsing the KNOWN ephemeris (no TLS search): "
                  f"P = {P_orb} d, T0 = {T0} {TIME_LABEL}, T14 = {dur * 24:.2f} h, "
                  f"depth = {depth_tls * 1e6:.0f} ppm  "
                  f"[check that this belongs to TIC {tic}]")
        P_orb, T0 = refine_known_ephemeris(time, flux, ferr, P_orb, T0,
                                           dur, cfg, log=log)
        # re-detrend with the known ephemeris so transits are preserved
        w2 = max(cfg.window_factor * dur, cfg.window_fallback)
        print(f"\nDetrending with known-ephemeris transit mask, "
              f"window = {w2 * 24:.1f} h")
        for sd in sectors_data:
            detrend_sector(sd, w2, cfg,
                           mask_transits=transit_mask(sd.time, P_orb, dur, T0))
            renormalize_sector_oot(sd, cfg, P=P_orb, T0=T0, dur=dur)
        sectors_data = filter_bad_sectors(sectors_data, cfg, log=log,
                                          tag="known-ephemeris detrend")
        sectors = np.array([sd.sector for sd in sectors_data])
        time, flux, ferr, sector_ids = stitch(sectors_data)
        time, flux, ferr, sector_ids = clip_outliers(
            time, flux, ferr, sector_ids, sigma=cfg.outlier_sigma,
            P=P_orb, T0=T0, dur=dur)
        plot_transits_on_sectors(sectors_data, P_orb, T0, dur, cfg, tic)
    else:
        if cfg.window_length_grid and len(cfg.window_length_grid) > 1:
            results, w_best, (time, flux, ferr, sector_ids), _rows = \
                search_multi_window(sectors_data, star, cfg, log=log)
            _log(log, f"  first-pass window adopted: {w_best:g} d")
        else:
            bt, bf, be = bin_for_search(time, flux, ferr,
                                        cfg.tls_search_bin_minutes)
            results = run_tls(bt, bf, be, star, cfg, tag=" (pass 1, binned)")
        _ef, _nin = candidate_edge_fraction(time, results.period, results.T0,
                                            results.duration)
        if np.isfinite(_ef):
            _log(log, f"\n  first-pass candidate: P = {results.period:.4f} d, "
                      f"SDE = {results.SDE:.1f}; {_ef * 100:.0f} % of its "
                      f"{_nin} in-transit cadences lie within 0.5 d of a "
                      f"data-segment edge"
                      + ("  <-- suspicious (ramps / scattered light)"
                         if _ef > 0.3 else ""))
        if results.SDE < cfg.SDE_threshold:
            _log(log, f"\nNo significant detection: "
                      f"SDE = {results.SDE:.1f} < {cfg.SDE_threshold}")
            # still save diagnostics so the weak peak can be inspected
            plot_tls_diagnostics(results, time, flux, ferr, cfg, tic,
                                 tag=" (pass 1, weak)")
            plot_transits_on_sectors(sectors_data, results.period, results.T0,
                                     results.duration, cfg, tic)
            log.close()
            return
        P_orb, T0, dur = results.period, results.T0, results.duration
        depth_tls = 1.0 - results.depth

        # TLS duration is often far too short on binned FFIs.  Estimate a
        # realistic T14 from the (still first-pass) LC and use a *generous*
        # mask so the second detrend cannot eat the transit wings.
        dur_est, depth_est = estimate_duration_depth(
            time, flux, ferr, P_orb, T0, dur_guess=max(dur, 0.05))
        # physical duration from stellar density as a lower bound
        try:
            dur_phys = float(t14(R_s=star["rad"], M_s=star["mass"],
                                 P=P_orb, small_planet=True))
            if not np.isfinite(dur_phys) or dur_phys <= 0:
                dur_phys = 0.0
        except Exception:
            dur_phys = 0.0
        dur_mask = max(dur_est, dur_phys, 5.0 * dur, 0.05)
        lc_pass1 = (time.copy(), flux.copy(), ferr.copy())
        print(f"  transit mask duration for 2nd detrend: "
              f"{dur_mask * 24:.2f} h "
              f"(TLS={dur * 24:.2f} h, data={dur_est * 24:.2f} h, "
              f"physical={dur_phys * 24:.2f} h)")

        # ---- 4. re-detrend with the transits masked, then re-run TLS -----
        w2 = max(cfg.window_factor * dur_mask, cfg.window_fallback)
        print(f"\nSecond detrending pass with transits masked, "
              f"window = {w2 * 24:.1f} h")
        for sd in sectors_data:
            detrend_sector(sd, w2, cfg,
                           mask_transits=transit_mask(
                               sd.time, P_orb, dur_mask, T0))
            renormalize_sector_oot(sd, cfg, P=P_orb, T0=T0, dur=dur_mask)
        sectors_data = filter_bad_sectors(sectors_data, cfg, log=log,
                                          tag="second detrend")
        sectors = np.array([sd.sector for sd in sectors_data])
        time, flux, ferr, sector_ids = stitch(sectors_data)  # full resolution
        time, flux, ferr, sector_ids = clip_outliers(
            time, flux, ferr, sector_ids, sigma=cfg.outlier_sigma,
            P=P_orb, T0=T0, dur=dur_mask)
        bt, bf, be = bin_for_search(time, flux, ferr, cfg.tls_search_bin_minutes)
        results = run_tls(bt, bf, be, star, cfg, tag=" (pass 2, binned)")
        P_orb, T0, dur = results.period, results.T0, results.duration
        depth_tls = 1.0 - results.depth

        # Fine-tune P/T0 and re-estimate T14 + depth on the full-res LC
        P_orb, T0, dur, depth_ref = refine_ephemeris(
            time, flux, ferr, P_orb, T0, max(dur, dur_mask))
        if depth_ref > 0:
            depth_tls = depth_ref

        # Periodogram + folded transit on the final (full-res) detrended LC
        plot_tls_diagnostics(results, time, flux, ferr, cfg, tic,
                             tag=" (pass 2)", P=P_orb, T0=T0, dur=dur,
                             depth=depth_tls)
        plot_transits_on_sectors(sectors_data, P_orb, T0, dur, cfg, tic)

    # ---- 5. per-transit depths -------------------------------------------
    # Runs with and without a TLS result.  In known-ephemeris mode (results
    # is None) a trapezoid fitted to the folded LC provides the unit shape;
    # before, this whole block was skipped, so no individual transits were
    # ever detected or tabulated.
    shape_model, dur_pt = None, dur
    if results is None:
        try:
            t14_s, depth_s, t0s_s, mdl = fit_transit_shape(
                time, flux, ferr, P_orb, T0, dur_guess=dur,
                depth_guess=depth_tls)
            shape_model = (lambda ph, _m=mdl, _s=t0s_s:
                           _m(np.asarray(ph, float) + _s))
            if 0.3 * dur < t14_s < 3.0 * dur:
                dur_pt = float(t14_s)
            _log(log, f"\n  shape for the per-transit fits (trapezoid on the "
                      f"fold): T14 = {t14_s * 24:.2f} h, depth = "
                      f"{depth_s * 1e6:.0f} ppm, T0 shift = "
                      f"{t0s_s * 1440:+.1f} min")
        except Exception as exc:
            _log(log, f"\n  ! shape fit failed ({exc}); using a soft box")

    def _run_pt(P_, T0_):
        return per_transit_depths(time, flux, ferr, results, cfg, P=P_, T0=T0_,
                                  dur=dur_pt, shape_model=shape_model,
                                  return_rejected=True)

    dt, rejected = _run_pt(P_orb, T0)
    diag = depth_diagnostics(dt)

    _log(log, f"\n--- Per-transit depths ({dt.size} usable transits) ---")
    _R_pt = cfg.per_transit_tc_range * dur_pt if cfg.per_transit_fit_tc else 0.0
    _n_lim = 0
    for r in dt:
        _lim = _R_pt > 0 and abs(r['dTc']) >= 0.97 * _R_pt
        _n_lim += int(_lim)
        _log(log, f"  epoch {r['epoch']:4d}  Tc = {r['Tc']:.5f}  "
                  f"dTc = {r['dTc'] * 1440:+6.1f} min  "
                  f"depth = {r['depth'] * 1e6:8.0f} +/- "
                  f"{r['e_depth'] * 1e6:6.0f} ppm  "
                  f"snr = {r['snr']:5.1f}  ({r['npts']} pts)"
                  + ("  [dTc at search limit]" if _lim else ""))
    if _n_lim:
        _log(log, f"  ! {_n_lim} transit(s) hit the mid-time search limit "
                  f"(+-{_R_pt * 1440:.0f} min): the ephemeris is probably "
                  f"wrong; these depths are not reliable")
    if rejected:
        _log(log, f"  epochs not usable ({len(rejected)}):")
        for e_, tc_, why in rejected:
            _log(log, f"    epoch {e_:4d}  Tc = {tc_:.3f}  {why}")

    # refine P and T0 from the measured mid-times, then redo the fits
    if cfg.refine_from_transits and dt.size:
        ep_with_data = list(dt["epoch"]) + [e_ for e_, _, why in rejected
                                             if why != "no data"]
        ref = refine_ephemeris_from_transits(dt, P_orb, T0, dur_pt, cfg,
                                             epochs_data=ep_with_data)
        if ref is None or "P" not in ref:
            _log(log, f"  ephemeris refinement skipped: "
                      f"{'' if ref is None else ref['reason']}")
        else:
            _log(log, f"  linear ephemeris from {ref['n']} transits: "
                      f"P = {ref['P']:.5f} +/- {ref['eP']:.5f} d, "
                      f"T0 = {ref['T0']:.5f} +/- {ref['eT0']:.5f}; "
                      f"O-C rms = {ref['oc_rms_min']:.1f} min, "
                      f"chi2_red = {ref['chi2_red']:.2f}")
            if ref["updated"]:
                _log(log, f"  adopted refined ephemeris: P {P_orb:.5f} -> "
                          f"{ref['P']:.5f}, T0 {T0:.5f} -> {ref['T0']:.5f}")
                P_orb, T0 = ref["P"], ref["T0"]
                dt, rejected = _run_pt(P_orb, T0)
                diag = depth_diagnostics(dt)
                _log(log, f"  after refinement: {dt.size} usable transits, "
                          f"mean |dTc| = "
                          f"{np.mean(np.abs(dt['dTc'])) * 1440 if dt.size else float('nan'):.1f} min")
            else:
                _log(log, f"  ephemeris NOT updated: {ref['reason']}")

    if "depth" in diag:
        _log(log, f"  weighted mean depth = {diag['depth'] * 1e6:.0f} "
                  f"+/- {diag['depth_err'] * 1e6:.0f} ppm")
    if "chi2_red" in diag:
        _log(log, f"  constant-depth chi2/dof = {diag['chi2_red']:.2f} "
                  f"({diag['dof']} dof)")
    if "odd_even_sigma" in diag:
        _log(log, f"  odd/even difference = {diag['odd_even_sigma']:.2f} sigma"
                  + (f"  (TLS reports {results.odd_even_mismatch:.2f})"
                     if results is not None else ""))

    # P/2 alias test: depth at phase 0.5 (fixed mid-time, no shift search)
    try:
        dh, _ = per_transit_depths(
            time, flux, ferr, results, cfg, P=P_orb, T0=T0 + 0.5 * P_orb,
            dur=dur_pt, shape_model=shape_model, return_rejected=True,
            fit_tc=False)
        dd = depth_diagnostics(dh)
        if "depth" in dd and dd["depth_err"] > 0:
            _log(log, f"  depth at phase 0.5 (P/2 and secondary-eclipse "
                      f"test): {dd['depth'] * 1e6:.0f} +/- "
                      f"{dd['depth_err'] * 1e6:.0f} ppm from {dh.size} epochs "
                      f"({dd['depth'] / dd['depth_err']:.1f} sigma)")
            if "depth" in diag and diag.get("depth_err", 0) > 0:
                dz = (dd["depth"] - diag["depth"]) / np.hypot(
                    dd["depth_err"], diag["depth_err"])
                _log(log, f"    primary {diag['depth'] * 1e6:.0f} +/- "
                          f"{diag['depth_err'] * 1e6:.0f} ppm vs phase 0.5 "
                          f"{dd['depth'] * 1e6:.0f} +/- "
                          f"{dd['depth_err'] * 1e6:.0f} ppm: difference "
                          f"{dz:+.1f} sigma")
            # timing of the phase-0.5 events: exactly P/2 after the primary
            # for a planet at P/2 (or a circular twin EB); an eccentric
            # eclipsing binary has its secondary displaced
            dh2, _ = per_transit_depths(
                time, flux, ferr, results, cfg, P=P_orb, T0=T0 + 0.5 * P_orb,
                dur=dur_pt, shape_model=shape_model, return_rejected=True,
                fit_tc=True)
            okh = (dh2["snr"] >= cfg.per_transit_min_snr) & \
                np.isfinite(dh2["e_dTc"])
            if okh.sum() >= 1:
                w_ = 1.0 / np.maximum(dh2["e_dTc"][okh], 0.01 * dur_pt) ** 2
                off = float(np.sum(w_ * dh2["dTc"][okh]) / np.sum(w_))
                e_off = float(1.0 / np.sqrt(np.sum(w_)))
                _log(log, f"    timing of the phase-0.5 events relative to "
                          f"T0 + P/2: {off * 1440:+.1f} +/- {e_off * 1440:.1f} "
                          f"min from {int(okh.sum())} epochs "
                          f"(= {off / P_orb:+.5f} in phase)")
    except Exception as exc:
        _log(log, f"  phase-0.5 test failed: {exc}")

    plot_per_transit_depths(dt, diag, cfg, tic)
    if dt.size:
        np.savetxt(cfg.path(f"per_transit_depths_{tic}.txt"),
                   np.column_stack([dt["epoch"], dt["Tc"], dt["dTc"],
                                    dt["e_dTc"], dt["depth"], dt["e_depth"],
                                    dt["snr"], dt["npts"]]),
                   header="epoch Tc dTc e_dTc depth e_depth snr npts")

    # Prefer the weighted mean of individual transits when they are
    # mutually consistent.  At long P / low S/N, single-epoch fits often
    # return ~0 with underestimated errors (chi2_red >> 1); using that
    # mean would understate the depth relative to the stacked fold.
    chi2r = diag.get("chi2_red", np.nan)
    if "depth" in diag and np.isfinite(chi2r) and chi2r < 5.0:
        depth = diag["depth"]
        _log(log, f"  adopted depth = weighted per-transit mean "
                  f"({depth * 1e6:.0f} ppm, chi2_red = {chi2r:.2f})")
    else:
        depth = depth_tls
        why = (f"chi2_red = {chi2r:.2f} >> 1"
               if np.isfinite(chi2r) else "no per-transit mean")
        _log(log, f"  adopted depth = fold/TLS/known value "
                  f"({depth * 1e6:.0f} ppm; per-transit mean discarded, "
                  f"{why})")

    # ---- 5c. secondary eclipse at any phase (eccentric orbits) ------------
    if cfg.secondary_search:
        try:
            # first-pass light curve (long biweight window): a secondary that
            # is longer than the primary is not absorbed by the short
            # second-pass window, which was set around the primary only
            t_s, f_s, _ = lc_pass1
            sres = secondary_eclipse_search(
                t_s, f_s, P_orb, T0, max(dur, dur_pt),
                dur_factors=tuple(cfg.secondary_dur_factors),
                min_snr=cfg.secondary_min_snr)
            _log(log, f"\n--- Secondary-eclipse search at all phases "
                      f"(widths {list(cfg.secondary_dur_factors)} x T14 = "
                      f"{max(dur, dur_pt) * 24:.2f} h; first-pass light curve) ---")
            _log(log, f"  phase coverage {sres['coverage'] * 100:.0f} %; "
                      f"S/N over phases: median {sres['snr_median']:.1f}, "
                      f"scatter {sres['snr_mad']:.1f}")
            _shown = sres["candidates"][:5]
            _best = [c for c in sres["candidates"] if c["coherent"]][:1]
            if _best and _best[0] not in _shown:
                _shown = _shown + _best
            for cd in _shown:
                _log(log, f"  candidate phase {cd['phase']:.4f}  width "
                          f"{cd['width_factor']:.2f} T14  depth "
                          f"{cd['depth'] * 1e6:6.0f} ppm  S/N {cd['snr']:5.1f}  "
                          f"({cd['n']} events; without the strongest, epoch "
                          f"{cd['drop']}: S/N {cd['snr_loo']:.1f}; inflated by the "
                          f"event scatter: S/N {cd.get('snr_infl', float('nan')):.1f})  "
                          + ("COHERENT" if cd["coherent"] else
                             "rejected: " + "; ".join(cd.get("reasons", []))))
            if sres["detected"]:
                _log(log, f"  ! SECONDARY ECLIPSE candidate at phase "
                          f"{sres['phase']:.4f} (offset from 0.5: "
                          f"{(sres['phase'] - 0.5) * P_orb * 24:+.1f} h), depth "
                          f"{sres['depth'] * 1e6:.0f} ppm, width "
                          f"{sres['width_factor']:.2f} x T14, S/N {sres['snr']:.1f} "
                          f"from {sres['n_epochs']} events")
                _log(log, f"    e cos(w) ~ {sres['ecosw']:+.3f}; "
                          f"e sin(w) ~ {sres['esinw']:+.2f} (from the width ratio, coarse)")
                d_ref = sres["depth"]
            else:
                _log(log, f"  no coherent secondary eclipse; 3-sigma depth limit "
                          f"(width T14): {sres['lim3_median'] * 1e6:.0f} ppm median over "
                          f"phases, {sres['lim3_at_half'] * 1e6:.0f} ppm at phase 0.5, "
                          f"{sres['lim3_max'] * 1e6:.0f} ppm worst case "
                          f"(note: an eccentric, inclined orbit can avoid a "
                          f"secondary eclipse altogether)")
                d_ref = sres["lim3_median"]
            d_prim = float(depth) if np.isfinite(depth) and depth > 0 else float("nan")
            t_c = _planck_ratio_temperature(d_ref / d_prim, float(star["Teff"]))
            if np.isfinite(t_c):
                _log(log, f"  surface-brightness ratio {d_ref / d_prim:.3g} "
                          f"(secondary / primary depth, 800 nm) -> companion "
                          f"brightness temperature "
                          f"{'~' if sres['detected'] else '<'} {t_c:.0f} K "
                          f"(primary {star['Teff']:.0f} K; assumes equal "
                          f"occulted areas). A self-luminous M dwarf has "
                          f"T ~ 2800-3500 K.")
            _summary(cfg)["secondary"] = dict(
                detected=bool(sres["detected"]),
                phase=float(sres["phase"]) if sres["detected"] else None,
                depth_ppm=float(sres["depth"] * 1e6) if sres["detected"] else None,
                lim3_ppm_median=float(sres["lim3_median"] * 1e6),
                companion_T_K=float(t_c) if np.isfinite(t_c) else None,
                coverage=float(sres["coverage"]))
            plot_secondary_search(sres, t_s, f_s, P_orb, T0, max(dur, dur_pt), cfg, tic)
            np.savetxt(cfg.path(f"secondary_search_{tic}.txt"),
                       np.column_stack([sres["centres"], sres["snr_curve"],
                                        sres["depth_curve"], sres["lim3_curve"]]),
                       header="phase snr_best depth_best lim3_T14")
        except Exception as exc:
            _log(log, f"  secondary-eclipse search failed: {exc}")

    _summary(cfg).update(P_d=float(P_orb), T0_BTJD=float(T0),
                         T14_h=float(dur * 24), depth_ppm_adopted=float(depth * 1e6),
                         n_sectors=int(len(sectors_data)),
                         dilution_corrected_per_sector=bool(cfg._dilution_applied))
    _log(log, f"\nAdopted ephemeris: P = {P_orb:.6f} d, T0 = {T0:.5f}, "
              f"T14 = {dur * 24:.2f} h, depth = {depth * 1e6:.0f} ppm")

    # ---- 5b. save light curves + multi-panel figure ----------------------
    print("\nSaving light curves ...")
    save_sector_lightcurves(sectors_data, cfg, tic, tag="detrended")
    save_stitched_lightcurve(time, flux, ferr, sector_ids, cfg, tic)
    plot_sector_panels(sectors_data, cfg, tic, tag="detrended")

    # ---- 6. centroids -----------------------------------------------------
    cen = centroid_offsets(sectors_data, P_orb, T0, dur)
    if len(cen):
        _log(log, "\n--- In-transit centroid offsets (flux-weighted centroid of "
                  "the aperture; formal errors only - for the location of "
                  "the signal use the difference imaging below) ---")
        _log(log, cen.to_string(index=False))

    # ---- 7. folded light curve -------------------------------------------
    # The adopted T14 may be a catalogue guess shorter than the transit seen
    # in this LC; a short T14 puts ingress/egress into the 'out-of-transit'
    # regions (baseline, clipping, error estimate) and gives a fold window
    # that barely covers the transit.  Use the larger of the two.
    dur_fold = max(dur, dur_pt)
    if dur_fold > dur * 1.05:
        _log(log, f"\n  fold window / OOT definition use T14 = "
                  f"{dur_fold * 24:.2f} h (measured) instead of "
                  f"{dur * 24:.2f} h (adopted)")
    tp, fp, ep, lc = fold_and_bin(time, flux, ferr, P_orb, T0, dur_fold, cfg)
    ld_fit = None
    if cfg.ld_fit:
        try:
            ld_fit = fit_limb_darkened_lc(tp, lc, P_orb, dur_fold, depth, star,
                                          sectors_data, cfg, log=log)
            plot_ld_fit(tp, fp, ld_fit, cfg, tic)
            # Prefer the LD-fit depth (and duration) over depth_known / TLS
            # for calc_depths and TRICERATOPS -- but only when the fold is
            # trustworthy: a T0 shift of a sizeable fraction of T14 means the
            # folded transit is smeared (wrong P/T0), which makes the fit
            # too shallow and too long.
            t0s_ld = abs(float(ld_fit.get("t0_shift_d", 0.0)))
            if t0s_ld > cfg.ld_update_max_t0shift * dur_fold:
                _log(log, f"\n  ! LD fit T0 shift = {t0s_ld * 1440:.0f} min "
                          f"(> {cfg.ld_update_max_t0shift:.2f} x T14): the "
                          f"fold is probably smeared; adopted depth and T14 "
                          f"are NOT updated from the LD fit")
            else:
                d_ld = float(ld_fit["depth_ppm"]) * 1e-6
                if np.isfinite(d_ld) and d_ld > 0:
                    _log(log, f"\n  updating adopted depth from LD fit: "
                              f"{depth * 1e6:.0f} -> {d_ld * 1e6:.0f} ppm")
                    depth = d_ld
                t14_ld = float(ld_fit.get("T14_h", float("nan"))) / 24.0
                if np.isfinite(t14_ld) and t14_ld > 0:
                    _log(log, f"  updating adopted T14 from LD fit: "
                              f"{dur * 24:.2f} -> {t14_ld * 24:.2f} h")
                    dur = t14_ld
            _log(log, f"  adopted for TRICERATOPS: depth = {depth * 1e6:.0f} ppm, "
                      f"T14 = {dur * 24:.2f} h")
            _summary(cfg).update(T14_h=float(dur * 24),
                                 depth_ppm_adopted=float(depth * 1e6))
        except Exception as exc:
            _log(log, f"\n  ! limb-darkened fit failed ({exc}); "
                      f"the trapezoid is used as reference model")
    sigma_tri, sig_info = triceratops_flux_error(
        tp, fp, lc, max(dur, dur_fold), P_orb, cfg,
        ref_model=None if ld_fit is None else ld_fit["model"],
        ref_npar=0 if ld_fit is None else ld_fit["n_free"])
    _log(log, "\n--- Flux uncertainty for TRICERATOPS ---")
    for k, v in sig_info.items():
        _log(log, f"  {k:28s} {v:.4g}" if isinstance(v, float)
             else f"  {k:28s} {v}")
    np.savetxt(cfg.path(f"folded_lightcurve_{tic}.txt"),
               np.column_stack([tp, fp, ep]),
               header="phase_days flux flux_err")

    # ---- 7c. per-transit depths with the LD shape, per sector, timing ----
    dt_final = dt
    if ld_fit is not None:
        try:
            _dtl = per_transit_depths_ld(time, flux, ferr, sectors_data, P_orb,
                                         T0, dur_fold, ld_fit, cfg, log=log)
            if _dtl is not None and _dtl.size >= max(2, dt.size // 2):
                dt_final = _dtl
        except Exception as exc:
            _log(log, f"  ! per-transit LD fits failed: {exc}")
    try:
        report_depths_per_sector(dt_final, sectors_data, cfg, log=log)
    except Exception as exc:
        _log(log, f"  ! per-sector depth report failed: {exc}")
    # ---- 7d. ground-based light curves: fits and mid-times ----------------
    cfg._ground, cfg._ground_times = None, []
    if cfg.ground_specs:
        try:
            ground_photometry_fits(cfg, tic, P_orb, T0, max(dur, dur_fold),
                                   dt_final, ld_fit, shape_model, star, log=log)
        except Exception as exc:
            _log(log, f"  ! ground-based photometry failed: {exc}")
    if cfg.etv_analysis:
        try:
            etv_analysis(dt_final, P_orb, T0, cfg, tic, log=log)
        except Exception as exc:
            _log(log, f"  ! timing analysis failed: {exc}")

    plot_folded_lc(tp, fp, lc, cfg, tic, max(dur, dur_fold),
                   model=None if ld_fit is None else ld_fit["model"],
                   model_label="limb-darkened fit")

    # ---- 8. TRICERATOPS ---------------------------------------------------
    apertures = [sd.ap_pixels for sd in sectors_data]
    cfg._sector_frames = {int(sd.sector): (sd.wcs, int(sd.col0), int(sd.row0))
                          for sd in sectors_data}
    if cfg.dilution_corrected is None:
        if cfg._dilution_applied:
            cfg.dilution_corrected = True
            _log(log, "\n  light curve on the target's scale (per-sector "
                      "dilution correction): dilution_corrected = True for "
                      "calc_depths / calc_probs")
        elif src in ("tesscut", "sap"):
            cfg.dilution_corrected = False
            _log(log, "\n  aperture photometry is not crowding-corrected: "
                      "dilution_corrected = False for calc_depths / calc_probs "
                      "(override with --dilution-corrected)")
    et = (cfg.tri_exptime_d if cfg.tri_exptime_d is not None
          else effective_exposure_d(sectors_data, cfg.bin_minutes))
    cfg._tri_exptime = float(et) if et and et > 0 else None
    if cfg.use_known_ephemeris and cfg.depth_known > 0:
        try:
            _ph = np.asarray(getattr(lc.time, "value", lc.time), float)
            d_fold = 1.0 - float(np.nanmedian(np.sort(
                np.asarray(lc.flux.value, float)[np.abs(_ph) < 0.25 * dur])[:3]))
        except Exception:
            d_fold = float("nan")
        if np.isfinite(d_fold) and d_fold > 0 and \
                abs(depth - d_fold) > 0.3 * max(depth, d_fold):
            _log(log, f"  ! depth passed to TRICERATOPS ({depth * 1e6:.0f} ppm) "
                      f"differs from the depth seen in this light curve "
                      f"(~{d_fold * 1e6:.0f} ppm at the bottom of the binned "
                      f"fold). Give the depth on the scale of this light "
                      f"curve: on-target (like SPOC PDCSAP) with the default "
                      f"per-sector dilution correction, the raw aperture "
                      f"depth with --dilution-mode none.")
    target, all_ap = make_target(tic, sectors, apertures, depth, cfg)
    # cross-check: our PSF-model target fractions against triceratops'
    try:
        _fr = sector_target_fractions(target, all_ap, sigma=cfg.psf_sigma)
        diffs = [abs(_fr[int(sd.sector)] - sd.g_target) for sd in sectors_data
                 if int(sd.sector) in _fr and np.isfinite(sd.g_target)]
        if diffs:
            _log(log, f"  target flux fraction: pipeline vs triceratops field "
                      f"model, largest difference {max(diffs):.3f} over "
                      f"{len(diffs)} sectors")
    except Exception as exc:
        _log(log, f"  target-fraction cross-check failed: {exc}")

    drop_stars = set()

    # ---- 8a. difference imaging ------------------------------------------
    if cfg.difference_imaging:
        print("\nDifference imaging ...")
        di_summary, di_stars = difference_image_analysis(
            sectors_data, target.stars, P_orb, T0, dur, cfg, tic)
        if len(di_summary):
            _log(log, "\n--- Difference-image centroids ---")
            _log(log, di_summary.to_string(index=False))
            di_summary.to_csv(cfg.path(f"difference_imaging_{tic}.csv"),
                              index=False)
            _log(log, f"  (centroid errors include a systematic floor of "
                      f"{cfg.di_sys_floor_px:.2f} px in quadrature)")
            # the differential offset (difference image vs the target's own
            # PRF in the direct image) cancels most of the PRF-model error
            col = ("offset_direct_sigma" if "offset_direct_sigma" in di_summary
                   and di_summary["offset_direct_sigma"].notna().any()
                   else "offset_cat_sigma")
            good = di_summary["snr"] >= cfg.di_min_snr
            on_target = di_summary.loc[good, col].dropna()
            if len(on_target):
                med = float(on_target.median())
                _log(log, f"  difference-image centroid vs target "
                          f"({'direct-image PRF' if col.startswith('offset_direct') else 'catalogue position'}): "
                          f"median {med:.1f} sigma over {len(on_target)} "
                          f"sectors with S/N >= {cfg.di_min_snr:g}; "
                          f"{'consistent with' if med < 3 else 'OFFSET from'} "
                          f"the target")
                _summary(cfg)["di_offset_sigma_median"] = med
        if len(di_stars):
            di_stars.to_csv(cfg.path(f"difference_imaging_stars_{tic}.csv"),
                            index=False)
            di_drop = combine_difference_imaging(di_stars, cfg)
            _log(log, f"  stars excluded by difference imaging: {len(di_drop)}"
                      + (f" (first: {sorted(di_drop)[:8]})" if di_drop else ""))
            drop_stars.update(di_drop)

    # ---- 8b. vetting from the neighbours' own light curves ---------------
    if cfg.vet_neighbours:
        vet = vet_nearby_stars(tic, int(sectors[0]), target.stars,
                               P_orb, T0, dur, depth, cfg)
        if len(vet):
            _log(log, "\n--- Nearby-star vetting from their own light curves ---")
            _log(log, vet.to_string(index=False))
            vet.to_csv(cfg.path(f"neighbour_vetting_{tic}.csv"), index=False)
            drop_stars.update(vet.loc[vet["excluded"], "ID"].astype(int).tolist())

    # ---- 8c. ground-based photometry: star clearance -----------------------
    if getattr(cfg, "_ground", None):
        try:
            drop_stars.update(ground_photometry_field(target, cfg, tic, log=log))
        except Exception as exc:
            _log(log, f"  ! ground-based star clearance failed: {exc}")
    if cfg.exclude_stars:
        _ex = {int(x) for x in cfg.exclude_stars}
        _log(log, f"\n  stars excluded by the user (--exclude-stars): {sorted(_ex)}")
        drop_stars.update(_ex)

    # never drop the target itself
    drop_stars.discard(int(tic))
    # stars with tdepth == 0 cannot host the signal and are ignored by
    # triceratops anyway; removing them would only add noise to the log.
    if "tdepth" in target.stars.columns:
        relevant = set(target.stars.loc[target.stars["tdepth"] > 0, "ID"]
                       .astype(int))
        ignored = sorted(drop_stars - relevant)
        if ignored:
            _log(log, f"  {len(ignored)} excluded star(s) have tdepth = 0 "
                      f"(cannot host the signal); not passed to remove_star")
        drop_stars &= relevant

    # ---- 7b. red-noise scramble p-value ----------------------------------
    if cfg.red_noise_scrambles and int(cfg.red_noise_scrambles) > 0:
        print("\nRed-noise scramble test ...")
        try:
            _dur_rn = max(float(dur), float(locals().get("dur_fold", dur)))
            rn = red_noise_scramble_test(
                time, flux, P_orb, T0, _dur_rn,
                depth_obs=depth,
                n_scrambles=int(cfg.red_noise_scrambles),
                seed=int(cfg.red_noise_seed),
                log=log)
            if rn is not None:
                np.savetxt(
                    cfg.path(f"red_noise_scramble_{tic}.txt"),
                    np.array([[rn["depth_obs"], rn["p_value"],
                               rn["n_valid"], rn["n_ge"],
                               rn["depth_null_ppm"], rn["depth_null_p99_ppm"]]]),
                    header="depth_obs p_value n_valid n_ge null_med_ppm null_p99_ppm")
        except Exception as exc:
            _log(log, f"  red-noise scramble failed: {exc}")

    print("\nRunning TRICERATOPS ...")
    target, FPPs, NFPPs, sigma_used = run_triceratops(
        target, all_ap, tic, depth, P_orb, lc, cfg,
        drop_stars=sorted(drop_stars), log=log, sigma=sigma_tri)

    target.plot_fits(time=lc.time.value, flux_0=lc.flux.value,
                     flux_err_0=sigma_used,
                     save=True, fname=cfg.path(f"plot_fits_{tic}"))
    plt.close("all")
    write_run_summary(cfg, tic, log=log)
    log.close()


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also read by triceratops_gui.py)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tic", type=int, default=None,
                    help="TIC ID of the target (Kepler mode: looked up from "
                         "--kic when not given).")
    ap.add_argument("--mission", type=str, default="tess",
                    choices=("tess", "kepler"),
                    help="tess (default) or kepler: Kepler long-cadence "
                         "light curves and pixel files per quarter; times in "
                         "BKJD (BJD - 2454833); needs --kic.")
    ap.add_argument("--kic", type=int, default=None,
                    help="Kepler Input Catalog number (with --mission kepler).")
    ap.add_argument("--host-tic", type=int, default=None, metavar="TIC",
                    help="Star that TRICERATOPS treats as the host (e.g. a "
                         "close companion); the light curve and pixels stay "
                         "those of --tic / --kic.  The light curve is then "
                         "used on the aperture scale (SAP, no dilution "
                         "correction).")
    ap.add_argument("--star-source", type=str, default="tic",
                    choices=("tic", "kepler"),
                    help="Host-star R, M, Teff, logg: tic (default) or kepler "
                         "(Berger et al. 2020 Gaia-Kepler catalogue, else the "
                         "Kepler DR25 stellar table; needs --kic).")
    ap.add_argument("--stellar", type=float, nargs="+", default=None,
                    metavar="X",
                    help="Manual host-star parameters R[Rsun] M[Msun] Teff[K] "
                         "[logg]; override --star-source.")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory (default: TIC{tic}/ in the CWD). "
                         "Created automatically if missing.")
    ap.add_argument("--n-iter", type=int, default=Config.n_iter,
                    help="Minimum number of TRICERATOPS calc_probs runs "
                         "(more are added until converged).")
    ap.add_argument("--contrast-curve", type=str, default=None,
                    metavar="FILE",
                    help="High-resolution imaging contrast curve: "
                         "separation [arcsec], delta mag.")
    ap.add_argument("--vet-neighbours", action="store_true",
                    help="Vet nearby stars from their own TESS light curves "
                         "and exclude those that cannot host the signal.")
    ap.add_argument("--no-difference-imaging", action="store_true",
                    help="Skip the difference-image centroid analysis.")
    ap.add_argument("--exclude-sectors", type=int, nargs="+", default=None,
                    metavar="S",
                    help="Sector numbers to drop before extraction "
                         "(e.g. --exclude-sectors 56 79).")
    ap.add_argument("--no-auto-exclude-sectors", action="store_true",
                    help="Do not auto-drop sectors that fail normalisation "
                         "quality cuts after detrending.")
    ap.add_argument("--sector-max-rms", type=float, default=None,
                    help="Max robust RMS of norm_flux (fractional; default 0.02).")
    ap.add_argument("--red-noise-scrambles", type=int, default=0,
                    metavar="N",
                    help="Phase-randomisation trials for a red-noise "
                         "false-alarm p-value on the stacked depth "
                         "(0 disables; try 1000–5000).")
    ap.add_argument("--red-noise-seed", type=int, default=42,
                    help="RNG seed for --red-noise-scrambles.")
    ap.add_argument("--download-dir", type=str, default=None,
                    help="TESScut / SPOC cache directory (default: <outdir>/tesscut_cache). "
                         "Already-downloaded products are reused.")
    ap.add_argument("--lc-source", type=str, default="tesscut",
                    choices=("tesscut", "sap", "pdcsap"),
                    help="Photometry: tesscut (FFI custom aperture, default), "
                         "sap (SPOC SAP), or pdcsap (SPOC PDC). For sap/pdcsap "
                         "the SPOC TPF is still fetched when available.")
    ap.add_argument("--trilegal-fname", type=str, default=None,
                    help="Pre-downloaded TRILEGAL table for triceratops "
                         "(skips the SSL-sensitive online query).")
    ap.add_argument("--err-floor-ppm", type=float, default=Config.err_floor_ppm,
                    help="Lower bound on the flux uncertainty given to "
                         "calc_probs [ppm].")
    ap.add_argument("--no-inflate", action="store_true",
                    help="Do not inflate sigma to chi2_red = 1 of the "
                         "reference model.")
    ap.add_argument("--sensitivity", action="store_true",
                    help="Also report FPP/NFPP at several flux-uncertainty "
                         "scales (extra calc_probs runs).")
    ap.add_argument("--raw-aperture-flux", action="store_true",
                    help="Pass dilution_corrected=False to triceratops "
                         "(flux is an uncorrected aperture LC). Needs "
                         "triceratops >= 1.1.")
    ap.add_argument("--ld-fit", action="store_true",
                    help="Fit the folded LC with a limb-darkened transit "
                         "model (u1,u2 fixed to the Claret values of the "
                         "TIC/TLS catalogue) and use it as reference model "
                         "for the sigma inflation.")
    ap.add_argument("--ld-free", action="store_true",
                    help="Like --ld-fit, but fit the limb-darkening "
                         "coefficients (Kipping q1,q2).")
    ap.add_argument("--ld-coeffs", type=float, nargs=2, metavar=("U1", "U2"),
                    default=None,
                    help="Like --ld-fit, with fixed quadratic coefficients.")
    ap.add_argument("--n-samples", type=int, default=None, metavar="N",
                    help="Monte-Carlo draws per scenario in calc_probs "
                         "(default: triceratops default, 1e6). Raise when the "
                         "log reports that the evidence is dominated by one run.")
    ap.add_argument("--no-pool", action="store_true",
                    help="Do not pool the scenario evidences over the runs; "
                         "report the mean of the per-run FPPs instead.")
    ap.add_argument("--sens-runs", type=int, default=Config.tri_sens_runs,
                    help="calc_probs runs per scale for --sensitivity (pooled).")
    ap.add_argument("--no-refine-ephemeris", action="store_true",
                    help="Do not refine P and T0 from the individual transit "
                         "mid-times.")
    ap.add_argument("--refine-chi2-max", type=float, default=None,
                    metavar="X",
                    help="Max chi2_red of the linear timing fit before the "
                         "refined ephemeris is rejected (default: 8).  Raise "
                         "for long-period targets with few transits.")
    ap.add_argument("--ephem-adopt-max-sigma", type=float, default=None,
                    metavar="F",
                    help="A refined ephemeris is adopted only if it predicts "
                         "every epoch with data to within F x T14 (1 sigma; "
                         "default 0.25).")
    ap.add_argument("--no-tc-fit", action="store_true",
                    help="Keep the mid-time of each individual transit fixed "
                         "at the ephemeris value.")
    ap.add_argument("--ephemeris", type=float, nargs=4,
                    metavar=("P_D", "T0_BTJD", "DEPTH_PPM", "T14_H"),
                    default=None,
                    help="Known ephemeris for THIS target (skips the TLS "
                         "search): period [d], T0 [BTJD; BKJD with --mission "
                         "kepler; or the full BJD_TDB], depth [ppm], "
                         "duration [h].")
    ap.add_argument("--blind-search", action="store_true",
                    help="Force a blind TLS search (default when --ephemeris "
                         "is not given).")
    ap.add_argument("--p-min", type=float, default=None, metavar="D",
                    help="TLS period_min [d] (default: 0.5).  Narrow the grid "
                         "when you know the approximate period.")
    ap.add_argument("--p-max", type=float, default=None, metavar="D",
                    help="TLS period_max [d] (default: 100, also clipped to "
                         "baseline/n_transits_min at runtime).")
    ap.add_argument("--detrend-windows", type=float, nargs="+", default=None,
                    metavar="D",
                    help="First-pass biweight windows [d]; the blind TLS search "
                         "runs once per window and the highest-SDE candidate is "
                         "kept (e.g. 0.75 1.0 1.5 2.5).")
    ap.add_argument("--ephem-search", type=str, default="auto",
                    choices=("auto", "scan", "grid", "off"),
                    help="With --ephemeris: how to check/refine P and T0 "
                         "before detrending. scan = box search of each "
                         "transit + robust linear fit; grid = folded box "
                         "search on a (P, T0) grid (stacks all transits, for "
                         "shallow signals); auto = scan, grid if the scan is "
                         "inconclusive (default); off = keep the input.")
    ap.add_argument("--no-ephem-scan", action="store_true",
                    help="Same as --ephem-search off.")
    ap.add_argument("--ephem-scan-hours", type=float, default=None,
                    metavar="H",
                    help="Half-width of the search in T0 [h], for scan and "
                         "grid (default: max(2 x T14, 12 h)).")
    ap.add_argument("--ephem-grid-dp", type=float, default=None, metavar="D",
                    help="Grid search: half-range in P [d] (default: the "
                         "T0 half-range divided by the number of epochs "
                         "from the reference epoch).")
    ap.add_argument("--ephem-grid-min-snr", type=float, default=None,
                    metavar="X",
                    help="Grid search: minimum stacked S/N to adopt the "
                         "result (default 7).")
    ap.add_argument("--aperture", type=str, default="optimal",
                    choices=("optimal", "percentile"),
                    help="TESScut aperture: S/N-optimal from a catalogue PSF "
                         "model (default) or pixels above the 85th percentile.")
    ap.add_argument("--dilution-mode", type=str, default="sector",
                    choices=("sector", "none"),
                    help="sector (default): put every sector on the target's "
                         "flux scale with its own target fraction before "
                         "folding; none: keep raw aperture flux.")
    ap.add_argument("--tri-exptime-min", type=float, default=None, metavar="M",
                    help="Exposure smoothing of the TRICERATOPS models [min] "
                         "(default: bin width + native cadences; 0 = package "
                         "default of 2 min).")
    ap.add_argument("--tri-max-iter", type=int, default=None, metavar="N",
                    help="Maximum calc_probs runs while not converged "
                         "(default 3 x --n-iter).")
    ap.add_argument("--tri-time-budget", type=float, default=None, metavar="S",
                    help="Time budget for the extra runs [s] (default 3600).")
    ap.add_argument("--no-etv", action="store_true",
                    help="Skip the eclipse-timing / depth-variation analysis.")
    ap.add_argument("--extra-times", type=str, default=None, metavar="FILE",
                    help="Ground-based mid-times to add to the timing "
                         "analysis: 'BJD_TDB err_min [label]' per line.")
    ap.add_argument("--no-secondary-search", action="store_true",
                    help="Skip the secondary-eclipse search at all phases.")
    ap.add_argument("--secondary-min-snr", type=float, default=None,
                    metavar="X",
                    help="Stacked S/N for a secondary-eclipse detection "
                         "(default 7).")
    ap.add_argument("--ground-lc", action="append", default=None,
                    metavar="SPEC",
                    help="Ground-based light curve (repeatable): "
                         "'path,band=r,star=TIC,ap=4.0,fwhm=2.5,obs=CODE,"
                         "detrend=poly1,AIRMASS'. band: B V R I g r i z J H K "
                         "(Rc=R, Sloan r'=r); star: TIC at the aperture "
                         "centre (default: target); ap: aperture RADIUS "
                         "[arcsec]; fwhm: seeing [arcsec]; time=BJD_TDB "
                         "(default) | JD_UTC | HJD_UTC; fcol/ecol/tcol: "
                         "column names; mag=1 for magnitudes.")
    ap.add_argument("--ground-list", type=str, default=None, metavar="FILE",
                    help="File with one --ground-lc specification per line "
                         "(paths relative to that file).")
    ap.add_argument("--ground-no-clear", action="store_true",
                    help="Report the ground clearance but do not remove "
                         "stars from TRICERATOPS.")
    ap.add_argument("--ground-tc-window", type=float, default=None,
                    metavar="MIN", help="Minimum half-width of the ground "
                         "mid-time search [min] (default 60; widened to 3 "
                         "sigma of the TESS ephemeris).")
    ap.add_argument("--ground-min-snr", type=float, default=None,
                    help="Detection threshold for ground light curves "
                         "(default 5).")
    ap.add_argument("--ground-depth-tol", type=float, default=None,
                    help="Fractional tolerance on the expected ground depth "
                         "(default 0.25).")
    ap.add_argument("--ground-fwhm", type=float, default=None, metavar="ARCSEC",
                    help="Seeing FWHM used when a light curve gives none "
                         "(default 3.0).")
    ap.add_argument("--tri-plus", type=str, default="auto",
                    choices=("auto", "on", "off"),
                    help="Pass ground light curves to calc_probs when the "
                         "installed triceratops is TRICERATOPS+ (auto, "
                         "default), require it (on) or never (off).")
    ap.add_argument("--exclude-stars", type=int, nargs="+", default=None,
                    metavar="TIC", help="Stars cleared by other follow-up "
                         "(removed before calc_probs).")
    ap.add_argument("--dilution-corrected", action="store_true",
                    help="Tell TRICERATOPS the light curve is crowding-"
                         "corrected (automatic with --dilution-mode sector "
                         "and for pdcsap).")
    return ap


if __name__ == "__main__":
    ap = build_parser()
    args = ap.parse_args()
    if args.raw_aperture_flux and args.dilution_corrected:
        ap.error("--raw-aperture-flux and --dilution-corrected are mutually exclusive")
    if args.ephemeris is not None and args.blind_search:
        ap.error("--ephemeris and --blind-search are mutually exclusive")
    if args.ld_free and args.ld_coeffs is not None:
        ap.error("--ld-free and --ld-coeffs are mutually exclusive")

    set_mission(args.mission)
    if args.mission == "kepler":
        if args.kic is None:
            ap.error("--mission kepler needs --kic")
        if args.lc_source == "tesscut":
            args.lc_source = "pdcsap"       # Kepler: no FFI cutouts
        if args.tic is None:
            args.tic = resolve_tic_from_kic(args.kic)
            print(f"KIC {args.kic} = TIC {args.tic}")
    elif args.kic is not None:
        ap.error("--kic only with --mission kepler")
    if args.tic is None:
        args.tic = Config.tic
    if args.stellar is not None and len(args.stellar) not in (3, 4):
        ap.error("--stellar needs R M Teff [logg] (3 or 4 numbers)")
    if args.star_source == "kepler" and args.kic is None:
        ap.error("--star-source kepler needs --mission kepler --kic N")
    host_other = args.host_tic is not None and int(args.host_tic) != int(args.tic)
    if host_other:
        # the light curve must be on the aperture's flux scale: PDCSAP is
        # crowding-corrected for the catalogue target, not for the host
        if args.lc_source == "pdcsap":
            print("--host-tic: PDCSAP is corrected for the light of the other "
                  "stars as seen from the catalogue target; SAP is used instead")
            args.lc_source = "sap"
        if args.dilution_corrected:
            ap.error("--host-tic and --dilution-corrected exclude each other")
        args.dilution_mode = "none"
    cfg = Config(tic=args.tic, outdir=args.outdir, n_iter=args.n_iter,
                 mission=args.mission, kic=args.kic,
                 host_tic=args.host_tic, star_source=args.star_source,
                 stellar_manual=(tuple(args.stellar) if args.stellar else None),
                 contrast_curve_file=args.contrast_curve,
                 vet_neighbours=args.vet_neighbours,
                 difference_imaging=not args.no_difference_imaging,
                 exclude_sectors=(tuple(args.exclude_sectors)
                                  if args.exclude_sectors else None),
                 auto_exclude_bad_sectors=not args.no_auto_exclude_sectors,
                 red_noise_scrambles=int(args.red_noise_scrambles),
                 red_noise_seed=int(args.red_noise_seed),
                 download_dir=args.download_dir,
                 lc_source=str(args.lc_source).lower(),
                 trilegal_fname=args.trilegal_fname,
                 err_floor_ppm=args.err_floor_ppm,
                 tri_inflate_to_model=not args.no_inflate,
                 tri_sensitivity=args.sensitivity,
                 dilution_corrected=(False if (args.raw_aperture_flux or host_other)
                                     else None),
                 ld_fit=bool(args.ld_fit or args.ld_free
                             or args.ld_coeffs is not None),
                 ld_mode=("free" if args.ld_free else
                          "fixed" if args.ld_coeffs is not None else "claret"),
                 ld_coeffs=(tuple(args.ld_coeffs) if args.ld_coeffs is not None
                            else None),
                 tri_N=args.n_samples, pool_evidences=not args.no_pool,
                 tri_sens_runs=args.sens_runs,
                 refine_from_transits=not args.no_refine_ephemeris,
                 per_transit_fit_tc=not args.no_tc_fit)
    if args.ephemeris is not None:
        p_, t0_, d_ppm, t14_h = args.ephemeris
        if t0_ > 2.4e6:                      # full BJD_TDB -> mission time
            t0_ = t0_ - TIME_ZERO
        cfg.use_known_ephemeris = True
        cfg.P_orb_known, cfg.T0_known = float(p_), float(t0_)
        cfg.depth_known, cfg.dur_known = float(d_ppm) * 1e-6, float(t14_h) / 24.0
    else:
        # Default path: blind TLS search (Config.use_known_ephemeris is False)
        cfg.use_known_ephemeris = False
    if args.p_min is not None:
        cfg.P_min = float(args.p_min)
    if args.p_max is not None:
        cfg.P_max = float(args.p_max)
    if args.refine_chi2_max is not None:
        cfg.refine_chi2_max = float(args.refine_chi2_max)
    if args.ephem_adopt_max_sigma is not None:
        cfg.ephem_adopt_max_sigma = float(args.ephem_adopt_max_sigma)
    if getattr(args, "sector_max_rms", None) is not None:
        cfg.sector_max_rms = float(args.sector_max_rms)
    if args.detrend_windows:
        cfg.window_length_grid = tuple(args.detrend_windows)
    cfg.ephem_search = "off" if args.no_ephem_scan else args.ephem_search
    if args.ephem_grid_dp is not None:
        cfg.ephem_grid_dP = float(args.ephem_grid_dp)
    if args.ephem_grid_min_snr is not None:
        cfg.ephem_grid_min_snr = float(args.ephem_grid_min_snr)
    if args.ephem_scan_hours is not None:
        cfg.ephem_scan_range_d = float(args.ephem_scan_hours) / 24.0
    if args.dilution_corrected:
        cfg.dilution_corrected = True
    cfg.secondary_search = not args.no_secondary_search
    cfg.aperture_mode = args.aperture
    cfg.dilution_mode = "none" if args.raw_aperture_flux else args.dilution_mode
    if args.tri_exptime_min is not None:
        cfg.tri_exptime_d = float(args.tri_exptime_min) / 1440.0
    if args.tri_max_iter is not None:
        cfg.tri_max_iter = int(args.tri_max_iter)
    if args.tri_time_budget is not None:
        cfg.tri_time_budget = float(args.tri_time_budget)
    cfg.etv_analysis = not args.no_etv
    cfg.extra_times_file = args.extra_times
    if args.secondary_min_snr is not None:
        cfg.secondary_min_snr = float(args.secondary_min_snr)
    specs = []
    for g in (args.ground_lc or []):
        specs.append(parse_ground_spec(g))
    if args.ground_list:
        specs.extend(read_ground_list(args.ground_list))
    for sp_ in specs:
        if not os.path.isfile(sp_.path):
            ap.error(f"ground light curve not found: {sp_.path}")
    cfg.ground_specs = specs or None
    cfg.ground_clear = not args.ground_no_clear
    if args.ground_tc_window is not None:
        cfg.ground_tc_window_min = float(args.ground_tc_window)
    if args.ground_min_snr is not None:
        cfg.ground_min_snr = float(args.ground_min_snr)
    if args.ground_depth_tol is not None:
        cfg.ground_depth_tol = float(args.ground_depth_tol)
    if args.ground_fwhm is not None:
        cfg.ground_fwhm_default = float(args.ground_fwhm)
    cfg.tri_plus = args.tri_plus
    if args.exclude_stars:
        cfg.exclude_stars = tuple(int(x) for x in args.exclude_stars)
    import shlex
    import sys
    cfg._command_line = shlex.join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    main(cfg)

