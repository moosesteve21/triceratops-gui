"""Regression tests for triceratops_validation.py (synthetic data, offline).

Run from the folder that holds the script:

    python -m pytest -q tests

Every test builds its own synthetic data; no MAST, Gaia or TRILEGAL access
is needed.  The tests cover the failure modes found while validating
TIC 229788204 (October 2026): clipped eclipse bottoms, a drifting ephemeris,
the TRICERATOPS pixel frame, per-sector dilution, secondary eclipses carried
by one event, and the timing analysis.
"""
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_validation as tv  # noqa: E402

RNG = np.random.default_rng(20261029)
P_TRUE, T0_TRUE, DUR = 93.83471, 1733.85618, 4.46 / 24.0


# ----------------------------------------------------------------- helpers
def _cfg(tmp_path, **kw):
    cfg = tv.Config(tic=1, outdir=str(tmp_path / "out"))
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _times(cad_min=10.0, sectors=((1683, 1710), (1711, 1737), (1738, 1764),
                                  (1765, 1791), (1792, 1818), (1819, 1845),
                                  (2750, 2776), (2777, 2803), (2804, 2830),
                                  (2831, 2857), (2858, 2884))):
    """Two blocks of consecutive 27-d sectors (every phase seen >= twice)."""
    return np.concatenate([np.arange(a, b, cad_min / 1440.0) for a, b in sectors])


def _vshape(t, P, T0, dur, depth):
    """Symmetric V-shaped (grazing) eclipse of the given depth."""
    ph = (t - T0 + 0.5 * P) % P - 0.5 * P
    return 1.0 - depth * np.clip(1.0 - np.abs(ph) / (0.5 * dur), 0.0, None)


def _box(t, P, T0, dur, depth, phase=0.0):
    ph = (t - T0 - phase * P + 0.5 * P) % P - 0.5 * P
    return 1.0 - depth * (np.abs(ph) < 0.5 * dur)


# ------------------------------------------------------------- clipping
def test_clip_keeps_bottom_of_deep_eclipse():
    t = _times()
    f = _vshape(t, P_TRUE, T0_TRUE, DUR, 0.07) + RNG.normal(0, 0.002, t.size)
    e = np.full(t.size, 0.002)
    core = np.abs((t - T0_TRUE + 0.5 * P_TRUE) % P_TRUE - 0.5 * P_TRUE) < 0.02
    a = tv.clip_outliers(t, f, e, sigma=5.0)                    # no ephemeris
    b = tv.clip_outliers(t, f, e, sigma=5.0, P=P_TRUE, T0=T0_TRUE, dur=DUR)
    for tt in (a[0], b[0]):
        keep = np.abs((tt - T0_TRUE + 0.5 * P_TRUE) % P_TRUE - 0.5 * P_TRUE) < 0.02
        assert keep.sum() == core.sum()


def test_clip_removes_upward_spikes():
    t = _times()
    f = 1.0 + RNG.normal(0, 0.002, t.size)
    f[::500] += 0.05
    out = tv.clip_outliers(t, f, np.full(t.size, 0.002), sigma=5.0)
    assert np.nanmax(out[1]) < 1.02


# ------------------------------------------------------------- ephemeris
def test_scan_recovers_drifting_ephemeris(tmp_path):
    t = _times()
    f = _vshape(t, P_TRUE, T0_TRUE, DUR, 0.04) + RNG.normal(0, 0.002, t.size)
    cfg = _cfg(tmp_path)
    P1, T01, fit = tv.check_known_ephemeris(t, f, np.full(t.size, 0.002),
                                            93.846904, T0_TRUE, DUR, cfg)
    assert fit is not None and fit["ok"]
    assert abs(P1 - P_TRUE) < 2e-3


def test_scan_does_not_adopt_fit_to_clustered_transits(tmp_path):
    # Early sectors too noisy for single transits, the
    # last two sectors clean.  A fit to the late transits only must not be
    # extrapolated over the whole baseline.
    P, T0, dur, dep = 7.4421, 1570.30, 1.4 / 24, 1500e-6
    rng = np.random.default_rng(7)
    early = _times(10.0, ((1569, 1595), (2283, 2309), (3010, 3036)))
    late = _times(200 / 60, ((4080, 4106), (4107, 4133)))
    t = np.concatenate([early, late])
    sig = np.where(t < 4000, 0.010, 0.0006)
    f = _box(t, P, T0, dur, dep) + rng.normal(0, 1, t.size) * sig
    cfg = _cfg(tmp_path)
    P1, T01, fit = tv.check_known_ephemeris(t, f, sig, P, T0, dur, cfg)
    assert fit is not None and max(fit["epochs"]) - min(fit["epochs"]) < 10
    assert (P1, T01) == (P, T0)
    assert fit["ok"] is False          # 'auto' then falls back to the grid


def test_refine_from_transits_needs_full_baseline(tmp_path):
    cfg = _cfg(tmp_path)
    dur = 1.3 / 24

    def dt_for(epochs, slope_min):
        return np.array([(int(e), 1570.3 + e * 7.4421, 1.5e-3, 2e-4, 100, 1.0,
                          (e - epochs[0]) * slope_min / 1440 + 0.5 / 1440 * (-1) ** i,
                          3.0 / 1440, 7.0, 50.0)
                         for i, e in enumerate(epochs)], dtype=tv._PT_DTYPE)

    clustered = dt_for([312, 313, 315], 2.0)
    r = tv.refine_ephemeris_from_transits(clustered, 7.4421, 1570.3, dur, cfg,
                                          epochs_data=[0, 1, 90, 180, 312, 313, 315])
    assert r["updated"] is False and r["worst_epoch"] == 0
    spread = dt_for([0, 1, 90, 180, 312, 313, 315], 0.01)
    r = tv.refine_ephemeris_from_transits(spread, 7.4421, 1570.3, dur, cfg)
    assert r["updated"] is True


def test_grid_finds_shallow_transit(tmp_path):
    t = _times(cad_min=3.33)
    P, T0, dep = 50.123, 1700.40, 1500e-6
    f = _box(t, P, T0, DUR, dep) + RNG.normal(0, 0.002, t.size)
    res = tv.fold_grid_search(t, f, P + 0.002, T0 + 0.05, DUR, 0.5)
    assert res["snr"] > 7
    assert abs(res["P"] - P) < 1e-3


# ---------------------------------------------------------- dilution / PSF
def _wcs(crval=(283.8916, 67.9079)):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [5.5, 5.5]
    w.wcs.cdelt = [-21.0 / 3600.0, 21.0 / 3600.0]
    w.wcs.crval = list(crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _field():
    # target at the centre, a 1-mag fainter neighbour 1.6 px to the east
    ra0, de0 = 283.8916, 67.9079
    dra = 1.6 * 21.0 / 3600.0 / np.cos(np.radians(de0))
    return pd.DataFrame(dict(ID=[1, 2, 3], ra=[ra0, ra0 + dra, ra0 - 0.02],
                             dec=[de0, de0, de0 + 0.01], Tmag=[12.1, 13.1, 15.0]))


def test_optimal_aperture_is_less_diluted(tmp_path):
    cfg = _cfg(tmp_path)
    w, fdf = _wcs(), _field()
    rows, cols = np.mgrid[0:10, 0:10].astype(float)
    xs, ys = tv.star_pixels(fdf, w, 0, 0)
    _, tot = tv.psf_model_images(xs, ys, fdf["Tmag"], cols, rows, 0.75)
    img = 1e5 * tot / tot.max() + 50.0           # plus a flat sky
    mask = tv.optimal_aperture(img, w, 0, 0, fdf, cfg)
    ap_opt = np.column_stack([cols[mask], rows[mask]]).astype(int)
    perc = img > np.percentile(img, 85)
    ap_pct = np.column_stack([cols[perc], rows[perc]]).astype(int)
    g_opt = tv.target_fraction(ap_opt, w, 0, 0, fdf, 0.75)
    g_pct = tv.target_fraction(ap_pct, w, 0, 0, fdf, 0.75)
    assert mask.sum() >= cfg.aperture_min_pix
    assert g_opt > g_pct


def test_dilution_correction_applied_once(tmp_path):
    cfg = _cfg(tmp_path, dilution_mode="sector")
    t = np.arange(0, 10, 0.01)
    f = 1.0 - 0.02 * (np.abs(t - 5) < 0.1)
    sd = tv.SectorData(sector=1, time=t, flux=f * 1000, flux_err=np.full(t.size, 1.0),
                       ap_pixels=np.zeros((1, 2), int), cen_x=t * 0, cen_y=t * 0,
                       g_target=0.4)
    tv.detrend_sector(sd, 1.0, cfg)
    tv.renormalize_sector_oot(sd, cfg)
    d1 = 1 - np.min(sd.norm_flux)
    tv.renormalize_sector_oot(sd, cfg)           # a second call must not re-apply
    d2 = 1 - np.min(sd.norm_flux)
    assert abs(d1 - 0.05) < 0.005
    assert abs(d2 - d1) < 1e-3


def test_fit_source_position_guess_stays_on_target():
    rows, cols = np.mgrid[0:10, 0:10].astype(float)
    img = (3e4 * np.exp(-((cols - 2) ** 2 + (rows - 1.5) ** 2) / 2)   # brighter
           + 1e4 * np.exp(-((cols - 5.2) ** 2 + (rows - 4.8) ** 2) / 2))
    free = tv.fit_source_position(img, None, cfg=tv.Config.__new__(tv.Config))
    held = tv.fit_source_position(img, None, cfg=tv.Config.__new__(tv.Config),
                                  guess=(5.0, 5.0), box=1.0)
    assert abs(free["x"] - 2.0) < 0.3                  # jumps to the bright star
    assert abs(held["x"] - 5.2) < 0.3 and abs(held["y"] - 4.8) < 0.3


# --------------------------------------------------------- secondary
def test_secondary_detected_and_single_event_rejected():
    t = _times(cad_min=10.0)
    f = 1.0 + RNG.normal(0, 0.002, t.size)
    f2 = f * _box(t, P_TRUE, T0_TRUE, DUR, 3000e-6, phase=0.37)
    r = tv.secondary_eclipse_search(t, f2, P_TRUE, T0_TRUE, DUR)
    assert r["detected"] and abs(r["phase"] - 0.37) < 0.01
    # one 3000 ppm dip in a single orbit only
    f3 = f.copy()
    tc = T0_TRUE + (11 + 0.68) * P_TRUE
    sel = np.abs(t - tc) < DUR
    f3[sel] -= 0.004
    r3 = tv.secondary_eclipse_search(t, f3, P_TRUE, T0_TRUE, DUR)
    near = [c for c in r3["candidates"] if abs(c["phase"] - 0.68) < 0.01]
    assert not r3["detected"] or abs(r3["phase"] - 0.68) > 0.01
    assert all(not c["coherent"] for c in near)


def test_planck_ratio_temperature():
    assert abs(tv._planck_ratio_temperature(1.0, 4000.0) - 4000.0) < 1.0
    assert tv._planck_ratio_temperature(0.01, 4000.0) < 2500.0


# ----------------------------------------------------- per-transit / LD
def test_per_transit_ld_consistent_between_cadences(tmp_path):
    cfg = _cfg(tmp_path)
    P, p, aRs, b = P_TRUE, 0.3, 115.0, 1.05           # grazing, V-shaped
    model = lambda ph: tv.limb_darkened_lc(np.asarray(ph, float), P, p, aRs, b,
                                           0.34, 0.30)
    secs = []
    t_all, f_all = [], []
    for (a, c), cad in (((1720, 1745), 30.0), ((2850, 2875), 200 / 60.0),
                        ((1915, 1940), 30.0), ((3320, 3345), 200 / 60.0)):
        t = np.arange(a, c, cad / 1440.0)
        ph = (t - T0_TRUE + 0.5 * P) % P - 0.5 * P
        f = tv._cadence_smoothed(model, cad / 1440.0)(ph) + RNG.normal(0, 0.001, t.size)
        secs.append(SimpleNamespace(sector=len(secs) + 1, time=t, cadence_d=cad / 1440.0))
        t_all.append(t)
        f_all.append(f)
    t, f = np.concatenate(t_all), np.concatenate(f_all)
    dt = tv.per_transit_depths_ld(t, f, np.full(t.size, 0.001), secs, P,
                                  T0_TRUE, 0.25, {"model_intrinsic": model}, cfg)
    assert dt is not None and dt.size == 4
    # 30-min and 200-s transits give the same (intrinsic) depth
    grid = np.linspace(-0.3, 0.3, 3001)
    d_int = 1.0 - np.min(model(grid))
    long_ = np.isin(dt["epoch"], [0, 2])
    def wm(sel):
        w = 1 / dt["e_depth"][sel] ** 2
        return np.sum(w * dt["depth"][sel]) / np.sum(w), 1 / np.sqrt(np.sum(w))
    m30, e30 = wm(long_)
    m2, e2 = wm(~long_)
    assert abs(m30 - m2) < 3 * np.hypot(e30, e2)
    assert abs(m2 - d_int) < 4 * e2


# ------------------------------------------------------------ timing
def test_etv_recovers_ltte_and_odd_even(tmp_path):
    cfg = _cfg(tmp_path)
    E = np.arange(0, 40)
    t = T0_TRUE + E * P_TRUE + (6.0 / 1440) * np.sin(2 * np.pi * E * P_TRUE / 1200.0)
    dt = np.array([(int(e), T0_TRUE + e * P_TRUE, 0.05, 0.001, 50, 1.0,
                    tt - (T0_TRUE + e * P_TRUE), 0.5 / 1440, 50.0, 100.0)
                   for e, tt in zip(E, t)], dtype=tv._PT_DTYPE)
    out = tv.etv_analysis(dt, P_TRUE, T0_TRUE, cfg, 1)
    assert out["ltte_dchi2"] > 50
    assert abs(out["ltte_best_P"] - 1200) / 1200 < 0.1
    # odd eclipses 3 min late (binary at 2P with e cos w != 0)
    t2 = T0_TRUE + E * P_TRUE + (E % 2) * 3.0 / 1440
    dt2 = dt.copy()
    dt2["dTc"] = t2 - dt2["Tc"]
    out2 = tv.etv_analysis(dt2, P_TRUE, T0_TRUE, cfg, 1)
    assert abs(out2["odd_even_dt_min"] - 3.0) < 0.5


# ------------------------------------------------------------ helpers
def test_exptime_kwargs_and_effective_exposure(tmp_path):
    cfg = _cfg(tmp_path)
    cfg._tri_exptime = 0.01

    def calc_probs(time, flux_0, exptime=0.00139, nsamples=20):
        return None
    kw = tv._exptime_kwargs(calc_probs, cfg)
    assert kw == {"exptime": 0.01, "nsamples": cfg.tri_nsamples}
    secs = [SimpleNamespace(time=np.arange(0, 1, 30 / 1440)),
            SimpleNamespace(time=np.arange(0, 1, 2 / 1440))]
    et = tv.effective_exposure_d(secs, 10.0) * 1440
    assert 10.0 < et < 30.0


def test_scenario_classes_sum_to_one():
    names = ["TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P",
             "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P"]
    p = np.full(len(names), 1.0 / len(names))
    cl = tv.scenario_class_probs(p, names)
    assert abs(sum(v for _, v in cl) - 1.0) < 1e-9


def test_secondary_search_ignores_primary_eclipse():
    t = _times(cad_min=10.0)
    f = _vshape(t, P_TRUE, T0_TRUE, DUR, 0.07) + RNG.normal(0, 0.002, t.size)
    r = tv.secondary_eclipse_search(t, f, P_TRUE, T0_TRUE, DUR)
    assert not r["detected"]


class _MockTarget:
    """calc_probs with heavy-tailed Monte Carlo evidences (non-converged)."""
    names = ["TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB",
             "SEBx2P", "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P"]

    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)
        self.sectors = []
        self.stars = pd.DataFrame(dict(ID=[1], Tmag=[12.0], tdepth=[0.07],
                                       fluxratio=[1.0]))
        self.calls = []

    def calc_probs(self, time, flux_0, flux_err_0, P_orb, exptime=0.00139,
                   nsamples=20, **kw):
        self.calls.append(dict(exptime=exptime, nsamples=nsamples))
        base = np.array([-100, -80, -81, -110, -85, -86, -120, -90, -91,
                         -105, -95, -96, -300, -200, -150], float)
        # an occasional lucky draw dominates the evidence (heavy tail)
        lnZ = base + self.rng.normal(0, 0.3, base.size)
        if self.rng.random() < 0.15:
            lnZ[1] += 6.0
        self.lnZ = lnZ
        p, fpp, nfpp = tv._probs_from_lnZ(lnZ)
        self.FPP, self.NFPP = fpp, nfpp
        self.probs = pd.DataFrame(dict(scenario=self.names, prob=p))


def test_run_triceratops_converges_classes_and_exptime(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, n_iter=5, tri_max_iter=40, tri_time_budget=60,
               tri_sensitivity=False, tri_conv_max_share=0.5)
    cfg._tri_exptime = 14.0 / 1440
    monkeypatch.setattr(tv, "nfpp_input_summary", lambda *a, **k: None)
    tgt = _MockTarget()
    lc = SimpleNamespace(time=SimpleNamespace(value=np.linspace(-0.3, 0.3, 50)),
                         flux=SimpleNamespace(value=np.ones(50)),
                         flux_err=SimpleNamespace(value=np.full(50, 1e-3)))
    _, fpps, nfpps, _ = tv.run_triceratops(tgt, [], 1, 0.07, 93.8, lc, cfg,
                                           sigma=7e-4)
    assert len(tgt.calls) >= 5
    assert all(abs(c["exptime"] - 14.0 / 1440) < 1e-12 for c in tgt.calls)
    cls = cfg._summary.get("scenario_classes")
    assert cls and abs(sum(cls.values()) - 1.0) < 1e-6
    eb = [v for k, v in cls.items() if k.startswith("eclipsing binary")][0]
    assert eb > 0.99
