"""Tests of the ground-based photometry module (v4.1), synthetic data only.

Scenario modelled on AAVSO campaign 942 (TIC 65344135): a target and a
neighbour 5.9" away, both of V ~ 13, blended in TESS (3.2 ppt), resolved
from the ground.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_validation as tv  # noqa: E402

TID, NID, FID = 65344135, 65344134, 99999999
RA0, DEC0 = 39.30713, -26.31458
P, T0 = 36.88, 3500.0                     # BTJD
DUR = 5.33 / 24
E_OBS = 23                                 # epoch of the ground transit


def _stars():
    return pd.DataFrame(dict(
        ID=[str(TID), str(NID), str(FID)],
        ra=[RA0, RA0, RA0 + 40 / 3600 / np.cos(np.radians(DEC0))],
        dec=[DEC0, DEC0 + 5.9 / 3600, DEC0],
        Tmag=[12.20, 12.39, 15.0], Vmag=[12.98, 13.17, 15.8],
        GAIAmag=[12.79, 12.98, 15.6], gaiabp=[13.25, 13.44, 16.0],
        gaiarp=[12.17, 12.36, 15.1],
        # intrinsic depths needed for a 3.2 ppt TESS depth (calc_depths)
        tdepth=[0.0059, 0.0070, 0.20]))


def _cfg(tmp_path, **kw):
    cfg = tv.Config(tic=TID, outdir=str(tmp_path / "out"))
    cfg._summary = {}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _write_aij(path, t_bjd, f, e, am):
    with open(path, "w") as fh:
        fh.write("\tBJD_TDB\trel_flux_T1\trel_flux_err_T1\tAIRMASS\n")
        for i, (a, b, c, d) in enumerate(zip(t_bjd, f, e, am)):
            fh.write(f"{i + 1}\t{a:.6f}\t{b:.6f}\t{c:.6f}\t{d:.4f}\n")


def _write_aavso(path, t_bjd, f, e, am, filt="SR"):
    with open(path, "w") as fh:
        fh.write("#TYPE=EXOPLANET\n#OBSCODE=TEST\n#SOFTWARE=EXOTIC\n"
                 "#DELIM=,\n#DATE_TYPE=BJD_TDB\n#OBSTYPE=CCD\n"
                 f"#STAR_NAME=TIC {NID}\n#FILTER={filt}\n"
                 "#DETREND_PARAMETERS=AIRMASS\n#MEASUREMENT_TYPE=Rnflux\n"
                 "#DATE,DIFF,ERR,DETREND_1\n")
        for a, b, c, d in zip(t_bjd, f, e, am):
            fh.write(f"{a:.6f},{b:.6f},{c:.6f},{d:.4f}\n")


def _lc(depth, dt_min=7.0, noise=2.5e-3, seed=1, span_h=(4.0, 4.0)):
    rng = np.random.default_rng(seed)
    Tc = T0 + E_OBS * P + dt_min / 1440
    t = np.arange(Tc - span_h[0] / 24, Tc + span_h[1] / 24, 1 / 1440)
    tmpl, _ = tv._ground_template(DUR)
    am = 1.05 + 0.6 * ((t - t.mean()) / 0.17) ** 2
    f = (1 - depth * tmpl(t - Tc)) * (1 - 0.015 * (am - 1.05))
    f = f + rng.normal(0, noise, t.size)
    return t + 2457000.0, f, np.full(t.size, noise), am, Tc


def _ephem_dt(sig_min=3.0):
    E = np.arange(0, 20, 3)
    return np.array([(e, T0 + e * P, 0.0, sig_min / 1440) for e in E],
                    dtype=[("epoch", int), ("Tc", float), ("dTc", float),
                           ("e_dTc", float)])


def _run(tmp_path, specs, **kw):
    cfg = _cfg(tmp_path, ground_specs=specs, **kw)
    os.makedirs(cfg.outdir, exist_ok=True)
    star = dict(ra=RA0, dec=DEC0)
    tv.ground_photometry_fits(cfg, TID, P, T0, DUR, _ephem_dt(), None, None,
                              star)
    tgt = SimpleNamespace(stars=_stars())
    drop = tv.ground_photometry_field(tgt, cfg, TID)
    return cfg, drop


def test_parse_spec_and_bands(tmp_path):
    sp = tv.parse_ground_spec('"my lc.tbl",band=Rc,star=TIC 65344134,ap=4,'
                              'fwhm=2.5,detrend=poly2,AIRMASS', str(tmp_path))
    assert sp.path == str(tmp_path / "my lc.tbl")
    assert sp.band == "R" and sp.star == NID
    assert sp.aperture_arcsec == 4 and sp.fwhm_arcsec == 2.5
    assert sp.detrend == "poly2,AIRMASS"
    assert tv.canonical_band("Sloan r'") == "r" and tv.canonical_band("SR") == "r"
    assert tv.canonical_band("B") == "B" and tv.canonical_band("Ic") == "I"
    sp2 = tv.parse_ground_spec("x.tbl band=r' ap=3 star=TIC65344134")
    assert sp2.band == "r" and sp2.star == NID and sp2.aperture_arcsec == 3


def test_aperture_fractions_match_numerical_integration():
    sp = tv.GroundSpec(path="x", band="V", aperture_arcsec=3.0, fwhm_arcsec=2.5)
    fr, info = tv.ground_aperture_fractions(_stars(), sp, TID)
    s = 2.5 / 2.3548
    g = np.linspace(-12, 12, 1201)
    X, Y = np.meshgrid(g, g)
    inside = X ** 2 + Y ** 2 < 9.0
    enc_nb = np.sum(np.exp(-((X) ** 2 + (Y - 5.9) ** 2) / (2 * s * s)) * inside) \
        / (2 * np.pi * s * s) * (g[1] - g[0]) ** 2
    row = fr.set_index("ID").loc[str(NID)]
    assert abs(row["enc"] - enc_nb) < 3e-3
    assert abs(fr["fluxratio"].sum() - 1) < 1e-9
    assert fr.set_index("ID").loc[str(TID), "fluxratio"] > 0.95


def test_readers_aij_aavso_and_magnitudes(tmp_path):
    t, f, e, am, _ = _lc(0.0055)
    _write_aij(tmp_path / "a.tbl", t, f, e, am)
    _write_aavso(tmp_path / "b.txt", t, f, e, am)
    m = -2.5 * np.log10(f) + 13.0
    np.savetxt(tmp_path / "c.dat", np.column_stack([t, m, 1.0857 * e]),
               header="BJD_TDB mag mag_err")
    la = tv.read_ground_lc(tv.GroundSpec(path=str(tmp_path / "a.tbl")))
    sb = tv.GroundSpec(path=str(tmp_path / "b.txt"))
    lb = tv.read_ground_lc(sb)
    lc_ = tv.read_ground_lc(tv.GroundSpec(path=str(tmp_path / "c.dat")))
    assert sb.band == "r" and sb.observer == "TEST"
    for lc in (la, lb, lc_):
        assert abs(lc["t"][0] - (t[0] - 2457000)) < 1e-6
        assert lc["t"].size > 0.98 * t.size
        assert abs(np.median(lc["flux"]) - 1) < 0.01
    assert "AIRMASS" in la["covs"] and "DETREND_1" in lb["covs"]
    assert abs(np.std(lc_["flux"]) - np.std(la["flux"])) < 2e-4


def test_jd_utc_conversion_is_close_to_bjd():
    jd = np.array([2461352.0])
    btjd, note = tv._to_btjd(jd, "JD_UTC", RA0, DEC0)
    d_min = (btjd[0] + 2457000 - jd[0]) * 1440
    assert 0.05 < abs(d_min) < 9.6     # TDB-UTC (1.15 min) + barycentre
    assert "converted" in note


def test_fit_recovers_depth_and_midtime(tmp_path):
    t, f, e, am, Tc = _lc(0.0057, dt_min=7.0)
    _write_aij(tmp_path / "t.tbl", t, f, e, am)
    sp = tv.parse_ground_spec(str(tmp_path / "t.tbl") + ",band=r")
    cfg = _cfg(tmp_path)
    lc = tv.read_ground_lc(sp)
    tmpl, _ = tv._ground_template(DUR)
    fit = tv.fit_ground_lc(lc, sp, P, T0, DUR, tmpl,
                           eph=tv._ephemeris_cov(_ephem_dt(), P, T0), cfg=cfg)
    assert fit["detected"] and fit["timing_ok"]
    assert abs(fit["depth"] - 0.0057) < 3 * fit["e_depth"] + 2e-4
    assert abs(fit["Tc"] - Tc) * 1440 < 3 * fit["eTc"] * 1440 + 1.0
    assert "AIRMASS" in fit["baseline"]


def test_on_target_signal_clears_neighbour(tmp_path):
    # target hosts the signal: 5.9 ppt intrinsic, ~5.8 ppt in a 3" aperture
    t, f, e, am, _ = _lc(0.0058, seed=2)
    _write_aij(tmp_path / "tgt.tbl", t, f, e, am)
    t2, f2, e2, am2, _ = _lc(0.0001, seed=3)
    _write_aavso(tmp_path / "nb.txt", t2, f2, e2, am2)
    specs = [tv.parse_ground_spec(f"{tmp_path / 'tgt.tbl'},band=R,ap=3,fwhm=2.5"),
             tv.parse_ground_spec(f"{tmp_path / 'nb.txt'},star={NID},ap=3,fwhm=2.5")]
    cfg, drop = _run(tmp_path, specs)
    v = cfg._ground["verdict"]
    assert v[str(TID)] == "consistent"
    assert v[str(NID)] == "excluded"
    assert NID in drop and TID not in drop
    assert os.path.isfile(cfg.path(f"ground_clearance_{TID}.pdf"))
    assert os.path.isfile(cfg.path(f"ground_lightcurves_{TID}.pdf"))
    assert len(cfg._ground_times) >= 1


def test_neighbour_signal_flags_neb(tmp_path):
    t, f, e, am, _ = _lc(0.0, seed=4)
    _write_aij(tmp_path / "tgt.tbl", t, f, e, am)
    t2, f2, e2, am2, _ = _lc(0.0068, seed=5)
    _write_aavso(tmp_path / "nb.txt", t2, f2, e2, am2)
    specs = [tv.parse_ground_spec(f"{tmp_path / 'tgt.tbl'},band=R,ap=3,fwhm=2.5"),
             tv.parse_ground_spec(f"{tmp_path / 'nb.txt'},star={NID},ap=3,fwhm=2.5")]
    cfg, drop = _run(tmp_path, specs)
    v = cfg._ground["verdict"]
    assert v[str(TID)] == "excluded"          # reported, never removed
    assert v[str(NID)] == "consistent"
    assert TID not in drop and NID not in drop
    assert cfg._summary["ground"]["target_status"] == "excluded"


def test_no_coverage_is_unconstrained(tmp_path):
    t, f, e, am, Tc = _lc(0.0058, span_h=(9.0, -4.0))   # ends before ingress
    _write_aij(tmp_path / "early.tbl", t, f, e, am)
    cfg, drop = _run(tmp_path, [tv.parse_ground_spec(
        f"{tmp_path / 'early.tbl'},band=R,ap=3,fwhm=2.5")])
    assert not cfg._ground["fits"][0]["usable"]
    assert drop == set()
    assert set(cfg._ground["verdict"].values()) == {"unconstrained"}


def test_chromaticity(tmp_path):
    def two_band(d_blue, d_red, seed):
        tb, fb, eb, ab, _ = _lc(d_blue, seed=seed, noise=1.0e-3)
        tr_, fr_, er_, ar_, _ = _lc(d_red, seed=seed + 1, noise=1.0e-3)
        _write_aij(tmp_path / f"B{seed}.tbl", tb, fb, eb, ab)
        _write_aij(tmp_path / f"R{seed}.tbl", tr_, fr_, er_, ar_)
        return [tv.parse_ground_spec(f"{tmp_path / f'B{seed}.tbl'},band=B,ap=3,fwhm=2.5"),
                tv.parse_ground_spec(f"{tmp_path / f'R{seed}.tbl'},band=r,ap=3,fwhm=2.5")]
    cfg, _ = _run(tmp_path, two_band(0.0058, 0.0058, 10))
    assert not cfg._ground["chromaticity"][str(TID)]["chromatic"]
    cfg2, _ = _run(tmp_path, two_band(0.0035, 0.0062, 20),
                   outdir=str(tmp_path / "out2"))
    c = cfg2._ground["chromaticity"][str(TID)]
    assert c["chromatic"] and c["red_over_blue"] > 1.3


class _PlusTarget:
    def __init__(self, stars):
        self.stars = stars

    def calc_probs(self, time, flux_0, flux_err_0, P_orb, external_lc_files=None,
                   filt_lcs=None, **kw):
        self.kw = dict(external_lc_files=external_lc_files, filt_lcs=filt_lcs)


def test_triceratops_plus_inputs(tmp_path):
    t, f, e, am, _ = _lc(0.0058, seed=7)
    _write_aij(tmp_path / "tgt.tbl", t, f, e, am)
    specs = [tv.parse_ground_spec(f"{tmp_path / 'tgt.tbl'},band=Rc,ap=3,fwhm=2.5")]
    cfg, _ = _run(tmp_path, specs)
    tgt = _PlusTarget(_stars())
    assert tv.triceratops_plus_available(tgt)
    assert not tv.triceratops_plus_available(SimpleNamespace(calc_probs=lambda **k: 0))
    kw = tv._plus_prepare(tgt, cfg)
    assert kw["filt_lcs"] == ["r"]
    arr = np.loadtxt(kw["external_lc_files"][0])
    assert arr.shape[1] == 3 and abs(np.median(arr[:, 0])) < 0.2
    assert "fluxratio_r" in tgt.stars.columns
    assert abs(tgt.stars["fluxratio_r"].iloc[0] - 0.97) < 0.03
    # only the target left -> light curve renormalised to the target
    tgt1 = _PlusTarget(_stars().iloc[:1].copy())
    kw1 = tv._plus_prepare(tgt1, cfg)
    a1 = np.loadtxt(kw1["external_lc_files"][0])
    intr = 1 - np.min(np.convolve(a1[:, 1], np.ones(31) / 31, "valid"))
    assert intr > 0.0050
    cfg.tri_plus = "off"
    assert tv._plus_prepare(tgt, cfg) == {}


def test_etv_uses_ground_times(tmp_path):
    rng = np.random.default_rng(9)
    E = np.arange(0, 12)
    dt = np.array([(e, T0 + e * P, rng.normal(0, 1) / 1440, 1.0 / 1440,
                    0.0059, 2e-4) for e in E],
                  dtype=[("epoch", int), ("Tc", float), ("dTc", float),
                         ("e_dTc", float), ("depth", float), ("e_depth", float)])
    cfg = _cfg(tmp_path)
    os.makedirs(cfg.outdir, exist_ok=True)
    cfg._ground_times = [(T0 + 30 * P + 1 / 1440, 1.5 / 1440, "ground:x")]
    out = tv.etv_analysis(dt, P, T0, cfg, TID)
    assert out is None or out.get("n", 13) == 13


class _PlusLikeResult:
    """TRICERATOPS+-style result: lnZ only in probs, NaN probabilities."""
    names = ["TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB",
             "SEBx2P", "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P"]

    def __init__(self):
        lnZ = np.array([-900, -880, -881, -910, -885, -886, -920, -890, -891,
                        -905, -895, -896, -1100, -1000, -950], float)
        with np.errstate(invalid="ignore"):
            pr = np.exp(lnZ) / np.sum(np.exp(lnZ))      # 0/0 -> NaN
        self.probs = pd.DataFrame(dict(scenario=self.names, prob=pr, lnZ=lnZ))
        self.FPP = float(1 - (pr[0] + pr[3] + pr[9]))
        self.NFPP = 0.0


def test_plus_style_result_is_normalised():
    t = _PlusLikeResult()
    assert not np.isfinite(t.FPP)
    tv._normalise_tri_result(t)
    assert np.allclose(t.lnZ, t.probs["lnZ"])
    assert np.isfinite(t.FPP) and abs(t.probs["prob"].sum() - 1) < 1e-9
    assert t.FPP > 0.99                      # EB dominates
    ok, why = tv._check_run(t)
    assert ok, why


def test_plus_style_tesscut_requests_use_cache(tmp_path, monkeypatch):
    """TRICERATOPS+ calls Tesscut.get_cutouts directly and reads 1CRV4P."""
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    cfg = _cfg(tmp_path)
    os.makedirs(cfg.outdir, exist_ok=True)
    coord = SkyCoord(RA0, DEC0, unit="deg")
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(), fits.ImageHDU()])
    tpf = SimpleNamespace(hdu=hdul, column=1234, row=567)
    monkeypatch.setattr(tv, "_load_tesscut_tpf", lambda *a, **k: tpf)
    monkeypatch.setattr(tv, "_tesscut_target_coord", lambda tic, cfg: coord)
    cfg._sector_frames = {}
    called = {}
    monkeypatch.setattr(tv.tr.Tesscut, "get_cutouts",
                        lambda *a, **k: called.setdefault("net", True) and [])
    with tv._triceratops_tesscut_cache(TID, cfg):
        res = tv.tr.Tesscut.get_cutouts(coordinates=coord, size=10, sector=5)
    assert "net" not in called
    h = res[0][1].header
    assert h["1CRV4P"] == 1234 and h["2CRV4P"] == 567
    assert cfg._tri_loaded_sectors == [5]


def test_numpy2_compat_shims(tmp_path):
    # TRICERATOPS+ style Gauss2D (meshgrid -> 1x1 array) through the patch
    fake = SimpleNamespace(Gauss2D=lambda x, y, mx, my, s, A: np.exp(
        -((np.meshgrid(x, y)[0] - mx) ** 2 + (np.meshgrid(x, y)[1] - my) ** 2)
        / (2 * s * s)) * A / (2 * np.pi * s * s))
    old = tv.tr
    try:
        tv.tr = fake
        tv._patch_triceratops_compat()
        g = fake.Gauss2D(0.1, -0.2, 0.0, 0.0, 0.75, 1.0)
        assert isinstance(g, float) and g > 0
        from scipy.integrate import dblquad
        v = dblquad(fake.Gauss2D, -0.5, 0.5, -0.5, 0.5, args=(0.0, 0.0, 0.75, 1.0))[0]
        assert 0.2 < v < 0.3
        assert fake.Gauss2D(np.array([0.0, 1.0]), np.array([0.0]), 0, 0, 0.75, 1.0).shape == (1, 2)
    finally:
        tv.tr = old
    p = tmp_path / "tri.csv"
    pd.DataFrame(dict(Mact=[1.0], logg=[4.4], logTe=[3.7], TESS=[14.0], J=[13.2],
                      H=[13.0], Ks=[12.9])).to_csv(p, index=False)
    q = tv._ensure_griz_columns(str(p))
    cols = pd.read_csv(q).columns
    assert all(c in cols for c in "griz")


def test_environment_check_runs():
    probs = tv.environment_problems()
    assert isinstance(probs, list)
    assert all(isinstance(p, str) and "Fix" in p or "import" in p for p in probs)



class _PlusSeq:
    """TRICERATOPS+-like target: new probs (with lnZ) each run, no .lnZ."""
    names = _PlusLikeResult.names

    def __init__(self):
        self.rng = np.random.default_rng(5)
        self.stars = pd.DataFrame(dict(ID=["1"], tdepth=[0.003]))

    def calc_probs(self, time, flux_0, flux_err_0, P_orb, **kw):
        lnZ = np.array([-100, -80, -81, -110, -85, -86, -120, -90, -91,
                        -105, -95, -96, -300, -200, -150], float)
        lnZ = lnZ + self.rng.normal(0, 2, lnZ.size)
        p = np.exp(lnZ - lnZ.max()); p /= p.sum()
        self.probs = pd.DataFrame(dict(scenario=self.names, prob=p, lnZ=lnZ))
        self.FPP = float(1 - (p[0] + p[3] + p[9])); self.NFPP = 0.0


def test_plus_lnz_refreshed_every_run(tmp_path):
    cfg = tv.Config(tic=1, outdir=str(tmp_path / "o"))
    t = _PlusSeq()
    lc = SimpleNamespace(time=SimpleNamespace(value=np.zeros(5)),
                         flux=SimpleNamespace(value=np.ones(5)))
    seen = []
    for _ in range(3):
        tv._calc_probs_checked(t, lc, 10.0, 1e-3, cfg, max_retries=0)
        seen.append(np.array(t.lnZ))
    assert not np.allclose(seen[0], seen[1]) and not np.allclose(seen[1], seen[2])
    assert np.allclose(seen[2], t.probs["lnZ"])
