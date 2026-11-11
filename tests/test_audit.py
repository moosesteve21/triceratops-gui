"""Tests for triceratops_audit.py (prototype) on a synthetic output folder.

    python -m pytest tests/test_audit.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_audit as ta  # noqa: E402

TIC = 999000001
P, T0, T14_H, DEPTH = 12.3456, 2000.0, 4.0, 4000.0


def _make_folder(tmp, depth_ppm=DEPTH, weak_sector=None, log_extra="",
                 summary_extra=None, period=P):
    rng = np.random.default_rng(3)
    dur = T14_H / 24
    for k, s in enumerate((10, 11, 12, 37, 38)):
        t0 = 1990 + 27.4 * k * (1 if s < 30 else 3)
        t = np.arange(t0, t0 + 25, 2 / 1440)
        f = 1 + 6e-4 * rng.standard_normal(t.size) + 3e-4 * np.sin(t / 3)
        n = np.round((t - T0) / P)
        ph = t - T0 - n * P
        d = depth_ppm * 1e-6 * (0.2 if s == weak_sector else 1.0)
        f[np.abs(ph) < dur / 2] -= d
        np.savetxt(os.path.join(tmp, f"lc_corrected_{TIC}_s{s}.txt"), np.c_[t, f, np.full(t.size, 6e-4)])
    log = (f"triceratops_validation.py v4.1.1\n"
           f"Adopted ephemeris: P = {period} d, T0 = {T0}, T14 = {T14_H} h, depth = {depth_ppm:.0f} ppm\n"
           + log_extra)
    open(os.path.join(tmp, f"triceratops_output_TIC{TIC}.txt"), "w").write(log)
    json.dump({"tic": TIC, "P_d": period, "T0_BTJD": T0, "T14_h": T14_H,
               "depth_ppm_adopted": depth_ppm, **(summary_extra or {})},
              open(os.path.join(tmp, f"summary_{TIC}.json"), "w"))
    return tmp


def _tests(A):
    return {t.area: t for t in A.tests}


def test_recovers_injected_signal(tmp_path):
    A = ta.run_audit(_make_folder(str(tmp_path)))
    T = _tests(A)
    assert T["Recomputation: BLS"].status == "PASS", T["Recomputation: BLS"].evidence
    assert T["Recomputation: sectors"].status == "PASS", T["Recomputation: sectors"].evidence


def test_no_signal_is_not_recovered(tmp_path):
    A = ta.run_audit(_make_folder(str(tmp_path), depth_ppm=0.0))
    assert _tests(A)["Recomputation: BLS"].status == "WARN"


def test_small_period_error_that_drifts_is_not_a_pass(tmp_path):
    # 0.05 % in P is ~4 h of drift over this baseline (T14 = 4 h)
    A = ta.run_audit(_make_folder(str(tmp_path), period=P * 1.0005))
    t = _tests(A)["Recomputation: BLS"]
    assert t.status == "WARN" and "drift apart" in t.evidence, t.evidence


def test_deviant_sector_flagged(tmp_path):
    A = ta.run_audit(_make_folder(str(tmp_path), weak_sector=37))
    t = _tests(A)["Recomputation: sectors"]
    assert t.status == "WARN" and "sector 37" in t.evidence, t.evidence


def test_missing_files_give_na_not_crash(tmp_path):
    open(os.path.join(tmp_path, f"triceratops_output_TIC{TIC}.txt"), "w").write("nothing useful\n")
    A = ta.run_audit(str(tmp_path))
    assert all(t.status in ("PASS", "WARN", "FAIL", "NA") for t in A.tests)
    assert any(t.status == "NA" for t in A.tests)


def test_outputs_written(tmp_path):
    folder = _make_folder(str(tmp_path))
    ta.main([folder, "--no-recompute"])
    for ext in ("json", "txt", "pdf"):
        assert os.path.getsize(os.path.join(folder, f"audit_{TIC}.{ext}")) > 100
    d = json.load(open(os.path.join(folder, f"audit_{TIC}.json")))
    assert d["tic"] == TIC and "tests" in d


def test_threshold_override(tmp_path):
    folder = _make_folder(str(tmp_path))
    A = ta.run_audit(folder, thresholds={"bls_drift_tol": 1e-9}, recompute_lc=False)
    assert A.thresholds["bls_drift_tol"] == 1e-9


def test_failure_carried_by_edge_transits_is_a_warning(tmp_path):
    extra = dict(depth_chi2_red=9.5, depth_chi2_red_excl_edge=1.2,
                 odd_even_sigma=6.5, odd_even_sigma_excl_edge=0.8,
                 edge_transit_epochs=[57])
    A = ta.run_audit(_make_folder(str(tmp_path), summary_extra=extra), recompute_lc=False)
    T = _tests(A)
    for area in ("Depth consistency", "Odd/even"):
        assert T[area].status == "WARN" and "data gap" in T[area].evidence, T[area].evidence
    extra.update(depth_chi2_red_excl_edge=9.0, odd_even_sigma_excl_edge=6.0)
    A = ta.run_audit(_make_folder(str(tmp_path), summary_extra=extra), recompute_lc=False)
    assert _tests(A)["Depth consistency"].status == "FAIL"
    assert _tests(A)["Odd/even"].status == "FAIL"


def test_density_with_unconstrained_b(tmp_path):
    def shape(rho_fit, rho_cat):
        log = (f"  b    = 0.10000 +/- 40.00000\n"
               f"  rho_star (circular orbit) = {rho_fit} g/cm^3\n"
               f"  rho_star from TIC mass/radius       = {rho_cat} g/cm^3   (consistency check)\n")
        extra = {"ld_fit": {"rp": 0.0400, "b": 0.1000, "grazing": False, "T14_h": 1.5}}
        A = ta.run_audit(_make_folder(str(tmp_path), log_extra=log, summary_extra=extra),
                         recompute_lc=False)
        return _tests(A)["Shape"]
    t = shape(25.0, 2.5)                # a higher b explains it
    assert t.status == "PASS" and "b = 0.92" in t.evidence, t.evidence
    t = shape(1.0, 5.0)                 # too long even for a central transit
    assert t.status == "WARN" and "stellar density" in t.evidence, t.evidence


def test_centroid_names_the_star_it_points_to(tmp_path):
    folder = _make_folder(str(tmp_path), log_extra="NFPP = 0.2 +/- 0.01   [bootstrap over runs]\n")
    with open(os.path.join(folder, f"difference_imaging_{TIC}.csv"), "w") as fh:
        fh.write("sector,snr,offset_direct_sigma\n17,6.0,3.8\n24,30.0,4.4\n")
    with open(os.path.join(folder, f"difference_imaging_stars_{TIC}.csv"), "w") as fh:
        fh.write("sector,ID,is_target,sep_target_arcsec,n_sigma\n"
                 f"17,{TIC},True,0.0,3.6\n24,{TIC},True,0.0,4.8\n"
                 "17,111,False,75.0,0.3\n24,111,False,75.0,0.2\n"      # both sectors
                 "17,222,False,90.0,0.5\n24,222,False,90.0,5.9\n")     # one sector only
    with open(os.path.join(folder, f"field_TIC{TIC}.csv"), "w") as fh:
        fh.write(f"ID,ra,dec,Tmag\n{TIC},0,0,10.5\n111,0,0,9.8\n222,0,0,12.0\n")
    T = _tests(ta.run_audit(folder, recompute_lc=False))
    assert "TIC 111 (T = 9.80" in T["Centroids"].evidence and "TIC 222" not in T["Centroids"].evidence
    assert T["Neighbours"].status == "WARN" and "TIC 111" in T["Neighbours"].evidence
    # a marginal offset (WARN, median < 3 sigma) names no stars
    with open(os.path.join(folder, f"difference_imaging_{TIC}.csv"), "w") as fh:
        fh.write("sector,snr,offset_direct_sigma\n17,6.0,0.6\n24,30.0,3.4\n")
    T = _tests(ta.run_audit(folder, recompute_lc=False))
    assert T["Centroids"].status == "WARN" and "consistent" not in T["Centroids"].evidence
    assert T["Neighbours"].status == "PASS", T["Neighbours"].evidence


def test_audit_never_changes_inputs(tmp_path):
    folder = _make_folder(str(tmp_path))
    before = {f: os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder)}
    ta.main([folder])
    for f, m in before.items():
        assert os.path.getmtime(os.path.join(folder, f)) == m
