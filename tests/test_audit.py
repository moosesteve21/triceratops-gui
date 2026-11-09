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


def _make_folder(tmp, depth_ppm=DEPTH, weak_sector=None, log_extra=""):
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
           f"Adopted ephemeris: P = {P} d, T0 = {T0}, T14 = {T14_H} h, depth = {depth_ppm:.0f} ppm\n"
           + log_extra)
    open(os.path.join(tmp, f"triceratops_output_TIC{TIC}.txt"), "w").write(log)
    json.dump({"tic": TIC, "P_d": P, "T0_BTJD": T0, "T14_h": T14_H,
               "depth_ppm_adopted": depth_ppm},
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
    A = ta.run_audit(folder, thresholds={"bls_period_tol": 1e-9}, recompute_lc=False)
    assert A.thresholds["bls_period_tol"] == 1e-9


def test_audit_never_changes_inputs(tmp_path):
    folder = _make_folder(str(tmp_path))
    before = {f: os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder)}
    ta.main([folder])
    for f, m in before.items():
        assert os.path.getmtime(os.path.join(folder, f)) == m
