#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation and Skeptic audit of a triceratops_validation.py run (prototype).

    python triceratops_audit.py TIC65344135
    python triceratops_audit.py TIC65344135 --no-recompute
    python triceratops_audit.py TIC65344135 --thresholds my_thresholds.json

Reads the output folder of a finished run (summary_<TIC>.json, the log
triceratops_output_TIC<TIC>.txt, the CSV/TXT tables and the saved light
curves) and writes

    audit_<TIC>.json   every claim and every test, with its source
    audit_<TIC>.txt    the same as a readable table
    audit_<TIC>.pdf    one to three pages for a report

Design (see "Evaluation and Skeptic layer for the TRICERATOPS pipeline"):
  * deterministic code, no language model; the same folder gives the same
    report;
  * every number names its source (file and key, or log line);
  * a test without its data is NOT AVAILABLE, never PASS;
  * the audit does not change the pipeline's verdict;
  * two independent recomputations from the saved light curves (a box
    least-squares search with its own detrending, per-sector box depths).

Results: PASS, WARN, FAIL, NA (not available).  Thresholds are in
THRESHOLDS below and can be overridden with a JSON file.

Prototype: the checks and thresholds are proposals for discussion.
Needs numpy and matplotlib; astropy (BoxLeastSquares) for the BLS check.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

__version__ = "0.1"

THRESHOLDS = {
    "ephem_shift_warn_T14": 0.25,     # adopted vs input ephemeris, in T14
    "min_transits_fail": 3,
    "loo_shift_warn_sigma": 3.0,      # weighted depth without one transit
    "weak_transit_snr": 3.0,
    "depth_chi2_warn": 2.0,
    "depth_chi2_fail": 4.0,
    "odd_even_warn": 2.0,
    "odd_even_fail": 3.0,
    "density_ratio_warn": 1.5,        # transit vs catalogue stellar density
    "g_range_warn": 0.05,
    "g_model_diff_warn": 0.02,
    "di_sector_warn_sigma": 3.0,
    "di_median_fail_sigma": 3.0,
    "di_min_snr": 3.0,
    "nfpp_warn": 0.01,
    "dominance_warn": 0.2,
    "valid_frac_warn": 0.8,
    "ground_depth_tol": 0.25,
    "bls_drift_tol": 0.5,             # max BLS-vs-pipeline mid-time drift [T14]
    "bls_depth_tol": 0.30,            # fractional
    "sector_depth_chi2_warn": 3.0,
    "input_depth_tol": 0.30,          # reference depth vs measured depth
}

STATUS_ORDER = {"FAIL": 0, "WARN": 1, "NA": 2, "PASS": 3}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Claim:
    item: str
    value: object
    unit: str = ""
    kind: str = ""          # measured | derived | fitted | reference
    source: str = ""


@dataclass
class Test:
    area: str
    question: str
    status: str             # PASS | WARN | FAIL | NA
    evidence: str
    source: str = ""
    advice: str = ""


@dataclass
class Audit:
    tic: int
    outdir: str
    audit_version: str = __version__
    pipeline_version: str = ""
    pipeline_verdict: str = ""
    claims: list = field(default_factory=list)
    tests: list = field(default_factory=list)
    recomputations: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    def claim(self, *a, **k):
        self.claims.append(Claim(*a, **k))

    def test(self, *a, **k):
        self.tests.append(Test(*a, **k))

    def counts(self):
        out = {s: 0 for s in STATUS_ORDER}
        for t in self.tests:
            out[t.status] = out.get(t.status, 0) + 1
        return out


# --------------------------------------------------------------------------
# Reading the run
# --------------------------------------------------------------------------
class Run:
    """Everything the audit reads from one output folder."""

    def __init__(self, outdir: str, tic: Optional[int] = None):
        self.outdir = os.path.abspath(outdir)
        self.tic = tic or self._guess_tic()
        t = self.tic
        self.files = {}
        self.summary = self._json(f"summary_{t}.json")
        self.log_name = f"triceratops_output_TIC{t}.txt"
        self.log = self._text(self.log_name)
        self.log_lines = self.log.splitlines() if self.log else []
        self.per_transit = self._table(f"per_transit_depths_{t}.txt")
        self.di = self._csv(f"difference_imaging_{t}.csv")
        self.di_stars = self._csv(f"difference_imaging_stars_{t}.csv")
        self.field = self._csv(f"field_TIC{t}.csv")
        self.ground = self._csv(f"ground_clearance_{t}.csv")

    def _guess_tic(self):
        for p in glob.glob(os.path.join(self.outdir, "triceratops_output_TIC*.txt")):
            m = re.search(r"TIC(\d+)\.txt$", p)
            if m:
                return int(m.group(1))
        m = re.search(r"(\d{5,})", os.path.basename(self.outdir.rstrip("/")))
        if m:
            return int(m.group(1))
        raise SystemExit(f"cannot determine the TIC for {self.outdir}; use --tic")

    def path(self, name):
        return os.path.join(self.outdir, name)

    def _json(self, name):
        p = self.path(name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                self.files[name] = p
                return json.load(fh)
        return {}

    def _text(self, name):
        p = self.path(name)
        if os.path.isfile(p):
            self.files[name] = p
            with open(p, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        return ""

    def _table(self, name):
        p = self.path(name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as fh:
            head = fh.readline().lstrip("#").split()
        try:
            a = np.atleast_2d(np.loadtxt(p))
        except Exception:
            return None
        self.files[name] = p
        return {h: a[:, i] for i, h in enumerate(head) if i < a.shape[1]}

    def _csv(self, name):
        p = self.path(name)
        if not os.path.isfile(p):
            return None
        import csv
        with open(p, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.files[name] = p
        return rows

    # -- log helpers -------------------------------------------------------
    def grep(self, pattern, last=False):
        """(match, line number) of the first (or last) line matching."""
        rx = re.compile(pattern)
        hit = None
        for i, ln in enumerate(self.log_lines, 1):
            m = rx.search(ln)
            if m:
                hit = (m, i)
                if not last:
                    return hit
        return hit if hit else (None, None)

    def grep_all(self, pattern):
        rx = re.compile(pattern)
        return [(rx.search(ln), i) for i, ln in enumerate(self.log_lines, 1)
                if rx.search(ln)]

    def src(self, line):
        return f"{self.log_name}:{line}" if line else self.log_name

    def S(self, *keys, default=None):
        d = self.summary
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d


def _f(x, fmt="{:.3g}"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return fmt.format(x)
    except Exception:
        return str(x)


# --------------------------------------------------------------------------
# Evaluation: what the run claims
# --------------------------------------------------------------------------
def evaluate(run: Run, A: Audit):
    sj = f"summary_{run.tic}.json"
    m, ln = run.grep(r"triceratops version: (\S+)")
    if m:
        A.claim("triceratops version", m.group(1), kind="reference", source=run.src(ln))
    if run.S("script_version"):
        A.pipeline_version = str(run.S("script_version"))
        A.claim("pipeline version", A.pipeline_version, kind="reference",
                source=f"{sj}:script_version")
    if run.S("triceratops_plus") is not None:
        A.claim("TRICERATOPS+", bool(run.S("triceratops_plus")), kind="reference",
                source=f"{sj}:triceratops_plus")
    m, ln = run.grep(r"KNOWN ephemeris.*P = ([\d.]+) d, T0 = ([\d.]+) BTJD, "
                     r"T14 = ([\d.]+) h, depth = ([\d.]+) ppm")
    if m:
        A.claim("input ephemeris P, T0, T14, depth",
                f"{m.group(1)} d, {m.group(2)} BTJD, {m.group(3)} h, {m.group(4)} ppm",
                kind="reference", source=run.src(ln))
    m, ln = run.grep(r"Adopted ephemeris: P = ([\d.]+) d, T0 = ([\d.]+)")
    if m:
        A.claim("adopted ephemeris P, T0", f"{m.group(1)} d, {m.group(2)} BTJD",
                kind="fitted", source=run.src(ln))
    elif run.S("P_d"):
        A.claim("adopted ephemeris P, T0", f"{run.S('P_d'):.6f} d, {run.S('T0_BTJD'):.5f} BTJD",
                kind="fitted", source=f"{sj}:P_d,T0_BTJD")
    secs = run.grep_all(r"sector\s+(\d+): (optimal|percentile)\s+(\d+) px\s+g =\s+([\d.]+)")
    if secs:
        A.claim("sectors (aperture px, target fraction g)",
                ", ".join(f"{m.group(1)} ({m.group(3)} px, {m.group(4)})" for m, _ in secs),
                kind="measured", source=run.src(secs[0][1]))
    if run.per_transit is not None:
        d = run.per_transit
        A.claim("transits in the per-transit table", int(len(d["epoch"])), kind="measured",
                source=f"per_transit_depths_{run.tic}.txt")
    for key, lab, unit, kind in (("depth_ppm_adopted", "depth adopted (aperture)", "ppm", "fitted"),
                                 ("depth_on_target_ppm", "depth on the target", "ppm", "derived"),
                                 ("depth_chi2_red", "depth chi2/dof", "", "derived"),
                                 ("odd_even_sigma", "odd/even difference", "sigma", "derived")):
        v = run.S(key)
        if v is not None:
            A.claim(lab, _f(v), unit, kind, f"{sj}:{key}")
    if run.S("ld_fit"):
        L = run.S("ld_fit")
        A.claim("LD fit Rp/R*, b, grazing, T14",
                f"{_f(L.get('rp'))}, {_f(L.get('b'))}, {L.get('grazing')}, {_f(L.get('T14_h'))} h",
                kind="fitted", source=f"{sj}:ld_fit")
    m, ln = run.grep(r"rho_star \(circular orbit\) = ([\d.]+)")
    m2, ln2 = run.grep(r"rho_star from TIC mass/radius\s+= ([\d.]+)")
    if m and m2:
        A.claim("stellar density transit / TIC", f"{m.group(1)} / {m2.group(1)}", "g/cm3",
                "fitted vs reference", run.src(ln))
    if run.S("secondary"):
        s = run.S("secondary")
        A.claim("secondary eclipse", "detected" if s.get("detected") else
                f"none; 3-sigma limit {_f(s.get('lim3_ppm_median'), '{:.0f}')} ppm",
                kind="derived", source=f"{sj}:secondary")
    if run.S("etv"):
        e = run.S("etv")
        A.claim("ETV O-C rms, LTTE limit", f"{_f(e.get('rms_min'))} min, "
                f"{_f(e.get('ltte_lim3_min'))} min", kind="derived", source=f"{sj}:etv")
    if run.di:
        A.claim("difference-imaging sectors", len(run.di), kind="measured",
                source=f"difference_imaging_{run.tic}.csv")
    hosts = _neighbour_hosts(run)
    if hosts is not None:
        A.claim("neighbours that can host the signal",
                ", ".join(f"TIC {h['ID']} ({h['sep']}\", f {h['fluxratio']})" for h in hosts) or "none",
                kind="derived", source=run.src(hosts and hosts[0].get("line")))
    if run.S("ground"):
        G = run.S("ground")
        A.claim("ground light curves (detected, timing)",
                f"{G.get('n_lc')} ({G.get('n_detected')}, {G.get('n_timing')})",
                kind="measured", source=f"{sj}:ground")
    for key, lab in (("FPP", "FPP"), ("NFPP", "NFPP")):
        if run.S(key) is not None:
            A.claim(lab, _f(run.S(key), "{:.4g}"), kind="fitted", source=f"{sj}:{key}")
    m, ln = run.grep(r"Giacalone.*: (PASSED|NOT passed.*)$", last=True)
    if run.S("verdict"):
        A.pipeline_verdict = str(run.S("verdict"))
    elif m:
        A.pipeline_verdict = m.group(1)
    if A.pipeline_verdict:
        A.claim("pipeline verdict", A.pipeline_verdict, kind="derived",
                source=f"{sj}:verdict" if run.S("verdict") else run.src(ln))


def _neighbour_hosts(run: Run):
    """Rows of the 'neighbours that can host the signal' table in the log."""
    m, ln = run.grep(r"neighbours that can host the signal")
    if not m:
        m2, ln2 = run.grep(r"no neighbour contributes enough flux")
        return [] if m2 else None
    out = []
    for j in range(ln + 1, min(ln + 40, len(run.log_lines))):
        parts = run.log_lines[j - 1].split()
        if len(parts) >= 5 and parts[0].isdigit():
            out.append(dict(ID=parts[0], Tmag=parts[1], sep=parts[2],
                            fluxratio=parts[3], tdepth=parts[4], line=j))
        elif out and not parts:
            break
    return out


def _centroid_stars(run: Run, sectors, max_sigma):
    """Catalogued stars (not the target) within max_sigma of the
    difference-image centroid in every given sector, brightest first."""
    if not run.di_stars:
        return []
    want = {int(s) for s in sectors}
    per = {}
    for r in run.di_stars:
        try:
            s = int(float(r["sector"]))
            if s in want and str(r.get("is_target")) != "True":
                per.setdefault(r["ID"], {})[s] = (float(r["n_sigma"]),
                                                  float(r["sep_target_arcsec"]))
        except (KeyError, ValueError):
            continue
    tmag = {r["ID"]: r.get("Tmag", "") for r in (run.field or [])}
    out = []
    for sid, d in per.items():
        if set(d) == want and all(ns < max_sigma for ns, _ in d.values()):
            out.append(dict(ID=sid, sep=next(iter(d.values()))[1],
                            Tmag=_f(float(tmag[sid]), "{:.2f}") if tmag.get(sid) else "?"))
    return sorted(out, key=lambda h: float(h["Tmag"]) if h["Tmag"] != "?" else 99.0)


# --------------------------------------------------------------------------
# Skeptic: tests that try to break the claim
# --------------------------------------------------------------------------
def skeptic(run: Run, A: Audit, T: dict):
    sj = f"summary_{run.tic}.json"

    # 1. environment
    m, ln = run.grep(r"cannot run TRICERATOPS|Traceback")
    vr, lnv = run.grep(r"valid runs: (\d+)/(\d+)")
    if m:
        A.test("Environment", "Did the run complete in a working environment?", "FAIL",
               "the log contains an environment error or a traceback", run.src(ln),
               "check the package versions (GUI: Tools -> Check interpreter)")
    elif vr:
        A.test("Environment", "Did the run complete in a working environment?", "PASS",
               f"TRICERATOPS completed {vr.group(1)}/{vr.group(2)} valid runs", run.src(lnv))
    else:
        A.test("Environment", "Did the run complete in a working environment?", "NA",
               "no TRICERATOPS result in the log", run.log_name)

    # 2. ephemeris
    mi, lni = run.grep(r"KNOWN ephemeris.*T14 = ([\d.]+) h")
    mad, lnad = run.grep(r"misses the transits by up to ([\d.]+) h \(= ([\d.]+) T14\)")
    mok, lnok = run.grep(r"known ephemeris confirmed")
    mfit, lnfit = run.grep(r"linear fit of (\d+) transits")
    if not mi:
        A.test("Ephemeris", "Did the input ephemeris hold?", "NA",
               "blind search (no input ephemeris)", run.log_name)
    else:
        n = int(mfit.group(1)) if mfit else None
        if n is not None and n < T["min_transits_fail"]:
            A.test("Ephemeris", "Did the input ephemeris hold?", "FAIL",
                   f"only {n} transits in the ephemeris fit", run.src(lnfit))
        elif mad and float(mad.group(2)) > T["ephem_shift_warn_T14"]:
            A.test("Ephemeris", "Did the input ephemeris hold?", "WARN",
                   f"input ephemeris missed the transits by up to {mad.group(1)} h "
                   f"({mad.group(2)} T14); refined from {n} transits", run.src(lnad),
                   "report the refined ephemeris to the observers")
        elif mok or mad:
            A.test("Ephemeris", "Did the input ephemeris hold?", "PASS",
                   f"confirmed from {n} transits" if mok else "small correction",
                   run.src(lnok or lnad))
        else:
            A.test("Ephemeris", "Did the input ephemeris hold?", "NA",
                   "no ephemeris check in the log", run.log_name)

    # 3. single events (leave one out + weak transits)
    d = run.per_transit
    if d is None or len(d.get("depth", [])) < 3:
        A.test("Single events", "Does one transit carry the signal?", "NA",
               "per-transit table missing or < 3 transits", f"per_transit_depths_{run.tic}.txt")
    else:
        dep, err = d["depth"], d["e_depth"]
        snr = d.get("snr", dep / err)
        w = 1 / err ** 2
        mu = np.sum(w * dep) / np.sum(w)
        shifts = []
        for i in range(dep.size):
            k = np.arange(dep.size) != i
            mu_i = np.sum(w[k] * dep[k]) / np.sum(w[k])
            se = 1 / np.sqrt(np.sum(w[k]))
            shifts.append(abs(mu_i - mu) / se)
        imax = int(np.argmax(shifts))
        weak = [int(e) for e, s in zip(d["epoch"], snr) if s < T["weak_transit_snr"]]
        src = f"per_transit_depths_{run.tic}.txt"
        if max(shifts) > T["loo_shift_warn_sigma"] or weak:
            ev = []
            if weak:
                ev.append(f"weak transit(s) S/N < {T['weak_transit_snr']:g} at epoch {weak}")
            if max(shifts) > T["loo_shift_warn_sigma"]:
                ev.append(f"leaving out epoch {int(d['epoch'][imax])} moves the mean depth "
                          f"{shifts[imax]:.1f} sigma")
            A.test("Single events", "Does one transit carry the signal?", "WARN",
                   "; ".join(ev), src, "re-run with that sector excluded and compare")
        else:
            A.test("Single events", "Does one transit carry the signal?", "PASS",
                   f"{dep.size} transits; largest leave-one-out shift {max(shifts):.1f} sigma",
                   src)

    # 4. depth consistency
    c = run.S("depth_chi2_red")
    if c is None:
        mm = run.grep_all(r"constant-depth chi2/dof = ([\d.]+)")
        c = float(mm[-1][0].group(1)) if mm else None
        src = run.src(mm[-1][1]) if mm else run.log_name
    else:
        src = f"{sj}:depth_chi2_red"
    if c is None:
        A.test("Depth consistency", "Are the transits equally deep?", "NA", "not reported", src)
    else:
        st = "FAIL" if c > T["depth_chi2_fail"] else "WARN" if c > T["depth_chi2_warn"] else "PASS"
        txt, adv = f"chi2/dof = {c:.2f}", ""
        cx, edge = run.S("depth_chi2_red_excl_edge"), run.S("edge_transit_epochs")
        if st != "PASS" and cx is not None and edge:
            txt += f"; {cx:.2f} without the transit(s) next to a data gap (epochs {edge})"
            if st == "FAIL" and cx <= T["depth_chi2_fail"]:
                # not downgraded: a transit next to a gap can be as real as
                # any other (e.g. an absent odd transit when the period is 2P)
                adv = ("the failure rests on transits next to a data gap; inspect "
                       "them before discarding them")
        A.test("Depth consistency", "Are the transits equally deep?", st, txt, src, adv)

    # 5. odd/even
    oe = run.S("odd_even_sigma")
    src = f"{sj}:odd_even_sigma"
    if oe is None:
        mm = run.grep_all(r"odd/even difference(?: \(on-target\)| \(all cadences\))? = ([\d.]+) sigma")
        oe = float(mm[-1][0].group(1)) if mm else None
        src = run.src(mm[-1][1]) if mm else run.log_name
    if oe is None:
        A.test("Odd/even", "Could the true period be 2P?", "NA", "not reported", src)
    else:
        st = "FAIL" if oe > T["odd_even_fail"] else "WARN" if oe > T["odd_even_warn"] else "PASS"
        txt, adv = f"{oe:.2f} sigma", ""
        ox, edge = run.S("odd_even_sigma_excl_edge"), run.S("edge_transit_epochs")
        if st != "PASS" and ox is not None and edge:
            txt += f"; {ox:.2f} sigma without the transit(s) next to a data gap (epochs {edge})"
            if st == "FAIL" and ox <= T["odd_even_fail"]:
                adv = ("the failure rests on transits next to a data gap; inspect "
                       "them before discarding them")
        mp = run.S("missing_parity")
        if st != "PASS" and mp in ("odd", "even"):
            other = "even" if mp == "odd" else "odd"
            d, e = run.S(f"depth_{mp}_ppm"), run.S(f"depth_{mp}_err_ppm")
            do, eo = run.S(f"depth_{other}_ppm"), run.S(f"depth_{other}_err_ppm")
            if None not in (d, e, do, eo):
                txt += (f"; no transit at the {mp} epochs ({mp} {d:.0f} +/- {e:.0f} ppm, "
                        f"{other} {do:.0f} +/- {eo:.0f} ppm)")
            P = run.S("P_d")
            adv = (f"rerun at 2P = {2 * P:.6f} d" if P else "rerun at twice the period") \
                + (f"; {adv}" if adv else "")
        A.test("Odd/even", "Could the true period be 2P?", st, txt, src, adv)

    # 6. secondary
    s = run.S("secondary")
    if s:
        if s.get("detected"):
            A.test("Secondary", "Is there light from a companion?", "FAIL",
                   f"secondary at phase {s.get('phase')}, depth {s.get('depth_ppm')} ppm",
                   f"{sj}:secondary")
        else:
            A.test("Secondary", "Is there light from a companion?", "PASS",
                   f"none; 3-sigma limit {_f(s.get('lim3_ppm_median'), '{:.0f}')} ppm, "
                   f"companion < {_f(s.get('companion_T_K'), '{:.0f}')} K", f"{sj}:secondary")
    else:
        m, ln = run.grep(r"SECONDARY ECLIPSE (?:candidate|DETECTED)[^\n]*")
        m0, ln0 = run.grep(r"no coherent secondary eclipse")
        if m:
            A.test("Secondary", "Is there light from a companion?", "WARN",
                   m.group(0).strip()[:140], run.src(ln),
                   "older pipeline versions accepted single-event candidates: check the figure")
        elif m0:
            A.test("Secondary", "Is there light from a companion?", "PASS",
                   "no coherent secondary eclipse", run.src(ln0))
        else:
            A.test("Secondary", "Is there light from a companion?", "NA", "no search in the log",
                   run.log_name)

    # 7. shape / density
    L = run.S("ld_fit") or {}
    mg, lng = run.grep(r"GRAZING geometry: b \+ Rp/Rs = ([\d.]+)")
    m, ln = run.grep(r"rho_star \(circular orbit\) = ([\d.]+)")
    m2, _ = run.grep(r"rho_star from TIC mass/radius\s+= ([\d.]+)")
    mb, lnb = run.grep(r"^\s+b\s+=\s+([\d.]+) \+/- ([\d.]+)")
    issues, notes = [], []
    if L.get("grazing") or mg:
        issues.append(f"grazing (b + k = {mg.group(1)})" if mg else "grazing")
    b_free = bool(mb and float(mb.group(2)) > 1)
    if m and m2:
        rho0, rho_c = float(m.group(1)), float(m2.group(1))
        r = rho0 / rho_c
        if r > T["density_ratio_warn"] or r < 1 / T["density_ratio_warn"]:
            # With b unconstrained, T14 fixes a/R* only up to
            # sqrt((1+k)^2 - b^2): rho(b) = rho0 [((1+k)^2-b^2)/((1+k)^2-b0^2)]^1.5.
            # Look for a b in [0, 1+k) that gives the catalogue density.
            k = L.get("rp")
            b0 = L.get("b", float(mb.group(1)) if mb else None)
            b_match = None
            if b_free and k is not None and b0 is not None:
                s2 = (1 + k) ** 2 - b0 ** 2
                b2 = (1 + k) ** 2 - s2 * (rho_c / rho0) ** (2.0 / 3.0)
                if 0.0 <= b2 < (1 + k) ** 2:
                    b_match = float(np.sqrt(b2))
            if b_match is not None:
                notes.append(f"stellar density {rho0:g} at b = {b0:.2f} vs {rho_c:g} g/cm3, "
                             f"but b is unconstrained and b = {b_match:.2f} gives the "
                             f"catalogue density")
            else:
                issues.append(f"stellar density {m.group(1)} vs {m2.group(1)} g/cm3 (ratio {r:.2f})")
    if b_free and not notes:
        issues.append(f"b unconstrained ({mb.group(1)} +/- {float(mb.group(2)):.3g})")
    if not (L or m):
        A.test("Shape", "Is the shape physical for this star?", "NA",
               "no limb-darkened fit (--ld-fit)", run.log_name)
    elif issues:
        A.test("Shape", "Is the shape physical for this star?", "WARN", "; ".join(issues),
               run.src(lng or ln), "an eccentric orbit, a wrong host or an EB can cause this")
    else:
        A.test("Shape", "Is the shape physical for this star?", "PASS",
               "not grazing; " + ("; ".join(notes) if notes else "stellar density consistent"),
               run.src(ln))

    # 8. dilution
    gr = run.S("g_target_range")
    secs = run.grep_all(r"sector\s+\d+: (?:optimal|percentile)\s+\d+ px\s+g =\s+([\d.]+)")
    gs = [float(m.group(1)) for m, _ in secs]
    if gr is None and gs:
        gr = [min(gs), max(gs)]
    md, lnd = run.grep(r"largest difference ([\d.]+) over (\d+) sectors")
    if gr is None:
        A.test("Dilution", "Do the sectors agree on the target fraction?", "NA",
               "no per-sector target fractions", run.log_name)
    else:
        spread = gr[1] - gr[0]
        diff = float(md.group(1)) if md else None
        bad = spread > T["g_range_warn"] or (diff is not None and diff > T["g_model_diff_warn"])
        A.test("Dilution", "Do the sectors agree on the target fraction?",
               "WARN" if bad else "PASS",
               f"g = {gr[0]:.3f} to {gr[1]:.3f}" + (f"; PSF models differ by {diff:.3f}"
                                                   if diff is not None else ""),
               run.src(lnd) if md else f"{sj}:g_target_range")

    # 9. centroids
    di_hosts = []
    if not run.di:
        A.test("Centroids", "Is the dip on the target?", "NA", "no difference imaging",
               f"difference_imaging_{run.tic}.csv")
    else:
        key = "offset_direct_sigma" if "offset_direct_sigma" in run.di[0] else "offset_cat_sigma"
        vals = []
        for r in run.di:
            try:
                if float(r.get("snr", "nan")) >= T["di_min_snr"]:
                    vals.append((int(float(r["sector"])), float(r[key])))
            except (ValueError, KeyError):
                pass
        if not vals:
            A.test("Centroids", "Is the dip on the target?", "NA",
                   "no sector with difference-image S/N >= 3", f"difference_imaging_{run.tic}.csv")
        else:
            med = float(np.median([v for _, v in vals]))
            hi = [f"s{s}: {v:.1f}" for s, v in vals if v > T["di_sector_warn_sigma"]]
            st = ("FAIL" if med > T["di_median_fail_sigma"] else "WARN" if hi else "PASS")
            txt = (f"median offset {med:.1f} sigma over {len(vals)} sectors ({key})"
                   + (f"; > {T['di_sector_warn_sigma']:g} sigma in {', '.join(hi)}" if hi else ""))
            # only when the offset itself is significant: near the target
            # many faint stars are "consistent" with a noisy centroid
            if st == "FAIL":
                di_hosts = _centroid_stars(run, [s for s, _ in vals], T["di_sector_warn_sigma"])
                if di_hosts:
                    txt += ("; the centroid is consistent (< "
                            f"{T['di_sector_warn_sigma']:g} sigma in every sector) with "
                            + ", ".join(f"TIC {h['ID']} (T = {h['Tmag']}, {h['sep']:.1f}\")"
                                        for h in di_hosts[:3])
                            + (f" and {len(di_hosts) - 3} fainter star(s)" if len(di_hosts) > 3 else ""))
            A.test("Centroids", "Is the dip on the target?", st, txt,
                   f"difference_imaging_{run.tic}.csv")

    # 10. neighbours
    hosts = _neighbour_hosts(run)
    nfpp = run.S("NFPP")
    if nfpp is None:
        m, ln = run.grep(r"NFPP\s+=\s+([\d.eE+-]+)\s+\+/-.*bootstrap")
        nfpp = float(m.group(1)) if m else None
    if hosts is None and nfpp is None:
        A.test("Neighbours", "Which stars can still host the signal?", "NA",
               "no NFPP input in the log", run.log_name)
    else:
        # a star the difference image points to but the flux budget leaves out
        known = {str(h["ID"]) for h in (hosts or [])}
        missed = [h for h in di_hosts if h["ID"] not in known]
        miss_txt = ("; difference imaging points to "
                    + ", ".join(f"TIC {h['ID']} ({h['sep']:.1f}\")" for h in missed[:3])
                    + ", which the flux budget does not list") if missed else ""
        miss_adv = (" - and run the flux budget for the star(s) the difference image "
                    "points to") if missed else ""
        if hosts:
            names = ", ".join(f"TIC {h['ID']} ({h['sep']}\", flux ratio {h['fluxratio']})"
                              for h in hosts)
            st = "WARN" if (nfpp or 0) > T["nfpp_warn"] or missed else "PASS"
            A.test("Neighbours", "Which stars can still host the signal?", st,
                   f"NFPP = {_f(nfpp, '{:.3g}')}; possible hosts: {names}{miss_txt}",
                   run.src(hosts[0]["line"]),
                   "ground photometry with a separate aperture on each named star clears it"
                   + miss_adv)
        else:
            A.test("Neighbours", "Which stars can still host the signal?",
                   "WARN" if missed else "PASS",
                   f"no neighbour can host the signal (flux budget); NFPP = "
                   f"{_f(nfpp, '{:.3g}')}{miss_txt}", run.log_name, miss_adv.lstrip(" -"))

    # 11. Monte Carlo
    dom = run.S("dominance")
    nv, nr = run.S("n_valid"), run.S("n_runs")
    mdom, lndom = run.grep(r"largest share of a single run.*: ([\d.]+)")
    if dom is None and mdom:
        dom = float(mdom.group(1))
    if nv is None and vr:
        nv, nr = int(vr.group(1)), int(vr.group(2))
    mb0, lnb0 = run.grep(r"FPP\s+=\s+[\d.]+\s+\+/-\s+([\d.]+)\s+\[bootstrap")
    if dom is None and nv is None:
        A.test("Monte Carlo", "Is the FPP converged and stable?", "NA", "no TRICERATOPS runs",
               run.log_name)
    else:
        ev, st = [], "PASS"
        if nv is not None and nr:
            ev.append(f"{nv}/{nr} valid runs")
            if nv / nr < T["valid_frac_warn"]:
                st = "WARN"
        if dom is not None:
            ev.append(f"dominance {dom:.2f}")
            if dom > T["dominance_warn"]:
                st = "WARN"
        if mb0 is not None:
            ev.append(f"bootstrap error {mb0.group(1)}")
            fpp = run.S("FPP")
            if float(mb0.group(1)) == 0 and (fpp is None or 0.001 < float(fpp) < 0.999):
                st = "WARN"
                ev.append("zero bootstrap error with an intermediate FPP: pooling suspect")
        mnc, _ = run.grep(r"Monte Carlo not converged")
        if mnc:
            st = "WARN"
        A.test("Monte Carlo", "Is the FPP converged and stable?", st, "; ".join(ev),
               f"{sj}:n_valid,dominance" if run.S("dominance") is not None else run.src(lndom),
               "raise --n-samples / --tri-max-iter" if st == "WARN" else "")

    # 12. sensitivity
    sens = run.grep_all(r"scale\s+([\d.]+)\s+sigma =\s+\d+ ppm\s+FPP = ([\d.eE+-]+)\s+NFPP = ([\d.eE+-]+)")
    if not sens:
        A.test("Sensitivity", "Does the verdict survive a larger sigma?", "NA",
               "run without --sensitivity", run.log_name)
    else:
        rows = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m, _ in sens]
        ok = {sc: (f < 0.015 and n < 0.001) for sc, f, n in rows}
        ev = "; ".join(f"x{sc:g}: FPP {f:.3g}" + (" (passes)" if ok[sc] else "")
                       for sc, f, n in rows)
        adv = ""
        if len(set(ok.values())) > 1:
            passing = sorted(sc for sc in ok if ok[sc])
            nominal = ok.get(1.0)
            if nominal is False and passing and max(passing) < 1:
                ev += ("; the verdict flips: the Giacalone criteria are met only when the noise "
                       "is assumed SMALLER than measured (x" + ", x".join(f"{p:g}" for p in passing) + ")")
                adv = ("not a validation: a pass that needs an optimistic sigma is not robust; "
                       "check that the nominal sigma matches the out-of-transit scatter (chi2_red ~ 1)")
            elif nominal is True:
                ev += "; the verdict flips: the nominal pass is lost at a larger sigma"
                adv = "the validation is fragile: report it with the sensitivity table"
            else:
                ev += "; the verdict changes with sigma"
                adv = "report the sensitivity table with the FPP"
        A.test("Sensitivity", "Does the verdict survive a different sigma?",
               "PASS" if len(set(ok.values())) == 1 else "WARN", ev,
               run.src(sens[0][1]), adv)

    # 13. ground data
    G = run.S("ground")
    if not G and not run.ground:
        A.test("Ground data", "Do ground and TESS agree?", "NA", "no ground light curves",
               f"ground_clearance_{run.tic}.csv")
    else:
        G = G or {}
        tstat = G.get("target_status", "untested")
        verd = G.get("verdict", {})
        conflict = [k for k, v in verd.items() if v == "conflict"]
        chrom = [h for h, c in (G.get("chromaticity") or {}).items() if c.get("chromatic")]
        if tstat == "excluded":
            st, ev = "FAIL", "the ground data exclude the target as the host"
        elif conflict or chrom:
            st = "WARN"
            ev = "; ".join(([f"conflicting results for {conflict}"] if conflict else [])
                           + ([f"chromatic depth for host(s) {chrom}"] if chrom else []))
        elif tstat == "consistent":
            st = "PASS"
            ev = f"target consistent; cleared: {G.get('cleared', [])}"
        else:
            st, ev = "NA", f"target {tstat}"
        A.test("Ground data", "Do ground and TESS agree?", st, ev, f"{sj}:ground")

    # 14. provenance
    mi, lni = run.grep(r"KNOWN ephemeris.*depth = ([\d.]+) ppm")
    ma, lna = run.grep(r"adopted for TRICERATOPS: depth = ([\d.]+) ppm")
    mw, lnw = run.grep(r"depth passed to TRICERATOPS \(([\d.]+) ppm\) differs")
    if mw:
        A.test("Provenance", "Are reference values kept apart from measurements?", "FAIL",
               f"depth passed to TRICERATOPS ({mw.group(1)} ppm) differs from the light curve",
               run.src(lnw), "give the depth on the scale of the light curve")
    elif mi and ma:
        di, da = float(mi.group(1)), float(ma.group(1))
        same = abs(di - da) / max(di, da) < 1e-3
        far = abs(di - da) / max(di, da) > T["input_depth_tol"]
        A.test("Provenance", "Are reference values kept apart from measurements?", "PASS",
               (f"input depth {di:.0f} ppm (reference) replaced by the measured {da:.0f} ppm"
                if not same else f"measured depth {da:.0f} ppm equals the input")
               + ("; input was far off" if far and not same else ""), run.src(lna))
    else:
        A.test("Provenance", "Are reference values kept apart from measurements?", "NA",
               "depths not found in the log", run.log_name)


# --------------------------------------------------------------------------
# Independent recomputations from the saved light curves
# --------------------------------------------------------------------------
def _running_median(t, f, window, mask=None):
    """Median of f within +-window/2 (in t), evaluated at every point;
    points where mask is True are left out of the medians."""
    out = np.empty_like(f)
    use = np.ones(f.size, bool) if mask is None else ~mask
    j0 = j1 = 0
    n = t.size
    for i in range(n):
        while t[j0] < t[i] - window / 2:
            j0 += 1
        while j1 < n and t[j1] <= t[i] + window / 2:
            j1 += 1
        seg = f[j0:j1][use[j0:j1]]
        out[i] = np.median(seg) if seg.size else np.median(f[j0:j1])
    return out


def _load_corrected(run: Run):
    out = {}
    for p in sorted(glob.glob(run.path(f"lc_corrected_{run.tic}_s*.txt"))):
        m = re.search(r"_s(\d+)\.txt$", p)
        try:
            a = np.loadtxt(p)
        except Exception:
            continue
        if a.ndim == 2 and a.shape[0] > 50:
            ok = np.all(np.isfinite(a[:, :2]), axis=1)
            out[int(m.group(1))] = a[ok]
    return out


def _ephemeris(run: Run):
    P, T0 = run.S("P_d"), run.S("T0_BTJD")
    T14 = (run.S("ld_fit", "T14_h") or run.S("T14_h"))
    depth = run.S("ld_fit", "depth_ppm") or run.S("depth_ppm_adopted")
    if P is None:
        m, _ = run.grep(r"Adopted ephemeris: P = ([\d.]+) d, T0 = ([\d.]+), T14 = ([\d.]+) h, "
                        r"depth = ([\d.]+) ppm")
        if m:
            P, T0, T14, depth = (float(m.group(i)) for i in (1, 2, 3, 4))
    if None in (P, T0, T14):
        return None
    return float(P), float(T0), float(T14) / 24.0, (float(depth) if depth else None)


def recompute(run: Run, A: Audit, T: dict):
    eph = _ephemeris(run)
    lcs = _load_corrected(run)
    if eph is None or not lcs:
        A.test("Recomputation", "Do independent fits recover the signal?", "NA",
               "no ephemeris or no lc_corrected_*.txt files", run.outdir)
        return
    P, T0, dur, depth_pipe = eph
    t_all, f_all, m_all, s_all, e_all = [], [], [], [], []
    win_bls = max(5 * dur, 0.75)
    for s, a in lcs.items():
        t, f = a[:, 0], a[:, 1]
        ph = (t - T0 + 0.5 * P) % P - 0.5 * P
        # (a) for the BLS: running median over 5 x T14, transits NOT masked
        #     (no knowledge of the ephemeris enters the search)
        f_bls = f / _running_median(t, f, win_bls)
        # (b) for the sector depths: 3 x T14, transits masked
        f_msk = f / _running_median(t, f, max(3 * dur, 0.5), np.abs(ph) < 0.75 * dur)
        # per-sector robust scatter; clip > 5 sigma upward, and downward only
        # isolated points (a transit is a run of low points, a cosmic ray or a
        # bad cadence is single), so deep transits survive the clip
        r = f_bls - 1
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        low = r < -5 * sig
        nb = np.zeros_like(low)
        nb[1:] |= r[:-1] < -3 * sig
        nb[:-1] |= r[1:] < -3 * sig
        keep = (r < 5 * sig) & ~(low & ~nb)
        t_all.append(t[keep])
        f_all.append(f_bls[keep])
        m_all.append(f_msk[keep])
        e_all.append(np.full(keep.sum(), sig))
        s_all.append(np.full(keep.sum(), s))
    t = np.concatenate(t_all)
    f_b = np.concatenate(f_all)
    f = np.concatenate(m_all)
    e_b = np.concatenate(e_all)
    sec = np.concatenate(s_all)
    o = np.argsort(t)
    t, f, f_b, e_b, sec = t[o], f[o], f_b[o], e_b[o], sec[o]
    res = {}

    # (a) BLS near P with the audit's own detrending.  The peak must rest on
    #     >= 3 transits with data (sparse multi-year data make aliases that
    #     rest on two events), and the signal at P is compared with the same
    #     box placed at random phases (an empirical false-alarm level).
    try:
        from astropy.timeseries import BoxLeastSquares
        good = np.isfinite(f_b)
        tb, fb, eb = t[good], f_b[good], e_b[good]
        bls = BoxLeastSquares(tb, fb, dy=eb)
        base = float(np.ptp(tb))
        step = 0.25 * dur * P / base
        n = int(min(max(2 * 0.004 * P / step, 200), 20000))
        periods = np.linspace(P * (1 - 0.004), P * (1 + 0.004), n)
        r = bls.power(periods, [0.8 * dur, dur, 1.2 * dur])

        def n_events(Pc, t0c, dc):
            st = bls.compute_stats(Pc, dc, t0c)
            return int(np.sum(np.asarray(st["per_transit_count"]) >= 5))

        # transits with data at the pipeline ephemeris; a peak that rests on
        # fewer of them is an alias that ignores observed events
        n_at_P = n_events(P, T0, dur)
        need = max(3, n_at_P - 1)
        order = np.argsort(r.power)[::-1]
        best, skipped = None, 0
        for i in order[:400]:
            ne = n_events(r.period[i], r.transit_time[i], r.duration[i])
            if ne >= need:
                best = (i, ne)
                break
            skipped += 1
        # signal at the pipeline ephemeris vs random phases
        def box_snr(t0c):
            ph = (tb - t0c + 0.5 * P) % P - 0.5 * P
            inn = np.abs(ph) < 0.5 * dur
            oot = (np.abs(ph) > 0.75 * dur) & (np.abs(ph) < 3 * dur)
            if inn.sum() < 5 or oot.sum() < 20:
                return np.nan
            d = np.median(fb[oot]) - np.mean(fb[inn])
            return d / (np.std(fb[oot]) / np.sqrt(inn.sum()))
        snr_p = box_snr(T0)
        rng = np.random.default_rng(1)
        offs = rng.uniform(2 * dur, P - 2 * dur, 300)
        null = np.array([box_snr(T0 + o) for o in offs])
        null = null[np.isfinite(null)]
        p_emp = float((np.sum(null >= snr_p) + 1) / (null.size + 1)) if null.size else np.nan
        res["bls"] = dict(snr_at_P=float(snr_p), null_p=p_emp,
                          null_max=float(np.max(null)) if null.size else None)
        if best is None:
            A.test("Recomputation: BLS", "Does an independent box search find the same signal?",
                   "WARN", f"no BLS peak within 0.4 % of P rests on >= {need} transits; box S/N at the "
                   f"pipeline ephemeris {snr_p:.1f}", f"lc_corrected_{run.tic}_s*.txt")
        else:
            i, ne = best
            Pb, db = float(r.period[i]), float(r.depth[i] * 1e6)
            dp = abs(Pb - P) / P
            res["bls"].update(period=Pb, depth_ppm=db, n_transits=ne, n_at_P=n_at_P,
                              top_period=float(r.period[order[0]]),
                              aliases_skipped=skipped)
            # BLS power at the pipeline period (best T0/duration) relative to the peak:
            # with only a few transits several nearby periods fit almost equally well
            rP = bls.power(np.array([P]), [0.8 * dur, dur, 1.2 * dur])
            prel = float(rP.power[0] / r.power[i]) if r.power[i] > 0 else np.nan
            res["bls"]["power_ratio_at_P"] = prel
            # A small fractional period difference can still put the two
            # ephemerides days apart over a long baseline: compare the
            # predicted mid-times at the pipeline's epochs with data, with
            # the BLS epoch numbering anchored to the pipeline's.
            Tb = float(r.transit_time[i])
            k0 = round((Tb - T0) / P)

            def t_bls(e):
                return Tb + (np.asarray(e, float) - k0) * Pb

            st_p = bls.compute_stats(P, dur, T0)
            tt_p = np.asarray(st_p["transit_times"], float)
            ep_p = np.round((tt_p[np.asarray(st_p["per_transit_count"]) >= 5] - T0) / P)
            drift = float(np.max(np.abs(t_bls(ep_p) - (T0 + ep_p * P)))) if ep_p.size else np.nan
            same = np.isfinite(drift) and drift < T["bls_drift_tol"] * dur
            res["bls"]["drift_h"] = drift * 24.0
            ok = p_emp < 0.01 and (same or prel >= 0.8)
            if p_emp >= 0.01:
                adv = "the signal is not recovered above the noise with independent detrending"
            else:
                # timing difference the two periods predict for a transit now
                import time as _time
                tzero = 2454833.0 if str(run.S("mission") or "").lower() == "kepler" else 2457000.0
                btjd_now = _time.time() / 86400.0 + 2440587.5 - tzero
                n_now = round((btjd_now - T0) / P)
                t_pipe = T0 + n_now * P
                dt_h = abs(float(t_bls(n_now)) - t_pipe) * 24.0
                res["bls"]["timing_difference_now_h"] = dt_h
                adv = (f"period ambiguity in the {'Kepler' if tzero < 2456000 else 'TESS'} data: "
                       f"at epoch {n_now} ({'BKJD' if tzero < 2456000 else 'BTJD'} "
                       f"{t_pipe:.2f}) the two ephemerides predict transit times {dt_h:.1f} h apart - a ground "
                       f"transit at the predicted time settles it")
            A.test("Recomputation: BLS", "Does an independent box search find the same signal?",
                   "PASS" if ok else "WARN",
                   f"own detrending (running median {win_bls * 24:.0f} h, transits not masked): "
                   f"best peak with data in >= {need} of the {n_at_P} transit windows at P = {Pb:.5f} d "
                   f"({dp * 100:.3f} % from {P:.5f}: the two ephemerides drift apart by up to "
                   f"{drift * 24:.1f} h = {drift / dur:.2f} T14 over the data; {skipped} stronger "
                   f"alias(es) with data in fewer windows "
                   f"skipped, top peak {float(r.period[order[0]]):.5f} d), box depth {db:.0f} ppm; box S/N at the pipeline ephemeris "
                   f"{snr_p:.1f}, at 300 random phases max {np.max(null):.1f} "
                   f"(p = {p_emp:.3g}); BLS power at P is {min(prel, 1) * 100:.0f} % of the peak",
                   f"lc_corrected_{run.tic}_s*.txt",
                   "" if ok else adv)
    except Exception as exc:
        A.test("Recomputation: BLS", "Does an independent box search find the same signal?",
               "NA", f"BLS not run ({type(exc).__name__}: {exc})", "")

    # (b) per-sector box depth / g_s
    gmap = {}
    for m, _ in run.grep_all(r"sector\s+(\d+): (?:optimal|percentile)\s+\d+ px\s+g =\s+([\d.]+)"):
        gmap[int(m.group(1))] = float(m.group(2))
    ph = (t - T0 + 0.5 * P) % P - 0.5 * P
    inn = np.abs(ph) < 0.35 * dur
    oot = (np.abs(ph) > 0.75 * dur) & (np.abs(ph) < 2.5 * dur)
    rows = []
    for s in sorted(set(sec.astype(int))):
        k = sec == s
        if (inn & k).sum() < 3 or (oot & k).sum() < 10:
            continue
        fi, fo = f[inn & k], f[oot & k]
        dpt = np.median(fo) - np.mean(fi)
        e = np.std(fo) / np.sqrt(fi.size) * 1.25
        g = gmap.get(int(s))
        rows.append(dict(sector=int(s), n_in=int(fi.size), depth_ppm=float(dpt * 1e6),
                         err_ppm=float(e * 1e6), g=g,
                         on_target_ppm=float(dpt / g * 1e6) if g else None))
    res["sector_depths"] = rows
    if len(rows) >= 2:
        d = np.array([r["depth_ppm"] for r in rows])
        e = np.array([r["err_ppm"] for r in rows])
        w = 1 / e ** 2
        mu = np.sum(w * d) / np.sum(w)
        chi = float(np.sum(w * (d - mu) ** 2) / (len(rows) - 1))
        st = "WARN" if chi > T["sector_depth_chi2_warn"] else "PASS"
        worst = rows[int(np.argmax(np.abs(d - mu) / e))]
        A.test("Recomputation: sectors", "Do the sectors give the same depth?", st,
               f"{len(rows)} sectors, mean box depth {mu:.0f} ppm, chi2/dof {chi:.2f}; "
               f"most deviant sector {worst['sector']} ({worst['depth_ppm']:.0f} +/- "
               f"{worst['err_ppm']:.0f} ppm)", f"lc_corrected_{run.tic}_s*.txt")
    else:
        A.test("Recomputation: sectors", "Do the sectors give the same depth?", "NA",
               "fewer than two sectors with in-transit data", "")
    res["_series"] = (t, f, P, T0, dur)
    A.recomputations = res


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def overall(A: Audit):
    c = A.counts()
    if c["FAIL"]:
        return f"{c['FAIL']} FAIL, {c['WARN']} WARN: the claim does not survive all tests"
    if c["WARN"]:
        return f"no FAIL, {c['WARN']} WARN: the claim survives, with the risks listed"
    return "all available tests passed"


def to_json(A: Audit, path):
    d = asdict(A)
    rc = dict(A.recomputations)
    rc.pop("_series", None)
    d["recomputations"] = rc
    d["counts"] = A.counts()
    d["overall"] = overall(A)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, default=str)


def to_text(A: Audit) -> str:
    L = [f"AUDIT of TIC {A.tic}  (triceratops_audit.py v{A.audit_version}; "
         f"pipeline v{A.pipeline_version or '?'})", f"folder: {A.outdir}",
         f"pipeline verdict (unchanged): {A.pipeline_verdict or 'n/a'}",
         f"audit: {overall(A)}", "", "EVALUATION - what the run claims"]
    for c in A.claims:
        L.append(f"  {c.item:42s} {str(c.value)} {c.unit}".rstrip()
                 + f"   [{c.kind}; {c.source}]")
    L += ["", "SKEPTIC - tests that try to break the claim"]
    for t in sorted(A.tests, key=lambda t: STATUS_ORDER.get(t.status, 9)):
        L.append(f"  {t.status:4s}  {t.area:24s} {t.evidence}")
        if t.source:
            L.append(f"        {'':24s} source: {t.source}")
        if t.advice:
            L.append(f"        {'':24s} -> {t.advice}")
    return "\n".join(L)


def to_pdf(A: Audit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap
    col = {"PASS": "#2e7d32", "WARN": "#e08a00", "FAIL": "#c62828", "NA": "#777777"}
    with PdfPages(path) as pdf:
        # page 1: header + skeptic table
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.965
        fig.text(0.06, y, f"Audit of TIC {A.tic}", fontsize=16, weight="bold")
        y -= 0.025
        fig.text(0.06, y, f"triceratops_audit.py v{A.audit_version} (prototype) - pipeline "
                          f"v{A.pipeline_version or '?'} - deterministic checks, no language model",
                 fontsize=8, color="0.35")
        y -= 0.03
        fig.text(0.06, y, f"Pipeline verdict (unchanged): {A.pipeline_verdict or 'n/a'}", fontsize=10)
        y -= 0.022
        c = A.counts()
        fig.text(0.06, y, f"Audit: {overall(A)}", fontsize=10, weight="bold",
                 color=col["FAIL"] if c["FAIL"] else col["WARN"] if c["WARN"] else col["PASS"])
        y -= 0.02
        fig.text(0.06, y, "  ".join(f"{k} {v}" for k, v in c.items()), fontsize=8.5, color="0.3")
        y -= 0.035
        fig.text(0.06, y, "Skeptic - tests that try to break the claim", fontsize=11.5, weight="bold")
        y -= 0.02
        for t in sorted(A.tests, key=lambda t: STATUS_ORDER.get(t.status, 9)):
            lines = textwrap.wrap(t.evidence, 92)
            adv = textwrap.wrap("-> " + t.advice, 92) if t.advice else []
            src = textwrap.wrap("source: " + t.source, 100) if t.source else []
            need = 0.0145 * (len(lines) + len(adv) + len(src)) + 0.012
            if y - need < 0.04:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.96
            fig.text(0.06, y, t.status, fontsize=8.5, weight="bold", color=col.get(t.status, "k"))
            fig.text(0.13, y, t.area, fontsize=8.5, weight="bold")
            yy = y
            for ln in lines:
                fig.text(0.34, yy, ln, fontsize=7.6)
                yy -= 0.0145
            for ln in adv:
                fig.text(0.34, yy, ln, fontsize=7.3, color="#1a4f8a")
                yy -= 0.0145
            for ln in src:
                fig.text(0.34, yy, ln, fontsize=6.6, color="0.45")
                yy -= 0.0145
            y = yy - 0.012
        pdf.savefig(fig)
        plt.close(fig)
        # page: evaluation table
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.96
        fig.text(0.06, y, "Evaluation - what the run claims (with sources)", fontsize=11.5,
                 weight="bold")
        y -= 0.03
        for cl in A.claims:
            val = textwrap.wrap(f"{cl.value} {cl.unit}".strip(), 60)
            if y - 0.0145 * (len(val) + 1) < 0.04:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.96
            fig.text(0.06, y, cl.item, fontsize=7.6, weight="bold")
            yy = y
            for ln in val:
                fig.text(0.38, yy, ln, fontsize=7.6)
                yy -= 0.0145
            fig.text(0.38, yy, f"{cl.kind}  |  {cl.source}", fontsize=6.4, color="0.45")
            y = yy - 0.02
        pdf.savefig(fig)
        plt.close(fig)
        # page: recomputations
        R = A.recomputations
        if R.get("_series"):
            t, f, P, T0, dur = R["_series"]
            fig, axs = plt.subplots(2, 1, figsize=(8.27, 8.5))
            ph = ((t - T0 + 0.5 * P) % P - 0.5 * P) * 24
            k = np.abs(ph) < 3 * dur * 24
            axs[0].plot(ph[k], f[k], ".", ms=1, color="0.6")
            if k.sum() > 30:
                edges = np.arange(ph[k].min(), ph[k].max() + 0.25, 0.25)
                idx = np.digitize(ph[k], edges)
                xb = [ph[k][idx == i].mean() for i in np.unique(idx) if (idx == i).sum() > 3]
                yb = [np.median(f[k][idx == i]) for i in np.unique(idx) if (idx == i).sum() > 3]
                axs[0].plot(xb, yb, "o", ms=3, color="k")
            axs[0].set_xlabel("hours from mid-transit (pipeline ephemeris)")
            axs[0].set_ylabel("flux (audit's own detrending)")
            axs[0].set_title("Fold of the lc_corrected files with the audit's own detrending "
                             "(running median 3 x T14, transits masked)", fontsize=9)
            rows = R.get("sector_depths", [])
            if rows:
                axs[1].errorbar([r["sector"] for r in rows], [r["depth_ppm"] for r in rows],
                                yerr=[r["err_ppm"] for r in rows], fmt="o", capsize=3)
                axs[1].set_xlabel("sector")
                axs[1].set_ylabel("box depth in the aperture [ppm]")
                axs[1].set_title("Per-sector box depths (audit recomputation)", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def run_audit(outdir, tic=None, recompute_lc=True, thresholds=None):
    T = dict(THRESHOLDS)
    if thresholds:
        T.update(thresholds)
    run = Run(outdir, tic)
    A = Audit(tic=run.tic, outdir=run.outdir, thresholds=T)
    if not run.log and not run.summary:
        raise SystemExit(f"no summary or log for TIC {run.tic} in {run.outdir}")
    evaluate(run, A)
    skeptic(run, A, T)
    if recompute_lc:
        recompute(run, A, T)
    return A


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir", help="output folder of a triceratops_validation.py run")
    ap.add_argument("--tic", type=int, default=None)
    ap.add_argument("--no-recompute", action="store_true",
                    help="skip the recomputations from the saved light curves")
    ap.add_argument("--thresholds", default=None, help="JSON file overriding THRESHOLDS")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    thr = None
    if a.thresholds:
        with open(a.thresholds, encoding="utf-8") as fh:
            thr = json.load(fh)
    A = run_audit(a.outdir, a.tic, not a.no_recompute, thr)
    base = os.path.join(A.outdir, f"audit_{A.tic}")
    to_json(A, base + ".json")
    txt = to_text(A)
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    if not a.no_pdf:
        try:
            to_pdf(A, base + ".pdf")
            print(f"\nwritten: {base}.json, .txt, .pdf")
        except Exception as exc:
            print(f"\n! PDF not written ({exc}); written: {base}.json, .txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
