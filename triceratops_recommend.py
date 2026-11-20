#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommend the next observations from the TRICERATOPS scenario probabilities.

Used by triceratops_validation.py (after calc_probs; written to the log,
summary_<TIC>.json and recommendations_<TIC>.txt) and by triceratops_audit.py
(section "Next observations").  Pure numpy, no network, no language model.

Idea
----
FPP is the summed probability of the false-positive scenarios; each kind of
follow-up observation can remove a known subset of them:

  ground photometry, separate apertures   NTP, NEB, NEBx2P on stars the
  (seeing-limited, TFOP SG1 / AAVSO)      seeing resolves
  high-resolution imaging (speckle / AO)  bound companions (PEB, PEBx2P, STP,
                                          SEB, SEBx2P) and unresolved
                                          background stars (DEB, DEBx2P, BTP,
                                          BEB, BEBx2P) outside its inner
                                          working angle; contrast curve ->
                                          --contrast-curve
  multi-band transits (chromaticity)      scenarios in which the eclipsed /
                                          transited star has another colour
                                          than the light that dilutes it
                                          (STP, SEB, SEBx2P, background EBs);
                                          estimated from the best-fit
                                          companion of the last run
  radial velocities (~1 km/s)             EB, EBx2P on the target (stellar
                                          mass companion: K of km/s)

For every observation the module gives the probability it addresses, the FPP
and NFPP *if* those scenarios were ruled out completely (a best case:
imaging cannot see companions inside its inner working angle, a chromatic
test needs enough precision), whether that alone would meet the criteria
of Giacalone et al. 2021 (FPP < 0.015, NFPP < 0.001), and the shortest
combination that would.  These are upper limits on what each observation
can achieve, not predictions of its outcome: if the observation finds the
signal on another star, the candidate is a false positive.
"""
from __future__ import annotations

import math

FPP_LIMIT, NFPP_LIMIT = 0.015, 0.001

PLANET = ("TP", "PTP", "DTP")
NEARBY = ("NTP", "NEB", "NEBx2P")
BOUND = ("PEB", "PEBx2P", "STP", "SEB", "SEBx2P")
BACKGROUND = ("DEB", "DEBx2P", "BTP", "BEB", "BEBx2P")
TARGET_EB = ("EB", "EBx2P")
# scenarios whose eclipsed / transited star is not the target: their depth
# changes with the band when its colour differs from the diluting light
CHROMATIC = ("STP", "SEB", "SEBx2P", "PEB", "PEBx2P", "BTP", "BEB", "BEBx2P",
             "DEB", "DEBx2P")

BANDS = {"B": 0.44, "V": 0.55, "g'": 0.48, "r'": 0.62, "i'": 0.76,
         "z'": 0.91, "TESS": 0.80, "Kepler": 0.64,
         # pipeline band names (canonical_band) and other filters
         "g": 0.48, "r": 0.62, "i": 0.76, "z": 0.91, "R": 0.64, "I": 0.79,
         "J": 1.25, "H": 1.65, "K": 2.2, "CBB": 0.70, "CV": 0.60}


def _planck(T, lam_um):
    """Relative blackbody intensity (only ratios are used)."""
    if not (T and T > 0):
        return float("nan")
    x = 14387.77 / (lam_um * T)
    if x > 700:
        return 0.0
    return 1.0 / (lam_um ** 5 * (math.exp(x) - 1.0))


def teff_from_mass(m):
    """Rough main-sequence Teff [K] from mass [Msun] (L ~ M^4, R ~ M^0.8)."""
    if not (m and m > 0):
        return float("nan")
    return 5772.0 * min(max(m, 0.08), 3.0) ** 0.6


def band_fraction(fr_ref, t_host, t_other, band, ref_band="TESS"):
    """Fraction of the light from the 'host' star in `band`, given its
    fraction fr_ref in ref_band and blackbody colours of host and the
    diluting light (t_other)."""
    if not (0 < fr_ref < 1):
        return fr_ref
    lr, lb = BANDS[ref_band], BANDS[band]
    ratio_ref = fr_ref / (1.0 - fr_ref)                    # F_host / F_other
    ratio_b = ratio_ref * (_planck(t_host, lb) / _planck(t_host, lr)) \
        * (_planck(t_other, lr) / _planck(t_other, lb))
    return ratio_b / (1.0 + ratio_b)


def _fpp_after(rows, removed):
    """FPP, NFPP after removing scenarios.  `removed` is a set of row
    indices (removed completely) or a dict index -> fraction removed."""
    w = removed if isinstance(removed, dict) else {i: 1.0 for i in removed}
    pr = [r["prob"] * (1.0 - min(max(w.get(i, 0.0), 0.0), 1.0))
          for i, r in enumerate(rows)]
    tot = sum(pr)
    if tot <= 0:
        return float("nan"), float("nan")
    fp = sum(p for p, r in zip(pr, rows) if r["scenario"] not in PLANET) / tot
    nf = sum(p for p, r in zip(pr, rows) if r["scenario"] in NEARBY) / tot
    return fp, nf


def _merge(a: dict, b: dict) -> dict:
    """Combine two removal maps: fractions removed independently."""
    out = dict(a)
    for k, v in b.items():
        out[k] = 1.0 - (1.0 - out.get(k, 0.0)) * (1.0 - v)
    return out


def bound_fraction_outside(sep_arcsec, dist_pc, a_peak_au=50.0, sig_dex=1.52):
    """Fraction of bound stellar companions of a Sun-like star wider than a
    projected separation (log-normal in a: Raghavan et al. 2010, peak ~50 au,
    1.5 dex).  Typical value, not a measurement."""
    if not (dist_pc and dist_pc > 0):
        return 0.5
    a = sep_arcsec * dist_pc
    z = (math.log10(a) - math.log10(a_peak_au)) / sig_dex
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chromatic_fraction(scenario, m_target, band_pair=("g'", "z'"), min_change=0.2,
                       n=181, depth=None, ref_band="TESS"):
    """Fraction of the possible bound companions (mass ratio q uniform in
    0.1 - 1, Raghavan et al. 2010) for which `scenario` would change the
    transit depth between the two bands by more than min_change (relative),
    with main-sequence colours (Teff ~ M^0.6, R ~ M^0.8, blackbodies).
    Companions too faint to produce the observed `depth` (fraction) are
    left out: for a transit on the companion (STP) the planet can be at
    most ~2 Jupiter radii (0.2 Rsun), for an eclipse (SEB) at most 100 %.
    Returns (fraction, list of (q, ratio)) over the possible companions."""
    b1, b2 = band_pair
    if not (m_target and m_target > 0):
        m_target = 1.0
    t_t, r_t = teff_from_mass(m_target), m_target ** 0.8
    out, hit, n_ok = [], 0, 0
    for k in range(n):
        q = 0.1 + 0.9 * k / (n - 1)
        m_c = q * m_target
        t_c, r_c = teff_from_mass(m_c), m_c ** 0.8

        def share_c(band):
            fc = r_c ** 2 * _planck(t_c, BANDS[band])
            ft = r_t ** 2 * _planck(t_t, BANDS[band])
            return fc / (fc + ft)
        if scenario.startswith("S"):      # transit / eclipse on the companion
            if depth:
                dmax = 1.0 if "EB" in scenario else min(1.0, (0.2 / r_c) ** 2)
                if share_c(ref_band) * dmax < depth:
                    continue                  # too faint to make the dip
            ratio = share_c(b1) / share_c(b2)
        else:                             # on the target, diluted by it
            ratio = (1 - share_c(b1)) / (1 - share_c(b2))
        out.append((q, ratio))
        n_ok += 1
        hit += abs(ratio - 1) > min_change
    return (hit / n_ok if n_ok else 0.0), out


def scenario_band_ratio(row, teff_target, band_blue, band_red, ref_band="TESS"):
    """Predicted depth(band_red) / depth(band_blue) for one TRICERATOPS
    scenario row (best-fit companion), with blackbody colours.

    S*, B* scenarios: the dip is on the other star (companion or background
    star of mass M_s), diluted by the target; P*, D* scenarios: the dip is
    on the target, diluted by the other star (mass M_comp).  TP, EB, EBx2P,
    N*: no colour change from dilution (returns 1).  Returns (ratio,
    T_other) or (nan, nan) when the row lacks the needed values."""
    sc = row.get("scenario", "")
    fr = row.get("flux_ratio_comp_T")
    if sc in ("TP", "EB", "EBx2P") or sc in NEARBY:
        return 1.0, float("nan")
    if not (fr and 0 < fr < 1):
        return float("nan"), float("nan")
    if sc[0] in ("S", "B"):
        t_o = teff_from_mass(row.get("M_s"))
        f_b = band_fraction(fr, t_o, teff_target, band_blue, ref_band)
        f_r = band_fraction(fr, t_o, teff_target, band_red, ref_band)
    else:                                   # P*, D*: dip on the target
        t_o = teff_from_mass(row.get("M_comp"))
        f_b = 1 - band_fraction(fr, t_o, teff_target, band_blue, ref_band)
        f_r = 1 - band_fraction(fr, t_o, teff_target, band_red, ref_band)
    if not (f_b and f_b > 0):
        return float("nan"), t_o
    return f_r / f_b, t_o


def chromatic_verdicts(summary: dict, host=None, nsig=3.0, ld_tol=0.03):
    """What the measured multi-band depths say about each scenario.

    Uses summary['ground']['chromaticity'][host] (depth ratio reddest /
    bluest band with its error, from triceratops_validation.py) and the
    scenario table.  For every scenario with probability > 0 the predicted
    ratio of its best-fit companion is compared with the measurement.
    Returns a dict with the measurement, a list of per-scenario verdicts
    ('ruled out' / 'consistent' / 'no prediction'), the probability that
    the data rule out, and the FPP / NFPP if those scenarios were removed.
    A planet on the target predicts a ratio of 1 within limb darkening
    (ld_tol, a few %)."""
    G = (summary or {}).get("ground") or {}
    ch = G.get("chromaticity") or {}
    if not ch:
        return None
    tid = str(summary.get("tic", ""))
    h = str(host) if host else (tid if tid in ch else next(iter(ch)))
    c = ch.get(h)
    if not c:
        return None
    rb, e = c.get("red_over_blue"), c.get("e_red_over_blue")
    bb, br = c.get("blue_band"), c.get("red_band")
    out = dict(host=h, blue_band=bb, red_band=br, ratio=rb, e_ratio=e,
               p=c.get("p"), chromatic=c.get("chromatic"), rows=[],
               detect_limit=(nsig * e if e and e == e else float("nan")))
    if not (bb and br and rb == rb and e and e == e and e > 0):
        out["usable"] = False
        return out
    out["usable"] = True
    out["planet_nsigma"] = abs(rb - 1.0) / math.hypot(e, ld_tol)
    teff_t = summary.get("Teff") or 5772.0
    rows = summary.get("scenario_probs") or []
    removed = {}
    for i, r in enumerate(rows):
        if r.get("prob", 0) <= 0:
            continue
        sc = r["scenario"]
        pred, t_o = scenario_band_ratio(r, teff_t, bb if bb in BANDS else "V",
                                        br if br in BANDS else "I")
        if sc in PLANET or sc in TARGET_EB or sc in NEARBY:
            continue
        if pred != pred:
            out["rows"].append(dict(scenario=sc, prob=r["prob"], pred=None,
                                    verdict="no prediction"))
            continue
        ns = abs(pred - rb) / math.hypot(e, ld_tol)
        v = "ruled out" if ns > nsig else "consistent"
        if v == "ruled out":
            removed[i] = 1.0
        out["rows"].append(dict(scenario=sc, prob=r["prob"], pred=pred,
                                T_other=t_o, nsigma=ns, verdict=v))
    out["prob_ruled_out"] = sum(rows[i]["prob"] for i in removed)
    fpp, nfpp = _fpp_after(rows, removed) if rows else (float("nan"),) * 2
    out["fpp_after"], out["nfpp_after"] = fpp, nfpp
    return out


def recommend(summary: dict, seeing_fwhm=2.5, hires_iwa=0.1, rv_precision_kms=1.0,
              band_pair=("g'", "z'"), ref_band=None, chromatic_min=0.2):
    """Recommendations from a pipeline summary.

    summary needs "scenario_probs": [{ID, scenario, prob, ...}, ...] (pooled
    probabilities), and may have "neighbours" (ID, sep, Tmag, Teff,
    fluxratio, tdepth), "Teff", "distance_pc", "P_d", "T0_BTJD", "T14_h",
    "depth_ppm_adopted", "mission".
    Returns dict(fpp, nfpp, items=[...], plan=[...], text=[...]).
    """
    rows = [dict(r) for r in (summary.get("scenario_probs") or [])
            if r.get("prob") is not None]
    if not rows:
        return dict(items=[], plan=[], text=["no scenario probabilities: run "
                                             "TRICERATOPS first"])
    tot = sum(r["prob"] for r in rows) or 1.0
    for r in rows:
        r["prob"] = float(r["prob"]) / tot
    fpp0, nfpp0 = _fpp_after(rows, set())
    ref_band = ref_band or ("Kepler" if str(summary.get("mission", "")).lower()
                            == "kepler" else "TESS")
    teff_t = float(summary.get("Teff") or float("nan"))
    dist = summary.get("distance_pc")
    tid = rows[0].get("ID")
    neigh = {int(n["ID"]): n for n in (summary.get("neighbours") or [])
             if n.get("ID") is not None}
    items = []

    def add(key, title, idx, detail, how, typical=None):
        """typical: dict index -> fraction this observation usually removes
        (default: all of it)."""
        if not idx:
            return
        mass = sum(rows[i]["prob"] for i in idx)
        fp, nf = _fpp_after(rows, set(idx))
        typ = typical if typical is not None else {i: 1.0 for i in idx}
        fpt, nft = _fpp_after(rows, typ)
        items.append(dict(key=key, title=title, removes=sorted(idx), typical=typ,
                          fpp_typical=fpt, nfpp_typical=nft,
                          scenarios=sorted({f"{rows[i]['scenario']}"
                                            + ("" if rows[i].get("ID") == tid
                                               else f" (TIC {rows[i].get('ID')})")
                                            for i in idx}),
                          prob_addressed=mass, fpp_after=fp, nfpp_after=nf,
                          meets=bool(fp < FPP_LIMIT and nf < NFPP_LIMIT),
                          detail=detail, how=how))

    # 1. ground photometry with a separate aperture on each resolved neighbour
    idx_res, idx_unres, lines = [], [], []
    for i, r in enumerate(rows):
        if r["scenario"] not in NEARBY:
            continue
        n = neigh.get(int(r["ID"])) if r.get("ID") is not None else None
        sep = float(n["sep"]) if n and n.get("sep") is not None else float(r.get("sep", float("nan")))
        (idx_res if (sep == sep and sep >= seeing_fwhm) else idx_unres).append(i)
    for sid in sorted({int(rows[i]["ID"]) for i in idx_res + idx_unres}):
        n = neigh.get(sid, {})
        p = sum(rows[i]["prob"] for i in idx_res + idx_unres if int(rows[i]["ID"]) == sid)
        sep = n.get("sep")
        td = n.get("tdepth")
        lines.append(f"TIC {sid}: {sep:.1f}\" , Tmag {n.get('Tmag', float('nan')):.2f}, "
                     f"P(signal on it) = {p:.3f}"
                     + (f", it would have to dim by {td * 1e6:.0f} ppm" if td else "")
                     + ("" if sep is None or sep >= seeing_fwhm else
                        f" - closer than the seeing ({seeing_fwhm:g}\"): "
                        f"needs high-resolution imaging"))
    tgt_td = None
    for n in neigh.values():
        if n.get("ID") == tid:
            tgt_td = n.get("tdepth")
    add("ground", "Ground photometry, separate apertures on the target and "
        "each nearby star", idx_res,
        lines, (f"one transit, seeing <= {seeing_fwhm:g}\"; the star that shows "
                f"the dip is the host"
                + (f" (on the target: {tgt_td * 1e6:.0f} ppm)" if tgt_td else "")
                + "; feed the light curves back with --ground-lc"))

    # 2. high-resolution imaging
    idx_hr = [i for i, r in enumerate(rows) if r["scenario"] in BOUND + BACKGROUND]
    idx_hr += idx_unres
    det = []
    for i in idx_hr:
        r = rows[i]
        fr = r.get("flux_ratio_comp_T")
        if r["scenario"] in BOUND and fr and fr > 0:
            dmag = -2.5 * math.log10(fr / max(1e-9, 1 - fr)) if fr < 1 else 0.0
            mc = r.get("M_s") if r["scenario"].startswith("S") else r.get("M_comp")
            det.append(f"{r['scenario']}: best-fit companion {mc if mc else float('nan'):.2f} Msun, "
                       f"{fr * 100:.0f} % of the {ref_band} light (contrast ~{abs(dmag):.1f} mag)")
    iwa_au = (f" = {hires_iwa * float(dist):.0f} au at {float(dist):.0f} pc"
              if dist else "")
    f_out = bound_fraction_outside(hires_iwa, float(dist) if dist else None)
    typ_hr = {i: (f_out if rows[i]["scenario"] in BOUND else 0.95) for i in idx_hr}
    det.append(f"typically {f_out * 100:.0f} % of bound companions lie outside "
               f"{hires_iwa:g}\" (separation distribution of Raghavan et al. 2010); "
               f"background stars: ~95 %")
    add("hires", "High-resolution imaging (speckle / adaptive optics)", idx_hr, det,
        (f"rules out companions and background stars outside ~{hires_iwa:g}\"{iwa_au} "
         f"down to its contrast limit; closer companions remain - best case shown; "
         f"give the contrast curve to the pipeline with --contrast-curve"),
        typical=typ_hr)

    # 3. multi-band transits.  A planet on the target has the same depth in
    #    every band; a transit / eclipse on a star of another colour, or one
    #    diluted by a star of another colour, does not.  How much the depth
    #    changes depends on the companion, which the data barely constrain:
    #    the fraction of all possible bound companions (mass ratio 0.1 - 1)
    #    that would change the depth by more than chromatic_min is used as
    #    the typical fraction removed; the best-fit companion of the last
    #    run is shown as an example.
    b1, b2 = band_pair
    idx_mb, typ_mb, det_mb = [], {}, []
    m_t = summary.get("M_star")
    d_tgt = summary.get("depth_on_target_ppm")
    for i, r in enumerate(rows):
        if r["scenario"] not in CHROMATIC or r["prob"] <= 0:
            continue
        sc = r["scenario"]
        if sc in ("STP", "SEB", "SEBx2P", "PEB", "PEBx2P"):
            frac, _ = chromatic_fraction(sc, m_t, band_pair, chromatic_min,
                                         depth=(float(d_tgt) * 1e-6 if d_tgt else None),
                                         ref_band=ref_band)
            idx_mb.append(i)
            typ_mb[i] = frac
            fr = r.get("flux_ratio_comp_T")
            ex = ""
            if fr and 0 < fr < 1:
                if sc.startswith("P"):
                    t_c = teff_from_mass(r.get("M_comp"))
                    f1 = 1 - band_fraction(fr, t_c, teff_t, b1, ref_band)
                    f2 = 1 - band_fraction(fr, t_c, teff_t, b2, ref_band)
                else:
                    t_c = teff_from_mass(r.get("M_s"))
                    f1 = band_fraction(fr, t_c, teff_t, b1, ref_band)
                    f2 = band_fraction(fr, t_c, teff_t, b2, ref_band)
                ratio = f1 / f2 if f2 else float("nan")
                ex = (f"; example (best-fit companion of the last run, ~{t_c:.0f} K): "
                      f"depth({b1}) / depth({b2}) = {ratio:.2f}")
                if d_tgt and ratio == ratio and abs(ratio - 1) > 0.02:
                    dd = abs(ratio - 1) / (1 + ratio) * 2 * float(d_tgt)
                    ex += (f", ~{dd:.0f} ppm apart -> needs a depth error of "
                           f"<= {dd / 3:.0f} ppm per band for 3 sigma")
            det_mb.append(f"{sc} (P = {r['prob']:.4f}): {frac * 100:.0f} % of the companions "
                          f"bright enough to make the dip change the depth by > {chromatic_min * 100:.0f} % "
                          f"between {b1} and {b2}" + ex)
        elif sc in ("BTP", "BEB", "BEBx2P", "DEB", "DEBx2P"):
            idx_mb.append(i)
            typ_mb[i] = 0.5
    if any(rows[i]["scenario"] in ("BTP", "BEB", "BEBx2P", "DEB", "DEBx2P") for i in idx_mb):
        det_mb.append("background / foreground stars: colour unknown, counted as 50 %")
    add("multiband", f"Transits in two or more filters ({b1} and {b2}, simultaneous "
        f"if possible)", idx_mb, det_mb,
        "a planet on the target gives the same depth in every band (apart from limb "
        "darkening, a few %); a companion of another colour changes it.  Typical = the "
        "share of possible companions with a colour difference large enough to see; "
        "feed the light curves to TRICERATOPS+ (--ground-lc with band)",
        typical=typ_mb)

    # 4. reconnaissance spectroscopy: RV of the target (EB, EBx2P) and a
    #    bright bound companion as a second set of lines (SB2) - also inside
    #    the inner working angle of the imaging, where it is hardest
    idx_rv, typ_rv, det_rv = [], {}, []
    P = summary.get("P_d")
    m_t = summary.get("M_star")
    for i, r in enumerate(rows):
        if r["scenario"] in TARGET_EB:
            idx_rv.append(i)
            typ_rv[i] = 1.0
            m2 = r.get("M_EB")
            if m2 and P and m_t:
                pp = float(P) * (2 if r["scenario"] == "EBx2P" else 1)
                k = 28.43 * (m2 / 9.546e-4) * (pp / 365.25) ** (-1 / 3) \
                    * (float(m_t) + m2) ** (-2 / 3) / 1000.0
                det_rv.append(f"{r['scenario']}: best-fit companion {m2:.2f} Msun -> "
                              f"K ~ {k:.1f} km/s")
        elif r["scenario"] in BOUND:
            fr = r.get("flux_ratio_comp_T") or 0.0
            if fr >= 0.2:
                idx_rv.append(i)
                typ_rv[i] = 0.7
                det_rv.append(f"{r['scenario']}: companion with {fr * 100:.0f} % of the "
                              f"light - a second set of lines (SB2) unless the two "
                              f"velocities coincide")
    add("spectroscopy", f"Reconnaissance spectroscopy (RV to ~{rv_precision_kms:g} km/s, "
        f"2-3 epochs; line profile / SB2)", idx_rv, det_rv,
        "a stellar-mass companion on the target gives K of km/s (a planet m/s); a "
        "bright companion shows its own lines, also closer than imaging can resolve",
        typical=typ_rv)

    # greedy plan: biggest FPP drop first, until the criteria are met
    # greedy plan with the TYPICAL removal fractions
    plan, removed = [], {}
    fp, nf = fpp0, nfpp0
    pool = list(items)
    while pool and not (fp < FPP_LIMIT and nf < NFPP_LIMIT):
        def score(it):
            a, b = _fpp_after(rows, _merge(removed, it["typical"]))
            return a + b
        best = min(pool, key=score)
        new = _merge(removed, best["typical"])
        fp2, nf2 = _fpp_after(rows, new)
        if fp2 >= fp - 1e-6 and nf2 >= nf - 1e-6:
            break
        removed, fp, nf = new, fp2, nf2
        plan.append(dict(key=best["key"], title=best["title"], fpp_after=fp,
                         nfpp_after=nf))
        pool.remove(best)
    items.sort(key=lambda it: -it["prob_addressed"])

    text = [f"FPP = {fpp0:.4f}, NFPP = {nfpp0:.4f} now (criteria: FPP < {FPP_LIMIT}, "
            f"NFPP < {NFPP_LIMIT})"]
    for k, it in enumerate(items, 1):
        text.append(f"{k}. {it['title']}")
        text.append(f"   addresses {', '.join(it['scenarios'])}: P = "
                    f"{it['prob_addressed']:.4f} of the total")
        text.append(f"   if all of these are ruled out: FPP = {it['fpp_after']:.4f}, "
                    f"NFPP = {it['nfpp_after']:.4f} -> "
                    + ("criteria met by this alone" if it["meets"]
                       else "not enough by itself"))
        if any(v < 1.0 for v in it["typical"].values()):
            text.append(f"   typical outcome: FPP = {it['fpp_typical']:.4f}, "
                        f"NFPP = {it['nfpp_typical']:.4f}")
        for d in it["detail"]:
            text.append(f"     - {d}")
        text.append(f"   how: {it['how']}")
    if plan:
        ok = fp < FPP_LIMIT and nf < NFPP_LIMIT
        text.append("Shortest combination (typical outcomes): "
                    + " + ".join(p["key"] for p in plan)
                    + f" -> FPP = {fp:.4f}, NFPP = {nf:.4f}"
                    + ("" if ok else "  (criteria still not met: more data or "
                       "a mass measurement needed)"))
    elif fpp0 < FPP_LIMIT and nfpp0 < NFPP_LIMIT:
        text.append("The criteria are already met; follow-up still checks the "
                    "assumptions (contrast curve, ground transit).")
    return dict(fpp=fpp0, nfpp=nfpp0, items=items, plan=plan, text=text)


def parse_log_table(log_text: str):
    """Fallback for older runs: the 'Scenario probabilities (last iteration)'
    table of the log (last iteration, not pooled) -> scenario_probs rows."""
    lines = log_text.splitlines()
    try:
        k = max(i for i, l in enumerate(lines)
                if l.startswith("Scenario probabilities (last iteration)"))
    except ValueError:
        return []
    head = lines[k + 1].split()
    out = []
    for l in lines[k + 2:]:
        p = l.split()
        if len(p) != len(head) + 1:
            break
        d = dict(zip(head, p[1:]))
        row = {"ID": int(float(d["ID"])), "scenario": d["scenario"]}
        for c in ("prob", "M_s", "R_s", "M_EB", "M_comp", "R_comp",
                  "flux_ratio_comp_T", "R_p"):
            if c in d:
                try:
                    row[c] = float(d[c])
                except ValueError:
                    row[c] = None
        out.append(row)
    return out
