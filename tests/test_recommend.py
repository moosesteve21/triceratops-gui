"""Observation recommendations from the scenario probabilities.

    python -m pytest tests/test_recommend.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_recommend as rc  # noqa: E402

T = 100


def _rows(**p):
    names = ["TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P",
             "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P"]
    rows = [dict(ID=T, scenario=n, prob=p.get(n, 0.0), M_s=1.0, M_comp=0.0,
                 flux_ratio_comp_T=0.0, M_EB=0.0) for n in names]
    for n in ("NTP", "NEB", "NEBx2P"):
        rows.append(dict(ID=200, scenario=n, prob=p.get(n, 0.0)))
    return rows


def _summary(rows, sep=5.9):
    return dict(scenario_probs=rows, Teff=5500, M_star=1.0, P_d=10.0,
                distance_pc=100.0,
                neighbours=[dict(ID=T, sep=0.0, tdepth=0.005),
                            dict(ID=200, sep=sep, Tmag=12.5, tdepth=0.006)])


def test_nearby_star_needs_ground_photometry_first():
    r = rc.recommend(_summary(_rows(TP=0.62, NTP=0.36, STP=0.02)))
    assert r["items"][0]["key"] == "ground"
    assert r["items"][0]["nfpp_after"] == 0.0
    assert r["plan"][0]["key"] == "ground"


def test_unresolved_neighbour_goes_to_imaging():
    r = rc.recommend(_summary(_rows(TP=0.7, NTP=0.3), sep=1.0), seeing_fwhm=2.5)
    keys = [it["key"] for it in r["items"]]
    assert "ground" not in keys and "hires" in keys


def test_background_and_bound_go_to_imaging_with_typical_fraction():
    r = rc.recommend(_summary(_rows(TP=0.9, PEB=0.05, BEB=0.05)))
    hr = [it for it in r["items"] if it["key"] == "hires"][0]
    assert hr["prob_addressed"] == pytest.approx(0.10)
    assert hr["fpp_after"] == pytest.approx(0.0)
    assert hr["fpp_typical"] > hr["fpp_after"]


def test_chromatic_fraction_over_possible_companions():
    # transit on a companion: most possible companions are redder
    f_stp, _ = rc.chromatic_fraction("STP", 1.0, depth=0.005)
    # eclipse on the target diluted by a companion: hardly chromatic
    f_peb, _ = rc.chromatic_fraction("PEB", 1.0, depth=0.005)
    assert f_stp > 0.5 and f_peb < 0.05
    # companions too faint to make the dip are left out
    _, lst = rc.chromatic_fraction("STP", 1.0, depth=0.005)
    assert min(q for q, _ in lst) > 0.3


def test_multiband_typical_uses_that_fraction():
    rows = _rows(TP=0.8, STP=0.1, PEB=0.1)
    res = rc.recommend(_summary(rows) | {"depth_on_target_ppm": 5000.0})
    mb = [it for it in res["items"] if it["key"] == "multiband"][0]
    assert 0.0 < mb["fpp_typical"] < 0.2 and mb["fpp_after"] == pytest.approx(0.0)


def test_rv_amplitude_for_stellar_companion():
    rows = _rows(TP=0.9, EB=0.1)
    for r in rows:
        if r["scenario"] == "EB":
            r["M_EB"] = 0.2
    res = rc.recommend(_summary(rows))
    sp = [it for it in res["items"] if it["key"] == "spectroscopy"][0]
    assert sp["fpp_after"] == pytest.approx(0.0)
    assert any("km/s" in ln for ln in res["text"])


def test_band_fraction_limits():
    assert rc.band_fraction(0.5, 5500, 5500, "B") == pytest.approx(0.5)
    assert rc.band_fraction(0.1, 3500, 5800, "z'") > rc.band_fraction(0.1, 3500, 5800, "B")


def test_parse_log_table():
    log = """Scenario probabilities (last iteration):
          ID scenario       M_s  flux_ratio_comp_T          prob         lnZ
0   65344135       TP  0.970000           0.000000  4.591004e-01  567.1
1   65344134      NTP  0.940000           0.000000  3.302952e-01  566.8

lnZ per scenario"""
    rows = rc.parse_log_table(log)
    assert len(rows) == 2 and rows[1]["ID"] == 65344134 and rows[1]["prob"] == pytest.approx(0.3302952)
