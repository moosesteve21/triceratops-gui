"""Kepler mode (v4.2) of triceratops_validation.py, with lightkurve mocked.

    python -m pytest tests/test_kepler_mode.py -q
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_validation as tv  # noqa: E402


class _Tab(dict):
    @property
    def colnames(self):
        return list(self.keys())


class _FakeLC:
    def __init__(self, q):
        rng = np.random.default_rng(q)
        t0 = 100 + 90 * q
        self.time = SimpleNamespace(value=np.arange(t0, t0 + 85, 0.0204))
        n = self.time.value.size
        self.pdcsap_flux = SimpleNamespace(value=1e5 * (1 + 1e-4 * rng.standard_normal(n)))
        self.pdcsap_flux_err = SimpleNamespace(value=np.full(n, 10.0))
        self.sap_flux = self.pdcsap_flux
        self.sap_flux_err = self.pdcsap_flux_err
        self.mom_centr1 = SimpleNamespace(value=np.full(n, 500.0))
        self.mom_centr2 = SimpleNamespace(value=np.full(n, 300.0))

    def remove_nans(self):
        return self


class _Item:
    def __init__(self, q, kind):
        self.q, self.kind = q, kind

    def download(self, **kw):
        if self.kind == "lc":
            return _FakeLC(self.q)
        return None


class _Search(list):
    def __init__(self, qs, kind):
        super().__init__(_Item(q, kind) for q in qs)
        self.table = _Tab(sequence_number=list(qs),
                          mission=[f"Kepler Quarter {q:02d}" for q in qs])


@pytest.fixture
def kepler(monkeypatch, tmp_path):
    calls = []

    def fake_search(kind):
        def f(target, **kw):
            calls.append((kind, target, kw))
            qs = [4, 9, 12, 15] if "quarter" not in kw else [kw["quarter"]]
            return _Search(qs, kind)
        return f

    monkeypatch.setattr(tv.lk, "search_lightcurve", fake_search("lc"))
    monkeypatch.setattr(tv.lk, "search_targetpixelfile", fake_search("tpf"))
    tv.set_mission("kepler")
    cfg = tv.Config(tic=123, mission="kepler", kic=3755522, lc_source="pdcsap",
                    outdir=str(tmp_path / "out"))
    yield cfg, calls
    tv.set_mission("tess")


def test_time_system_switch():
    tv.set_mission("kepler")
    assert tv.TIME_ZERO == 2454833.0 and tv.TIME_LABEL == "BKJD"
    tv.set_mission("tess")
    assert tv.TIME_ZERO == 2457000.0 and tv.TIME_LABEL == "BTJD"


def test_quarters_found(kepler):
    cfg, calls = kepler
    q = tv.find_tess_sectors(cfg.tic, cfg=cfg)
    assert list(q) == [4, 9, 12, 15]
    kind, target, kw = calls[0]
    assert target == "KIC 3755522" and kw["mission"] == "Kepler"


def test_extract_quarter(kepler):
    cfg, calls = kepler
    sd = tv.extract_sector(cfg.tic, 9, cfg)
    assert sd.sector == 9 and sd.time.size > 1000
    assert "Kepler" in sd.aperture_kind
    assert abs(sd.cadence_d * 1440 - 29.4) < 0.5
    assert any(kw.get("quarter") == 9 for _, _, kw in calls)


def test_default_outdir_is_kic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tv.Config(tic=1, mission="kepler", kic=3755522)
    assert os.path.basename(cfg.outdir) == "KIC3755522"


def test_parser_has_mission_options():
    ap = tv.build_parser()
    a = ap.parse_args(["--mission", "kepler", "--kic", "3755522", "--tic", "5",
                       "--ephemeris", "251.9607", "1123.2506", "198", "5.2"])
    assert a.mission == "kepler" and a.kic == 3755522
    assert a.ephemeris[1] == pytest.approx(1123.2506)


def test_extra_times_use_mission_zero(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("2455956.2506 3.0 ground\n")
    tv.set_mission("kepler")
    try:
        rows = tv._read_extra_times(str(p))
    finally:
        tv.set_mission("tess")
    assert rows[0][0] == pytest.approx(1123.2506)


def test_masked_quarter_numbers_fall_back_to_mission_text():
    import numpy.ma as ma
    s = _Search([4, 9], "lc")
    s.table["sequence_number"] = ma.masked_array([0, 0], mask=[True, True])
    assert tv._quarters_of(s) == [4, 9]


class _Row(dict):
    pass


class _TicTab(list):
    colnames = ["ID", "disposition", "duplicate_id", "ra", "dec", "Tmag", "dstArcSec"]


def test_kic_resolution_skips_split_entry(monkeypatch):
    split = _Row(ID=138303937, disposition="SPLIT", duplicate_id=None,
                 ra=293.6538, dec=38.8340, Tmag=11.8, dstArcSec=0.0)
    a = _Row(ID=1000001, disposition="", duplicate_id=None, ra=293.6538,
             dec=38.8340, Tmag=12.0, dstArcSec=0.1)
    b = _Row(ID=1000002, disposition="", duplicate_id=None, ra=293.6537,
             dec=38.8333, Tmag=12.9, dstArcSec=2.5)
    monkeypatch.setattr(tv.Catalogs, "query_criteria",
                        lambda **kw: _TicTab([split]))
    monkeypatch.setattr(tv.Catalogs, "query_region",
                        lambda *a_, **kw: _TicTab([split, a, b]))
    assert tv.resolve_tic_from_kic(3755522) == 1000001


def test_kic_resolution_follows_duplicate(monkeypatch):
    dup = _Row(ID=5, disposition="DUPLICATE", duplicate_id=77, ra=1.0, dec=2.0,
               Tmag=10.0, dstArcSec=0.0)
    monkeypatch.setattr(tv.Catalogs, "query_criteria", lambda **kw: _TicTab([dup]))
    assert tv.resolve_tic_from_kic(1) == 77


def test_flagged_rows_dropped_from_triceratops_table():
    import pandas as pd
    t = SimpleNamespace(
        stars=pd.DataFrame(dict(ID=[1, 2, 3], disposition=["", "SPLIT", None])),
        pix_coords=[np.array([[0, 0], [1, 1], [2, 2]])] * 2)
    tv._drop_flagged_tic_rows(t, 1)
    assert list(t.stars["ID"]) == [1, 3]
    assert all(pc.shape == (2, 2) and pc[1, 0] == 2 for pc in t.pix_coords)


def test_flagged_target_row_is_an_error():
    import pandas as pd
    t = SimpleNamespace(stars=pd.DataFrame(dict(ID=[9, 1], disposition=["SPLIT", ""])),
                        pix_coords=[np.zeros((2, 2))])
    with pytest.raises(RuntimeError):
        tv._drop_flagged_tic_rows(t, 1)


# ---- host-star options (--host-tic, --star-source, --stellar) -------------
class _VRow(dict):
    @property
    def colnames(self):
        return list(self.keys())


def _fake_vizier(rows_by_cat):
    class V:
        def __init__(self, *a, **kw):
            pass

        def query_constraints(self, KIC=None, catalog=None):
            r = rows_by_cat.get(catalog)
            return [[r]] if r is not None else []
    return V


def test_kepler_params_prefer_berger(monkeypatch):
    import astroquery.vizier as av
    monkeypatch.setattr(tv, "_VIZIER_HOST", "vizier.test")
    berger = _VRow(KIC=3755522, Rad=1.05, E_Rad=0.05, e_Rad=0.04, Mass=1.02,
                   E_Mass=0.06, e_Mass=0.05, Teff=5790.0, logg=4.40)
    monkeypatch.setattr(av, "Vizier", _fake_vizier({"J/AJ/159/280/table2": berger}))
    p = tv.kepler_stellar_params(3755522)
    assert p["rad"] == pytest.approx(1.05) and p["mass"] == pytest.approx(1.02)
    assert p["rad_max"] == pytest.approx(1.095) and "Berger" in p["source"]


def test_kepler_params_fall_back_to_dr25(monkeypatch):
    import astroquery.vizier as av
    monkeypatch.setattr(tv, "_VIZIER_HOST", "vizier.test")
    dr25 = _VRow(KIC=1, Rad=0.9, Mass=0.95, Teff=5500.0, logg=4.5)
    monkeypatch.setattr(av, "Vizier", _fake_vizier({"J/ApJS/229/30/catalog": dr25}))
    p = tv.kepler_stellar_params(1)
    assert p["rad"] == pytest.approx(0.9) and "DR25" in p["source"]


def test_manual_stellar_overrides_everything(monkeypatch):
    cfg = SimpleNamespace(host_tic=None, star_source="tic", kic=None,
                          stellar_manual=(0.84, 0.89, 5310.0))
    star = dict(rad=1.2, mass=1.1, Teff=5800.0, ra=1.0, dec=2.0)
    out = tv.apply_star_overrides(star, cfg, 10)
    assert out["rad"] == 0.84 and out["mass"] == 0.89 and out["Teff"] == 5310.0
    assert out["source"].startswith("manual")


def test_host_tic_uses_host_parameters(monkeypatch):
    monkeypatch.setattr(tv, "stellar_params",
                        lambda tic, kic=None: dict(rad=0.8, mass=0.85, Teff=5300.0,
                                                   ra=5.0, dec=6.0))
    cfg = SimpleNamespace(host_tic=22, star_source="kepler", kic=3755522,
                          stellar_manual=None)
    out = tv.apply_star_overrides(dict(rad=1.2, mass=1.1, Teff=5800.0), cfg, 10)
    assert out["rad"] == 0.8 and out["source"] == "TIC 22"   # kepler ignored


def test_host_params_written_to_triceratops_row0():
    import pandas as pd
    t = SimpleNamespace(stars=pd.DataFrame(dict(ID=[1, 2], mass=[1.0, 0.5],
                                                rad=[1.0, 0.5], Teff=[5800.0, 4000.0])))
    cfg = SimpleNamespace(_star=dict(rad=0.84, mass=0.89, Teff=5310.0,
                                     source="manual (--stellar)"))
    tv._set_host_params(t, cfg)
    assert t.stars.loc[0, "rad"] == 0.84 and t.stars.loc[1, "rad"] == 0.5


def test_parser_host_options():
    a = tv.build_parser().parse_args(["--tic", "1", "--host-tic", "2",
                                      "--stellar", "0.84", "0.89", "5310"])
    assert a.host_tic == 2 and a.stellar == [0.84, 0.89, 5310.0]


def test_kic_resolution_real_case_prefers_star_at_position(monkeypatch):
    """KIC 3755522: SPLIT 2MASS entry, DUPLICATE Gaia entry of A at 0.00",
    unflagged companion B at 2.49" - B must NOT be chosen."""
    split = _Row(ID=138303937, disposition="SPLIT", duplicate_id=None,
                 ra=293.6538, dec=38.8340, Tmag=11.82, dstArcSec=0.41)
    a_dup = _Row(ID=1877145501, disposition="DUPLICATE", duplicate_id=138303937,
                 ra=293.6538, dec=38.8340, Tmag=11.82, dstArcSec=0.0)
    b = _Row(ID=1877145512, disposition="", duplicate_id=None, ra=293.6537,
             dec=38.8333, Tmag=12.66, dstArcSec=2.49)
    monkeypatch.setattr(tv.Catalogs, "query_criteria", lambda **kw: _TicTab([split]))
    monkeypatch.setattr(tv.Catalogs, "query_region",
                        lambda *a_, **kw: _TicTab([a_dup, split, b]))
    assert tv.resolve_tic_from_kic(3755522) == 1877145501


def test_flagged_target_is_kept_when_it_is_the_chosen_tic():
    import pandas as pd
    t = SimpleNamespace(stars=pd.DataFrame(dict(ID=[1877145501, 138303937, 1877145512],
                                                disposition=["DUPLICATE", "SPLIT", ""])),
                        pix_coords=[np.zeros((3, 2))])
    tv._drop_flagged_tic_rows(t, 1877145501)
    assert list(t.stars["ID"]) == [1877145501, 1877145512]


def test_missing_radius_mass_estimated_from_teff():
    cfg = SimpleNamespace(host_tic=None, star_source="tic", kic=None, stellar_manual=None)
    out = tv.apply_star_overrides(dict(rad=np.nan, mass=np.nan, Teff=5812.0), cfg, 1)
    assert 0.95 < out["mass"] < 1.1 and 0.95 < out["rad"] < 1.1
    assert "estimated" in out["source"]


def test_kic_position_answered_offline():
    from astroquery.vizier import Vizier
    with tv._kic_position_offline(3755522, 293.65, 38.83):
        r = Vizier(columns=["_RA", "_DE"]).query_constraints(
            KIC="3755522", catalog="J/ApJS/229/30/catalog")[0].as_array()
    assert r[0]["_RA"] == 293.65 and r[0]["_DE"] == 38.83
