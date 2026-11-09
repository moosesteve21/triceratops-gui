"""Non-graphical tests of triceratops_gui.py (no display needed)."""
import os
import sys

import pytest

pytest.importorskip("tkinter")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_gui as G  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "triceratops_validation.py")


def test_parser_is_read_without_heavy_imports():
    ap, ver = G.load_parser(SCRIPT)
    dests = {a.dest for a in ap._actions}
    for d in ("tic", "ephemeris", "ground_lc", "tri_plus", "exclude_stars"):
        assert d in dests
    assert ver >= "4.1"
    ns = ap.parse_args(["--tic", "1", "--ephemeris", "1", "2", "3", "4"])
    assert ns.ephemeris == [1.0, 2.0, 3.0, 4.0]
    placed = {d for _, ds in G.TABS for d in ds}
    assert dests - placed - {"help"} == set()     # every option has a tab


def test_ground_spec_roundtrip():
    d = dict(path="/a b/c,d.tbl", band="r'", star="65344134", ap="3.5",
             detrend="poly1,AIRMASS", fcol="rel_flux_T2")
    s = G.build_spec(d)
    back = G.parse_spec(s)
    for k, v in d.items():
        assert back[k] == v
    assert G.parse_spec("x.tbl band=R star=TIC 123")["star"] == "123"


def test_shipped_profiles_are_valid():
    import glob
    import json
    ap, _ = G.load_parser(SCRIPT)
    dests = {a.dest for a in ap._actions}
    files = glob.glob(os.path.join(os.path.dirname(SCRIPT), "profiles", "*.json"))
    assert files
    for f in files:
        with open(f, encoding="utf-8") as fh:
            st = json.load(fh)
        assert set(st["options"]) <= dests, f
        eph = st["options"].get("ephemeris")
        if eph:
            assert len(eph) == 4 and all(float(x) > 0 for x in eph)
