"""Tests for triceratops_batch.py (no network, no pipeline runs).

    python -m pytest tests/test_batch.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triceratops_batch as tb  # noqa: E402

PARSER = tb.load_parser()


def test_options_to_args_like_a_profile():
    opts = {"tic": "999000001", "ephemeris": ["3.1", "1500.2", "900", "1.5"],
            "ld_fit": True, "sensitivity": False, "ephem_scan_hours": "16",
            "mask_ephemeris": [11.13, 1490.5, 2.0], "exclude_sectors": "6 33",
            "ground_lc": ["a.tbl,band=r", "b.tbl,band=i"]}
    a = tb.options_to_args(opts, PARSER)
    assert a[:2] == ["--tic", "999000001"]
    assert a[a.index("--ephemeris") + 1:a.index("--ephemeris") + 5] == ["3.1", "1500.2", "900", "1.5"]
    assert "--ld-fit" in a and "--sensitivity" not in a
    assert a[a.index("--exclude-sectors") + 1:a.index("--exclude-sectors") + 3] == ["6", "33"]
    assert a.count("--ground-lc") == 2
    ns = PARSER.parse_args(a)
    assert ns.mask_ephemeris == [11.13, 1490.5, 2.0] and ns.ephem_scan_hours == 16


@pytest.mark.parametrize("opts, msg", [
    ({"tic": 1, "no_such_option": 1}, "unknown option"),
    ({"tic": 1, "ld_fit": "yes"}, "true or false"),
    ({"tic": 1, "ephemeris": [1, 2, 3]}, "ephemeris"),
])
def test_options_to_args_rejects_bad_options(opts, msg):
    with pytest.raises(ValueError, match=msg):
        tb.options_to_args(opts, PARSER)


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return str(path)


def test_load_batch_merges_defaults_profile_and_options(tmp_path):
    prof = _write(tmp_path / "p.json", {"options": {
        "tic": "999000001", "ephemeris": ["3.1", "1500.2", "900", "1.5"],
        "ephem_search": "auto", "outdir": "somewhere"}, "ground": ["g.tbl,band=r"]})
    batch = _write(tmp_path / "b.json", {
        "name": "r3", "results_dir": str(tmp_path / "res"),
        "defaults": {"ld_fit": True, "ephem_search": "scan"},
        "candidates": [
            {"label": "kX_off", "profile": prof, "options": {"ephem_search": "off"}},
            {"tic": 999000002, "options": {"ephemeris": [2, 1400, 500, 2]}}]})
    b = tb.load_batch(batch, PARSER)
    assert b["root"] == str(tmp_path / "res" / "r3")
    assert b["cache"] == str(tmp_path / "res" / "r3" / "tesscut_cache")
    j1, j2 = b["jobs"]
    assert j1["outdir"] == os.path.join(b["root"], "kX_off")
    ns = PARSER.parse_args(j1["args"])
    assert ns.ephem_search == "off" and ns.ld_fit and ns.ground_lc == ["g.tbl,band=r"]
    assert ns.outdir is None and j1["dropped"] == ["outdir"]   # set by the batch
    assert j2["label"] == "TIC999000002" and j2["tic"] == 999000002
    assert PARSER.parse_args(j2["args"]).ephem_search == "scan"


def test_load_batch_allows_notes_and_reports_unknown_keys(tmp_path):
    batch = _write(tmp_path / "b.json", {
        "notes": "round 3", "candidates": [
            {"tic": 1, "notes": "2P alias of an earlier candidate"},
            {"tic": 2, "optoins": {"ld_fit": True}}]})
    b = tb.load_batch(batch, PARSER)
    assert b["jobs"][0]["notes"] == "2P alias of an earlier candidate"
    assert len(b["warnings"]) == 1 and "optoins" in b["warnings"][0]


def test_load_batch_rejects_duplicate_labels(tmp_path):
    batch = _write(tmp_path / "b.json", {"candidates": [{"tic": 1}, {"tic": 1}]})
    with pytest.raises(ValueError, match="used twice"):
        tb.load_batch(batch, PARSER)


def _batch(tmp_path, n=2):
    return tb.load_batch(_write(tmp_path / "b.json", {
        "name": "r3", "results_dir": str(tmp_path / "res"),
        "candidates": [{"tic": 999000000 + i} for i in range(n)]}), PARSER)


def _fake_run(calls, results_on=lambda cmd, n: True):
    """Stand-in for tb._run: records calls; a test run writes a summary when
    results_on(cmd, n_previous_runs_of_that_folder) is true."""
    def run(cmd, cwd, log_path, timeout_min=None):
        n = sum(1 for c, w in calls if w == cwd and c[1] == tb.SCRIPT)
        calls.append((cmd, cwd))
        if cmd[1] == tb.SCRIPT and "--prefetch-only" not in cmd and results_on(cmd, n):
            tic = cmd[cmd.index("--tic") + 1]
            _write(os.path.join(cwd, f"summary_{tic}.json"),
                   {"verdict": "NOT passed", "FPP": 0.2, "NFPP": 0.01})
        return 0, False
    return run


def test_test_mode_skips_finished_and_uses_the_shared_cache(tmp_path, monkeypatch):
    b = _batch(tmp_path)
    done, todo = b["jobs"]
    os.makedirs(done["outdir"])
    _write(os.path.join(done["outdir"], "summary_999000000.json"), {"FPP": 0.1})
    calls = []
    monkeypatch.setattr(tb, "_run", _fake_run(calls))
    rc = tb.test(b, b["jobs"], tb.Status(b), workers=1)
    assert rc == 0
    runs = [c for c in calls if c[0][1] == tb.SCRIPT]
    assert len(runs) == 1 and runs[0][1] == todo["outdir"]
    cmd = runs[0][0]
    assert cmd[cmd.index("--download-dir") + 1] == b["cache"]
    assert cmd[cmd.index("--outdir") + 1] == todo["outdir"]
    assert any(c[0][1] == tb.AUDIT for c in calls)
    st = json.load(open(os.path.join(b["root"], "batch_status.json")))
    assert st["jobs"][todo["label"]]["FPP"] == 0.2
    # --force reruns the finished one too
    calls.clear()
    tb.test(b, b["jobs"], tb.Status(b), workers=1, force=True, audit=False)
    assert len([c for c in calls if c[0][1] == tb.SCRIPT]) == 2


def test_test_without_results_is_retried(tmp_path, monkeypatch):
    b = _batch(tmp_path, n=1)
    calls = []
    # first attempt dies without a summary (e.g. MAST down), the second works
    monkeypatch.setattr(tb, "_run", _fake_run(calls, results_on=lambda cmd, n: n >= 1))
    assert tb.test(b, b["jobs"], tb.Status(b), retries=1, wait_min=0, audit=False) == 0
    assert len(calls) == 2
    calls.clear()
    for f in os.listdir(b["jobs"][0]["outdir"]):
        os.remove(os.path.join(b["jobs"][0]["outdir"], f))
    monkeypatch.setattr(tb, "_run", _fake_run(calls, results_on=lambda cmd, n: False))
    assert tb.test(b, b["jobs"], tb.Status(b), retries=1, wait_min=0, audit=False) == 1
    assert len(calls) == 2                       # one retry, then it gives up


def test_prefetch_runs_each_candidate_and_retries_failures(tmp_path, monkeypatch):
    b = _batch(tmp_path)
    calls = []

    def run(cmd, cwd, log_path, timeout_min=None):
        calls.append(cwd)
        failing = cwd == b["jobs"][1]["outdir"] and calls.count(cwd) == 1
        return (1 if failing else 0), False          # 2nd candidate fails once

    monkeypatch.setattr(tb, "_run", run)
    assert tb.prefetch(b, b["jobs"], tb.Status(b), passes=3, wait_min=0) == 0
    assert calls.count(b["jobs"][0]["outdir"]) == 1 and calls.count(b["jobs"][1]["outdir"]) == 2
    assert os.path.isdir(b["cache"]) and all(os.path.isdir(j["outdir"]) for j in b["jobs"])
    calls.clear()
    monkeypatch.setattr(tb, "_run", lambda cmd, cwd, log_path, timeout_min=None:
                        calls.append(cwd) or (1, False))
    assert tb.prefetch(b, b["jobs"], tb.Status(b), passes=2, wait_min=0) == 1
    assert len(calls) == 4


def test_run_stops_a_hung_process(tmp_path):
    log = str(tmp_path / "hung.log")
    t0 = __import__("time").time()
    rc, timed_out = tb._run([sys.executable, "-c", "import time; print('start', flush=True); "
                             "time.sleep(120)"], str(tmp_path), log, timeout_min=0.05)
    assert timed_out and rc == -9 and __import__("time").time() - t0 < 60
    text = open(log, encoding="utf-8").read()
    assert "start" in text and "stopped by triceratops_batch.py" in text
    rc, timed_out = tb._run([sys.executable, "-c", "print('ok')"], str(tmp_path), log, 1)
    assert (rc, timed_out) == (0, False)


def test_clean_deletes_only_a_cutout_cache(tmp_path):
    b = _batch(tmp_path)
    sub = os.path.join(b["cache"], "tesscut")
    os.makedirs(sub)
    open(os.path.join(sub, "tess-s0010-1-1_1.0_2.0_10x10_astrocut.fits"), "w").close()
    open(os.path.join(b["cache"], "notes.txt"), "w").close()
    assert tb.clean(b) == 1 and os.path.isdir(b["cache"])      # refused
    os.remove(os.path.join(b["cache"], "notes.txt"))
    assert tb.clean(b, dry=True) == 0 and os.path.isdir(b["cache"])
    assert tb.clean(b) == 0 and not os.path.exists(b["cache"])
