#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run triceratops_validation.py on a list of candidates from a JSON file.

    python triceratops_batch.py BATCH.json --mode prefetch   (download cutouts)
    python triceratops_batch.py BATCH.json --mode test       (run + audit)
    python triceratops_batch.py BATCH.json --mode both
    python triceratops_batch.py BATCH.json --mode status
    python triceratops_batch.py BATCH.json --mode clean      (delete the cache)

'prefetch' creates every candidate's output folder and downloads all the
TESScut cutouts the runs will read into one shared cache folder (network
work, light on CPU).  'test' runs the pipeline and the audit for every
candidate with that cache (CPU work), a few at a time.  The cache is kept
until 'clean' deletes it, so the tests can be rerun with other settings
without downloading again.  A finished candidate (summary_*.json in its
folder) is skipped unless --force is given; change its label or outdir to
keep both results.

MAST stalls and refuses requests now and then.  Inside each run the script
times out stalled requests and retries them (--net-timeout, --net-retries;
put them in "defaults" to change them).  On top of that this script stops a
prefetch after --prefetch-timeout and a test after --test-timeout minutes
(the whole process tree), runs the failed prefetches again in up to
--prefetch-passes passes, and reruns a test that ended without results
--test-retries times, --retry-wait minutes later.  Prefetching again is
cheap: cutouts already in the cache are skipped.

Batch file (paths relative to the current directory):

    {
      "name": "round3",                  optional; default: the file name
      "results_dir": "results",          optional; parent folder of the batch
      "cache_dir": "...",                optional; default <results_dir>/<name>/tesscut_cache
      "workers": 3,                      optional; parallel tests (default 3)
      "audit": true,                     optional; run triceratops_audit.py after each test
      "defaults": {"ld_fit": true},      options for every candidate
      "candidates": [
        {"label": "kA_auto", "profile": "profiles/profile_X.json",
         "options": {"ephem_search": "auto"}},
        {"tic": 123456789, "options": {"ephemeris": [3.18, 1468.3, 900, 1.5]}}
      ]
    }

Options use the names in the profiles (the script's argument names with
underscores); values may be numbers or strings, lists for multi-value
options (ephemeris: P T0 depth_ppm T14_h; mask_ephemeris: P T0 T14_h per
planet).  For each candidate: defaults < profile < options; "tic" may also
sit on the candidate itself.  label defaults to TIC<tic>; outdir defaults to
<results_dir>/<name>/<label>.  outdir, download_dir and prefetch_only are
set by this script (a profile's own outdir is ignored).  "notes" (or
"description") may be added to the file and to any candidate; they are
not used.  Other unknown keys are reported, since a misspelt "options"
would otherwise drop a candidate's settings silently.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "triceratops_validation.py")
AUDIT = os.path.join(HERE, "triceratops_audit.py")
MANAGED = ("outdir", "download_dir", "prefetch_only")
BATCH_KEYS = {"name", "results_dir", "cache_dir", "workers", "audit", "defaults",
              "candidates", "notes", "description"}
CANDIDATE_KEYS = {"label", "tic", "profile", "options", "outdir", "notes", "description"}


# --------------------------------------------------------------------------
# Reading the batch file
# --------------------------------------------------------------------------
def load_parser():
    """The validation script's argument parser, without its heavy imports."""
    sys.path.insert(0, HERE)
    try:
        from triceratops_gui import load_parser as _lp
    except ImportError:                      # no tkinter: import the script itself
        import triceratops_validation as tv
        return tv.build_parser()
    return _lp(SCRIPT)[0]


def options_to_args(opts: dict, parser) -> list:
    """Command-line tokens for a profile-style options dict (validated)."""
    acts = {a.dest: a for a in parser._actions if a.option_strings}
    unknown = sorted(set(opts) - set(acts))
    if unknown:
        raise ValueError(f"unknown option(s): {', '.join(unknown)}")
    out = []
    for dest, val in opts.items():
        a = acts[dest]
        flag = a.option_strings[0]
        kind = type(a).__name__
        if val is None or val == "" or val == []:
            continue
        if kind == "_StoreTrueAction":
            if not isinstance(val, bool):
                raise ValueError(f"{dest}: true or false expected, got {val!r}")
            if val:
                out.append(flag)
            continue
        vals = list(val) if isinstance(val, (list, tuple)) else None
        if kind == "_AppendAction":
            for v in (vals if vals is not None else [val]):
                out += [flag, str(v)]
        elif a.nargs in (None, "?"):
            if vals is not None:
                raise ValueError(f"{dest}: one value expected, got a list")
            out += [flag, str(val)]
        else:
            toks = [str(v) for v in vals] if vals is not None else shlex.split(str(val))
            out += [flag] + toks
    import contextlib
    import io
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parser.parse_args(out)
    except SystemExit:
        msg = err.getvalue().strip().splitlines()
        raise ValueError(msg[-1] if msg else "invalid options")
    return out


def load_batch(path: str, parser) -> dict:
    """Resolve the batch file into {name, root, cache, workers, audit, jobs}."""
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    root = os.path.abspath(os.path.join(spec.get("results_dir", "results"), name))
    cache = os.path.abspath(spec.get("cache_dir") or os.path.join(root, "tesscut_cache"))
    defaults = dict(spec.get("defaults") or {})
    warnings = []
    extra = sorted(set(spec) - BATCH_KEYS)
    if extra:
        warnings.append(f"batch file: unknown key(s) {', '.join(extra)} ignored")
    jobs, seen = [], set()
    for i, c in enumerate(spec.get("candidates") or []):
        where = f"candidate {i + 1}"
        extra = sorted(set(c) - CANDIDATE_KEYS)
        if extra:
            warnings.append(f"{where} ({c.get('label', '?')}): unknown key(s) "
                            f"{', '.join(extra)} ignored (options go inside \"options\")")
        opts = dict(defaults)
        ground = []
        if c.get("profile"):
            with open(c["profile"], encoding="utf-8") as fh:
                prof = json.load(fh)
            opts.update(prof.get("options") or {})
            ground = list(prof.get("ground") or [])
        opts.update(c.get("options") or {})
        if c.get("tic") is not None:
            opts["tic"] = c["tic"]
        dropped = [k for k in MANAGED if opts.pop(k, None) not in (None, False, "")]
        if opts.get("tic") in (None, ""):
            raise ValueError(f"{where}: no tic")
        label = str(c.get("label") or f"TIC{opts['tic']}")
        if label in seen:
            raise ValueError(f"{where}: label {label!r} used twice")
        seen.add(label)
        try:
            args = options_to_args(opts, parser)
        except ValueError as exc:
            raise ValueError(f"{where} ({label}): {exc}") from None
        for g in ground:
            args += ["--ground-lc", str(g)]
        outdir = os.path.abspath(c.get("outdir") or os.path.join(root, label))
        jobs.append(dict(label=label, tic=int(opts["tic"]), args=args, outdir=outdir,
                         dropped=dropped, notes=c.get("notes") or c.get("description")))
    if not jobs:
        raise ValueError("the batch file has no candidates")
    return dict(name=name, root=root, cache=cache, jobs=jobs, warnings=warnings,
                workers=int(spec.get("workers", 3)), audit=bool(spec.get("audit", True)))


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------
def _read_json(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    try:
        with open(hits[0], encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def job_state(job) -> dict:
    out = job["outdir"]
    pre = _read_json(os.path.join(out, f"prefetch_TIC{job['tic']}.json"))
    summ = _read_json(os.path.join(out, "summary_*.json"))
    aud = _read_json(os.path.join(out, "audit_*.json"))
    st = dict(label=job["label"], tic=job["tic"], outdir=out,
              prefetched=None, finished=summ is not None)
    if pre is not None:
        st["prefetched"] = not pre.get("failed")
        st["prefetch"] = {k: pre.get(k) for k in ("sectors", "sizes", "cached",
                                                  "downloaded", "failed")}
    if summ is not None:
        st.update(verdict=summ.get("verdict"), FPP=summ.get("FPP"), NFPP=summ.get("NFPP"))
    if aud is not None:
        counts = {}
        for t in aud.get("tests", []):
            counts[t.get("status")] = counts.get(t.get("status"), 0) + 1
        st["audit"] = {k: counts.get(k, 0) for k in ("FAIL", "WARN", "PASS", "NA")}
    return st


class Status:
    """batch_status.json in the batch folder: one record per candidate."""

    def __init__(self, batch):
        self.path = os.path.join(batch["root"], "batch_status.json")
        self.lock = threading.Lock()
        self.data = dict(name=batch["name"], cache_dir=batch["cache"], jobs={})
        old = _read_json(self.path)
        if old and isinstance(old.get("jobs"), dict):
            self.data["jobs"].update(old["jobs"])

    def update(self, label, **kw):
        with self.lock:
            self.data["jobs"].setdefault(label, {}).update(kw)
            self.data["updated"] = _now()
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp, self.path)


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def _env():
    return dict(os.environ, PYTHONIOENCODING="utf-8")


def _validation_cmd(job, batch, *extra):
    return ([sys.executable, SCRIPT, *job["args"], "--outdir", job["outdir"],
             "--download-dir", batch["cache"], *extra])


def _kill_tree(p):
    """Kill a process and its children (TRICERATOPS may run a worker pool)."""
    try:
        if os.name == "nt":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import signal
            os.killpg(p.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        p.kill()
        p.wait(timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run(cmd, cwd, log_path, timeout_min=None):
    """Run cmd with its output in log_path; kill it after timeout_min minutes.

    Returns (returncode, timed_out).  A run killed for taking too long gets
    a note at the end of its log and returncode -9.
    """
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
          else {"start_new_session": True})
    with open(log_path, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                             env=_env(), **kw)
        try:
            return p.wait(timeout=timeout_min * 60 if timeout_min else None), False
        except subprocess.TimeoutExpired:
            _kill_tree(p)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n!!! stopped by triceratops_batch.py after {timeout_min:g} min\n")
    return -9, True


def prefetch(batch, jobs, status, workers=1, dry=False, passes=3, wait_min=5.0,
             timeout_min=120.0):
    """Download the cutouts; candidates of the same TIC run one after another.

    A candidate whose prefetch fails or times out is tried again in up to
    `passes` passes, `wait_min` minutes apart; cutouts already in the cache
    are not downloaded again.
    """
    if not dry:
        os.makedirs(batch["cache"], exist_ok=True)

    def one_tic(group):
        res = {}
        for j in group:
            cmd = _validation_cmd(j, batch, "--prefetch-only")
            if dry:
                print("   ", shlex.join(cmd))
                res[j["label"]] = 0
                continue
            os.makedirs(j["outdir"], exist_ok=True)
            t0 = time.time()
            rc, timed_out = _run(cmd, j["outdir"], os.path.join(j["outdir"], "prefetch.log"),
                                 timeout_min)
            p = job_state(j).get("prefetch") or {}
            msg = (f"{j['label']}: prefetch rc={rc}"
                   + (f" (stopped after {timeout_min:g} min)" if timed_out else "")
                   + f", {p.get('downloaded', '?')} downloaded, "
                     f"{p.get('cached', '?')} already cached, "
                     f"{len(p.get('failed') or [])} failed, {(time.time() - t0) / 60:.1f} min")
            print(msg, flush=True)
            status.update(j["label"], prefetch_rc=rc, prefetch_timed_out=timed_out,
                          prefetched_at=_now(), prefetch=p)
            res[j["label"]] = rc
        return res

    todo = list(jobs)
    for n in range(1, max(1, passes) + 1):
        by_tic = {}
        for j in todo:
            by_tic.setdefault(j["tic"], []).append(j)
        rcs = {}
        with ThreadPoolExecutor(max(1, workers)) as ex:
            for r in ex.map(one_tic, by_tic.values()):
                rcs.update(r)
        todo = [j for j in todo if rcs.get(j["label"]) != 0]
        if not todo or dry:
            return 0
        if n < passes:
            print(f"{len(todo)} candidate(s) not fully prefetched; pass {n + 1}/{passes} "
                  f"in {wait_min:g} min", flush=True)
            time.sleep(wait_min * 60)
    print(f"! still incomplete after {passes} pass(es): "
          f"{', '.join(j['label'] for j in todo)}", flush=True)
    return 1


def test(batch, jobs, status, workers=3, force=False, audit=True, dry=False,
         timeout_min=300.0, retries=1, wait_min=5.0):
    """Pipeline + audit for every candidate, `workers` at a time.

    A test that ends without a summary (crash, network failure, or stopped
    after timeout_min minutes) is run again up to `retries` times,
    `wait_min` minutes later.
    """
    if not os.path.isdir(batch["cache"]) and not dry:
        print(f"! no cache at {batch['cache']}: every cutout will be downloaded "
              f"during the tests (run --mode prefetch first)", flush=True)

    def one(j):
        st = job_state(j)
        if st["finished"] and not force:
            print(f"{j['label']}: skipped (finished; --force reruns it)", flush=True)
            return 0
        cmd = _validation_cmd(j, batch)
        if dry:
            print("   ", shlex.join(cmd))
            return 0
        if st["prefetched"] is not True:
            print(f"{j['label']}: ! not (fully) prefetched; missing cutouts will be "
                  f"downloaded during the test", flush=True)
        os.makedirs(j["outdir"], exist_ok=True)
        log = os.path.join(j["outdir"], "stdout.log")
        for attempt in range(1, max(0, retries) + 2):
            for old in glob.glob(os.path.join(j["outdir"], "summary_*.json")):
                os.remove(old)                  # "finished" must mean this run
            status.update(j["label"], test_started=_now(), test_rc=None, attempt=attempt)
            t0 = time.time()
            rc, timed_out = _run(cmd, j["outdir"], log, timeout_min)
            if job_state(j)["finished"] or attempt > retries:
                break
            print(f"{j['label']}: test rc={rc}"
                  + (f" (stopped after {timeout_min:g} min)" if timed_out else "")
                  + f" without results; attempt {attempt + 1}/{retries + 1} in "
                    f"{wait_min:g} min", flush=True)
            time.sleep(wait_min * 60)
        rca = None
        if audit:
            rca, _ = _run([sys.executable, AUDIT, j["outdir"]], j["outdir"],
                          os.path.join(j["outdir"], "audit_stdout.log"), 30)
        try:
            with open(log, encoding="utf-8", errors="replace") as fh:
                n_dl = sum("not in cache" in line for line in fh)
        except OSError:
            n_dl = None
        st = job_state(j)
        a = st.get("audit")
        msg = (f"{j['label']}: test rc={rc}"
               + (f" (stopped after {timeout_min:g} min)" if timed_out else "")
               + (f", attempt {attempt}" if attempt > 1 else "")
               + (f", audit rc={rca}" if audit else "")
               + f", {(time.time() - t0) / 60:.1f} min, {n_dl} cutout(s) downloaded"
               + (f"; {st.get('verdict')}, FPP {_g(st.get('FPP'))}, NFPP {_g(st.get('NFPP'))}"
                  if st["finished"] else "")
               + (f"; audit {a['FAIL']} FAIL {a['WARN']} WARN" if a else ""))
        print(msg, flush=True)
        status.update(j["label"], test_rc=rc, test_timed_out=timed_out, audit_rc=rca,
                      test_finished=_now(),
                      minutes=round((time.time() - t0) / 60, 1), downloads_during_test=n_dl,
                      verdict=st.get("verdict"), FPP=st.get("FPP"), NFPP=st.get("NFPP"),
                      audit=a)
        return 0 if rc == 0 and st["finished"] and rca in (0, None) else 1

    with ThreadPoolExecutor(max(1, workers)) as ex:
        rcs = list(ex.map(one, jobs))
    return 0 if all(rc == 0 for rc in rcs) else 1


def show_status(batch, jobs):
    print(f"batch {batch['name']}: {len(jobs)} candidate(s); cache {batch['cache']} "
          f"({_size_mb(batch['cache']):.0f} MB)" if os.path.isdir(batch["cache"])
          else f"batch {batch['name']}: {len(jobs)} candidate(s); no cache yet")
    for j in jobs:
        st = job_state(j)
        pre = {True: "prefetched", False: "prefetch INCOMPLETE",
               None: "not prefetched"}[st["prefetched"]]
        res = "not tested"
        if st["finished"]:
            res = f"{st.get('verdict')}, FPP {_g(st.get('FPP'))}, NFPP {_g(st.get('NFPP'))}"
            if st.get("audit"):
                a = st["audit"]
                res += f"; audit {a['FAIL']} FAIL {a['WARN']} WARN"
        print(f"  {j['label']:28s} {pre:20s} {res}")
    return 0


def _g(x):
    return f"{x:.3g}" if isinstance(x, (int, float)) else "-"


def _size_mb(path):
    n = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n / 2 ** 20


def clean(batch, dry=False):
    """Delete the shared cache folder (only TESScut FITS files are expected there)."""
    cache = batch["cache"]
    if not os.path.isdir(cache):
        print(f"no cache at {cache}")
        return 0
    other = []
    for dp, _, fs in os.walk(cache):
        other += [os.path.join(dp, f) for f in fs
                  if not f.lower().endswith((".fits", ".fits.gz", ".fit"))]
    if other:
        print(f"! {cache} holds files other than FITS cutouts (e.g. {other[0]}); "
              f"not deleted")
        return 1
    mb = _size_mb(cache)
    if dry:
        print(f"would delete {cache} ({mb:.0f} MB)")
        return 0
    shutil.rmtree(cache)
    print(f"deleted {cache} ({mb:.0f} MB)")
    return 0


# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Prefetch, test and audit a list of candidates (see the module docstring).")
    ap.add_argument("batch", help="batch file (JSON)")
    ap.add_argument("--mode", required=True,
                    choices=("prefetch", "test", "both", "status", "clean"))
    ap.add_argument("--only", nargs="+", metavar="LABEL",
                    help="only these candidates (labels)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel tests (default: the batch file's 'workers', else 3)")
    ap.add_argument("--prefetch-workers", type=int, default=1,
                    help="parallel downloads, one TIC per worker (default 1)")
    ap.add_argument("--prefetch-timeout", type=float, default=120.0, metavar="MIN",
                    help="stop one candidate's prefetch after this many minutes "
                         "(default 120); it is tried again in the next pass")
    ap.add_argument("--prefetch-passes", type=int, default=3,
                    help="passes over the candidates whose prefetch failed (default 3)")
    ap.add_argument("--test-timeout", type=float, default=300.0, metavar="MIN",
                    help="stop one test after this many minutes (default 300)")
    ap.add_argument("--test-retries", type=int, default=1,
                    help="reruns of a test that ended without results (default 1)")
    ap.add_argument("--retry-wait", type=float, default=5.0, metavar="MIN",
                    help="pause before a retry or another prefetch pass (default 5)")
    ap.add_argument("--force", action="store_true",
                    help="rerun candidates that already have results")
    ap.add_argument("--no-audit", action="store_true", help="skip the audit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands, run nothing")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    batch = load_batch(a.batch, load_parser())
    for w in batch["warnings"]:
        print(f"! {w}")
    jobs = batch["jobs"]
    if a.only:
        want = set(a.only)
        missing = want - {j["label"] for j in jobs}
        if missing:
            raise SystemExit(f"no candidate labelled {', '.join(sorted(missing))}")
        jobs = [j for j in jobs if j["label"] in want]
    for j in jobs:
        if j["dropped"]:
            print(f"{j['label']}: ignoring {', '.join(j['dropped'])} from the options "
                  f"(set by the batch)")
    if a.mode == "status":
        return show_status(batch, jobs)
    if a.mode == "clean":
        return clean(batch, dry=a.dry_run)
    if not a.dry_run:
        os.makedirs(batch["root"], exist_ok=True)
    status = Status(batch)
    print(f"batch {batch['name']}: {len(jobs)} candidate(s); results in {batch['root']}; "
          f"cache {batch['cache']}", flush=True)
    rc = 0
    if a.mode in ("prefetch", "both"):
        rc |= prefetch(batch, jobs, status, workers=a.prefetch_workers, dry=a.dry_run,
                       passes=a.prefetch_passes, wait_min=a.retry_wait,
                       timeout_min=a.prefetch_timeout)
    if a.mode in ("test", "both"):
        rc |= test(batch, jobs, status, workers=a.workers or batch["workers"],
                   force=a.force, audit=batch["audit"] and not a.no_audit, dry=a.dry_run,
                   timeout_min=a.test_timeout, retries=a.test_retries,
                   wait_min=a.retry_wait)
    print("ALL DONE", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
