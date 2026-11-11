#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphical front end for triceratops_validation.py (Tkinter, standard library).

    python triceratops_gui.py            (in the folder of the script)
    python triceratops_gui.py --script /path/to/triceratops_validation.py

The options are read from the build_parser() function of the validation
script itself (without importing its heavy dependencies), so every option of
the command line appears in the GUI, also options added in later versions
(those land on the "Other" tab).  The GUI builds the command, shows it, runs
it with the Python interpreter you choose (e.g. the 'triplus' conda
environment for TRICERATOPS+), streams the log, and shows the summary and the
output files afterwards.  Settings can be saved as profiles (JSON) and a
command line can be imported.

Needs Python >= 3.8 with tkinter.  The validation itself runs in the selected
interpreter, which needs the pipeline's packages; the GUI does not.
"""
from __future__ import annotations

import __future__ as _future
import argparse
import ast
import datetime as _dt
import glob
import json
import os
import platform
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

__version__ = "1.2"
SCRIPT_NAME = "triceratops_validation.py"
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".triceratops_gui.json")

# --------------------------------------------------------------------------
# Reading the parser of the validation script
# --------------------------------------------------------------------------


def load_parser(script_path):
    """(parser, script_version) from the script's Config + build_parser().

    Only those two definitions are compiled and executed, so lightkurve,
    triceratops etc. are not imported and the GUI starts immediately.
    """
    with open(script_path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    keep = [n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in ("Config", "build_parser")]
    if len([n for n in keep if n.name == "build_parser"]) != 1:
        raise RuntimeError(f"{script_path} has no build_parser() function "
                           f"(the GUI needs triceratops_validation.py >= 4.1)")
    for n in keep:                 # Config as a plain class: only its
        if isinstance(n, ast.ClassDef):    # class-level defaults are used
            n.decorator_list = []
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"__doc__": doc, "argparse": argparse, "dataclass": dataclass,
          "field": field, "Optional": Optional, "os": os,
          "__name__": "triceratops_validation_parser"}
    try:
        import numpy as np  # noqa: F401  (only if a default needs it)
        ns["np"] = np
    except Exception:
        pass
    code = compile(mod, script_path, "exec",
                   flags=_future.annotations.compiler_flag, dont_inherit=True)
    exec(code, ns)
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)', src, re.M)
    return ns["build_parser"](), (m.group(1) if m else "?")


def action_help(a) -> str:
    h = a.help or ""
    if "%(" in h:
        try:
            h = h % dict(default=a.default, prog="triceratops_validation.py",
                         dest=a.dest, choices=a.choices, metavar=a.metavar)
        except Exception:
            pass
    return h


# --------------------------------------------------------------------------
# Layout: which option goes on which tab, friendly labels
# --------------------------------------------------------------------------
TABS = [
    ("Target & ephemeris", [
        "tic", "mission", "kic", "host_tic", "star_source", "stellar",
        "outdir", "ephemeris", "mask_ephemeris", "blind_search", "p_min", "p_max",
        "ephem_search", "no_ephem_scan", "ephem_scan_hours", "ephem_grid_dp",
        "ephem_grid_min_snr", "no_refine_ephemeris", "refine_chi2_max",
        "ephem_adopt_max_sigma", "no_tc_fit"]),
    ("Light curve", [
        "lc_source", "download_dir", "exclude_sectors", "no_auto_exclude_sectors",
        "sector_max_rms", "aperture", "dilution_mode", "raw_aperture_flux",
        "dilution_corrected", "detrend_windows", "err_floor_ppm", "no_inflate"]),
    ("Shape & tests", [
        "ld_fit", "ld_free", "ld_coeffs", "no_secondary_search",
        "secondary_min_snr", "red_noise_scrambles", "red_noise_seed",
        "no_etv", "extra_times"]),
    ("Field", [
        "contrast_curve", "vet_neighbours", "no_difference_imaging",
        "exclude_stars"]),
    ("TRICERATOPS", [
        "n_iter", "n_samples", "no_pool", "sensitivity", "sens_runs",
        "tri_exptime_min", "tri_max_iter", "tri_time_budget", "trilegal_fname",
        "tri_plus"]),
    ("Ground photometry", [
        "ground_lc", "ground_list", "ground_no_clear", "ground_tc_window",
        "ground_min_snr", "ground_depth_tol", "ground_fwhm"]),
]
LABELS = {
    "tic": "TIC ID", "mission": "Mission (TESS / Kepler)", "kic": "KIC ID (Kepler)",
    "host_tic": "Host star for TRICERATOPS (TIC)",
    "star_source": "Stellar parameters from", "stellar": "Manual R M Teff [logg]",
    "outdir": "Output folder", "ephemeris": "Known ephemeris",
    "blind_search": "Blind TLS search", "p_min": "TLS period min [d]",
    "p_max": "TLS period max [d]", "ephem_search": "Ephemeris check",
    "no_ephem_scan": "No ephemeris check", "ephem_scan_hours": "Scan half-width [h]",
    "ephem_grid_dp": "Grid half-range in P [d]",
    "ephem_grid_min_snr": "Grid min. S/N", "no_refine_ephemeris":
    "Do not refine P, T0 from transits", "refine_chi2_max": "Refine: max chi2_red",
    "ephem_adopt_max_sigma": "Refine: max prediction error [T14]",
    "mask_ephemeris": "Mask other planets (P T0 T14_h ...)",
    "no_tc_fit": "Fixed mid-times in per-transit fits",
    "lc_source": "Light-curve source", "download_dir": "TESScut cache folder",
    "exclude_sectors": "Exclude sectors", "no_auto_exclude_sectors":
    "No automatic sector exclusion", "sector_max_rms": "Max sector rms",
    "aperture": "Aperture", "dilution_mode": "Dilution correction",
    "raw_aperture_flux": "Raw aperture flux", "dilution_corrected":
    "Light curve is dilution corrected", "detrend_windows":
    "Detrend windows [d]", "err_floor_ppm": "Error floor [ppm]",
    "no_inflate": "Do not inflate sigma",
    "ld_fit": "Limb-darkened fit", "ld_free": "LD coefficients free",
    "ld_coeffs": "LD coefficients u1 u2", "no_secondary_search":
    "No secondary-eclipse search", "secondary_min_snr": "Secondary min. S/N",
    "red_noise_scrambles": "Red-noise scrambles", "red_noise_seed":
    "Red-noise seed", "no_etv": "No timing analysis", "extra_times":
    "Extra mid-times file",
    "contrast_curve": "Contrast curve", "vet_neighbours": "Vet neighbours",
    "no_difference_imaging": "No difference imaging",
    "exclude_stars": "Exclude stars (TIC)",
    "n_iter": "Min. runs (n_iter)", "n_samples": "Samples per run (N)",
    "no_pool": "Do not pool runs", "sensitivity": "Sensitivity runs",
    "sens_runs": "Runs per sensitivity case", "tri_exptime_min":
    "Model exposure [min]", "tri_max_iter": "Max. runs", "tri_time_budget":
    "Time budget [s]", "trilegal_fname": "TRILEGAL / field-star file",
    "tri_plus": "TRICERATOPS+",
    "ground_lc": "Ground light curves", "ground_list": "Ground list file",
    "ground_no_clear": "Do not remove cleared stars", "ground_tc_window":
    "Mid-time window [min]", "ground_min_snr": "Detection S/N",
    "ground_depth_tol": "Depth tolerance", "ground_fwhm": "Default seeing [\"]",
}
DIR_OPTIONS = {"outdir", "download_dir"}
FILE_OPTIONS = {"contrast_curve", "trilegal_fname", "extra_times", "ground_list"}
EPHEM_FIELDS = ("P [d]", "T0 [BTJD/BKJD or BJD]", "Depth [ppm]", "T14 [h]")
GROUND_BANDS = ["", "R", "V", "B", "I", "g", "r", "i", "z", "J", "H", "K",
                "CBB", "CV", "TESS"]
GROUND_TIME = ["", "BJD_TDB", "JD_UTC", "HJD_UTC"]


def nice_label(dest):
    return LABELS.get(dest, dest.replace("_", " ").capitalize())


# --------------------------------------------------------------------------
# Small widgets
# --------------------------------------------------------------------------
class ToolTip:
    def __init__(self, widget, text, wrap=420):
        self.widget, self.text, self.wrap, self.tip = widget, text, wrap, None
        if text:
            widget.bind("<Enter>", self.show, add="+")
            widget.bind("<Leave>", self.hide, add="+")

    def show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", background="#ffffe6",
                 relief="solid", borderwidth=1, wraplength=self.wrap,
                 font=("TkDefaultFont", 9)).pack(ipadx=4, ipady=2)

    def hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _fmt_default(a):
    d = a.default
    if d is None or d is False or d == argparse.SUPPRESS:
        return ""
    if isinstance(d, (list, tuple)):
        return " ".join(str(x) for x in d)
    return str(d)


class OptionWidget:
    """One argparse option as a row of widgets on a tab."""

    def __init__(self, parent, row, action, on_change):
        self.a = action
        self.dest = action.dest
        self.flag = action.option_strings[0]
        self.kind = type(action).__name__
        self.on_change = on_change
        lab = ttk.Label(parent, text=nice_label(self.dest))
        lab.grid(row=row, column=0, sticky="w", padx=(6, 8), pady=2)
        help_ = action_help(action)
        ToolTip(lab, f"{self.flag}\n{help_}")
        self.vars = []
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=1, sticky="we", pady=2)
        self.frame = frm
        if self.kind == "_StoreTrueAction":
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(frm, variable=v)
            cb.pack(side="left")
            ToolTip(cb, help_)
            self.vars = [v]
        elif self.dest == "ephemeris":
            for i, name in enumerate(EPHEM_FIELDS):
                ttk.Label(frm, text=name).grid(row=0, column=2 * i, padx=(0, 3))
                v = tk.StringVar()
                e = ttk.Entry(frm, textvariable=v, width=13)
                e.grid(row=0, column=2 * i + 1, padx=(0, 10))
                ToolTip(e, help_)
                self.vars.append(v)
        elif action.choices:
            v = tk.StringVar(value=str(action.default))
            cb = ttk.Combobox(frm, textvariable=v, values=list(action.choices),
                              state="readonly", width=14)
            cb.pack(side="left")
            ToolTip(cb, help_)
            self.vars = [v]
        else:
            v = tk.StringVar()
            wide = self.dest in DIR_OPTIONS | FILE_OPTIONS
            e = ttk.Entry(frm, textvariable=v, width=60 if wide else 22)
            e.pack(side="left", fill="x", expand=wide)
            ToolTip(e, help_)
            if self.dest in DIR_OPTIONS:
                ttk.Button(frm, text="…", width=3,
                           command=lambda: self._browse(dir_=True)).pack(side="left", padx=3)
            elif self.dest in FILE_OPTIONS:
                ttk.Button(frm, text="…", width=3,
                           command=lambda: self._browse(dir_=False)).pack(side="left", padx=3)
            self.vars = [v]
        d = _fmt_default(action) if self.dest != "tic" else ""
        hint = ttk.Label(parent, text=(f"default {d}" if d else ""),
                         foreground="#777777")
        hint.grid(row=row, column=2, sticky="w", padx=6)
        for v in self.vars:
            v.trace_add("write", lambda *_: self.on_change())

    def _browse(self, dir_):
        cur = self.vars[0].get()
        if dir_:
            p = filedialog.askdirectory(initialdir=cur or os.getcwd())
        else:
            p = filedialog.askopenfilename(initialdir=os.path.dirname(cur) or os.getcwd())
        if p:
            self.vars[0].set(p)

    # value <-> widgets
    def get_value(self):
        if self.kind == "_StoreTrueAction":
            return bool(self.vars[0].get())
        if self.dest == "ephemeris":
            return [v.get().strip() for v in self.vars]
        return self.vars[0].get().strip()

    def set_value(self, val):
        if self.kind == "_StoreTrueAction":
            self.vars[0].set(bool(val))
        elif self.dest == "ephemeris":
            vals = list(val or []) + [""] * 4
            for v, x in zip(self.vars, vals[:4]):
                v.set("" if x is None else str(x))
        elif self.a.choices:
            self.vars[0].set(str(val) if val not in (None, "") else str(self.a.default))
        else:
            if isinstance(val, (list, tuple)):
                val = " ".join(str(x) for x in val)
            self.vars[0].set("" if val is None else str(val))

    def reset(self):
        if self.kind == "_StoreTrueAction":
            self.set_value(False)
        elif self.dest == "ephemeris":
            self.set_value([])
        elif self.a.choices:
            self.set_value(self.a.default)
        else:
            self.set_value("")

    def args(self, notes):
        """Command-line tokens for this option ([] = default)."""
        if self.kind == "_StoreTrueAction":
            return [self.flag] if self.vars[0].get() else []
        if self.dest == "ephemeris":
            vals = self.get_value()
            if not any(vals):
                return []
            if not all(vals):
                raise ValueError("Known ephemeris: give all four values "
                                 "(P, T0, depth, T14) or none")
            nums = []
            for name, x in zip(EPHEM_FIELDS, vals):
                try:
                    nums.append(float(x))
                except ValueError:
                    raise ValueError(f"Known ephemeris: '{x}' is not a number ({name})")
            if nums[1] > 2.4e6:
                notes.append(f"T0 {nums[1]} is a full BJD: the script converts it "
                             f"to BTJD (BJD - 2457000) or, for Kepler, BKJD "
                             f"(BJD - 2454833)")
            elif 5.0e4 < nums[1] < 1.0e5:
                raise ValueError("Known ephemeris: T0 looks like an MJD; give "
                                 "BTJD (BJD - 2457000), BKJD (BJD - 2454833, "
                                 "Kepler) or the full BJD")
            out = [v.strip() for v in vals]          # keep the digits as typed
            return [self.flag] + out
        val = self.get_value()
        if self.a.choices:
            return [] if val in ("", str(self.a.default)) else [self.flag, val]
        if val == "":
            return []
        toks = val.split() if self.a.nargs not in (None, "?") else [val]
        if self.a.nargs not in (None, "?", "+", "*") and isinstance(self.a.nargs, int) \
                and len(toks) != self.a.nargs:
            raise ValueError(f"{nice_label(self.dest)}: {self.a.nargs} values expected")
        typ = self.a.type
        if typ in (int, float):
            for t in toks:
                try:
                    typ(t)
                except ValueError:
                    raise ValueError(f"{nice_label(self.dest)}: '{t}' is not "
                                     f"{'an integer' if typ is int else 'a number'}")
        if self.a.nargs in (None, "?") and _fmt_default(self.a) == val:
            return []
        return [self.flag] + toks


# --------------------------------------------------------------------------
# Ground light curves: spec <-> fields
# --------------------------------------------------------------------------
GROUND_FIELDS = [("path", "File"), ("band", "Band"), ("star", "Aperture star (TIC)"),
                 ("ap", "Aperture radius [\"]"), ("fwhm", "Seeing FWHM [\"]"),
                 ("obs", "Observer"), ("time", "Time system"),
                 ("fcol", "Flux column"), ("ecol", "Error column"),
                 ("tcol", "Time column"), ("mag", "Magnitudes (1)"),
                 ("detrend", "Detrend"), ("ra", "Centre RA [deg]"),
                 ("dec", "Centre Dec [deg]"), ("pixscale", "Pixel scale [\"/px]"),
                 ("plus", "TRICERATOPS+ band"), ("label", "Label")]
_GKEY = {"band": "band", "filter": "band", "filt": "band", "star": "star",
         "tic": "star", "center": "star", "ra": "ra", "dec": "dec",
         "ap": "ap", "aperture": "ap", "r_ap": "ap", "radius": "ap",
         "fwhm": "fwhm", "seeing": "fwhm", "pixscale": "pixscale",
         "scale": "pixscale", "obs": "obs", "observer": "obs", "obscode": "obs",
         "tcol": "tcol", "time_col": "tcol", "fcol": "fcol", "flux_col": "fcol",
         "column": "fcol", "ecol": "ecol", "err_col": "ecol", "time": "time",
         "timesys": "time", "time_system": "time", "mag": "mag",
         "detrend": "detrend", "plus": "plus", "plus_band": "plus",
         "label": "label"}


def parse_spec(text):
    """Same grammar as triceratops_validation.parse_ground_spec (dict)."""
    text = re.sub(r"(?i)\b(star|tic|center)\s*=\s*TIC\s*(\d+)", r"\1=\2", text)
    toks = []
    for m in re.finditer(r'"([^"]*)"|(\S+)', text):
        t = m.group(1) if m.group(1) is not None else m.group(2)
        if m.group(1) is None and ("=" in t or "," in t):
            toks.extend([x for x in t.split(",") if x])
        else:
            toks.append(t)
    if not toks:
        return {}
    d = {"path": toks[0]}
    last = None
    for t in toks[1:]:
        if "=" not in t:
            if last == "detrend":
                d["detrend"] = d.get("detrend", "") + "," + t
            continue
        k, v = t.split("=", 1)
        key = _GKEY.get(k.strip().lower())
        if key:
            d[key] = v.strip()
            last = key
    return d


def build_spec(d):
    path = d.get("path", "").strip()
    if not path:
        return ""
    parts = [f'"{path}"' if (" " in path or "," in path) else path]
    for k, _ in GROUND_FIELDS[1:]:
        v = str(d.get(k, "") or "").strip()
        if v:
            parts.append(f"{k}={v}")
    return ",".join(parts)


class GroundDialog(tk.Toplevel):
    def __init__(self, master, values=None):
        super().__init__(master)
        self.title("Ground light curve")
        self.transient(master)
        self.result = None
        self.vars = {}
        values = values or {}
        hints = {"band": "B V R I (Rc = R), g r i z (Sloan), J H K",
                 "star": "empty = target",
                 "ap": "aperture RADIUS in arcsec",
                 "time": "default BJD_TDB",
                 "fcol": "e.g. rel_flux_T2 for star T2 in AstroImageJ",
                 "detrend": "auto | poly0/1/2, plus columns: poly1,AIRMASS",
                 "plus": "override the band passed to TRICERATOPS+",
                 "mag": "1 if the flux column holds magnitudes",
                 "ra": "instead of an aperture star",
                 "pixscale": "AstroImageJ: converts Source_Radius, FWHM_Mean"}
        for i, (k, lab) in enumerate(GROUND_FIELDS):
            ttk.Label(self, text=lab).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            v = tk.StringVar(value=values.get(k, ""))
            self.vars[k] = v
            if k == "band":
                w = ttk.Combobox(self, textvariable=v, values=GROUND_BANDS, width=12)
            elif k == "time":
                w = ttk.Combobox(self, textvariable=v, values=GROUND_TIME,
                                 width=12, state="readonly")
            else:
                w = ttk.Entry(self, textvariable=v, width=58 if k == "path" else 22)
            w.grid(row=i, column=1, sticky="w", padx=4, pady=2)
            if k == "path":
                ttk.Button(self, text="…", width=3, command=self._browse).grid(
                    row=i, column=2, padx=3)
            if k in hints:
                ttk.Label(self, text=hints[k], foreground="#777777").grid(
                    row=i, column=2 if k != "path" else 3, sticky="w", padx=4)
        b = ttk.Frame(self)
        b.grid(row=len(GROUND_FIELDS), column=0, columnspan=4, pady=8)
        ttk.Button(b, text="OK", command=self._ok).pack(side="left", padx=5)
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="left", padx=5)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()

    def _browse(self):
        p = filedialog.askopenfilename(parent=self, title="Ground light curve")
        if p:
            self.vars["path"].set(p)

    def _ok(self):
        d = {k: v.get().strip() for k, v in self.vars.items()}
        if not d["path"]:
            messagebox.showerror("Ground light curve", "Choose a file.", parent=self)
            return
        for k in ("ap", "fwhm", "ra", "dec", "pixscale"):
            if d[k]:
                try:
                    float(d[k])
                except ValueError:
                    messagebox.showerror("Ground light curve",
                                         f"{k}: '{d[k]}' is not a number", parent=self)
                    return
        if d["star"]:
            s = d["star"].upper().replace("TIC", "").strip()
            if not s.isdigit():
                messagebox.showerror("Ground light curve",
                                     "Aperture star: give a TIC number", parent=self)
                return
            d["star"] = s
        self.result = d
        self.destroy()


class GroundPanel(ttk.LabelFrame):
    COLS = ("path", "band", "star", "ap", "fwhm", "obs", "other")

    def __init__(self, master, on_change):
        super().__init__(master, text="Ground light curves (--ground-lc)")
        self.on_change = on_change
        self.specs = []
        self.tree = ttk.Treeview(self, columns=self.COLS, show="headings", height=7)
        widths = (330, 50, 90, 60, 60, 80, 220)
        heads = ("File", "Band", "Star", "Ap [\"]", "FWHM", "Observer", "Other options")
        for c, w, h in zip(self.COLS, widths, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, stretch=(c in ("path", "other")))
        self.tree.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=4, pady=4)
        self.tree.bind("<Double-1>", lambda e: self.edit())
        for i, (t, cmd) in enumerate((("Add files…", self.add),
                                      ("Edit…", self.edit),
                                      ("Remove", self.remove),
                                      ("Up", lambda: self.move(-1)),
                                      ("Down", lambda: self.move(1)),
                                      ("Import list…", self.import_list),
                                      ("Save as list…", self.save_list))):
            ttk.Button(self, text=t, command=cmd).grid(row=1, column=i, padx=3, pady=3, sticky="w")
        self.columnconfigure(7, weight=1)
        self.rowconfigure(0, weight=1)

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for s in self.specs:
            d = parse_spec(s)
            other = ",".join(f"{k}={d[k]}" for k, _ in GROUND_FIELDS
                             if k not in self.COLS and d.get(k))
            self.tree.insert("", "end", values=(
                os.path.basename(d.get("path", "")), d.get("band", ""), d.get("star", ""),
                d.get("ap", ""), d.get("fwhm", ""), d.get("obs", ""), other))
        self.on_change()

    def _sel(self):
        s = self.tree.selection()
        return self.tree.index(s[0]) if s else None

    def add(self):
        paths = filedialog.askopenfilenames(title="Ground light curves")
        for p in paths:
            dlg = GroundDialog(self, {"path": p})
            self.wait_window(dlg)
            if dlg.result:
                self.specs.append(build_spec(dlg.result))
        self.refresh()

    def edit(self):
        i = self._sel()
        if i is None:
            return
        dlg = GroundDialog(self, parse_spec(self.specs[i]))
        self.wait_window(dlg)
        if dlg.result:
            self.specs[i] = build_spec(dlg.result)
            self.refresh()

    def remove(self):
        i = self._sel()
        if i is not None:
            del self.specs[i]
            self.refresh()

    def move(self, d):
        i = self._sel()
        if i is None or not (0 <= i + d < len(self.specs)):
            return
        self.specs[i], self.specs[i + d] = self.specs[i + d], self.specs[i]
        self.refresh()
        self.tree.selection_set(self.tree.get_children()[i + d])

    def import_list(self):
        p = filedialog.askopenfilename(title="Ground list file")
        if not p:
            return
        base = os.path.dirname(os.path.abspath(p))
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip() if line.lstrip().startswith('"') \
                    else line.split("#", 1)[0].strip()
                if not line:
                    continue
                d = parse_spec(line)
                if d.get("path") and not os.path.isabs(d["path"]):
                    d["path"] = os.path.join(base, d["path"])
                self.specs.append(build_spec(d))
        self.refresh()

    def save_list(self):
        if not self.specs:
            return
        p = filedialog.asksaveasfilename(title="Save ground list",
                                         defaultextension=".txt")
        if not p:
            return
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# ground light curves for triceratops_validation.py --ground-list\n")
            for s in self.specs:
                d = parse_spec(s)
                path = d.pop("path")
                fh.write(f'"{path}" ' + " ".join(f"{k}={v}" for k, v in d.items() if v) + "\n")

    def args(self):
        out = []
        for s in self.specs:
            out += ["--ground-lc", s]
        return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def open_path(path):
    """Open a file or folder with the system viewer (Windows, WSL, macOS, Linux)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif "microsoft" in platform.release().lower():
            try:
                w = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
                subprocess.Popen(["explorer.exe", w])
            except Exception:
                subprocess.Popen(["xdg-open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        messagebox.showerror("Open", f"Could not open {path}:\n{exc}")


def find_interpreters():
    """sys.executable plus the conda environments that can be found."""
    out = [sys.executable]
    home = os.path.expanduser("~")
    pats = []
    for base in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda"):
        pats += [os.path.join(home, base, "envs", "*", "bin", "python"),
                 os.path.join(home, base, "bin", "python"),
                 os.path.join(home, base, "envs", "*", "python.exe"),
                 os.path.join(home, base, "python.exe")]
    pats += ["/opt/conda/envs/*/bin/python", "/opt/conda/bin/python"]
    for p in pats:
        for f in sorted(glob.glob(p)):
            if os.path.isfile(f) and f not in out:
                out.append(f)
    return out


def jd_to_utc_string(jd_tdb):
    """Approximate UTC calendar date of a BJD_TDB (no barycentric correction:
    accurate to about +-8 min, which is fine for planning)."""
    t = _dt.datetime(2000, 1, 1, 12, 0, 0) + _dt.timedelta(days=jd_tdb - 2451545.0)
    t -= _dt.timedelta(seconds=69.184)          # TDB/TT - UTC (leap seconds 2017+)
    return t.strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self, script_path):
        super().__init__()
        self.title("TRICERATOPS validation")
        self.geometry("1180x800")
        self.minsize(900, 600)
        self.proc = None
        self.q = queue.Queue()
        self.t_start = None
        self.script_path = os.path.abspath(script_path)
        try:
            self.parser, self.script_version = load_parser(self.script_path)
        except Exception as exc:
            messagebox.showerror("TRICERATOPS GUI", str(exc))
            raise
        self.title(f"TRICERATOPS validation — {os.path.basename(script_path)} "
                   f"v{self.script_version} — GUI v{__version__}")
        self.widgets = {}
        self._cmd_after = None
        self._build_menu()
        self._build_top()
        self._build_notebook()
        self._build_bottom()
        self.load_settings(SETTINGS_FILE, quiet=True)
        self.update_command()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._poll)

    # ---- layout -----------------------------------------------------------
    def _build_menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="New (reset options)", command=self.reset)
        fm.add_command(label="Open profile…", command=self.open_profile)
        fm.add_command(label="Save profile…", command=self.save_profile)
        fm.add_command(label="Save target profile (options only)…",
                       command=lambda: self.save_profile(options_only=True))
        fm.add_separator()
        fm.add_command(label="Import command line…", command=self.import_command)
        fm.add_command(label="Copy command", command=self.copy_command)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.on_close)
        m.add_cascade(label="File", menu=fm)
        self.prof_menu = tk.Menu(m, tearoff=0, postcommand=self._fill_profiles_menu)
        m.add_cascade(label="Profiles", menu=self.prof_menu)
        tm = tk.Menu(m, tearoff=0)
        tm.add_command(label="Check interpreter (triceratops / TRICERATOPS+)",
                       command=self.check_interpreter)
        tm.add_command(label="Predicted transit times…", command=self.transit_times)
        tm.add_command(label="Run regression tests", command=self.run_tests)
        tm.add_separator()
        tm.add_command(label="Audit output folder (current target)",
                       command=self.run_audit)
        tm.add_command(label="Audit another output folder…",
                       command=lambda: self.run_audit(ask=True))
        m.add_cascade(label="Tools", menu=tm)
        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Observer's Guide (PDF)", command=self.open_guide)
        hm.add_command(label="Technical manual (PDF)", command=self.open_manual)
        hm.add_command(label="Command-line help", command=self.show_cli_help)
        hm.add_command(label="About", command=self.about)
        m.add_cascade(label="Help", menu=hm)
        self.config(menu=m)

    def _build_top(self):
        top = ttk.LabelFrame(self, text="Run environment")
        top.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(top, text="Python interpreter").grid(row=0, column=0, sticky="w", padx=6)
        self.interp = tk.StringVar(value=sys.executable)
        cb = ttk.Combobox(top, textvariable=self.interp, values=find_interpreters(), width=70)
        cb.grid(row=0, column=1, sticky="we", padx=4, pady=2)
        ToolTip(cb, "Python that runs the validation, e.g. the conda environment "
                    "with triceratops 1.1, or the 'triplus' environment with "
                    "TRICERATOPS+.")
        ttk.Button(top, text="…", width=3, command=self._browse_interp).grid(row=0, column=2)
        ttk.Button(top, text="Check", command=self.check_interpreter).grid(row=0, column=3, padx=4)
        self.interp_info = ttk.Label(top, text="", foreground="#335")
        self.interp_info.grid(row=0, column=4, sticky="w", padx=6)
        ttk.Label(top, text="Working folder").grid(row=1, column=0, sticky="w", padx=6)
        self.workdir = tk.StringVar(value=os.path.dirname(self.script_path))
        e = ttk.Entry(top, textvariable=self.workdir, width=72)
        e.grid(row=1, column=1, sticky="we", padx=4, pady=2)
        ToolTip(e, "The run starts here; without an output folder the results "
                   "go to TIC<tic>/ in this folder.")
        ttk.Button(top, text="…", width=3, command=self._browse_workdir).grid(row=1, column=2)
        top.columnconfigure(1, weight=1)
        self.interp.trace_add("write", lambda *_: self.update_command())
        self.workdir.trace_add("write", lambda *_: self.update_command())

    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=4)
        acts = {a.dest: a for a in self.parser._actions if a.option_strings
                and a.dest != "help"}
        placed = set()
        for title, dests in TABS:
            frame = self._scroll_tab(title)
            r = 0
            for d in dests:
                if d not in acts:
                    continue
                placed.add(d)
                if d == "ground_lc":
                    self.ground = GroundPanel(frame, self.update_command)
                    self.ground.grid(row=r, column=0, columnspan=3, sticky="nsew",
                                     padx=4, pady=4)
                    r += 1
                    continue
                self.widgets[d] = OptionWidget(frame, r, acts[d], self.update_command)
                r += 1
            frame.columnconfigure(1, weight=1)
            if title == "Target & ephemeris":
                ttk.Label(frame, foreground="#555", justify="left", text=(
                    "Without a known ephemeris the script runs a blind TLS search.\n"
                    "T0 may be given as BTJD (BJD - 2457000) or as the full BJD_TDB; "
                    "the GUI converts.\nDepth: on the target's flux scale with the "
                    "default per-sector dilution correction; the aperture depth with "
                    "'Dilution correction = none' or TRICERATOPS+.")).grid(
                    row=r, column=0, columnspan=3, sticky="w", padx=6, pady=10)
        rest = [a for d, a in acts.items() if d not in placed]
        if rest:
            frame = self._scroll_tab("Other")
            for r, a in enumerate(rest):
                if type(a).__name__ == "_AppendAction":
                    continue
                self.widgets[a.dest] = OptionWidget(frame, r, a, self.update_command)
            frame.columnconfigure(1, weight=1)
        if not hasattr(self, "ground"):
            self.ground = GroundPanel(ttk.Frame(self), self.update_command)
        # run tab
        run = ttk.Frame(self.nb)
        self.nb.add(run, text="Run & log")
        bar = ttk.Frame(run)
        bar.pack(fill="x", pady=4)
        self.btn_run = ttk.Button(bar, text="▶ Run", command=self.run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(bar, text="■ Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(bar, text="Save log…", command=self.save_log).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=4)
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Follow", variable=self.autoscroll).pack(side="left", padx=8)
        self.status = ttk.Label(bar, text="idle")
        self.status.pack(side="left", padx=12)
        self.pbar = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.pbar.pack(side="right", padx=6)
        lf = ttk.Frame(run)
        lf.pack(fill="both", expand=True)
        self.log = tk.Text(lf, wrap="none", font=("Courier", 9), undo=False)
        ys = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(lf, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="we")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        self.log.tag_configure("warn", foreground="#b03000")
        self.log.tag_configure("head", foreground="#00308f", font=("Courier", 9, "bold"))
        self.log.tag_configure("gui", foreground="#666666")
        self.log.tag_configure("good", foreground="#006400")
        # results tab
        res = ttk.Frame(self.nb)
        self.nb.add(res, text="Results")
        rb = ttk.Frame(res)
        rb.pack(fill="x", pady=4)
        ttk.Button(rb, text="Refresh", command=self.refresh_results).pack(side="left", padx=4)
        ttk.Button(rb, text="Open output folder", command=lambda: open_path(self.outdir())).pack(side="left", padx=4)
        b = ttk.Button(rb, text="Audit run", command=self.run_audit)
        b.pack(side="left", padx=4)
        ToolTip(b, "Runs triceratops_audit.py on the output folder: deterministic "
                   "checks (no language model) that try to break the claim, plus an "
                   "independent recomputation from the saved light curves. Writes "
                   "audit_<TIC>.pdf/.txt/.json and opens the PDF. The run's own "
                   "files are not changed.")
        self.res_label = ttk.Label(rb, text="")
        self.res_label.pack(side="left", padx=10)
        pw = ttk.Panedwindow(res, orient="horizontal")
        pw.pack(fill="both", expand=True)
        f1 = ttk.LabelFrame(pw, text="Summary (summary_<TIC>.json)")
        self.sum_tree = ttk.Treeview(f1, columns=("v",), show="tree headings")
        self.sum_tree.heading("#0", text="Item")
        self.sum_tree.heading("v", text="Value")
        self.sum_tree.column("#0", width=260)
        self.sum_tree.column("v", width=260)
        self.sum_tree.pack(fill="both", expand=True)
        f2 = ttk.LabelFrame(pw, text="Output files (double-click to open)")
        self.file_tree = ttk.Treeview(f2, columns=("size", "time"), show="tree headings")
        self.file_tree.heading("#0", text="File")
        self.file_tree.heading("size", text="kB")
        self.file_tree.heading("time", text="Modified")
        self.file_tree.column("#0", width=330)
        self.file_tree.column("size", width=60, anchor="e")
        self.file_tree.column("time", width=130)
        self.file_tree.pack(fill="both", expand=True)
        self.file_tree.bind("<Double-1>", self._open_selected_file)
        pw.add(f1, weight=1)
        pw.add(f2, weight=1)

    def _scroll_tab(self, title):
        outer = ttk.Frame(self.nb)
        self.nb.add(outer, text=title)
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return inner

    def _build_bottom(self):
        bot = ttk.LabelFrame(self, text="Command")
        bot.pack(fill="x", padx=8, pady=(2, 8))
        self.cmd_text = tk.Text(bot, height=4, wrap="word", font=("Courier", 9),
                                background="#f4f4f4")
        csb = ttk.Scrollbar(bot, orient="vertical", command=self.cmd_text.yview)
        self.cmd_text.configure(yscrollcommand=csb.set)
        self.cmd_text.pack(side="left", fill="x", expand=True, padx=(4, 0), pady=4)
        csb.pack(side="left", fill="y", pady=4)
        fr = ttk.Frame(bot)
        fr.pack(side="right", padx=4)
        ttk.Button(fr, text="Copy", command=self.copy_command).pack(fill="x", pady=1)
        ttk.Button(fr, text="▶ Run", command=self.run).pack(fill="x", pady=1)
        self.cmd_msg = ttk.Label(self, text="", foreground="#b03000")
        self.cmd_msg.pack(fill="x", padx=10)

    def _browse_interp(self):
        p = filedialog.askopenfilename(title="Python interpreter")
        if p:
            self.interp.set(p)

    def _browse_workdir(self):
        p = filedialog.askdirectory(initialdir=self.workdir.get())
        if p:
            self.workdir.set(p)

    # ---- command ----------------------------------------------------------
    def build_args(self):
        notes = []
        args = []
        tic = self.widgets["tic"].get_value() if "tic" in self.widgets else ""
        kepler = self._mission() == "kepler"
        kic = self.widgets["kic"].get_value() if "kic" in self.widgets else ""
        if kepler and not kic:
            raise ValueError("Kepler mission: give the KIC ID of the target")
        if not tic and not kepler:
            raise ValueError("Give the TIC ID of the target")
        if tic and not tic.isdigit():
            raise ValueError("TIC ID must be a whole number")
        if tic:
            args += ["--tic", tic]
        for d, w in self.widgets.items():
            if d == "tic":
                continue
            args += w.args(notes)
        args += self.ground.args()
        g = lambda d: self.widgets[d].get_value() if d in self.widgets else None  # noqa: E731
        if g("raw_aperture_flux") and g("dilution_corrected"):
            raise ValueError("'Raw aperture flux' and 'Light curve is dilution "
                             "corrected' exclude each other")
        if g("blind_search") and any(g("ephemeris") or []):
            raise ValueError("'Blind TLS search' and a known ephemeris exclude each other")
        if g("ld_free") and g("ld_coeffs"):
            raise ValueError("'LD coefficients free' and fixed LD coefficients "
                             "exclude each other")
        # final check with the script's own parser
        import contextlib
        import io
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                self.parser.parse_args(args)
        except SystemExit:
            msg = err.getvalue().strip().splitlines()
            raise ValueError(msg[-1] if msg else "invalid options")
        return args, notes

    def command_list(self):
        args, notes = self.build_args()
        return [self.interp.get().strip() or sys.executable, "-u",
                self.script_path] + args, notes

    def command_string(self, cmd):
        if sys.platform.startswith("win"):
            return subprocess.list2cmdline(cmd)
        return " ".join(shlex.quote(c) for c in cmd)

    def update_command(self):
        if self._cmd_after:
            self.after_cancel(self._cmd_after)
        self._cmd_after = self.after(150, self._update_command_now)

    def _update_command_now(self):
        self._cmd_after = None
        try:
            cmd, notes = self.command_list()
            # show without -u
            show = [c for i, c in enumerate(cmd) if not (i == 1 and c == "-u")]
            s = self.command_string([show[0], os.path.basename(show[1])] + show[2:])
            self.cmd_msg.configure(text="; ".join(notes), foreground="#335")
        except ValueError as exc:
            s = ""
            self.cmd_msg.configure(text=str(exc), foreground="#b03000")
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", s)
        self.cmd_text.configure(state="disabled")

    def copy_command(self):
        s = self.cmd_text.get("1.0", "end").strip()
        if s:
            self.clipboard_clear()
            self.clipboard_append(s)

    # ---- running ----------------------------------------------------------
    def _log(self, text, tag=None):
        if tag is None:
            s = text.lstrip()
            if s.startswith("!") or "WARNING" in text or "Traceback" in text \
                    or "Error" in text[:40]:
                tag = "warn"
            elif s.startswith("---") or s.startswith("===") or s.startswith("SUMMARY"):
                tag = "head"
        self.log.insert("end", text, tag)
        if self.autoscroll.get():
            self.log.see("end")

    def run(self):
        if self.proc is not None:
            return
        try:
            cmd, notes = self.command_list()
        except ValueError as exc:
            messagebox.showerror("Options", str(exc))
            return
        wd = self.workdir.get().strip() or os.path.dirname(self.script_path)
        if not os.path.isdir(wd):
            messagebox.showerror("Working folder", f"{wd} does not exist")
            return
        self.nb.select(len(self.nb.tabs()) - 2)
        self._log(f"\n[GUI] {time.strftime('%Y-%m-%d %H:%M:%S')}  in {wd}\n", "gui")
        self._log("[GUI] " + self.command_string(cmd) + "\n", "gui")
        for n in notes:
            self._log(f"[GUI] note: {n}\n", "gui")
        env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
                   PYTHONIOENCODING="utf-8")
        try:
            self.proc = subprocess.Popen(cmd, cwd=wd, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace",
                                         bufsize=1, env=env)
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Run", f"Could not start:\n{exc}")
            return
        self.t_start = time.time()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.pbar.start(12)
        self.save_settings(SETTINGS_FILE)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(("line", line))
        proc.wait()
        self.q.put(("done", proc.returncode))

    def _poll(self):
        n = 0
        try:
            while n < 500:
                kind, val = self.q.get_nowait()
                n += 1
                if kind == "line":
                    self._log(val)
                    s = val.strip()
                    if s.startswith("---") or s.endswith("...") or s.startswith("Running"):
                        self._phase = s.strip("- ")[:70]
                elif kind == "done":
                    self._finished(val)
                elif kind == "info":
                    self.interp_info.configure(text=val)
        except queue.Empty:
            pass
        if self.proc is not None and self.t_start:
            el = int(time.time() - self.t_start)
            self.status.configure(text=f"running {el // 60:d}:{el % 60:02d}  "
                                       f"{getattr(self, '_phase', '')}")
        self.after(100, self._poll)

    def _finished(self, rc):
        el = int(time.time() - (self.t_start or time.time()))
        self.proc = None
        self.pbar.stop()
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        ok = rc == 0
        self.status.configure(text=f"{'finished' if ok else 'stopped / failed'} "
                                   f"(exit code {rc}) after {el // 60:d}:{el % 60:02d}")
        self._log(f"[GUI] exit code {rc}\n", "good" if ok else "warn")
        self.refresh_results()
        if ok:
            self.nb.select(len(self.nb.tabs()) - 1)
            self._open_audit_pdf()
        else:
            self._after_audit = None

    def stop(self):
        if self.proc is None:
            return
        if not messagebox.askyesno("Stop", "Stop the running validation?"):
            return
        try:
            self.proc.terminate()
        except Exception:
            pass
        self._log("[GUI] stop requested\n", "warn")

    def save_log(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         initialfile="gui_log.txt")
        if p:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self.log.get("1.0", "end"))

    def run_tests(self):
        if self.proc is not None:
            return
        wd = os.path.dirname(self.script_path)
        cmd = [self.interp.get().strip() or sys.executable, "-m", "pytest", "-q", "tests"]
        self.nb.select(len(self.nb.tabs()) - 2)
        self._log(f"\n[GUI] regression tests: {self.command_string(cmd)}\n", "gui")
        try:
            self.proc = subprocess.Popen(cmd, cwd=wd, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1)
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Tests", str(exc))
            return
        self.t_start = time.time()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.pbar.start(12)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def run_audit(self, ask=False):
        """Run triceratops_audit.py on an output folder (Tools menu / Results tab)."""
        if self.proc is not None:
            messagebox.showinfo("Audit", "Wait until the current run has finished.")
            return
        audit = os.path.join(os.path.dirname(self.script_path), "triceratops_audit.py")
        if not os.path.isfile(audit):
            messagebox.showerror("Audit", f"{audit} not found: put triceratops_audit.py "
                                          f"next to {os.path.basename(self.script_path)}.")
            return
        tic = ""
        if ask:
            od = filedialog.askdirectory(title="Output folder of a validation run",
                                         initialdir=self.outdir() if os.path.isdir(self.outdir())
                                         else self.workdir.get())
            if not od:
                return
        else:
            od = self.outdir()
            tic = self.widgets["tic"].get_value() if "tic" in self.widgets else ""
        if not os.path.isdir(od):
            messagebox.showerror("Audit", f"{od} does not exist (run the validation first).")
            return
        cmd = [self.interp.get().strip() or sys.executable, audit, od]
        if tic:
            cmd += ["--tic", str(tic)]
        self.nb.select(len(self.nb.tabs()) - 2)
        self._log(f"\n[GUI] audit: {self.command_string(cmd)}\n", "gui")
        try:
            self.proc = subprocess.Popen(cmd, cwd=od, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1)
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Audit", str(exc))
            return
        self._after_audit = (od, tic)
        self.t_start = time.time()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.pbar.start(12)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _open_audit_pdf(self):
        od, tic = getattr(self, "_after_audit", (None, None)) or (None, None)
        self._after_audit = None
        if not od:
            return
        pdfs = ([os.path.join(od, f"audit_{tic}.pdf")] if tic else
                sorted(glob.glob(os.path.join(od, "audit_*.pdf")), key=os.path.getmtime))
        pdfs = [p for p in pdfs if os.path.isfile(p)]
        if pdfs:
            open_path(pdfs[-1])

    # ---- results ----------------------------------------------------------
    def outdir(self):
        od = self.widgets["outdir"].get_value() if "outdir" in self.widgets else ""
        wd = self.workdir.get().strip() or os.path.dirname(self.script_path)
        if od:
            return od if os.path.isabs(od) else os.path.join(wd, od)
        tic = self.widgets["tic"].get_value() if "tic" in self.widgets else ""
        if self._mission() == "kepler":
            kic = self.widgets["kic"].get_value() if "kic" in self.widgets else ""
            if kic:
                return os.path.join(wd, f"KIC{kic}")
        return os.path.join(wd, f"TIC{tic}")

    def _mission(self):
        w = self.widgets.get("mission")
        return (w.get_value() if w else "tess") or "tess"

    def refresh_results(self):
        od = self.outdir()
        self.sum_tree.delete(*self.sum_tree.get_children())
        self.file_tree.delete(*self.file_tree.get_children())
        if not os.path.isdir(od):
            self.res_label.configure(text=f"{od} does not exist yet")
            return
        self.res_label.configure(text=od)
        tic = self.widgets["tic"].get_value() if "tic" in self.widgets else ""
        js = os.path.join(od, f"summary_{tic}.json")
        if not os.path.isfile(js):          # e.g. Kepler run without --tic
            found = sorted(glob.glob(os.path.join(od, "summary_*.json")),
                           key=os.path.getmtime)
            if found:
                js = found[-1]
        if os.path.isfile(js):
            try:
                with open(js, encoding="utf-8") as fh:
                    data = json.load(fh)
                self._fill_tree("", data)
            except Exception as exc:
                self.sum_tree.insert("", "end", text="error", values=(str(exc),))
        else:
            self.sum_tree.insert("", "end", text="(no summary yet)", values=("",))
        files = []
        for f in os.listdir(od):
            p = os.path.join(od, f)
            if os.path.isfile(p):
                files.append((os.path.getmtime(p), f, os.path.getsize(p)))
        for mt, f, sz in sorted(files, reverse=True):
            self.file_tree.insert("", "end", text=f, values=(
                f"{sz / 1024:.0f}", time.strftime("%Y-%m-%d %H:%M", time.localtime(mt))))

    def _fill_tree(self, parent, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and v:
                    node = self.sum_tree.insert(parent, "end", text=str(k), values=("",),
                                                open=parent == "")
                    self._fill_tree(node, v)
                else:
                    self.sum_tree.insert(parent, "end", text=str(k),
                                         values=(self._fmt(v),))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    node = self.sum_tree.insert(parent, "end", text=f"[{i}]", values=("",))
                    self._fill_tree(node, v)
                else:
                    self.sum_tree.insert(parent, "end", text=f"[{i}]", values=(self._fmt(v),))

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _open_selected_file(self, _e=None):
        s = self.file_tree.selection()
        if s:
            open_path(os.path.join(self.outdir(), self.file_tree.item(s[0], "text")))

    # ---- settings / profiles ---------------------------------------------
    def _changed_options(self):
        """Options that differ from the script's defaults."""
        out = {}
        for d, w in self.widgets.items():
            v = w.get_value()
            if w.kind == "_StoreTrueAction":
                if v:
                    out[d] = True
            elif d == "ephemeris":
                if any(v):
                    out[d] = v
            elif w.a.choices:
                if v not in ("", str(w.a.default)):
                    out[d] = v
            elif v != "" and (d == "tic" or v != _fmt_default(w.a)):
                out[d] = v
        return out

    def get_state(self, options_only=False, description=None):
        st = dict(gui_version=__version__,
                  description=description or "",
                  options=self._changed_options(),
                  ground=list(self.ground.specs))
        if not options_only:
            st.update(script=self.script_path, interpreter=self.interp.get(),
                      workdir=self.workdir.get())
        return st

    def profile_dir(self):
        return os.path.join(os.path.dirname(self.script_path), "profiles")

    def _fill_profiles_menu(self):
        m = self.prof_menu
        m.delete(0, "end")
        files = sorted(glob.glob(os.path.join(self.profile_dir(), "*.json")))
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    desc = json.load(fh).get("description") or ""
            except Exception:
                desc = "(unreadable)"
            lab = os.path.splitext(os.path.basename(f))[0]
            m.add_command(label=f"{lab}  —  {desc}" if desc else lab,
                          command=lambda f=f: self.load_settings(f))
        if not files:
            m.add_command(label="(no profiles in profiles/)", state="disabled")
        m.add_separator()
        m.add_command(label="Open profiles folder",
                      command=lambda: (os.makedirs(self.profile_dir(), exist_ok=True),
                                       open_path(self.profile_dir())))

    def set_state(self, st):
        if st.get("interpreter"):
            self.interp.set(st["interpreter"])
        if st.get("workdir"):
            self.workdir.set(st["workdir"])
        for d, w in self.widgets.items():
            if d in st.get("options", {}):
                w.set_value(st["options"][d])
            else:
                w.reset()
        self.ground.specs = list(st.get("ground", []))
        self.ground.refresh()
        self.update_command()

    def save_settings(self, path, options_only=False, description=None):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.get_state(options_only, description), fh, indent=2)
        except Exception as exc:
            if path != SETTINGS_FILE:
                messagebox.showerror("Profile", f"Could not write {path}:\n{exc}")

    def load_settings(self, path, quiet=False):
        try:
            with open(path, encoding="utf-8") as fh:
                st = json.load(fh)
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Profile", f"Could not read {path}:\n{exc}")
            return
        self.set_state(st)

    def save_profile(self, options_only=False):
        from tkinter import simpledialog
        tic = self.widgets["tic"].get_value() if "tic" in self.widgets else ""
        os.makedirs(self.profile_dir(), exist_ok=True)
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         initialfile=(f"profile_TIC{tic}.json" if tic else
                                                      "profile_KIC" + str(self.widgets["kic"].get_value()
                                                                          if "kic" in self.widgets else "")
                                                      + ".json"),
                                         initialdir=self.profile_dir())
        if not p:
            return
        desc = simpledialog.askstring("Profile", "Short description (shown in the "
                                      "Profiles menu):", parent=self,
                                      initialvalue=f"TIC {tic}") or ""
        self.save_settings(p, options_only=options_only, description=desc)

    def open_profile(self):
        p = filedialog.askopenfilename(filetypes=[("Profiles", "*.json"), ("All", "*")],
                                       initialdir=self.profile_dir()
                                       if os.path.isdir(self.profile_dir())
                                       else self.workdir.get())
        if p:
            self.load_settings(p)

    def reset(self):
        for w in self.widgets.values():
            w.reset()
        self.ground.specs = []
        self.ground.refresh()

    def import_command(self):
        dlg = tk.Toplevel(self)
        dlg.title("Import command line")
        ttk.Label(dlg, text="Paste a triceratops_validation.py command line "
                            "(e.g. from the manual):").pack(anchor="w", padx=8, pady=4)
        txt = tk.Text(dlg, width=100, height=6, font=("Courier", 9))
        txt.pack(padx=8, pady=4)

        def do():
            s = txt.get("1.0", "end").strip().replace("\\\n", " ").replace("^\n", " ")
            try:
                self.apply_command_line(s)
            except Exception as exc:
                messagebox.showerror("Import", str(exc), parent=dlg)
                return
            dlg.destroy()
        ttk.Button(dlg, text="Import", command=do).pack(pady=6)

    def apply_command_line(self, s):
        toks = shlex.split(s, posix=not sys.platform.startswith("win"))
        while toks and not toks[0].startswith("--"):
            toks.pop(0)            # python, script name
        ns = self.parser.parse_args(toks)
        self.reset()
        for d, w in self.widgets.items():
            v = getattr(ns, d, None)
            if w.kind == "_StoreTrueAction":
                w.set_value(bool(v))
            elif v is not None and v != w.a.default:
                w.set_value(v)
            elif d == "tic" and v is not None:
                w.set_value(v)
        self.ground.specs = list(getattr(ns, "ground_lc", None) or [])
        self.ground.refresh()

    def on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Quit", "A validation is running. Stop it and quit?"):
                return
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.save_settings(SETTINGS_FILE)
        self.destroy()

    # ---- tools ------------------------------------------------------------
    def check_interpreter(self):
        py = self.interp.get().strip() or sys.executable
        self.interp_info.configure(text="checking …")
        code = ("import inspect,sys\n"
                "import numpy as np, scipy.integrate as si\n"
                "for n,v in (('int',int),('float',float),('bool',bool),('NaN',np.nan),"
                "('Inf',np.inf),('trapz',getattr(np,'trapezoid',None)),"
                "('in1d',getattr(np,'isin',None))):\n"
                " if v is not None and not hasattr(np,n): setattr(np,n,v)\n"
                "if not hasattr(si,'trapz') and hasattr(si,'trapezoid'): si.trapz=si.trapezoid\n"
                "try:\n import triceratops.triceratops as tr\n"
                " import importlib.metadata as m\n"
                " v=m.version('triceratops')\n"
                " p='external_lc_files' in str(inspect.signature(tr.target.calc_probs))\n"
                " import numpy as np\n"
                " nv=tuple(int(x) for x in np.__version__.split('.')[:2])\n"
                " w=' - needs numpy<2.4!' if p and nv>=(2,4) else ''\n"
                " try:\n"
                "  from pytransit import QuadraticModel as Q\n"
                "  pt=m.version('pytransit')\n"
                "  if not hasattr(Q,'evaluate_pv'): w+=' - needs pytransit==2.2.0!'\n"
                " except Exception as e:\n"
                "  pt='not importable'\n"
                " print(('TRICERATOPS+ ' if p else 'triceratops ')+v+', numpy '+np.__version__+', pytransit '+pt+w)\n"
                "except Exception as e:\n print('triceratops not importable: '+type(e).__name__)\n"
                "print('Python '+sys.version.split()[0])\n")

        def work():
            try:
                out = subprocess.run([py, "-c", code], capture_output=True, text=True,
                                     timeout=180).stdout.strip().replace("\n", ", ")
            except Exception as exc:
                out = f"cannot run: {exc}"
            self.q.put(("info", out))       # handled in _poll (main thread)
        threading.Thread(target=work, daemon=True).start()

    def transit_times(self):
        w = self.widgets.get("ephemeris")
        vals = w.get_value() if w else ["", "", "", ""]
        dlg = tk.Toplevel(self)
        dlg.title("Predicted transit times")
        e = {}
        zero = 2454833.0 if self._mission() == "kepler" else 2457000.0
        tlab = "BKJD" if zero < 2456000 else "BTJD"
        for i, (k, v) in enumerate((("P [d]", vals[0]), (f"T0 [{tlab} or BJD]", vals[1]),
                                    ("T14 [h]", vals[3]), ("Number", "6"),
                                    ("From (BJD, empty = now)", ""))):
            ttk.Label(dlg, text=k).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            sv = tk.StringVar(value=v)
            ttk.Entry(dlg, textvariable=sv, width=20).grid(row=i, column=1, padx=6)
            e[k] = sv
        out = tk.Text(dlg, width=88, height=12, font=("Courier", 9))
        out.grid(row=6, column=0, columnspan=3, padx=6, pady=6)

        def calc():
            try:
                P = float(e["P [d]"].get())
                T0 = float(e[f"T0 [{tlab} or BJD]"].get())
                T0 = T0 if T0 > 2.4e6 else T0 + zero
                d = float(e["T14 [h]"].get() or 0) / 24
                n = int(e["Number"].get() or 6)
                start = e["From (BJD, empty = now)"].get().strip()
                now = (float(start) if start else
                       2440587.5 + time.time() / 86400.0)
            except ValueError:
                messagebox.showerror("Transit times", "Check the numbers", parent=dlg)
                return
            E = int((now - T0) // P) + 1
            out.delete("1.0", "end")
            out.insert("end", f"{'epoch':>6} {'mid BJD_TDB':>15}  {'mid UTC (approx.)':>17}"
                              f"  {'ingress UTC':>17}  {'egress UTC':>17}\n")
            for k in range(n):
                tc = T0 + (E + k) * P
                out.insert("end", f"{E + k:6d} {tc:15.5f}  {jd_to_utc_string(tc):>17}"
                                  f"  {jd_to_utc_string(tc - d / 2):>17}"
                                  f"  {jd_to_utc_string(tc + d / 2):>17}\n")
            out.insert("end", "\nUTC without barycentric correction: +-8 min. "
                              "Add the ephemeris uncertainty before planning.\n")
        ttk.Button(dlg, text="Compute", command=calc).grid(row=5, column=1, pady=4)

    def open_manual(self):
        d = os.path.dirname(self.script_path)
        cands = sorted(glob.glob(os.path.join(
            d, "TRICERATOPS_Validation_Pipeline_Documentation_v*.pdf")),
            key=lambda p: [int(x) if x.isdigit() else x
                           for x in re.split(r"(\d+)", p)])
        if cands:
            open_path(cands[-1])
        else:
            messagebox.showinfo("Manual", "No manual PDF found next to the script.")

    def open_guide(self):
        d = os.path.dirname(self.script_path)
        for p in (os.path.join(d, "TRICERATOPS_Observers_Guide.pdf"),
                  os.path.join(d, "observers_guide", "TRICERATOPS_Observers_Guide.pdf")):
            if os.path.isfile(p):
                open_path(p)
                return
        messagebox.showinfo("Observer's Guide",
                            "TRICERATOPS_Observers_Guide.pdf not found next to the script.")

    def show_cli_help(self):
        dlg = tk.Toplevel(self)
        dlg.title("triceratops_validation.py --help")
        t = tk.Text(dlg, width=110, height=40, font=("Courier", 9))
        t.pack(fill="both", expand=True)
        t.insert("1.0", self.parser.format_help())

    def about(self):
        messagebox.showinfo(
            "About", f"TRICERATOPS validation GUI v{__version__}\n"
                     f"Script: {self.script_path} (v{self.script_version})\n"
                     f"Python {sys.version.split()[0]}, Tk {tk.TkVersion}")


def main():
    ap = argparse.ArgumentParser(description="GUI for triceratops_validation.py")
    ap.add_argument("--script", default=None,
                    help="path of triceratops_validation.py (default: next to this file)")
    ap.add_argument("--profile", default=None, help="profile (JSON) to load")
    a = ap.parse_args()
    script = a.script or os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT_NAME)
    if not os.path.isfile(script):
        script = os.path.join(os.getcwd(), SCRIPT_NAME)
    if not os.path.isfile(script):
        print(f"{SCRIPT_NAME} not found; use --script PATH")
        sys.exit(1)
    app = App(script)
    if a.profile:
        app.load_settings(a.profile)
    app.mainloop()


if __name__ == "__main__":
    main()
