# TRICERATOPS validation pipeline

Statistical validation of **TESS** and **Kepler** planet candidates with [TRICERATOPS](https://github.com/stevengiacalone/triceratops) (Giacalone et al. 2021), including ground-based photometry, difference imaging, and a graphical front-end.

| Component | File | Version |
|-----------|------|---------|
| Pipeline | `triceratops_validation.py` | **4.3** |
| GUI | `triceratops_gui.py` | **1.2** |
| Audit (prototype) | `triceratops_audit.py` | 0.1 |
| Follow-up recommendations | `triceratops_recommend.py` | — |
| Batch runs | `triceratops_batch.py` | — |

Observer-oriented documentation: **Observer’s Guide v4.3** (`TRICERATOPS_Observers_Guide.pdf`, LaTeX sources in `observers_guide/`).  
Full methods and tests: technical manual (`TRICERATOPS_Validation_Pipeline_Documentation.pdf`).

**Author / contact:** Siegfried Vanaverbeke  
GitHub: [@svaverbe](https://github.com/svaverbe) · Repository: [triceratops-gui](https://github.com/svaverbe/triceratops-gui)  
E-mail: [siegfriedvanaverbeke@gmail.com](mailto:siegfriedvanaverbeke@gmail.com)

---

### Summary

This repository provides a pipeline for the **statistical validation** of TESS and Kepler planet candidates with TRICERATOPS. It builds light curves, runs diagnostic tests (odd/even, secondary eclipse, ETV, difference imaging, …), computes false-positive probabilities (FPP/NFPP), and can include ground-based photometry (AAVSO, AstroImageJ, …) to clear nearby stars. A graphical interface (`triceratops_gui.py`) and a post-run audit script are included. From version 4.2, **Kepler** targets are supported (`--mission kepler --kic`). See the Observer’s Guide for observer-oriented documentation.

---

## New in version 4.3

- **Audit with recommendations.** `triceratops_audit.py` tries to break the claim of a run (17 deterministic tests) and lists, from the TRICERATOPS scenario probabilities, which follow-up observation would lower the FPP most: high-resolution imaging, ground photometry with separate apertures, transits in two or more filters, or reconnaissance spectroscopy (`triceratops_recommend.py`; also written to `recommendations_<TIC>.txt`).
- **Multi-band ground photometry.** The colour test (intrinsic depth per band, $\chi^2$ test and red/blue ratio) and the use of the light curves by TRICERATOPS+ are recorded in `summary_<TIC>.json`; the audit test *Multi-band* states which false-positive scenarios the colours rule out and whether the FPP already includes the ground data.
- **Companions limited by Gaia (MOLUSC).** `--molusc-file`, `--molusc-n`, `--molusc-multiplicity`: the bound companions still allowed by Gaia RUWE and Gaia imaging (MOLUSC, Wood et al. 2021) replace the default companion prior. A patched copy of MOLUSC is in `MOLUSC/` (see `MOLUSC/README_pipeline.txt`).
- **Robustness fixes** found with test targets: ephemeris refinement in the blind search, per-transit template, secondary-eclipse centre on stars with a phase curve, hot-planet occultations not counted as EB evidence, difference imaging in sectors with scattered light, limb-darkened fit with a stellar-density prior when the free fit is implausible, `--bin-minutes` for the TRICERATOPS fold, audit that leaves out the sectors the pipeline excluded.
- **Worked examples in the Observer’s Guide:** TIC 65344135 (AAVSO campaign 942), WASP-18 b (control target with its occultation), TIC 229701403 (two candidates that are one eccentric eclipsing binary on a neighbour), L 279-94 (M-dwarf candidate; default vs Gaia-limited companions).

---

## What it does

1. Builds multi-sector (TESS) or multi-quarter (Kepler) light curves.
2. Detrends, searches or refines the ephemeris, and runs diagnostic tests (odd/even, secondary eclipse, ETV, difference imaging, …).
3. Calls TRICERATOPS to compute false-positive probabilities (FPP / NFPP).
4. Optionally folds in **ground-based light curves** (AAVSO, AstroImageJ, EXOTIC, plain tables) to clear nearby stars and, with TRICERATOPS+, to use multi-colour photometry in the likelihood.
5. Writes a structured log, `summary_<TIC>.json`, plots, and tables.

A candidate is reported **PASSED** when FPP &lt; 0.015 and NFPP &lt; 0.001 (Giacalone et al. 2021 criteria), provided the diagnostics show nothing suspicious.

---

## Requirements

- Python ≥ 3.8 (3.11 recommended)
- Core packages: `triceratops` (≥ 1.1), `lightkurve`, `astroquery`, `wotan`, `transitleastsquares`, `numpy`, `scipy`, `matplotlib`, `astropy`
- GUI: Tkinter (included with most conda Pythons; on Ubuntu/WSL: `sudo apt install python3-tk`)
- Optional: [TRICERATOPS+](https://github.com/JGB276/TRICERATOPS-plus) in a separate environment for multi-colour ground photometry in `calc_probs`

Example conda environment for TRICERATOPS+:

```bash
conda create -n triplus -c conda-forge python=3.11 "numpy<2.4" scipy pandas matplotlib \
  astropy astroquery numba celerite emcee seaborn arviz corner pytest
conda activate triplus
pip install lightkurve "pytransit==2.2.0" mechanicalsoup pyrr wotan transitleastsquares "setuptools<81"
git clone https://github.com/JGB276/TRICERATOPS-plus.git
cd TRICERATOPS-plus && pip install .
```

---

## Quick start

### Command line

```bash
# TESS candidate with known ephemeris
python triceratops_validation.py \
  --tic 65344135 \
  --ephemeris 36.88 4352.3681 5900 5.33 \
  --ld-fit

# Kepler candidate
python triceratops_validation.py \
  --mission kepler --kic 3755522 \
  --ephemeris 10.5 245.3 1200 3.2 \
  --ld-fit
```

T0 may be given as mission time (BTJD / BKJD) or as full BJD_TDB; the script converts automatically.  
Depth is in **ppm**, duration in **hours**.

```bash
python triceratops_validation.py --help   # full option list
```

### Graphical interface

```bash
python triceratops_gui.py
# or point at a specific validation script:
python triceratops_gui.py --script /path/to/triceratops_validation.py
```

The GUI reads every option from `build_parser()` in the validation script, so new flags appear automatically.  
Settings can be saved as **profiles** (JSON) in `profiles/`.

Useful menu items:
- **Tools → Check interpreter** — which `triceratops` / NumPy / pytransit the selected Python has
- **Tools → Run regression tests** — `pytest` on `tests/`
- **Tools → Predicted transit times**
- **Help → Observer’s Guide / Technical manual / Command-line help**

### Audit a finished run

```bash
python triceratops_audit.py TIC65344135
python triceratops_audit.py path/to/output_folder --tic 65344135
```

Writes `audit_<TIC>.json`, `.txt` and `.pdf` with independent checks of the claims in the run (deterministic, no language model).

---

## Kepler mode (v4.2)

```text
--mission kepler --kic <KIC>
```

- Long-cadence Kepler SAP/PDCSAP and target-pixel files **per quarter**
- Times in **BKJD** (BJD_TDB − 2454833)
- Kepler-band limb darkening; TRICERATOPS with `mission="Kepler"`
- TIC looked up from KIC when `--tic` is omitted
- Default output folder: `KIC<kic>/`

Related options:

| Option | Meaning |
|--------|---------|
| `--host-tic` | Star treated as host by TRICERATOPS (e.g. close companion) |
| `--star-source tic\|kepler` | Source of R, M, Teff, logg |
| `--stellar R M Teff [logg]` | Manual stellar parameters |

See the chapter *Kepler targets* of the Observer’s Guide.

---

## Ground-based photometry

```bash
--ground-lc "file.csv,band=R,star=65344134,ap=3.5,fwhm=2.4,obs=CODE"
--ground-list my_list.txt
```

Supported formats: AstroImageJ tables, AAVSO/EXOTIC reports, simple time/flux/error tables.  
Stars inconsistent with the observed depth are cleared before `calc_probs`; with TRICERATOPS+ the light curves can enter the likelihood (`--tri-plus auto|on|off`).  
Only `g r i z J H K` (and Rc, Ic) are TRICERATOPS+ bands; B, V and clear-filter light curves enter the clearance and the colour test.

---

## Companions limited by Gaia (MOLUSC)

```bash
bash MOLUSC/run_molusc_L27994.sh          # example script; needs access to the Gaia archive or VizieR
python triceratops_validation.py ... \
  --molusc-file TIC100796276_molusc/L27994_N1000000_kept.csv --molusc-n 1000000
```

TRICERATOPS uses only the MOLUSC companions whose periastron is beyond 10 au and treats closer binaries as planet-free; report an FPP obtained this way together with that assumption (Observer’s Guide, Section 5.4).

---

## Repository layout (suggested)

```text
.
├── triceratops_validation.py   # main pipeline
├── triceratops_gui.py          # Tkinter front-end
├── triceratops_audit.py        # post-run audit
├── triceratops_recommend.py    # next-observation recommendations
├── triceratops_batch.py        # batch runs from a JSON list
├── requirements.txt
├── profiles/                   # saved GUI profiles (JSON)
├── tests/                      # pytest regression tests
├── MOLUSC/                     # MOLUSC (Wood et al. 2021, MIT) with small patches
├── observers_guide/            # LaTeX sources of the Observer’s Guide
├── TIC65344135/                # example output (AAVSO campaign 942)
├── TRICERATOPS_Observers_Guide.pdf
├── TRICERATOPS_Validation_Pipeline_Documentation.pdf
└── README.md
```

Large data products (cutouts, run outputs, TRILEGAL tables) should stay out of git (see `.gitignore`).

---

## Acknowledgements

Steve Price tested the pipeline independently with his own pipeline (KESTREL) and contributed `triceratops_batch.py` and `requirements.txt`. MOLUSC is by Mackenna L. Wood (MIT licence, included in `MOLUSC/`).

---

## Citation

If you use this pipeline in a publication, please cite:

- Giacalone et al. (2021), *TRICERATOPS*, [AJ](https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G)
- Wood, Mann & Kraus (2021), *MOLUSC*, AJ 162, 128 — when `--molusc-file` is used
- and the specific version of this pipeline (version string is written into every log and `summary_*.json`).

---

## Licence

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Siegfried Vanaverbeke (https://github.com/svaverbe)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

Questions, bug reports and contributions: open an issue on this repository, or contact the author listed at the top of this README.
