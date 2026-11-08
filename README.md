# TRICERATOPS validation pipeline

Statistical validation of **TESS** and **Kepler** planet candidates with [TRICERATOPS](https://github.com/stevengiacalone/triceratops) (Giacalone et al. 2021), including ground-based photometry, difference imaging, and a graphical front-end.

| Component | File | Version |
|-----------|------|---------|
| Pipeline | `triceratops_validation.py` | **4.2** |
| GUI | `triceratops_gui.py` | **1.2** |
| Audit (prototype) | `triceratops_audit.py` | 0.1 |

Observer-oriented documentation: **Observer’s Guide** (PDF) + Kepler update addendum.  
Full methods and tests: technical manual (separate PDF).

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

See the **Kepler update addendum** (PDF) for the full observer-facing description.

---

## Ground-based photometry

```bash
--ground-lc "file.csv,band=R,star=65344134,ap=3.5,fwhm=2.4,obs=CODE"
--ground-list my_list.txt
```

Supported formats: AstroImageJ tables, AAVSO/EXOTIC reports, simple time/flux/error tables.  
Stars inconsistent with the observed depth are cleared before `calc_probs`; with TRICERATOPS+ the light curves can enter the likelihood (`--tri-plus auto|on|off`).

---

## Repository layout (suggested)

```text
.
├── triceratops_validation.py   # main pipeline
├── triceratops_gui.py          # Tkinter front-end
├── triceratops_audit.py        # post-run audit
├── profiles/                   # saved GUI profiles (JSON)
├── tests/                      # pytest regression tests
├── README.md
└── docs/                       # optional: Observer’s Guide + addendum PDFs
```

Large data products (cutouts, run outputs, TRILEGAL tables) should stay out of git (see `.gitignore`).

---

## Citation

If you use this pipeline in a publication, please cite:

- Giacalone et al. (2021), *TRICERATOPS*, [AJ](https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G)
- and the specific version of this pipeline (version string is written into every log and `summary_*.json`).

---

## Licence / contact

Add your preferred licence and contact details here.
