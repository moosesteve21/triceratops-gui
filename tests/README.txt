Regression tests for triceratops_validation.py (v4.0, v4.1)

Run from the folder that contains triceratops_validation.py:

    python -m pytest -q tests

All tests use synthetic data and need no internet access (about 10 s).
Run them after every change to the script.  They cover: outlier clipping of
deep eclipses, ephemeris scan and grid search, optimal aperture and
per-sector dilution, difference-image centroid fits, the secondary-eclipse
search (detection, single-event rejection, no confusion with the primary),
per-transit depths with the cadence-smoothed limb-darkened shape, the
eclipse-timing analysis, the TRICERATOPS exposure and convergence logic.

v4.1: tests/test_ground_photometry.py covers the ground-based photometry
module: specification parsing and band names, AstroImageJ / AAVSO-EXOTIC /
magnitude readers, JD_UTC -> BJD_TDB, the transit fit (depth, mid-time,
airmass baseline), aperture flux fractions against a numerical integral,
star clearance (signal on target, nearby eclipsing binary, no coverage),
chromaticity, the TRICERATOPS+ inputs and the ground mid-times in the ETV
analysis.
