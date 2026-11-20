MOLUSC (Wood, Mann & Kraus 2021, AJ 162, 128), https://github.com/woodml/MOLUSC
copied 2026-11-20 for the TRICERATOPS validation pipeline, with three small
patches in code/BinaryStarGUI.py (marked "pipeline patch" / "compatibility shim"):
  1. scipy.interpolate.interp2d (removed in SciPy 1.14) replaced by a
     RegularGridInterpolator with interp2d's behaviour (nearest-value
     extrapolation, 1-D result for scalar input);
  2. float(f(...)) on a 1-element array (an error in NumPy 2) -> float(np.ravel(...)[0]);
  3. the command line stopped with AttributeError 'star_jitter': set to -1.
Tested with a simulated Gaia answer (100,000 companions, 1 min).

Use with the pipeline:
  bash run_molusc_L27994.sh
  then in the GUI (tab Field): MOLUSC companions file = ..._kept.csv,
  MOLUSC: companions generated = the N of the run (1000000).
  4. Gaia query with a fallback: when the Gaia ESA archive does not answer
     (connection reset; archive down from 25 Nov 2026 for DR4), the same
     values (parallax, G, BP-RP, RUWE, solution type) are read from VizieR
     (Gaia DR3, I/355/gaiadr3), with the mirrors the pipeline also uses.
  5. The DeprecationWarning of interp2d (SciPy 1.10-1.13, one per simulated
     companion) is switched off; it does not change the results.
