These are reusable method-specific reproducibility script templates for SpatChat.

Export behavior:
- The app copies only the scripts relevant to methods requested in the current session.
- It also bundles:
  - input_data.csv : the exact standardized dataset used in the session
  - repro_config.json : the session-specific argument values used by SpatChat

Usage:
- Rename any copied .txt script to .py if you want to run it directly.
- Each script has a USER ARGUMENTS section at the top.
- By default, scripts read values from repro_config.json when present.
- Reviewers can inspect or modify those arguments explicitly without using SpatChat or an LLM.

Current state:
- mcp_reproduce.txt is standalone and does not depend on this repository's Python modules.
- kde_reproduce.txt is standalone and reproduces the KDE raster/contour export files.
- locoh_reproduce.txt is standalone and reproduces the LoCoH JSON/GeoJSON export files.
- akde_reproduce.txt is standalone and reproduces the AKDE raster/contour/variogram export files.
- dbbmm_reproduce.txt is standalone and reproduces the dBBMM raster/isopleth export files.
- displacement_reproduce.txt, step_lengths_reproduce.txt, turning_angles_reproduce.txt,
  autocorrelation_reproduce.txt, and hmm_reproduce.txt are also available and use the
  bundled movement_common.py.txt helper.
- The app copies only the templates relevant to the methods actually run in the session.
