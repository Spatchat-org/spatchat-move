# ctmm_py

Pure-Python ctmm runtime used by SpatChat AKDE.

This package is not an R wrapper. The translated ctmm 1.3.0 source modules live directly in this package. Public module names mirror the ctmm R source files where practical, with app-facing helpers folded into the same modules.

The app imports the production path through `methods.akde`, which uses:
- `telemetry.as_telemetry`
- `variogram.variogram`
- `models.ctmm`, `models.ctmm_guess`, `models.ctmm_fit`, `models.ctmm_select`
- `home_range.akde`
- selected translated R-source helpers in this package
