"""ctmm_py: pure-Python ctmm 1.3.0 runtime used by SpatChat."""

try:
    from .api import *  # noqa: F401,F403
except ImportError:
    # Minimal environments without pandas / full stack: submodule imports still work
    # (e.g. ``import ctmm_py.covm``).
    pass
