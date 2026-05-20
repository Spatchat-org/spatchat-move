"""Parity-focused translation of ctmm 1.3.0 ``R/parallel.R``."""
from __future__ import annotations

import os

from .optim import optimizer


def detectCores(*args, fast: bool = True, **kwargs) -> int:
    del args, kwargs
    if fast and os.name == "nt":
        return 1
    return max(int(os.cpu_count() or 1), 1)


def plapply(items, fn, cores: int = 1, fast: bool = False):
    del fast
    cores = resolveCores(cores, fast=False)
    # Keep deterministic behavior; callers get the same values as lapply.
    return [fn(x) for x in items]


def resolveCores(cores: int = 1, fast: bool = False) -> int:
    if cores is None:
        cores = detectCores(fast=fast)
    cores = int(cores)
    if cores < 1:
        cores = max(1, detectCores(fast=fast) + cores)
    if fast and os.name == "nt":
        cores = 1
    return max(cores, 1)


__all__ = ["detectCores", "optimizer", "plapply", "resolveCores"]
