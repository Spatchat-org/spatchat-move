"""Partial parity translation of ctmm 1.3.0 ``R/speeds.R``."""

from __future__ import annotations

from .speed_ops import abs_bivar, abs_data, speeds, speeds_fast


def speeds_telemetry(object, CTMM, **kwargs):
    return speeds(object, CTMM=CTMM, **kwargs)


def speeds_ctmm(object, data=None, **kwargs):
    return speeds(object, data=data, **kwargs)


def speeds_slow(data, CTMM=None, **kwargs):
    return speeds(data, CTMM=CTMM, fast=False, **kwargs)


def spds_fn(*args, **kwargs):
    return speeds(*args, **kwargs)


__all__ = ["abs_bivar", "abs_data", "spds_fn", "speeds", "speeds_ctmm", "speeds_fast", "speeds_slow", "speeds_telemetry"]
