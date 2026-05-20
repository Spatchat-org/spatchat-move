"""Partial parity translation of ctmm 1.3.0 ``R/speed.R``."""

from __future__ import annotations

from .speed_ops import speed, speed_deterministic, speed_variance


def speed_telemetry(object, CTMM, **kwargs):
    return speed(object, CTMM=CTMM, **kwargs)


def speed_ctmm(object, **kwargs):
    return speed(object, **kwargs)


def speed_rand(CTMM, data=None, **kwargs):
    del data, kwargs
    return speed_deterministic(CTMM)


def spd_fn(*args, **kwargs):
    return speed(*args, **kwargs)


def fn(*args, **kwargs):
    return speed(*args, **kwargs)


__all__ = ["fn", "spd_fn", "speed", "speed_ctmm", "speed_deterministic", "speed_rand", "speed_telemetry", "speed_variance"]
