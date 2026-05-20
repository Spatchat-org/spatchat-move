"""Partial parity translation of ctmm 1.3.0 ``R/str.R``."""

from __future__ import annotations

from pprint import pformat
from typing import Any


def str_custom(object: Any) -> str:
    cls = object.__class__.__name__
    header = f"Class '{cls}' [ctmm_py parity]"
    body = pformat(getattr(object, "__dict__", object), width=100, compact=False)
    return header + "\n" + body


def str_ctmm(object: Any) -> str:
    return str_custom(object)


def str_UERE(object: Any) -> str:
    return str_custom(object)


def str_covm(object: Any) -> str:
    return str_custom(object)


__all__ = ["str_custom", "str_ctmm", "str_UERE", "str_covm"]
