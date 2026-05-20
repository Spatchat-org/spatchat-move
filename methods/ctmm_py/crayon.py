"""Partial parity translation of ctmm 1.3.0 ``R/crayon.R``."""

from __future__ import annotations

import sys


_ANSI = {
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
}
_RESET = "\x1b[0m"


def message(*args, fg: str = "black", bg: str = "bgWhite"):
    del bg
    text = " ".join(str(a) for a in args)
    prefix = _ANSI.get(str(fg).lower(), "")
    sys.stderr.write(f"{prefix}{text}{_RESET}\n")
    return text


__all__ = ["message"]
