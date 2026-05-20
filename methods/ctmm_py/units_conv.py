from __future__ import annotations

from typing import Any


_DAY = 86400.0
_YEAR = 365.24217 * _DAY
_MONTH = 2.8 + 60.0 * (44.0 + 60.0 * (12.0 + 24.0 * 29.0))

_SCALES = {
    # time
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": _DAY,
    "days": _DAY,
    "mon": _MONTH,
    "month": _MONTH,
    "months": _MONTH,
    "yr": _YEAR,
    "year": _YEAR,
    "years": _YEAR,
    # distance
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
}


def _canonical_unit(unit: str) -> str:
    s = str(unit).strip().lower()
    return " ".join(s.split())


def unit_convert(value: float, unit_expr: str, *, to_si: bool) -> float:
    """Subset of ctmm `%#%` conversion for common scalar units."""
    unit_expr = _canonical_unit(unit_expr)
    if unit_expr == "":
        return float(value)

    if "*" in unit_expr:
        out = float(value)
        for part in unit_expr.split("*"):
            out = unit_convert(out, part, to_si=to_si)
        return out

    if "/" in unit_expr:
        parts = unit_expr.split("/")
        out = unit_convert(value, parts[0], to_si=to_si)
        for den in parts[1:]:
            out = unit_convert(out, den, to_si=not to_si)
        return out

    if unit_expr not in _SCALES:
        raise ValueError(f"Unit {unit_expr!r} unknown.")
    scale = _SCALES[unit_expr]
    return float(value) * scale if to_si else float(value) / scale


def pct_hash_pct(x: Any, y: Any) -> float:
    """ctmm `%#%` operator subset.

    Numeric `%#%` unit -> convert to SI.
    Unit `%#%` numeric -> convert from SI.
    """
    if isinstance(x, (int, float)) and not isinstance(y, (int, float)):
        return unit_convert(float(x), str(y), to_si=True)
    if not isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return unit_convert(float(y), str(x), to_si=False)
    if not isinstance(x, (int, float)) and not isinstance(y, (int, float)):
        return pct_hash_pct(x, pct_hash_pct(1.0, y))
    raise TypeError("Unsupported `%#%` operand types.")

