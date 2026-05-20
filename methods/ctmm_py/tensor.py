"""Parity translation of ctmm 1.3.0 ``R/tensor.R`` helpers."""

from __future__ import annotations

import numpy as np


def Assign(x, i, value, index: int = 1):
    out = np.asarray(x).copy()
    sl = [slice(None)] * out.ndim
    sl[int(index) - 1] = i
    out[tuple(sl)] = value
    return out


def arrayify(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def fbind(x, y):
    xx = arrayify(x)
    yy = arrayify(y)
    return np.concatenate([xx, yy], axis=0)


def lbind(x, y):
    xx = arrayify(x)
    yy = arrayify(y)
    return np.concatenate([xx, yy], axis=xx.ndim - 1)


def tensor_product(x, y):
    xx = arrayify(x)
    yy = arrayify(y)
    return np.tensordot(xx, yy, axes=([-1], [0]))


def tensor_contract2(x, y):
    xx = arrayify(x)
    yy = arrayify(y)
    return np.tensordot(xx, yy, axes=([-2, -1], [0, 1]))

__ = tensor_contract2

__all__ = ["Assign", "__", "arrayify", "fbind", "lbind", "tensor_contract2", "tensor_product"]
