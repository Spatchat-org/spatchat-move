from __future__ import annotations

from .types import CTMMModel, Telemetry


def akde(telem: Telemetry, model: CTMMModel, **kwargs):
    from .kde import akde as akde_port

    return akde_port(telem, model, **kwargs)


def homerange(telem: Telemetry, model: CTMMModel, **kwargs):
    return akde(telem, model, **kwargs)
