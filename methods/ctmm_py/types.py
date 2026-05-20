from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class Telemetry:
    data: pd.DataFrame
    id_col: str = "animal_id"
    time_col: str = "timestamp"
    x_col: str = "longitude"
    y_col: str = "latitude"
    crs: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CTMMModel:
    model: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariogramFit:
    model: CTMMModel
    fit: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParityTolerance:
    rtol: float = 1e-10
    atol: float = 1e-12
