from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.utils import load_json, unit

@dataclass
class DrillTarget:
    target_id: int
    position: np.ndarray
    normal: np.ndarray
    depth: float
    diameter: float

def load_targets(path: str) -> list[DrillTarget]:
    data = load_json(path)
    out=[]
    for item in data['targets']:
        out.append(DrillTarget(int(item['id']), np.array(item['position'], dtype=float), unit(np.array(item['normal'], dtype=float)), float(item['depth']), float(item['diameter'])))
    return out
