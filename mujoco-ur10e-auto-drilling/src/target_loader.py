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
    out: list[DrillTarget] = []
    for item in data["targets"]:
        out.append(
            DrillTarget(
                target_id=int(item["id"]),
                position=np.array(item["position"], dtype=float),
                normal=unit(np.array(item["normal"], dtype=float)),
                depth=float(item["depth"]),
                diameter=float(item["diameter"]),
            )
        )
    return out
