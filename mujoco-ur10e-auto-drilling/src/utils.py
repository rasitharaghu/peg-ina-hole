from __future__ import annotations
from pathlib import Path
from typing import Any
import json, yaml, numpy as np

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v.copy() if n < 1e-12 else v / n
