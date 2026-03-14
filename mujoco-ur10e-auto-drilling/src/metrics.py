from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import csv


@dataclass
class DrillMetrics:
    target_id: int
    success: int
    steps: int
    retries: int
    peak_force_z: float
    peak_force_xy: float
    spindle_time_s: float
    jam_events: int


def append_metrics_csv(path: str | Path, row: DrillMetrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def write_summary_json(path: str | Path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
