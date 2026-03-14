from pathlib import Path
from src.metrics import DrillMetrics, append_metrics_csv


def test_metrics_csv(tmp_path: Path):
    path = tmp_path / "drill.csv"
    row = DrillMetrics(
        target_id=0,
        success=1,
        steps=100,
        retries=0,
        peak_force_z=12.0,
        peak_force_xy=3.0,
        spindle_time_s=1.2,
        jam_events=0,
    )
    append_metrics_csv(path, row)
    assert path.exists()
