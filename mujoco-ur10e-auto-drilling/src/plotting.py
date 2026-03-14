from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmark(csv_path: str | Path, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.hist(df["peak_force_z"], bins=20)
    plt.xlabel("Peak axial force")
    plt.ylabel("Count")
    plt.title("Peak axial force")
    plt.tight_layout()
    plt.savefig(out_dir / "peak_force_z.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["peak_force_xy"], bins=20)
    plt.xlabel("Peak lateral force")
    plt.ylabel("Count")
    plt.title("Peak lateral force")
    plt.tight_layout()
    plt.savefig(out_dir / "peak_force_xy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["spindle_time_s"], bins=20)
    plt.xlabel("Spindle time [s]")
    plt.ylabel("Count")
    plt.title("Spindle time")
    plt.tight_layout()
    plt.savefig(out_dir / "spindle_time.png", dpi=150)
    plt.close()
