from __future__ import annotations
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

def plot_benchmark(csv_path, out_dir):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(csv_path)
    plt.figure(figsize=(8,5)); plt.hist(df['steps'], bins=10); plt.xlabel('Steps'); plt.ylabel('Count'); plt.tight_layout(); plt.savefig(out/'steps.png', dpi=150); plt.close()
