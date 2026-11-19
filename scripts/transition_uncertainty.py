"""Settings and helpers shared by the statistics of the SCHC size scan.

Imported by scripts/npj_rev_ext2_tables.py. It defines

  * the runaway threshold (final mean component size > 100);
  * the bootstrap settings: 10,000 resamples of whole runs with a fixed seed;
  * the time windows of the Late/Early ratio (steps 15,000-20,000 over 2,000-5,000) and of the
    size-score correlation (steps 2,000-10,000);
  * the Wilson score interval, window masks, row-wise Pearson correlation and percentile
    intervals.
"""
from __future__ import annotations

import numpy as np

RUNAWAY = 100.0
B = 10_000
SEED = 20260925
EARLY, LATE, CORR = (2000, 5000), (15000, 20000), (2000, 10000)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def window(steps: np.ndarray, w: tuple[int, int]) -> np.ndarray:
    return (steps >= w[0]) & (steps <= w[1])


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(axis=1) / den


def pct(a: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
