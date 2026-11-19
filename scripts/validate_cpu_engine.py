#!/usr/bin/env python3
"""
Distributional validation of the CPU engine (src/schc_fast.c) against the JAX runs at L=100.

  C:    results/cpu_scan_check/original_mix/L100/runs            (200 runs, 20,000 steps)
  JAX:  results/jax/transition_check/L100/runs   (50 runs, 20,000 steps)

The two implementations use different random number generators, so the comparison concerns
distributions, not the trajectories of identical seeds.

Compared: number of runaway runs (final mean component size > 100; Fisher exact) and, among
non-runaway runs, final mean size, final max size and final occupancy (medians; Mann-Whitney).
Final values are those of the last recorded step of a run: step 20000, or the step at which
the run went extinct, where sizes and occupancy are zero.

Output: results/npj_rev/validation/c_vs_jax_L100.csv, one row per compared quantity
(the same table is printed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
L = 100
C_RUNS = ROOT / "results/cpu_scan_check/original_mix/L100/runs"
JAX_RUNS = ROOT / "results/jax/transition_check/L100/runs"
OUT = ROOT / "results/npj_rev/validation/c_vs_jax_L100.csv"


def summarize(run_dir: Path, L: int):
    out = []
    for f in sorted(run_dir.glob("seed_*.csv")):
        if f.name.endswith(".partial.csv"):
            continue
        d = pd.read_csv(f)
        last = d.iloc[-1]
        dens = last["n_active"] / L**2 if "n_active" in d else np.nan
        out.append(dict(runaway=bool(last["mean_size"] > 100),
                        mean_size=float(last["mean_size"]), max_size=float(last["max_size"]), density=dens))
    return pd.DataFrame(out)


def main():
    for run_dir in (C_RUNS, JAX_RUNS):
        if not run_dir.is_dir():
            raise SystemExit(f"runs not found: {run_dir}")
    c, j = summarize(C_RUNS, L), summarize(JAX_RUNS, L)
    kc, kj = int(c.runaway.sum()), int(j.runaway.sum())
    p_run = stats.fisher_exact([[kc, len(c) - kc], [kj, len(j) - kj]])[1]
    rows = [dict(quantity="runaway (final mean component size > 100)", statistic="number of runs",
                 c=kc, jax=kj, n_c=len(c), n_jax=len(j), test="Fisher exact", p=p_run)]
    cn, jn = c[~c.runaway], j[~j.runaway]
    for col, quantity in (("mean_size", "final mean component size"),
                          ("max_size", "final largest component size"),
                          ("density", "final occupancy")):
        if cn[col].notna().sum() > 2 and jn[col].notna().sum() > 2:
            p = stats.mannwhitneyu(cn[col].dropna(), jn[col].dropna()).pvalue
            rows.append(dict(quantity=quantity, statistic="median of non-runaway runs",
                             c=cn[col].median(), jax=jn[col].median(),
                             n_c=int(cn[col].notna().sum()), n_jax=int(jn[col].notna().sum()),
                             test="Mann-Whitney U", p=p))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
