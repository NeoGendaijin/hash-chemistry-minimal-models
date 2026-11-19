#!/usr/bin/env python3
"""Extension II (Sec. 4): statistics of the SCHC size scan, Table 2 and supplementary numbers.

Runs of the C implementation of SCHC (src/schc_fast.c), seeds 0-99 per L. Everything is computed
from the per-seed CSVs. A condition whose runs are not all present is reported as "pending data"
together with the number of runs actually used.

Inputs (read only)
  results/cpu_scan_full/original_mix/L{L}/runs/seed_{s}.csv   size scan: 20k steps, no early stop
      Size columns (final mean/max, Late/Early, r, fill) of an L with missing runs stay "pending"
      (use --allow-partial to print provisional values computed from the runs present; runaway runs
      take longer to simulate, so an unfinished scan under-represents them).
  results/cpu_scan_40k/original_mix/L{L}/...   40k steps, runs stopped at mean size >= 2000 (horizon check)
  results/cpu_sensitivity/original_mix[_mu..|_pd..]/L{L}/...  10k steps, 30 seeds, no stop
  results/cpu_scan_check/original_mix/L100/...  200 seeds at L = 100, 20k steps, no stop (extra-seed check)
  results/cpu_snapshots/t{T}/original_mix/L{L}/configs/seed_{s}_final.npy   (caption numbers)
  results/cpu_scan_full/original_mix/L{L}/configs/seed_{s}_final.npy        (component definition check)

Definitions (paper rules; the runaway threshold, the time windows and the bootstrap settings are set
in scripts/transition_uncertainty.py)
  runaway        final mean component size > 100 at the run's last row (t = 20,000 unless extinct).
                 Extinct runs (n_active == 0) are non-runaway with all sizes 0 from extinction on.
  final sizes    mean_size / max_size at the last row (0 for extinct runs).
  Late/Early     per run, the mean of mean_size over steps 15,000-20,000 and over 2,000-5,000
                 (inclusive windows; samples every 10 steps), then the ratio of the ensemble means,
                 mean(late) / mean(early).
  r (paper)      Pearson r between the ensemble-mean size trajectory and the ensemble-mean score
                 (mean_fitness) trajectory over steps 2,000-10,000. Extinct runs contribute size 0;
                 the score is averaged over the runs alive at each sample.
  r (per run)    Alternatives that remove the common time trend that dominates the ensemble r:
                 r_run  = per-run Pearson r of mean_size(t) vs mean_fitness(t) over 2,000-10,000;
                 r_detr = the same after removing a least-squares linear trend in t from each
                 series of each run. Summarised as the back-transformed mean Fisher z over runs
                 (runs with an undefined r are dropped) with a bootstrap interval over runs.
                 Also grouped by the run's state DURING the window (not by its outcome at 20,000):
                 runaway throughout (mean size > 100 by t = 2,000; runaway is absorbing), compact
                 throughout (no crossing by t = 10,000), transitional (crossing in (2,000, 10,000]);
                 runs extinct by 10,000 without crossing are left out. Each group has its own
                 bootstrap interval over the runs of the group.
  intervals      Wilson score 95% for fractions; percentile bootstrap over runs (10,000 resamples,
                 seed 20260925 combined with L) for everything else. The ensemble r also gets a
                 BCa interval (jackknife acceleration), because its bootstrap distribution is skewed.
  nucleation     first sampled step with mean size > 100 (also > 10 and > 1000); runs that never
                 cross are censored at their last row (extinction or 20,000).
  seeds          Seeds 0-99 (and therefore the initial cell types) are the same at every L, and the
                 early dynamics are identical until a structure meets the boundary. Besides the
                 independent-groups tests, the script reports the outcome coupling of the same seed
                 across L (phi of all and of adjacent pairs, with a seed-shuffle null), Cochran's Q
                 and within-seed permutation versions of the homogeneity and trend tests, and a
                 seed-cluster bootstrap for pooled fractions.
  snapshots      Caption numbers for the Fig. 'phenomenon' snapshots are read from the 20k CSVs at
                 the snapshot times (a T-step run is the prefix of the 20k run); the saved .npy
                 configurations of results/cpu_snapshots, when present, are labelled as a check.

Outputs: results/npj_rev/ext2_tables/  (CSV, MD, PNG, PDF and the LaTeX rows of Table 2)

Usage
  .venv/bin/python scripts/npj_rev_ext2_tables.py                 # default, pending where incomplete
  .venv/bin/python scripts/npj_rev_ext2_tables.py --allow-partial # provisional size columns
  options: --boot 10000 --perm 100000 --out <dir>
"""
from __future__ import annotations

import os

# BLAS runs single-threaded unless these variables are already set
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import transition_uncertainty as tu  # noqa: E402  (wilson, pct, window, rowwise_pearson, windows)

RES = ROOT / "results"
FULL = RES / "cpu_scan_full" / "original_mix"
H40K = RES / "cpu_scan_40k" / "original_mix"
SENS = RES / "cpu_sensitivity"
SNAP = RES / "cpu_snapshots"
EXTRA_L100 = RES / "cpu_scan_check" / "original_mix" / "L100"   # 200 seeds, no stop

TABLE_L = list(range(200, 401, 20))
ONSET_L = [100, 120, 140, 160, 180, 200, 210, 220, 230, 240, 250, 260]
ALL_L = sorted(set(TABLE_L) | set(ONSET_L))
SEEDS = list(range(100))
T_END, DT, STOP = 20000, 10, 2000.0
RUN = tu.RUNAWAY                       # 100 cells
SEED = tu.SEED                         # 20260925
EARLY, LATE, CORR = tu.EARLY, tu.LATE, tu.CORR
THRESH_MEAN = [50, 100, 500, 2000]
FILL_THR = 0.3
SURV_S = [10, 30, 100, 300, 1000]
CROSS_THR = [10, 100, 1000]
COLS = ["step", "max_fitness", "mean_fitness", "max_size", "mean_size", "cum_pattern_types",
        "n_active", "n_components"]
KNOWN = ("complete", "extinct")

SENS_L = [100, 200, 240, 300, 320, 400]
SENS_GRID = {   # grid -> [(value, label, directory name)]
    "mu": [(0.001, "0.001", "original_mix_mu0.001"), (0.002, "0.002", "original_mix_mu0.002"),
           (0.005, "0.005", "original_mix_mu0.005"), (0.01, "0.01", "original_mix_mu0.01"),
           (0.002 / 0.999, "0.002/0.999 (baseline)", "original_mix")],
    "p_death": [(0.0005, "0.0005", "original_mix_pd0.0005"), (0.001, "0.001 (baseline)", "original_mix"),
                (0.005, "0.005", "original_mix_pd0.005"), (0.01, "0.01", "original_mix_pd0.01")],
}
H40K_L = [240, 320, 400]
H40K_SEEDS, SENS_SEEDS = 100, 30
# state of a run during the correlation window CORR (see window_state)
WSTATES = [("compact", "compact throughout (no crossing by 10k)"),
           ("transitional", "nucleated in the window (crossing in (2k, 10k])"),
           ("runaway", "runaway throughout (crossed by 2k)")]
# snapshot panels of the 'phenomenon' figure: (L, seed, snapshot times)
SNAP_PANELS = [(200, 0, (5000, 10000)), (400, 5, (2000, 5000, 10000))]
N_PERM = [100_000]   # set from --perm in main(); used to print permutation p-values at their floor
MIN_GROUP_CI = 5     # smallest group of runs that gets a bootstrap interval (window-state groups)

# ---------------- figure style ----------------
sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "figure.dpi": 150, "savefig.dpi": 300,
})
# colour-blind-safe categorical colours: seaborn "colorblind" (Okabe-Ito based);
# series that share a panel also differ in line style and marker. Ordered series (L, s) use a
# truncated viridis ramp (wide lightness range) plus line styles.
PAL = sns.color_palette("colorblind")
C_MAIN, C_ALT, C_GREY, C_MED = PAL[0], PAL[1], "#8c8c8c", "#2b2b2b"
LSTYLES = ["-", "--", "-.", ":", (0, (5, 1.2)), (0, (3, 1, 1, 1, 1, 1))]


def ordered_colors(n: int) -> list:
    return [plt.get_cmap("viridis")(x) for x in np.linspace(0.0, 0.85, max(n, 1))][:n]
MM = 1 / 25.4
DOUBLE_COL = 180 * MM


# =============================== small helpers ===============================
def wilson(k, n):
    return tu.wilson(int(k), int(n)) if n else (np.nan, np.nan)


def boot_weights(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Resampling weights: row b gives how often each run is drawn, divided by n."""
    idx = rng.integers(0, n, size=(B, n))
    W = np.zeros((B, n))
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return W / n


def fisher_mean(r: np.ndarray, W: np.ndarray) -> tuple[float, float, float]:
    """Back-transformed mean Fisher z over runs with a bootstrap interval."""
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    m = np.isfinite(z)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    w = W[:, m]
    zb = (w @ z[m]) / w.sum(axis=1)
    lo, hi = tu.pct(zb)
    return float(np.tanh(z[m].mean())), float(np.tanh(lo)), float(np.tanh(hi))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


def detrend(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    tc = t - t.mean()
    b = (tc * (x - x.mean())).sum() / (tc * tc).sum()
    return x - x.mean() - b * tc


def first_step(cond: np.ndarray, steps: np.ndarray) -> float:
    i = np.flatnonzero(cond)
    return float(steps[i[0]]) if i.size else np.nan


def window_state(t_cross: float, status: str, final_step: int) -> str:
    """State of a run during the correlation window CORR = (2,000, 10,000), from its first crossing of
    mean size 100 (runaway is absorbing: no run that crossed fell back, see absorbing_check.csv)."""
    crossed = np.isfinite(t_cross)
    if crossed and t_cross <= CORR[0]:
        return "runaway"
    if status == "extinct" and final_step <= CORR[1] and not crossed:
        return "extinct"
    if not crossed or t_cross > CORR[1]:
        return "compact"
    return "transitional"


def bca(theta: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron 1987) from bootstrap replicates and
    leave-one-out (jackknife) estimates: (lo, hi, adjusted lower level, adjusted upper level).
    NaN when the bias correction is undefined."""
    boot, jack = boot[np.isfinite(boot)], jack[np.isfinite(jack)]
    if len(boot) < 100 or len(jack) < 3 or not np.isfinite(theta):
        return np.nan, np.nan, np.nan, np.nan
    p0 = (boot < theta).mean() + 0.5 * (boot == theta).mean()
    if not 0 < p0 < 1:
        return np.nan, np.nan, np.nan, np.nan
    z0 = stats.norm.ppf(p0)
    d = jack.mean() - jack
    den = 6.0 * (d ** 2).sum() ** 1.5
    a = (d ** 3).sum() / den if den > 0 else 0.0
    zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    q = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    lo, hi = np.quantile(boot, q)
    return float(lo), float(hi), float(q[0]), float(q[1])


def p_str(p, perm: bool = False) -> str:
    """p-value text; a permutation p at its floor 1/(n_perm+1) is printed as '<= floor'."""
    if p is None or not np.isfinite(p):
        return "n/a"
    floor = 1.0 / (N_PERM[0] + 1)
    if perm and p <= floor * (1 + 1e-9):
        return f"<= {floor:.1e}"
    return fmt_p(p)


def tidy_p(text: str) -> str:
    """'p = <= 1e-05' (a permutation p at its floor inserted after 'p = ') -> 'p <= 1e-05'."""
    return text.replace("p = <= ", "p <= ")


def fmt_num(v, nd=1, big=True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "pending data"
    if big and abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3g}"


def md_table(df: pd.DataFrame, floatfmt: dict | None = None, default: str = "{:.3g}") -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.get(c, default).format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =============================== run loading ===============================
def read_run(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path, usecols=COLS)
    except Exception:          # missing, empty or being written
        return None
    if df.empty:
        return None
    return {c: df[c].to_numpy(float) for c in COLS}


def classify(r: dict, T: int, stop: float) -> str:
    st = r["step"]
    last = int(st[-1])
    if last > T or not np.array_equal(st, np.arange(0, last + 1, DT)):
        return "bad_grid"
    if r["n_active"][-1] == 0:
        return "extinct"
    if last == T:
        return "complete"
    if stop and r["mean_size"][-1] >= stop:
        return "stopped"
    return "truncated"


def padded(r: dict, status: str, T: int) -> dict:
    """Trajectories on the full grid 0..T: zeros after extinction, NaN after an early stop."""
    n, m = T // DT + 1, len(r["step"])
    out = {"step": np.arange(0, T + 1, DT, dtype=float)}
    for c in COLS[1:]:
        a = np.full(n, np.nan)
        a[:m] = r[c]
        if status == "extinct":
            if c in ("max_size", "mean_size", "n_active", "n_components"):
                a[m:] = 0.0
            elif c == "cum_pattern_types":
                a[m:] = r[c][-1]
        out[c] = a
    dead = out["n_active"] == 0
    for c in ("mean_fitness", "max_fitness"):
        out[c][dead] = np.nan
    return out


def summarize(r: dict, status: str, L: int, T: int) -> tuple[dict, dict]:
    p = padded(r, status, T)
    st, ms, mx = r["step"], r["mean_size"], r["max_size"]
    known = status in KNOWN
    rec = dict(status=status, final_step=int(st[-1]),
               final_mean=float(ms[-1]), final_max=float(mx[-1]),
               final_n_active=float(r["n_active"][-1]), final_n_components=float(r["n_components"][-1]),
               final_cum_patterns=float(r["cum_pattern_types"][-1]),
               lifetime_max_size=float(mx.max()), lifetime_max_mean=float(ms.max()))
    rec["fill"] = rec["final_max"] / L ** 2 if known else np.nan
    rec["density"] = rec["final_n_active"] / L ** 2 if known else np.nan
    if not known:
        for c in ("final_mean", "final_max", "final_n_active", "final_n_components", "final_cum_patterns"):
            rec[c + "_at_stop"] = rec[c]
            rec[c] = np.nan
    # runaway indicators (a stopped run had mean >= 2000 at the stop: runaway if runaway is absorbing)
    rec["runaway"] = float(rec["final_mean"] > RUN) if known else (1.0 if status == "stopped" else np.nan)
    for thr in THRESH_MEAN:
        rec[f"run_mean{thr}"] = (float(rec["final_mean"] > thr) if known
                                 else (1.0 if status == "stopped" and thr < STOP else np.nan))
    rec[f"run_fill{FILL_THR:g}"] = float(rec["fill"] > FILL_THR) if known else np.nan
    # crossing (nucleation) times
    for thr in CROSS_THR:
        rec[f"t_cross{thr}"] = first_step(ms > thr, st)
    rec["t_fill"] = first_step(mx / L ** 2 > FILL_THR, st)
    rec["window_state"] = window_state(rec["t_cross100"], status, rec["final_step"])
    # absorbing check: behaviour after the first crossing of 100 / of fill 0.3
    tc = rec["t_cross100"]
    after = ms[st > tc] if np.isfinite(tc) else np.array([])
    rec["n_dips_after_cross"] = int((after <= RUN).sum())
    rec["min_mean_after_cross"] = float(after.min()) if after.size else np.nan
    tf = rec["t_fill"]
    afterf = mx[st > tf] / L ** 2 if np.isfinite(tf) else np.array([])
    rec["n_fill_dips"] = int((afterf <= FILL_THR).sum())
    rec["min_fill_after_cross"] = float(afterf.min()) if afterf.size else np.nan
    # windows (NaN when the run stopped before the window ends)
    pst, pms, pq = p["step"], p["mean_size"], p["mean_fitness"]
    rec["early"] = float(pms[tu.window(pst, EARLY)].mean()) if T >= EARLY[1] else np.nan
    rec["late"] = float(pms[tu.window(pst, LATE)].mean()) if T >= LATE[1] else np.nan
    w = tu.window(pst, CORR)
    x, y, t = pms[w], pq[w], pst[w]
    rec["r_run"] = rec["r_detr"] = np.nan
    if T >= CORR[1] and not np.isnan(x).any():
        a = np.isfinite(y)
        if a.sum() >= 20:
            rec["r_run"] = pearson(x[a], y[a])
            rec["r_detr"] = pearson(detrend(t[a], x[a]), detrend(t[a], y[a]))
    return rec, p


def load_main(L: int) -> tuple[list[dict], dict]:
    """Record and padded trajectory of every seed of the size scan at this L."""
    recs, trajs = [], {}
    for s in SEEDS:
        f = FULL / f"L{L}" / "runs" / f"seed_{s}.csv"
        r = read_run(f) if f.exists() else None
        st = classify(r, T_END, 0) if r is not None else None
        if st not in KNOWN:
            recs.append(dict(L=L, seed=s, status="missing", source="", note=f"no-stop file {st}" if st else ""))
            continue
        rec, p = summarize(r, st, L, T_END)
        rec.update(L=L, seed=s, source="full", note="")
        recs.append(rec)
        trajs[s] = p
    return recs, trajs


def load_simple(d: Path, seeds, L: int, T: int, stop: float) -> list[dict]:
    recs = []
    for s in seeds:
        f = d / "runs" / f"seed_{s}.csv"
        r = read_run(f) if f.exists() else None
        if r is None:
            continue
        st = classify(r, T, stop)
        if st not in ("complete", "extinct", "stopped"):
            continue
        rec, _ = summarize(r, st, L, T)
        rec.update(L=L, seed=s)
        recs.append(rec)
    return recs


# =============================== Table 2 ===============================
def table2_row(L: int, df: pd.DataFrame, trajs: dict, B: int, allow_partial: bool,
               ens_store: list) -> dict:
    rng = np.random.default_rng([SEED, L])
    present = df[df.status != "missing"]
    known = df[df.status.isin(KNOWN)].sort_values("seed")
    complete = len(known) == len(SEEDS)
    k = int(np.nansum(present.runaway))
    lo, hi = wilson(k, len(present))
    row = dict(L=L, n_seeds=len(SEEDS), n_present=len(present), n_nostop=len(known),
               n_missing=int((df.status == "missing").sum()),
               complete=complete,
               runaway_source="no-stop" if complete else "partial",
               runaway_k=k, runaway_n=len(present), runaway_frac=k / len(present) if len(present) else np.nan,
               runaway_lo=lo, runaway_hi=hi,
               n_extinct=int((known.status == "extinct").sum()))
    state_cols = [f"{c}_{g}{s}" for g, _ in WSTATES for c in ("r_detr", "r_run")
                  for s in ("_n", "", "_lo", "_hi", "_median")] + [f"r_detr_{g}_frac_neg" for g, _ in WSTATES]
    size_cols = ["mean_size", "mean_size_lo", "mean_size_hi", "mean_size_q25", "mean_size_median",
                 "mean_size_q75", "max_size", "max_size_lo", "max_size_hi", "max_size_q25",
                 "max_size_median", "max_size_q75", "late_early", "late_early_lo", "late_early_hi",
                 "late_early_nonrunaway", "r", "r_lo", "r_hi", "r_bca_lo", "r_bca_hi", "r_boot_median",
                 "r_boot_p_below", "r_bca_ok", "r_bca_level_lo", "r_bca_level_hi", "r_jack_lo", "r_jack_hi",
                 "r_run", "r_run_lo", "r_run_hi",
                 "r_run_median", "r_detr", "r_detr_lo", "r_detr_hi", "r_detr_median",
                 "r_detr_by_final_outcome_runaway", "r_detr_by_final_outcome_compact", "r_detr_frac_neg"
                 ] + state_cols
    row.update({c: np.nan for c in size_cols})
    row["size_status"] = "complete" if complete else "pending data"
    if not (complete or (allow_partial and len(known) >= 5)):
        return row
    if not complete:
        row["size_status"] = f"PROVISIONAL ({len(known)}/{len(SEEDS)} runs; runaway runs under-represented)"
    n = len(known)
    W = boot_weights(n, B, rng)
    fm, fx = known.final_mean.to_numpy(), known.final_max.to_numpy()
    q_m, q_x = np.percentile(fm, [25, 50, 75]), np.percentile(fx, [25, 50, 75])
    row.update(mean_size=fm.mean(), mean_size_lo=tu.pct(W @ fm)[0], mean_size_hi=tu.pct(W @ fm)[1],
               mean_size_q25=q_m[0], mean_size_median=q_m[1], mean_size_q75=q_m[2],
               max_size=fx.mean(), max_size_lo=tu.pct(W @ fx)[0], max_size_hi=tu.pct(W @ fx)[1],
               max_size_q25=q_x[0], max_size_median=q_x[1], max_size_q75=q_x[2])
    early, late = known.early.to_numpy(), known.late.to_numpy()
    le_b = (W @ late) / (W @ early)
    row.update(late_early=late.mean() / early.mean(), late_early_lo=tu.pct(le_b)[0],
               late_early_hi=tu.pct(le_b)[1])
    # compact runs = non-runaway and not extinct (as in part (e))
    nr = (known.runaway.to_numpy() == 0) & (known.status.to_numpy() != "extinct")
    if nr.any():
        row["late_early_nonrunaway"] = late[nr].mean() / early[nr].mean()
    # ensemble r (paper definition) with run-level bootstrap
    seeds = known.seed.tolist()
    w = tu.window(trajs[seeds[0]]["step"], CORR)
    S = np.stack([trajs[s]["mean_size"][w] for s in seeds])
    Q = np.stack([trajs[s]["mean_fitness"][w] for s in seeds])
    A = np.isfinite(Q).astype(float)
    Qz = np.where(A > 0, Q, 0.0)
    row["r"] = pearson(S.mean(axis=0), Qz.sum(axis=0) / A.sum(axis=0))
    boot_r = np.concatenate([tu.rowwise_pearson(W[i:i + 500] @ S, (W[i:i + 500] @ Qz) / (W[i:i + 500] @ A))
                             for i in range(0, B, 500)])
    row["r_lo"], row["r_hi"] = tu.pct(boot_r)
    row["r_boot_median"] = float(np.nanmedian(boot_r))
    # BCa: the percentile interval is strongly skewed (the estimate can sit at its edge)
    with np.errstate(invalid="ignore", divide="ignore"):
        jack = tu.rowwise_pearson((S.sum(axis=0)[None] - S) / (n - 1),
                                  (Qz.sum(axis=0)[None] - Qz) / (A.sum(axis=0)[None] - A))
    row["r_bca_lo"], row["r_bca_hi"], q1, q2 = bca(row["r"], boot_r, jack)
    # diagnostics: share of bootstrap r below the estimate (0.5 = unbiased; the resampling noise in the
    # ensemble-mean trajectory attenuates r, so it is usually small) and whether BCa is usable (it
    # extrapolates into the extreme tail and can exclude the estimate when that share is near 0)
    row["r_boot_p_below"] = float((boot_r[np.isfinite(boot_r)] < row["r"]).mean())
    # usable only if >= 10 bootstrap replicates lie beyond each adjusted quantile level and the
    # interval contains the estimate
    row["r_bca_ok"] = bool(np.isfinite(row["r_bca_lo"]) and q1 * B >= 10 and (1 - q2) * B >= 10
                           and row["r_bca_lo"] <= row["r"] <= row["r_bca_hi"])
    row["r_bca_level_lo"], row["r_bca_level_hi"] = q1, q2
    # jackknife interval on the Fisher z scale (stable alternative; contains the estimate by construction)
    zj = np.arctanh(np.clip(jack[np.isfinite(jack)], -0.999999, 0.999999))
    if len(zj) >= 3:
        se = np.sqrt((len(zj) - 1) / len(zj) * ((zj - zj.mean()) ** 2).sum())
        z_r = np.arctanh(np.clip(row["r"], -0.999999, 0.999999))
        row["r_jack_lo"], row["r_jack_hi"] = float(np.tanh(z_r - 1.959964 * se)), float(np.tanh(z_r + 1.959964 * se))
    rr, rd = known.r_run.to_numpy(), known.r_detr.to_numpy()
    row["r_run"], row["r_run_lo"], row["r_run_hi"] = fisher_mean(rr, W)
    row["r_detr"], row["r_detr_lo"], row["r_detr_hi"] = fisher_mean(rd, W)
    row["r_run_median"] = float(np.nanmedian(rr))
    row["r_detr_median"] = float(np.nanmedian(rd))
    zr = np.arctanh(np.clip(rd, -0.999999, 0.999999))
    ok = np.isfinite(zr)
    # grouping by the outcome at t = 20,000: this puts runs that nucleated inside the 2k-10k window
    # into the 'runaway' group (the window-state groups below keep them apart)
    rw = known.runaway.to_numpy() == 1
    row["r_detr_by_final_outcome_runaway"] = float(np.tanh(zr[ok & rw].mean())) if (ok & rw).any() else np.nan
    row["r_detr_by_final_outcome_compact"] = float(np.tanh(zr[ok & nr].mean())) if (ok & nr).any() else np.nan
    row["r_detr_frac_neg"] = float((rd[ok] < 0).mean()) if ok.any() else np.nan
    # per-run r grouped by the state of the run DURING the correlation window
    ws = known.window_state.to_numpy()
    for gi, (g, _) in enumerate(WSTATES):
        for ci, (col, arr) in enumerate((("r_detr", rd), ("r_run", rr))):
            x = arr[(ws == g) & np.isfinite(arr)]
            row[f"{col}_{g}_n"] = len(x)
            if len(x):
                row[f"{col}_{g}_median"] = float(np.median(x))
                row[f"{col}_{g}"] = float(np.tanh(np.arctanh(np.clip(x, -0.999999, 0.999999)).mean()))
            if len(x) >= MIN_GROUP_CI:
                Wg = boot_weights(len(x), B, np.random.default_rng([SEED, L, 4, gi, ci]))
                row[f"{col}_{g}"], row[f"{col}_{g}_lo"], row[f"{col}_{g}_hi"] = fisher_mean(x, Wg)
            if col == "r_detr" and len(x):
                row[f"r_detr_{g}_frac_neg"] = float((x < 0).mean())
    # ensemble trajectories every 100 steps
    sel = slice(0, None, 10)
    st = trajs[seeds[0]]["step"][sel]
    M = {c: np.stack([trajs[s][c][sel] for s in seeds]) for c in ("mean_size", "max_size", "mean_fitness",
                                                                   "max_fitness", "cum_pattern_types")}
    with np.errstate(all="ignore"):
        ens_store.append(pd.DataFrame(dict(
            L=L, step=st, n_runs=len(seeds), n_alive=np.isfinite(M["mean_fitness"]).sum(axis=0),
            mean_size_mean=M["mean_size"].mean(axis=0), mean_size_median=np.median(M["mean_size"], axis=0),
            max_size_mean=M["max_size"].mean(axis=0), mean_fitness_mean=np.nanmean(M["mean_fitness"], axis=0),
            max_fitness_mean=np.nanmean(M["max_fitness"], axis=0),
            cum_patterns_median=np.median(M["cum_pattern_types"], axis=0))))
    return row


def fmt_cell(v, lo, hi, kind) -> str:
    """One Table 2 cell in the form \\valci{value}{lo, hi}."""
    if not np.isfinite(v):
        return "pending"

    def f(x):
        if kind == "frac":
            s = f"{x:.2f}"
        elif kind == "r":
            s = f"{x:.2f}"
        elif kind == "ratio":
            s = f"{x:.2f}" if abs(x) < 100 else f"{x:,.0f}"
        else:
            s = f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}"
        return s.replace(",", "{,}").replace("-", "$-$")

    return f"\\valci{{{f(v)}}}{{{f(lo)}, {f(hi)}}}"


def table2_tex(t2: pd.DataFrame) -> str:
    out = ["% Table 2 rows (generated by scripts/npj_rev_ext2_tables.py; paste into tabular*)",
           "% L & Mean size & Max size & Late/Early & Runaway frac. & r(sbar,qbar)"]
    for _, r in t2.iterrows():
        cells = [str(int(r.L)),
                 fmt_cell(r.mean_size, r.mean_size_lo, r.mean_size_hi, "size"),
                 fmt_cell(r.max_size, r.max_size_lo, r.max_size_hi, "size"),
                 fmt_cell(r.late_early, r.late_early_lo, r.late_early_hi, "ratio"),
                 fmt_cell(r.runaway_frac, r.runaway_lo, r.runaway_hi, "frac"),
                 fmt_cell(r.r, r.r_lo, r.r_hi, "r")]
        flag = "" if r.complete else f"  % runaway: {r.runaway_source}; sizes: {r.size_status}"
        out.append(" & ".join(cells) + r" \\" + flag)
    return "\n".join(out) + "\n"


# =============================== tests ===============================
def perm_tests(ks, ns, scores, n_perm: int, rng: np.random.Generator) -> dict:
    """Permutation tests on a k-by-2 table of runaway counts, all from the same permutations:
    chi-square homogeneity, Cochran-Armitage trend in `scores` (two-sided), and the maximum 2x2
    chi-square over all cut points between consecutive groups (a change-point test whose p is
    corrected for the choice of cut)."""
    ks, ns, scores = np.asarray(ks, int), np.asarray(ns, int), np.asarray(scores, float)
    K, N, Kt = len(ns), int(ns.sum()), int(ks.sum())
    y = np.concatenate([[1] * k + [0] * (n - k) for k, n in zip(ks, ns)]).astype(np.float32)
    G = np.zeros((N, K), np.float32)
    G[np.arange(N), np.repeat(np.arange(K), ns)] = 1
    p0 = Kt / N
    e1, e0 = ns * p0, ns * (1 - p0)
    cn = np.cumsum(ns)[:-1]
    ET = p0 * (scores * ns).sum()

    def stats_of(o1):  # o1: (b, K) runaway counts per group
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = (((o1 - e1) ** 2 / e1) + (((ns - o1) - e0) ** 2 / e0)).sum(axis=1)
            trend = np.abs(o1 @ scores - ET)
            c1 = np.cumsum(o1, axis=1)[:, :-1]
            n1, n2 = cn, N - cn
            k2 = Kt - c1
            split = N * (c1 * (n2 - k2) - k2 * (n1 - c1)) ** 2 / (n1 * n2 * Kt * (N - Kt))
        return np.nan_to_num(chi), trend, np.nan_to_num(split)

    chi_o, tr_o, sp_o = stats_of(ks[None].astype(float))
    hits = np.zeros(3)
    done, batch = 0, 2000
    while done < n_perm:
        b = min(batch, n_perm - done)
        P = rng.permuted(np.tile(y, (b, 1)), axis=1)
        chi, tr, sp = stats_of((P @ G).astype(float))
        hits += [(chi >= chi_o[0] - 1e-9).sum(), (tr >= tr_o[0] - 1e-9).sum(),
                 (sp.max(axis=1) >= sp_o[0].max() - 1e-9).sum()]
        done += b
    pv = (hits + 1) / (n_perm + 1)
    var_t = p0 * (1 - p0) * ((scores ** 2 * ns).sum() - (scores * ns).sum() ** 2 / N)
    z = ((ks * scores).sum() - ET) / np.sqrt(var_t) if var_t > 0 else np.nan
    j = int(np.argmax(sp_o[0]))
    return dict(chi2=float(chi_o[0]), p_chi2_perm=float(pv[0]),
                p_chi2_asym=float(stats.chi2.sf(chi_o[0], K - 1)),
                trend_z=float(z), p_trend_perm=float(pv[1]),
                p_trend_asym=float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                best_cut_index=j, max_split_chi2=float(sp_o[0][j]), p_max_split_perm=float(pv[2]))


def matched_tests(X: np.ndarray, scores, n_perm: int, B: int, rng: np.random.Generator) -> dict:
    """Tests that respect the reuse of seeds 0..S-1 at every size. X: seeds x sizes 0/1 runaway matrix.
    Cochran's Q (asymptotic chi2, K-1 df); within-seed permutation tests (each seed's outcomes are
    permuted across the sizes, which is the exact null of Q) of homogeneity (same chi2 statistic as
    perm_tests) and of a linear trend in `scores` (stratified Cochran-Armitage; z from the exact
    permutation variance); seed-cluster bootstrap (resampling seeds) of the pooled fraction."""
    X = np.asarray(X, float)
    S, K = X.shape
    s = np.asarray(scores, float)
    C, R = X.sum(axis=0), X.sum(axis=1)
    N = C.sum()
    den = K * N - (R ** 2).sum()
    Qc = (K - 1) * (K * (C ** 2).sum() - N ** 2) / den if den > 0 else np.nan
    p0 = N / (S * K)
    e1, e0 = S * p0, S * (1 - p0)
    ET = N * s.mean()

    def chi_of(c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(((c - e1) ** 2 / e1 + ((S - c) - e0) ** 2 / e0).sum(axis=-1))

    chi_o, tr_o = float(chi_of(C)), float(C @ s - ET)
    var_t = (R * (K - R) / (K - 1) * ((s - s.mean()) ** 2).mean()).sum()
    mixed = (R > 0) & (R < K)
    Xm, fixed = X[mixed], X[~mixed].sum(axis=0)
    hits = np.zeros(2)
    done = 0
    while done < n_perm:
        b = min(1000, n_perm - done)
        P = rng.permuted(np.broadcast_to(Xm, (b,) + Xm.shape).copy(), axis=2)
        c = P.sum(axis=1) + fixed
        hits += [(chi_of(c) >= chi_o - 1e-9).sum(), (np.abs(c @ s - ET) >= abs(tr_o) - 1e-9).sum()]
        done += b
    pv = (hits + 1) / (n_perm + 1)
    W = boot_weights(S, B, rng)
    lo, hi = tu.pct(W @ X.mean(axis=1))
    return dict(cochran_Q=float(Qc), p_cochran_Q=float(stats.chi2.sf(Qc, K - 1)) if np.isfinite(Qc) else np.nan,
                n_informative_seeds=int(mixed.sum()), chi2_within=chi_o, p_chi2_within=float(pv[0]),
                trend_z_within=float(tr_o / np.sqrt(var_t)) if var_t > 0 else np.nan,
                p_trend_within=float(pv[1]), pooled=float(X.mean()), pooled_seedboot_lo=lo,
                pooled_seedboot_hi=hi)


def phi_matrix(X: np.ndarray) -> np.ndarray:
    """phi (Pearson r of 0/1 columns) for every pair of columns of X (... x seeds x sizes)."""
    Xc = X - X.mean(axis=-2, keepdims=True)
    cov = np.einsum("...sk,...sl->...kl", Xc, Xc)
    sd = np.sqrt(np.einsum("...kk->...k", cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        return cov / (sd[..., :, None] * sd[..., None, :])


def logistic(x: np.ndarray, y: np.ndarray, quad: bool = False) -> dict:
    """Maximum-likelihood logistic regression by Newton-Raphson (no statsmodels)."""
    xc = (x - 300.0) / 100.0
    X = np.column_stack([np.ones_like(xc), xc] + ([xc ** 2] if quad else []))
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))
    p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    ll0 = float((y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum())
    return dict(beta=b, se=se, ll=ll, ll0=ll0, n=len(y))


def logrank(times, events, groups, scores=None) -> dict:
    """k-sample log-rank test (and the trend version with group scores)."""
    times, events, groups = map(np.asarray, (times, events, groups))
    labs = np.unique(groups)
    K = len(labs)
    gi = np.searchsorted(labs, groups)
    U = np.zeros(K)
    V = np.zeros((K, K))
    for t in np.unique(times[events == 1]):
        at = times >= t
        nj = np.bincount(gi[at], minlength=K).astype(float)
        dj = np.bincount(gi[(times == t) & (events == 1)], minlength=K).astype(float)
        n, d = nj.sum(), dj.sum()
        if n < 2:
            continue
        U += dj - nj * d / n
        f = d * (n - d) / (n - 1)
        V += f * (np.diag(nj / n) - np.outer(nj, nj) / n ** 2)
    out = {}
    try:
        chi = float(U[:-1] @ np.linalg.solve(V[:-1, :-1], U[:-1]))
        out.update(logrank_chi2=chi, logrank_df=K - 1, p_logrank=float(stats.chi2.sf(chi, K - 1)))
    except np.linalg.LinAlgError:
        out.update(logrank_chi2=np.nan, logrank_df=K - 1, p_logrank=np.nan)
    if scores is not None:
        w = np.asarray([scores[l] for l in labs], float)
        z = float(w @ U / np.sqrt(w @ V @ w))
        out.update(trend_z=z, p_trend=float(2 * stats.norm.sf(abs(z))))
    return out


def km_incidence(times, events, grid) -> np.ndarray:
    """1 - Kaplan-Meier survival of 'not yet nucleated' evaluated on grid."""
    times, events = np.asarray(times, float), np.asarray(events, int)
    ev_t = np.unique(times[events == 1])
    S, s = [], 1.0
    surv_at = {}
    for t in ev_t:
        n = (times >= t).sum()
        d = ((times == t) & (events == 1)).sum()
        s *= 1 - d / n
        surv_at[t] = s
    keys = np.array(sorted(surv_at))
    vals = np.array([surv_at[k] for k in keys])
    for g in grid:
        i = np.searchsorted(keys, g, side="right") - 1
        S.append(vals[i] if i >= 0 else 1.0)
    return 1 - np.array(S)


def poisson_ci(k, exposure, a=0.05):
    lo = stats.chi2.ppf(a / 2, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


# =============================== main analysis ===============================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true",
                    help="compute size columns on incomplete L as PROVISIONAL (biased)")
    ap.add_argument("--boot", type=int, default=tu.B)
    ap.add_argument("--perm", type=int, default=100_000)
    ap.add_argument("--out", default=str(RES / "npj_rev" / "ext2_tables"))
    args = ap.parse_args()
    N_PERM[0] = args.perm
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matched: dict = {}       # matched-seed test results per subset of sizes
    issues: list[str] = []
    md: list[str] = ["# Extension II (Sec. 4): statistics of the size scan\n",
                     "Generated by `scripts/npj_rev_ext2_tables.py` from the runs of the C implementation, "
                     "seeds 0-99 per L. See the script docstring for all definitions.\n"]

    # ---------------- load the main scan ----------------
    per_run, t2_rows, ens = [], [], []
    for L in ALL_L:
        recs, trajs = load_main(L)
        df = pd.DataFrame(recs)
        if (df.status == "missing").all():
            per_run.append(df)
            continue
        per_run.append(df)
        if L in TABLE_L or L == 100 or (df.status.isin(KNOWN).all()):
            row = table2_row(L, df, trajs, args.boot, args.allow_partial, ens)
            t2_rows.append(row)
        del trajs
        print(f"L={L}: {int(df.status.isin(KNOWN).sum())} complete, {int((df.status == 'missing').sum())} missing",
              flush=True)
    pr = pd.concat(per_run, ignore_index=True)
    lead = ["L", "seed", "status", "source", "note"]
    pr = pr[lead + [c for c in pr.columns if c not in lead]]
    pr.to_csv(out / "per_run.csv", index=False)
    present = pr[pr.status != "missing"]

    status = (pr.assign(nostop=pr.status.isin(KNOWN), missing=pr.status == "missing")
              .groupby("L").agg(n_nostop=("nostop", "sum"), n_missing=("missing", "sum"))
              .reset_index())
    status["complete"] = status.n_nostop == len(SEEDS)
    status.to_csv(out / "data_status.csv", index=False)

    # ---------------- (a) Table 2 ----------------
    t2 = pd.DataFrame(t2_rows)
    t2tab = t2[t2.L.isin(TABLE_L)].reset_index(drop=True)
    t2tab.to_csv(out / "table2.csv", index=False)
    (out / "table2_valci.tex").write_text(table2_tex(t2tab))
    if len(ens):
        pd.concat(ens, ignore_index=True).to_csv(out / "ensemble_trajectories.csv", index=False)
    ensdf = pd.concat(ens, ignore_index=True) if ens else pd.DataFrame()

    md.append("## Data status\n")
    md.append("`n_nostop` runs have the full 20k trajectory of the size scan (no early stop); `n_missing` have "
              "no usable file. A condition is complete when all 100 seeds have the full trajectory.\n")
    md.append(md_table(status, default="{:.0f}"))
    md.append("")

    md.append("## (a) Table 2 (L = 200..400)\n")
    md.append("Mean/max size at t = 20,000 (extinct runs as 0), ensemble mean [95% percentile bootstrap]; "
              "Late/Early = mean(late)/mean(early) of per-run window means; runaway fraction with Wilson "
              "interval; r = ensemble-mean size vs ensemble-mean score over 2k-10k. `r_run`/`r_detr` are the "
              "per-run alternatives (raw / linearly detrended; mean Fisher z over runs).\n")
    view = []
    for _, r in t2tab.iterrows():
        cell = lambda v, lo, hi, nd=1: ("pending" if not np.isfinite(v) else
                                        f"{fmt_num(v, nd)} [{fmt_num(lo, nd)}, {fmt_num(hi, nd)}]")
        view.append({
            "L": int(r.L), "n": int(r.runaway_n),
            "runaway": f"{int(r.runaway_k)}/{int(r.runaway_n)} = {r.runaway_frac:.2f} [{r.runaway_lo:.2f}, {r.runaway_hi:.2f}]",
            "mean size": cell(r.mean_size, r.mean_size_lo, r.mean_size_hi),
            "mean size q25/med/q75": "pending" if not np.isfinite(r.mean_size_median) else
            f"{r.mean_size_q25:.2f}/{r.mean_size_median:.2f}/{fmt_num(r.mean_size_q75, 2)}",
            "max size": cell(r.max_size, r.max_size_lo, r.max_size_hi),
            "max q25/med/q75": "pending" if not np.isfinite(r.max_size_median) else
            f"{r.max_size_q25:.0f}/{r.max_size_median:.0f}/{r.max_size_q75:,.0f}",
            "Late/Early": cell(r.late_early, r.late_early_lo, r.late_early_hi, 2),
            "r (paper)": cell(r.r, r.r_lo, r.r_hi, 2),
            "r BCa / jackknife-z": "pending" if not np.isfinite(r.r) else
            ((f"[{fmt_num(r.r_bca_lo, 2)}, {fmt_num(r.r_bca_hi, 2)}]" if r.r_bca_ok else "unreliable")
             + f" / [{fmt_num(r.r_jack_lo, 2)}, {fmt_num(r.r_jack_hi, 2)}]"),
            "r_run": cell(r.r_run, r.r_run_lo, r.r_run_hi, 2),
            "r_detr": cell(r.r_detr, r.r_detr_lo, r.r_detr_hi, 2),
        })
    md.append(md_table(pd.DataFrame(view)))
    md.append("\nLaTeX rows: `table2_valci.tex`.\n")
    rr = t2[np.isfinite(t2.r)]
    if len(rr):
        md.append("Interval for the ensemble r. Its bootstrap distribution is skewed and biased towards weaker "
                  "correlation (resampling duplicates runs, and the extra noise in the ensemble-mean size "
                  "trajectory attenuates r), so the estimate sits at or beyond the lower edge of its percentile "
                  "interval. BCa corrects for the bias but then reads quantiles far out in the tail (adjusted "
                  "levels in table2.csv `r_bca_level_*`); it is marked `unreliable` when fewer than 10 bootstrap "
                  "replicates lie beyond an adjusted level or the BCa interval excludes the estimate. The jackknife "
                  "interval on the "
                  "Fisher z scale (r = tanh(z +- 1.96 SE_jack)) is a stable alternative. Per L: "
                  + "; ".join(f"L={int(x.L)}: r = {x.r:.3f}, percentile [{x.r_lo:.3f}, {x.r_hi:.3f}] (bootstrap "
                              f"median {x.r_boot_median:.3f}, share below r {x.r_boot_p_below:.3f}), BCa "
                              f"[{x.r_bca_lo:.3f}, {x.r_bca_hi:.3f}]{'' if x.r_bca_ok else ' unreliable'}, "
                              f"jackknife-z [{x.r_jack_lo:.3f}, {x.r_jack_hi:.3f}]"
                              for _, x in rr.iterrows())
                  + ".\n")
    if rr.r_detr.notna().any():
        md.append("### Per-run size-score correlation by the run's state during the 2k-10k window\n")
        md.append("Groups: " + "; ".join(f"`{g}` = {lab}" for g, lab in WSTATES) + " (runaway is absorbing; runs "
                  "extinct by 10k without crossing are excluded). Back-transformed mean Fisher z with a bootstrap "
                  "95% interval over the runs of each group; median and fraction < 0 of the per-run values. The "
                  "last column groups the runs by their outcome at t = 20,000 instead (table2.csv "
                  "`r_detr_by_final_outcome_*`), which puts runs that nucleated inside the window into the "
                  "'runaway' group.\n")
        sv_rows = []
        for _, x in rr.iterrows():
            for g, _lab in WSTATES:
                n_g = x.get(f"r_detr_{g}_n", np.nan)
                if not np.isfinite(n_g) or n_g == 0:
                    continue
                ci = lambda c: (f"{x[c]:.2f} [{x[c + '_lo']:.2f}, {x[c + '_hi']:.2f}]" if np.isfinite(x[c + "_lo"])
                                else f"{x[c]:.2f}")
                sv_rows.append({"L": int(x.L), "state in window": g, "n": int(n_g),
                                "r_detr [95% CI]": ci(f"r_detr_{g}"), "r_detr median": f"{x[f'r_detr_{g}_median']:.2f}",
                                "frac r_detr<0": f"{x[f'r_detr_{g}_frac_neg']:.2f}",
                                "r_run [95% CI]": ci(f"r_run_{g}"),
                                "r_detr by final outcome": "" if g == "transitional" else
                                f"{x['r_detr_by_final_outcome_' + ('runaway' if g == 'runaway' else 'compact')]:.2f}"})
        md.append(md_table(pd.DataFrame(sv_rows)))
        interp, pattern = [], []
        for _, x in rr.iterrows():
            c_lo, c_hi = x.get("r_detr_compact_lo", np.nan), x.get("r_detr_compact_hi", np.nan)
            r_lo, r_hi = x.get("r_detr_runaway_lo", np.nan), x.get("r_detr_runaway_hi", np.nan)
            if not (np.isfinite(c_lo) and np.isfinite(r_lo)):
                continue
            interp.append(f"L={int(x.L)}: compact throughout {x.r_detr_compact:.2f} (CI "
                          f"{'excludes' if c_hi < 0 or c_lo > 0 else 'includes'} 0, n={int(x.r_detr_compact_n)}), "
                          f"runaway throughout {x.r_detr_runaway:.2f} (CI "
                          f"{'excludes' if r_hi < 0 or r_lo > 0 else 'includes'} 0, n={int(x.r_detr_runaway_n)}), "
                          f"transitional {x.get('r_detr_transitional', np.nan):.2f} "
                          f"(n={int(x.get('r_detr_transitional_n', 0))})")
            pattern.append((int(x.L), bool(c_hi < 0 and r_lo <= 0 <= r_hi)))
        if interp:
            md.append("\nReading: " + "; ".join(interp) + ".")
            yes = [L_ for L_, ok_ in pattern if ok_]
            no = [L_ for L_, ok_ in pattern if not ok_]
            md.append(f"Pattern 'weakly negative while compact (CI < 0), indistinguishable from zero while runaway "
                      f"(CI includes 0)' holds at L = {yes if yes else 'none'}"
                      + (f", not at L = {no}" if no else "") + " (sizes with >= "
                      f"{MIN_GROUP_CI} runaway-throughout runs). When the runs are grouped by their final outcome, "
                      "the negative average of the 'runaway' group comes from the runs that nucleated inside the "
                      "window (transitional group).\n")

    # hash scores of the components present: ensemble means of the maximum and of the mean score
    trends = []
    for L in (100, 200):
        e = ensdf[ensdf.L == L] if len(ensdf) else ensdf
        if not len(e):
            continue
        mf = e.set_index("step")
        pk = mf.mean_fitness_mean.idxmax()
        trends.append(f"L={L}: ensemble max score {mf.max_fitness_mean.loc[1000]:.4f} (t=1k) -> "
                      f"{mf.max_fitness_mean.loc[20000]:.4f} (t=20k); mean score peaks at t={int(pk)} "
                      f"({mf.mean_fitness_mean.max():.3f}) and is {mf.mean_fitness_mean.loc[20000]:.3f} at t=20k")
    if trends:
        md.append("### Hash scores of the components present (ensemble means, sampled every 100 steps)\n")
        md.append("; ".join(trends) + ".\n")

    # ---------------- (b) transition statistics ----------------
    rb = []
    for L in ALL_L:
        d = present[present.L == L]
        if not len(d):
            rb.append(dict(L=L, n=0, k=np.nan, frac=np.nan, lo=np.nan, hi=np.nan, source="pending data"))
            continue
        k, n = int(np.nansum(d.runaway)), len(d)
        lo, hi = wilson(k, n)
        rb.append(dict(L=L, n=n, k=k, frac=k / n, lo=lo, hi=hi,
                       source="no-stop" if n == len(SEEDS) else f"PROVISIONAL partial ({n}/{len(SEEDS)})"))
    rb = pd.DataFrame(rb)
    # extra-seed check (L=100: 200 no-stop seeds)
    extra = []
    for L, d, T in [(100, EXTRA_L100, T_END)]:
        recs = load_simple(d, range(200), L, T, STOP)
        if recs:
            e = pd.DataFrame(recs)
            k = int(np.nansum(e.runaway))
            extra.append(dict(L=L, n=len(e), k=k, frac=k / len(e), lo=wilson(k, len(e))[0],
                              hi=wilson(k, len(e))[1], source=str(d.relative_to(ROOT))))
    rb.to_csv(out / "runaway_by_L.csv", index=False)

    tests = []
    # tests use only sizes with all seeds present: a partially filled size holds the runs that
    # finished first, which are the compact ones (runaway runs are slower)
    avail = rb[rb.n == len(SEEDS)].set_index("L")
    L_ok = sorted(avail.index)
    present_ok = present[present.L.isin(L_ok)]
    partial_L = rb[(rb.n > 0) & (rb.n < len(SEEDS))]
    if len(partial_L):
        issues.append("sizes with only some seeds present were left out of the tests (provisional, biased "
                      "towards fast compact runs): " + ", ".join(f"L={int(r.L)} ({int(r.n)}/{len(SEEDS)})"
                                                              for _, r in partial_L.iterrows()))

    def group(Ls):
        Ls = [L for L in Ls if L in avail.index]
        return Ls, [int(avail.loc[L, "k"]) for L in Ls], [int(avail.loc[L, "n"]) for L in Ls]

    def fisher_split(name, A, Bs):
        La, ka, na = group(A)
        Lb, kb, nb = group(Bs)
        if not La or not Lb:
            tests.append(dict(test="Fisher exact", subset=name, statistic=np.nan, p=np.nan, detail="pending data"))
            return
        odds, p = stats.fisher_exact([[sum(ka), sum(na) - sum(ka)], [sum(kb), sum(nb) - sum(kb)]])
        tests.append(dict(test="Fisher exact", subset=name, statistic=odds, p=p,
                          detail=f"{sum(ka)}/{sum(na)} (L={La[0]}..{La[-1]}) vs {sum(kb)}/{sum(nb)} "
                                 f"(L={Lb[0]}..{Lb[-1]})"))

    fisher_split("split L<=300 vs L>=320", [L for L in TABLE_L if L <= 300], [L for L in TABLE_L if L >= 320])
    fisher_split("L=300 vs L=320", [300], [320])
    fisher_split("onset split L<=220 vs L>=240 (post hoc)", [L for L in TABLE_L if L <= 220],
                 [L for L in TABLE_L if L >= 240])
    fisher_split("L=220 vs L=240", [220], [240])
    fisher_split("L=100 vs L=200", [100], [200])
    # adjacent pairs on the coarse grid with Holm correction
    adj = []
    Lc = [L for L in TABLE_L if L in avail.index]
    for a, b in zip(Lc[:-1], Lc[1:]):
        _, p = stats.fisher_exact([[avail.loc[a, "k"], avail.loc[a, "n"] - avail.loc[a, "k"]],
                                   [avail.loc[b, "k"], avail.loc[b, "n"] - avail.loc[b, "k"]]])
        adj.append((a, b, p))
    if adj:
        ps = np.array([x[2] for x in adj])
        order = np.argsort(ps)
        holm = np.empty_like(ps)
        run_max = 0
        for i, j in enumerate(order):
            run_max = max(run_max, min(1, (len(ps) - i) * ps[j]))
            holm[j] = run_max
        for (a, b, p), h in zip(adj, holm):
            tests.append(dict(test="Fisher exact, adjacent (Holm)", subset=f"L={a} vs L={b}", statistic=np.nan,
                              p=p, p_holm=h, detail=""))

    rngp = np.random.default_rng([SEED, 1])
    rngm = np.random.default_rng([SEED, 5])           # matched-seed tests (separate stream)
    Mseed = present.pivot(index="seed", columns="L", values="runaway")
    for name, Ls in [("coarse L=200..400", TABLE_L), ("plateau L=240..400", [L for L in TABLE_L if L >= 240]),
                     ("plateau L=260..400", [L for L in TABLE_L if L >= 260]),
                     ("L=320..400", [L for L in TABLE_L if L >= 320]),
                     ("all available L=100..400", ALL_L), ("onset L=100..260", ONSET_L)]:
        Ls2, ks, ns = group(Ls)
        if len(Ls2) < 3:
            tests.append(dict(test="permutation chi2 / trend / max-split", subset=name, statistic=np.nan,
                              p=np.nan, detail=f"pending data ({len(Ls2)} sizes available)"))
            continue
        pt = perm_tests(ks, ns, Ls2, args.perm, rngp)
        cut = Ls2[pt["best_cut_index"]]
        k_all, n_all = sum(ks), sum(ns)
        lo, hi = wilson(k_all, n_all)
        Xs = Mseed[Ls2].dropna()
        mt = matched_tests(Xs.to_numpy(), Ls2, args.perm, args.boot, rngm) if len(Xs) == len(SEEDS) else None
        tests.append(dict(test="permutation chi2 homogeneity", subset=name, statistic=pt["chi2"],
                          p=pt["p_chi2_perm"], p_asym=pt["p_chi2_asym"],
                          detail=f"L={Ls2[0]}..{Ls2[-1]} ({len(Ls2)} sizes), pooled {k_all}/{n_all} = "
                                 f"{k_all / n_all:.3f} Wilson [{lo:.3f}, {hi:.3f}]"
                                 + ("" if mt is None else f", seed-cluster bootstrap [{mt['pooled_seedboot_lo']:.3f}, "
                                                          f"{mt['pooled_seedboot_hi']:.3f}]")))
        tests.append(dict(test="Cochran-Armitage trend in L (permutation)", subset=name, statistic=pt["trend_z"],
                          p=pt["p_trend_perm"], p_asym=pt["p_trend_asym"], detail="z > 0: rises with L"))
        tests.append(dict(test="max-chi2 change point (permutation-corrected)", subset=name,
                          statistic=pt["max_split_chi2"], p=pt["p_max_split_perm"],
                          detail=f"best cut: L<={cut} vs L>{cut}"))
        if mt is not None:
            tests.append(dict(test="Cochran's Q homogeneity (matched seeds)", subset=name, statistic=mt["cochran_Q"],
                              p=mt["p_cochran_Q"], detail=f"chi2 with {len(Ls2) - 1} df; {mt['n_informative_seeds']} "
                                                          "seeds with mixed outcomes across L"))
            tests.append(dict(test="within-seed permutation chi2 homogeneity (matched seeds)", subset=name,
                              statistic=mt["chi2_within"], p=mt["p_chi2_within"],
                              detail="each seed's outcomes permuted across L (exact null of Cochran's Q)"))
            tests.append(dict(test="within-seed permutation trend in L (matched seeds)", subset=name,
                              statistic=mt["trend_z_within"], p=mt["p_trend_within"],
                              detail="stratified Cochran-Armitage; z > 0: rises with L"))
            matched[name] = mt

    # logistic regression of runaway on L (per-run indicator)
    logit_rows = []
    for name, Ls, quad in [("coarse L=200..400", TABLE_L, False), ("plateau L=240..400", range(240, 401, 20), False),
                           ("all available", ALL_L, False), ("all available, quadratic", ALL_L, True)]:
        d = present_ok[present_ok.L.isin(list(Ls))]
        if d.L.nunique() < 3:
            logit_rows.append(dict(model=name, n=len(d), note="pending data"))
            continue
        fit = logistic(d.L.to_numpy(float), d.runaway.to_numpy(float), quad)
        b, se = fit["beta"], fit["se"]
        lr = 2 * (fit["ll"] - fit["ll0"])
        row = dict(model=name, n=fit["n"], n_sizes=d.L.nunique(), intercept_at_L300=b[0],
                   slope_per_100=b[1], slope_se=se[1], slope_z=b[1] / se[1],
                   p_wald_slope=2 * stats.norm.sf(abs(b[1] / se[1])),
                   odds_ratio_per_20=np.exp(0.2 * b[1]),
                   or20_lo=np.exp(0.2 * (b[1] - 1.959964 * se[1])), or20_hi=np.exp(0.2 * (b[1] + 1.959964 * se[1])),
                   LR_chi2=lr, p_LR=stats.chi2.sf(lr, len(b) - 1))
        if quad:
            lin = logistic(d.L.to_numpy(float), d.runaway.to_numpy(float), False)
            lr2 = 2 * (fit["ll"] - lin["ll"])
            row.update(quad_per_100sq=b[2], quad_se=se[2], p_LR_quadratic_term=stats.chi2.sf(lr2, 1))
        logit_rows.append(row)
    logit = pd.DataFrame(logit_rows)
    logit.to_csv(out / "logistic.csv", index=False)

    # Welch / Mann-Whitney / Cohen's d on final mean sizes (needs complete no-stop runs)
    def size_test(a, b):
        da = pr[(pr.L == a) & pr.status.isin(KNOWN)].final_mean.to_numpy()
        db = pr[(pr.L == b) & pr.status.isin(KNOWN)].final_mean.to_numpy()
        if len(da) < len(SEEDS) or len(db) < len(SEEDS):
            tests.append(dict(test="Welch t / Mann-Whitney U / Cohen's d (final mean size)", subset=f"L={a} vs L={b}",
                              statistic=np.nan, p=np.nan, detail="pending data (needs complete no-stop runs)"))
            return
        t, p = stats.ttest_ind(da, db, equal_var=False)
        u, pu = stats.mannwhitneyu(da, db, alternative="two-sided")
        sp = np.sqrt(((len(da) - 1) * da.var(ddof=1) + (len(db) - 1) * db.var(ddof=1)) / (len(da) + len(db) - 2))
        dd = (da.mean() - db.mean()) / sp
        tests.append(dict(test="Welch t (final mean size)", subset=f"L={a} vs L={b}", statistic=t, p=p,
                          detail=f"means {da.mean():.1f} vs {db.mean():.1f}"))
        tests.append(dict(test="Mann-Whitney U (final mean size)", subset=f"L={a} vs L={b}", statistic=u, p=pu,
                          detail=f"n={len(da)},{len(db)}"))
        tests.append(dict(test="Cohen's d (final mean size)", subset=f"L={a} vs L={b}", statistic=dd, p=np.nan,
                          detail="pooled SD"))

    size_test(300, 320)
    size_test(220, 240)
    tests = pd.DataFrame(tests)
    tests["p_at_perm_floor"] = [("permutation" in str(t)) and np.isfinite(p) and p <= (1 + 1e-9) / (args.perm + 1)
                                for t, p in zip(tests.test, tests.p)]
    tests.to_csv(out / "transition_tests.csv", index=False)

    md.append("## (b) Transition statistics\n")
    md.append("### Runaway fraction per L (onset region and coarse grid)\n")
    rbv = rb.copy()
    md.append(md_table(rbv, floatfmt={"frac": "{:.2f}", "lo": "{:.2f}", "hi": "{:.2f}", "k": "{:.0f}"}))
    if extra:
        md.append("\nExtra-seed check: " + "; ".join(
            f"L={e['L']}: {e['k']}/{e['n']} = {e['frac']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}] ({e['source']})"
            for e in extra) + "\n")
    md.append("### Tests\n")
    md.append(f"Permutation p-values are (hits + 1)/(n_perm + 1) with n_perm = {args.perm:,}; a value at that floor "
              f"is printed as `<= {1 / (args.perm + 1):.1e}`. The matched-seed rows account for the reuse of seeds "
              "0-99 at every L.\n")
    tview = tests.copy()
    tview["p"] = [p_str(p, perm="permutation" in str(t)) if np.isfinite(p) else ""
                  for p, t in zip(tview.p, tview.test)]
    md.append(md_table(tview.fillna(""), floatfmt={"statistic": "{:.3g}", "p_holm": "{:.3g}", "p_asym": "{:.3g}"}))
    md.append("\n### Logistic regression of the runaway indicator on L (L centred at 300, per 100 cells)\n")
    md.append(md_table(logit, default="{:.3g}"))
    md.append("")

    # threshold sensitivity
    thr_rows = []
    defs = [(f"final mean > {t}", f"run_mean{t}") for t in THRESH_MEAN] + [(f"fill > {FILL_THR:g}", f"run_fill{FILL_THR:g}")]
    for L in ALL_L:
        d = present_ok[present_ok.L == L]
        if not len(d):
            continue
        for lab, col in defs:
            v = d[col].to_numpy()
            kn = np.isfinite(v)
            k = int(np.nansum(v))
            n_unknown = int((~kn).sum())
            lo, hi = wilson(k, len(d)) if n_unknown == 0 else (np.nan, np.nan)
            thr_rows.append(dict(L=L, definition=lab, k=k, n=len(d), n_unknown=n_unknown,
                                 frac=k / len(d) if n_unknown == 0 else np.nan, lo=lo, hi=hi))
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(out / "threshold_sensitivity.csv", index=False)
    if len(thr):
        piv = thr.assign(cell=[("pending (%d unknown)" % r.n_unknown) if r.n_unknown else f"{r.k}/{r.n}"
                               for _, r in thr.iterrows()]).pivot(index="L", columns="definition", values="cell")
        piv = piv[[d[0] for d in defs]].reset_index()
        md.append("### Threshold sensitivity of the runaway definition (k/n at the last row)\n")
        md.append(md_table(piv))
        known_all = present[present.status.isin(KNOWN)]
        if len(known_all):
            dis = {lab: int((known_all[col] != known_all["runaway"]).sum()) for lab, col in defs}
            md.append("\nRuns (with the full trajectory) classified differently from the paper rule: "
                      + ", ".join(f"{k}: {v}" for k, v in dis.items()) + f" (of {len(known_all)}).\n")

    # absorbing check
    ab_rows, ab_ex = [], []
    for L in ALL_L:
        d = pr[(pr.L == L) & pr.status.isin(KNOWN)]
        if not len(d):
            continue
        crossed = d[np.isfinite(d.t_cross100)]
        rev = crossed[crossed.final_mean <= RUN]
        dips = crossed[crossed.n_dips_after_cross > 0]
        fcross = d[np.isfinite(d.t_fill)]
        ab_rows.append(dict(L=L, n_full_trajectory=len(d), n_crossed_100=len(crossed),
                            n_final_le_100_after_crossing=len(rev),
                            n_extinct_after_crossing=int((rev.status == "extinct").sum()),
                            n_with_any_later_sample_le_100=len(dips),
                            min_mean_after_crossing=crossed.min_mean_after_cross.min() if len(crossed) else np.nan,
                            n_crossed_fill=len(fcross), n_final_fill_le_thr=int((fcross.fill <= FILL_THR).sum()),
                            n_with_any_later_fill_le_thr=int((fcross.n_fill_dips > 0).sum())))
        for _, x in dips.iterrows():
            ab_ex.append(dict(L=L, seed=int(x.seed), t_cross100=x.t_cross100, n_dips=int(x.n_dips_after_cross),
                              min_mean_after=x.min_mean_after_cross, final_mean=x.final_mean,
                              final_max=x.final_max, status=x.status))
    ab = pd.DataFrame(ab_rows)
    ab.to_csv(out / "absorbing_check.csv", index=False)
    pd.DataFrame(ab_ex).to_csv(out / "absorbing_exceptions.csv", index=False)
    if len(ab):
        md.append("### Is runaway absorbing? (runs with the full trajectory only)\n")
        md.append(md_table(ab, default="{:.4g}"))
        md.append(f"\nRuns with the full trajectory that crossed mean 100: {int(ab.n_crossed_100.sum())} (of "
                  f"{int(ab.n_full_trajectory.sum())}); of these, {int(ab.n_final_le_100_after_crossing.sum())} "
                  f"ended <= 100 and {len(ab_ex)} dipped to <= 100 at any later sample "
                  "(listed in `absorbing_exceptions.csv`).\n")

    # survival-style P(runaway | lifetime max component >= s)
    sv = []
    for L in ALL_L:
        d = present_ok[present_ok.L == L]
        if not len(d):
            continue
        for s in SURV_S:
            reach = d[d.lifetime_max_size >= s]
            k = int(np.nansum(reach.runaway))
            lo, hi = wilson(k, len(reach))
            lo2, hi2 = wilson(len(reach), len(d))
            sv.append(dict(L=L, s=s, n=len(d), n_reach=len(reach), p_reach=len(reach) / len(d), p_reach_lo=lo2,
                           p_reach_hi=hi2, k_runaway_given_reach=k,
                           p_runaway_given_reach=k / len(reach) if len(reach) else np.nan, lo=lo, hi=hi))
    sv = pd.DataFrame(sv)
    sv.to_csv(out / "survival_given_maxsize.csv", index=False)
    if len(sv):
        md.append("### Survival-style table: P(runaway | largest component ever >= s)\n")
        pv_ = sv.assign(cell=[f"{int(r.k_runaway_given_reach)}/{int(r.n_reach)} = {r.p_runaway_given_reach:.2f}"
                              if r.n_reach else "0/0" for _, r in sv.iterrows()]
                        ).pivot(index="L", columns="s", values="cell").reset_index()
        pv_.columns = ["L"] + [f"s>={c}" for c in pv_.columns[1:]]
        md.append(md_table(pv_))
        md.append("")

    # crossing (nucleation) times
    ct_rows, km_rows = [], []
    grid = np.arange(0, T_END + 1, 100)
    for L in ALL_L:
        d = present_ok[present_ok.L == L]
        if not len(d):
            continue
        tt = np.where(np.isfinite(d.t_cross100), d.t_cross100, d.final_step.astype(float))
        ev = np.isfinite(d.t_cross100).astype(int)
        inc = km_incidence(tt, ev, grid)
        km_rows.append(pd.DataFrame(dict(L=L, step=grid, frac_nucleated=inc)))
        row = dict(L=L, n=len(d), n_nucleated=int(ev.sum()))
        for th in CROSS_THR:
            c = d[f"t_cross{th}"].dropna().to_numpy()
            row[f"n_cross{th}"] = len(c)
            if len(c):
                row.update({f"t{th}_median": np.median(c), f"t{th}_q25": np.percentile(c, 25),
                            f"t{th}_q75": np.percentile(c, 75), f"t{th}_min": c.min(), f"t{th}_max": c.max()})
                if th == 100 and len(c) >= 3:
                    row["t100_mean"] = c.mean()
                    row["t100_skew"] = stats.skew(c)
        for t in (5000, 10000, 15000, 20000):
            row[f"km_frac_by_{t}"] = inc[grid == t][0]
        ct_rows.append(row)
    ct = pd.DataFrame(ct_rows)
    ct.to_csv(out / "crossing_times.csv", index=False)
    km = pd.concat(km_rows, ignore_index=True) if km_rows else pd.DataFrame()
    km.to_csv(out / "km_nucleation.csv", index=False)
    lr_rows = []
    for name, Ls in [("coarse L=200..400", TABLE_L), ("plateau L=240..400", [L for L in TABLE_L if L >= 240]),
                     ("L=320..400", [L for L in TABLE_L if L >= 320])]:
        d = present_ok[present_ok.L.isin(Ls)]
        if d.L.nunique() < 2:
            continue
        tt = np.where(np.isfinite(d.t_cross100), d.t_cross100, d.final_step.astype(float))
        ev = np.isfinite(d.t_cross100).astype(int)
        lr_ = logrank(tt, ev, d.L.to_numpy(), scores={L: float(L) for L in Ls})
        c = d[np.isfinite(d.t_cross100)]
        grp = [g.t_cross100.to_numpy() for _, g in c.groupby("L") if len(g) > 1]
        kw = stats.kruskal(*grp) if len(grp) >= 2 else SimpleNamespace(statistic=np.nan, pvalue=np.nan)
        sp = stats.spearmanr(c.L, c.t_cross100)
        lr_rows.append(dict(subset=name, n=len(d), n_events=int(ev.sum()), **lr_,
                            kruskal_H_crossed=kw.statistic, p_kruskal=kw.pvalue,
                            spearman_rho_L_vs_t=sp.statistic, p_spearman=sp.pvalue))
    lrdf = pd.DataFrame(lr_rows)
    lrdf.to_csv(out / "crossing_time_tests.csv", index=False)
    if len(ct):
        md.append("### Crossing (nucleation) times: first sample with mean size > 100\n")
        md.append(md_table(ct[["L", "n", "n_nucleated", "t100_median", "t100_q25", "t100_q75", "t100_min",
                               "t100_max", "t100_skew", "t10_median", "t1000_median", "km_frac_by_5000",
                               "km_frac_by_10000", "km_frac_by_15000", "km_frac_by_20000"]],
                           default="{:.4g}"))
        md.append("\nLog-rank tests (censoring at extinction or 20,000; trend z > 0 means faster nucleation "
                  "at larger L), and tests on the crossing times of the runs that crossed:\n")
        md.append(md_table(lrdf, default="{:.3g}"))
        md.append("")

    # seed coupling: seeds 0-99 (and so the initial cell types) are reused at every L, and the early
    # dynamics are identical until a structure meets the boundary
    Lsc = [L for L in TABLE_L if L in avail.index]
    if len(Lsc) >= 2:
        X = Mseed[Lsc].dropna().to_numpy(float)
        S_, K_ = X.shape
        iu = np.triu_indices(K_, 1)
        adj_i = np.arange(K_ - 1)
        big = np.array([Lsc[i] >= 340 for i in adj_i])
        Pm = phi_matrix(X)
        obs = dict(all_pairs=float(np.nanmean(Pm[iu])), adjacent=float(np.nanmean(Pm[adj_i, adj_i + 1])),
                   adjacent_L340=float(np.nanmean(Pm[adj_i[big], adj_i[big] + 1])) if big.any() else np.nan)
        rr_ = np.random.default_rng([SEED, 2])
        n_null, hits, null_all = 10_000, {k: 0 for k in obs}, []
        for i in range(0, n_null, 1000):
            Pn = phi_matrix(rr_.permuted(np.broadcast_to(X, (1000, S_, K_)).copy(), axis=1))
            with np.errstate(invalid="ignore"):
                st_ = dict(all_pairs=np.nanmean(Pn[:, iu[0], iu[1]], axis=1),
                           adjacent=np.nanmean(Pn[:, adj_i, adj_i + 1], axis=1),
                           adjacent_L340=(np.nanmean(Pn[:, adj_i[big], adj_i[big] + 1], axis=1) if big.any()
                                          else np.full(1000, np.nan)))
            for k in obs:
                hits[k] += int((st_[k] >= obs[k] - 1e-12).sum())
            null_all.append(st_["all_pairs"])
        null_all = np.concatenate(null_all)
        p_null = {k: (hits[k] + 1) / (n_null + 1) for k in obs}
        pair_rows = []
        for i in range(K_):
            for j in range(i + 1, K_):
                a, b = X[:, i], X[:, j]
                n11, n10, n01 = int((a * b).sum()), int((a * (1 - b)).sum()), int(((1 - a) * b).sum())
                n00 = S_ - n11 - n10 - n01
                pair_rows.append(dict(L_a=Lsc[i], L_b=Lsc[j], adjacent=j == i + 1, k_a=int(a.sum()), k_b=int(b.sum()),
                                      both_runaway=n11, expected_both=a.sum() * b.sum() / S_, phi=Pm[i, j],
                                      p_fisher=stats.fisher_exact([[n11, n10], [n01, n00]])[1]))
        cpl = pd.DataFrame(pair_rows)
        cpl.to_csv(out / "seed_coupling.csv", index=False)
        md.append("### Seed coupling across L (seeds 0-99 are reused at every L)\n")
        md.append(f"phi = correlation of the runaway indicators of the same seed at two sizes; null = shuffling the "
                  f"seeds independently at each size ({n_null:,} shuffles; one-sided p). Mean phi: all pairs "
                  f"{obs['all_pairs']:.3f} (p = {p_str(p_null['all_pairs'], True)}), adjacent pairs "
                  f"{obs['adjacent']:.3f} (p = {p_str(p_null['adjacent'], True)}), adjacent pairs with L >= 340 "
                  f"{obs['adjacent_L340']:.3f} (p = {p_str(p_null['adjacent_L340'], True)}). Adjacent pairs:\n")
        md.append(md_table(cpl[cpl.adjacent].drop(columns="adjacent"),
                           floatfmt={"expected_both": "{:.1f}", "phi": "{:.3f}", "p_fisher": "{:.3g}"},
                           default="{:.0f}"))
        md.append("\nThe matched-seed tests in the test table (Cochran's Q, within-seed permutation) and the "
                  "seed-cluster bootstrap intervals of the pooled fractions account for this coupling.\n")
        ext_same = pr[(pr.status == "extinct")].groupby("seed").agg(nL=("L", "nunique"), steps=("final_step", "nunique"))
        common = ext_same[(ext_same.nL >= 2) & (ext_same.steps == 1)]
        adj = cpl[cpl.adjacent]
        top = adj.sort_values("phi", ascending=False).head(3)

        def mt_txt(name):
            m = matched.get(name)
            return "pending" if m is None else (f"Cochran's Q = {m['cochran_Q']:.1f} (p = {fmt_p(m['p_cochran_Q'])}), "
                                                f"within-seed permutation chi2 p = {p_str(m['p_chi2_within'], True)}, "
                                                f"trend z = {m['trend_z_within']:.2f} (p = "
                                                f"{p_str(m['p_trend_within'], True)})")
        issues.append(
            f"Seeds 0-99 (and therefore the initial cell types) are reused at every L, and the early dynamics do "
            f"not depend on L until a structure meets the boundary (e.g. seed 0 has identical rows at L=200 and 400 "
            f"up to step 100). Seed(s) {', '.join(map(str, common.index.tolist())) or 'none'} go extinct at the "
            f"same step at every L ({', '.join(str(int(x)) for x in common.nL.tolist())} sizes). Runaway outcomes "
            f"of the same seed across L: mean phi over all pairs of L={Lsc[0]}..{Lsc[-1]} = {obs['all_pairs']:.3f} "
            f"(seed-shuffle null mean {np.mean(null_all):.3f}, p = {p_str(p_null['all_pairs'], True)}); between "
            f"neighbouring sizes (mean adjacent phi "
            f"{obs['adjacent']:.3f}, p = {p_str(p_null['adjacent'], True)}; for L >= 340 {obs['adjacent_L340']:.3f}, "
            f"p = {p_str(p_null['adjacent_L340'], True)}; largest: "
            + ", ".join(f"L={int(r.L_a)} vs {int(r.L_b)} phi = {r.phi:.2f}, {int(r.both_runaway)} seeds run away at "
                        f"both vs {r.expected_both:.1f} expected, Fisher p = {fmt_p(r.p_fisher)}"
                        for _, r in top.iterrows())
            + "; `seed_coupling.csv`). The independent-groups tests are therefore complemented by matched-seed "
              f"tests (transition_tests.csv): plateau L=240..400: {mt_txt('plateau L=240..400')}; L=320..400: "
              f"{mt_txt('L=320..400')}.")

    # ---------------- (c) 40k horizon ----------------
    h_rows, h_km = [], []
    for L in H40K_L:
        recs = load_simple(H40K / f"L{L}", range(H40K_SEEDS), L, 40000, STOP)
        if not recs:
            h_rows.append(dict(L=L, n=0, status="pending data", note="pending data"))
            continue
        d = pd.DataFrame(recs)
        tc = d.t_cross100.to_numpy()
        by20, by40 = np.isfinite(tc) & (tc <= 20000), np.isfinite(tc)
        alive20 = (d.status != "extinct") | (d.final_step > 20000)
        at_risk = (~by20) & alive20.to_numpy()
        new = int((by40 & ~by20).sum())
        lo20, hi20 = wilson(by20.sum(), len(d))
        lo40, hi40 = wilson(by40.sum(), len(d))
        clo, chi_ = wilson(new, at_risk.sum())
        # constant-hazard rates in (5k, 20k] and (20k, 40k] per 10^4 steps
        exit_t = np.where(by40, tc, d.final_step.to_numpy(float))
        def rate(a, b):
            expo = np.clip(np.minimum(exit_t, b) - a, 0, None).sum() / 1e4
            ev = int(((tc > a) & (tc <= b)).sum())
            return ev, expo
        e1, x1 = rate(5000, 20000)
        e2, x2 = rate(20000, 40000)
        p_rr = stats.binomtest(e2, e1 + e2, x2 / (x1 + x2)).pvalue if (e1 + e2) else np.nan
        # agreement with the 20k data of the size scan
        m = present[(present.L == L)].set_index("seed")
        mism = 0
        for _, x in d.iterrows():
            if x.seed in m.index:
                a20 = m.loc[x.seed, "t_cross100"]
                b20 = x.t_cross100 if (np.isfinite(x.t_cross100) and x.t_cross100 <= 20000) else np.nan
                if not ((np.isnan(a20) and np.isnan(b20)) or a20 == b20):
                    mism += 1
        h_rows.append(dict(L=L, n=len(d),
                           status="complete" if len(d) == H40K_SEEDS else
                           f"PROVISIONAL ({len(d)}/{H40K_SEEDS} runs; early-stop scan: runaway runs finish first, "
                           "so partial fractions are biased high)",
                           k_by_20k=int(by20.sum()), frac_20k=by20.mean(), lo_20k=lo20, hi_20k=hi20,
                           k_by_40k=int(by40.sum()), frac_40k=by40.mean(), lo_40k=lo40, hi_40k=hi40,
                           new_in_20k_40k=new, at_risk_at_20k=int(at_risk.sum()),
                           p_new_given_at_risk=new / at_risk.sum() if at_risk.sum() else np.nan,
                           p_new_lo=clo, p_new_hi=chi_,
                           events_5k_20k=e1, rate_5k_20k=e1 / x1 if x1 else np.nan,
                           rate_5k_20k_lo=poisson_ci(e1, x1)[0] if x1 else np.nan,
                           rate_5k_20k_hi=poisson_ci(e1, x1)[1] if x1 else np.nan,
                           events_20k_40k=e2, rate_20k_40k=e2 / x2 if x2 else np.nan,
                           rate_20k_40k_lo=poisson_ci(e2, x2)[0] if x2 else np.nan,
                           rate_20k_40k_hi=poisson_ci(e2, x2)[1] if x2 else np.nan,
                           p_rate_ratio=p_rr, mismatch_vs_20k_scan=mism,
                           n_complete_40k=int((d.status == "complete").sum())))
        g = np.arange(0, 40001, 200)
        tt = np.where(np.isfinite(tc), tc, d.final_step.to_numpy(float))
        h_km.append(pd.DataFrame(dict(L=L, step=g, frac_nucleated=km_incidence(tt, np.isfinite(tc).astype(int), g))))
    hz = pd.DataFrame(h_rows)
    hz.to_csv(out / "horizon_40k.csv", index=False)
    md.append("## (c) 40k horizon vs 20k\n")
    if "k_by_20k" in hz:
        md.append("Nucleated = mean size > 100 at some sample. Rates are nucleations per 10^4 steps per run at "
                  "risk (constant-hazard estimate, exact Poisson 95% CI); `p_rate_ratio` tests equal hazards "
                  "in (5k,20k] and (20k,40k]. `mismatch_vs_20k_scan` counts seeds whose crossing by 20k differs "
                  "from the 20k scan (should be 0: same seeds, same engine). `status` PROVISIONAL: the 40k scan "
                  "stops runs at mean size 2,000, so the runaway runs finish first and fractions from a partly "
                  "filled L are biased high.\n")
        md.append(md_table(hz, default="{:.3g}"))
    else:
        md.append("pending data (results/cpu_scan_40k not present)\n")
    md.append("")

    # ---------------- (d) sensitivity grids ----------------
    sens_rows = []
    for gname, entries in SENS_GRID.items():
        for val, lab, dname in entries:
            for L in SENS_L:
                recs = load_simple(SENS / dname / f"L{L}", range(SENS_SEEDS), L, 10000, 0)
                if not recs:
                    sens_rows.append(dict(grid=gname, value=val, label=lab, dir=dname, L=L, n=0,
                                          status="pending data"))
                    continue
                d = pd.DataFrame(recs)
                d = d[d.status.isin(KNOWN)]
                k = int(d.runaway.sum())
                lo, hi = wilson(k, len(d))
                fm = d.final_mean.to_numpy()
                rw = d[d.runaway == 1]
                sens_rows.append(dict(grid=gname, value=val, label=lab, dir=dname, L=L, n=len(d),
                                      status="complete" if len(d) == SENS_SEEDS else
                                      f"PROVISIONAL ({len(d)}/{SENS_SEEDS} runs; no stop: runaway runs finish last, "
                                      "so partial fractions are biased low)", k_runaway=k,
                                      frac=k / len(d), lo=lo, hi=hi, median_final_mean=np.median(fm),
                                      q25_final_mean=np.percentile(fm, 25), q75_final_mean=np.percentile(fm, 75),
                                      mean_final_mean=fm.mean(), median_final_max=np.median(d.final_max),
                                      median_fill_runaway=rw.fill.median() if len(rw) else np.nan,
                                      n_extinct=int((d.status == "extinct").sum()),
                                      median_t_cross100=d.t_cross100.median()))
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out / "sensitivity_summary.csv", index=False)
    md.append("## (d) Mutation / death-probability sensitivity (10k steps, 30 seeds, no stop)\n")
    if (sens.n > 0).any():
        md.append("`status` PROVISIONAL: these runs have no early stop, so the runaway runs finish last and "
                  "fractions from a partly filled condition are biased low.\n")
        md.append(md_table(sens[sens.n > 0].drop(columns=["dir"]), default="{:.3g}"))
    else:
        md.append("pending data (results/cpu_sensitivity not present)\n")
    md.append("")

    # ---------------- (e) fill fraction and the compact regime ----------------
    fe = []
    for L in ALL_L:
        d = pr[(pr.L == L) & pr.status.isin(KNOWN)]
        if not len(d):
            continue
        complete = len(d) == len(SEEDS)
        rw = d[d.runaway == 1]
        nr = d[(d.runaway == 0) & (d.status != "extinct")]
        row = dict(L=L, status="complete" if complete else
                   f"PROVISIONAL ({len(d)}/{len(SEEDS)} runs; runaway runs missing)",
                   n_runaway=len(rw), n_nonrunaway=len(nr), n_extinct=int((d.status == "extinct").sum()))
        if len(rw):
            f = rw.fill.to_numpy()
            W = boot_weights(len(f), args.boot, np.random.default_rng([SEED, L, 3]))
            row.update(fill_median=np.median(f), fill_q25=np.percentile(f, 25), fill_q75=np.percentile(f, 75),
                       fill_mean=f.mean(), fill_mean_lo=tu.pct(W @ f)[0], fill_mean_hi=tu.pct(W @ f)[1],
                       fill_min=f.min(), smax_median=np.median(rw.final_max), meansize_runaway_median=np.median(rw.final_mean))
        if len(nr):
            row.update(nonrun_mean_size_median=nr.final_mean.median(), nonrun_mean_size_q25=nr.final_mean.quantile(.25),
                       nonrun_mean_size_q75=nr.final_mean.quantile(.75), nonrun_mean_size_mean=nr.final_mean.mean(),
                       nonrun_max_size_median=nr.final_max.median(), nonrun_density_median=nr.density.median(),
                       nonrun_density_q25=nr.density.quantile(.25), nonrun_density_q75=nr.density.quantile(.75),
                       nonrun_density_mean=nr.density.mean())
        fe.append(row)
    fe = pd.DataFrame(fe)
    fe.to_csv(out / "fill_nonrunaway.csv", index=False)
    md.append("## (e) Fill fraction of runaway runs and the compact regime at t = 20,000\n")
    md.append("Fill = s_max/L^2 of runs with final mean > 100; compact = non-runaway, non-extinct runs "
              "(density = active cells / L^2). PROVISIONAL rows are computed from the runs present.\n")
    md.append(md_table(fe, default="{:.3g}"))
    md.append("")

    # ---------------- snapshot numbers (captions) ----------------
    def largest(cfg_path: Path):
        """(largest 8-connected component of the non-empty cells, number of components, active cells)."""
        occ = np.load(cfg_path) > 0
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        return (int(np.bincount(lab.ravel())[1:].max()) if n else 0), int(n), int(occ.sum())

    snaps = []
    for L, s, Ts in SNAP_PANELS:
        # the T-step snapshot run is the prefix of the 20k run of the same seed (deterministic engine)
        f = FULL / f"L{L}" / "runs" / f"seed_{s}.csv"
        r = read_run(f) if f.exists() else None
        src = "cpu_scan_full" if r is not None else ""
        # confirmation that the CSV's max_size is the same component definition as the labelling below
        fin_csv = fin_npy = np.nan
        g = FULL / f"L{L}" / "configs" / f"seed_{s}_final.npy"
        if r is not None and g.exists() and r["step"][-1] == T_END:
            fin_csv, fin_npy = float(r["max_size"][-1]), float(largest(g)[0])
        between = np.nan
        if r is not None and r["step"][-1] >= max(Ts):
            w_ = (r["step"] >= min(Ts)) & (r["step"] <= max(Ts))
            between = float(r["max_size"][w_].max())
        for T in Ts:
            row = dict(T=T, L=L, seed=s, source_csv=src, largest_component_csv=np.nan, mean_size_csv=np.nan,
                       n_active_csv=np.nan, n_components_csv=np.nan, max_over_panel_times_csv=np.nan,
                       max_between_panel_times_csv=between, final_t20000_csv=fin_csv, final_t20000_npy=fin_npy,
                       largest_component_npy=np.nan, n_components_npy=np.nan, n_active_npy=np.nan,
                       npy_matches_csv=np.nan, note="")
            if r is not None and r["step"][-1] >= T:
                i = int(np.flatnonzero(r["step"] == T)[0])
                row.update(largest_component_csv=float(r["max_size"][i]), mean_size_csv=float(r["mean_size"][i]),
                           n_active_csv=float(r["n_active"][i]), n_components_csv=float(r["n_components"][i]))
                if r["step"][-1] >= max(Ts):
                    row["max_over_panel_times_csv"] = float(max(r["max_size"][r["step"] == t_][0] for t_ in Ts))
            else:
                row["note"] = "pending data (no CSV row at T)"
            f = SNAP / f"t{T}" / "original_mix" / f"L{L}" / "configs" / f"seed_{s}_final.npy"
            if f.exists():
                big, n_, act = largest(f)
                row.update(largest_component_npy=big, n_components_npy=n_, n_active_npy=act,
                           npy_matches_csv=float(big == row["largest_component_csv"]))
            else:
                row["note"] = (row["note"] + "; " if row["note"] else "") + "snapshot .npy pending (check only)"
            snaps.append(row)
    snaps = pd.DataFrame(snaps)
    snaps.to_csv(out / "snapshot_components.csv", index=False)
    md.append("## Snapshot caption numbers (Fig. 'phenomenon' a,b)\n")
    md.append("Largest 8-connected component at the snapshot times, read from the 20k CSV of the same seed (a "
              "T-step run is the prefix of the 20k run). `max_between` = largest value at any sample between the "
              "first and last snapshot time; `final_*` = t = 20,000 value in the CSV and from labelling the saved "
              "final configuration (same component definition); `*_npy` = labelled snapshot configuration "
              "(results/cpu_snapshots), a confirmation only.\n")
    md.append(md_table(snaps[["L", "seed", "T", "largest_component_csv", "mean_size_csv", "n_active_csv",
                              "max_over_panel_times_csv", "max_between_panel_times_csv", "final_t20000_csv",
                              "final_t20000_npy", "largest_component_npy", "note"]], default="{:,.0f}",
                       floatfmt={"mean_size_csv": "{:.2f}"}))
    md.append("")
    bad_snap = snaps[snaps.npy_matches_csv == 0]
    if len(bad_snap):
        issues.append("snapshot .npy largest component differs from the 20k CSV row at the same step for "
                      + ", ".join(f"L={int(r.L)} seed {int(r.seed)} t={int(r['T'])}" for _, r in bad_snap.iterrows())
                      + " (the snapshot run is not the prefix of the 20k run?)")

    # ---------------- data issues ----------------
    bad = pr[pr.note.fillna("").astype(str).str.len() > 0]
    if len(bad):
        issues.append(f"{len(bad)} no-stop files were unreadable or incomplete when read (being written?): "
                      + ", ".join(f"L{int(r.L)}/seed_{int(r.seed)} ({r.note})" for _, r in bad.head(10).iterrows()))

    # ---------------- figures ----------------
    size_L = set(status.L[status.complete]) | (set(status.L[status.n_nostop > 0]) if args.allow_partial else set())
    figures(out, t2tab, rb, pr, present_ok, km, thr, sv, hz, h_km, sens, size_L)

    md.append("## Data issues\n")
    md.extend(f"- {s}" for s in issues)
    md.append("\n## Files\n")
    md.extend(f"- `{p.name}`" for p in sorted(out.iterdir()) if p.is_file() and p.name != "summary.md")
    (out / "summary.md").write_text(tidy_p("\n".join(md) + "\n"))
    print(f"wrote {out}/summary.md, tables and figures")


# =============================== figures ===============================
def pending(ax, text="pending data"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=C_GREY, fontsize=8)


def logaxis(ax, plain=False):
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if plain:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def kticks(ax, top):
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5000 if top <= 20000 else 10000))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))


def yerr(df, m, lo, hi):
    v = df[m].to_numpy(float)
    return np.vstack([v - df[lo].to_numpy(float), df[hi].to_numpy(float) - v]).clip(min=0)


def plabel(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def figures(out, t2, rb, pr, present, km, thr, sv, hz, h_km, sens, size_L):
    # ---- Fig A: Table 2 overview ----
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    ax = axs[0, 0]
    part = rb[(rb.n > 0) & (rb.n < len(SEEDS))]     # partly filled L: biased towards compact runs
    d = rb[rb.n == len(SEEDS)]
    ax.fill_between(d.L, d.lo, d.hi, color=C_MAIN, alpha=0.15, lw=0)
    ax.plot(d.L, d.frac, "-", color=C_MAIN, lw=1.3, zorder=2)
    ax.plot(d.L, d.frac, "o", color=C_MAIN, ms=4.5, mec="white", mew=0.5, zorder=3)
    if len(part):
        ax.plot(part.L, part.frac, "^", mfc="white", mec=C_MED, ms=4, mew=0.8, zorder=3)
    ax.set_ylim(-0.02, 0.88)
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("Runaway fraction")
    hd = [Line2D([], [], color=C_MAIN, marker="o", mec="white", label=f"{len(SEEDS)} runs per $L$")]
    if len(part):
        hd.append(Line2D([], [], color=C_MED, marker="^", mfc="white", ls="", ms=4,
                         label="partial (provisional, biased low)"))
    ax.legend(handles=hd, loc="upper left", frameon=False, fontsize=6.3)
    sns.despine(ax=ax)
    plabel(ax, "a")

    def strip(ax, col, ylabel, mean_col, lo_col, hi_col, ylim):
        ok = t2[np.isfinite(t2[mean_col])]
        any_pts = any_ext = False
        for L in ok.L:
            d_ = pr[(pr.L == L) & pr.status.isin(("complete", "extinct"))]
            v = d_[col].to_numpy()
            ext = (d_.status == "extinct").to_numpy()
            x = L + np.random.RandomState(int(L)).uniform(-4, 4, size=len(v))
            ax.scatter(x[~ext], np.clip(v[~ext], 1, None), s=5, color=C_MAIN, alpha=0.35, lw=0, zorder=2)
            if ext.any():   # size 0 cannot be drawn on the log axis: own marker at the bottom edge
                ax.scatter(x[ext], np.full(ext.sum(), ylim[0] * 1.12), s=11, marker="x", color=C_MED, lw=0.7,
                           zorder=3, clip_on=False)
                any_ext = True
            any_pts = True
        if any_pts:
            med = [np.median(pr[(pr.L == L) & pr.status.isin(("complete", "extinct"))][col]) for L in ok.L]
            ax.plot(ok.L, np.clip(med, 1, None), "-", color=C_MED, lw=1.3, zorder=4, label="median")
            ax.errorbar(ok.L.to_numpy(), ok[mean_col].to_numpy(), yerr=yerr(ok, mean_col, lo_col, hi_col),
                        fmt="D", color=C_ALT, ms=3.5, capsize=2, lw=1, zorder=5, label="mean [95% CI]")
            h, lb = ax.get_legend_handles_labels()
            if any_ext:
                h.append(Line2D([], [], color=C_MED, marker="x", ls="", ms=4, mew=0.7))
                lb.append("extinct (size 0)")
            ax.legend(h, lb, loc="center left", frameon=False, fontsize=6.3)
        else:
            pending(ax)
        ax.set_yscale("log")
        logaxis(ax)
        ax.set_ylim(*ylim)
        ax.set_xlim(190, 410)
        ax.set_xlabel("Space size $L$")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)

    strip(axs[0, 1], "final_mean", "Final mean size", "mean_size", "mean_size_lo", "mean_size_hi", (0.8, 2e4))
    plabel(axs[0, 1], "b")
    strip(axs[0, 2], "final_max", "Final max size", "max_size", "max_size_lo", "max_size_hi", (0.8, 3e5))
    plabel(axs[0, 2], "c")

    ax = axs[1, 0]
    ok = t2[np.isfinite(t2.late_early)]
    if len(ok):
        ax.errorbar(ok.L.to_numpy(), ok.late_early.to_numpy(), yerr=yerr(ok, "late_early", "late_early_lo", "late_early_hi"),
                    fmt="o-", color=C_MAIN, ms=4, capsize=2, lw=1.2, mec="white", mew=0.4, label="all runs")
        okn = ok[np.isfinite(ok.late_early_nonrunaway)]
        ax.plot(okn.L, okn.late_early_nonrunaway, "s--", color=C_GREY, ms=3, lw=1,
                label="compact runs only\n(non-runaway, not extinct)")
        ax.set_yscale("log")
        logaxis(ax, plain=True)
        ax.axhline(1, color=C_GREY, lw=0.7, ls=":")
        ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    else:
        pending(ax)
    ax.set_xlim(190, 410)
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("Late/Early mean size")
    sns.despine(ax=ax)
    plabel(ax, "d")

    ax = axs[1, 1]
    ok = t2[np.isfinite(t2.r)]
    if len(ok):
        # ensemble r (paper) and the detrended per-run r by the run's state during the 2k-10k window
        for col, c, mk, ls, lab, off in [
                ("r", C_MAIN, "o", "-", "ensemble (paper)", -7.5),
                ("r_detr", C_MED, "s", "--", "per run, detrended: all", -2.5),
                ("r_detr_compact", C_ALT, "D", "-.", "  compact in window", 2.5),
                ("r_detr_runaway", C_GREY, "^", ":", "  runaway in window", 7.5)]:
            okc = ok[np.isfinite(ok[col])]
            if not len(okc):
                continue
            ax.errorbar(okc.L.to_numpy() + off, okc[col].to_numpy(), yerr=yerr(okc, col, col + "_lo", col + "_hi"),
                        fmt=mk, ls=ls, color=c, ms=3.5, capsize=1.5, lw=1.1, mec="white", mew=0.3, label=lab)
        ax.legend(loc="upper left", frameon=False, fontsize=6.0)
    else:
        pending(ax)
    ax.axhline(0, color=C_GREY, lw=0.7, ls=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(190, 410)
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel(r"Size-score correlation $r$ (2k-10k)")
    sns.despine(ax=ax)
    plabel(ax, "e")

    ax = axs[1, 2]
    rw = pr[(pr.runaway == 1) & pr.status.isin(("complete", "extinct")) & pr.L.isin(size_L) & pr.L.isin(TABLE_L)]
    if len(rw):
        x = rw.L + np.random.RandomState(1).uniform(-4, 4, size=len(rw))
        ax.scatter(x, rw.fill, s=6, color=C_MAIN, alpha=0.45, lw=0)
        g = rw.groupby("L").fill.median()
        ax.plot(g.index, g.values, "-", color=C_MED, lw=1.3, label="median")
        ax.legend(loc="lower right", frameon=False, fontsize=6.3)
        ax.set_ylim(0, 0.85)
    else:
        pending(ax)
    ax.set_xlim(190, 410)
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel(r"Fill $s_{\max}/L^2$ (runaway runs)")
    sns.despine(ax=ax)
    plabel(ax, "f")
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_table2_overview.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)

    # ---- Fig B: nucleation, thresholds, survival ----
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.66))
    ax = axs[0, 0]
    Ls = [L for L in TABLE_L[::2] if len(km) and (km.L == L).any()]
    cols = ordered_colors(len(Ls))
    for i, (c, L) in enumerate(zip(cols, Ls)):
        k = km[km.L == L]
        ax.plot(k.step, k.frac_nucleated, color=c, lw=1.3, ls=LSTYLES[i % len(LSTYLES)], label=f"{L}")
    if Ls:
        ax.legend(title="$L$", ncol=2, frameon=False, fontsize=6, title_fontsize=6.5, loc="upper left")
    ax.set_xlim(0, T_END)
    kticks(ax, T_END)
    ax.set_ylim(-0.01, 0.55)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Fraction nucleated")
    ax.set_title("Kaplan-Meier, mean size > 100", fontsize=7.5)
    sns.despine(ax=ax)
    plabel(ax, "a", dx=-0.12)

    ax = axs[0, 1]
    c = present[np.isfinite(present.t_cross100) & present.L.isin(ALL_L)]
    if len(c):
        Ls_c = sorted(c.L.unique())
        data = [c[c.L == L].t_cross100.to_numpy() for L in Ls_c]
        bp = ax.boxplot(data, positions=range(len(Ls_c)), widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color=C_MED, lw=1.2))
        for b in bp["boxes"]:
            b.set(facecolor=(*C_MAIN, 0.2), edgecolor=C_MAIN, lw=0.8)
        for i, v in enumerate(data):
            ax.scatter(i + np.random.RandomState(i).uniform(-0.2, 0.2, len(v)), v, s=4, color=C_MAIN, alpha=0.5, lw=0)
        ax.set_xticks(range(len(Ls_c)))
        ax.set_xticklabels([str(L) for L in Ls_c], rotation=60)
        ax.set_ylim(0, T_END)
    else:
        pending(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("Nucleation step")
    ax.set_title("Runs that nucleated by t = 20,000", fontsize=7.5)
    sns.despine(ax=ax)
    plabel(ax, "b", dx=-0.12)

    ax = axs[1, 0]
    if len(thr):
        defs = list(dict.fromkeys(thr.definition))
        pal = [PAL[j] for j in (0, 1, 2, 4, 7, 5, 9)][:len(defs)]
        for i, (dname, col) in enumerate(zip(defs, pal)):
            t = thr[(thr.definition == dname)].sort_values("L")
            ax.plot(t.L.to_numpy() + (i - 2) * 1.2, t.frac.to_numpy(), marker="os^vDph"[i % 7], ms=3.5, lw=1.1,
                    ls=LSTYLES[i % len(LSTYLES)], color=col, label=dname)
        ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    else:
        pending(ax)
    ax.set_ylim(-0.02, 0.6)
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("Runaway fraction")
    ax.set_title("Definition of runaway", fontsize=7.5)
    sns.despine(ax=ax)
    plabel(ax, "c", dx=-0.12)

    ax = axs[1, 1]
    if len(sv):
        pal = ordered_colors(len(SURV_S))
        for i, (col, s) in enumerate(zip(pal, SURV_S)):
            t = sv[(sv.s == s) & (sv.n_reach > 0)].sort_values("L")
            ax.plot(t.L, t.p_runaway_given_reach, marker="os^vD"[i % 5], ms=3, lw=1.1, ls=LSTYLES[i % len(LSTYLES)],
                    color=col, label=f"$s\\geq{s}$")
        t = sv[sv.s == 100].sort_values("L")
        ax.fill_between(t.L, t.lo, t.hi, color=pal[2], alpha=0.12, lw=0)
        ax.legend(frameon=False, fontsize=6.2, loc="upper left", title="largest comp. ever", title_fontsize=6.3)
        ax.set_title(r"Band: 95% CI for $s\geq100$", fontsize=7.5)
        ax.set_ylim(-0.02, 1.02)
    else:
        pending(ax)
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("P(runaway | largest $\\geq s$)")
    sns.despine(ax=ax)
    plabel(ax, "d", dx=-0.12)
    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_nucleation_thresholds.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)

    # ---- Fig C: 40k horizon ----
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.5, DOUBLE_COL * 0.34))
    if h_km:
        hk = pd.concat(h_km)
        pal = ordered_colors(len(H40K_L))
        for i, (col, L) in enumerate(zip(pal, H40K_L)):
            k = hk[hk.L == L]
            if len(k):
                nL = int(hz.loc[hz.L == L, "n"].iloc[0]) if (hz.L == L).any() else 0
                ax.plot(k.step, k.frac_nucleated, color=col, lw=1.4, ls=LSTYLES[i % len(LSTYLES)],
                        label=f"$L={L}$" + ("" if nL == H40K_SEEDS else f" (PROVISIONAL n={nL})"))
        ax.axvline(20000, color=C_GREY, lw=0.8, ls=":")
        ax.legend(frameon=False, loc="upper left")
    else:
        pending(ax, "pending data (cpu_scan_40k)")
    ax.set_xlim(0, 40000)
    kticks(ax, 40000)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Fraction nucleated")
    sns.despine(ax=ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_horizon_40k.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)

    # ---- Fig D: sensitivity heatmaps (colour-blind-safe sequential maps with colour bars) ----
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    for j, (gname, ylabel) in enumerate([("mu", r"Mutation rate $\mu$"), ("p_death", r"Death prob. $p_{\rm death}$")]):
        g = sens[(sens.grid == gname) & (sens.n > 0) & ~sens.label.str.contains(r"/0\.999")] if len(sens) else sens
        for i, (col, cmap, vmin, vmax, lab) in enumerate([
                ("frac", "mako_r", 0, 1, "Runaway fraction (t = 10,000)"),
                ("mean_final_mean", "rocket_r", 0, 4, r"$\log_{10}$ mean final mean size")]):
            ax = axs[i, j]
            if not len(g):
                pending(ax, "pending data (cpu_sensitivity)")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            P = g.pivot(index="value", columns="L", values=col).sort_index(ascending=False)
            N = g.pivot(index="value", columns="L", values="n").reindex(index=P.index, columns=P.columns)
            Z = np.log10(np.clip(P, 1, None)) if col != "frac" else P
            ann = P.map(lambda v: "" if not np.isfinite(v) else (f"{v:.2f}" if col == "frac" else
                                                                 (f"{v:.1f}" if v < 100 else f"{v / 1000:.1f}k")))
            ann = ann + N.map(lambda v: "" if not np.isfinite(v) or v == SENS_SEEDS else f"\n(n={int(v)}*)")
            sns.heatmap(Z, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=ann, fmt="", annot_kws={"size": 6.3},
                        linewidths=1, linecolor="white", cbar_kws={"label": lab},
                        yticklabels=[f"{v:g}" for v in P.index])
            ax.set_xlabel("Space size $L$")
            ax.set_ylabel(ylabel)
            ax.tick_params(left=False, bottom=False)
            plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    if len(sens) and ((sens.n > 0) & (sens.n < SENS_SEEDS)).any():
        fig.subplots_adjust(bottom=0.12)
        fig.text(0.01, 0.01, f"* PROVISIONAL: fewer than {SENS_SEEDS} finished runs; these runs have no early stop, "
                             "so runaway runs finish last and are under-represented.", fontsize=6, color=C_MED)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_sensitivity.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
