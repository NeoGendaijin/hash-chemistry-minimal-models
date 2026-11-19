#!/usr/bin/env python3
"""Supplementary observables, matched-seed checks and the rows of the supplementary tables."""
from __future__ import annotations

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/npj_rev/supplement"


KFMT = FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def spatial_observables():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.2), layout="constrained")
    metrics = [("occupancy", r"Occupancy $N_{\mathrm{active}}/L^2$"),
               ("dominant_share", r"Largest-component share $s_{\max}/N_{\mathrm{active}}$"),
               ("n_components", "Number of components")]
    colors = {200: "#0072B2", 240: "#D55E00", 400: "#009E73"}
    grid = np.arange(0, 20001, 10)
    rows = []
    for L in (200, 240, 400):
        groups = {False: [], True: []}
        for seed in range(100):
            path = ROOT / f"results/cpu_scan_full/original_mix/L{L}/runs/seed_{seed}.csv"
            d = pd.read_csv(path)
            runaway = bool(d.mean_size.iloc[-1] > 100)
            x = d.set_index("step").reindex(grid).fillna(0)
            x["occupancy"] = x.n_active / L**2
            x["dominant_share"] = np.divide(x.max_size, x.n_active,
                                             out=np.zeros(len(x)), where=x.n_active > 0)
            groups[runaway].append(x[[m for m, _ in metrics]].to_numpy())
            rows.append(dict(L=L, seed=seed, runaway=runaway,
                             occupancy=x.occupancy.iloc[-1], dominant_share=x.dominant_share.iloc[-1],
                             n_components=x.n_components.iloc[-1]))
        for runaway, members in groups.items():
            arr = np.stack(members)
            low, med, high = np.percentile(arr, [25, 50, 75], axis=0)
            label = f"$L$ = {L}, {'runaway' if runaway else 'compact or extinct'} ($n$ = {len(arr)})"
            style = "-" if runaway else "--"
            for k, ax in enumerate(axs):
                ax.plot(grid, med[:, k], color=colors[L], ls=style, lw=1.2, label=label)
                ax.fill_between(grid, low[:, k], high[:, k], color=colors[L], alpha=0.07)
    for k, (ax, (_, label)) in enumerate(zip(axs, metrics)):
        ax.set(xlabel="Time (steps)", ylabel=label, title=f"({chr(97+k)})")
        ax.xaxis.set_major_formatter(KFMT)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    save(fig, "spatial_observables")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "spatial_observables.csv", index=False)
    return out


def novelty_rate():
    rates = []
    bins = np.arange(0, 20001, 1000)
    for seed in range(40):
        d = pd.read_csv(ROOT / f"results/cpu_novelty/original_mix/L200/runs/seed_{seed}.csv")
        s = d.set_index("step").cum_pattern_types.reindex(bins).ffill()
        for t, rate in zip(bins[1:], np.diff(s.to_numpy()) / 1000):
            rates.append(dict(seed=seed, window_end=t, novel_patterns_per_step=rate))
    out = pd.DataFrame(rates)
    out.to_csv(OUT / "novelty_rate.csv", index=False)
    q = out.groupby("window_end").novel_patterns_per_step.quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(figsize=(5.4, 2.8), layout="constrained")
    ax.plot(q.index, q[0.5], color="#0072B2", lw=1.4)
    ax.fill_between(q.index, q[0.25], q[0.75], color="#0072B2", alpha=0.18)
    ax.set(xlabel="Time (steps), end of 1,000-step window", ylabel="New component-pattern\ntypes per step")
    ax.xaxis.set_major_formatter(KFMT)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "novelty_rate")
    return out


def matched_seed_checks():
    d = pd.read_csv(ROOT / "results/npj_rev/ext1/per_run.csv")
    d = d[d.rule == "intended"].copy()
    records = []
    for index, col in enumerate(("B_run", "C_run")):
        rng = np.random.default_rng(20261103 + index)
        ok = d[d.used]
        a = ok[ok.P == 0][["D", "seed", col]]
        b = ok[ok.P == 1][["D", "seed", col]]
        pairs = a.merge(b, on=["D", "seed"], suffixes=("_0", "_1"))
        pairs["difference"] = pairs[f"{col}_1"] - pairs[f"{col}_0"]
        diff = pairs.groupby("seed").difference.mean().to_numpy()
        boots = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        records.append(dict(comparison="P=1 minus P=0, matched D and seed; averaged within seed",
                            measure=col, n=len(diff), mean_difference=diff.mean(), ci_lo=lo, ci_hi=hi,
                            p=stats.wilcoxon(diff).pvalue, test="paired Wilcoxon across seed blocks",
                            bootstrap_seed=20261103 + index))
    out = pd.DataFrame(records)
    out.to_csv(OUT / "matched_seed_checks.csv", index=False)
    return out


def seed_counts(d, mask, col):
    """Number of runs with `col` true per seed (seeds 0-29) among the runs selected by `mask`."""
    return d[mask].groupby("seed")[col].sum().reindex(range(30), fill_value=0).to_numpy()


def success_criterion_checks():
    """Seed-level comparisons of the number of successful runs (A) under the criterion of the paper
    and under criteria based on the number of active cells in the last recorded population (after
    1,999 steps) instead of the log size."""
    d = pd.read_csv(ROOT / "results/npj_rev/ext1/per_run.csv")
    d = d[d.rule == "intended"].copy()
    explosive = d.explosive.astype(bool)
    criteria = [("paper", d.used.astype(bool))]
    criteria += [(f"active>={t}", (d.n_active_final >= t) & ~explosive) for t in (50, 100, 200, 500, 1000)]
    low = d.P <= 0.4
    records = []
    for index, (name, ok) in enumerate(criteria):
        d["ok"] = ok
        row = dict(criterion=name)
        for P in sorted(d.P.unique()):
            row[f"n_P{P:.1f}"] = int(d[d.P == P].ok.sum())
        contrasts = (("dyadicity", d.P == 1, d.P == 0, 20261118), ("locality", low & (d.D <= 50), low & (d.D >= 60), 20261218))
        for label, first, second, seed in contrasts:
            a, b = seed_counts(d, first, "ok"), seed_counts(d, second, "ok")
            diff = (a - b).astype(float)
            rng = np.random.default_rng(seed + index)
            boots = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row.update({f"{label}_first": int(a.sum()), f"{label}_second": int(b.sum()),
                        f"{label}_seeds_higher": int((diff > 0).sum()), f"{label}_seeds_lower": int((diff < 0).sum()),
                        f"{label}_mean_difference": diff.mean(), f"{label}_ci_lo": lo, f"{label}_ci_hi": hi,
                        f"{label}_p": stats.wilcoxon(diff).pvalue, f"{label}_bootstrap_seed": seed + index})
        records.append(row)
    out = pd.DataFrame(records)
    out.to_csv(OUT / "success_criterion_checks.csv", index=False)
    return out


def active_cells_by_P():
    d = pd.read_csv(ROOT / "results/npj_rev/ext1/per_run.csv")
    return d[d.rule == "intended"].groupby("P").n_active_final.median().to_dict()


def tex_int(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def tex_p(p):
    """Two significant digits; powers of ten below 0.001."""
    if p >= 0.001:
        return f"${p:.{1 - int(np.floor(np.log10(p)))}f}$"
    exponent = int(np.floor(np.log10(p)))
    return f"${p / 10**exponent:.1f}\\times10^{{{exponent}}}$"


def supplementary_tables(success):
    """Rows of the supplementary tables, written as LaTeX for the Supplementary Information."""
    c = pd.read_csv(ROOT / "results/npj_rev/ext1/conditions.csv")
    c = c[c.rule == "intended"].sort_values(["D", "P"])
    rows = [f"{int(r.D)} & {r.P:.1f} & {int(r.A)} & [{r.A_wilson_lo:.1f}, {r.A_wilson_hi:.1f}] & {r.B:.2f} & "
            f"[{r.B_lo:.2f}, {r.B_hi:.2f}] & {tex_int(r.C)} & [{tex_int(r.C_lo)}, {tex_int(r.C_hi)}] \\\\"
            for r in c.itertuples()]
    (OUT / "table_conditions.tex").write_text("\n".join(rows) + "\n")

    labels = {"paper": "Criterion of the main text"}
    rows = []
    count_columns = [name for name in success.columns if name.startswith("n_P")]
    for r in success.to_dict("records"):
        criterion = r["criterion"]
        name = labels.get(criterion) or f"At least {tex_int(float(criterion.split('>=')[1]))} active cells"
        counts = " & ".join(str(r[column]) for column in count_columns)
        rows.append(f"{name} & {counts} & {r['dyadicity_seeds_higher']}/{r['dyadicity_seeds_lower']} & "
                    f"{tex_p(r['dyadicity_p'])} & {r['locality_first']}/{r['locality_second']} & "
                    f"{r['locality_seeds_higher']}/{r['locality_seeds_lower']} & {tex_p(r['locality_p'])} \\\\")
    (OUT / "table_success_criterion.tex").write_text("\n".join(rows) + "\n")

    s = pd.read_csv(ROOT / "results/npj_rev/ext2_tables/survival_given_maxsize.csv")
    s = s[s.s == 100].sort_values("L")
    rows = []
    for r in s.itertuples():
        if r.n_reach:
            outcome = f"{int(r.k_runaway_given_reach)} & {r.p_runaway_given_reach:.2f} & [{r.lo:.2f}, {r.hi:.2f}]"
        else:
            outcome = "-- & -- & --"
        rows.append(f"{int(r.L)} & {int(r.n)} & {int(r.n_reach)} & {outcome} \\\\")
    (OUT / "table_conditional_outcomes.tex").write_text("\n".join(rows) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    spatial = spatial_observables()
    novelty = novelty_rate()
    paired = matched_seed_checks()
    success = success_criterion_checks()
    supplementary_tables(success)
    lines = ["# Supplementary checks", "",
             "## Final spatial observables (median)", "",
             spatial.groupby(["L", "runaway"])[["occupancy", "dominant_share", "n_components"]]
                    .median().to_string(), "", "## Late novelty rate", ""]
    late = novelty[novelty.window_end > 15000].groupby("seed").novel_patterns_per_step.mean()
    lines += [f"L=200, steps 15,001-20,000, 40 runs: median {late.median():.3f} new sampled pattern "
              f"types/step, IQR {late.quantile(.25):.3f}-{late.quantile(.75):.3f}.",
              "Counts refer to sampled components, not exclusively replicated winners.", "",
              "## Matched-seed sensitivity checks", "", paired.to_string(index=False), "",
              "These exploratory contrasts preserve seed dependence across conditions. B/C comparisons "
              "are conditional on success in both matched conditions. Their intervals resample seeds, "
              "not individual contests or time points. Each observable has its own fixed bootstrap seed. "
              "These checks address the effect of dyadicity, not equivalence of implementations.", "",
              "## Number of successful runs under different criteria (seed-level comparisons)", "",
              success.to_string(index=False), "",
              "dyadicity: successful runs per seed over the 10 values of D, P=1 minus P=0. locality: successful runs "
              "per seed over the 15 conditions with P<=0.4, D<=50 minus D>=60. p values are paired Wilcoxon tests "
              "across the 30 seeds; intervals are percentile bootstrap intervals of the mean difference over seeds.", "",
              "Median number of active cells in the last recorded population by P (all runs with the intended rule): "
              + ", ".join(f"P={P:g}: {m:g}" for P, m in active_cells_by_P().items()) + "."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
