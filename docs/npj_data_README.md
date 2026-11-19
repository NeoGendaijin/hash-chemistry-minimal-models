# Data for "Hash Chemistry: Minimal Models for Evolutionary Growth of Complexity"

This archive accompanies the paper by I. Horiguchi and H. Sayama. It contains the outputs of
the simulations of both extensions, the runs used for validation and the derived statistics.
It belongs to the code release `npj-complexity-rev1`.

Extract it at the root of the code:

```bash
tar -xzf npj-complexity-data.tar.gz
sha256sum -c results/MANIFEST_npj.sha256
```

`results/MANIFEST_npj.sha256` lists every file of the archive with its SHA-256 hash.
`results/PROVENANCE_npj.json` records the commit and the hashes of the source files at
packaging, the versions of the analysis packages, the simulation settings, the seeds and an
inventory of the datasets.

## Datasets

All paths are below `results/`. The directory names are identifiers only.

| Directory | Content | Seeds / steps | Use in the paper |
|---|---|---|---|
| `jax/spatial_dyadic/intended` | JAX; 60 (D, P) conditions at L=100, intended dyadic rule | 0-29 / 2,000 | Extension I, Figs. 5-7 |
| `jax/spatial_dyadic/as_coded` | JAX; the 50 conditions with P>0, as-coded dyadic rule | 0-29 / 2,000 | comparison with the earlier Mathematica runs |
| `jax/spatial_dyadic/stats` | statistics of that comparison | derived | Methods |
| `jax/spatial_dyadic_snapshots/intended` | configurations of four conditions at 1,500, 1,750 and 2,000 steps | 0-7 | Fig. 7 |
| `jax/original_comparison/runs` | JAX; the setting of the original SCHC study (L=100) | 0-99 / 2,000 | comparison with the original study |
| `jax/original_comparison/analysis` | statistics of that comparison | derived | Methods |
| `jax/transition_check/L100` | JAX; L=100 | 0-49 / 20,000 | comparison with the C implementation |
| `cpu_scan_check/original_mix/L100` | C; L=100 | 0-199 / 20,000 | comparison with the JAX implementation |
| `jax/adaptive_validation.json` | record of the test of batch splitting | | Methods |
| `cpu_scan_full/original_mix` | C; size scan at 19 space sizes | 0-99 / 20,000 | Extension II, Table 2, Figs. 8-9 |
| `cpu_novelty/original_mix/L200` | C; all metrics recorded at every step | 0-39 / 20,000 | Fig. 8e, Supplementary Fig. S2 |
| `cpu_sensitivity` | C; mutation and death probabilities at six space sizes | 0-29 / 10,000 | Fig. 9c,d |
| `cpu_scan_40k/original_mix` | C; L=240, 320 and 400; a run stops when its mean component size reaches 2,000 | 0-99 / up to 40,000 | Supplementary Fig. S3 |
| `cpu_snapshots` | C; L=200 seed 0 and L=400 seed 5, rerun up to the times shown | stated seeds | Fig. 8a,b |
| `npj_rev` | tables, statistics and test results derived from the datasets above | derived | all results |

The 19 space sizes of the size scan are 100, 120, 140, 160, 180, 200, 210, 220, 230, 240, 250,
260, 280, 300, 320, 340, 360, 380 and 400. Unless a directory name says otherwise, the runs
use k=1000, n=10, mu=0.002/0.999 and death_prob=0.001. In `cpu_sensitivity`, the suffixes
`_mu...` and `_pd...` give the mutation and death probabilities; the directory without a
suffix holds the runs with the default values, and `original_mix_mu0.002` holds additional
runs with mu exactly 0.002.

## Files and columns

Runs of the C implementation (`runs/seed_{s}.csv`) have the columns step, max_fitness,
mean_fitness, max_size, mean_size, cum_cell_types, cum_pattern_types, n_active and
n_components.

- max_fitness and mean_fitness are hash scores, not realized fitness. Scores and sizes are
  measured over the 8-connected components present at the recorded step.
- cum_pattern_types counts the distinct patterns among the components present at the
  recorded steps, whether or not they replicated. Patterns are identified by a 64-bit
  identifier (two 32-bit hashes) of the shape and cell types, up to translation.
- Rows are recorded every ten steps, except in `cpu_novelty` (every step). A run that goes
  extinct ends at extinction; the analysis counts its sizes as zero afterwards.
- `configs/` and `cpu_snapshots` hold configurations as integer arrays; zero is an empty
  cell and positive entries are cell types.

Runs of Extension I (`D{D}_P{P}/seed_{s}.csv`) record the replication events of each step:
max_fitness, mean_fitness, their distances from 1 (max_fitness_gap, mean_fitness_gap),
num_replicated_cells, max_rep_size, mean_rep_size, cum_cell_types, cum_pattern_types,
n_events, n_active and log_bytes_cum.

- Row t records the events of step t+1; n_active is the number of active cells before that
  step. The last row, numbered 1999, therefore holds the cumulative counts after 2,000
  steps.
- cum_pattern_types counts the patterns of the winners of replication events. It differs
  from the count of Extension II, which includes every component present.
- log_bytes_cum is the size that the log of replication events would have as a text file
  in the format of the original Mathematica program; the success criterion of the paper
  uses it.

The per-run data of the earlier Mathematica runs of the spatial-dyadic SCHC are not part of
this archive; they are available from H. Sayama on reasonable request. The statistics derived
from them are in `jax/spatial_dyadic/stats` and in the Mathematica rows of
`npj_rev/ext1/regression.csv` and `joint_tests.csv`. Without that file, `npj_rev_ext1.py`
writes these two tables without those rows. The data of the original SCHC study are public
at https://github.com/hsayama/Structural-Cellular-Hash-Chemistry.

## Reproduction

```bash
uv run --no-sync python scripts/npj_rev_ext1.py
uv run --no-sync python scripts/ext1_novelty_per_event.py
uv run --no-sync python scripts/npj_rev_ext2_tables.py
uv run --no-sync python scripts/npj_rev_ext2_figs.py
uv run --no-sync python scripts/npj_revision_schematics.py
uv run --no-sync python scripts/npj_revision_supplement.py
JAX_PLATFORMS=cpu uv run --no-sync python scripts/validate_model_rules.py
uv run --no-sync python scripts/validate_cpu_engine.py
```

The outputs are written to `results/npj_rev/` (ext1, ext2_tables, ext2fig, schematics,
supplement, validation).
