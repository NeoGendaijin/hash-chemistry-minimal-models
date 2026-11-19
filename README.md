# Hash Chemistry: Minimal Models for Evolutionary Growth of Complexity

Code for the paper by I. Horiguchi and H. Sayama, *Hash Chemistry: Minimal Models for
Evolutionary Growth of Complexity*. The repository contains

- a JAX implementation of Structural Cellular Hash Chemistry (SCHC) with its spatial-dyadic
  extension (`src/simulation.py`), used for Extension I of the paper;
- a C implementation of SCHC (`src/schc_fast.c`, called through `src/fast.py`), used for
  Extension II;
- the scripts that ran the simulations, and the scripts that produce the figures, tables
  and statistics of the paper.

The version used in the paper is the release `npj-complexity-rev1`. The data of the paper are
attached to that release as `npj-complexity-data.tar.gz` and are described in
[the data guide](docs/npj_data_README.md).

## Model

SCHC runs on an L x L grid with open boundaries. A cell is empty or carries one of k types.
The individuals are the 8-connected components of the non-empty (active) cells.

- A contest samples two active cells uniformly with replacement and takes their components.
- Each component gets a hash score (Equation 1 of the paper). The component with the higher
  score wins; exact ties are broken uniformly at random.
- The winner's bounding box with a one-cell margin is read from the grid as it was before the
  contest and written so that it is centered on the loser. Death and mutation apply to every
  copied cell, including empty cells. Cells whose source or target lies outside the grid are
  not copied. If both sampled cells belong to the same component, it is copied onto itself.
- A step is a sequence of contests. Each contest advances the time by the size of the winner
  divided by the number of active cells, and the step ends when the time reaches 1.

The spatial-dyadic extension has two parameters. `D` is the largest chessboard distance
between the two sampled cells. `P` is the probability that a contest is decided by a hash of
the pair of patterns instead of their individual hash scores. With `dyadic_rule="intended"`
the pair is put into a canonical order, so the winner does not depend on the sampling order.
`dyadic_rule="as_coded"` is the rule as coded in the earlier Mathematica program of the
spatial-dyadic SCHC; the paper uses it for a comparison with the runs of that program.

## Install

The simulations and analyses of the paper used Python 3.10. The C implementation needs
`gcc`; it is compiled on first use.

```bash
uv venv --python 3.10
uv pip install -e ".[paper]" -c docs/npj-analysis-constraints.txt
```

`docs/npj-analysis-constraints.txt` pins the package versions used for the paper.

## Reproduce the figures and tables

Extract the data archive at the root of the repository and run the analysis scripts. They
need no GPU.

```bash
tar -xzf npj-complexity-data.tar.gz
sha256sum -c results/MANIFEST_npj.sha256
uv run --no-sync python scripts/npj_rev_ext1.py
uv run --no-sync python scripts/ext1_novelty_per_event.py
uv run --no-sync python scripts/npj_rev_ext2_tables.py
uv run --no-sync python scripts/npj_rev_ext2_figs.py
uv run --no-sync python scripts/npj_revision_schematics.py
uv run --no-sync python scripts/npj_revision_supplement.py
```

The outputs are written to `results/npj_rev/`:

| Script | Output directory | Content |
|---|---|---|
| `npj_rev_ext1.py`, `ext1_novelty_per_event.py` | `ext1/` | Extension I: Figures 5 to 7, per-run and per-condition tables, tests |
| `npj_rev_ext2_tables.py` | `ext2_tables/` | Extension II: Table 2, tests, Supplementary Fig. S3 |
| `npj_rev_ext2_figs.py` | `ext2fig/` | Figures 8 and 9 |
| `npj_revision_schematics.py` | `schematics/` | schematic of Figure 1 |
| `npj_revision_supplement.py` | `supplement/` | Supplementary Tables S2 to S4, Supplementary Figs. S1 and S2 |

## Run the simulations

```python
from hash_chemistry.fast import run                 # C implementation
rows, final_grid = run(L=100, seed=0, steps=2000)

from hash_chemistry.simulation import SCHCParams    # JAX implementation
params = SCHCParams(L=100, D=10, P=0.6)
```

The datasets of the paper were produced by the following commands.

| Dataset (below `results/`) | Command |
|---|---|
| `jax/spatial_dyadic` | `scripts/run_spatial_dyadic_sweep.py --part i/3` for i = 0, 1, 2 (JAX, one GPU each) |
| `jax/spatial_dyadic_snapshots` | `scripts/run_dyadic_snapshots.py` |
| `jax/original_comparison/runs` | `scripts/compare_original_schc.py --seeds 0-99 --batch 50` |
| `jax/transition_check` | `scripts/run_transition_batched.py --size 100` |
| `cpu_scan_full`, `cpu_novelty`, `cpu_snapshots`, `cpu_sensitivity`, `cpu_scan_40k`, `cpu_scan_check` | `scripts/run_size_scans.sh` (C, CPU only) |

The scripts do not write into a directory that already holds runs of another origin. To
rerun a dataset while the data archive is extracted, give another `--output-root`.

The two implementations use different random number generators, so equal seed numbers do
not give the same run in JAX and in C. JAX runs are reproduced bit for bit only on the same
platform (GPU or CPU).

## Checks

| Command | What it checks |
|---|---|
| `JAX_PLATFORMS=cpu uv run --no-sync python scripts/validate_model_rules.py` | the model rules in small deterministic cases, and the agreement of the JAX and C implementations with an independent NumPy version of a contest |
| `uv run --no-sync python scripts/validate_adaptive_batch.py` | that moving runs between `vmap` batches changes no state, random number stream or event record |
| `uv run --no-sync python scripts/validate_cpu_engine.py` | the C runs against the JAX runs at L=100 (Methods of the paper) |
| `uv run --no-sync python scripts/analyze_original_comparison.py --original-dir DIR` | the JAX runs against the data of the original SCHC study; `DIR` is a checkout of https://github.com/hsayama/Structural-Cellular-Hash-Chemistry |
| `uv run --no-sync python scripts/analyze_extension1_stats.py --mathematica-csv FILE` | the JAX runs with the as-coded dyadic rule against the earlier Mathematica runs of the spatial-dyadic SCHC |

The per-run data of the earlier Mathematica runs of the spatial-dyadic SCHC are not part of
the data archive; they are available from H. Sayama on reasonable request. The statistics
derived from them are in the archive. `scripts/npj_rev_ext1.py` takes the same file through
the optional `--mathematica-csv`; without it, the script writes `regression.csv` and
`joint_tests.csv` without the rows of the Mathematica runs, and every other output is the
same.
