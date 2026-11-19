#!/usr/bin/env python3
"""
Size scan with the C implementation (src/schc_fast.c), one run per worker process.

Output per (L, seed), with the columns of src/fast.py:
  <output-root>/original_mix[_mu..][_pd..]/L{L}/runs/seed_{s}.csv
  <output-root>/original_mix[_mu..][_pd..]/L{L}/configs/seed_{s}_final.npy
Existing CSVs are skipped. A run is determined by L, the seed and the parameters. The random
streams differ from those of the JAX implementation, so equal seed numbers do not give
the same run in the two implementations.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# importing src pulls in JAX; keep it off the GPUs (this script only uses the C implementation)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import fast

HEADER = ",".join(fast.COLUMNS)
FMT = ["%d", "%.8f", "%.8f", "%d", "%.8f", "%d", "%d", "%d", "%d"]


def one(job):
    L, seed, a = job
    run_dir = Path(a["out"]) / f"L{L}" / "runs"
    cfg_dir = Path(a["out"]) / f"L{L}" / "configs"
    out = run_dir / f"seed_{seed}.csv"
    if out.exists():
        return L, seed, None, 0.0
    t0 = time.time()
    rows, final = fast.run(L, seed, a["steps"], mu=a["mu"], death_prob=a["death_prob"],
                           sample_interval=a["sample_interval"], stop_mean_size=a["stop_mean_size"],
                           init_config=None if a["init"] is None else np.load(a["init"]).astype(np.int32))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, rows, delimiter=",", header=HEADER, comments="", fmt=FMT)
    np.save(cfg_dir / f"seed_{seed}_final.npy", final.astype(np.int16))
    return L, seed, rows[-1], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--stop-mean-size", type=float, default=0.0,
                    help="stop a run when its mean component size reaches this value (0: never)")
    ap.add_argument("--mu", type=str, default="0.002/0.999")
    ap.add_argument("--death-prob", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output-root", default=str(ROOT / "results" / "cpu_scan_full"))
    ap.add_argument("--init-config", default=None, help=".npy configuration shared by all seeds (seeds then only set the RNG)")
    args = ap.parse_args()

    lo, hi = (map(int, args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds), int(args.seeds)))
    seeds = list(range(lo, hi + 1))
    mu = float(eval(args.mu, {"__builtins__": {}}))
    tag = "original_mix"  # name of the output directory below --output-root
    if args.mu != "0.002/0.999":
        tag += f"_mu{mu:g}"
    if args.death_prob != 0.001:
        tag += f"_pd{args.death_prob:g}"
    a = dict(out=str(Path(args.output_root) / tag), steps=args.steps, mu=mu, death_prob=args.death_prob,
             sample_interval=args.sample_interval, stop_mean_size=args.stop_mean_size, init=args.init_config)
    fast._load()  # build once before forking
    # largest L first so the slowest jobs do not form the tail
    jobs = [(L, s, a) for L in sorted(args.sizes, reverse=True) for s in seeds]
    print(f"{tag}: {len(jobs)} runs, sizes={args.sizes}, seeds={lo}-{hi}, workers={args.workers}", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for L, seed, last, dt in pool.imap_unordered(one, jobs):
            if last is None:
                continue
            print(f"L={L} seed={seed} steps={int(last[0])} mean_size={last[4]:.1f} max_size={int(last[3])} "
                  f"n_active={int(last[7])} ({dt:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
