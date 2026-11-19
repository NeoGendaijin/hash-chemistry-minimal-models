"""Record the parameters and the sources of a simulation dataset next to its outputs."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
JAX_ROOT = ROOT / "results" / "jax"  # default location of the JAX datasets


def complete_outputs(directory, definition):
    seeds, steps = definition["seeds"], definition["steps"]
    if "save_steps" in definition:
        return all((directory / f"seed_{seed}_t{t}.npy").is_file()
                   for seed in seeds for t in definition["save_steps"])
    for seed in seeds:
        path = directory / f"seed_{seed}.csv"
        if not path.is_file():
            return False
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return False
        times = [int(row["step"]) for row in rows]
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            return False
        if "sample_interval" not in definition:
            if len(rows) != steps or times[-1] != steps - 1:
                return False
        else:
            last = rows[-1]
            threshold = definition.get("stop_mean_size", 0)
            if not (times[-1] == steps or int(last["n_active"]) == 0 or
                    (threshold > 0 and float(last["mean_size"]) >= threshold)):
                return False
    return True


def register_run(directory, params, seeds, steps, **extra):
    """Write run_metadata.json for a new dataset, or check that an existing one has the same definition."""
    directory = Path(directory)
    definition = dict(model=asdict(params), seeds=list(seeds), steps=steps, **extra)
    path = directory / "run_metadata.json"
    if path.exists():
        if json.loads(path.read_text())["definition"] != definition:
            raise ValueError(f"{path} describes a different simulation; choose another output directory")
        return
    if directory.exists() and any(directory.glob("seed_*")):
        raise ValueError(f"{directory} holds simulation outputs without metadata; choose another output directory")
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "src/simulation.py", ROOT / "src/metrics.py",
               ROOT / "scripts/compare_original_schc.py", ROOT / "scripts/adaptive_batch.py",
               ROOT / "scripts/run_spatial_dyadic_sweep.py", ROOT / "scripts/run_dyadic_snapshots.py",
               ROOT / "scripts/run_transition_batched.py", Path(__file__)]
    metadata = dict(definition=definition,
                    source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources},
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    jax_version=jax.__version__, devices=list(map(str, jax.devices())))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    if commit.returncode == 0:
        metadata["code_commit"] = commit.stdout.strip()
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
        name = stream.name
    os.replace(name, path)
