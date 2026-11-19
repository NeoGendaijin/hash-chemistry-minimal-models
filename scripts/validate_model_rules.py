#!/usr/bin/env python3
"""Small, deterministic checks of the model rules, including the grid corners and the
agreement of the JAX and C implementations.

Run with JAX_PLATFORMS=cpu python scripts/validate_model_rules.py.
These tests check the rules, not the statistical agreement of ensembles of runs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import fast
from src.simulation import (SCHCParams, _component_hash,
                            SCHCState, _connected_component, _contest, _dyadic_winner, _mutate, _rewrite,
                            advance_one_step, advance_one_step_logged, initialize_state)


def numpy_contest(grid, start1, start2, coin):
    """Independent NumPy version of one contest without death and mutation."""
    labels, _ = ndimage.label(grid > 0, structure=np.ones((3, 3)))
    masks = [labels == labels.flat[s] for s in (start1, start2)]
    cells = [np.flatnonzero(m).astype(np.int32) for m in masks]
    hashes = [fast.component_hash(grid, c) for c in cells]
    f1, f2 = [np.float32(h) / np.float32(2**32) for h in hashes]
    winner = 0 if f1 > f2 else (1 if f2 > f1 else coin)
    points = [np.argwhere(m) for m in masks]
    boxes = [(p.min(0), p.max(0)) for p in points]
    wm, wx = boxes[winner]
    lm, lx = boxes[1 - winner]
    start = np.rint((lm + lx) / 2).astype(int) + wm - np.rint((wm + wx) / 2).astype(int) - 1
    expected = grid.copy()
    for ox in range(int(wx[0] - wm[0] + 3)):
        for oy in range(int(wx[1] - wm[1] + 3)):
            gx, gy = wm - 1 + [ox, oy]
            tx, ty = start + [ox, oy]
            if all(0 <= a < len(grid) for a in (gx, gy, tx, ty)):
                expected[tx, ty] = grid[gx, gy]
    return expected, boxes[winner], boxes[1 - winner]


class ModelRules(unittest.TestCase):
    def test_defaults_and_parameter_checks(self):
        p = SCHCParams()
        self.assertEqual((p.D, p.P, p.dyadic_rule), (None, 0.0, "intended"))
        self.assertTrue(p.global_competition)
        self.assertTrue(SCHCParams(L=100, D=100).global_competition)
        self.assertFalse(SCHCParams(L=100, D=90).global_competition)
        for bad in (dict(D=0), dict(P=1.5), dict(dyadic_rule="other")):
            with self.assertRaises(ValueError):
                SCHCParams(**bad)

    def test_hash_matches_c_and_is_translation_invariant(self):
        rng = np.random.default_rng(20261023)
        for _ in range(40):
            grid = np.zeros((12, 12), np.int32)
            grid[2:6, 2:6] = rng.integers(1, 1001, (4, 4))
            mask = grid > 0
            cells = np.flatnonzero(mask).astype(np.int32)
            for seed in (0, 101, 202):
                jh = int(_component_hash(jnp.array(grid), jnp.array(mask), seed))
                self.assertEqual(jh, fast.component_hash(grid, cells, seed))
                shifted = np.roll(grid, (3, 4), axis=(0, 1))
                self.assertEqual(jh, int(_component_hash(jnp.array(shifted), jnp.array(shifted > 0), seed)))

    def test_open_components_include_diagonals(self):
        grid = np.eye(8, dtype=bool)
        grid[0, 7] = True
        got = np.asarray(_connected_component(jnp.array(grid), jnp.array([0, 0])))
        self.assertEqual(int(got.sum()), 8)
        self.assertFalse(got[0, 7])

    def test_contest_matches_independent_reference(self):
        rng = np.random.default_rng(23)
        checks = 0
        for _ in range(12):
            grid = np.zeros((10, 10), np.int32)
            grid[0, 0] = int(rng.integers(1, 1001))
            grid[4:7, 4:7] = rng.integers(1, 1001, (3, 3))
            grid[9, 9] = int(rng.integers(1, 1001))
            for a, b in ((0, 44), (44, 0), (99, 44), (44, 99), (0, 0), (44, 44)):
                for coin in (0, 1):
                    expected, wb, lb = numpy_contest(grid, a, b, coin)
                    cpu = fast.forced_contest(grid, a, b, coin)
                    np.testing.assert_array_equal(cpu, expected)
                    params = SCHCParams(L=10, mu=0, death_prob=0)
                    jax_out, _ = _rewrite(jnp.array(grid), tuple(map(jnp.array, wb)),
                                           tuple(map(jnp.array, lb)), random.PRNGKey(0), params)
                    np.testing.assert_array_equal(jax_out, expected)
                    checks += 1
        self.assertEqual(checks, 144)

    def test_copy_onto_the_corner_cell_is_written(self):
        grid = np.zeros((8, 8), np.int32)
        grid[0, 0], grid[4, 4] = 1, 17
        source = (jnp.array([4, 4]), jnp.array([4, 4]))
        target = (jnp.array([0, 0]), jnp.array([0, 0]))
        params = SCHCParams(L=8, mu=0, death_prob=0)
        got, _ = _rewrite(jnp.array(grid), source, target, random.PRNGKey(0), params)
        self.assertEqual(int(got[0, 0]), 17)

    def test_dyadic_winner_does_not_depend_on_sampling_order(self):
        for a in range(1, 16):
            grid = np.zeros((8, 8), np.int32)
            grid[1, 1], grid[5, 5] = a, 731
            g = jnp.array(grid)
            first = jnp.zeros_like(g, dtype=bool).at[1, 1].set(True)
            second = jnp.zeros_like(g, dtype=bool).at[5, 5].set(True)
            forward = int(_dyadic_winner(g, first, second, "intended"))
            backward = int(_dyadic_winner(g, second, first, "intended"))
            self.assertEqual(forward, 1 - backward)

    def test_jax_contests_never_replace_the_stronger_pattern_with_the_weaker(self):
        grid = np.zeros((10, 10), np.int32)
        grid[2, 2], grid[7, 7] = 17, 731
        masks = [jnp.zeros_like(jnp.array(grid), dtype=bool).at[x, x].set(True) for x in (2, 7)]
        hashes = [int(_component_hash(jnp.array(grid), mask)) for mask in masks]
        x = (2, 7)[int(hashes[1] > hashes[0])]
        params = SCHCParams(L=10, mu=0, death_prob=0)
        def contest(key):
            state, _, _ = _contest(SCHCState(jnp.array(grid), key, jnp.int32(0)), params)
            return state.config
        configs = np.asarray(jax.jit(jax.vmap(contest))(random.split(random.PRNGKey(23), 256)))
        self.assertTrue(np.all(configs[:, x, x] == grid[x, x]))
        self.assertTrue(np.any(configs != grid))

    def test_mutation_includes_empty_cells_and_death_has_priority(self):
        values = jnp.array([0, 3, 7], dtype=jnp.int32)
        for death, mu in ((0, 0), (1, 1), (0, 1)):
            result, _ = _mutate(values, random.PRNGKey(0), SCHCParams(mu=mu, death_prob=death))
            if death:
                np.testing.assert_array_equal(result, np.zeros(3))
            elif not mu:
                np.testing.assert_array_equal(result, values)
            else:
                self.assertTrue(np.all((np.asarray(result) >= 1) & (np.asarray(result) <= 1000)))

    def test_event_logging_does_not_change_dynamics(self):
        params = SCHCParams(L=10)
        plain = initialize_state(random.PRNGKey(23), params)
        logged = plain
        seen = jnp.zeros(params.k + 1, dtype=bool)
        plain_step = jax.jit(lambda state: advance_one_step(state, params))
        logged_step = jax.jit(lambda state, seen: advance_one_step_logged(state, params, seen, 101))
        for _ in range(4):
            plain = plain_step(plain)
            logged, seen, _ = logged_step(logged, seen)
            np.testing.assert_array_equal(plain.config, logged.config)
            np.testing.assert_array_equal(plain.rng, logged.rng)

    def test_c_implementation_reproduces_stored_run(self):
        path = ROOT / "results/cpu_scan_full/original_mix/L100/runs/seed_0.csv"
        if not path.exists():
            self.skipTest("extract the data archive to compare with a stored run")
        ref = np.genfromtxt(path, delimiter=",", names=True)
        rows, _ = fast.run(100, 0, 100, sample_interval=10)
        expected = np.column_stack([ref[c][:len(rows)] for c in fast.COLUMNS])
        # The stored CSV keeps eight decimals of the score and size columns.
        np.testing.assert_array_equal(np.round(rows, 8), expected)

    def test_stored_runs_do_not_depend_on_batch_size(self):
        # The 30 runs of the original setting exist in two datasets, generated in batches of 30
        # and of 50 seeds (D=100 at L=100 is global competition).
        a_dir = ROOT / "results/jax/spatial_dyadic/intended/D100_P0.0"
        b_dir = ROOT / "results/jax/original_comparison/runs"
        if not (a_dir.is_dir() and b_dir.is_dir()):
            self.skipTest("extract the data archive to compare the two stored datasets")
        columns = ["step", "num_replicated_cells", "max_rep_size", "mean_rep_size", "cum_cell_types",
                   "cum_pattern_types", "n_events", "n_active", "log_bytes_cum"]
        for seed in range(30):
            a = np.genfromtxt(a_dir / f"seed_{seed}.csv", delimiter=",", names=True)
            b = np.genfromtxt(b_dir / f"seed_{seed}.csv", delimiter=",", names=True)
            for name in columns:
                np.testing.assert_array_equal(a[name], b[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
