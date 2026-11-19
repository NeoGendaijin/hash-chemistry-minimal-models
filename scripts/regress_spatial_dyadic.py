"""Quadratic response surface of the three measurements of the spatial-dyadic SCHC (Extension I).

Model:  Y = a + b D + c D^2 + d P + e P^2 + f D P + error
with Y = A (successful runs per condition, 60 points), B (mean active cells per replication event in
the final 500 steps, per successful run) and C (cumulative pattern types at t=2000, per successful
run).

This module holds what scripts/npj_rev_ext1.py imports: the design matrix, an OLS fit with classical
and HC3 robust standard errors, and the loader of the per-run data of the earlier Mathematica runs of
the spatial-dyadic SCHC (Sayama 2026), which used the as-coded dyadic rule. That CSV has no header; its
columns are D, P, the name of the measurement (starting with A, B or C) and the value: A once per
condition, B and C once per successful run. It is not part of the data archive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def design(D, P):
    D, P = np.asarray(D, float), np.asarray(P, float)
    return np.column_stack([np.ones_like(D), D, D ** 2, P, P ** 2, D * P])


def ols(X, y):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    r = y - X @ beta
    se = np.sqrt(np.diag(XtXi) * (r @ r) / (n - k))
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    se_hc3 = np.sqrt(np.diag(XtXi @ (X.T * (r / (1 - h)) ** 2) @ X @ XtXi))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    p_hc3 = 2 * stats.t.sf(np.abs(beta / se_hc3), n - k)
    return beta, se, p, p_hc3, n


def load_mathematica(path):
    """A, B and C of the Mathematica runs from their per-run CSV (see the module docstring for its layout)."""
    d = pd.read_csv(path, header=None, names=["D", "P", "measure", "value"])
    d["P"] = d["P"].round(1)
    d["m"] = d["measure"].str[0]
    return {m: d[d["m"] == m][["D", "P", "value"]].reset_index(drop=True) for m in "ABC"}
