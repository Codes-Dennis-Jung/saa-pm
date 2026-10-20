"""Efficient frontiers - what private markets and the policy limits are worth.

Three reference frontiers (max expected return for each volatility):

* ``policy``         all bounds and group limits, privates free within policy
* ``no_privates``    the same, with private sleeves forced to zero
* ``unconstrained``  long-only, fully invested, no bounds or groups

The vertical gap between ``policy`` and ``no_privates`` is the return the
private-markets allocation adds at equal risk; between ``unconstrained`` and
``policy`` it is the cost of the policy limits.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from ..config import OptimizerConfig
from ..optimizer.constraints import ConstraintSpec, solve_problem
from ..risk.covariance import align_to, nearest_psd
from ..universe import Universe


def frontier(
    universe: Universe, sigma: pd.DataFrame, vols: Sequence[float], config: Optional[OptimizerConfig] = None
) -> pd.DataFrame:
    """Max-return portfolio at each volatility cap (one parametrised problem)."""
    config = config or OptimizerConfig()
    S = nearest_psd(align_to(sigma, universe.names)).values
    L = np.linalg.cholesky(S + 1e-12 * np.eye(len(S)))
    w = cp.Variable(universe.n)
    cap = cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Maximize(universe.mu @ w), ConstraintSpec(universe, config).build(w) + [cp.norm(L.T @ w, 2) <= cap]
    )
    rows: List[dict] = []
    for v in vols:
        cap.value = float(v)
        if not solve_problem(prob, config.solver):
            continue
        wv = np.where(np.abs(w.value) < 1e-8, 0.0, w.value)
        vol = float(np.sqrt(wv @ S @ wv))
        if rows and vol < rows[-1]["vol"] - 1e-6:  # flat tail: risk budget unused
            continue
        rows.append(
            dict(
                vol_cap=float(v),
                vol=vol,
                exp_return=float(universe.mu @ wv),
                income=float(universe.income @ wv),
                illiquid=float(wv[~universe.liquid_mask].sum()),
                **dict(zip(universe.names, wv)),
            )
        )
    return pd.DataFrame(rows)


def reference_frontiers(
    universe: Universe,
    sigma: pd.DataFrame,
    vols: Optional[Sequence[float]] = None,
    config: Optional[OptimizerConfig] = None,
) -> Dict[str, pd.DataFrame]:
    config = config or OptimizerConfig()
    grid: Sequence[float] = list(np.linspace(0.03, 0.15, 25)) if vols is None else vols
    no_priv = universe.updated({n: {"lb": 0.0, "ub": 0.0} for n in universe.illiquid_names})
    # only floors of purely-private groups are dropped; mixed groups keep theirs
    no_priv = no_priv.with_groups(
        [replace(g, lo=0.0) if set(g.members) <= set(universe.illiquid_names) else g for g in no_priv.groups]
    )
    free = Universe([replace(a, lb=0.0, ub=1.0) for a in universe], [])
    return {
        "policy": frontier(universe, sigma, grid, config),
        "no_privates": frontier(no_priv, sigma, grid, config),
        "unconstrained": frontier(free, sigma, grid, config),
    }


def frontier_gaps(frontiers: Dict[str, pd.DataFrame], at_vols: Sequence[float]) -> pd.DataFrame:
    """Expected return of each frontier interpolated at the given vols."""
    out = {}
    for k, f in frontiers.items():
        if len(f) < 2:
            out[k] = [np.nan] * len(at_vols)
            continue
        x, y = f["vol"].values, f["exp_return"].values
        out[k] = [
            float(np.interp(v, x, y)) if x.min() - 1e-6 <= v <= x.max() + 1e-6 else np.nan for v in at_vols
        ]
    df = pd.DataFrame(out, index=pd.Index(at_vols, name="vol"))
    if {"policy", "no_privates"} <= set(df):
        df["privates_add"] = df["policy"] - df["no_privates"]
    if {"policy", "unconstrained"} <= set(df):
        df["policy_cost"] = df["unconstrained"] - df["policy"]
    return df
