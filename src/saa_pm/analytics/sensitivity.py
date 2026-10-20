"""Sensitivity of the strategic allocation, and which constraints bind.

One-at-a-time perturbations of the inputs the allocation depends on most -
expected returns per asset class, the covariance level, the targets and
the risk band - each re-solved with the SAME goal optimiser. Reported per
case: expected return / income / vol, turnover vs the base solution and the
largest weight move. Grid discretisation can make small shocks jump between
grid points; treat single-digit-bp changes as noise.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import OptimizerConfig
from ..optimizer.goal_optimizer import GoalOptimizer, InfeasibleError
from ..universe import Universe

Inputs = Tuple[Universe, pd.DataFrame, OptimizerConfig]
Perturbation = Callable[[Universe, pd.DataFrame, OptimizerConfig], Inputs]
Case = Tuple[str, Perturbation]


def shift_cma(members: List[str], d: float) -> Perturbation:
    """Expected return of ``members`` shifted by ``d``."""

    def f(u: Universe, S: pd.DataFrame, c: OptimizerConfig) -> Inputs:
        return u.updated({n: {"exp_return": u[n].exp_return + d} for n in members}), S, c

    return f


def scale_covariance(k: float) -> Perturbation:
    """Every variance and covariance multiplied by ``k``."""

    def f(u: Universe, S: pd.DataFrame, c: OptimizerConfig) -> Inputs:
        return u, S * k, c

    return f


def change_config(**deltas: float) -> Perturbation:
    """Add ``deltas`` to optimiser fields, e.g. ``change_config(target_return=0.005)``."""

    def f(u: Universe, S: pd.DataFrame, c: OptimizerConfig) -> Inputs:
        new = {k: getattr(c, k) + v for k, v in deltas.items()}
        if "vol_lo" in new:
            new["vol_lo"] = max(0.0, new["vol_lo"])
        return u, S, replace(c, **new)

    return f


def default_cases(
    universe: Universe, cma_shock: float = 0.01, vol_scale: float = 0.2, target_shock: float = 0.005
) -> List[Case]:
    cases: List[Case] = []
    for cls in sorted({a.group for a in universe}):
        members = [a.name for a in universe if a.group == cls]
        for d in (+cma_shock, -cma_shock):
            cases.append((f"CMA {cls} {d:+.1%}", shift_cma(members, d)))
    for k in (1 + vol_scale, 1 - vol_scale):
        cases.append((f"covariance x{k:.2f}", scale_covariance(k)))
    for d in (+target_shock, -target_shock):
        cases.append((f"return target {d:+.1%}", change_config(target_return=d)))
        cases.append((f"income target {d:+.1%}", change_config(target_income=d)))
    for d in (+0.01, -0.01):
        cases.append((f"risk band {d:+.0%}", change_config(vol_lo=d, vol_hi=d)))
    return cases


def sensitivity(
    universe: Universe, sigma: pd.DataFrame, config: OptimizerConfig, cases: Optional[List[Case]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per case, weights per case)."""
    base = GoalOptimizer(universe, sigma, config).run(allow_fallback=True)
    w0 = base.weights
    rows, weights = [], {"base": w0}
    rows.append(
        dict(
            case="base",
            exp_return=base.solution.exp_return,
            income=base.solution.income,
            vol=base.solution.vol,
            turnover=0.0,
            largest_move="",
            rule=base.solution.selection_rule,
        )
    )
    for name, fn in cases or default_cases(universe):
        u, S, c = fn(universe, sigma, config)
        try:
            r = GoalOptimizer(u, S, c).run(allow_fallback=True)
        except InfeasibleError:
            rows.append(dict(case=name, rule="infeasible"))
            continue
        w = r.weights
        d = w - w0
        i = int(np.argmax(np.abs(d)))
        rows.append(
            dict(
                case=name,
                exp_return=r.solution.exp_return,
                income=r.solution.income,
                vol=r.solution.vol,
                turnover=0.5 * float(np.abs(d).sum()),
                largest_move=f"{universe.names[i]} {d[i]:+.1%}",
                rule=r.solution.selection_rule,
            )
        )
        weights[name] = w
    return (pd.DataFrame(rows).set_index("case"), pd.DataFrame(weights, index=universe.names))


def binding_constraints(w: np.ndarray, universe: Universe, tol: float = 1e-4) -> pd.DataFrame:
    """Asset bounds and group limits the solution sits on."""
    rows = []
    for i, a in enumerate(universe):
        if a.lb > 0 and w[i] <= a.lb + tol:
            rows.append(dict(constraint=f"asset {a.name}", side="lower", limit=a.lb, value=w[i]))
        if w[i] >= a.ub - tol:
            rows.append(dict(constraint=f"asset {a.name}", side="upper", limit=a.ub, value=w[i]))
    for g, sel in universe.group_selectors():
        x = float(sel @ w)
        if g.lo > 0 and x <= g.lo + tol:
            rows.append(dict(constraint=f"group {g.name}", side="lower", limit=g.lo, value=x))
        if x >= g.hi - tol:
            rows.append(dict(constraint=f"group {g.name}", side="upper", limit=g.hi, value=x))
    return pd.DataFrame(rows)
