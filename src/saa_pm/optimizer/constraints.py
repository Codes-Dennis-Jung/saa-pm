"""The ONE place the portfolio constraint set is defined.

Used by the optimiser, the feasibility diagnostics, the max-attainable LP
behind floor clamping, and (numerically) by the post-hoc audit, so all four
always agree on what "feasible" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from ..config import OptimizerConfig
from ..privates.ladder import Floor
from ..universe import Universe

STRATEGIC = "strategic"  # destination portfolio: no turnover / no-trade bands
REBALANCE = "rebalance"  # trade from w_prev: turnover cap + no-trade bands


@dataclass
class ConstraintSpec:
    universe: Universe
    config: OptimizerConfig
    floors: Sequence[Floor] = ()
    w_prev: Optional[np.ndarray] = None
    mode: str = STRATEGIC

    def __post_init__(self):
        if self.mode not in (STRATEGIC, REBALANCE):
            raise ValueError(f"mode must be '{STRATEGIC}' or '{REBALANCE}', got {self.mode!r}")
        self.floors = list(self.floors or [])
        if self.w_prev is not None:
            self.w_prev = np.asarray(self.w_prev, float)

    @property
    def trading_rules_active(self) -> bool:
        return self.mode == REBALANCE and self.w_prev is not None

    def trade_caps(self) -> np.ndarray:
        """Per-asset |w - w_prev| cap (inf = none). Privates default to a tight
        band: their NAV drifts, they cannot be traded."""
        caps = []
        for a in self.universe:
            cap = a.max_trade
            if cap is None and not a.liquid:
                cap = self.config.illiquid_max_trade
            caps.append(np.inf if cap is None else float(cap))
        return np.array(caps)

    def build(self, w: cp.Variable, include_trading: bool = True) -> List[cp.Constraint]:
        u = self.universe
        cons = [cp.sum(w) == 1, w >= u.lb, w <= u.ub]
        for g, sel in u.group_selectors():
            cons += [sel @ w >= g.lo, sel @ w <= g.hi]
        for f in self.floors:
            if f.level > 0:
                cons.append(f.selector @ w >= f.level)
        if include_trading and self.trading_rules_active:
            assert self.w_prev is not None
            tl = self.config.turnover_limit
            if tl is not None:
                cons.append(0.5 * cp.norm1(w - self.w_prev) <= tl)
            caps = self.trade_caps()
            fin = np.isfinite(caps)
            if fin.any():
                idx = np.where(fin)[0]
                cons.append(cp.abs(w[idx] - self.w_prev[idx]) <= caps[idx])
            gated = self.redemption_caps()
            if gated:
                idx = np.array(list(gated))
                cons.append(self.w_prev[idx] - w[idx] <= np.array([gated[i] for i in idx]))
        return cons

    def redemption_caps(self) -> dict:
        """{index: max weight that can be SOLD} for semi-liquid (gated) assets;
        buying is not restricted."""
        assert self.w_prev is not None
        return {
            i: float(a.redemption_limit) * max(float(self.w_prev[i]), 0.0)
            for i, a in enumerate(self.universe)
            if a.semi_liquid
        }

    def audit(
        self, w: np.ndarray, sigma: pd.DataFrame, vol_ceiling: float, tol: float = 1e-6
    ) -> pd.DataFrame:
        """Independent check that ``w`` satisfies every constraint."""
        u, c = self.universe, self.config
        w = np.asarray(w, float)
        rows = [("budget sums to 1", abs(w.sum() - 1) <= 1e-7, f"{w.sum():.8f}")]
        viol = [(a.name, round(w[i], 5)) for i, a in enumerate(u) if w[i] < a.lb - tol or w[i] > a.ub + tol]
        rows.append(("single-asset bounds", not viol, str(viol) if viol else "ok"))
        gv = [
            (g.name, round(float(sel @ w), 4))
            for g, sel in u.group_selectors()
            if not g.lo - tol <= float(sel @ w) <= g.hi + tol
        ]
        rows.append(("group bounds", not gv, str(gv) if gv else "ok"))
        vol = float(np.sqrt(max(w @ sigma.values @ w, 0.0)))
        rows.append(("vol <= ceiling", vol <= vol_ceiling + 1e-5, f"{vol:.4%}"))
        rows.append(("vol >= band floor", vol >= c.vol_lo - 1e-4, f"{vol:.4%}"))
        gtol = c.return_tol + 1e-6
        inc = float(u.income @ w)
        ret = float(u.mu @ w)
        rows.append(("income target", inc >= c.target_income - gtol, f"{inc:.4%}"))
        rows.append(("return target", ret >= c.target_return - gtol, f"{ret:.4%}"))
        if self.trading_rules_active:
            assert self.w_prev is not None
            t = 0.5 * float(np.abs(w - self.w_prev).sum())
            rows.append(
                ("turnover limit", c.turnover_limit is None or t <= c.turnover_limit + 1e-6, f"{t:.4%}")
            )
            caps = self.trade_caps()
            moved = [u.names[i] for i in range(u.n) if abs(w[i] - self.w_prev[i]) > caps[i] + tol]
            rows.append(("no-trade bands", not moved, str(moved) if moved else "ok"))
            gated = self.redemption_caps()
            if gated:
                over = [u.names[i] for i, c in gated.items() if self.w_prev[i] - w[i] > c + tol]
                rows.append(("redemption gates", not over, str(over) if over else "ok"))
        if self.floors:
            bad = [
                (f.bucket, round(float(f.selector @ w), 4), round(f.level, 4))
                for f in self.floors
                if float(f.selector @ w) < f.level - tol
            ]
            rows.append(("ladder floors", not bad, str(bad) if bad else "ok"))
        return pd.DataFrame(rows, columns=["check", "pass", "value"])


def solve_problem(problem: cp.Problem, solver: str) -> bool:
    """Solve with the configured solver, falling back to SCS. Never raises;
    returns True if a usable (optimal / optimal_inaccurate) solution exists."""
    for s in dict.fromkeys([solver, "SCS"]):
        try:
            problem.solve(solver=s)
        except (cp.SolverError, ValueError, ArithmeticError):
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return True
        if problem.status in ("infeasible", "unbounded"):
            return False
    return False


def max_attainable(spec: ConstraintSpec, selector: np.ndarray, include_trading: bool = False) -> float:
    """Maximum weight ``selector`` can hold under the POLICY constraints of
    ``spec`` (bounds, groups, floors; trading frictions only if asked; no risk
    model). NaN if the constraint set itself is infeasible."""
    w = cp.Variable(spec.universe.n)
    prob = cp.Problem(cp.Maximize(selector @ w), spec.build(w, include_trading))
    if solve_problem(prob, spec.config.solver):
        return float(prob.value)
    return float("nan")
