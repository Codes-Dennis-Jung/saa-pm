"""THE one place point-in-time decision inputs are assembled.

A decision indexed at ``t`` (a row position in the monthly return history)
sees only information available when month ``t`` opens:

* policy      policy_history rows effective <= last closed month
* goals       the income target IN FORCE at that date (optional glide)
* CMAs        the cma_history vintage in force; else a blend with trailing
              realised returns (if configured); else assets.csv
* pacing      growth_rate re-synced to the CMA (vintage IRR invariant)
* covariance  estimated on returns.iloc[:t] only

The point-in-time run uses ``t = len(returns)``; the walk-forward backtest
calls the SAME builder at every decision month, so both see identical
inputs at the same date by construction. All objects are immutable, so the
state at ``t`` is a pure function of (files, t).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..config import Goals, OptimizerConfig
from ..io.loaders import MandateData
from ..privates.pacing import PacingParams
from ..risk.covariance import RiskEstimate, RiskModel
from ..universe import GroupConstraint, Universe


@dataclass
class DecisionState:
    t: int
    asof: pd.Timestamp  # last CLOSED month = returns.index[t-1]
    universe: Universe  # policy bounds + CMAs in force
    risk: RiskEstimate
    goals: Goals
    optimizer: OptimizerConfig  # targets in force at asof
    pacing: Dict[str, PacingParams]
    cma_source: str  # vintage | blend | static
    policy: Dict[str, float]

    @property
    def sigma(self) -> pd.DataFrame:
        return self.risk.sigma

    @property
    def sigma_trace(self) -> float:
        return float(np.trace(self.sigma.values))

    @property
    def mu_sum(self) -> float:
        return float(self.universe.mu.sum())


def apply_policy(
    policy: Dict[str, float], universe: Universe, goals: Goals, opt: OptimizerConfig
) -> tuple[Universe, Goals, OptimizerConfig]:
    """Return copies with the policy keys applied (keys validated at load)."""
    if not policy:
        return universe, goals, opt
    asset_changes: Dict[str, Dict[str, float]] = {}
    for k, v in policy.items():
        parts = k.split(".")
        if len(parts) == 3 and parts[0] == "asset":
            asset_changes.setdefault(parts[1], {})[parts[2]] = float(v)
    if asset_changes:
        universe = universe.updated(asset_changes)
    groups = [
        GroupConstraint(
            g.name, g.members, policy.get(f"group.{g.name}.lo", g.lo), policy.get(f"group.{g.name}.hi", g.hi)
        )
        for g in universe.groups
    ]
    universe = universe.with_groups(groups)
    goal_kw = {
        k: float(policy[k])
        for k in ("cpi", "return_spread", "target_income", "vol_lo", "vol_hi")
        if k in policy
    }
    if goal_kw:
        goals = replace(goals, **goal_kw)
    opt_kw: Dict[str, Any] = {}
    if "turnover_limit" in policy:
        tl = float(policy["turnover_limit"])
        opt_kw["turnover_limit"] = None if tl <= 0 else tl
    if "income_penalty" in policy:
        opt_kw["income_penalty"] = float(policy["income_penalty"])
    if opt_kw:
        opt = replace(opt, **opt_kw)
    return universe, goals, opt


class StateBuilder:
    """Builds a :class:`DecisionState` for any month of the history."""

    def __init__(self, data: MandateData, vol_steps: Optional[int] = None):
        self.data = data
        self.returns = data.common_returns
        self.risk_model = RiskModel(data.settings.risk)
        opt = data.settings.optimizer
        self.base_optimizer = replace(opt, vol_steps=vol_steps) if vol_steps else opt

    def __len__(self) -> int:
        return len(self.returns)

    def date(self, t: int) -> pd.Timestamp:
        return self.returns.index[t - 1]

    def cmas_asof(self, universe: Universe, t: int) -> tuple[Universe, str]:
        data, asof = self.data, self.date(t)
        vint = data.panel.cma_asof(asof)
        if vint:
            return universe.with_cmas(vint), "vintage"
        bt = data.settings.backtest
        if bt.cma_mode == "static":
            return universe, "static"
        n = min(t, bt.cma_trailing_years * 12)
        win = self.returns.iloc[t - n : t]
        realised = (1 + win).prod() ** (12 / len(win)) - 1
        return universe.updated(
            {
                a.name: {
                    "exp_return": (1 - bt.cma_blend) * a.exp_return + bt.cma_blend * float(realised[a.name])
                }
                for a in universe
            }
        ), "blend"

    def build(self, t: Optional[int] = None) -> DecisionState:
        T = len(self.returns)
        t = T if t is None else t
        if not 1 <= t <= T:
            raise ValueError(f"t={t} outside 1..{T}")
        asof = self.date(t)
        data = self.data
        policy = data.panel.policy_asof(asof)
        universe, goals, opt = apply_policy(policy, data.universe, data.settings.goals, self.base_optimizer)
        opt = opt.with_goals(goals, asof)
        universe, source = self.cmas_asof(universe, t)
        pacing = {n: p.synced_to(universe[n].exp_return, universe[n].income) for n, p in data.pacing.items()}
        risk = self.risk_model.estimate(universe, self.returns, t)
        return DecisionState(t, asof, universe, risk, goals, opt, pacing, source, policy)
