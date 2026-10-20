"""Point-in-time allocation: the strategic target and today's portfolio.

STRATEGIC TARGET  the destination policy portfolio - private sleeves free
                  within their policy bounds, no funding floors. Where the
                  book SHOULD go; the gap to NAV + scheduled calls is the
                  commitment decision.
TODAY             the implementable portfolio - private sleeves pinned at
                  their current NAV (they cannot be traded), funding-ladder
                  floors sized from the known call schedule, liquid book
                  optimised. This is what you trade to now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import OptimizerConfig
from ..io.loaders import MandateData
from ..optimizer.constraints import REBALANCE, STRATEGIC
from ..optimizer.goal_optimizer import GoalOptimizer, InfeasibleError, OptimizationResult, clamp_group_floors
from ..privates.ladder import FundingLadder, LadderRequirement
from ..universe import GroupBreach, Universe
from .state import DecisionState, StateBuilder


@dataclass
class TodayProblem:
    """Everything that defines the TODAY solve - reused by side analyses
    (sensitivity, binding constraints) so they perturb exactly this problem."""

    universe: Universe  # privates pinned, unreachable group floors clamped
    sigma: pd.DataFrame
    config: OptimizerConfig
    ladder: LadderRequirement
    w_prev: np.ndarray
    breaches: List[GroupBreach] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def optimizer(self, **overrides) -> GoalOptimizer:
        from dataclasses import replace

        cfg = replace(self.config, **overrides) if overrides else self.config
        return GoalOptimizer(self.universe, self.sigma, cfg, self.ladder.floors, self.w_prev, REBALANCE)

    def solve(self, **overrides) -> OptimizationResult:
        return self.optimizer(**overrides).run(allow_fallback=True)


@dataclass
class AllocationResult:
    state: DecisionState
    current: np.ndarray
    portfolio_value: float
    strategic: OptimizationResult
    today: OptimizationResult
    ladder: LadderRequirement
    ladder_table: pd.DataFrame
    diagnostics: dict
    notes: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return self.state.universe.names

    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"current": self.current, "strategic": self.strategic.weights, "today": self.today.weights},
            index=self.names,
        )

    def summary(self) -> pd.DataFrame:
        u = self.state.universe
        S = self.state.sigma.values
        rows = {}
        for k, w in (
            ("current", self.current),
            ("strategic", self.strategic.weights),
            ("today", self.today.weights),
        ):
            rows[k] = dict(
                exp_return=float(u.mu @ w),
                income=float(u.income @ w),
                vol=float(np.sqrt(max(w @ S @ w, 0))),
                illiquid=float(w[~u.liquid_mask].sum()),
                turnover_vs_current=float(0.5 * np.abs(w - self.current).sum()),
            )
        df = pd.DataFrame(rows).T
        cfg = self.state.optimizer
        df["return_target"] = cfg.target_return
        df["income_target"] = cfg.target_income
        df["vol_band"] = f"{cfg.vol_lo:.1%}-{cfg.vol_hi:.1%}"
        df.loc["strategic", "rule"] = self.strategic.solution.selection_rule
        df.loc["today", "rule"] = self.today.solution.selection_rule
        return df

    def commitment_gap(self) -> pd.DataFrame:
        """Strategic private weight vs NAV today + scheduled calls."""
        rows = []
        idx = {n: i for i, n in enumerate(self.names)}
        sched = self.ladder.detail
        for n in self.state.universe.illiquid_names:
            planned = (
                (float(sched.loc[sched["asset"] == n, "amount"].sum()) / self.portfolio_value)
                if len(sched)
                else 0.0
            )
            tgt = float(self.strategic.weights[idx[n]])
            nav = float(self.current[idx[n]])
            rows.append(
                dict(
                    asset=n,
                    strategic_target=tgt,
                    nav_today=nav,
                    scheduled_calls=planned,
                    gap=tgt - nav - planned,
                )
            )
        return pd.DataFrame(rows).set_index("asset") if rows else pd.DataFrame()


class PointInTimeAllocator:
    def __init__(self, data: MandateData, builder: Optional[StateBuilder] = None):
        self.data = data
        self.builder = builder or StateBuilder(data)

    def today_problem(
        self,
        state: DecisionState,
        w_prev: Optional[np.ndarray] = None,
        schedule: Optional[pd.DataFrame] = None,
        portfolio_value: Optional[float] = None,
        cover: Optional[float] = None,
    ) -> TodayProblem:
        data = self.data
        w_prev = data.holdings.weights if w_prev is None else np.asarray(w_prev, float)
        pv = data.holdings.total if portfolio_value is None else portfolio_value
        pinned, breaches = state.universe.pin_privates(w_prev)
        pinned, notes = clamp_group_floors(pinned, state.optimizer)
        ladder_cfg = data.settings.ladder
        if cover is not None:
            from dataclasses import replace

            ladder_cfg = replace(ladder_cfg, cover=cover)
        ladder = FundingLadder(state.universe, ladder_cfg)
        req = ladder.requirement(data.schedule if schedule is None else schedule, state.asof, pv)
        notes += [
            f"pinned private NAV breaches group '{b.group}' ({b.weight:.2%} > {b.cap:.2%}) - cap widened"
            for b in breaches
        ]
        return TodayProblem(pinned, state.sigma, state.optimizer, req, w_prev, breaches, notes)

    def strategic(self, state: DecisionState, w_prev: Optional[np.ndarray] = None) -> OptimizationResult:
        w_prev = self.data.holdings.weights if w_prev is None else w_prev
        opt = GoalOptimizer(state.universe, state.sigma, state.optimizer, (), w_prev, STRATEGIC)
        try:
            return opt.run()
        except InfeasibleError:
            return opt.run(allow_fallback=True)

    def run(self, t: Optional[int] = None) -> AllocationResult:
        data = self.data
        state = self.builder.build(t)
        current = data.holdings.weights
        pv = data.holdings.total
        strategic = self.strategic(state, current)
        problem = self.today_problem(state)
        today = problem.solve()
        ladder = FundingLadder(state.universe, data.settings.ladder)
        table = ladder.table(problem.ladder, today.weights, pv) if ladder.active else pd.DataFrame()
        diag = GoalOptimizer(state.universe, state.sigma, state.optimizer).diagnose()
        notes = list(problem.notes)
        for r, label in ((strategic, "strategic"), (today, "today")):
            if r.fallback:
                notes.append(f"{label}: {r.fallback}")
            notes += [f"{label} audit: {m}" for m in r.audit_failures]
            notes += [
                f"{label}: ladder bucket {s.bucket} needs {s.need:.2%} but policy "
                f"allows {s.granted:.2%} - shortfall not pre-funded"
                for s in r.ladder_shortfalls
            ]
        return AllocationResult(state, current, pv, strategic, today, problem.ladder, table, diag, notes)
