"""Walk-forward backtest of the allocation PROCESS, with benchmarks.

Every month after the burn-in:

1. at decision months build the as-of state (same builder as the point-in-time
   run: covariance from data before t, CMA vintage and policy in force);
2. mark liquid assets (income accrues as cash) and the private book (realised
   reported returns; Takahashi-Alexander rates only drive calls/distributions);
3. pay capital calls through the waterfall: cash -> ladder buckets in order ->
   forced pro-rata sale (costed) -> overdraft (recorded);
4. apply external flows;
5. annual strategic review: destination portfolio, and - if no historical
   call file exists - the top-up commitment that closes the private gap;
6. quarterly rebalance of the liquid book with privates pinned at NAV and
   ladder floors sized from the remaining calls.

Benchmarks run through the same mechanics with fixed liquid weights.
Three modelling choices drive the answer:
* private NAV is marked with REALISED returns, never with growth_rate;
* private weights are pinned at each rebalance (they cannot be traded);
* forced sales are counted and costed - a backtest in which the ladder is
  never stressed has not tested the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .._logging import logger
from ..analytics.performance import backtest_metrics
from ..io.loaders import MandateData
from ..optimizer.constraints import REBALANCE, STRATEGIC
from ..optimizer.goal_optimizer import GoalOptimizer, InfeasibleError, clamp_group_floors
from ..privates.ladder import FundingLadder
from ..privates.pacing import Vintage, simulate_pacing
from .portfolio_state import PortfolioState
from .state import DecisionState, StateBuilder


@dataclass
class BacktestResult:
    label: str
    monthly: pd.DataFrame
    weights: pd.DataFrame
    reviews: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def returns(self) -> pd.Series:
        return self.monthly["ret"]

    def metrics(self, vol_lo: float, vol_hi: float, window: int = 60) -> pd.Series:
        return backtest_metrics(self, vol_lo, vol_hi, window)


class WalkForwardBacktest:
    """One run. ``fixed_weights=None`` runs the full optimisation process;
    an array runs a buy-and-rebalance benchmark of the liquid book."""

    def __init__(
        self,
        data: MandateData,
        label: str = "process",
        fixed_weights: Optional[np.ndarray] = None,
        use_ladder: bool = True,
        builder: Optional[StateBuilder] = None,
        start_weights: Optional[np.ndarray] = None,
        replay_calls: bool = True,
        with_privates: bool = True,
        verbose: bool = False,
    ):
        self.data = data
        self.label = label
        self.fixed = None if fixed_weights is None else np.asarray(fixed_weights, float)
        self.use_ladder = use_ladder
        bt = data.settings.backtest
        self.builder = builder or StateBuilder(data, vol_steps=bt.vol_steps)
        self.w0 = data.holdings.weights if start_weights is None else np.asarray(start_weights, float)
        calls = data.panel.calls
        self.replay = replay_calls and calls is not None and len(calls) > 0
        self.with_privates = with_privates
        self.verbose = verbose

    def _projected_floors(
        self, pf: PortfolioState, state: DecisionState, date: pd.Timestamp, ladder: FundingLadder
    ):
        """Simulate mode: floors from what the book ITSELF will call - instalments
        already fixed this year plus the TA projection of every cohort."""
        rows = []
        horizon = int(np.ceil(ladder.config.edges[-1])) + 1 if ladder.config.edges else 1
        for n, bk in pf.books.items():
            pending = bk.pending_calls
            if pending > 0:
                rows.append(dict(date=date + pd.DateOffset(months=6), asset=n, amount=pending))
            legacy = [Vintage(c.nav, c.uncalled, c.age_m // 12) for c in bk.cohorts]
            path = simulate_pacing([], state.pacing[n], horizon=horizon, legacy=legacy)
            for yr, c in path["call"].items():
                if c > 0:
                    rows.append(
                        dict(date=date + pd.DateOffset(years=int(yr), months=-6), asset=n, amount=float(c))
                    )
        sched = pd.DataFrame(rows, columns=["date", "asset", "amount"])
        return ladder.requirement(sched, date, pf.pv)

    def run(self) -> BacktestResult:
        data = self.data
        bt = data.settings.backtest
        hist = self.builder.returns
        T = len(hist)
        if T <= bt.burn_in_months + 12:
            raise ValueError(f"{T} months of history but burn-in is {bt.burn_in_months}")
        u0 = data.universe
        names = u0.names
        ladder = FundingLadder(u0, data.settings.ladder)
        ladder_order = u0.ladder_order if (self.use_ladder and ladder.active) else []
        sweep_to = u0.sweep_asset
        pv0 = data.holdings.total

        undrawn = (
            {}
            if (self.replay or not self.with_privates)
            else {n: float(data.holdings.undrawn.get(n, 0.0)) for n in data.pacing}
        )
        state0 = self.builder.build(bt.burn_in_months)
        pf = PortfolioState.from_weights(u0, self.w0, pv0, state0.pacing, undrawn, bt.default_tx_cost_bps)
        sched = data.panel.calls if self.replay else None

        state: DecisionState = state0
        floors_req = None
        rows, reviews, events, wrows = [], [], [], []
        pv_prev = pf.pv

        def event(date, kind, amount=np.nan, cost=0.0):
            events.append(dict(date=date, kind=kind, amount=amount, cost=cost))

        for t in range(bt.burn_in_months, T):
            date = hist.index[t]
            m = t - bt.burn_in_months
            is_review = self.fixed is None and m > 0 and m % bt.review_months == 0
            is_rebal = m % bt.rebalance_months == 0

            # 0. as-of state (also for benchmarks: income / pacing in force)
            if is_review or is_rebal or m % 12 == 0:
                state = self.builder.build(t)
                for n, bk in pf.books.items():
                    bk.set_params(state.pacing[n])
            u = state.universe

            # 1. mark markets
            r = hist.iloc[t]
            income = 0.0
            for a in u:
                if a.liquid:
                    v = pf.value[a.name]
                    income += v * a.income / 12.0
                    pf.value[a.name] = v * (1.0 + r[a.name] - a.income / 12.0)
            calls = dists = 0.0
            for n, bk in pf.books.items():
                c, d = bk.step_month(float(r[n]))
                calls += c
                dists += d
            if sched is not None:
                due = sched[(sched["date"] > hist.index[t - 1]) & (sched["date"] <= date)]
                for row in due.itertuples():
                    pf.books[row.asset].call_now(float(row.amount))
                    calls += float(row.amount)
            pf.sync_private_values()
            pf.cash += income + dists

            # 2. fund calls
            fr = pf.fund(calls, ladder_order, bt.forced_sale_extra_bps)
            forced, cost_forced = fr.forced, fr.cost
            if fr.forced > 1e-6:
                event(date, "forced_sale", fr.forced / pv_prev, fr.cost / pv_prev)
            if fr.shortfall > 1e-6:
                event(date, "call_unfunded", fr.shortfall / pv_prev)
            pf.sweep_cash(sweep_to)

            # 3. external flows
            flow = data.panel.flow_in_month(date)
            if flow > 0:
                pf.value[sweep_to] += flow
            elif flow < 0:
                fo = pf.fund(-flow, ladder_order, bt.forced_sale_extra_bps)
                if fo.forced > 1e-6:
                    forced += fo.forced
                    cost_forced += fo.cost
                    event(date, "forced_sale_outflow", fo.forced / pv_prev, fo.cost / pv_prev)
                pf.sweep_cash(sweep_to)

            pv = pf.pv
            turnover = tx_cost = 0.0
            sel_rule, exp = "", dict(exp_return=np.nan, exp_vol=np.nan, exp_income=np.nan)

            # 4. strategic review (+ simulated commitments)
            if is_review:
                try:
                    res = GoalOptimizer(u, state.sigma, state.optimizer, (), pf.weights(), STRATEGIC).run(
                        allow_fallback=True
                    )
                    w_star = res.weights
                    commits = {}
                    if not self.replay:
                        for n, bk in pf.books.items():
                            exposure = (bk.nav + bk.uncalled) / pv
                            amt = (
                                max(0.0, w_star[u.index(n)] - exposure)
                                * pv
                                / max(state.pacing[n].drawdown_ratio, 1e-9)
                            )
                            bk.commit(amt)
                            commits[n] = amt / pv
                    if not any(s.risk_binding for s in res.scan):
                        event(date, "frontier_collapsed")
                    reviews.append(
                        dict(
                            date=date,
                            ok=True,
                            rule=res.solution.selection_rule,
                            target_illiquid=float(w_star[~u.liquid_mask].sum()),
                            **{f"commit_{n}": v for n, v in commits.items()},
                            cma_source=state.cma_source,
                            sigma_trace=state.sigma_trace,
                            mu_sum=state.mu_sum,
                            target_return=state.optimizer.target_return,
                            target_income=state.optimizer.target_income,
                        )
                    )
                except InfeasibleError as e:
                    reviews.append(dict(date=date, ok=False, error=str(e)[:120]))
                    event(date, "review_infeasible")

            # 5. ladder floors from the remaining calls
            if self.fixed is None and (is_review or is_rebal):
                if not ladder_order:
                    floors_req = None
                elif self.replay:
                    floors_req = ladder.requirement(sched, date, pv)
                else:
                    floors_req = self._projected_floors(pf, state, date, ladder)

            # 6. rebalance the liquid book
            if is_rebal:
                w_now = pf.weights()
                w_tgt = None
                if self.fixed is not None:
                    w_tgt = self.fixed.copy()
                    priv = ~u.liquid_mask
                    w_tgt[priv] = w_now[priv]
                    s = w_tgt[~priv].sum()
                    if s > 1e-9:
                        w_tgt[~priv] *= (1.0 - w_now[priv].sum()) / s
                else:
                    pinned, breaches = u.pin_privates(w_now)
                    pinned, _ = clamp_group_floors(pinned, state.optimizer)
                    for b in breaches:
                        event(date, f"group_breach:{b.group}", b.weight)
                    floors = floors_req.floors if floors_req is not None else ()
                    try:
                        res = GoalOptimizer(
                            pinned, state.sigma, state.optimizer, floors, w_now, REBALANCE
                        ).run(allow_fallback=True)
                        w_tgt = res.weights
                        sel_rule = res.solution.selection_rule
                        exp = dict(
                            exp_return=res.solution.exp_return,
                            exp_vol=res.solution.vol,
                            exp_income=res.solution.income,
                        )
                        if res.fallback:
                            event(date, "rebalance_relaxed", res.solution.vol)
                        for s in res.ladder_shortfalls:
                            event(date, f"ladder_underfunded:b{s.bucket}", s.need - s.granted)
                    except InfeasibleError:
                        event(date, "rebalance_infeasible")
                if w_tgt is not None:
                    turnover, tx_cost = pf.trade_to(w_tgt)
                    pv = pf.pv

            w_end = pf.weights()
            wrows.append(pd.Series(w_end, index=names, name=date))
            rows.append(
                dict(
                    date=date,
                    pv=pv,
                    ret=(pv - flow) / pv_prev - 1.0,
                    flow=flow,
                    income=income / pv_prev,
                    calls=calls / pv_prev,
                    dists=dists / pv_prev,
                    illiquid=float(w_end[~u.liquid_mask].sum()),
                    ladder_held=float(sum(w_end[u.index(n)] for n in ladder_order)),
                    turnover=turnover,
                    tx_cost=(tx_cost + cost_forced) / pv_prev,
                    forced=forced / pv_prev,
                    overdraft=min(pf.cash, 0.0) / pv,
                    review=is_review,
                    rebal=is_rebal,
                    rule=sel_rule,
                    target_return=state.optimizer.target_return,
                    target_income=state.optimizer.target_income,
                    vol_lo=state.optimizer.vol_lo,
                    vol_hi=state.optimizer.vol_hi,
                    **exp,
                )
            )
            pv_prev = pv

        wt = pd.DataFrame(wrows)
        wt.index.name = "date"

        def idx(rows_):
            return pd.DataFrame(rows_).set_index("date") if rows_ else pd.DataFrame()

        return BacktestResult(self.label, idx(rows), wt, idx(reviews), idx(events))


def sixty_forty_weights(data: MandateData) -> Optional[np.ndarray]:
    """60% in the first liquid equity sleeve, 40% in the first liquid rates/credit
    sleeve (by group name)."""
    u = data.universe
    eq = next((a.name for a in u if a.liquid and "equit" in a.group.lower()), None)
    bd = next(
        (a.name for a in u if a.liquid and a.group.lower() in ("rates", "credit", "bonds", "fixed income")),
        None,
    )
    if not (eq and bd):
        return None
    w = np.zeros(u.n)
    w[u.index(eq)], w[u.index(bd)] = 0.6, 0.4
    return w


class BacktestSuite:
    """Process run + ablation + benchmarks, and the comparison table."""

    def __init__(self, data: MandateData, verbose: bool = False):
        self.data = data
        self.builder = StateBuilder(data, vol_steps=data.settings.backtest.vol_steps)
        self.verbose = verbose

    def run(
        self, include: tuple = ("process", "no_ladder", "hold_current", "60_40")
    ) -> Dict[str, BacktestResult]:
        d, b = self.data, self.builder
        runs: Dict[str, BacktestResult] = {}
        specs = {
            "process": lambda: WalkForwardBacktest(d, "process", builder=b),
            "no_ladder": lambda: WalkForwardBacktest(d, "no_ladder", use_ladder=False, builder=b),
            "hold_current": lambda: WalkForwardBacktest(
                d, "hold_current", fixed_weights=d.holdings.weights, builder=b
            ),
        }
        w6040 = sixty_forty_weights(d)
        if w6040 is not None:
            specs["60_40"] = lambda: WalkForwardBacktest(
                d,
                "60_40",
                fixed_weights=w6040,
                start_weights=w6040,
                builder=b,
                replay_calls=False,
                with_privates=False,
            )
        for name in include:
            if name in specs:
                logger.info(f"  backtest: {name} ...")
                runs[name] = specs[name]().run()
        return runs

    def metrics(self, runs: Dict[str, BacktestResult]) -> pd.DataFrame:
        bt = self.data.settings.backtest
        g = self.data.settings.goals
        return pd.concat(
            [r.metrics(g.vol_lo, g.vol_hi, bt.target_window_months) for r in runs.values()], axis=1
        )
