"""Forward transition plan: the optimal allocation at each capital-call date.

Question: "Starting from today's portfolio and paying the calls in
commitment_future.csv on their dates, how does the optimal allocation look
at each call date, and which assets fund each call?"

Model, stated plainly:
* the MARKET state is frozen at the last closed month (CMAs, covariance,
  policy, targets); only the BOOK evolves;
* between call dates, month by month, liquid assets grow at their CMA total
  return and called private capital grows at its CMA and distributes on the
  Takahashi-Alexander bow; distributions are swept into ladder bucket 1;
* on a call date the calls are paid down the ladder, then pro rata from the
  rest of the liquid book (anything left is an overdraft and reported), the
  called capital becomes private NAV, privates are pinned, ladder floors are
  re-sized from the REMAINING schedule and the liquid book is re-optimised.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import pandas as pd

from ..io.loaders import MandateData
from ..optimizer.constraints import REBALANCE, STRATEGIC
from ..optimizer.goal_optimizer import GoalOptimizer, InfeasibleError, OptimizationResult, clamp_group_floors
from ..privates.book import monthly_growth
from ..privates.ladder import FundingLadder
from .portfolio_state import PortfolioState
from .state import DecisionState, StateBuilder


@dataclass
class ProjectionResult:
    state: DecisionState
    allocations: pd.DataFrame  # date x (assets + exp_return/income/vol/rule)
    fund_plan: pd.DataFrame  # long: date, asset, flow_ccy, kind
    private_book: pd.DataFrame  # per date: NAV per sleeve, calls, distributions
    ladder: pd.DataFrame  # per date x bucket: need vs held
    notes: List[str]

    @property
    def weights(self) -> pd.DataFrame:
        return self.allocations[self.state.universe.names]

    def funding_table(self, scale: float = 1e6) -> pd.DataFrame:
        """Wide sources (-) and uses (+) per asset and date, in ``scale`` units."""
        if self.fund_plan.empty:
            return pd.DataFrame()
        t = (
            self.fund_plan.pivot_table(
                index="asset", columns="date", values="flow_ccy", aggfunc="sum"
            ).fillna(0.0)
            / scale
        )
        order = [n for n in self.state.universe.names if n in t.index]
        return t.loc[order]


class CallDateProjector:
    def __init__(self, data: MandateData, builder: Optional[StateBuilder] = None):
        self.data = data
        self.builder = builder or StateBuilder(data)

    def _solve(
        self, state: DecisionState, pf: PortfolioState, floors, mode: str, label: str, notes: List[str]
    ) -> OptimizationResult:
        w_now = pf.weights()
        pinned, breaches = state.universe.pin_privates(w_now)
        pinned, gnotes = clamp_group_floors(pinned, state.optimizer)
        notes += [f"{label}: {m}" for m in gnotes]
        notes += [
            f"{label}: private NAV breaches '{b.group}' ({b.weight:.2%} > {b.cap:.2%})" for b in breaches
        ]
        try:
            res = GoalOptimizer(pinned, state.sigma, state.optimizer, floors, w_now, mode).run(
                allow_fallback=True
            )
        except InfeasibleError:
            # the private book has outgrown the liquid lower bounds: scale them
            # to the space that is left - loudly - and retry once
            priv = float(w_now[~pinned.liquid_mask].sum())
            lb_liq = float(pinned.lb[pinned.liquid_mask].sum())
            room = max(0.0, 1.0 - priv)
            if not (lb_liq > room - 1e-6 and lb_liq > 0):
                raise
            k = max(0.0, (room - 1e-4) / lb_liq)
            notes.append(
                f"{label}: pinned privates {priv:.2%} + liquid lower bounds "
                f"{lb_liq:.2%} exceed 100% - liquid lower bounds scaled x{k:.2f}"
            )
            scaled = pinned.updated({a.name: {"lb": a.lb * k} for a in pinned if a.liquid})
            res = GoalOptimizer(scaled, state.sigma, state.optimizer, floors, w_now, mode).run(
                allow_fallback=True
            )
        if res.fallback:
            notes.append(f"{label}: {res.fallback}")
        notes += [
            f"{label}: ladder bucket {s.bucket} needs {s.need:.2%}, policy allows {s.granted:.2%}"
            for s in res.ladder_shortfalls
        ]
        return res

    def run(
        self,
        schedule: Optional[pd.DataFrame] = None,
        planning_schedule: Optional[pd.DataFrame] = None,
        ladder_cover: Optional[float] = None,
        distributions: bool = True,
    ) -> ProjectionResult:
        """``schedule``: calls actually PAID (default commitment_future.csv);
        ``planning_schedule``: what the ladder is sized on (default = schedule;
        a different one models a surprise); ``distributions=False`` delays all
        private distributions (stress)."""
        data = self.data
        state = self.builder.build()
        u = state.universe
        ladder_cfg = (
            data.settings.ladder
            if ladder_cover is None
            else replace(data.settings.ladder, cover=ladder_cover)
        )
        ladder = FundingLadder(u, ladder_cfg)
        if not ladder.active:
            raise ValueError(
                "no asset has a ladder_bucket - the projection needs the "
                "funding ladder (e.g. Cash=1, T-bills=2, IG=3)"
            )
        sched = (data.schedule if schedule is None else schedule).copy()
        sched["date"] = pd.to_datetime(sched["date"])
        plan = sched.copy() if planning_schedule is None else planning_schedule.copy()
        plan["date"] = pd.to_datetime(plan["date"])
        sched = sched[sched["date"] > state.asof]
        if sched.empty:
            raise ValueError("no future capital calls in commitment_future.csv")
        missing = sorted(set(sched["asset"]) - set(state.pacing))
        if missing:
            raise ValueError(f"scheduled sleeves without pacing rows: {missing}")

        notes: List[str] = []
        pv0 = data.holdings.total
        pf = PortfolioState.from_weights(u, data.holdings.weights, pv0, state.pacing, {})
        ladder_order, sweep = u.ladder_order, u.sweep_asset
        alloc_rows, fund_rows, book_rows, ladder_rows = [], [], [], []

        def record(label, res: OptimizationResult, req, extra: Optional[dict] = None):
            w = res.weights
            alloc_rows.append(
                dict(
                    date=label,
                    **dict(zip(u.names, w)),
                    exp_return=res.solution.exp_return,
                    income=res.solution.income,
                    vol=res.solution.vol,
                    rule=res.solution.selection_rule,
                )
            )
            for row in ladder.table(req, w, pf.pv).to_dict("records"):
                ladder_rows.append(dict(date=label, **row))
            if extra is not None:
                book_rows.append(
                    dict(
                        date=label,
                        portfolio_value=pf.pv,
                        **extra,
                        **{f"{n}_nav": bk.nav for n, bk in pf.books.items()},
                    )
                )

        def rebalance(label, kind, res):
            before = dict(pf.value)
            pf.trade_to(res.weights)
            for n in u.liquid_names:
                d = pf.value[n] - before[n]
                if abs(d) > 1e-6 * pv0:
                    fund_rows.append(dict(date=label, asset=n, flow_ccy=d, kind=kind))

        # t0: today, trading rules of the mandate apply
        req = ladder.requirement(plan, state.asof, pf.pv)
        lab0 = f"{state.asof:%Y-%m-%d}"
        res = self._solve(state, pf, req.floors, REBALANCE, "today", notes)
        rebalance(lab0, "initial_rebalance", res)
        record(lab0, res, req)

        # roll from call date to call date
        mu = {a.name: monthly_growth(a.exp_return) for a in u}
        g_priv = {n: monthly_growth(p.growth_rate) for n, p in state.pacing.items()}
        prev = state.asof
        for cdate in sorted(sched["date"].unique()):
            cdate = pd.Timestamp(cdate)
            months = max(1, (cdate.year - prev.year) * 12 + cdate.month - prev.month)
            dists = 0.0
            for _ in range(months):
                for n in u.liquid_names:
                    pf.value[n] *= 1.0 + mu[n]
                for n, bk in pf.books.items():
                    _, d = bk.step_month(g_priv[n], distribute=distributions)
                    dists += d
                pf.sync_private_values()
            pf.value[sweep] += dists
            lab = f"{cdate:%Y-%m-%d}"
            if dists > 0:
                fund_rows.append(dict(date=lab, asset=sweep, flow_ccy=dists, kind="distributions"))

            due = sched[sched["date"] == cdate]
            calls = float(due["amount"].sum())
            before = dict(pf.value)
            fr = pf.fund(calls, ladder_order)
            for n in u.liquid_names:
                d = pf.value[n] - before[n]
                if abs(d) > 1e-6 * pv0:
                    fund_rows.append(dict(date=lab, asset=n, flow_ccy=d, kind="call_funding"))
            if fr.shortfall > 1e-6:
                notes.append(
                    f"{lab}: liquid book exhausted - {fr.shortfall:,.0f} of calls unfunded (overdraft)"
                )
            for r in due.itertuples():
                pf.books[r.asset].call_now(float(r.amount))
                fund_rows.append(dict(date=lab, asset=r.asset, flow_ccy=float(r.amount), kind="capital_call"))
            pf.sync_private_values()

            req = ladder.requirement(plan[plan["date"] > cdate], cdate, pf.pv)
            res = self._solve(state, pf, req.floors, STRATEGIC, lab, notes)
            rebalance(lab, "rebalance", res)
            remaining = sched[sched["date"] > cdate]
            record(
                lab,
                res,
                req,
                dict(
                    calls_paid=calls,
                    calls_unfunded=fr.shortfall,
                    distributions=dists,
                    **{
                        f"{n}_scheduled_remaining": float(
                            remaining.loc[remaining["asset"] == n, "amount"].sum()
                        )
                        for n in pf.books
                    },
                ),
            )
            prev = cdate

        alloc = pd.DataFrame(alloc_rows).set_index("date")
        return ProjectionResult(
            state,
            alloc,
            pd.DataFrame(fund_rows),
            pd.DataFrame(book_rows).set_index("date"),
            pd.DataFrame(ladder_rows),
            notes,
        )
