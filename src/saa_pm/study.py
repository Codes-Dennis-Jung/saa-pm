"""SAAStudy - one object that runs the whole analysis and writes the results.

    from saa_pm import SAAStudy
    study = SAAStudy("examples/data", "results")
    results = study.run()            # allocation, projection, backtest, analyses
    results.allocation.weights()     # DataFrames for notebook use

Every step loads from the SAME validated data and the SAME as-of state, so the
strategic target, today's portfolio, the forward plan and every side analysis
describe one consistent picture.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._logging import enable_console_logging, logger
from .analytics import frontier as fr
from .analytics import liquidity as liq
from .analytics import private_markets as pm
from .analytics import risk as rk
from .analytics import sensitivity as sens
from .analytics import stress as st
from .analytics.performance import performance_table
from .analytics.wealth import simulate_wealth
from .engine.allocation import AllocationResult, PointInTimeAllocator
from .engine.backtest import BacktestResult, BacktestSuite
from .engine.projection import CallDateProjector, ProjectionResult
from .engine.state import StateBuilder
from .io.loaders import MandateData, load_mandate
from .io.validation import ValidationReport, validate
from .reporting.writer import ResultWriter


class _NoCharts:
    """Stand-in when matplotlib is not installed: every chart call is a no-op."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _charts_module():
    try:
        from .reporting import charts

        return charts
    except ImportError:
        logger.warning("matplotlib is not installed - charts skipped (pip install 'saa-pm[report]')")
        return _NoCharts()


ALL_STEPS = ("allocation", "projection", "backtest", "analyses")


@dataclass
class StudyResults:
    data: MandateData
    validation: ValidationReport
    allocation: Optional[AllocationResult] = None
    projection: Optional[ProjectionResult] = None
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)
    backtest_metrics: Optional[pd.DataFrame] = None
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    frontiers: Dict[str, pd.DataFrame] = field(default_factory=dict)
    charts: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class SAAStudy:
    def __init__(
        self,
        data_dir: str,
        out_dir: Optional[str] = "results",
        verbose: bool = True,
        input_type: str = "index",
    ):
        self.data_dir, self.out_dir, self.verbose = data_dir, out_dir, verbose
        if verbose:
            enable_console_logging()
        self.data = load_mandate(data_dir, input_type)
        self.validation = validate(self.data)
        self.builder = StateBuilder(self.data)

    def log(self, msg: str, level: int = logging.INFO) -> None:
        logger.log(level, msg)

    def portfolios(self, res: StudyResults) -> Dict[str, np.ndarray]:
        ports = {"current": self.data.holdings.weights}
        if res.allocation:
            ports["strategic"] = res.allocation.strategic.weights
            ports["today"] = res.allocation.today.weights
        if res.projection:
            for d, row in res.projection.weights.iloc[1:].iterrows():
                ports[f"plan_{d[:7]}"] = row.values.astype(float)
        return ports

    def run(
        self, steps: Sequence[str] = ALL_STEPS, write: bool = True, mc_paths: int = 10_000
    ) -> StudyResults:
        unknown = set(steps) - set(ALL_STEPS)
        if unknown:
            raise ValueError(f"unknown steps {sorted(unknown)}; choose from {ALL_STEPS}")
        d = self.data
        res = StudyResults(d, self.validation)
        self.log(
            f"Loaded {d.universe.n} assets, {len(d.common_returns)} months "
            f"({d.common_returns.index[0]:%Y-%m} to {d.asof:%Y-%m}), "
            f"portfolio {d.holdings.total:,.0f}"
        )
        if not self.validation.ok:
            self.log(self.validation.text())
            self.validation.raise_if_errors()
        for w in self.validation.warnings:
            self.log(f"  ! {w}", logging.WARNING)

        def timed(name, fn):
            t = time.time()
            out = fn()
            res.timings[name] = round(time.time() - t, 1)
            self.log(f"  {name:<28} {res.timings[name]:>6.1f}s")
            return out

        if "allocation" in steps:
            res.allocation = timed("allocation", lambda: PointInTimeAllocator(d, self.builder).run())
        if "projection" in steps and len(d.schedule) and d.universe.ladder_map:
            res.projection = timed("projection", lambda: CallDateProjector(d, self.builder).run())
        if "backtest" in steps:
            bt = d.settings.backtest
            if len(d.common_returns) > bt.burn_in_months + 24:
                suite = BacktestSuite(d)
                res.backtests = timed("backtest", suite.run)
                res.backtest_metrics = suite.metrics(res.backtests)
            else:
                self.log("  backtest skipped: history shorter than burn-in + 24 months")
        if "analyses" in steps:
            timed("analyses", lambda: self._analyses(res, mc_paths))
        if write and self.out_dir:
            timed("write results", lambda: self._write(res))
        return res

    def _analyses(self, res: StudyResults, mc_paths: int) -> None:
        d, T = self.data, res.tables
        state = res.allocation.state if res.allocation else self.builder.build()
        u, S = state.universe, state.sigma
        ports = self.portfolios(res)
        W = pd.DataFrame(ports, index=u.names).T

        att = rk.attribution(ports, S, u)
        T["risk/attribution_assets"] = att
        T["risk/attribution_classes"] = rk.attribution_by(att, u, "asset_class")
        T["risk/attribution_groups"] = rk.attribution_by(att, u, "groups")
        T["risk/concentration"] = rk.concentration(ports, S, u)
        T["risk/correlation"] = rk.correlation_of_record(S)
        T["risk/model_vs_realised_vol"] = rk.realised_vs_model_vol(S, d.common_returns)

        # performance of the fixed portfolios on history (pro-forma)
        R = d.common_returns[u.names]
        T["performance/proforma"] = performance_table(
            pd.DataFrame({p: R.values @ w for p, w in ports.items()}, index=R.index)
        )

        T["stress/historical_episodes"] = st.historical_episodes(
            R,
            {k: v for k, v in ports.items() if k in ("current", "strategic", "today")} or ports,
            reference="today" if "today" in ports else None,
        )
        if d.stress_factors is not None and d.stress_scenarios is not None:
            try:
                fs = st.FactorStress(u, d.common_returns, d.stress_factors)
                pnl, moves = fs.run(st.scenario_shocks(d.stress_scenarios, d.stress_factors), ports)
                T["stress/factor_scenarios_pnl"] = pnl
                T["stress/factor_scenarios_asset_moves"] = moves
            except ValueError as exc:
                self.log(f"  factor stress skipped: {exc}", logging.WARNING)

        fronts = fr.reference_frontiers(u, S, config=state.optimizer)
        for k, f in fronts.items():
            T[f"frontier/{k}"] = f
        T["frontier/gaps"] = fr.frontier_gaps(fronts, [round(v, 2) for v in np.arange(0.04, 0.121, 0.01)])
        res.frontiers = fronts

        summ, wts = sens.sensitivity(u, S, state.optimizer)
        T["sensitivity/summary"] = summ
        T["sensitivity/weights"] = wts
        if res.allocation:
            T["sensitivity/binding_strategic"] = sens.binding_constraints(res.allocation.strategic.weights, u)

        T["income/decomposition"] = liq.income_decomposition(W, u, state.optimizer.target_income)
        dates, pvs = {}, {}
        for p in ports:
            dates[p], pvs[p] = state.asof, d.holdings.total
        if res.projection is not None:
            book = res.projection.private_book
            for p in ports:
                if p.startswith("plan_"):
                    key = next((i for i in book.index if i.startswith(p[5:])), None)
                    if key is not None:
                        dates[p], pvs[p] = pd.Timestamp(key), float(book.loc[key, "portfolio_value"])
        T["liquidity/profile"] = liq.liquidity_profile(W, u, d.schedule, pvs, dates, d.settings.ladder.cover)

        if d.pacing and len(d.schedule):
            legacy = {n: float(d.holdings.market_value[n]) for n in d.pacing}
            cf = pm.project_cashflows(state.pacing, d.schedule, state.asof, legacy)
            if not cf.empty:
                T["private_markets/cashflows_monthly"] = cf
                T["private_markets/annual"] = pm.annual_summary(cf)
                T["private_markets/performance_by_sleeve"] = pm.performance(cf, ("asset",))
                T["private_markets/performance_total"] = pm.performance(cf, ())
                T["private_markets/j_curve"] = pm.j_curve(cf)
                T["private_markets/vintages"] = pm.vintage_diversification(cf)

        # wealth Monte Carlo on the projected allocation path
        wpath = (
            res.projection.weights.copy()
            if res.projection is not None
            else pd.DataFrame(
                [ports.get("today", ports["current"])], columns=u.names, index=[f"{state.asof:%Y-%m-%d}"]
            )
        )
        wpath.index = pd.to_datetime([str(i)[:10] for i in wpath.index])
        g = state.goals
        wp = simulate_wealth(
            u.mu,
            S.values,
            wpath,
            years=10,
            n_paths=mc_paths,
            start_value=d.holdings.total,
            cpi=g.cpi,
            target_return=state.optimizer.target_return,
            start=state.asof,
        )
        T["wealth/nominal"] = wp.percentiles
        T["wealth/real"] = wp.real_percentiles
        T["wealth/probabilities"] = wp.probabilities

    def _write(self, res: StudyResults) -> None:
        assert self.out_dir is not None
        w = ResultWriter(self.out_dir)
        charts = _charts_module()
        d = self.data
        w.table("00_inputs/validation_coverage", res.validation.coverage)
        w.table("00_inputs/settings", d.settings.describe())
        w.text("00_inputs/validation.txt", res.validation.text())
        classes = {a.name: a.group for a in d.universe}

        if res.allocation:
            a = res.allocation
            w.table("01_allocation/weights", a.weights())
            w.table("01_allocation/summary", a.summary())
            w.table("01_allocation/ladder", a.ladder_table)
            w.table("01_allocation/ladder_calls", a.ladder.detail)
            w.table("01_allocation/commitment_gap", a.commitment_gap())
            w.table("01_allocation/covariance", a.state.sigma)
            w.table("01_allocation/feasibility", pd.Series(a.diagnostics, name="value").to_frame())
            w.table("01_allocation/today_audit", a.today.audit)
            res.charts.append(
                charts.allocation_by_class(
                    a.weights(),
                    classes,
                    w.chart_path("allocation_by_class.png"),
                    "Current, strategic target and today's implementable portfolio",
                )
            )
        if res.projection is not None:
            p = res.projection
            w.table("02_projection/allocations", p.allocations)
            w.table("02_projection/fund_plan", p.fund_plan)
            w.table("02_projection/funding_table_millions", p.funding_table())
            w.table("02_projection/private_book", p.private_book)
            w.table("02_projection/ladder", p.ladder)
            res.charts.append(charts.transition_path(p.weights, classes, w.chart_path("transition_path.png")))
        if res.backtests:
            w.table("03_backtest/metrics", res.backtest_metrics)
            rets = pd.DataFrame({k: r.returns for k, r in res.backtests.items()})
            w.table("03_backtest/monthly_returns", rets)
            for k, r in res.backtests.items():
                w.table(f"03_backtest/{k}_monthly", r.monthly)
                w.table(f"03_backtest/{k}_weights", r.weights)
                if len(r.events):
                    w.table(f"03_backtest/{k}_events", r.events)
                if len(r.reviews):
                    w.table(f"03_backtest/{k}_reviews", r.reviews)
            res.charts.append(charts.backtest_growth(rets, w.chart_path("backtest.png")))
        for name, t in res.tables.items():
            sec = name.split("/")[0]
            w.table(f"04_{sec}/{name.split('/', 1)[1]}", t)

        T = res.tables
        if "risk/attribution_classes" in T and res.allocation:
            tc = T["risk/attribution_classes"]
            tc = tc[tc["portfolio"] == "strategic"].set_index("asset_class")
            res.charts.append(
                charts.risk_vs_weight(
                    tc,
                    w.chart_path("risk_attribution.png"),
                    "Strategic target: share of capital vs share of risk",
                )
            )
        if res.frontiers and res.allocation:
            a = res.allocation
            pts = {}
            for k in ("current", "strategic", "today"):
                ww = dict(current=a.current, strategic=a.strategic.weights, today=a.today.weights)[k]
                pts[k] = (float(np.sqrt(ww @ a.state.sigma.values @ ww)), float(a.state.universe.mu @ ww))
            cfg = a.state.optimizer
            res.charts.append(
                charts.frontiers(
                    res.frontiers,
                    pts,
                    w.chart_path("frontiers.png"),
                    cfg.target_return,
                    (cfg.vol_lo, cfg.vol_hi),
                )
            )
        if "private_markets/annual" in T:
            res.charts.append(
                charts.pm_cashflows(
                    T["private_markets/annual"], w.chart_path("private_markets_cashflows.png")
                )
            )
        if "wealth/nominal" in T:
            pv = d.holdings.total
            tr = (
                res.allocation.state.optimizer.target_return
                if res.allocation
                else d.settings.goals.target_return
            )
            yrs = T["wealth/nominal"].index
            res.charts.append(
                charts.wealth_fan(
                    T["wealth/nominal"],
                    w.chart_path("wealth.png"),
                    pv,
                    target_path=pd.Series(pv * (1 + tr) ** yrs, yrs),
                )
            )
        if "stress/factor_scenarios_pnl" in T and "today" in T["stress/factor_scenarios_pnl"]:
            res.charts.append(
                charts.stress_bars(
                    T["stress/factor_scenarios_pnl"]["today"],
                    w.chart_path("stress.png"),
                    "Factor stress scenarios: today's portfolio",
                )
            )
        res.charts = [c for c in res.charts if c]
        w.summary(res)
        w.workbook()
        self.log(f"  results written to {os.path.abspath(w.out)}")
