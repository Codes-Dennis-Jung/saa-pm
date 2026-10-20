"""Unit tests for the numerical building blocks."""

import numpy as np
import pandas as pd
import pytest

from saa_pm.analytics.performance import diversification_ratio, max_drawdown, risk_contributions
from saa_pm.analytics.private_markets import xirr
from saa_pm.config import OptimizerConfig, Settings
from saa_pm.engine.portfolio_state import PortfolioState
from saa_pm.io.csvio import parse_dates, read_csv, to_float
from saa_pm.optimizer.goal_optimizer import GoalOptimizer
from saa_pm.privates.book import PrivateBook
from saa_pm.privates.ladder import FundingLadder, ladder_bucket, years_between
from saa_pm.privates.pacing import PacingParams, implied_irr, single_vintage_profile
from saa_pm.risk.covariance import (
    aggregate_returns,
    desmooth_returns,
    ewma_covariance,
    nearest_psd,
    shrink_constant_correlation,
)
from saa_pm.universe import Asset, GroupConstraint, Universe


def test_german_and_us_csv_parse_identically(tmp_path):
    de = tmp_path / "de.csv"
    us = tmp_path / "us.csv"
    de.write_text("date;a;b\n31.01.2020;1.114,07;3,5%\n29.02.2020;1.120,00;4,0%\n", encoding="utf-8")
    us.write_text("date,a,b\n2020-01-31,1114.07,3.5%\n2020-02-29,1120.00,4.0%\n", encoding="utf-8")
    d, u = read_csv(str(de)), read_csv(str(us))
    assert [to_float(x) for x in d["a"]] == [to_float(x) for x in u["a"]] == [1114.07, 1120.0]
    assert to_float(d["b"].iloc[0]) == pytest.approx(0.035)
    assert list(parse_dates(d["date"], "x")) == list(parse_dates(u["date"], "x"))


def test_day_first_dates_not_swapped():
    s = parse_dates(pd.Series(["01.02.2020", "01.03.2020", "01.04.2020"]), "x")
    assert list(s.dt.month) == [2, 3, 4]


def test_blank_config_value_keeps_default():
    s = Settings.from_mapping({"vol_steps": 15})
    assert s.optimizer.vol_steps == 15 and s.optimizer.turnover_limit == 0.15


def test_turnover_zero_means_no_cap():
    assert Settings.from_mapping({"turnover_limit": 0}).optimizer.turnover_limit is None


def test_geltner_recovers_true_series():
    rng = np.random.default_rng(0)
    true = pd.Series(rng.normal(0.01, 0.04, 400))
    phi = 0.5
    obs = true.copy()
    for t in range(1, len(obs)):
        obs.iloc[t] = (1 - phi) * true.iloc[t] + phi * obs.iloc[t - 1]
    rec, _ = desmooth_returns(obs, phi=phi)
    assert np.allclose(rec.iloc[1:].values, true.iloc[1:].values, atol=1e-12)


def test_ewma_equals_sample_cov_for_long_halflife():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(0, 0.02, (300, 3)), columns=list("abc"))
    S, _ = ewma_covariance(X, halflife=1e9, periods_per_year=1)
    assert np.allclose(S.values, X.cov().values, rtol=1e-6)


def test_shrinkage_keeps_variances_and_psd():
    A = pd.DataFrame(
        [[0.04, 0.019, 0.0], [0.019, 0.01, 0.002], [0.0, 0.002, 0.09]], index=list("abc"), columns=list("abc")
    )
    S = nearest_psd(shrink_constant_correlation(A, 0.3))
    assert np.allclose(np.diag(S), np.diag(A), atol=1e-9)
    assert np.linalg.eigvalsh(S.values).min() > 0


def test_aggregation_aligned_to_window_end():
    idx = pd.date_range("2020-01-31", periods=7, freq="ME")
    r = pd.DataFrame({"x": [0.01] * 7}, index=idx)
    q = aggregate_returns(r, 3)
    assert list(q.index) == [idx[3], idx[6]]
    assert q["x"].iloc[0] == pytest.approx(1.01**3 - 1)


P = PacingParams(
    "PE",
    life_years=10,
    commitment_period=4,
    contribution_rates=(0.25, 0.5, 0.75, 1.0),
    bow_factor=2.5,
    growth_rate=0.10,
    yield_rate=0.02,
    drawdown_ratio=0.9,
)


def test_vintage_irr_equals_growth_rate():
    assert implied_irr(P) == pytest.approx(P.growth_rate, abs=1e-6)


def test_commitment_fully_drawn():
    prof = single_vintage_profile(P)
    assert prof["call"].sum() == pytest.approx(P.drawdown_ratio)
    assert P.problems(0.10) == []


def test_monthly_book_pays_full_annual_ta_rate():
    """A year of monthly payouts must distribute the annual TA fraction."""
    p = PacingParams(
        "X",
        life_years=10,
        commitment_period=2,
        contribution_rates=(0.5, 1.0),
        bow_factor=1.0,
        growth_rate=0.0,
        yield_rate=0.0,
    )
    b = PrivateBook("X", p)
    b.call_now(100.0)
    b.cohorts[0].age_m = 48  # start of year 5: RD = 0.5
    paid = sum(b.step_month(0.0)[1] for _ in range(12))
    assert paid == pytest.approx(50.0, rel=1e-9)


def test_xirr():
    d = pd.to_datetime(["2021-01-01", "2022-01-01"])  # 365 days (ACT/365)
    assert xirr([-100, 110], d) == pytest.approx(0.10, abs=1e-4)


def test_ladder_buckets_on_exact_anniversary():
    t = years_between(pd.Timestamp("2026-03-31"), ["2028-03-31", "2029-03-31", "2026-09-30"])
    assert list(ladder_bucket(t, (2.0, 3.0))) == [2, 3, 1]


def _small_universe():
    assets = [
        Asset("Cash", "Liquidity", 0.03, 0.03, ladder_bucket=1, ub=0.5),
        Asset("Bond", "Rates", 0.045, 0.035, ub=0.8),
        Asset("Equity", "Equities", 0.08, 0.02, ub=0.7),
        Asset("PE", "Private", 0.11, 0.02, liquid=False, ub=0.2),
    ]
    groups = [GroupConstraint("Growth", ("Equity", "PE"), 0.2, 0.75)]
    S = pd.DataFrame(
        np.diag([0.0001, 0.0036, 0.0256, 0.0625]),
        index=[a.name for a in assets],
        columns=[a.name for a in assets],
    )
    S.loc["Equity", "PE"] = S.loc["PE", "Equity"] = 0.6 * 0.16 * 0.25
    return Universe(assets, groups), S


def test_ladder_requirement_and_floor():
    u, _ = _small_universe()
    lad = FundingLadder(u, Settings.from_mapping({}, n_ladder_buckets=1).ladder)
    sched = pd.DataFrame(
        dict(date=pd.to_datetime(["2027-01-31", "2025-01-31"]), asset=["PE", "PE"], amount=[10.0, 99.0])
    )
    req = lad.requirement(sched, pd.Timestamp("2026-01-31"), 100.0)
    assert req.need.tolist() == [0.10]  # past call ignored


def test_ladder_needs_currency_portfolio_value():
    u, _ = _small_universe()
    lad = FundingLadder(u, Settings.from_mapping({}, n_ladder_buckets=1).ladder)
    with pytest.raises(ValueError):
        lad.requirement(
            pd.DataFrame(dict(date=["2027-01-31"], asset=["PE"], amount=[1.0])),
            pd.Timestamp("2026-01-31"),
            float("nan"),
        )


def test_goal_order_and_audit():
    u, S = _small_universe()
    cfg = OptimizerConfig(target_return=0.06, target_income=0.025, vol_lo=0.06, vol_hi=0.10, vol_steps=9)
    res = GoalOptimizer(u, S, cfg).run()
    sol = res.solution
    assert abs(sol.weights.sum() - 1) < 1e-7
    assert sol.exp_return >= cfg.target_return - 1e-5
    assert res.audit["pass"].all(), res.audit_failures
    assert sol.vol >= cfg.vol_lo - 1e-4  # band floor honoured


def test_band_floor_preferred_when_frontier_is_flat():
    """When return is capped by bounds, the selection must not pick a point
    below the band floor if one inside the band exists."""
    u, S = _small_universe()
    cfg = OptimizerConfig(target_return=0.05, target_income=0.0, vol_lo=0.05, vol_hi=0.12, vol_steps=8)
    sol = GoalOptimizer(u, S, cfg).run().solution
    assert sol.vol >= cfg.vol_lo - 1e-4


def test_infeasible_bounds_are_reported():
    u, S = _small_universe()
    bad = u.updated({"Cash": {"lb": 0.6}, "Bond": {"lb": 0.6}})
    assert any("lower bounds" in p for p in bad.structural_problems())


def _gated_universe(limit=0.25):
    u, S = _small_universe()
    eltif = Asset("ELTIF", "Private", 0.09, 0.05, liquid=True, ub=0.4, redemption_limit=limit)
    assets = [a for a in u if a.name != "PE"] + [eltif]
    S = S.rename(index={"PE": "ELTIF"}, columns={"PE": "ELTIF"})
    return Universe(assets, [GroupConstraint("Growth", ("Equity", "ELTIF"), 0.2, 0.75)]), S


def test_semi_liquid_sleeve_buys_freely_but_sells_through_the_gate():
    """An evergreen / ELTIF sleeve: subscriptions unrestricted, redemptions
    capped at redemption_limit x holding per rebalance."""
    from saa_pm.optimizer.constraints import REBALANCE, ConstraintSpec

    u, _ = _gated_universe(0.25)
    cfg = OptimizerConfig(turnover_limit=None)
    import cvxpy as cp

    def extreme(w_prev, sense):
        w = cp.Variable(u.n)
        spec = ConstraintSpec(u, cfg, w_prev=np.array(w_prev), mode=REBALANCE)
        obj = cp.Maximize(w[3]) if sense == "max" else cp.Minimize(w[3])
        cp.Problem(obj, spec.build(w)).solve()
        return float(w.value[3]), spec

    hi, _ = extreme([0.5, 0.3, 0.2, 0.0], "max")
    assert hi == pytest.approx(0.4, abs=1e-6)  # bought up to ub in one step
    lo, spec = extreme([0.2, 0.2, 0.2, 0.4], "min")
    assert lo == pytest.approx(0.3, abs=1e-6)  # only 25% of the 40% holding redeemed
    assert u.semi_liquid_names == ["ELTIF"] and not u.fully_liquid_mask[3]
    no_ladder = Universe([eltif_first := Asset("E", "P", 0.09, 0.05, redemption_limit=0.1)] + list(u)[1:], [])
    assert no_ladder.sweep_asset != eltif_first.name  # cash is never swept into a gated fund


def test_semi_liquid_sleeve_gets_the_illiquidity_uplift_and_no_forced_sale():
    from saa_pm.risk.covariance import illiquidity_uplift

    u, S = _gated_universe()
    up = illiquidity_uplift(S, u.fully_liquid_mask, 0.15)
    assert up.loc["ELTIF", "ELTIF"] == pytest.approx(S.loc["ELTIF", "ELTIF"] * 1.15**2)
    assert up.loc["Bond", "Bond"] == pytest.approx(S.loc["Bond", "Bond"])
    pf = PortfolioState.from_weights(u, np.array([0.0, 0.1, 0.1, 0.8]), 100.0, {}, {})
    res = pf.fund(30.0, ["Cash"])
    assert pf.value["ELTIF"] == pytest.approx(80.0)  # never sold to meet a call
    assert res.shortfall == pytest.approx(10.0)


def test_pinned_privates_plus_liquid_floors_widen_mixed_group_cap():
    """Pinned private NAV that fits a group cap on its own, but not together
    with the liquid members' lower bounds, must widen the cap, not make the
    solve infeasible."""
    u = Universe(
        [
            Asset("Cash", "L", 0.03, 0.03, True, 0.0, 1.0),
            Asset("Gold", "R", 0.05, 0.0, True, 0.03, 0.1),
            Asset("Infra", "R", 0.09, 0.05, False, 0.0, 0.3),
        ],
        [GroupConstraint("Real", ("Gold", "Infra"), 0.0, 0.20), GroupConstraint("Liq", ("Cash",), 0.0, 1.0)],
    )
    pinned, breaches = u.pin_privates(np.array([0.8, 0.0, 0.19]))
    assert [b.group for b in breaches] == ["Real"]
    assert breaches[0].weight == pytest.approx(0.22)
    assert not pinned.structural_problems()


def test_unfunded_call_creates_no_value():
    """A call the liquid book cannot pay becomes an overdraft."""
    u, _ = _small_universe()
    p = PacingParams("PE", 10, 4, (0.25, 0.5, 0.75, 1.0), 2.5, 0.1, 0.0)
    pf = PortfolioState.from_weights(u, np.array([0.1, 0.0, 0.0, 0.9]), 100.0, {"PE": p}, {})
    pv0 = pf.pv
    pf.books["PE"].call_now(50.0)
    pf.sync_private_values()
    res = pf.fund(50.0, ["Cash"])
    assert res.shortfall == pytest.approx(40.0)
    assert pf.pv == pytest.approx(pv0)  # value conserved


def test_drawdown_counts_first_month():
    assert max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_euler_contributions_sum_to_vol():
    _, S = _small_universe()
    w = np.array([0.1, 0.4, 0.4, 0.1])
    rc = risk_contributions(w, S)
    assert rc["ccr"].sum() == pytest.approx(np.sqrt(w @ S.values @ w))
    assert rc["pct_of_risk"].sum() == pytest.approx(1.0)
    assert diversification_ratio(w, S) >= 1.0
