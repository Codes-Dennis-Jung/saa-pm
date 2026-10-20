"""End-to-end tests on the synthetic sample mandate."""

import numpy as np
import pandas as pd
import pytest

from saa_pm import (
    BacktestSuite,
    CallDateProjector,
    PointInTimeAllocator,
    SAAStudy,
    StateBuilder,
    load_mandate,
    validate,
)
from saa_pm.sample_data import write_sample_data


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sample")
    write_sample_data(str(d), seed=42)
    return str(d)


@pytest.fixture(scope="module")
def data(data_dir):
    return load_mandate(data_dir)


def test_sample_data_validates(data):
    rep = validate(data)
    assert rep.ok, rep.errors


def test_holdings_with_unknown_asset_rejected(data_dir, tmp_path):
    import shutil

    d = tmp_path / "bad"
    shutil.copytree(data_dir, d)
    h = pd.read_csv(d / "holdings.csv")
    h.loc[len(h)] = ["Privat Equity", 1e6, ""]  # typo must not vanish from PV
    h.to_csv(d / "holdings.csv", index=False)
    with pytest.raises(ValueError, match="not in assets.csv"):
        load_mandate(str(d))


def test_state_is_a_pure_function_of_t(data):
    """Point-in-time and walk-forward see identical inputs at the same date."""
    b = StateBuilder(data)
    direct = b.build(len(b) - 1)
    for t in range(60, len(b) - 1, 12):
        b.build(t)
    again = b.build(len(b) - 1)
    assert np.array_equal(direct.sigma.values, again.sigma.values)
    assert np.array_equal(direct.universe.mu, again.universe.mu)


def test_allocation_meets_goals_and_ladder(data):
    res = PointInTimeAllocator(data).run()
    cfg = res.state.optimizer
    s = res.strategic.solution
    assert s.exp_return >= cfg.target_return - 1e-5
    assert cfg.vol_lo - 1e-4 <= s.vol <= cfg.vol_hi + 1e-5
    held = res.ladder_table["held_pct_pv"].values
    need = res.ladder_table["need_pct_pv"].values
    assert (held >= need - 1e-6).all()  # calls pre-funded
    assert abs(res.today.weights.sum() - 1) < 1e-6


def test_projection_starts_from_todays_solution(data):
    alloc = PointInTimeAllocator(data).run()
    proj = CallDateProjector(data).run()
    assert np.allclose(proj.weights.iloc[0].values, alloc.today.weights, atol=1e-6)
    assert (proj.private_book["calls_unfunded"] == 0).all()


def test_backtest_process_runs_and_is_consistent(data):
    runs = BacktestSuite(data).run(include=("process",))
    m = runs["process"].monthly
    assert len(m) == len(data.common_returns) - data.settings.backtest.burn_in_months
    assert np.isfinite(m["ret"]).all()
    pv_path = (1 + m["ret"]).cumprod() * data.holdings.total
    flows = m["flow"].sum()
    assert m["pv"].iloc[-1] == pytest.approx(pv_path.iloc[-1] + flows, rel=0.02)


def test_full_study_writes_results(data_dir, tmp_path):
    out = tmp_path / "results"
    res = SAAStudy(data_dir, str(out), verbose=False).run(
        steps=("allocation", "projection", "analyses"), mc_paths=500
    )
    assert (out / "saa_results.xlsx").exists()
    assert (out / "SUMMARY.md").exists()
    assert (out / "charts" / "allocation_by_class.png").exists()
    assert "risk/attribution_assets" in res.tables
    pm = res.tables["private_markets/performance_by_sleeve"].set_index("asset")
    cma = {a.name: a.exp_return for a in res.allocation.state.universe}
    for n, row in pm.iterrows():  # IRR of the TA model == CMA
        assert row["irr"] == pytest.approx(cma[n], abs=0.01)
