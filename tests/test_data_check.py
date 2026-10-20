"""Input data check (examples/data_check.py) and semi-liquid (ELTIF) sleeves end to end."""

import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from saa_pm import PointInTimeAllocator, load_mandate, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "examples", "data")
sys.path.insert(0, os.path.join(ROOT, "examples"))

from data_check import FAIL, PASS, check_inputs, stress_test_inputs  # noqa: E402


@pytest.fixture
def folder(tmp_path):
    for f in os.listdir(DATA):
        if f.endswith(".csv"):
            shutil.copy(os.path.join(DATA, f), tmp_path)
    return tmp_path


def _edit_assets(folder, fn):
    p = os.path.join(folder, "assets.csv")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    fn(df)
    df.to_csv(p, index=False)


def test_example_data_reconciles_without_failures():
    rep = check_inputs(DATA)
    assert rep.ok, rep.text()
    t = rep.table
    for check in (
        "values reconcile with the model",
        "returns reconcile (file levels -> model returns)",
        "total reconciles",
        "every position reconciles",
        "groups reconcile",
        "values reconcile",
        "pacing reconciles",
        "rows and total reconcile",
    ):
        rows = t[t["check"] == check]
        assert len(rows) and (rows["status"] == PASS).all(), (check, rows)


def test_fault_injection_every_case_behaves_as_expected():
    res = stress_test_inputs(DATA)
    bad = res[res["result"] != "PASS"]
    assert bad.empty, bad.to_string()
    assert {"variant", "fault"} <= set(res["kind"])


def test_missing_required_file_fails_early(folder):
    os.remove(os.path.join(folder, "holdings.csv"))
    rep = check_inputs(str(folder))
    assert not rep.ok and rep.data is None
    assert (rep.table["status"] == FAIL).any()


def _gate(df, name, limit):
    if "redemption_limit" not in df:
        df["redemption_limit"] = ""
    df.loc[df["name"] == name, "redemption_limit"] = str(limit)


def test_redemption_limit_column_loads(folder):
    _edit_assets(folder, lambda df: _gate(df, "US Nasdaq Eq", 0.25))
    data = load_mandate(str(folder))
    a = data.universe["US Nasdaq Eq"]
    assert a.redemption_limit == pytest.approx(0.25) and a.semi_liquid
    assert data.universe.semi_liquid_names == ["US Nasdaq Eq"]
    assert validate(data).ok


def test_invalid_redemption_limits_are_reported(folder):
    _edit_assets(folder, lambda df: (_gate(df, "US Nasdaq Eq", -0.1), _gate(df, "Cash", 0.5)))
    rep = validate(load_mandate(str(folder)))
    assert any("redemption_limit" in e and "US Nasdaq Eq" in e for e in rep.errors)
    assert any("gated (semi-liquid) asset 'Cash'" in e for e in rep.errors)


def test_redemption_limit_on_closed_end_sleeve_warns(folder):
    _edit_assets(folder, lambda df: _gate(df, "Priv Eq", 0.05))
    rep = validate(load_mandate(str(folder)))
    assert any("ignored for an illiquid" in w for w in rep.warnings)


def test_today_portfolio_respects_the_gate(folder):
    """US Nasdaq Eq is held at 8% and its strategic weight is 0%: with a 25% gate
    today's portfolio may sell at most a quarter of it."""
    _edit_assets(folder, lambda df: _gate(df, "US Nasdaq Eq", 0.25))
    data = load_mandate(str(folder))
    alloc = PointInTimeAllocator(data).run()
    i = data.universe.index("US Nasdaq Eq")
    w0 = data.holdings.weights[i]
    assert alloc.today.weights[i] >= 0.75 * w0 - 1e-6
    audit = alloc.today.audit.set_index("check")
    assert bool(audit.loc["redemption gates", "pass"])
    # the strategic target is a destination, not a trade: the gate does not bind it
    assert alloc.strategic.weights[i] < 0.75 * w0


def test_gated_sleeve_counts_as_its_own_liquidity_tier(folder):
    from saa_pm.analytics.liquidity import liquidity_profile

    _edit_assets(folder, lambda df: _gate(df, "US Nasdaq Eq", 0.25))
    u = load_mandate(str(folder)).universe
    w = pd.DataFrame([np.full(u.n, 1.0 / u.n)], columns=u.names, index=["eq"])
    lp = liquidity_profile(w, u).iloc[0]
    assert lp["semi_liquid"] == pytest.approx(1.0 / u.n)
    assert lp["ladder"] + lp["other_liquid"] + lp["semi_liquid"] + lp["illiquid"] == pytest.approx(1.0)
