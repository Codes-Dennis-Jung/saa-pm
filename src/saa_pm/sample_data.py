"""Synthetic example mandate - every input file, fully reproducible from a seed.

The data are SIMULATED (no market data is shipped): monthly log returns for 24
asset classes and 5 stress factors are drawn jointly from a multivariate normal
whose volatilities and correlations are rounded, plausible long-run values
(vols to 0.5%, correlations to 0.05). Each asset is drifted to its CMA, three
stylised sell-offs are injected through the equity beta of every series, and
the private sleeves report quarterly, appraisal-smoothed NAVs (so the
desmoothing and mixed-frequency logic is exercised).

    saa-pm sample-data examples/data
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# fmt: off
# name, class, exp_return, income, liquid, lb, ub, appraisal, ladder, tx_bps, freq, vol (annual)
_ASSETS = [
    ("Cash",         "Liquidity",       0.037, 0.037,  True,  0.01, 0.28, False, 1,    0,  "M", 0.005),
    ("US STR",       "Liquidity",       0.038, 0.038,  True,  0.00, 0.24, False, 2,    1,  "M", 0.015),
    ("US Govt",      "Rates",           0.047, 0.038,  True,  0.03, 0.25, False, None, 2,  "M", 0.045),
    ("UK Govt",      "Rates",           0.0455, 0.0435, True, 0.00, 0.08, False, None, 3,  "M", 0.025),
    ("JP Govt",      "Rates",           0.034, 0.025,  True,  0.00, 0.08, False, None, 3,  "M", 0.025),
    ("EU Govt",      "Rates",           0.050, 0.0335, True,  0.00, 0.10, False, None, 3,  "M", 0.050),
    ("EM HC",        "Credit",          0.060, 0.060,  True,  0.03, 0.08, False, None, 20, "M", 0.085),
    ("US IG",        "Credit",          0.0535, 0.0545, True, 0.03, 0.22, False, 3,    4,  "M", 0.065),
    ("EU IG",        "Credit",          0.054, 0.0365, True,  0.02, 0.10, False, None, 5,  "M", 0.045),
    ("EU HY",        "Credit",          0.051, 0.059,  True,  0.00, 0.05, False, None, 15, "M", 0.065),
    ("US HY",        "Credit",          0.0635, 0.0725, True, 0.00, 0.08, False, None, 15, "M", 0.070),
    ("US Dvd Eq",    "Equities",        0.081, 0.0245, True,  0.03, 0.15, False, None, 3,  "M", 0.140),
    ("US EW Eq",     "Equities",        0.077, 0.0175, True,  0.00, 0.15, False, None, 3,  "M", 0.155),
    ("US Nasdaq Eq", "Equities",        0.0985, 0.006, True,  0.00, 0.12, False, None, 3,  "M", 0.175),
    ("EU ex UK Eq",  "Equities",        0.0715, 0.0285, True, 0.02, 0.12, False, None, 5,  "M", 0.160),
    ("EU Dvd Eq",    "Equities",        0.059, 0.043,  True,  0.02, 0.12, False, None, 5,  "M", 0.155),
    ("UK Eq",        "Equities",        0.0835, 0.039, True,  0.02, 0.06, False, None, 5,  "M", 0.150),
    ("JP Eq",        "Equities",        0.085, 0.020,  True,  0.02, 0.06, False, None, 5,  "M", 0.150),
    ("Pac ex JP Eq", "Equities",        0.0745, 0.0355, True, 0.01, 0.08, False, None, 8,  "M", 0.175),
    ("EM Eq",        "Equities",        0.088, 0.025,  True,  0.03, 0.10, False, None, 15, "M", 0.170),
    ("Gold",         "Real Assets",     0.056, 0.000,  True,  0.03, 0.07, False, None, 5,  "M", 0.155),
    ("Priv Infra",   "Private Markets", 0.104, 0.0555, False, 0.02, 0.15, True,  None, 0,  "Q", 0.085),
    ("Priv Eq",      "Private Markets", 0.1135, 0.022, False, 0.00, 0.15, True,  None, 0,  "Q", 0.210),
    ("Priv Debt",    "Private Markets", 0.079, 0.075,  False, 0.02, 0.14, True,  None, 0,  "Q", 0.050),
]
# stress factors: name, annual drift (log), vol (annual)
_FACTORS = [
    ("Global_Equity", 0.070,  0.140),
    ("EUR_USD",       0.000,  0.075),
    ("Oil_Brent",     0.020,  0.440),
    ("Global_Govt",   0.030,  0.035),
    ("Global_Credit", 0.040,  0.055),
]
# correlation of [assets..., factors...] in the order above: lower triangle, row by row
_CORR_LT = """
+0.35
+0.10 +0.85
+0.15 +0.65 +0.65
 0.00 +0.35 +0.50 +0.30
+0.05 +0.60 +0.70 +0.65 +0.50
+0.15 +0.50 +0.45 +0.35 +0.35 +0.55
+0.10 +0.70 +0.75 +0.50 +0.45 +0.75 +0.80
+0.10 +0.55 +0.50 +0.55 +0.40 +0.80 +0.75 +0.80
+0.10 +0.25 +0.10 +0.20 +0.20 +0.45 +0.75 +0.70 +0.85
+0.05 +0.35 +0.25 +0.25 +0.25 +0.45 +0.80 +0.75 +0.75 +0.90
-0.05 +0.20 +0.15 +0.25 +0.20 +0.35 +0.65 +0.55 +0.55 +0.65 +0.70
-0.05 +0.15 +0.10 +0.20 +0.15 +0.35 +0.65 +0.55 +0.60 +0.75 +0.80 +0.95
+0.05 +0.20 +0.20 +0.20 +0.15 +0.40 +0.60 +0.55 +0.60 +0.65 +0.70 +0.65 +0.75
+0.05 +0.30 +0.20 +0.20 +0.10 +0.30 +0.75 +0.55 +0.55 +0.75 +0.75 +0.80 +0.85 +0.70
+0.05 +0.25 +0.10 +0.20 +0.05 +0.25 +0.70 +0.50 +0.50 +0.70 +0.70 +0.80 +0.80 +0.55 +0.95
+0.05 +0.20 +0.05 +0.15  0.00 +0.20 +0.65 +0.45 +0.50 +0.70 +0.70 +0.75 +0.80 +0.60 +0.90 +0.95
+0.05 -0.05 -0.20  0.00 -0.05 +0.10 +0.30 +0.15 +0.25 +0.50 +0.45 +0.55 +0.60 +0.60 +0.60 +0.50 +0.55
+0.05 +0.25 +0.15 +0.20 +0.15 +0.30 +0.70 +0.60 +0.55 +0.70 +0.75 +0.75 +0.80 +0.60 +0.80 +0.80 +0.80 +0.50
+0.10 +0.30 +0.20 +0.20 +0.10 +0.25 +0.75 +0.55 +0.50 +0.60 +0.65 +0.60 +0.65 +0.65 +0.75 +0.70 +0.70 +0.50 +0.80
+0.20 +0.40 +0.40 +0.30 +0.25 +0.25 +0.35 +0.35 +0.25 +0.10 +0.20 +0.10 +0.05 +0.05 +0.20 +0.20 +0.20 -0.15 +0.25 +0.35
+0.05 +0.15 +0.10 +0.15 +0.10 +0.15 +0.45 +0.35 +0.30 +0.35 +0.45 +0.45 +0.40 +0.30 +0.45 +0.45 +0.45 +0.15 +0.50 +0.40 +0.30
 0.00 +0.20 +0.10 +0.20 +0.10 +0.35 +0.70 +0.60 +0.65 +0.80 +0.85 +0.80 +0.90 +0.70 +0.80 +0.75 +0.75 +0.55 +0.75 +0.60  0.00 +0.40
+0.10 +0.05 -0.15  0.00 +0.10 +0.20 +0.55 +0.45 +0.60 +0.85 +0.80 +0.55 +0.65 +0.45 +0.55 +0.55 +0.55 +0.40 +0.55 +0.50  0.00 +0.35 +0.70
+0.05 +0.25 +0.15 +0.25 +0.15 +0.35 +0.75 +0.60 +0.65 +0.75 +0.85 +0.85 +0.95 +0.90 +0.90 +0.80 +0.85 +0.65 +0.85 +0.80 +0.15 +0.45 +0.85 +0.60
-0.10 -0.35 -0.25 -0.10 -0.05 -0.05 -0.55 -0.35 -0.10 -0.20 -0.30 -0.30 -0.30 -0.25 -0.55 -0.55 -0.45 -0.05 -0.45 -0.50 -0.40 -0.30 -0.30 -0.10 -0.40
 0.00 -0.15 -0.35 -0.20 -0.15 -0.20 +0.25 +0.05 +0.10 +0.40 +0.45 +0.30 +0.35 +0.15 +0.30 +0.40 +0.40 +0.25 +0.30 +0.25 -0.05 +0.30 +0.40 +0.60 +0.30 -0.15
+0.10 +0.75 +0.90 +0.70 +0.65 +0.90 +0.55 +0.80 +0.70 +0.30 +0.35 +0.25 +0.20 +0.30 +0.25 +0.20 +0.10 -0.05 +0.25 +0.25 +0.35 +0.15 +0.20 +0.05 +0.25 -0.15 -0.30
+0.10 +0.70 +0.70 +0.55 +0.45 +0.75 +0.85 +0.95 +0.90 +0.75 +0.75 +0.55 +0.60 +0.55 +0.60 +0.55 +0.50 +0.15 +0.60 +0.55 +0.35 +0.35 +0.60 +0.50 +0.60 -0.30 +0.05 +0.80
"""
_SMOOTH = 0.45          # appraisal smoothing of private NAVs (AR(1) weight on the past)
_EPISODES = [(0.30, -0.15), (0.55, -0.08), (0.80, -0.18)]  # (position in sample, Global_Equity move over 3m)
_EPISODE_SHAPE = np.array([0.7, 0.6, -0.3])  # month-by-month share of the move (sharp fall, partial rebound)
# fmt: on


def _correlation() -> np.ndarray:
    rows = [r.split() for r in _CORR_LT.strip("\n").split("\n")]
    n = len(rows) + 1
    C = np.eye(n)
    for i, r in enumerate(rows, start=1):
        assert len(r) == i, f"_CORR_LT row {i} has {len(r)} entries"
        C[i, :i] = C[:i, i] = [float(x) for x in r]
    # rounding breaks positive definiteness: clip eigenvalues, rescale to unit diagonal
    w, V = np.linalg.eigh(C)
    C = (V * np.maximum(w, 1e-3)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def _simulate(seed: int, start: str, end: str):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="ME")
    n = len(dates)
    names = [a[0] for a in _ASSETS] + [f[0] for f in _FACTORS]
    vol = np.array([a[-1] for a in _ASSETS] + [f[2] for f in _FACTORS])
    C = _correlation()
    assert C.shape[0] == len(names)
    cov = np.outer(vol, vol) * C / 12.0
    X = rng.multivariate_normal(np.zeros(len(names)), cov, size=n)
    # stylised sell-offs: a 3-month equity drawdown, every series moves by its beta
    eq = names.index("Global_Equity")
    beta = cov[:, eq] / cov[eq, eq]
    for m, move in _EPISODES:
        i = int(m * n)
        X[i : i + 3] += np.outer(_EPISODE_SHAPE, beta) * move
    X = pd.DataFrame(X, index=dates, columns=names)
    drift = {a[0]: np.log1p(a[2]) for a in _ASSETS} | {f[0]: f[1] for f in _FACTORS}
    for c in names:  # hit the CMA (assets) or the factor drift on average
        x = X[c]
        X[c] = x - x.mean() + drift[c] / 12 - 0.5 * x.var()
    R = np.expm1(X[[a[0] for a in _ASSETS]])
    true = R.copy()
    for name, *_r in _ASSETS:
        if not _r[3]:  # illiquid -> appraisal smoothing
            s = R[name].values.copy()
            for t in range(1, n):
                s[t] = (1 - _SMOOTH) * s[t] + _SMOOTH * s[t - 1]
            R[name] = s
    return dates, R, true, X[[f[0] for f in _FACTORS]]


def write_sample_data(
    folder: str, seed: int = 42, start: str = "2011-01-31", end: str = "2026-09-30"
) -> None:
    os.makedirs(folder, exist_ok=True)
    dates, R, _, F = _simulate(seed, start, end)
    rng = np.random.default_rng(seed + 1)

    def w(name, df):
        return df.to_csv(os.path.join(folder, name), index=False)

    # assets.csv
    w(
        "assets.csv",
        pd.DataFrame(
            [
                dict(
                    name=a[0],
                    group=a[1],
                    exp_return=a[2],
                    income=a[3],
                    liquid=a[4],
                    lb=a[5],
                    ub=a[6],
                    appraisal_based=a[7],
                    max_trade="",
                    ladder_bucket="" if a[8] is None else a[8],
                    tx_cost_bps=a[9],
                    nav_frequency=a[10],
                    price_column=a[0],
                )
                for a in _ASSETS
            ]
        ),
    )

    # ---- prices.csv: month-end total-return levels (quarterly for privates) --
    lvl = 100 * (1 + R).cumprod()
    first = lvl.iloc[[0]].copy()
    first.iloc[:, :] = 100.0
    first.index = [dates[0] - pd.offsets.MonthEnd(1)]
    lvl = pd.concat([first, lvl])
    for a in _ASSETS:
        if a[10] == "Q":  # only quarter-ends reported
            lvl.loc[~lvl.index.month.isin([3, 6, 9, 12]), a[0]] = np.nan
    out = lvl.round(4)
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    w("prices.csv", out)

    # groups.csv
    groups = [
        ("Liquidity", "Cash|US STR", 0.02, 0.40),
        ("Rates", "US Govt|UK Govt|JP Govt|EU Govt", 0.08, 0.32),
        ("Non-USD govt", "UK Govt|JP Govt|EU Govt", 0.0, 0.20),
        ("Credit", "US IG|EU IG|EU HY|US HY|EM HC", 0.12, 0.32),
        ("Investment grade credit", "US IG|EU IG", 0.06, 0.30),
        ("Sub-investment grade", "EU HY|US HY|EM HC", 0.05, 0.15),
        (
            "Equities",
            "US Dvd Eq|US EW Eq|US Nasdaq Eq|EU ex UK Eq|EU Dvd Eq|UK Eq|JP Eq|Pac ex JP Eq|EM Eq",
            0.25,
            0.60,
        ),
        ("US equities", "US Dvd Eq|US EW Eq|US Nasdaq Eq", 0.10, 0.28),
        ("Non-US equities", "EU ex UK Eq|EU Dvd Eq|UK Eq|JP Eq|Pac ex JP Eq|EM Eq", 0.12, 0.32),
        ("EM exposure", "EM HC|EM Eq", 0.06, 0.18),
        ("Real Assets", "Gold", 0.03, 0.07),
        ("Private Markets", "Priv Eq|Priv Debt|Priv Infra", 0.06, 0.32),
        ("Income assets", "EM HC|US HY|EU HY|Priv Debt|US Dvd Eq|EU Dvd Eq|UK Eq", 0.18, 0.45),
    ]
    w("groups.csv", pd.DataFrame(groups, columns=["name", "members", "lo", "hi"]))

    # pacing.csv
    pacing = [
        ("Priv Eq", 10, 4, "0.25|0.50|0.75|1.00", 2.8, "", 0.90),
        ("Priv Infra", 10, 3, "0.40|0.60|1.00", 2.0, "", 0.90),
        ("Priv Debt", 7, 2, "0.50|1.00", 1.2, "", 0.85),
    ]
    cols = [
        "asset",
        "life_years",
        "commitment_period",
        "contribution_rates",
        "bow_factor",
        "yield_rate",
        "drawdown_ratio",
    ]
    w("pacing.csv", pd.DataFrame(pacing, columns=cols))

    # holdings.csv: USD 100m, a legacy private-equity NAV, undrawn book
    hold = {
        "Cash": 40e6,
        "US Govt": 36e6,
        "US Dvd Eq": 7e6,
        "US Nasdaq Eq": 8e6,
        "Gold": 2e6,
        "Priv Eq": 7e6,
    }
    undrawn = {"Priv Eq": 10e6, "Priv Infra": 8e6, "Priv Debt": 12e6}
    w(
        "holdings.csv",
        pd.DataFrame(
            [
                dict(asset=a[0], market_value=hold.get(a[0], 0.0), commitments_undrawn=undrawn.get(a[0], ""))
                for a in _ASSETS
            ]
        ),
    )

    # ---- commitment_future.csv: the KNOWN call schedule of the undrawn book --
    asof = dates[-1]
    calls = []
    plan = {
        "Priv Eq": [0.25, 0.35, 0.25, 0.15],
        "Priv Infra": [0.40, 0.35, 0.25],
        "Priv Debt": [0.50, 0.50],
    }
    for asset, shares in plan.items():
        for k, s in enumerate(shares):
            d = (asof + pd.DateOffset(months=6 + 12 * k)) + pd.offsets.QuarterEnd(0)
            calls.append(dict(date=f"{d:%Y-%m-%d}", asset=asset, amount=round(s * undrawn[asset])))
    w("commitment_future.csv", pd.DataFrame(calls).sort_values(["date", "asset"]))

    # commitments_actual.csv: historical calls replayed by the backtest
    hist_calls = []
    for k, y in enumerate(range(dates[36].year + 1, asof.year)):
        hist_calls.append(dict(date=f"{y}-06-30", asset="Priv Eq", amount=1.5e6))
        if k % 2 == 0:
            hist_calls.append(dict(date=f"{y}-09-30", asset="Priv Infra", amount=2.0e6))
        if k % 3 == 1:
            hist_calls.append(dict(date=f"{y}-03-31", asset="Priv Debt", amount=2.5e6))
    w("commitments_actual.csv", pd.DataFrame(hist_calls))

    # config.csv
    cfg = [
        ("cpi", 0.0325, "inflation assumption for the return goal"),
        ("return_spread", 0.030, "required real return over CPI"),
        ("target_income", 0.035, "minimum expected cash income"),
        ("vol_lo", 0.06, "lower edge of the risk band"),
        ("vol_hi", 0.08, "upper edge of the risk band"),
        ("vol_steps", 13, "grid points across the risk band"),
        ("turnover_limit", 0.30, "one-way turnover cap per rebalance (0 = none)"),
        ("hard_return_floor", 0, "1 makes the return goal a hard constraint"),
        ("ladder_edge_1", 2, "calls due within 2y are pre-funded in ladder bucket 1"),
        ("ladder_edge_2", 3, "calls due in 2-3y in bucket 2; later calls in bucket 3"),
        ("ladder_cover", 1.0, "multiple of scheduled calls to pre-fund"),
        ("halflife", 36, "EWMA half-life in months"),
        ("shrink", 0.25, "constant-correlation shrinkage"),
        ("uplift", 0.15, "extra volatility for illiquid sleeves"),
        ("max_lookback", 120, "estimation window cap in months"),
        ("burn_in_months", 48, "history used before the first backtest decision"),
        ("rebalance_months", 3, "liquid rebalance frequency in the backtest"),
        ("review_months", 12, "strategic review frequency in the backtest"),
        ("bt_vol_steps", 7, "grid points inside the backtest"),
        ("forced_sale_extra_bps", 50, "penalty on a forced (distressed) sale"),
        ("goal_return_window_months", 60, "rolling window for the return-goal hit rate"),
    ]
    w("config.csv", pd.DataFrame(cfg, columns=["key", "value", "description"]))

    # cma_history.csv: annual published CMA vintages
    rows = []
    for y in range(dates[0].year + 1, asof.year + 1):
        for a in _ASSETS:
            rows.append(
                dict(
                    effective_date=f"{y - 1}-12-31",
                    asset=a[0],
                    exp_return=round(a[2] + rng.normal(0, 0.006), 4),
                    income=round(max(0.0, a[3] + rng.normal(0, 0.003)), 4),
                )
            )
    w("cma_history.csv", pd.DataFrame(rows))

    # policy_history.csv
    y0 = dates[0].year
    w(
        "policy_history.csv",
        pd.DataFrame(
            [
                dict(effective_date=f"{y0 + 4}-12-31", key="target_income", value=0.030),
                dict(effective_date=f"{y0 + 9}-03-31", key="vol_hi", value=0.09),
                dict(effective_date=f"{y0 + 11}-12-31", key="vol_hi", value=0.08),
                dict(effective_date=f"{y0 + 12}-12-31", key="target_income", value=0.035),
            ]
        ),
    )

    # flows.csv
    w(
        "flows.csv",
        pd.DataFrame([dict(date=f"{y0 + 7}-06-30", amount=5e6), dict(date=f"{y0 + 11}-03-31", amount=-3e6)]),
    )

    # stress factors / scenarios
    lv = (100 * np.exp(F.cumsum())).round(4)
    lv.insert(0, "date", lv.index.strftime("%Y-%m-%d"))
    w("stress_factors.csv", lv)
    ev = [int(m * len(dates)) for m, _ in _EPISODES]
    sc = [
        dict(name="Equities -10%", Global_Equity=-0.10),
        dict(name="Equities +10%", Global_Equity=0.10),
        dict(name="EUR -10% vs USD", EUR_USD=-0.10),
        dict(name="EUR +10% vs USD", EUR_USD=0.10),
        dict(name="Oil price shock +50%", Oil_Brent=0.50),
        dict(name="Rates shock (govt bonds -8%)", Global_Govt=-0.08),
        dict(name="Credit spread widening", Global_Credit=-0.06),
        dict(name="Stagflation", Global_Equity=-0.15, Global_Govt=-0.06, Oil_Brent=0.40),
    ]
    for k, i in enumerate(ev, start=1):
        sc.append(
            dict(
                name=f"Historical episode {k} (simulated)",
                start=f"{dates[i - 1]:%Y-%m-%d}",
                end=f"{dates[i + 2]:%Y-%m-%d}",
            )
        )
    cols = ["name", "start", "end"] + [f[0] for f in _FACTORS]
    w("stress_scenarios.csv", pd.DataFrame(sc).reindex(columns=cols))

    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Example input folder (synthetic)\n\nAll numbers are SIMULATED (seed "
            f"{seed}) from rounded, plausible volatilities and correlations - no market\n"
            "data. Regenerate with `saa-pm sample-data <folder>`. See the main README for "
            "the meaning\nof every file and column.\n"
        )
