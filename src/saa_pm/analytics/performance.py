"""Performance and risk measures.

Conventions: inputs are MONTHLY simple returns; returns annualise
geometrically (``**12/n``), volatilities with ``sqrt(12)``. Drawdowns are
anchored at 1.0 BEFORE the first return, so a loss in the first month counts.
Covariance arguments are ANNUALISED.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

_EPS = 1e-12
ArrayLike = Union[np.ndarray, pd.Series, list]


def _w(w: ArrayLike, sigma: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Weights as an array; a labelled Series is aligned to sigma's labels."""
    if isinstance(w, pd.Series) and isinstance(sigma, pd.DataFrame):
        return w.reindex(sigma.index).fillna(0.0).values.astype(float)
    return np.asarray(w, dtype=float).ravel()


def portfolio_vol(w: ArrayLike, sigma) -> float:
    w = _w(w, sigma)
    S = np.asarray(sigma, float)
    return float(np.sqrt(max(w @ S @ w, 0.0)))


def risk_contributions(w: ArrayLike, sigma: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition: marginal (MCR), absolute (CCR, sums to vol) and
    percentage (PCR, sums to 1) contributions."""
    ww = _w(w, sigma)
    S = np.asarray(sigma, float)
    vol = portfolio_vol(ww, S)
    mcr = (S @ ww) / vol if vol > _EPS else np.zeros_like(ww)
    ccr = ww * mcr
    idx = list(sigma.index) if isinstance(sigma, pd.DataFrame) else list(range(len(ww)))
    return pd.DataFrame(
        {
            "weight": ww,
            "vol": np.sqrt(np.diag(S)),
            "mcr": mcr,
            "ccr": ccr,
            "pct_of_risk": ccr / vol if vol > _EPS else 0.0,
        },
        index=idx,
    )


def diversification_ratio(w: ArrayLike, sigma) -> float:
    """sum_i w_i sigma_i / portfolio vol (>= 1 for long-only)."""
    ww = _w(w, sigma)
    S = np.asarray(sigma, float)
    vol = portfolio_vol(ww, S)
    return float((ww * np.sqrt(np.diag(S))).sum() / vol) if vol > _EPS else np.nan


def effective_n(shares: ArrayLike) -> float:
    """1 / HHI of non-negative shares (renormalised)."""
    s = np.clip(np.asarray(shares, float), 0, None)
    tot = s.sum()
    return float(1.0 / np.sum((s / tot) ** 2)) if tot > _EPS else np.nan


def _clean(r) -> pd.Series:
    return pd.Series(r, dtype=float).dropna()


def ann_return(r) -> float:
    r = _clean(r)
    return float((1 + r).prod() ** (12 / len(r)) - 1) if len(r) else np.nan


def ann_vol(r) -> float:
    r = _clean(r)
    return float(r.std() * np.sqrt(12)) if len(r) > 1 else np.nan


def sharpe_ratio(r, rf=0.0) -> float:
    """Annualised mean excess return / annualised excess vol. ``rf`` is either
    an annual rate or a monthly series aligned to ``r``."""
    r = _clean(r)
    rf_m = (
        pd.Series(rf, dtype=float).reindex(r.index).fillna(0.0)
        if isinstance(rf, pd.Series)
        else (1 + float(rf)) ** (1 / 12) - 1
    )
    ex = r - rf_m
    v = ex.std() * np.sqrt(12)
    return float(ex.mean() * 12 / v) if len(ex) > 1 and v > _EPS else np.nan


def cumulative(r) -> pd.Series:
    """Growth of 1, anchored at 1.0 one period before the first return."""
    r = pd.Series(r, dtype=float).fillna(0.0)
    return (1 + r).cumprod()


def drawdown_series(r) -> pd.Series:
    cum = cumulative(r)
    peak = np.maximum(cum.cummax(), 1.0)  # the starting capital is a peak
    return cum / peak - 1.0


def max_drawdown(r) -> float:
    dd = drawdown_series(r)
    return float(dd.min()) if len(dd) else np.nan


def rolling_ann_return(r, months: int = 60) -> pd.Series:
    r = pd.Series(r, dtype=float)
    return (1 + r).rolling(months).apply(np.prod, raw=True) ** (12 / months) - 1


def rolling_ann_vol(r, months: int = 36) -> pd.Series:
    return pd.Series(r, dtype=float).rolling(months).std() * np.sqrt(12)


def tracking_error(r, bench) -> float:
    a = (pd.Series(r, dtype=float) - pd.Series(bench, dtype=float)).dropna()
    return float(a.std() * np.sqrt(12)) if len(a) > 1 else np.nan


def information_ratio(r, bench) -> float:
    a = (pd.Series(r, dtype=float) - pd.Series(bench, dtype=float)).dropna()
    te = tracking_error(r, bench)
    return float(a.mean() * 12 / te) if len(a) > 1 and te > _EPS else np.nan


def beta(r, bench) -> float:
    df = pd.concat([pd.Series(r, dtype=float), pd.Series(bench, dtype=float)], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    v = df.iloc[:, 1].var()
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / v) if v > _EPS else np.nan


def historical_var(r, level: float = 0.95) -> float:
    r = _clean(r)
    return float(-np.percentile(r, 100 * (1 - level))) if len(r) >= 12 else np.nan


def historical_cvar(r, level: float = 0.95) -> float:
    r = _clean(r)
    if len(r) < 12:
        return np.nan
    cut = np.percentile(r, 100 * (1 - level))
    tail = r[r <= cut]
    return float(-tail.mean()) if len(tail) else np.nan


def performance_table(returns: pd.DataFrame, rf=0.0) -> pd.DataFrame:
    """Standard measures for every column of a monthly return frame."""
    rows = {}
    for c in returns.columns:
        r = returns[c]
        rows[c] = dict(
            ann_return=ann_return(r),
            ann_vol=ann_vol(r),
            sharpe=sharpe_ratio(r, rf),
            max_drawdown=max_drawdown(r),
            var_95_monthly=historical_var(r),
            cvar_95_monthly=historical_cvar(r),
            worst_month=float(_clean(r).min()),
            best_month=float(_clean(r).max()),
            months=int(_clean(r).size),
        )
    return pd.DataFrame(rows).T


def backtest_metrics(result, vol_lo: float, vol_hi: float, window: int = 60) -> pd.Series:
    """Goal-oriented scorecard of one backtest run. Targets in force each month
    (policy history) are used when the run recorded them."""
    m = result.monthly
    r = m["ret"]
    n_years = len(r) / 12.0
    target = m["target_return"] if "target_return" in m else pd.Series(np.nan, index=m.index)
    roll_vol = rolling_ann_vol(r, 36)
    lo = m["vol_lo"] if "vol_lo" in m else vol_lo
    hi = m["vol_hi"] if "vol_hi" in m else vol_hi
    valid = roll_vol.notna()
    in_band = ((roll_vol >= lo) & (roll_vol <= hi))[valid]
    roll_ret = rolling_ann_return(r, window)
    hit = (roll_ret >= target)[roll_ret.notna()]
    ann = ann_return(r)
    ev = result.events

    def count(pred):
        return int(pred(ev["kind"]).sum()) if len(ev) else 0

    return pd.Series(
        {
            "ann_return": ann,
            "excess_vs_target": ann - float(target.mean()) if target.notna().any() else np.nan,
            "ann_vol": ann_vol(r),
            "return_per_vol": ann / ann_vol(r) if ann_vol(r) > 0 else np.nan,
            "max_drawdown": max_drawdown(r),
            "realised_income": float(m["income"].sum() / n_years),
            "share_36m_vol_in_band": float(in_band.mean()) if len(in_band) else np.nan,
            f"share_{window}m_windows_hitting_target": float(hit.mean()) if len(hit) else np.nan,
            "turnover_pa": float(m["turnover"].sum() / n_years),
            "costs_pa": float(m["tx_cost"].sum() / n_years),
            "illiquid_min": float(m["illiquid"].min()),
            "illiquid_max": float(m["illiquid"].max()),
            "ladder_min": float(m["ladder_held"].min()),
            "forced_sales": count(lambda k: k.str.startswith("forced_sale")),
            "forced_amount": float(m["forced"].sum()),
            "unfunded_calls": count(lambda k: k == "call_unfunded"),
            "group_breaches": count(lambda k: k.str.startswith("group_breach")),
            "infeasible_solves": count(lambda k: k.str.contains("infeasible")),
            "ladder_underfunded": count(lambda k: k.str.startswith("ladder_underfunded")),
        },
        name=result.label,
    )
