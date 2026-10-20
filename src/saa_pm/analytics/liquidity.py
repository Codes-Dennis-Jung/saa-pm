"""Liquidity profile and income decomposition.

Liquidity tiers per portfolio: the funding ladder (pre-funding assets),
other liquid assets, semi-liquid (gated, e.g. ELTIF) sleeves and illiquid
private sleeves. Coverage compares the
ladder with the capital calls due in the next 1 / 2 / 3 years measured from
EACH portfolio's own date and value (a future portfolio is not judged
against calls that have already been paid).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from ..universe import Universe


def liquidity_profile(
    weights: pd.DataFrame,
    universe: Universe,
    schedule: Optional[pd.DataFrame] = None,
    portfolio_values: Optional[Mapping[str, float]] = None,
    dates: Optional[Mapping[str, pd.Timestamp]] = None,
    cover: float = 1.0,
) -> pd.DataFrame:
    """``weights``: portfolio x asset. ``dates`` / ``portfolio_values``: per
    portfolio as-of date and currency value (needed for call coverage)."""
    ladder = set(universe.ladder_order)
    liq = universe.liquid_mask
    rows = {}
    for p, w in weights.iterrows():
        w = w[universe.names].values.astype(float)
        lad = float(sum(w[universe.index(n)] for n in ladder))
        semi = float(sum(w[universe.index(n)] for n in universe.semi_liquid_names))
        rec = dict(
            ladder=lad,
            other_liquid=float(w[liq].sum()) - lad - semi,
            semi_liquid=semi,
            illiquid=float(w[~liq].sum()),
        )
        if schedule is not None and len(schedule) and dates and portfolio_values:
            d0, pv = pd.Timestamp(dates[p]), float(portfolio_values[p])
            s = schedule.assign(date=pd.to_datetime(schedule["date"]))
            for y in (1, 2, 3):
                due = s[(s["date"] > d0) & (s["date"] <= d0 + pd.DateOffset(years=y))]
                need = cover * float(due["amount"].sum()) / pv
                rec[f"calls_{y}y"] = need
                rec[f"ladder_cover_{y}y"] = lad / need if need > 1e-9 else np.nan
                rec[f"liquid_cover_{y}y"] = (lad + rec["other_liquid"]) / need if need > 1e-9 else np.nan
        rows[p] = rec
    return pd.DataFrame(rows).T


def income_decomposition(
    weights: pd.DataFrame, universe: Universe, target_income: Optional[float] = None
) -> pd.DataFrame:
    """Expected cash income (weight x income CMA) by asset class per portfolio."""
    inc = pd.Series(universe.income, index=universe.names)
    cls = pd.Series({a.name: a.group for a in universe})
    contrib = weights[universe.names].mul(inc, axis=1)
    out = contrib.T.groupby(cls).sum().T
    out["total"] = out.sum(axis=1)
    if target_income is not None:
        out["target"] = target_income
        out["gap"] = out["total"] - target_income
    return out
