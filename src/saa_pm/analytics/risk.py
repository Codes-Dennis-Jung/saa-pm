"""Risk attribution, concentration and the correlation of record.

All measures use the risk model the optimiser actually saw (the as-of
covariance: aggregated, desmoothed, shrunk, illiquidity uplift), never raw
monthly returns.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

from ..universe import Universe
from .performance import diversification_ratio, effective_n, portfolio_vol, risk_contributions


def attribution(
    portfolios: Mapping[str, np.ndarray], sigma: pd.DataFrame, universe: Universe
) -> pd.DataFrame:
    """Long table: portfolio x asset with weight, vol, MCR, CCR, % of risk."""
    frames = []
    for name, w in portfolios.items():
        rc = risk_contributions(np.asarray(w, float), sigma.loc[universe.names, universe.names])
        rc.insert(0, "portfolio", name)
        rc["asset_class"] = [universe[n].group for n in rc.index]
        frames.append(rc.rename_axis("asset").reset_index())
    return pd.concat(frames, ignore_index=True)


def attribution_by(table: pd.DataFrame, universe: Universe, by: str = "asset_class") -> pd.DataFrame:
    """Aggregate an :func:`attribution` table to asset classes (non-overlapping,
    sums to 100%) or to the policy groups (``by='groups'``; groups may overlap)."""
    if by == "asset_class":
        g = table.groupby(["portfolio", "asset_class"])[["weight", "ccr", "pct_of_risk"]].sum()
        return g.reset_index()
    rows = []
    for port, sub in table.groupby("portfolio"):
        sub = sub.set_index("asset")
        for grp in universe.groups:
            m = [n for n in grp.members if n in sub.index]
            rows.append(
                dict(
                    portfolio=port,
                    group=grp.name,
                    weight=sub.loc[m, "weight"].sum(),
                    ccr=sub.loc[m, "ccr"].sum(),
                    pct_of_risk=sub.loc[m, "pct_of_risk"].sum(),
                    lo=grp.lo,
                    hi=grp.hi,
                )
            )
    return pd.DataFrame(rows)


def concentration(
    portfolios: Mapping[str, np.ndarray], sigma: pd.DataFrame, universe: Universe, top: int = 5
) -> pd.DataFrame:
    """Weight and risk concentration per portfolio. Effective N uses
    |risk shares| renormalised so hedging assets do not break the measure."""
    S = sigma.loc[universe.names, universe.names]
    rows = {}
    for name, w in portfolios.items():
        w = np.asarray(w, float)
        rc = risk_contributions(w, S)
        pct = rc["pct_of_risk"].values
        top_w = np.sort(w)[::-1][:top].sum()
        top_r = np.sort(np.abs(pct))[::-1][:top].sum() / max(np.abs(pct).sum(), 1e-12)
        rows[name] = dict(
            vol=portfolio_vol(w, S.values),
            diversification_ratio=diversification_ratio(w, S.values),
            n_holdings=int((w > 1e-4).sum()),
            effective_n_weight=effective_n(w),
            effective_n_risk=effective_n(np.abs(pct)),
            **{f"top{top}_weight": top_w, f"top{top}_risk": top_r},
            largest_risk=str(rc["pct_of_risk"].idxmax()),
            largest_risk_share=float(rc["pct_of_risk"].max()),
        )
    return pd.DataFrame(rows).T


def correlation_of_record(sigma: pd.DataFrame) -> pd.DataFrame:
    v = np.sqrt(np.diag(sigma.values))
    C = sigma.values / np.outer(v, v)
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=sigma.index, columns=sigma.columns)


def realised_vs_model_vol(sigma: pd.DataFrame, returns: pd.DataFrame, months: int = 36) -> pd.DataFrame:
    """Model vol next to the raw trailing realised vol (a stale or smoothed
    series shows a large gap)."""
    raw = returns[sigma.index].iloc[-months:].std() * np.sqrt(12)
    return pd.DataFrame(
        {"model_vol": np.sqrt(np.diag(sigma.values)), f"realised_vol_{months}m": raw.values},
        index=sigma.index,
    )


def eigen_diagnostics(sigma: pd.DataFrame) -> Dict[str, float]:
    C = correlation_of_record(sigma).values
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return dict(
        largest_eigen_share=float(ev[0] / ev.sum()),
        condition_number=float(ev[0] / max(ev[-1], 1e-12)),
        effective_factors=float(ev.sum() ** 2 / (ev**2).sum()),
    )
