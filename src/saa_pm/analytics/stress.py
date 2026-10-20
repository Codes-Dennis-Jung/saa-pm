"""Stress testing.

1. Conditional factor stress (delta approach). Assets and stress factors
   form one joint return panel; a scenario shocks a subset K of factors by s
   and each asset moves by its conditional expectation

        m = Sigma_aK Sigma_KK^-1 s

   so correlated assets move even when the scenario names one factor. P&L is
   w'm: linear, first-order - what a monthly SAA book can honestly support.
   Scenarios are explicit shocks, or historical windows whose factor moves
   are read from the factor levels (explicit values override).

2. Historical episodes. Fixed-weight portfolios replayed through dated
   windows of the asset history (named windows plus the worst non-overlapping
   12-month windows found in the data).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..risk.covariance import aggregate_returns, desmooth_returns
from ..universe import Universe
from .performance import max_drawdown


def scenario_shocks(scenarios: pd.DataFrame, factor_levels: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """{scenario: {factor: shock}} from explicit values and/or date windows."""
    factors = [c for c in scenarios.columns if c not in ("name", "start", "end")]
    unknown = [c for c in factors if c not in factor_levels.columns]
    if unknown:
        raise ValueError(f"stress scenarios use factors {unknown} that are not in the factor file")
    out = {}
    for _, row in scenarios.iterrows():
        shocks: Dict[str, float] = {}
        start, end = row.get("start"), row.get("end")
        if pd.notna(start) and pd.notna(end):
            s, e = start + pd.offsets.MonthEnd(0), end + pd.offsets.MonthEnd(0)
            for c in factors:
                col = factor_levels[c].dropna()
                if s in col.index and e in col.index:
                    shocks[c] = float(col.loc[e] / col.loc[s] - 1.0)
        elif pd.notna(start) != pd.notna(end):
            raise ValueError(f"scenario '{row['name']}' needs both start and end")
        for c in factors:
            if pd.notna(row.get(c)):
                v = float(row[c])
                if abs(v) > 1.5:
                    raise ValueError(
                        f"scenario '{row['name']}': shock {v} in '{c}' > 150% - write shocks as '10%' or 0.10"
                    )
                shocks[c] = v
        if shocks:
            out[str(row["name"])] = shocks
    return out


class FactorStress:
    """Joint asset/factor panel treated exactly as the risk model treats assets."""

    def __init__(
        self, universe: Universe, returns: pd.DataFrame, factor_levels: pd.DataFrame, min_obs: int = 20
    ):
        self.universe = universe
        k = universe.estimation_months
        fac = factor_levels.pct_change(fill_method=None)
        clash = set(fac.columns) & set(returns.columns)
        if clash:
            fac = fac.rename(columns={c: f"factor:{c}" for c in clash})
        joint = returns[universe.names].join(fac, how="inner").dropna()
        agg = aggregate_returns(joint, k)
        for a in universe:
            if a.appraisal_based:
                agg[a.name], _ = desmooth_returns(agg[a.name])
        self.panel = agg.dropna()
        self.factor_names = list(fac.columns)
        if len(self.panel) < min_obs:
            raise ValueError(
                f"only {len(self.panel)} joint observations of assets and "
                f"stress factors - extend the factor history"
            )
        self.cov = self.panel.cov()

    def asset_moves(self, shocks: Mapping[str, float]) -> pd.Series:
        K = list(shocks)
        S_aK = self.cov.loc[self.universe.names, K].values
        S_KK = self.cov.loc[K, K].values
        beta = S_aK @ np.linalg.pinv(S_KK)
        return pd.Series(beta @ np.array([shocks[c] for c in K]), index=self.universe.names)

    def run(
        self,
        scenarios: Mapping[str, Mapping[str, float]],
        portfolios: Mapping[str, np.ndarray],
        portfolio_value: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (P&L per scenario x portfolio in %, asset moves per scenario)."""
        moves = pd.DataFrame({s: self.asset_moves(sh) for s, sh in scenarios.items()})
        pnl = pd.DataFrame(
            {p: moves.T.values @ np.asarray(w, float) for p, w in portfolios.items()}, index=moves.columns
        )
        pnl.index.name = "scenario"
        return pnl, moves


DEFAULT_EPISODES: List[Tuple[str, str, str]] = [
    ("Global financial crisis", "2007-10-31", "2009-02-28"),
    ("Euro sovereign crisis", "2011-04-30", "2011-09-30"),
    ("Taper tantrum", "2013-04-30", "2013-08-31"),
    ("China / oil slump", "2015-05-31", "2016-02-29"),
    ("Q4 2018 sell-off", "2018-09-30", "2018-12-31"),
    ("Covid crash", "2020-01-31", "2020-03-31"),
    ("Rates shock 2022", "2021-12-31", "2022-09-30"),
]


def worst_windows(r: pd.Series, months: int = 12, n: int = 2) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """The n worst non-overlapping rolling windows of a return series."""
    roll = (1 + r).rolling(months).apply(np.prod, raw=True) - 1
    roll = roll.dropna().sort_values()
    picked: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    pos = {d: i for i, d in enumerate(r.index)}
    for end, _ in roll.items():
        i = pos[end]
        if all(abs(i - pos[e]) >= months for _, e in picked):
            picked.append((r.index[i - months + 1], end))
        if len(picked) == n:
            break
    return picked


def historical_episodes(
    returns: pd.DataFrame,
    portfolios: Mapping[str, np.ndarray],
    episodes: Optional[Sequence[Tuple[str, str, str]]] = None,
    reference: Optional[str] = None,
    n_worst: int = 2,
) -> pd.DataFrame:
    """Cumulative return and max drawdown of each fixed-weight portfolio over
    each episode ``(name, start, end)``: from the level at the start month-end
    to the level at the end month-end (returns strictly after ``start``).
    The worst non-overlapping 12-month windows of ``reference`` are added."""
    paths = pd.DataFrame(
        {p: returns.values @ np.asarray(w, float) for p, w in portfolios.items()}, index=returns.index
    )
    eps = list(episodes if episodes is not None else DEFAULT_EPISODES)
    ref = reference or next(iter(portfolios))
    for i, (s, e) in enumerate(worst_windows(paths[ref], 12, n_worst), start=1):
        s = s - pd.offsets.MonthEnd(1)  # window starts at the prior level
        eps.append((f"worst 12m #{i} ({ref})", f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}"))
    rows = []
    first, last = returns.index[0], returns.index[-1]
    for name, s, e in eps:
        s, e = pd.Timestamp(s) + pd.offsets.MonthEnd(0), pd.Timestamp(e) + pd.offsets.MonthEnd(0)
        if s < first - pd.offsets.MonthEnd(1) or e > last:
            continue
        win = paths[(paths.index > s) & (paths.index <= e)]
        for p in portfolios:
            r = win[p]
            rows.append(
                dict(
                    episode=name,
                    start=s,
                    end=e,
                    months=len(r),
                    portfolio=p,
                    cum_return=float((1 + r).prod() - 1),
                    max_drawdown=max_drawdown(r),
                )
            )
    return pd.DataFrame(rows)
