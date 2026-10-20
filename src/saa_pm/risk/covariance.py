"""Covariance estimation for a universe with appraisal-based private assets.

Pipeline (:meth:`RiskModel.estimate`), using data strictly before ``t``:

1. aggregate monthly returns to the coarsest NAV frequency in the universe
   (interpolated quarterly NAVs would otherwise fake autocorrelation);
2. Geltner-desmooth the appraisal-based sleeves;
3. EWMA covariance (half-life in months, converted to the estimation
   frequency), bias-corrected and annualised;
4. constant-correlation shrinkage;
5. volatility uplift for illiquid and semi-liquid (gated) sleeves (correlations preserved);
6. nearest positive semi-definite repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import RiskModelConfig
from ..universe import Universe


def desmooth_returns(
    r: pd.Series, phi: Optional[float] = None, phi_cap: float = 0.6
) -> Tuple[pd.Series, float]:
    """Geltner (1993) first-order unsmoothing of appraisal-based returns.

        r_true_t = (r_obs_t - phi * r_obs_{t-1}) / (1 - phi)

    ``phi`` defaults to the lag-1 autocorrelation, clipped to [0, phi_cap].
    The first observation has no predecessor and becomes NaN.
    """
    x = r.dropna()
    if phi is None:
        phi = float(np.corrcoef(x.values[1:], x.values[:-1])[0, 1]) if len(x) >= 3 else 0.0
    if not np.isfinite(phi):
        phi = 0.0  # constant series: never NaN-poison Sigma
    phi = float(np.clip(phi, 0.0, phi_cap))
    if phi < 1e-8:
        return r.copy(), 0.0
    out = (x - phi * x.shift(1)) / (1.0 - phi)
    return out.reindex(r.index), phi


def aggregate_returns(win: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compound monthly returns into k-month buckets aligned to the END of the
    window (the ragged head is dropped). The index is the bucket's last date."""
    if k <= 1:
        return win
    n = (len(win) // k) * k
    w = win.iloc[len(win) - n :]
    grp = np.arange(n) // k
    out = (1.0 + w).groupby(grp).prod() - 1.0
    out.index = w.index[k - 1 :: k]
    return out


def ewma_covariance(
    returns: pd.DataFrame, halflife: float = 36.0, periods_per_year: float = 12.0, demean: bool = True
) -> Tuple[pd.DataFrame, float]:
    """Annualised EWMA covariance and the effective sample size 1/sum(w^2).

    ``halflife`` is in observation periods: the observation ``halflife``
    periods ago carries half the weight of the newest one.
    """
    X = returns.dropna().values
    n_obs, n_assets = X.shape
    if n_obs < 2:
        raise ValueError(
            f"ewma_covariance: {n_obs} usable observation(s) - need >= 2 "
            f"(window too short or a column is all-NaN)"
        )
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n_obs - 1, -1, -1)
    w = w / w.sum()
    mu = w @ X if demean else np.zeros(n_assets)
    Xc = X - mu
    S = (Xc * w[:, None]).T @ Xc
    S = S / (1.0 - np.sum(w**2))  # bias correction for weighted variance
    S *= periods_per_year
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns), 1.0 / np.sum(w**2)


def shrink_constant_correlation(S: pd.DataFrame, delta: float = 0.25) -> pd.DataFrame:
    """Blend the correlation matrix toward its average off-diagonal value
    (fixed intensity); variances are untouched."""
    v = np.sqrt(np.diag(S.values))
    if (v <= 0).any():
        bad = list(S.index[v <= 0])
        raise ValueError(f"zero variance for {bad} - constant price series?")
    C = S.values / np.outer(v, v)
    n = C.shape[0]
    if n < 2 or delta <= 0:
        return S.copy()
    r_bar = C[~np.eye(n, dtype=bool)].mean()
    T = np.full((n, n), r_bar)
    np.fill_diagonal(T, 1.0)
    C_s = (1 - delta) * C + delta * T
    return pd.DataFrame(C_s * np.outer(v, v), index=S.index, columns=S.columns)


def illiquidity_uplift(
    S: pd.DataFrame, liquid: Sequence[bool] | np.ndarray, uplift: float = 0.15
) -> pd.DataFrame:
    """Scale illiquid sleeves' volatility by (1 + uplift); correlations kept."""
    if uplift <= 0:
        return S
    f = np.array([1.0 if liq else 1.0 + uplift for liq in liquid])
    return pd.DataFrame(S.values * np.outer(f, f), index=S.index, columns=S.columns)


def nearest_psd(S: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """Symmetrise and clip eigenvalues at ``eps`` (well-posed SOC constraint)."""
    A = (S.values + S.values.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    A = vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T
    return pd.DataFrame((A + A.T) / 2.0, index=S.index, columns=S.columns)


def align_to(S: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Reorder a labelled covariance to ``names`` or raise on a set mismatch.

    Weights are positional; a silently misordered Sigma would compute every
    risk number against the wrong asset.
    """
    names = list(names)
    if list(S.index) == names and list(S.columns) == names:
        return S
    if set(S.index) == set(names) and set(S.columns) == set(names):
        return S.loc[names, names]
    raise ValueError(
        f"covariance does not match the universe: missing "
        f"{[n for n in names if n not in S.index]}, unexpected "
        f"{[n for n in S.index if n not in names]}"
    )


@dataclass
class RiskEstimate:
    """Output of :meth:`RiskModel.estimate`."""

    sigma: pd.DataFrame  # annualised, PSD, universe order
    phi: Dict[str, float]  # Geltner phi per appraisal-based asset
    n_obs: int  # observations at the estimation frequency
    n_eff: float  # EWMA effective sample size
    agg_months: int

    @property
    def vols(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.sigma.values)), index=self.sigma.index)

    @property
    def correlation(self) -> pd.DataFrame:
        v = self.vols.values
        return pd.DataFrame(
            self.sigma.values / np.outer(v, v), index=self.sigma.index, columns=self.sigma.columns
        )


class RiskModel:
    """Point-in-time covariance estimator (see module docstring)."""

    def __init__(self, config: Optional[RiskModelConfig] = None):
        self.config = config or RiskModelConfig()

    def agg_months(self, universe: Universe) -> int:
        return self.config.agg_months or universe.estimation_months

    def estimate(self, universe: Universe, returns: pd.DataFrame, t: Optional[int] = None) -> RiskEstimate:
        """Estimate Sigma from ``returns.iloc[:t]`` (default: all rows)."""
        c = self.config
        t = len(returns) if t is None else t
        k = self.agg_months(universe)
        win = returns[universe.names].iloc[max(0, t - c.max_lookback) : t]
        agg = aggregate_returns(win, k)
        clean = agg.copy()
        phis: Dict[str, float] = {}
        for a in universe:
            if a.appraisal_based:
                clean[a.name], phis[a.name] = desmooth_returns(agg[a.name], phi_cap=c.phi_cap)
        clean = clean.dropna()
        S, n_eff = ewma_covariance(clean, halflife=max(2.0, c.halflife / k), periods_per_year=12.0 / k)
        S = shrink_constant_correlation(S, c.shrink)
        S = illiquidity_uplift(S, universe.fully_liquid_mask, c.uplift)  # privates + gated sleeves
        S = nearest_psd(S)
        return RiskEstimate(align_to(S, universe.names), phis, len(clean), n_eff, k)
