"""Monte Carlo wealth projection.

Monthly asset returns are joint lognormal, calibrated so that each asset's
expected ANNUAL simple return equals its CMA and its volatility matches the
covariance of record:

    log-drift (annual) = log(1 + mu) - sigma^2 / 2

(using ``mu - sigma^2/2`` would overstate every asset by ~mu^2/2 p.a.).
The portfolio is rebalanced monthly to a weight path (e.g. the projected
allocation at each call date, held until the next one).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class WealthProjection:
    percentiles: pd.DataFrame  # year x {p5, p25, p50, p75, p95, mean}
    real_percentiles: pd.DataFrame  # deflated by CPI
    probabilities: pd.DataFrame  # year x {P(beat target), P(loss), P(real loss)}


def simulate_wealth(
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: pd.DataFrame | np.ndarray,
    years: int = 10,
    n_paths: int = 10_000,
    start_value: float = 1.0,
    cpi: float = 0.0,
    target_return: Optional[float] = None,
    start: Optional[pd.Timestamp] = None,
    seed: int = 7,
) -> WealthProjection:
    """``weights`` is a single vector, or a frame indexed by date whose rows
    apply from that date onward (``start`` = projection start date)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    S = np.asarray(sigma, float)
    n_m = years * 12
    drift_m = (np.log1p(mu) - 0.5 * np.diag(S)) / 12.0
    L = np.linalg.cholesky(S / 12.0 + 1e-14 * np.eye(len(S)))

    if isinstance(weights, pd.DataFrame):
        start = pd.Timestamp(start or weights.index[0])
        dates = pd.date_range(start + pd.offsets.MonthEnd(1), periods=n_m, freq="ME")
        idx = pd.to_datetime(weights.index)
        pos = np.searchsorted(idx.values, dates.values, side="right") - 1
        W = weights.values[np.clip(pos, 0, len(weights) - 1)]
    else:
        W = np.tile(np.asarray(weights, float), (n_m, 1))

    wealth = np.full(n_paths, float(start_value))
    yearly = np.empty((years + 1, n_paths))
    yearly[0] = wealth
    for m in range(n_m):
        z = rng.standard_normal((n_paths, len(mu)))
        r = np.expm1(drift_m + z @ L.T)
        wealth = wealth * (1.0 + r @ W[m])
        if (m + 1) % 12 == 0:
            yearly[(m + 1) // 12] = wealth
    yrs = np.arange(years + 1)
    q = [5, 25, 50, 75, 95]

    def table(x):
        df = pd.DataFrame(np.percentile(x, q, axis=1).T, index=yrs, columns=[f"p{k}" for k in q])
        df["mean"] = x.mean(axis=1)
        df.index.name = "year"
        return df

    defl = (1 + cpi) ** yrs[:, None]
    real = yearly / defl
    probs = pd.DataFrame(index=yrs[1:])
    if target_return is not None:
        probs["p_beat_target"] = (yearly[1:] >= start_value * (1 + target_return) ** yrs[1:, None]).mean(
            axis=1
        )
    probs["p_nominal_loss"] = (yearly[1:] < start_value).mean(axis=1)
    probs["p_real_loss"] = (real[1:] < start_value).mean(axis=1)
    probs.index.name = "year"
    return WealthProjection(table(yearly), table(real), probs)
