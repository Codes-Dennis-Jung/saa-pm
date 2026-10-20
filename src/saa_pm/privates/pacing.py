"""Takahashi-Alexander (2002) commitment pacing, per vintage.

Each commitment cohort ("vintage") ages on its own clock:

    Call_t = RC(age) * Uncalled_{t-1}
    Dist_t = min(RD(age) * NAV_{t-1} * (1 + G) + Y * NAV_{t-1},  NAV available)
    NAV_t  = NAV_{t-1} * (1 + G) + Call_t - Dist_t
    RD(age) = min(1, (age / L) ** B)

Calibration rules (checked by :meth:`PacingParams.problems`):

* contribution rates apply to REMAINING unfunded capital and must reach 1.0
  by the end of the commitment period, otherwise the fund calls forever;
* ``growth_rate`` equals the sleeve's CMA total return: the vintage IRR of a
  TA path equals ``growth_rate`` and ``yield_rate`` only splits it between
  income and appreciation. Loading forces this invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PacingParams:
    """Takahashi-Alexander parameters for ONE private programme."""

    name: str = "generic"
    life_years: int = 12
    commitment_period: int = 5
    contribution_rates: Tuple[float, ...] = (0.25, 0.33, 0.50, 0.75, 1.00)
    bow_factor: float = 2.5
    growth_rate: float = 0.095
    yield_rate: float = 0.0
    drawdown_ratio: float = 0.95  # share of a commitment ever called
    yield_from_cma: bool = True  # True -> yield re-synced to CMA income

    def __post_init__(self):
        object.__setattr__(self, "contribution_rates", tuple(float(r) for r in self.contribution_rates))

    def contribution_rate(self, age: int) -> float:
        rates = self.contribution_rates
        return rates[min(max(age, 1), len(rates)) - 1]

    def distribution_rate(self, age: float) -> float:
        return min(1.0, (max(age, 0.0) / self.life_years) ** self.bow_factor)

    def synced_to(self, exp_return: float, income: float) -> PacingParams:
        """Re-establish growth == CMA total return (and yield == CMA income
        unless the yield was stated explicitly), capping the yield at the
        total return so NAV is never distributed away."""
        y = income if self.yield_from_cma else self.yield_rate
        return replace(self, growth_rate=float(exp_return), yield_rate=float(min(y, exp_return)))

    def problems(self, cma_total_return: Optional[float] = None, tol: float = 0.005) -> List[str]:
        msgs = []
        r = np.array(self.contribution_rates)
        if len(r) == 0:
            msgs.append(f"[{self.name}] no contribution rates")
            return msgs
        if (r <= 0).any() or (r > 1 + 1e-12).any():
            msgs.append(
                f"[{self.name}] contribution rates must lie in (0, 1], got "
                f"{list(r)} (percent written without '%'?)"
            )
        resid = float(np.prod(1.0 - np.clip(r, 0, 1)))
        if resid > 1e-6:
            msgs.append(
                f"[{self.name}] contribution rates leave {resid:.1%} undrawn - the last rate must be 1.0"
            )
        if len(r) > self.commitment_period:
            msgs.append(
                f"[{self.name}] {len(r)} contribution rates but the commitment "
                f"period is {self.commitment_period}y - the tail expires"
            )
        if self.commitment_period >= self.life_years:
            msgs.append(f"[{self.name}] commitment period >= fund life")
        if self.yield_rate > self.growth_rate + 1e-9:
            msgs.append(
                f"[{self.name}] yield {self.yield_rate:.2%} exceeds total return {self.growth_rate:.2%}"
            )
        if not 0.5 <= self.drawdown_ratio <= 1.0:
            msgs.append(f"[{self.name}] implausible drawdown_ratio {self.drawdown_ratio}")
        if cma_total_return is not None and abs(self.growth_rate - cma_total_return) > tol:
            msgs.append(f"[{self.name}] growth {self.growth_rate:.2%} != CMA {cma_total_return:.2%}")
        return msgs


class Vintage:
    """One commitment cohort on an annual clock."""

    __slots__ = ("nav", "uncalled", "age")

    def __init__(self, nav: float, uncalled: float, age: int = 0):
        self.nav, self.uncalled, self.age = float(nav), float(uncalled), int(age)

    def copy(self) -> Vintage:
        return Vintage(self.nav, self.uncalled, self.age)

    def step(self, p: PacingParams) -> Tuple[float, float]:
        """Advance one year; returns (call, distribution)."""
        self.age += 1
        call = p.contribution_rate(self.age) * self.uncalled
        self.uncalled -= call
        if self.age >= p.commitment_period:
            self.uncalled = 0.0  # unfunded expires with the LPA period
        grown = self.nav * (1.0 + p.growth_rate)
        dist = min(p.distribution_rate(self.age) * grown + p.yield_rate * self.nav, grown + call)
        self.nav = grown + call - dist
        return call, dist


def single_vintage_profile(p: PacingParams, max_age: Optional[int] = None) -> pd.DataFrame:
    """Path of a commitment of 1.0 by age: call, distribution, nav, uncalled."""
    max_age = max_age or (p.life_years + 3)
    v = Vintage(0.0, p.drawdown_ratio)
    rows = []
    for _ in range(max_age):
        c, d = v.step(p)
        rows.append(dict(age=v.age, call=c, distribution=d, nav=v.nav, uncalled=v.uncalled))
    return pd.DataFrame(rows).set_index("age")


def implied_irr(p: PacingParams) -> float:
    """IRR of a single-vintage TA path (should equal ``growth_rate``)."""
    prof = single_vintage_profile(p, p.life_years + 5)
    cf = np.concatenate([[0.0], (prof["distribution"] - prof["call"]).values])
    cf[-1] += prof["nav"].iloc[-1]
    roots = np.roots(cf[::-1])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return float((1 / roots - 1).max()) if len(roots) else float("nan")


def seed_legacy_vintages(nav0: float, uncalled0: float, p: PacingParams) -> Tuple[List[Vintage], float]:
    """Spread an EXISTING programme across ages with the steady-state profile.

    NAV and uncalled are scaled independently; the returned maturity ratio
    (> 1 younger than steady state, < 1 mature) is a diagnostic.
    """
    prof = single_vintage_profile(p, p.life_years + 3)
    nav_shape, unc_shape = prof["nav"].values, prof["uncalled"].values
    k_nav = nav0 / nav_shape.sum() if nav_shape.sum() > 0 else 0.0
    k_unc = uncalled0 / unc_shape.sum() if unc_shape.sum() > 0 else 0.0
    vintages = [
        Vintage(k_nav * n, k_unc * u, int(a))
        for a, n, u in zip(prof.index.values, nav_shape, unc_shape)
        if k_nav * n > 1e-12 or k_unc * u > 1e-12
    ]
    return vintages, (k_unc / k_nav) if k_nav > 0 else float("nan")


def simulate_pacing(
    commitments: Sequence[float],
    p: PacingParams,
    nav0: float = 0.0,
    uncalled0: float = 0.0,
    horizon: Optional[int] = None,
    legacy: Optional[List[Vintage]] = None,
) -> pd.DataFrame:
    """Aggregate annual call / distribution / NAV path over all vintages."""
    T = horizon or len(commitments)
    commitments = list(commitments) + [0.0] * max(0, T - len(commitments))
    if legacy is None:
        legacy, _ = seed_legacy_vintages(nav0, uncalled0, p)
    vintages = [v.copy() for v in legacy]
    rows = []
    for t in range(1, T + 1):
        vintages.append(Vintage(0.0, commitments[t - 1] * p.drawdown_ratio))
        call_t = dist_t = 0.0
        for v in vintages:
            c, d = v.step(p)
            call_t += c
            dist_t += d
        vintages = [v for v in vintages if v.nav > 1e-10 or v.uncalled > 1e-10]
        rows.append(
            dict(
                year=t,
                commitment=commitments[t - 1],
                call=call_t,
                distribution=dist_t,
                net_cashflow=dist_t - call_t,
                nav=sum(v.nav for v in vintages),
                uncalled=sum(v.uncalled for v in vintages),
            )
        )
    return pd.DataFrame(rows).set_index("year")
