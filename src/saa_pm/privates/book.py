"""The private-markets book on a MONTHLY clock - one model for the backtest and
the forward projection.

Each cohort (a commitment, a paid capital call, or a slice of legacy NAV)
ages on its own clock. At each cohort anniversary the Takahashi-Alexander
rates for the coming year are fixed:

* the year's capital call  = RC(year) * uncalled, paid in 12 equal instalments
* the year's payout rate   = RD(year) + yield, converted to the monthly rate m
  with 1 - (1 - m)^12 = min(RD + yield, 1), so a year of monthly payouts
  distributes exactly the annual TA fraction (a naive (RD + y) / 12 would
  pay out only ~61% instead of 90% when RD = 0.9)
* in the final year of fund life the remaining NAV is paid out linearly.

NAV is marked with whatever monthly return the caller supplies: realised
(reported) returns in the backtest, the CMA growth rate in a projection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .pacing import PacingParams, seed_legacy_vintages


class Cohort:
    __slots__ = ("nav", "uncalled", "age_m", "call_m", "rate_m", "liquidating")

    def __init__(self, nav: float = 0.0, uncalled: float = 0.0, age_m: int = 0):
        self.nav, self.uncalled, self.age_m = float(nav), float(uncalled), int(age_m)
        self.call_m = 0.0
        self.rate_m = 0.0
        self.liquidating = False

    def _start_year(self, p: PacingParams) -> None:
        year = self.age_m // 12 + 1
        call = p.contribution_rate(year) * self.uncalled
        self.uncalled -= call
        if year >= p.commitment_period:
            self.uncalled = 0.0  # unfunded expires with the LPA period
        self.call_m = call / 12.0
        annual = min(1.0, p.distribution_rate(year) + p.yield_rate)
        self.rate_m = 1.0 - (1.0 - annual) ** (1.0 / 12.0)
        self.liquidating = year >= p.life_years

    def step(self, p: PacingParams, r: float, distribute: bool = True) -> Tuple[float, float]:
        """One month: mark NAV, pay this month's call instalment, distribute.
        Returns (call, distribution)."""
        if self.age_m % 12 == 0:
            self._start_year(p)
        self.nav *= 1.0 + r
        call = self.call_m
        self.nav += call
        dist = 0.0
        if distribute:
            if self.liquidating:
                dist = self.nav / max(1, 12 - self.age_m % 12)
            else:
                dist = self.nav * self.rate_m
            dist = min(dist, self.nav)
            self.nav -= dist
        self.age_m += 1
        return call, dist

    @property
    def alive(self) -> bool:
        return self.nav > 1e-12 or self.uncalled > 1e-12 or self.call_m > 1e-12


class PrivateBook:
    """One private programme: cohorts, NAV, undrawn, and the cash it throws off."""

    def __init__(self, name: str, params: PacingParams, nav0: float = 0.0, uncalled0: float = 0.0):
        self.name, self.params = name, params
        self.cohorts: List[Cohort] = []
        self.maturity = float("nan")
        if nav0 > 1e-12 or uncalled0 > 1e-12:
            legacy, self.maturity = seed_legacy_vintages(nav0, uncalled0, params)
            self.cohorts = [Cohort(v.nav, v.uncalled, 12 * v.age) for v in legacy]
            # the steady-state shape is scaled to the inputs exactly
            nav = self.nav
            if nav > 1e-12:
                for c in self.cohorts:
                    c.nav *= nav0 / nav

    @property
    def nav(self) -> float:
        return sum(c.nav for c in self.cohorts)

    @property
    def uncalled(self) -> float:
        return sum(c.uncalled for c in self.cohorts)

    @property
    def pending_calls(self) -> float:
        """Instalments already fixed for the current cohort years."""
        return sum(c.call_m * (12 - c.age_m % 12) for c in self.cohorts if c.age_m % 12 != 0)

    def set_params(self, params: PacingParams) -> None:
        self.params = params

    def commit(self, amount: float) -> None:
        """A new commitment, drawn with the TA contribution rates."""
        if amount > 1e-12:
            self.cohorts.append(Cohort(0.0, amount * self.params.drawdown_ratio))

    def call_now(self, amount: float) -> None:
        """A capital call paid today: becomes NAV of a fresh cohort."""
        if amount > 1e-12:
            self.cohorts.append(Cohort(amount, 0.0))

    def step_month(self, r: float, distribute: bool = True) -> Tuple[float, float]:
        call = dist = 0.0
        for c in self.cohorts:
            ci, di = c.step(self.params, r, distribute)
            call += ci
            dist += di
        self.cohorts = [c for c in self.cohorts if c.alive]
        return call, dist

    def scale_nav(self, target: float) -> None:
        nav = self.nav
        if nav > 1e-12:
            for c in self.cohorts:
                c.nav *= target / nav


def monthly_growth(annual: float) -> float:
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def books_from_state(
    pacing: Dict[str, PacingParams], navs: Dict[str, float], undrawn: Dict[str, float]
) -> Dict[str, PrivateBook]:
    return {n: PrivateBook(n, p, navs.get(n, 0.0), undrawn.get(n, 0.0)) for n, p in pacing.items()}
