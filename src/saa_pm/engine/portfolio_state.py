"""Portfolio state in currency, with the capital-call cash waterfall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from ..privates.book import PrivateBook
from ..privates.pacing import PacingParams
from ..universe import Universe


@dataclass
class FundingResult:
    paid_from_cash: float = 0.0
    paid_from_ladder: float = 0.0
    forced: float = 0.0  # pro-rata sale of non-ladder liquid assets
    cost: float = 0.0  # transaction cost of the forced sale
    shortfall: float = 0.0  # could not be funded at all -> overdraft


class PortfolioState:
    """Market values per asset (currency), cash, and one book per private sleeve.

    Private values always equal their book NAV. Cash is normally swept into the
    first ladder asset; it goes NEGATIVE only when a capital call cannot be
    funded (an overdraft that is recorded, never silently created value).
    """

    def __init__(
        self,
        universe: Universe,
        values: Dict[str, float],
        books: Dict[str, PrivateBook],
        default_tx_bps: float = 10.0,
    ):
        self.universe = universe
        self.names = list(universe.names)
        self.value = {n: float(values.get(n, 0.0)) for n in self.names}
        self.books = books
        self.cash = 0.0
        self.default_tx_bps = default_tx_bps
        for n, bk in self.books.items():
            target = self.value[n]
            if target > 1e-9 and bk.nav <= 1e-12:
                raise ValueError(f"'{n}' holds {target:,.0f} but its book has no NAV")
            bk.scale_nav(target)
            self.value[n] = bk.nav

    @classmethod
    def from_weights(
        cls,
        universe: Universe,
        w: np.ndarray,
        pv: float,
        pacing: Dict[str, PacingParams],
        undrawn: Dict[str, float],
        default_tx_bps: float = 10.0,
    ) -> PortfolioState:
        values = {n: float(w[i] * pv) for i, n in enumerate(universe.names)}
        books = {n: PrivateBook(n, p, values.get(n, 0.0), undrawn.get(n, 0.0)) for n, p in pacing.items()}
        return cls(universe, values, books, default_tx_bps)

    @property
    def pv(self) -> float:
        return sum(self.value.values()) + self.cash

    def weights(self) -> np.ndarray:
        pv = self.pv
        return np.array([self.value[n] / pv for n in self.names])

    def tx_bps(self, name: str) -> float:
        a = self.universe[name]
        return a.tx_cost_bps if a.tx_cost_bps is not None else self.default_tx_bps

    def sync_private_values(self) -> None:
        for n, bk in self.books.items():
            self.value[n] = bk.nav

    def fund(self, need: float, ladder: Sequence[str], forced_extra_bps: float = 0.0) -> FundingResult:
        """Pay ``need`` from cash, then the ladder in order, then a forced
        pro-rata sale of the other liquid assets; any remainder is an overdraft."""
        res = FundingResult()
        if need <= 1e-12:
            return res
        take = min(max(self.cash, 0.0), need)
        self.cash -= take
        need -= take
        res.paid_from_cash = take
        for n in ladder:
            if need <= 1e-12:
                break
            take = min(max(self.value[n], 0.0), need)
            self.value[n] -= take
            need -= take
            res.paid_from_ladder += take
        if need <= 1e-12:
            return res
        # gated (semi-liquid) sleeves cannot be sold at short notice to meet a call
        others = [
            n for n in self.universe.liquid_names if n not in ladder and not self.universe[n].semi_liquid
        ]
        pool = sum(max(self.value[n], 0.0) for n in others)
        forced = min(need, pool)
        if pool > 1e-12:
            for n in others:
                amt = forced * max(self.value[n], 0.0) / pool
                self.value[n] -= amt
                res.cost += amt * (self.tx_bps(n) + forced_extra_bps) / 1e4
        res.forced = forced
        need -= forced
        self.cash -= res.cost
        if need > 1e-9:
            res.shortfall = need
            self.cash -= need  # overdraft: the call is owed, not free
        return res

    def sweep_cash(self, to: str) -> None:
        if self.cash > 1e-12:
            self.value[to] += self.cash
            self.cash = 0.0

    def trade_to(self, w_target: np.ndarray, liquid_only: bool = True) -> tuple[float, float]:
        """Trade the liquid book to ``w_target`` (privates cannot be traded).
        Costs are charged pro rata to the liquid book. Returns (turnover, cost)."""
        pv = self.pv
        w_now = self.weights()
        trade = (np.asarray(w_target) - w_now) * pv
        cost = 0.0
        for i, a in enumerate(self.universe):
            if liquid_only and not a.liquid:
                continue
            cost += abs(trade[i]) * self.tx_bps(a.name) / 1e4
            self.value[a.name] += trade[i]
        liq = self.universe.liquid_names
        tot = sum(max(self.value[n], 0.0) for n in liq)
        if tot > 0 and cost > 0:
            for n in liq:
                self.value[n] -= cost * max(self.value[n], 0.0) / tot
        turnover = 0.5 * float(np.abs(trade[self.universe.liquid_mask]).sum()) / pv
        return turnover, cost
