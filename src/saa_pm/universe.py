"""The investment universe: assets, group constraints and the funding ladder.

All objects here are immutable. Anything that changes over time (a CMA
vintage, a policy bound, private weights pinned at their NAV) produces a
*new* :class:`Universe` via :meth:`Universe.with_assets` and friends, so a
decision at one date can never leak state into another.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

FREQ_MONTHS: Dict[str, int] = {"M": 1, "Q": 3, "SA": 6, "A": 12}


@dataclass(frozen=True)
class Asset:
    """Capital-market assumptions and trading rules for one asset class.

    All rates are annual decimals. ``exp_return`` is the arithmetic expected
    TOTAL return; ``income`` is the part of it paid out as cash.
    """

    name: str
    group: str
    exp_return: float
    income: float
    liquid: bool = True
    lb: float = 0.0
    ub: float = 1.0
    appraisal_based: bool = False  # True -> desmooth the return history
    max_trade: Optional[float] = None  # per-rebalance |w - w_prev| cap
    ladder_bucket: Optional[int] = None  # funding-ladder bucket (1 = soonest)
    tx_cost_bps: float = 10.0
    nav_frequency: str = "M"  # M / Q / SA / A
    price_column: str = ""  # prices.csv column (default: name)
    # semi-liquid (evergreen / ELTIF) vehicle: bought at NAV like a liquid
    # asset, but at most this share of the HOLDING can be redeemed per
    # rebalance (the fund's gate); None = no redemption limit
    redemption_limit: Optional[float] = None

    @property
    def semi_liquid(self) -> bool:
        return self.liquid and self.redemption_limit is not None

    @property
    def nav_months(self) -> int:
        return FREQ_MONTHS.get(self.nav_frequency.upper(), 1)

    @property
    def price_col(self) -> str:
        return self.price_column or self.name


@dataclass(frozen=True)
class GroupConstraint:
    """``lo <= sum(w[members]) <= hi`` - groups may overlap (look-through)."""

    name: str
    members: Tuple[str, ...]
    lo: float = 0.0
    hi: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "members", tuple(self.members))


@dataclass(frozen=True)
class GroupBreach:
    """A pinned private book (plus the liquid members' lower bounds) that
    already exceeds a group cap."""

    group: str
    weight: float
    cap: float


class Universe:
    """Ordered, validated collection of assets plus group constraints.

    Weight vectors throughout the package are indexed by position in
    :attr:`names`; this class is the single source of that order.
    """

    def __init__(self, assets: Sequence[Asset], groups: Sequence[GroupConstraint] = ()):
        self.assets: Tuple[Asset, ...] = tuple(assets)
        self.groups: Tuple[GroupConstraint, ...] = tuple(groups)
        self.names: List[str] = [a.name for a in self.assets]
        dupes = sorted({n for n in self.names if self.names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate asset names: {dupes}")
        self._idx = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self):
        return iter(self.assets)

    def __contains__(self, name: str) -> bool:
        return name in self._idx

    def __getitem__(self, name: str) -> Asset:
        return self.assets[self._idx[name]]

    def index(self, name: str) -> int:
        return self._idx[name]

    @property
    def n(self) -> int:
        return len(self.assets)

    @property
    def mu(self) -> np.ndarray:
        return np.array([a.exp_return for a in self.assets], dtype=float)

    @property
    def income(self) -> np.ndarray:
        return np.array([a.income for a in self.assets], dtype=float)

    @property
    def lb(self) -> np.ndarray:
        return np.array([a.lb for a in self.assets], dtype=float)

    @property
    def ub(self) -> np.ndarray:
        return np.array([a.ub for a in self.assets], dtype=float)

    @property
    def liquid_mask(self) -> np.ndarray:
        return np.array([a.liquid for a in self.assets], dtype=bool)

    @property
    def liquid_names(self) -> List[str]:
        return [a.name for a in self.assets if a.liquid]

    @property
    def illiquid_names(self) -> List[str]:
        return [a.name for a in self.assets if not a.liquid]

    @property
    def semi_liquid_names(self) -> List[str]:
        return [a.name for a in self.assets if a.semi_liquid]

    @property
    def fully_liquid_mask(self) -> np.ndarray:
        """Liquid and not gated: tradable at short notice in any size."""
        return np.array([a.liquid and not a.semi_liquid for a in self.assets], dtype=bool)

    @property
    def estimation_months(self) -> int:
        """Coarsest NAV frequency in the universe (risk-model frequency)."""
        return max([a.nav_months for a in self.assets] + [1])

    def selector(self, members: Iterable[str]) -> np.ndarray:
        m = set(members)
        return np.array([1.0 if n in m else 0.0 for n in self.names])

    def group_selectors(self) -> List[Tuple[GroupConstraint, np.ndarray]]:
        return [(g, self.selector(g.members)) for g in self.groups]

    def series(self, w: np.ndarray):
        import pandas as pd

        return pd.Series(np.asarray(w, float), index=self.names)

    @property
    def ladder_map(self) -> Dict[int, List[str]]:
        """{bucket: [asset names]} in bucket order, liquid assets only."""
        out: Dict[int, List[str]] = {}
        for a in self.assets:
            if a.ladder_bucket is not None:
                out.setdefault(int(a.ladder_bucket), []).append(a.name)
        return dict(sorted(out.items()))

    @property
    def ladder_order(self) -> List[str]:
        """Assets in the order calls are paid from them (bucket 1 first)."""
        return [n for k in sorted(self.ladder_map) for n in self.ladder_map[k]]

    @property
    def sweep_asset(self) -> str:
        """Where cash (income, distributions, inflows) is swept to."""
        order = self.ladder_order
        if order:
            return order[0]
        liq = [a.name for a in self.assets if a.liquid and not a.semi_liquid]  # never a gated sleeve
        if not liq:
            raise ValueError("universe has no liquid asset to hold cash")
        return liq[0]

    def with_assets(self, assets: Sequence[Asset]) -> Universe:
        return Universe(assets, self.groups)

    def with_groups(self, groups: Sequence[GroupConstraint]) -> Universe:
        return Universe(self.assets, groups)

    def updated(self, changes: Mapping[str, Mapping[str, Any]]) -> Universe:
        """New universe with per-asset field changes ``{name: {field: value}}``."""
        unknown = sorted(set(changes) - set(self.names))
        if unknown:
            raise KeyError(f"unknown assets {unknown}")
        return self.with_assets(
            [replace(a, **changes[a.name]) if a.name in changes else a for a in self.assets]
        )

    def with_cmas(self, cmas: Mapping[str, Tuple[float, float]]) -> Universe:
        """Apply ``{name: (exp_return, income)}``; unknown names are ignored."""
        return self.updated(
            {n: {"exp_return": float(r), "income": float(i)} for n, (r, i) in cmas.items() if n in self}
        )

    def pin_privates(self, w: np.ndarray, eps: float = 1e-6) -> Tuple[Universe, List[GroupBreach]]:
        """Fix illiquid weights at ``w`` (they cannot be traded) and widen any
        group cap the pinned book already exceeds, recording the breach. In a
        mixed group the liquid members' lower bounds count as forced too."""
        w = np.asarray(w, float)
        assets = [
            replace(a, lb=max(0.0, w[i] - eps), ub=w[i] + eps) if not a.liquid else a
            for i, a in enumerate(self.assets)
        ]
        groups, breaches = [], []
        for g in self.groups:
            members = [self[m] for m in g.members if m in self._idx]
            forced = float(sum(w[self._idx[a.name]] if not a.liquid else a.lb for a in members))
            if all(a.liquid for a in members):
                forced = 0.0  # purely liquid groups are not affected by pinning
            hi = g.hi
            if forced > hi + 1e-9:
                breaches.append(GroupBreach(g.name, round(forced, 4), hi))
                hi = forced + eps
            groups.append(replace(g, hi=hi))
        return Universe(assets, groups), breaches

    def structural_problems(self) -> List[str]:
        """Contradictions in bounds / groups that make every solve infeasible."""
        probs: List[str] = []
        names = set(self.names)
        for a in self.assets:
            if not (np.isfinite(a.lb) and np.isfinite(a.ub)):
                probs.append(f"asset '{a.name}': non-finite bounds [{a.lb}, {a.ub}]")
            elif a.lb < 0 or a.lb > a.ub + 1e-9:
                probs.append(f"asset '{a.name}': need 0 <= lb <= ub, got [{a.lb:.2%}, {a.ub:.2%}]")
        s_lb, s_ub = float(self.lb.sum()), float(self.ub.sum())
        if s_lb > 1 + 1e-9:
            probs.append(f"sum of lower bounds {s_lb:.2%} > 100%")
        if s_ub < 1 - 1e-9:
            probs.append(f"sum of upper bounds {s_ub:.2%} < 100%")
        for g in self.groups:
            bad = [m for m in g.members if m not in names]
            if bad:
                probs.append(f"group '{g.name}' references unknown assets {bad}")
            if g.lo > g.hi + 1e-9:
                probs.append(f"group '{g.name}': lo {g.lo:.2%} > hi {g.hi:.2%}")
            mem = [a for a in self.assets if a.name in g.members]
            oth = [a for a in self.assets if a.name not in g.members]
            m_lb, m_ub = sum(a.lb for a in mem), sum(a.ub for a in mem)
            o_lb, o_ub = sum(a.lb for a in oth), sum(a.ub for a in oth)
            if m_lb > g.hi + 1e-9:
                probs.append(f"group '{g.name}': member lower bounds {m_lb:.2%} > cap {g.hi:.2%}")
            if m_ub < g.lo - 1e-9:
                probs.append(f"group '{g.name}': floor {g.lo:.2%} but members hold at most {m_ub:.2%}")
            if 1 - o_ub > g.hi + 1e-9:
                probs.append(
                    f"group '{g.name}': non-members hold at most {o_ub:.2%}, "
                    f"forcing the group above its cap {g.hi:.2%}"
                )
            if 1 - o_lb < g.lo - 1e-9:
                probs.append(
                    f"group '{g.name}': floor {g.lo:.2%} unreachable, non-member floors claim {o_lb:.2%}"
                )
        return probs

    def __repr__(self) -> str:
        return f"Universe({self.n} assets, {len(self.groups)} groups)"
