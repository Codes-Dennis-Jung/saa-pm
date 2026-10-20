"""The funding ladder: pre-fund KNOWN capital calls by time-to-call.

Bucket k of the ladder pre-funds exactly the scheduled calls whose
time-to-call (years from the as-of date) falls in its window:

    bucket 1   0 <= t < edges[0]          (e.g. Cash,  calls due < 2y)
    bucket 2   edges[0] <= t < edges[1]   (e.g. T-bills, 2-3y)
    bucket N   t >= edges[-1]             (e.g. IG credit, 3y+)

    floor_k = cover * sum(scheduled amounts in bucket k) / portfolio value

Nothing is invented or stressed: what the schedule says is what is
pre-funded. As time passes the same schedule rolls down the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import LadderConfig
from ..universe import Universe

SCHEDULE_COLUMNS = ["date", "asset", "amount"]


def years_between(asof: pd.Timestamp, dates) -> np.ndarray:
    """Calendar time-to-call in years (whole years + months/12 + days/365.25),
    so a call on the exact 2-year anniversary is 2.00y."""
    a = pd.Timestamp(asof)
    d = pd.to_datetime(pd.Index(dates))
    return ((d.year - a.year) + (d.month - a.month) / 12.0 + (d.day - a.day) / 365.25).values.astype(float)


def ladder_bucket(years_to_call, edges: Sequence[float]) -> np.ndarray:
    """1-based bucket for each time-to-call (see module docstring)."""
    e = np.asarray(edges, dtype=float)
    t = np.atleast_1d(np.asarray(years_to_call, dtype=float))
    return 1 + np.searchsorted(e, t, side="right")


@dataclass
class Floor:
    """``selector @ w >= level`` for one ladder bucket."""

    bucket: int
    selector: np.ndarray
    level: float
    assets: Tuple[str, ...] = ()


@dataclass
class LadderRequirement:
    need: np.ndarray  # per bucket, fraction of PV
    detail: pd.DataFrame  # one row per future call
    floors: List[Floor] = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(self.need.sum()) if len(self.need) else 0.0


class FundingLadder:
    """Turns a call schedule into per-bucket weight floors for a universe."""

    def __init__(self, universe: Universe, config: Optional[LadderConfig] = None):
        config = config or LadderConfig()
        self.bucket_map: Dict[int, List[str]] = universe.ladder_map
        self.names = list(universe.names)
        self.config = config
        if self.bucket_map:
            if sorted(self.bucket_map) != list(range(1, len(self.bucket_map) + 1)):
                raise ValueError(f"ladder buckets must be 1..N, got {sorted(self.bucket_map)}")
            if len(config.edges) != len(self.bucket_map) - 1:
                raise ValueError(
                    f"{len(self.bucket_map)} ladder bucket(s) need "
                    f"{len(self.bucket_map) - 1} year edge(s), got {config.edges}"
                )
            illiquid = [n for ns in self.bucket_map.values() for n in ns if not universe[n].liquid]
            if illiquid:
                raise ValueError(f"ladder buckets contain illiquid assets {illiquid}")

    @property
    def active(self) -> bool:
        return bool(self.bucket_map)

    @property
    def n_buckets(self) -> int:
        return len(self.bucket_map)

    def bucket_label(self, k: int) -> str:
        lo = ([0.0] + list(self.config.edges))[k - 1]
        hi = (list(self.config.edges) + [np.inf])[k - 1]
        return f"{lo:g}y <= t < {hi:g}y" if np.isfinite(hi) else f"t >= {lo:g}y"

    def requirement(
        self, schedule: pd.DataFrame | None, asof: pd.Timestamp, portfolio_value: float
    ) -> LadderRequirement:
        """Per-bucket need from the calls dated strictly after ``asof``."""
        if not self.active:
            return LadderRequirement(np.zeros(0), pd.DataFrame(columns=SCHEDULE_COLUMNS))
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise ValueError(
                f"funding ladder needs a positive portfolio value in currency, got {portfolio_value}"
            )
        need = np.zeros(self.n_buckets)
        cols = SCHEDULE_COLUMNS + ["years_to_call", "bucket", "pct_of_pv"]
        s = pd.DataFrame(columns=cols)
        if schedule is not None and len(schedule):
            s = schedule.copy()
            s["date"] = pd.to_datetime(s["date"])
            s = s[s["date"] > pd.Timestamp(asof)].sort_values(["date", "asset"])
            if len(s):
                s["years_to_call"] = years_between(asof, s["date"])
                s["bucket"] = ladder_bucket(s["years_to_call"].values, self.config.edges)
                s["pct_of_pv"] = self.config.cover * s["amount"].values / float(portfolio_value)
                for k in range(1, self.n_buckets + 1):
                    need[k - 1] = float(s.loc[s["bucket"] == k, "pct_of_pv"].sum())
            s = s.reindex(columns=cols).reset_index(drop=True)
        return LadderRequirement(need, s, self.floors(need))

    def floors(self, need: np.ndarray) -> List[Floor]:
        """Bucket k alone pre-funds need[k-1]: no substitution between buckets."""
        out = []
        for k, members in self.bucket_map.items():
            sel = np.array([1.0 if n in members else 0.0 for n in self.names])
            out.append(Floor(k, sel, float(need[k - 1]), tuple(members)))
        return out

    def held(self, w: np.ndarray) -> np.ndarray:
        """Weight held in each bucket."""
        idx = {n: i for i, n in enumerate(self.names)}
        return np.array([sum(w[idx[n]] for n in self.bucket_map[k]) for k in sorted(self.bucket_map)])

    def table(self, req: LadderRequirement, w: np.ndarray, pv: float) -> pd.DataFrame:
        held = self.held(w)
        rows = []
        for k in sorted(self.bucket_map):
            nd = float(req.need[k - 1])
            rows.append(
                dict(
                    bucket=k,
                    assets="|".join(self.bucket_map[k]),
                    window=self.bucket_label(k),
                    need_pct_pv=nd,
                    need_ccy=nd * pv,
                    held_pct_pv=float(held[k - 1]),
                    cover=float(held[k - 1] / nd) if nd > 1e-6 else np.nan,
                )
            )
        return pd.DataFrame(rows)
