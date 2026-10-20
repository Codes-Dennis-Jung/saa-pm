"""Typed settings.

``config.csv`` is a flat ``key;value`` file. :meth:`Settings.from_mapping`
maps it onto five small, immutable config objects:

* :class:`Goals`            - the mandate: return / income / risk targets
* :class:`OptimizerConfig`  - how each convex solve is run
* :class:`RiskModelConfig`  - covariance estimation
* :class:`LadderConfig`     - funding-ladder year edges and cover
* :class:`BacktestConfig`   - walk-forward mechanics

Unknown keys are kept in ``Settings.unknown`` and reported by validation
instead of being silently ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


@dataclass(frozen=True)
class Goals:
    """Three ranked goals: (1) return >= CPI + spread, (2) income >= target,
    (3) volatility inside [vol_lo, vol_hi]. The order is binding.

    The income goal may be phased in: with ``income_target_year`` set, the
    binding income target ramps linearly from ``income_glide_start_income``
    (at ``income_glide_start_year``) to ``target_income`` (at the target year).
    """

    cpi: float = 0.024
    return_spread: float = 0.030
    target_income: float = 0.035
    vol_lo: float = 0.06
    vol_hi: float = 0.08
    target_return_override: float = float("nan")  # set to bypass CPI + spread
    income_target_year: float = float("nan")
    income_glide_start_year: float = float("nan")
    income_glide_start_income: float = float("nan")

    @property
    def target_return(self) -> float:
        if not _isnan(self.target_return_override):
            return float(self.target_return_override)
        return self.cpi + self.return_spread

    def income_target_at(self, date: pd.Timestamp) -> float:
        """Binding income target at ``date`` (a floor; earlier is welcome)."""
        y_final = self.income_target_year
        if _isnan(y_final):
            return self.target_income
        yr = date.year + (date.month - 1) / 12.0
        if yr >= y_final:
            return self.target_income
        y0, inc0 = self.income_glide_start_year, self.income_glide_start_income
        if _isnan(y0) or _isnan(inc0) or y0 >= y_final:
            return self.target_income
        if yr <= y0:
            return inc0
        return inc0 + (self.target_income - inc0) * (yr - y0) / (y_final - y0)


@dataclass(frozen=True)
class OptimizerConfig:
    """Targets plus solver settings for ONE decision date."""

    target_return: float = 0.054
    target_income: float = 0.035
    vol_lo: float = 0.06
    vol_hi: float = 0.08
    vol_steps: int = 21
    turnover_limit: Optional[float] = None  # one-way, sum|dw|/2; None = no cap
    income_penalty: float = 25.0  # only used if lexicographic=False
    hard_return_floor: bool = False  # True -> infeasible if unreachable
    lexicographic: bool = True  # strict goal order inside a solve
    return_tol: float = 1e-5  # give-back tolerance between stages
    vol_relax_max: float = 0.0  # > vol_hi lets risk yield to return
    illiquid_max_trade: float = 0.01  # default no-trade band, privates
    solver: str = "CLARABEL"

    def __post_init__(self):
        if self.turnover_limit is not None and self.turnover_limit <= 0:
            object.__setattr__(self, "turnover_limit", None)
        if self.vol_lo > self.vol_hi:
            raise ValueError(f"vol_lo {self.vol_lo} > vol_hi {self.vol_hi}")
        if self.vol_steps < 1:
            raise ValueError("vol_steps must be >= 1")

    @property
    def vol_grid(self):
        import numpy as np

        return np.linspace(self.vol_lo, self.vol_hi, self.vol_steps)

    def with_goals(self, goals: Goals, date: Optional[pd.Timestamp] = None) -> OptimizerConfig:
        inc = goals.income_target_at(date) if date is not None else goals.target_income
        return replace(
            self,
            target_return=goals.target_return,
            target_income=inc,
            vol_lo=goals.vol_lo,
            vol_hi=goals.vol_hi,
        )


@dataclass(frozen=True)
class RiskModelConfig:
    halflife: float = 36.0  # EWMA half-life in MONTHS
    shrink: float = 0.25  # constant-correlation shrinkage intensity
    uplift: float = 0.15  # extra vol on illiquid sleeves
    max_lookback: int = 180  # estimation window cap, months
    agg_months: Optional[int] = None  # None -> coarsest nav_frequency
    phi_cap: float = 0.6  # cap on the Geltner smoothing parameter


@dataclass(frozen=True)
class LadderConfig:
    edges: Tuple[float, ...] = (2.0, 3.0)  # year edges between buckets
    cover: float = 1.0  # multiple on scheduled calls

    def __post_init__(self):
        e = tuple(float(x) for x in self.edges)
        if any(b <= a for a, b in zip(e, e[1:])):
            raise ValueError(f"ladder edges must increase, got {e}")
        object.__setattr__(self, "edges", e)


@dataclass(frozen=True)
class BacktestConfig:
    burn_in_months: int = 60
    rebalance_months: int = 3
    review_months: int = 12
    vol_steps: int = 9  # cheaper grid inside the backtest
    cma_blend: float = 0.0  # > 0 -> blend CMAs with trailing realised
    cma_trailing_years: int = 10
    forced_sale_extra_bps: float = 75.0
    default_tx_cost_bps: float = 10.0
    use_ladder: bool = True
    target_window_months: int = 60  # rolling window for the return-goal hit rate

    @property
    def cma_mode(self) -> str:
        return "blend" if self.cma_blend > 0 else "static"


_DEFAULTS: Dict[str, float] = dict(
    cpi=0.024,
    return_spread=0.030,
    target_return=float("nan"),
    target_income=0.035,
    vol_lo=0.06,
    vol_hi=0.08,
    vol_steps=21,
    turnover_limit=0.15,
    income_penalty=25.0,
    hard_return_floor=0.0,
    lexicographic=1.0,
    vol_relax_max=0.0,
    income_target_year=float("nan"),
    income_glide_start_year=float("nan"),
    income_glide_start_income=float("nan"),
    ladder_edge_1=2.0,
    ladder_edge_2=3.0,
    ladder_cover=1.0,
    halflife=36.0,
    shrink=0.25,
    uplift=0.15,
    max_lookback=180,
    burn_in_months=60,
    rebalance_months=3,
    review_months=12,
    bt_vol_steps=9,
    cma_blend=0.0,
    cma_trailing_years=10,
    forced_sale_extra_bps=75.0,
    goal_return_window_months=60,
    illiquid_max_trade=0.01,
)

# accepted for backward compatibility, without effect
_OBSOLETE = {"bt_max_iter"}

# keys whose values are rates in [0, 1] ("4,2" without % is read as 4.2%)
RATE_KEYS = {
    "cpi",
    "return_spread",
    "target_return",
    "target_income",
    "vol_lo",
    "vol_hi",
    "vol_relax_max",
    "turnover_limit",
    "shrink",
    "uplift",
    "cma_blend",
    "income_glide_start_income",
    "illiquid_max_trade",
}


@dataclass(frozen=True)
class Settings:
    goals: Goals = field(default_factory=Goals)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    risk: RiskModelConfig = field(default_factory=RiskModelConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    unknown: Tuple[str, ...] = ()
    raw: Tuple[Tuple[str, float], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, float], n_ladder_buckets: Optional[int] = None) -> Settings:
        s = dict(_DEFAULTS)
        s.update({k: v for k, v in values.items()})
        unknown = tuple(
            sorted(
                k
                for k in values
                if k not in _DEFAULTS and k not in _OBSOLETE and not k.startswith("ladder_edge_")
            )
        )

        goals = Goals(
            cpi=float(s["cpi"]),
            return_spread=float(s["return_spread"]),
            target_income=float(s["target_income"]),
            vol_lo=float(s["vol_lo"]),
            vol_hi=float(s["vol_hi"]),
            target_return_override=float(s["target_return"]),
            income_target_year=float(s["income_target_year"]),
            income_glide_start_year=float(s["income_glide_start_year"]),
            income_glide_start_income=float(s["income_glide_start_income"]),
        )
        tl = float(s["turnover_limit"])
        opt = OptimizerConfig(
            vol_steps=int(s["vol_steps"]),
            turnover_limit=None if tl <= 0 else tl,
            income_penalty=float(s["income_penalty"]),
            hard_return_floor=bool(s["hard_return_floor"]),
            lexicographic=bool(s["lexicographic"]),
            vol_relax_max=float(s["vol_relax_max"]),
            illiquid_max_trade=float(s["illiquid_max_trade"]),
        ).with_goals(goals)
        risk = RiskModelConfig(
            halflife=float(s["halflife"]),
            shrink=float(s["shrink"]),
            uplift=float(s["uplift"]),
            max_lookback=int(s["max_lookback"]),
        )

        # ladder edges: ladder_edge_1 .. ladder_edge_{N-1}
        k_max = (
            (n_ladder_buckets - 1)
            if n_ladder_buckets
            else max([int(k.rsplit("_", 1)[1]) for k in s if k.startswith("ladder_edge_")] + [0])
        )
        edges = []
        for k in range(1, k_max + 1):
            v = s.get(f"ladder_edge_{k}", float("nan"))
            if _isnan(v):
                raise ValueError(
                    f"config.csv: ladder_edge_{k} missing but the "
                    f"universe defines {n_ladder_buckets} ladder buckets"
                )
            edges.append(float(v))
        ladder = LadderConfig(edges=tuple(edges), cover=float(s["ladder_cover"]))

        bt = BacktestConfig(
            burn_in_months=int(s["burn_in_months"]),
            rebalance_months=int(s["rebalance_months"]),
            review_months=int(s["review_months"]),
            vol_steps=int(s["bt_vol_steps"]),
            cma_blend=float(s["cma_blend"]),
            cma_trailing_years=int(s["cma_trailing_years"]),
            forced_sale_extra_bps=float(s["forced_sale_extra_bps"]),
            target_window_months=int(s["goal_return_window_months"]),
        )
        return cls(goals, opt, risk, ladder, bt, unknown, tuple(sorted((k, v) for k, v in values.items())))

    def describe(self) -> pd.DataFrame:
        rows = []
        for section in ("goals", "optimizer", "risk", "ladder", "backtest"):
            obj = getattr(self, section)
            for f in fields(obj):
                rows.append(dict(section=section, key=f.name, value=getattr(obj, f.name)))
        return pd.DataFrame(rows)
