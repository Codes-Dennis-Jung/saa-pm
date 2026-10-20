"""Load an input folder into a :class:`MandateData` bundle.

Required files
    assets.csv     one row per asset class: CMAs, liquidity, bounds, ladder bucket
    prices.csv     month-end TOTAL-RETURN index levels, one column per asset

Optional files
    groups.csv             group (look-through) limits
    pacing.csv             Takahashi-Alexander parameters for private sleeves
    holdings.csv           current market values and undrawn commitments
    config.csv             targets, risk band, model settings (key;value)
    commitment_future.csv  KNOWN future capital calls (sizes the funding ladder)

Optional time-varying files (as-of semantics, used by the backtest)
    cma_history.csv        effective_date, asset, exp_return, income
    policy_history.csv     effective_date, key, value
    flows.csv              date, amount         external contributions (+) / withdrawals (-)
    commitments_actual.csv date, asset, amount  historical capital calls

Optional analysis files
    stress_factors.csv     month-end factor levels for the factor stress test
    stress_scenarios.csv   named stress scenarios (factor shocks or date windows)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import RATE_KEYS, Settings
from ..privates.pacing import PacingParams
from ..universe import FREQ_MONTHS, Asset, GroupConstraint, Universe
from .csvio import as_rate, parse_dates, read_csv, to_bool, to_float, to_list, to_numeric_column, to_str

POLICY_ATTRS_ASSET = {"lb", "ub", "max_trade"}
POLICY_SETTINGS = {
    "target_income",
    "vol_lo",
    "vol_hi",
    "turnover_limit",
    "income_penalty",
    "cpi",
    "return_spread",
}


@dataclass
class Holdings:
    """Current portfolio in currency."""

    market_value: pd.Series  # per asset (universe order)
    undrawn: pd.Series  # undrawn commitments per private sleeve

    @property
    def total(self) -> float:
        return float(self.market_value.sum())

    @property
    def weights(self) -> np.ndarray:
        return (self.market_value / self.total).values

    def private_state(self) -> Dict[str, Tuple[float, float]]:
        """{sleeve: (nav weight, undrawn / PV)}."""
        pv = self.total
        return {
            n: (float(self.market_value[n] / pv), float(self.undrawn.get(n, 0.0) / pv))
            for n in self.undrawn.index
        }


@dataclass
class Panel:
    """Optional time-varying inputs with as-of lookups (no look-ahead)."""

    cma: Optional[pd.DataFrame] = None  # long: effective_date, asset, exp_return, income
    policy: Optional[pd.DataFrame] = None  # wide: effective_date x key
    flows: Optional[pd.Series] = None  # month-end -> currency amount
    calls: Optional[pd.DataFrame] = None  # long: date, asset, amount (historical)

    def cma_asof(self, date) -> Dict[str, Tuple[float, float]]:
        if self.cma is None:
            return {}
        v = self.cma[self.cma["effective_date"] <= date]
        if v.empty:
            return {}
        last = v.sort_values("effective_date").groupby("asset").last()
        return {a: (float(r["exp_return"]), float(r["income"])) for a, r in last.iterrows()}

    def cma_start(self) -> Optional[pd.Timestamp]:
        return None if self.cma is None else self.cma["effective_date"].min()

    def policy_asof(self, date) -> Dict[str, float]:
        if self.policy is None:
            return {}
        v = self.policy[self.policy.index <= date]
        if v.empty:
            return {}
        return {k: float(x) for k, x in v.ffill().iloc[-1].items() if pd.notna(x)}

    def flow_in_month(self, date: pd.Timestamp) -> float:
        if self.flows is None:
            return 0.0
        key = pd.Timestamp(date) + pd.offsets.MonthEnd(0)
        return float(self.flows.get(key, 0.0))

    @property
    def active(self) -> List[str]:
        return [n for n in ("cma", "policy", "flows", "calls") if getattr(self, n) is not None]


@dataclass
class MandateData:
    """Everything read from an input folder."""

    folder: str
    universe: Universe
    returns: pd.DataFrame  # monthly total returns, decimal
    pacing: Dict[str, PacingParams]
    holdings: Holdings
    settings: Settings
    schedule: pd.DataFrame  # future capital calls (long)
    panel: Panel
    stress_factors: Optional[pd.DataFrame] = None
    stress_scenarios: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return self.universe.names

    @property
    def common_returns(self) -> pd.DataFrame:
        """Months in which every asset has data (the estimation window)."""
        return self.returns.dropna()

    @property
    def asof(self) -> pd.Timestamp:
        return self.common_returns.index[-1]

    @classmethod
    def from_frames(
        cls,
        assets: pd.DataFrame,
        prices: pd.DataFrame,
        holdings: pd.DataFrame,
        *,
        config: Optional[Dict[str, float]] = None,
        groups: Optional[pd.DataFrame] = None,
        pacing: Optional[pd.DataFrame] = None,
        commitment_future: Optional[pd.DataFrame] = None,
        cma_history: Optional[pd.DataFrame] = None,
        policy_history: Optional[pd.DataFrame] = None,
        flows: Optional[pd.DataFrame] = None,
        commitments_actual: Optional[pd.DataFrame] = None,
        stress_factors: Optional[pd.DataFrame] = None,
        stress_scenarios: Optional[pd.DataFrame] = None,
        input_type: str = "index",
    ) -> MandateData:
        """Build a mandate from in-memory DataFrames instead of a folder.

        Each frame has the columns of the CSV file of the same name (see the
        module docstring); ``prices`` and ``stress_factors`` are indexed by date,
        ``config`` is a plain ``{key: value}`` dict. The frames go through the
        very same parser and checks as files do.
        """
        import tempfile

        frames = {
            "assets": assets,
            "holdings": holdings,
            "groups": groups,
            "pacing": pacing,
            "commitment_future": commitment_future,
            "cma_history": cma_history,
            "policy_history": policy_history,
            "flows": flows,
            "commitments_actual": commitments_actual,
            "stress_scenarios": stress_scenarios,
        }
        with tempfile.TemporaryDirectory(prefix="saa_pm_") as tmp:
            for name, df in frames.items():
                if df is not None:
                    df.to_csv(os.path.join(tmp, f"{name}.csv"), index=False)
            for name, df in (("prices", prices), ("stress_factors", stress_factors)):
                if df is not None:
                    out = df.copy()
                    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
                    out.index.name = "date"
                    out.to_csv(os.path.join(tmp, f"{name}.csv"))
            if config:
                pd.DataFrame({"key": list(config), "value": list(config.values())}).to_csv(
                    os.path.join(tmp, "config.csv"), index=False
                )
            data = load_mandate(tmp, input_type)
        data.folder = "<in-memory>"
        return data


def _rate(v, where, notes):
    return as_rate(v, where, notes)


def load_assets(path: str, notes: List[str]) -> List[Asset]:
    df = read_csv(path)
    missing = {"name", "group", "exp_return", "income", "liquid"} - set(df.columns)
    if missing:
        raise ValueError(f"assets.csv missing columns: {sorted(missing)}")
    out = []
    for _, r in df.iterrows():
        nm = to_str(r["name"])
        if not nm:
            raise ValueError("assets.csv contains a row without a name")
        er = _rate(to_float(r["exp_return"]), f"assets.csv '{nm}' exp_return", notes)
        inc = _rate(to_float(r["income"]), f"assets.csv '{nm}' income", notes)
        if not (np.isfinite(er) and np.isfinite(inc)):
            raise ValueError(f"assets.csv '{nm}': exp_return and income must be numbers")
        liquid_raw = r.get("liquid")
        if to_str(liquid_raw) == "":
            raise ValueError(f"assets.csv '{nm}': the 'liquid' flag is blank")
        mt = _rate(to_float(r.get("max_trade")), f"assets.csv '{nm}' max_trade", notes)
        bucket = to_float(r.get("ladder_bucket"))
        if np.isfinite(bucket) and bucket != int(bucket):
            raise ValueError(f"assets.csv '{nm}': ladder_bucket must be an integer")
        rl = _rate(to_float(r.get("redemption_limit")), f"assets.csv '{nm}' redemption_limit", notes)
        freq = to_str(r.get("nav_frequency"), "M").upper()
        if freq not in FREQ_MONTHS:
            raise ValueError(f"assets.csv '{nm}': nav_frequency {freq!r} not in {list(FREQ_MONTHS)}")
        out.append(
            Asset(
                name=nm,
                group=to_str(r["group"]),
                exp_return=er,
                income=inc,
                liquid=to_bool(liquid_raw),
                lb=_rate(to_float(r.get("lb"), 0.0), f"assets.csv '{nm}' lb", notes),
                ub=_rate(to_float(r.get("ub"), 1.0), f"assets.csv '{nm}' ub", notes),
                appraisal_based=to_bool(r.get("appraisal_based"), False),
                max_trade=None if not np.isfinite(mt) else mt,
                ladder_bucket=None if not np.isfinite(bucket) else int(bucket),
                tx_cost_bps=to_float(r.get("tx_cost_bps"), 10.0),
                nav_frequency=freq,
                price_column=to_str(r.get("price_column"), nm),
                redemption_limit=None if not np.isfinite(rl) else rl,
            )
        )
    return out


def load_prices(path: str, assets: List[Asset], notes: List[str], input_type: str = "index") -> pd.DataFrame:
    """Month-end total-return levels -> monthly returns (universe order).

    Low-frequency NAVs (nav_frequency Q/SA/A) are CARRIED flat between
    reports: the return of a quarter is booked in the month it is reported.
    Interpolating would leak the next report into earlier months (look-ahead)
    and fake autocorrelation; the risk model aggregates to the reporting
    frequency anyway.
    """
    raw = read_csv(path)
    date_col = raw.columns[0]
    idx = parse_dates(raw[date_col], f"{os.path.basename(path)} column '{date_col}'")
    raw = raw.drop(columns=[date_col])
    raw.index = pd.DatetimeIndex(idx)
    raw = raw.sort_index()
    if raw.index.has_duplicates:
        raise ValueError(
            f"{os.path.basename(path)}: duplicate dates {list(raw.index[raw.index.duplicated()][:3])}"
        )
    num = raw.apply(to_numeric_column)
    blank = raw.isna() | raw.apply(
        lambda c: c.astype(str).str.strip().str.lower().isin(["", "nan", "na", "none"])
    )
    bad = (~blank) & num.isna()
    if bad.any().any():
        c = bad.any()[bad.any()].index[0]
        raise ValueError(
            f"{os.path.basename(path)}: non-numeric value in column '{c}' at {bad[c][bad[c]].index[0].date()}"
        )
    missing = [a.price_col for a in assets if a.price_col not in num.columns]
    if missing:
        raise ValueError(f"prices.csv has no column for {missing}")
    lvl = (1.0 + num).cumprod() if input_type == "returns" else num
    lvl = lvl.resample("ME").last()

    out = {}
    for a in assets:
        s = lvl[a.price_col].astype(float)
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            raise ValueError(f"'{a.name}': price column '{a.price_col}' is empty")
        s = s.loc[first:last]
        if (s.dropna() <= 0).any():
            raise ValueError(f"'{a.name}': non-positive index level - is this a return series?")
        gaps = int(s.isna().sum())
        if gaps:
            if a.nav_months == 1:
                notes.append(
                    f"'{a.name}': {gaps} missing month(s) carried flat - "
                    f"set nav_frequency if it reports less often"
                )
            s = s.ffill()
        out[a.name] = s
    ret = pd.DataFrame(out).pct_change(fill_method=None).iloc[1:]
    return ret


def load_groups(path: str, notes: List[str]) -> List[GroupConstraint]:
    df = read_csv(path)
    out = []
    for _, r in df.iterrows():
        nm = to_str(r["name"])
        lo = _rate(to_float(r.get("lo"), 0.0), f"groups.csv '{nm}' lo", notes)
        hi = _rate(to_float(r.get("hi"), 1.0), f"groups.csv '{nm}' hi", notes)
        members = to_list(r["members"])
        if not members:
            raise ValueError(f"groups.csv '{nm}' has no members (use '|' between names)")
        out.append(GroupConstraint(nm, tuple(members), lo, hi))
    return out


def load_pacing(path: str, assets: Dict[str, Asset], notes: List[str]) -> Dict[str, PacingParams]:
    """``growth_rate`` is FORCED to the sleeve's CMA total return (vintage IRR
    invariant); a growth_rate column in the file is ignored with a note."""
    df = read_csv(path)
    if "growth_rate" in df.columns:
        notes.append(
            "pacing.csv: growth_rate column ignored - growth is set to the CMA "
            "total return of each sleeve (vintage IRR = CMA)"
        )
    out = {}
    for _, r in df.iterrows():
        nm = to_str(r["asset"])
        if nm not in assets:
            raise ValueError(f"pacing.csv references unknown asset '{nm}'")
        rates = tuple(to_float(x) for x in to_list(r["contribution_rates"]))
        if not rates or any(not np.isfinite(v) for v in rates):
            raise ValueError(f"pacing.csv '{nm}': unparseable contribution_rates {r['contribution_rates']!r}")
        if any(v > 1 for v in rates):
            notes.append(f"pacing.csv '{nm}': contribution rates > 1 read as percent")
            rates = tuple(v / 100.0 for v in rates)
        y = _rate(to_float(r.get("yield_rate")), f"pacing.csv '{nm}' yield_rate", notes)
        a = assets[nm]
        p = PacingParams(
            name=nm,
            life_years=int(to_float(r["life_years"])),
            commitment_period=int(to_float(r["commitment_period"])),
            contribution_rates=rates,
            bow_factor=to_float(r.get("bow_factor"), 2.0),
            growth_rate=a.exp_return,
            yield_rate=a.income if np.isnan(y) else y,
            drawdown_ratio=_rate(
                to_float(r.get("drawdown_ratio"), 0.95), f"pacing.csv '{nm}' drawdown_ratio", notes
            ),
            yield_from_cma=bool(np.isnan(y)),
        )
        if p.yield_rate > p.growth_rate + 1e-9:
            notes.append(
                f"pacing.csv '{nm}': yield {p.yield_rate:.2%} > CMA total return "
                f"{p.growth_rate:.2%} - capped (yield only splits the return)"
            )
        out[nm] = p.synced_to(a.exp_return, a.income)
    return out


def load_holdings(path: str, universe: Universe) -> Holdings:
    df = read_csv(path)
    if "asset" not in df.columns or "market_value" not in df.columns:
        raise ValueError("holdings.csv needs columns asset;market_value[;commitments_undrawn]")
    df["asset"] = df["asset"].map(to_str)
    dupes = sorted(set(df["asset"][df["asset"].duplicated()]))
    if dupes:
        raise ValueError(f"holdings.csv has duplicate rows for {dupes}")
    unknown = sorted(set(df["asset"]) - set(universe.names))
    if unknown:
        raise ValueError(f"holdings.csv lists assets not in assets.csv: {unknown}")
    df = df.set_index("asset")
    mv = pd.Series({n: to_float(df["market_value"].get(n), 0.0) for n in universe.names})
    if not np.isfinite(mv).all() or (mv < 0).any():
        raise ValueError(
            f"holdings.csv: blank, NaN or negative market values for "
            f"{list(mv.index[~np.isfinite(mv) | (mv < 0)])}"
        )
    if mv.sum() <= 0:
        raise ValueError("holdings.csv: market values sum to zero")
    und_col = df["commitments_undrawn"] if "commitments_undrawn" in df.columns else None
    undrawn = pd.Series(
        {n: (to_float(und_col.get(n), 0.0) if und_col is not None else 0.0) for n in universe.illiquid_names},
        dtype=float,
    )
    liquid_with_undrawn = [
        n for n in universe.liquid_names if und_col is not None and to_float(und_col.get(n), 0.0) > 0
    ]
    if liquid_with_undrawn:
        raise ValueError(f"holdings.csv: undrawn commitments on liquid assets {liquid_with_undrawn}")
    return Holdings(mv, undrawn)


def load_call_schedule(path: str, universe: Universe, label: str) -> pd.DataFrame:
    df = read_csv(path)
    need = {"date", "asset", "amount"}
    if not need <= set(df.columns):
        raise ValueError(f"{label} needs columns {sorted(need)}, got {list(df.columns)}")
    df = df[df["date"].map(to_str) != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "asset", "amount"])
    df["date"] = parse_dates(df["date"], f"{label} column 'date'")
    df["asset"] = df["asset"].map(to_str)
    df["amount"] = df["amount"].map(to_float)
    bad = sorted(set(df["asset"]) - set(universe.names))
    if bad:
        raise ValueError(f"{label} references unknown assets {bad}")
    liq = sorted({n for n in df["asset"] if universe[n].liquid})
    if liq:
        raise ValueError(f"{label} lists liquid assets {liq} - only private sleeves take calls")
    if df["amount"].isna().any() or (df["amount"] < 0).any():
        raise ValueError(f"{label}: blank or negative amount")
    return df[["date", "asset", "amount"]].sort_values(["date", "asset"]).reset_index(drop=True)


def load_config_values(path: str, notes: List[str]) -> Dict[str, float]:
    df = read_csv(path)
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        k = to_str(r["key"])
        if not k:
            continue
        v = to_float(r["value"])
        if not np.isfinite(v):
            notes.append(f"config.csv: '{k}' is blank - default used")
            continue
        out[k] = _rate(v, f"config.csv {k}", notes) if k in RATE_KEYS else v
    return out


def _is_rate_policy_key(k: str) -> bool:
    p = k.split(".")
    return k in RATE_KEYS or (
        len(p) == 3
        and ((p[0] == "asset" and p[2] in POLICY_ATTRS_ASSET) or (p[0] == "group" and p[2] in ("lo", "hi")))
    )


def load_panel(folder: str, universe: Universe, notes: List[str]) -> Panel:
    p = Panel()
    f = os.path.join(folder, "cma_history.csv")
    if os.path.exists(f):
        df = read_csv(f)
        df["effective_date"] = parse_dates(df["effective_date"], "cma_history.csv effective_date")
        df["asset"] = df["asset"].map(to_str)
        for c in ("exp_return", "income"):
            df[c] = [_rate(to_float(v), f"cma_history.csv {c}", notes) for v in df[c]]
        if df[["exp_return", "income"]].isna().any().any():
            raise ValueError("cma_history.csv has blank exp_return or income")
        unknown = sorted(set(df["asset"]) - set(universe.names))
        if unknown:
            raise ValueError(f"cma_history.csv references unknown assets {unknown}")
        p.cma = df[["effective_date", "asset", "exp_return", "income"]]

    f = os.path.join(folder, "policy_history.csv")
    if os.path.exists(f):
        df = read_csv(f)
        df["effective_date"] = parse_dates(df["effective_date"], "policy_history.csv effective_date")
        df["key"] = df["key"].map(to_str)
        bad = [k for k in df["key"].unique() if not _valid_policy_key(k, universe)]
        if bad:
            raise ValueError(
                f"policy_history.csv: unrecognised keys {bad}. Valid: "
                f"{sorted(POLICY_SETTINGS)}, asset.<name>.lb|ub|max_trade, "
                f"group.<name>.lo|hi"
            )
        df["value"] = [
            _rate(to_float(v), f"policy_history.csv {k}", notes) if _is_rate_policy_key(k) else to_float(v)
            for k, v in zip(df["key"], df["value"])
        ]
        p.policy = df.pivot_table(
            index="effective_date", columns="key", values="value", aggfunc="last"
        ).sort_index()

    f = os.path.join(folder, "flows.csv")
    if os.path.exists(f):
        df = read_csv(f)
        df["date"] = parse_dates(df["date"], "flows.csv date")
        df["amount"] = df["amount"].map(to_float)
        s = df.groupby("date")["amount"].sum().sort_index().resample("ME").sum()
        p.flows = s[s != 0] if (s != 0).any() else None

    f = os.path.join(folder, "commitments_actual.csv")
    if os.path.exists(f):
        calls = load_call_schedule(f, universe, "commitments_actual.csv")
        p.calls = calls if len(calls) else None
    return p


def _valid_policy_key(k: str, universe: Universe) -> bool:
    parts = k.split(".")
    if len(parts) == 1:
        return k in POLICY_SETTINGS
    if len(parts) == 3 and parts[0] == "asset":
        return parts[1] in universe and parts[2] in POLICY_ATTRS_ASSET
    if len(parts) == 3 and parts[0] == "group":
        return parts[1] in {g.name for g in universe.groups} and parts[2] in ("lo", "hi")
    return False


def load_levels(path: str) -> pd.DataFrame:
    """Generic month-end level file (first column = date) -> monthly levels."""
    raw = read_csv(path)
    date_col = raw.columns[0]
    idx = parse_dates(raw[date_col], f"{os.path.basename(path)} column '{date_col}'")
    df = raw.drop(columns=[date_col]).apply(to_numeric_column)
    df.index = pd.DatetimeIndex(idx)
    return df.sort_index().resample("ME").last()


def load_stress_scenarios(path: str) -> pd.DataFrame:
    df = read_csv(path)
    df["name"] = df["name"].map(to_str)
    for c in ("start", "end"):
        if c in df.columns:
            vals = df[c].map(to_str)
            df[c] = pd.NaT
            mask = vals != ""
            if mask.any():
                df.loc[mask, c] = parse_dates(vals[mask], f"stress_scenarios.csv {c}")
            df[c] = pd.to_datetime(df[c])
    for c in df.columns:
        if c not in ("name", "start", "end"):
            df[c] = df[c].map(to_float)
    return df


def load_mandate(folder: str, input_type: str = "index") -> MandateData:
    """Load and parse every input file in ``folder`` (see module docstring)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"input folder not found: {folder}")

    def p(f):
        return os.path.join(folder, f)

    notes: List[str] = []

    assets = load_assets(p("assets.csv"), notes)
    groups = load_groups(p("groups.csv"), notes) if os.path.exists(p("groups.csv")) else []
    universe = Universe(assets, groups)
    returns = load_prices(p("prices.csv"), assets, notes, input_type)
    pacing = (
        load_pacing(p("pacing.csv"), {a.name: a for a in assets}, notes)
        if os.path.exists(p("pacing.csv"))
        else {}
    )
    config_values = load_config_values(p("config.csv"), notes) if os.path.exists(p("config.csv")) else {}
    settings = Settings.from_mapping(config_values, n_ladder_buckets=len(universe.ladder_map) or None)
    if settings.unknown:
        notes.append(f"config.csv: unknown keys ignored {list(settings.unknown)}")

    if os.path.exists(p("holdings.csv")):
        holdings = load_holdings(p("holdings.csv"), universe)
    else:
        raise FileNotFoundError(
            f"{p('holdings.csv')} is required: the current portfolio in currency "
            f"(asset;market_value;commitments_undrawn)"
        )

    schedule = (
        load_call_schedule(p("commitment_future.csv"), universe, "commitment_future.csv")
        if os.path.exists(p("commitment_future.csv"))
        else pd.DataFrame(columns=["date", "asset", "amount"])
    )
    panel = load_panel(folder, universe, notes)

    stress_f = load_levels(p("stress_factors.csv")) if os.path.exists(p("stress_factors.csv")) else None
    stress_s = (
        load_stress_scenarios(p("stress_scenarios.csv"))
        if os.path.exists(p("stress_scenarios.csv"))
        else None
    )
    return MandateData(
        folder, universe, returns, pacing, holdings, settings, schedule, panel, stress_f, stress_s, notes
    )
