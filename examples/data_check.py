"""Input data check: does every input file arrive in the model correctly?

Two tools, usable on any input folder (the example data or your own):

``check_inputs(folder)``
    1. inventory of the input files (present, rows, columns, separator);
    2. loads the folder exactly as the study does (``load_mandate`` +
       ``validate``) - blocking problems become FAIL rows;
    3. RECONCILES every file against what the model holds: each raw value is
       compared with the loaded value, so a dropped row, a mapped-away column,
       a percent read as a fraction or a silently carried price shows up;
    4. plausibility checks the loader does not make: return outliers and
       typo spikes, stale prices, gaps, reporting months of quarterly NAVs,
       several dates in one month, series ending before the others, unknown
       config keys, scenarios that reference unknown factors, ...

``stress_test_inputs(folder)``
    fault injection: copies the folder, applies a FORMAT VARIANT (German
    locale, percent strings, shuffled rows/columns, BOM, padding, tabs - must
    load *identically*) or an injected FAULT (typo in a name, text in a price,
    duplicate dates, a x10 spike, stale prices, bad bounds, bad pacing, ... -
    must be *caught*), and reports whether each expectation holds.

Command line::

    python examples/data_check.py examples/data            # data check
    python examples/data_check.py examples/data --stress   # + fault injection
"""

from __future__ import annotations

import difflib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import saa_pm  # noqa: F401
except ImportError:  # running from a clone without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from saa_pm.config import _DEFAULTS, RATE_KEYS
from saa_pm.io.csvio import parse_dates, read_csv, sniff_separator, to_bool, to_float, to_list, to_str
from saa_pm.io.loaders import MandateData, load_mandate
from saa_pm.io.validation import validate

REQUIRED = ("assets", "prices", "holdings")
OPTIONAL = (
    "config",
    "groups",
    "pacing",
    "commitment_future",
    "commitments_actual",
    "cma_history",
    "policy_history",
    "flows",
    "stress_factors",
    "stress_scenarios",
)
TOL = 1e-9

PASS, INFO, WARN, FAIL = "PASS", "INFO", "WARN", "FAIL"
_ORDER = {FAIL: 0, WARN: 1, INFO: 2, PASS: 3}


@dataclass
class DataCheckReport:
    rows: List[dict] = field(default_factory=list)
    data: Optional[MandateData] = None

    def add(self, file: str, check: str, status: str, detail: str = "") -> None:
        self.rows.append(dict(file=file, check=check, status=status, detail=detail))

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["file", "check", "status", "detail"])

    @property
    def ok(self) -> bool:
        return not any(r["status"] == FAIL for r in self.rows)

    def summary(self) -> pd.DataFrame:
        """Count of PASS / INFO / WARN / FAIL per file."""
        t = self.table
        out = t.pivot_table(index="file", columns="status", values="check", aggfunc="count", fill_value=0)
        return out.reindex(columns=[PASS, INFO, WARN, FAIL], fill_value=0)

    def issues(self, levels=(FAIL, WARN)) -> pd.DataFrame:
        t = self.table
        t = t[t["status"].isin(levels)]
        return t.sort_values("status", key=lambda s: s.map(_ORDER), kind="stable").reset_index(drop=True)

    def raise_if_failed(self) -> None:
        if not self.ok:
            bad = self.issues((FAIL,))
            msg = "\n  - ".join(f"[{r.file}] {r.check}: {r.detail}" for r in bad.itertuples())
            raise ValueError("input data check failed:\n  - " + msg)

    def text(self) -> str:
        s = self.summary()
        lines = [
            f"DATA CHECK: {'OK' if self.ok else 'FAILED'} - "
            + ", ".join(f"{int(s[c].sum())} {c}" for c in s.columns)
        ]
        for r in self.issues().itertuples():
            lines.append(f"  {r.status} [{r.file}] {r.check}: {r.detail}")
        return "\n".join(lines)


def _path(folder: str, name: str) -> str:
    return os.path.join(folder, f"{name}.csv")


def _rate(v: float) -> float:
    """The loader's rule: a rate > 1 is read as a percent."""
    return v / 100.0 if np.isfinite(v) and abs(v) > 1.0 else v


def _same(a, b, tol: float = TOL) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def _fmt_dates(idx, n: int = 4) -> str:
    d = [f"{pd.Timestamp(x):%Y-%m}" for x in list(idx)[:n]]
    return ", ".join(d) + (f" (+{len(idx) - n} more)" if len(idx) > n else "")


def _robust_z(r: pd.Series) -> pd.Series:
    med = r.median()
    mad = (r - med).abs().median() * 1.4826
    return (r - med) / mad if mad > 0 else pd.Series(0.0, index=r.index)


def _inventory(folder: str, rep: DataCheckReport) -> Dict[str, pd.DataFrame]:
    raw: Dict[str, pd.DataFrame] = {}
    for name in REQUIRED + OPTIONAL:
        p = _path(folder, name)
        if not os.path.exists(p):
            rep.add(
                f"{name}.csv",
                "file present",
                FAIL if name in REQUIRED else INFO,
                "required file missing" if name in REQUIRED else "optional - not provided",
            )
            continue
        try:
            df = read_csv(p)
            sep = sniff_separator(p)
        except Exception as exc:
            rep.add(f"{name}.csv", "file readable", FAIL, str(exc))
            continue
        raw[name] = df
        sep_name = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(sep, repr(sep))
        rep.add(
            f"{name}.csv",
            "file present",
            PASS,
            f"{len(df)} rows x {df.shape[1]} columns, {sep_name}-separated",
        )
    known = {f"{n}.csv" for n in REQUIRED + OPTIONAL}
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".csv") and f not in known:
            close = difflib.get_close_matches(f, known, n=1)
            hint = f" - did you mean {close[0]}?" if close else ""
            rep.add(f, "file recognised", WARN, f"not an input file name, IGNORED{hint}")
    return raw


def _check_assets(df: pd.DataFrame, data: MandateData, rep: DataCheckReport) -> None:
    F = "assets.csv"
    u = data.universe
    names = df["name"].map(to_str)
    dup = sorted(set(names[names.duplicated()]))
    rep.add(
        F, "unique asset names", FAIL if dup else PASS, f"duplicates {dup}" if dup else f"{len(names)} names"
    )
    rep.add(F, "every row loaded", PASS if len(df) == u.n else FAIL, f"{len(df)} rows -> {u.n} assets")
    fields = {
        "exp_return": lambda r: _rate(to_float(r["exp_return"])),
        "income": lambda r: _rate(to_float(r["income"])),
        "lb": lambda r: _rate(to_float(r.get("lb"), 0.0)),
        "ub": lambda r: _rate(to_float(r.get("ub"), 1.0)),
        "liquid": lambda r: to_bool(r["liquid"]),
        "appraisal_based": lambda r: to_bool(r.get("appraisal_based"), False),
        "nav_frequency": lambda r: to_str(r.get("nav_frequency"), "M").upper(),
        "price_column": lambda r: to_str(r.get("price_column"), to_str(r["name"])),
    }
    bad, pct = [], []
    for _, r in df.iterrows():
        n = to_str(r["name"])
        if n not in u:
            bad.append(f"'{n}' not in the universe")
            continue
        a = u[n]
        for f, get in fields.items():
            if not _same(get(r), getattr(a, f)):
                bad.append(f"'{n}'.{f}: file {get(r)!r} vs model {getattr(a, f)!r}")
        lb_raw = to_float(r.get("ladder_bucket"))
        if not _same(lb_raw if np.isfinite(lb_raw) else None, a.ladder_bucket):
            bad.append(f"'{n}'.ladder_bucket: file {lb_raw} vs model {a.ladder_bucket}")
        for f in ("exp_return", "income", "lb", "ub"):
            v = to_float(r.get(f))
            if np.isfinite(v) and abs(v) > 1.0:
                pct.append(f"'{n}'.{f}={v:g} read as {v / 100:g}")
    rep.add(
        F,
        "values reconcile with the model",
        FAIL if bad else PASS,
        "; ".join(bad[:5]) if bad else f"{len(fields) + 1} fields x {u.n} assets identical",
    )
    if pct:
        rep.add(F, "rates written as percent", WARN, "; ".join(pct[:4]) + " - confirm the unit")
    odd = [f"'{a.name}' return {a.exp_return:.2%}" for a in u if not -0.05 <= a.exp_return <= 0.30]
    odd += [f"'{a.name}' income {a.income:.2%}" for a in u if not 0.0 <= a.income <= 0.20]
    rep.add(F, "return / income plausible (-5..30% / 0..20%)", WARN if odd else PASS, "; ".join(odd))
    flip = [a.name for a in u if a.lb > a.ub + TOL]
    rep.add(F, "lb <= ub", FAIL if flip else PASS, f"{flip}" if flip else "")
    s_lb, s_ub = float(u.lb.sum()), float(u.ub.sum())
    rep.add(
        F,
        "sum of bounds brackets 100%",
        PASS if s_lb <= 1 + TOL and s_ub >= 1 - TOL else FAIL,
        f"sum lb {s_lb:.1%}, sum ub {s_ub:.1%}",
    )
    no_lad = not u.ladder_map
    if no_lad and not u.illiquid_names:
        detail = "no ladder - not needed: no closed-end sleeves, so no capital calls"
    elif no_lad:
        detail = "no ladder_bucket - capital calls cannot be pre-funded"
    else:
        detail = "; ".join(f"bucket {k}: {'|'.join(v)}" for k, v in u.ladder_map.items())
    rep.add(F, "funding ladder defined", WARN if no_lad and u.illiquid_names else PASS, detail)


def _check_prices(
    df: pd.DataFrame, data: MandateData, rep: DataCheckReport, outlier_z: float, max_abs: float
) -> None:
    F = "prices.csv"
    u = data.universe
    dcol = df.columns[0]
    dates = pd.DatetimeIndex(parse_dates(df[dcol], F))
    lv = df.drop(columns=[dcol]).apply(lambda c: c.map(to_float))
    lv.index = dates
    lv = lv.sort_index()
    rep.add(F, "date range", INFO, f"{dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}, {len(dates)} rows")
    not_me = dates[~dates.is_month_end]
    if len(not_me):
        rep.add(
            F,
            "dates are month-ends",
            WARN,
            f"{len(not_me)} dates not month-end (e.g. {not_me[0]:%Y-%m-%d}) "
            "- each month keeps its LAST value",
        )
    per_month = pd.Series(1, index=dates).groupby(dates.to_period("M")).sum()
    multi = per_month[per_month > 1]
    rep.add(
        F,
        "one row per month",
        WARN if len(multi) else PASS,
        f"{len(multi)} month(s) with several rows, only the last is used: "
        f"{', '.join(str(p) for p in multi.index[:4])}"
        if len(multi)
        else "",
    )
    used = {a.price_col for a in u}
    unused = [c for c in lv.columns if c not in used]
    if unused:
        rep.add(F, "unused columns", INFO, f"not referenced by assets.csv: {unused}")

    # recompute monthly returns independently and compare with the model
    lvl = lv.resample("ME").last()
    worst, worst_name = 0.0, ""
    for a in u:
        s = lvl[a.price_col]
        s = s.loc[s.first_valid_index() : s.last_valid_index()].ffill()
        r = s.pct_change(fill_method=None).iloc[1:]
        m = data.returns[a.name].dropna()
        r = r.reindex(m.index)
        d = float((r - m).abs().max()) if len(m) else np.inf
        if not np.isfinite(d) or d > worst or not worst_name:
            worst, worst_name = d, a.name
    rep.add(
        F,
        "returns reconcile (file levels -> model returns)",
        PASS if worst <= 1e-10 else FAIL,
        f"max abs difference {worst:.1e} ({worst_name})",
    )

    asof = data.asof
    for a in u:
        s = lvl[a.price_col]
        first, last = s.first_valid_index(), s.last_valid_index()
        body = s.loc[first:last]
        tag = f"{a.name}"
        if last < asof or last < lvl.index.max() - pd.offsets.MonthEnd(0):
            rep.add(
                F,
                f"{tag}: series up to date",
                WARN,
                f"ends {last:%Y-%m} while the file runs to {lvl.index.max():%Y-%m} - the common "
                f"window (and the as-of date) is cut back to the shortest series",
            )
        rep_months = body.dropna().index
        if a.nav_months == 1:
            gaps = body.index[body.isna()]
            if len(gaps):
                rep.add(
                    F,
                    f"{tag}: no gaps",
                    WARN,
                    f"{len(gaps)} blank month(s) carried flat (zero return): {_fmt_dates(gaps)}",
                )
            r = body.ffill().pct_change(fill_method=None).iloc[1:]
            zero = (r.abs() < 1e-12).astype(int)
            runs = zero.groupby((zero != zero.shift()).cumsum()).transform("sum") * zero
            if runs.max() >= 3:
                end = runs.idxmax()
                rep.add(
                    F,
                    f"{tag}: no stale prices",
                    WARN,
                    f"{int(runs.max())} consecutive unchanged months ending {end:%Y-%m} - stale feed?",
                )
        else:
            off = [d for d in rep_months[1:] if d.month % a.nav_months != 0]
            if off:
                rep.add(
                    F,
                    f"{tag}: reports on {a.nav_frequency} calendar",
                    WARN,
                    f"{len(off)} report(s) outside the {a.nav_frequency} calendar: {_fmt_dates(off)}",
                )
            r = body.dropna().pct_change().iloc[1:]
        r = r.dropna()
        if len(r) < 8:
            continue
        lim = max_abs * np.sqrt(a.nav_months)
        z = _robust_z(r)
        out = r[(r.abs() > lim) | (z.abs() > outlier_z)]
        spikes = [
            d
            for d in out.index
            if d in r.index
            and r.index.get_loc(d) + 1 < len(r)
            and r.loc[d] * r.iloc[r.index.get_loc(d) + 1] < 0
            and min(abs(r.loc[d]), abs(r.iloc[r.index.get_loc(d) + 1])) > lim / 2
        ]
        if spikes:
            rep.add(
                F,
                f"{tag}: no typo spikes",
                FAIL,
                f"jump and reversal at {_fmt_dates(spikes)} "
                f"({r.loc[spikes[0]]:+.0%} then back) - a mistyped level?",
            )
        elif len(out):
            rep.add(
                F,
                f"{tag}: no return outliers",
                WARN,
                f"{len(out)} period(s) beyond {lim:.0%} or {outlier_z:.0f} robust sigma: "
                + ", ".join(f"{d:%Y-%m} {v:+.1%}" for d, v in out.head(3).items()),
            )
    n_flag = sum(1 for r in rep.rows if r["file"] == F and r["status"] in (WARN, FAIL) and ":" in r["check"])
    if n_flag == 0:
        rep.add(F, "per-series checks (gaps, stale, calendar, outliers, spikes)", PASS, f"{u.n} series clean")
    rep.add(F, "as-of date", INFO, f"{asof:%Y-%m-%d}; common window {len(data.common_returns)} months")


def _check_holdings(df: pd.DataFrame, data: MandateData, rep: DataCheckReport) -> None:
    F = "holdings.csv"
    h = data.holdings
    df = df.assign(asset=df["asset"].map(to_str))
    raw_mv = df.set_index("asset")["market_value"].map(lambda v: to_float(v, 0.0))
    rep.add(
        F,
        "total reconciles",
        PASS if _same(float(raw_mv.sum()), h.total, 1e-12) else FAIL,
        f"file {raw_mv.sum():,.2f} vs model {h.total:,.2f}",
    )
    diff = [n for n in raw_mv.index if n in h.market_value and not _same(raw_mv[n], float(h.market_value[n]))]
    rep.add(
        F, "every position reconciles", FAIL if diff else PASS, f"{diff}" if diff else f"{len(raw_mv)} rows"
    )
    missing = [n for n in data.names if n not in raw_mv.index]
    if missing:
        rep.add(
            F, "every asset listed", INFO, f"{len(missing)} asset(s) not listed, held at 0: {missing[:5]}"
        )
    rep.add(
        F, "weights sum to 100%", PASS if abs(h.weights.sum() - 1) < 1e-12 else FAIL, f"{h.weights.sum():.6f}"
    )
    if "commitments_undrawn" in df.columns:
        und = df.set_index("asset")["commitments_undrawn"].map(lambda v: to_float(v, 0.0))
        tot = float(und[[n for n in und.index if n in data.universe.illiquid_names]].sum())
        rep.add(
            F,
            "undrawn commitments reconcile",
            PASS if _same(tot, float(h.undrawn.sum())) else FAIL,
            f"file {tot:,.0f} vs model {h.undrawn.sum():,.0f}",
        )


def _check_groups(df: pd.DataFrame, data: MandateData, rep: DataCheckReport) -> None:
    F = "groups.csv"
    u = data.universe
    loaded = {g.name: g for g in u.groups}
    bad, unknown = [], []
    for _, r in df.iterrows():
        n = to_str(r["name"])
        g = loaded.get(n)
        if g is None:
            bad.append(f"'{n}' not loaded")
            continue
        if tuple(to_list(r["members"])) != tuple(g.members):
            bad.append(f"'{n}' members differ")
        if not _same(_rate(to_float(r.get("lo"), 0.0)), g.lo) or not _same(
            _rate(to_float(r.get("hi"), 1.0)), g.hi
        ):
            bad.append(f"'{n}' limits differ")
        unknown += [f"'{n}': {m}" for m in g.members if m not in u]
    rep.add(F, "groups reconcile", FAIL if bad else PASS, "; ".join(bad) if bad else f"{len(df)} groups")
    rep.add(F, "members exist in assets.csv", FAIL if unknown else PASS, "; ".join(unknown[:4]))
    in_group = {m for g in u.groups for m in g.members}
    free = [n for n in u.names if n not in in_group]
    if free:
        rep.add(F, "every asset in a group", INFO, f"only single-asset bounds apply to {free}")


def _check_config(df: pd.DataFrame, data: MandateData, rep: DataCheckReport) -> None:
    F = "config.csv"
    s = data.settings
    raw = dict(s.raw)
    known = set(_DEFAULTS) | {"bt_max_iter"}
    unknown, bad, pct = [], [], []
    for _, r in df.iterrows():
        k = to_str(r["key"])
        if not k:
            continue
        v = to_float(r["value"])
        if not k.startswith("ladder_edge_") and k not in known:
            close = difflib.get_close_matches(k, sorted(known), n=1)
            unknown.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        if not np.isfinite(v):
            continue
        exp = _rate(v) if k in RATE_KEYS else v
        if k in raw and not _same(exp, float(raw[k])):
            bad.append(f"{k}: file {v} vs model {raw[k]}")
        if k in RATE_KEYS and abs(v) > 1:
            pct.append(f"{k}={v:g} read as {v / 100:g}")
    rep.add(
        F,
        "known keys",
        WARN if unknown else PASS,
        ("IGNORED: " + ", ".join(unknown)) if unknown else f"{len(df)} keys",
    )
    rep.add(F, "values reconcile", FAIL if bad else PASS, "; ".join(bad))
    if pct:
        rep.add(F, "rates written as percent", WARN, "; ".join(pct))
    g = s.goals
    rep.add(
        F,
        "goals in force",
        INFO,
        f"return >= {g.target_return:.2%} (CPI {g.cpi:.2%} + {g.return_spread:.2%}), "
        f"income >= {g.target_income:.2%}, vol {g.vol_lo:.1%}-{g.vol_hi:.1%}",
    )


def _check_pacing(df: pd.DataFrame, data: MandateData, rep: DataCheckReport) -> None:
    F = "pacing.csv"
    u = data.universe
    bad = []
    for _, r in df.iterrows():
        n = to_str(r["asset"])
        p = data.pacing.get(n)
        if p is None:
            bad.append(f"'{n}' not loaded")
            continue
        rates = tuple(to_float(x) for x in to_list(r["contribution_rates"]))
        rates = tuple(v / 100 for v in rates) if any(v > 1 for v in rates) else rates
        for f, v in (
            ("life_years", int(to_float(r["life_years"]))),
            ("commitment_period", int(to_float(r["commitment_period"]))),
            ("contribution_rates", rates),
            ("bow_factor", to_float(r.get("bow_factor"), 2.0)),
        ):
            if not (np.allclose(v, getattr(p, f)) if f == "contribution_rates" else _same(v, getattr(p, f))):
                bad.append(f"'{n}'.{f}: file {v} vs model {getattr(p, f)}")
        if not _same(p.growth_rate, u[n].exp_return):
            bad.append(f"'{n}': growth {p.growth_rate} != CMA {u[n].exp_return}")
    rep.add(
        F, "pacing reconciles", FAIL if bad else PASS, "; ".join(bad[:4]) if bad else f"{len(df)} sleeves"
    )
    missing = [n for n in u.illiquid_names if n not in data.pacing]
    rep.add(F, "every private sleeve has pacing", WARN if missing else PASS, f"{missing}" if missing else "")
    probs = [m for p in data.pacing.values() for m in p.problems() if "yield" not in m]
    rep.add(F, "pacing calibration sane", FAIL if probs else PASS, "; ".join(probs[:3]))


def _check_schedule(
    df: pd.DataFrame, loaded: pd.DataFrame, name: str, data: MandateData, rep: DataCheckReport, future: bool
) -> None:
    F = f"{name}.csv"
    df = df[df["date"].map(to_str) != ""]
    tot = float(df["amount"].map(to_float).sum())
    rep.add(
        F,
        "rows and total reconcile",
        PASS if len(df) == len(loaded) and _same(tot, float(loaded["amount"].sum())) else FAIL,
        f"{len(df)} rows / {tot:,.0f} in file vs {len(loaded)} / {loaded['amount'].sum():,.0f} loaded",
    )
    if not len(loaded):
        return
    d = pd.to_datetime(loaded["date"])
    if future:
        past = loaded[d <= data.asof]
        rep.add(
            F,
            "calls lie after the as-of date",
            WARN if len(past) else PASS,
            f"{len(past)} call(s) on/before {data.asof:%Y-%m-%d} are treated as already paid"
            if len(past)
            else f"{d.min():%Y-%m} .. {d.max():%Y-%m}",
        )
    else:
        out = loaded[(d < data.returns.index.min()) | (d > data.asof)]
        rep.add(
            F,
            "calls lie inside the price history",
            WARN if len(out) else PASS,
            f"{len(out)} call(s) outside {data.returns.index.min():%Y-%m}..{data.asof:%Y-%m} "
            "are never replayed"
            if len(out)
            else f"{d.min():%Y-%m} .. {d.max():%Y-%m}",
        )
    by = loaded.groupby("asset")["amount"].sum()
    rep.add(F, "total by sleeve", INFO, ", ".join(f"{k} {v:,.0f}" for k, v in by.items()))


def _check_panel(raw: Dict[str, pd.DataFrame], data: MandateData, rep: DataCheckReport) -> None:
    p, u = data.panel, data.universe
    start, end = data.returns.index.min(), data.asof
    if "cma_history" in raw:
        F = "cma_history.csv"
        df = raw["cma_history"]
        rep.add(
            F,
            "rows reconcile",
            PASS if p.cma is not None and len(p.cma) == len(df) else FAIL,
            f"{len(df)} rows -> {0 if p.cma is None else len(p.cma)} loaded",
        )
        if p.cma is not None:
            cover = p.cma.groupby("effective_date")["asset"].nunique()
            partial = cover[cover < u.n]
            rep.add(
                F,
                "every vintage covers every asset",
                WARN if len(partial) else PASS,
                f"{len(partial)} vintage(s) incomplete - missing assets keep the previous value "
                f"(first: {partial.index[0]:%Y-%m-%d}, {int(partial.iloc[0])}/{u.n})"
                if len(partial)
                else f"{len(cover)} vintages x {u.n} assets",
            )
            last = p.cma[p.cma["effective_date"] == p.cma["effective_date"].max()].set_index("asset")
            gap = {a.name: last.loc[a.name, "exp_return"] - a.exp_return for a in u if a.name in last.index}
            big = {k: v for k, v in gap.items() if abs(v) > 0.01}
            rep.add(
                F,
                "latest vintage vs assets.csv",
                INFO if big else PASS,
                "the backtest uses the vintages, today's decision uses them too (as-of): "
                + ", ".join(f"{k} {v:+.2%}" for k, v in list(big.items())[:4])
                if big
                else "within 1%",
            )
    if "policy_history" in raw:
        F = "policy_history.csv"
        n = 0 if p.policy is None else int(p.policy.notna().sum().sum())
        rep.add(
            F,
            "rows reconcile",
            PASS if n == len(raw["policy_history"]) else WARN,
            f"{len(raw['policy_history'])} rows -> {n} values loaded",
        )
        if p.policy is not None:
            out = p.policy.index[(p.policy.index < start) | (p.policy.index > end)]
            rep.add(
                F, "dates inside the history", WARN if len(out) else PASS, _fmt_dates(out) if len(out) else ""
            )
    if "flows" in raw:
        F = "flows.csv"
        tot_raw = float(raw["flows"]["amount"].map(to_float).sum())
        tot = 0.0 if p.flows is None else float(p.flows.sum())
        rep.add(
            F, "total reconciles", PASS if _same(tot_raw, tot) else FAIL, f"file {tot_raw:,.0f} vs {tot:,.0f}"
        )
        if p.flows is not None:
            out = p.flows.index[(p.flows.index < start) | (p.flows.index > end)]
            rep.add(
                F,
                "dates inside the history",
                WARN if len(out) else PASS,
                f"never applied: {_fmt_dates(out)}" if len(out) else "",
            )


def _check_stress(raw: Dict[str, pd.DataFrame], data: MandateData, rep: DataCheckReport) -> None:
    fac, sc = data.stress_factors, data.stress_scenarios
    if fac is not None:
        F = "stress_factors.csv"
        both = fac.pct_change(fill_method=None).join(data.common_returns, how="inner").dropna()
        n_q = len(both) // max(1, data.universe.estimation_months)
        rep.add(
            F,
            "overlap with the asset history",
            PASS if n_q >= 20 else FAIL,
            f"{len(both)} joint months = {n_q} estimation periods (need >= 20)",
        )
        gaps = fac.index[fac.isna().any(axis=1)]
        rep.add(
            F,
            "no gaps",
            WARN if len(gaps) else PASS,
            _fmt_dates(gaps) if len(gaps) else f"{fac.shape[1]} factors",
        )
    if sc is not None:
        F = "stress_scenarios.csv"
        facs = set() if fac is None else set(fac.columns)
        shock_cols = [c for c in sc.columns if c not in ("name", "start", "end")]
        unknown = [c for c in shock_cols if c not in facs]
        rep.add(
            F,
            "shock columns are known factors",
            FAIL if unknown else PASS,
            f"not in stress_factors.csv: {unknown}" if unknown else f"{len(shock_cols)} factors",
        )
        empty, outside = [], []
        for r in sc.itertuples():
            has_shock = any(pd.notna(getattr(r, c, np.nan)) for c in shock_cols if c.isidentifier())
            has_win = "start" in sc and pd.notna(getattr(r, "start", pd.NaT))
            if not has_shock and not has_win:
                empty.append(r.name)
            if has_win and fac is not None and (r.start < fac.index.min() or r.end > fac.index.max()):
                outside.append(r.name)
        rep.add(
            F,
            "every scenario has shocks or a window",
            FAIL if empty else PASS,
            f"{empty}" if empty else f"{len(sc)}",
        )
        rep.add(
            F, "windows inside the factor history", FAIL if outside else PASS, f"{outside}" if outside else ""
        )


def check_inputs(folder: str, outlier_z: float = 8.0, max_abs_monthly: float = 0.25) -> DataCheckReport:
    """Inventory, load, validate, reconcile and plausibility-check a folder."""
    rep = DataCheckReport()
    raw = _inventory(folder, rep)
    if any(r["status"] == FAIL for r in rep.rows):
        return rep
    try:
        data = load_mandate(folder)
    except Exception as exc:
        rep.add("(load)", "folder loads", FAIL, f"{type(exc).__name__}: {exc}")
        return rep
    rep.data = data
    rep.add("(load)", "folder loads", PASS, f"{data.universe}, {len(data.returns)} months")
    try:
        v = validate(data)
        for e in v.errors:
            rep.add("(validate)", "blocking error", FAIL, e)
        for w in v.warnings:
            rep.add("(validate)", "warning", WARN, w)
        for n in v.notes:
            rep.add("(validate)", "note", INFO, n)
        if v.ok:
            rep.add("(validate)", "no blocking errors", PASS, f"{len(v.warnings)} warning(s)")
    except Exception as exc:
        rep.add("(validate)", "validation runs", FAIL, f"{type(exc).__name__}: {exc}")

    steps: List[Callable[[], None]] = [
        lambda: _check_assets(raw["assets"], data, rep),
        lambda: _check_prices(raw["prices"], data, rep, outlier_z, max_abs_monthly),
        lambda: _check_holdings(raw["holdings"], data, rep),
    ]
    if "groups" in raw:
        steps.append(lambda: _check_groups(raw["groups"], data, rep))
    if "config" in raw:
        steps.append(lambda: _check_config(raw["config"], data, rep))
    if "pacing" in raw:
        steps.append(lambda: _check_pacing(raw["pacing"], data, rep))
    if "commitment_future" in raw:
        steps.append(
            lambda: _check_schedule(
                raw["commitment_future"], data.schedule, "commitment_future", data, rep, True
            )
        )
    if "commitments_actual" in raw:
        calls = (
            data.panel.calls
            if data.panel.calls is not None
            else pd.DataFrame(columns=["date", "asset", "amount"])
        )
        steps.append(
            lambda: _check_schedule(raw["commitments_actual"], calls, "commitments_actual", data, rep, False)
        )
    steps.append(lambda: _check_panel(raw, data, rep))
    steps.append(lambda: _check_stress(raw, data, rep))
    for step in steps:
        try:
            step()
        except Exception as exc:  # a check itself must never hide a problem
            rep.add("(check)", "check ran", FAIL, f"{type(exc).__name__}: {exc}")
    return rep


def _read_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_raw(df: pd.DataFrame, path: str, sep: str = ",", encoding: str = "utf-8") -> None:
    df.to_csv(path, index=False, sep=sep, encoding=encoding)


def _edit(name: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[str], None]:
    def apply(folder: str) -> None:
        p = _path(folder, name)
        _write_raw(fn(_read_raw(p)), p)

    return apply


def _set(df: pd.DataFrame, row_mask, col: str, value) -> pd.DataFrame:
    df = df.copy()
    df.loc[row_mask, col] = value
    return df


def _german(folder: str) -> None:
    """Every file: ';' separator, decimal comma, dd.mm.yyyy dates."""
    for f in os.listdir(folder):
        if not f.endswith(".csv"):
            continue
        p = os.path.join(folder, f)
        df = _read_raw(p)

        def cell(v: str) -> str:
            v = v.strip()
            if len(v) == 10 and v[4] == "-" and v[7] == "-":
                return f"{v[8:10]}.{v[5:7]}.{v[0:4]}"
            if "|" in v:
                return "|".join(cell(x) for x in v.split("|"))
            try:
                float(v)
                return v.replace(".", ",")
            except ValueError:
                return v

        _write_raw(df.map(cell), p, sep=";")


def _percent_strings(folder: str) -> None:
    def fn(df):
        for c in ("exp_return", "income", "lb", "ub"):
            df[c] = [f"{float(v) * 100:.6g}%" if v.strip() else v for v in df[c]]
        return df

    _edit("assets", fn)(folder)


def _shuffle(folder: str) -> None:
    _edit("prices", lambda df: df.iloc[::-1][[df.columns[0]] + list(df.columns[1:][::-1])])(folder)
    _edit("holdings", lambda df: df.iloc[::-1])(folder)
    _edit("groups", lambda df: df.iloc[::-1])(folder)


def _bom_crlf(folder: str) -> None:
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            p = os.path.join(folder, f)
            df = _read_raw(p)
            df.to_csv(p, index=False, encoding="utf-8-sig", lineterminator="\r\n")


def _padding(folder: str) -> None:
    _edit("holdings", lambda df: df.assign(asset=[f"  {v} " for v in df["asset"]]))(folder)
    _edit("commitment_future", lambda df: df.assign(asset=[f" {v}" for v in df["asset"]]))(folder)


def _tabs(folder: str) -> None:
    p = _path(folder, "prices")
    _write_raw(_read_raw(p), p, sep="\t")


def _price_col(folder: str, col: str, fn: Callable[[pd.Series], pd.Series]) -> None:
    def edit(df):
        df = df.copy()
        df[col] = fn(df[col])
        return df

    _edit("prices", edit)(folder)


def _scale_at(i: int, k: float) -> Callable[[pd.Series], pd.Series]:
    def fn(s):
        s = s.copy()
        s.iloc[i] = f"{float(s.iloc[i]) * k:.4f}"
        return s

    return fn


def _flat(i0: int, n: int) -> Callable[[pd.Series], pd.Series]:
    def fn(s):
        s = s.copy()
        s.iloc[i0 : i0 + n] = s.iloc[i0 - 1]
        return s

    return fn


def _blank(i0: int, n: int) -> Callable[[pd.Series], pd.Series]:
    def fn(s):
        s = s.copy()
        s.iloc[i0 : i0 + n] = ""
        return s

    return fn


def _shift_private_reports(s: pd.Series) -> pd.Series:
    vals = s.copy()
    out = pd.Series([""] * len(s), index=s.index)
    nz = [i for i, v in enumerate(vals) if v.strip()]
    for i in nz[len(nz) // 2 :]:  # second half of the history reported one month late
        if i + 1 < len(s):
            out.iloc[i + 1] = vals.iloc[i]
    for i in nz[: len(nz) // 2]:
        out.iloc[i] = vals.iloc[i]
    return out


def _mid_month(folder: str) -> None:
    def fn(df):
        df = df.copy()
        extra = df.iloc[[40]].copy()
        extra.iloc[0, 0] = extra.iloc[0, 0][:8] + "15"
        return pd.concat([df.iloc[:40], extra, df.iloc[40:]], ignore_index=True)

    _edit("prices", fn)(folder)


def _cases() -> List[tuple]:
    """(name, kind, mutation, expected, keyword). kind: 'variant' must load identically;
    'fault' must be caught (FAIL) or flagged (WARN/FAIL whose text contains keyword)."""
    eq, liq, priv = "US Dvd Eq", "Cash", "Priv Eq"
    return [
        # format variants: must give exactly the same model
        ("German locale (';', decimal comma, dd.mm.yyyy)", "variant", _german, "identical", ""),
        ("percent strings in assets.csv ('4.7%')", "variant", _percent_strings, "identical", ""),
        ("rows and columns shuffled", "variant", _shuffle, "identical", ""),
        ("UTF-8 BOM and Windows line endings", "variant", _bom_crlf, "identical", ""),
        ("spaces around asset names", "variant", _padding, "identical", ""),
        ("tab-separated prices.csv", "variant", _tabs, "identical", ""),
        # structural faults: must be rejected (FAIL)
        (
            "typo in a holdings asset name",
            "fault",
            _edit("holdings", lambda d: _set(d, d["asset"] == "US Govt", "asset", "US Gov")),
            "caught",
            "",
        ),
        ("price column missing", "fault", _edit("prices", lambda d: d.drop(columns=[eq])), "caught", ""),
        ("text in a price cell", "fault", lambda f: _price_col(f, eq, _scale_text), "caught", ""),
        (
            "duplicate date row",
            "fault",
            _edit("prices", lambda d: pd.concat([d.iloc[:10], d.iloc[[9]], d.iloc[10:]])),
            "caught",
            "",
        ),
        ("negative price level", "fault", lambda f: _price_col(f, eq, _scale_at(50, -1.0)), "caught", ""),
        (
            "returns given instead of index levels",
            "fault",
            lambda f: _price_col(
                f,
                eq,
                lambda s: pd.Series(
                    [
                        f"{v:.5f}" if v == v else ""
                        for v in pd.to_numeric(s, errors="coerce").pct_change().fillna(0.0)
                    ],
                    index=s.index,
                ),
            ),
            "caught",
            "",
        ),
        (
            "lower bound above upper bound",
            "fault",
            _edit("assets", lambda d: _set(d, d["name"] == eq, "lb", "0.2")),
            "caught",
            "",
        ),
        (
            "group references an unknown asset",
            "fault",
            _edit("groups", lambda d: _set(d, d["name"] == "Liquidity", "members", "Cash|US TBill")),
            "caught",
            "",
        ),
        (
            "contribution rates do not reach 1.0",
            "fault",
            _edit(
                "pacing", lambda d: _set(d, d["asset"] == priv, "contribution_rates", "0.25|0.50|0.75|0.90")
            ),
            "caught",
            "",
        ),
        (
            "capital call on a liquid asset",
            "fault",
            _edit("commitment_future", lambda d: _set(d, d.index == 0, "asset", liq)),
            "caught",
            "",
        ),
        (
            "negative capital call",
            "fault",
            _edit("commitment_future", lambda d: _set(d, d.index == 0, "amount", "-500000")),
            "caught",
            "",
        ),
        (
            "negative market value",
            "fault",
            _edit("holdings", lambda d: _set(d, d["asset"] == "Gold", "market_value", "-1000")),
            "caught",
            "",
        ),
        (
            "vol_lo above vol_hi",
            "fault",
            _edit("config", lambda d: _set(d, d["key"] == "vol_lo", "value", "0.09")),
            "caught",
            "",
        ),
        (
            "unknown asset in cma_history",
            "fault",
            _edit("cma_history", lambda d: _set(d, d.index == 0, "asset", "Bitcoin")),
            "caught",
            "",
        ),
        (
            "unknown key in policy_history",
            "fault",
            _edit("policy_history", lambda d: _set(d, d.index == 0, "key", "target_incme")),
            "caught",
            "",
        ),
        (
            "stress scenario uses an unknown factor",
            "fault",
            _edit("stress_scenarios", lambda d: d.rename(columns={"Oil_Brent": "Oil_WTI"})),
            "caught",
            "",
        ),
        (
            "mistyped price level (x10 for one month)",
            "fault",
            lambda f: _price_col(f, eq, _scale_at(80, 10.0)),
            "caught",
            "spike",
        ),
        # silent faults: must be flagged (WARN or FAIL)
        (
            "stale price feed (6 months unchanged)",
            "fault",
            lambda f: _price_col(f, eq, _flat(100, 6)),
            "flagged",
            "stale",
        ),
        (
            "missing months in a monthly series",
            "fault",
            lambda f: _price_col(f, eq, _blank(60, 3)),
            "flagged",
            "blank month",
        ),
        (
            "one series ends 6 months early",
            "fault",
            lambda f: _price_col(
                f, eq, lambda s: pd.concat([s.iloc[:-6], pd.Series([""] * 6, index=s.index[-6:])])
            ),
            "flagged",
            "ends",
        ),
        (
            "private NAV reported off the quarter calendar",
            "fault",
            lambda f: _price_col(f, priv, _shift_private_reports),
            "flagged",
            "calendar",
        ),
        ("two price rows in one month (mid-month date)", "fault", _mid_month, "flagged", "several rows"),
        (
            "rate written as percent (4.7 instead of 0.047)",
            "fault",
            _edit("assets", lambda d: _set(d, d["name"] == eq, "exp_return", "8.1")),
            "flagged",
            "percent",
        ),
        (
            "typo in a config key",
            "fault",
            _edit("config", lambda d: _set(d, d["key"] == "vol_hi", "key", "vol_hgih")),
            "flagged",
            "IGNORED",
        ),
        (
            "file name typo (holdings_v2.csv extra file)",
            "fault",
            lambda f: shutil.copy(_path(f, "flows"), os.path.join(f, "flow.csv")),
            "flagged",
            "IGNORED",
        ),
    ]


def _scale_text(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.iloc[30] = "1O2.5"  # letter O instead of zero
    return s


def _fingerprint(data: MandateData) -> Dict[str, object]:
    u = data.universe
    return dict(
        returns=data.returns.sort_index(axis=1),
        assets={a.name: a for a in u},
        groups={g.name: g for g in u.groups},
        holdings=data.holdings.market_value.sort_index(),
        undrawn=data.holdings.undrawn.sort_index(),
        settings=dict(data.settings.raw),
        pacing=data.pacing,
        schedule=data.schedule.sort_values(["date", "asset"]).reset_index(drop=True),
        factors=None if data.stress_factors is None else data.stress_factors.sort_index(axis=1),
    )


def _identical(a: Dict[str, object], b: Dict[str, object]) -> str:
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (pd.DataFrame, pd.Series)):
            try:
                if isinstance(x, pd.DataFrame):
                    pd.testing.assert_frame_equal(
                        x, y, check_exact=False, rtol=1e-12, atol=1e-12, check_freq=False, check_dtype=False
                    )
                else:
                    pd.testing.assert_series_equal(
                        x, y, check_exact=False, rtol=1e-12, atol=1e-12, check_dtype=False
                    )
            except AssertionError:
                return f"'{k}' differs"
        elif isinstance(x, dict):
            if set(x) != set(y):
                return f"'{k}' differs (names)"
            for n in x:
                if not _same_obj(x[n], y[n]):
                    return f"'{k}' differs ({n})"
        elif x != y:
            return f"'{k}' differs"
    return ""


def _same_obj(x, y) -> bool:
    """Field-by-field equality with a float tolerance ('3.7%' parses to 0.037000000000000005)."""
    if hasattr(x, "__dataclass_fields__"):
        return all(_same_obj(getattr(x, f), getattr(y, f)) for f in x.__dataclass_fields__)
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(_same_obj(a, b) for a, b in zip(x, y))
    if isinstance(x, (float, np.floating)) or isinstance(y, (float, np.floating)):
        return _same(float(x), float(y), 1e-12)
    return x == y


def stress_test_inputs(folder: str) -> pd.DataFrame:
    """Run every format variant and injected fault against a copy of ``folder``."""
    base = _fingerprint(load_mandate(folder))
    rows = []
    for name, kind, mutate, expected, keyword in _cases():
        tmp = tempfile.mkdtemp(prefix="saa_check_")
        try:
            for f in os.listdir(folder):
                if f.endswith(".csv"):
                    shutil.copy(os.path.join(folder, f), tmp)
            mutate(tmp)
            rep = check_inputs(tmp)
            fails = rep.issues((FAIL,))
            flags = rep.issues((FAIL, WARN))
            if expected == "identical":
                if rep.data is None:
                    observed, ok = f"did not load: {fails['detail'].iloc[0][:90]}", False
                else:
                    diff = _identical(base, _fingerprint(rep.data))
                    observed, ok = ("identical model" if not diff else diff), not diff
            elif expected == "caught":
                hit = (
                    fails[
                        fails["detail"].str.contains(keyword, case=False)
                        | fails["check"].str.contains(keyword, case=False)
                    ]
                    if keyword
                    else fails
                )
                ok = len(hit) > 0
                observed = (
                    f"FAIL [{hit.iloc[0]['file']}] {hit.iloc[0]['detail'][:90]}" if ok else "NOT caught"
                )
            else:
                hit = flags[
                    flags["detail"].str.contains(keyword, case=False)
                    | flags["check"].str.contains(keyword, case=False)
                ]
                ok = len(hit) > 0
                observed = (
                    f"{hit.iloc[0]['status']} [{hit.iloc[0]['file']}] {hit.iloc[0]['detail'][:90]}"
                    if ok
                    else "NOT flagged"
                )
        except Exception as exc:
            ok, observed = False, f"check crashed: {type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rows.append(
            dict(case=name, kind=kind, expected=expected, result="PASS" if ok else "FAIL", observed=observed)
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 110)
    report = check_inputs(folder)
    print(report.text())
    print(report.summary())
    if "--stress" in sys.argv:
        res = stress_test_inputs(folder)
        print(res.to_string(index=False))
        print(f"\n{(res['result'] == 'PASS').sum()}/{len(res)} stress cases behave as expected")
        if (res["result"] != "PASS").any():
            sys.exit(1)
    sys.exit(0 if report.ok else 1)
