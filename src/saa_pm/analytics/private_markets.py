"""Private-markets cash flows: projected calls, distributions, NAV, J-curves
and fund performance (IRR, DPI, RVPI, TVPI).

Every scheduled call becomes its own cohort on its call date; NAV held today
("legacy") runs off from the as-of date. All cohorts step monthly with the
same Takahashi-Alexander book the engine uses, growing at the CMA.

For PERFORMANCE the legacy NAV enters as a contribution at the as-of date
(you could sell it at NAV today); otherwise its run-off distributions would
look like free money and inflate IRR / TVPI.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..privates.book import PrivateBook, monthly_growth
from ..privates.pacing import PacingParams


def xirr(cashflows: Sequence[float], dates: Sequence) -> float:
    """IRR on dated cash flows (ACT/365); NaN if flows do not change sign."""
    cfs = np.asarray(cashflows, float)
    d = pd.to_datetime(pd.Index(dates))
    if len(cfs) < 2 or (cfs >= 0).all() or (cfs <= 0).all():
        return np.nan
    t = np.asarray((d - d.min()).days, float) / 365.0

    def npv(r):
        return float(np.sum(cfs / (1.0 + r) ** t))

    lo, hi = -0.99, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def project_cashflows(
    pacing: Dict[str, PacingParams],
    schedule: pd.DataFrame,
    asof: pd.Timestamp,
    legacy_nav: Optional[Dict[str, float]] = None,
    horizon_years: Optional[int] = None,
) -> pd.DataFrame:
    """Monthly long table per cohort: asset, cohort, date, call, distribution, nav."""
    legacy_nav = legacy_nav or {}
    sched = schedule[pd.to_datetime(schedule["date"]) > asof] if len(schedule) else schedule
    H = horizon_years or (max([p.life_years for p in pacing.values()] + [10]) + 2)
    end = pd.Timestamp(asof) + pd.DateOffset(
        years=H + int((pd.to_datetime(sched["date"]).max() - asof).days / 365.25) if len(sched) else H
    )
    rows = []

    def run(asset, cohort, book, start, first_call):
        g = monthly_growth(book.params.growth_rate)
        rows.append(
            dict(asset=asset, cohort=cohort, date=start, call=first_call, distribution=0.0, nav=book.nav)
        )
        d = start
        while d < end and (book.nav > 1e-6 or not rows):
            d = d + pd.offsets.MonthEnd(1)
            _, dist = book.step_month(g)
            rows.append(dict(asset=asset, cohort=cohort, date=d, call=0.0, distribution=dist, nav=book.nav))

    for n, p in pacing.items():
        nav0 = float(legacy_nav.get(n, 0.0))
        if nav0 > 1e-9:
            run(n, "legacy", PrivateBook(n, p, nav0=nav0), pd.Timestamp(asof), 0.0)
    for r in sched.itertuples():
        if r.asset not in pacing:
            continue
        b = PrivateBook(r.asset, pacing[r.asset])
        b.call_now(float(r.amount))
        run(
            r.asset,
            f"{pd.Timestamp(r.date):%Y-%m}",
            b,
            pd.Timestamp(r.date) + pd.offsets.MonthEnd(0),
            float(r.amount),
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["net_cashflow"] = df["distribution"] - df["call"]
    df["year"] = df["date"].dt.year
    return df


def performance(cf: pd.DataFrame, by: Sequence[str] = ("asset",)) -> pd.DataFrame:
    """IRR / DPI / RVPI / TVPI per group. Legacy NAV counts as paid-in capital;
    NAV still held at the end of the projection is a terminal value."""
    rows = []
    keys = list(by)
    groups = cf.groupby(keys) if keys else [((), cf)]
    for key, g in groups:
        paid_in = g["call"].copy()
        first = g.groupby(["asset", "cohort"])["date"].transform("min")
        legacy_start = (g["cohort"] == "legacy") & (g["date"] == first)
        paid_in[legacy_start] += g.loc[legacy_start, "nav"]
        flows = pd.DataFrame({"date": g["date"], "flow": g["distribution"] - paid_in})
        final = g.sort_values("date").groupby(["asset", "cohort"]).tail(1)
        flows = pd.concat([flows, pd.DataFrame({"date": final["date"], "flow": final["nav"]})])
        flows = flows.groupby("date")["flow"].sum()
        pi, dist, nav = paid_in.sum(), g["distribution"].sum(), final["nav"].sum()
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update(
            paid_in=pi,
            distributions=dist,
            final_nav=nav,
            dpi=dist / pi if pi > 0 else np.nan,
            rvpi=nav / pi if pi > 0 else np.nan,
            tvpi=(dist + nav) / pi if pi > 0 else np.nan,
            irr=xirr(flows.values, flows.index),
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def annual_summary(cf: pd.DataFrame) -> pd.DataFrame:
    """Calls, distributions, net cash flow and year-end NAV by calendar year."""
    flows = cf.groupby("year")[["call", "distribution", "net_cashflow"]].sum()
    ye = cf.sort_values("date").groupby(["asset", "cohort", "year"]).tail(1)
    flows["nav_year_end"] = ye.groupby("year")["nav"].sum()
    flows["cum_net_cashflow"] = flows["net_cashflow"].cumsum()
    return flows


def j_curve(cf: pd.DataFrame) -> pd.DataFrame:
    """Cumulative net cash flow by years since the call, per sleeve (new calls only)."""
    new = cf[cf["cohort"] != "legacy"].copy()
    start = new.groupby(["asset", "cohort"])["date"].transform("min")
    new["years_since_call"] = ((new["date"] - start).dt.days / 365.25).round().astype(int)
    agg = new.groupby(["asset", "years_since_call"])["net_cashflow"].sum()
    return agg.groupby(level=0).cumsum().unstack(0).ffill()


def vintage_diversification(cf: pd.DataFrame) -> pd.DataFrame:
    """Year-end NAV by cohort and the effective number of vintages (1/HHI)."""
    ye = cf.sort_values("date").groupby(["asset", "cohort", "year"]).tail(1)
    nav = ye.pivot_table(index="year", columns="cohort", values="nav", aggfunc="sum").fillna(0.0)
    shares = nav.div(nav.sum(axis=1).replace(0, np.nan), axis=0)
    nav["effective_vintages"] = 1.0 / (shares**2).sum(axis=1).replace(0, np.nan)
    return nav
