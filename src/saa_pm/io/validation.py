"""Input validation - run before trusting any result.

:func:`validate` returns a :class:`ValidationReport` with blocking errors,
warnings and a per-asset data-coverage table. Nothing is printed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from ..risk.covariance import aggregate_returns, desmooth_returns
from .loaders import MandateData


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError("input validation failed:\n  - " + "\n  - ".join(self.errors))

    def text(self) -> str:
        out = []
        for title, items, mark in (
            ("ERRORS", self.errors, "x"),
            ("WARNINGS", self.warnings, "!"),
            ("NOTES", self.notes, "-"),
        ):
            if items:
                out.append(f"{title}")
                out += [f"  {mark} {m}" for m in items]
        return "\n".join(out) if out else "no issues"


def min_history_months(data: MandateData) -> int:
    return max(24, 3 * data.universe.estimation_months)


def validate(data: MandateData) -> ValidationReport:
    rep = ValidationReport(notes=list(data.notes))
    u, hist = data.universe, data.returns
    errs, warns = rep.errors, rep.warnings

    # structure
    errs += u.structural_problems()
    for g in u.groups:
        if not g.members:
            errs.append(f"group '{g.name}' has no members")
    gnames = [g.name for g in u.groups]
    dup = sorted({n for n in gnames if gnames.count(n) > 1})
    if dup:
        errs.append(f"duplicate group names {dup}")
    ladder = u.ladder_map
    if ladder and sorted(ladder) != list(range(1, len(ladder) + 1)):
        errs.append(f"ladder buckets must be numbered 1..N, got {sorted(ladder)}")
    for k, names in ladder.items():
        for n in names:
            if not u[n].liquid:
                errs.append(f"ladder bucket {k} contains illiquid asset '{n}'")
            elif u[n].semi_liquid:
                errs.append(f"ladder bucket {k} contains gated (semi-liquid) asset '{n}'")
    for a in u:
        rl = a.redemption_limit
        if rl is not None and not 0.0 < rl <= 1.0:
            errs.append(f"'{a.name}': redemption_limit {rl} must lie in (0, 1]")
        if rl is not None and not a.liquid:
            warns.append(
                f"'{a.name}': redemption_limit is ignored for an illiquid (closed-end) sleeve - "
                f"set liquid=True for an evergreen / ELTIF vehicle"
            )

    # private sleeves
    for n, p in data.pacing.items():
        if u[n].liquid:
            errs.append(f"pacing defined for liquid asset '{n}'")
        errs += [m for m in p.problems() if "yield" not in m]
    for a in u:
        if not a.liquid and a.name not in data.pacing:
            warns.append(f"'{a.name}' is illiquid but has no pacing row - no distributions or call modelling")
        if not a.liquid and not a.appraisal_based:
            warns.append(
                f"'{a.name}' is illiquid but not appraisal_based - its reported "
                f"(smoothed) volatility is used without desmoothing"
            )

    # history
    common = hist.dropna()
    need = min_history_months(data)
    cov_rows = []
    k_est = u.estimation_months
    for a in u:
        s = hist[a.name].dropna()
        sn = aggregate_returns(s.to_frame(), k_est)[a.name] if k_est > 1 else s
        ann = np.sqrt(12.0 / k_est)
        realised = (1 + s).prod() ** (12 / len(s)) - 1 if len(s) >= 12 else np.nan
        vol_rep = sn.std() * ann if len(sn) > 4 else np.nan
        vol_des, phi = vol_rep, 0.0
        if a.appraisal_based and len(sn) > 8:
            d, phi = desmooth_returns(sn)
            vol_des = d.dropna().std() * ann
        rho = float(np.corrcoef(sn.values[1:], sn.values[:-1])[0, 1]) if len(sn) >= 20 else np.nan
        cov_rows.append(
            dict(
                asset=a.name,
                first=s.index.min(),
                last=s.index.max(),
                months=len(s),
                realised_return=realised,
                cma=a.exp_return,
                vol_reported=vol_rep,
                vol_desmoothed=vol_des,
                autocorr=rho,
                phi=phi,
            )
        )
        if len(s) >= 24:
            # CMAs are ARITHMETIC; a realised series compounds GEOMETRICALLY,
            # roughly mu - vol^2/2. Compare like with like.
            v2 = (vol_des if np.isfinite(vol_des) else 0.0) ** 2
            geo = (1 + a.exp_return) * np.exp(-0.5 * v2) - 1
            if a.income > 0.015:
                d_tot = abs(realised - geo)
                d_px = abs(realised - (geo - a.income))
                if d_px < 0.5 * d_tot:
                    warns.append(
                        f"'{a.name}': realised {realised:.2%} p.a. is close to "
                        f"total-minus-income - is this a PRICE index?"
                    )
            if abs(realised - geo) > 0.06:
                warns.append(
                    f"'{a.name}': CMA {a.exp_return:.2%} (geometric ~{geo:.2%}) "
                    f"vs realised {realised:.2%} p.a. - large gap"
                )
        if np.isfinite(rho):
            if not a.appraisal_based and rho > 0.30:
                warns.append(
                    f"'{a.name}': autocorrelation {rho:.2f} but not marked "
                    f"appraisal_based - stale prices or smoothed NAV?"
                )
            if a.appraisal_based and rho >= 0.60:
                warns.append(
                    f"'{a.name}': autocorrelation {rho:.2f} hits the Geltner cap - "
                    f"desmoothing multiplies its vol by 2.5x"
                )
        if a.income > a.exp_return + 1e-9:
            rep.notes.append(
                f"'{a.name}': income {a.income:.2%} > total return "
                f"{a.exp_return:.2%} (normal for credit, but confirm)"
            )
    rep.coverage = pd.DataFrame(cov_rows).set_index("asset")

    if len(common) < need:
        short = rep.coverage["months"].idxmin()
        late = rep.coverage["first"].idxmax()
        errs.append(
            f"common history is {len(common)} months, need >= {need}. "
            f"Shortest series '{short}', latest start '{late}' "
            f"({rep.coverage.loc[late, 'first']:%Y-%m})"
        )
    elif len(common) < 60:
        warns.append(
            f"common history only {len(common)} months - short for a "
            f"{data.settings.risk.halflife:.0f}m EWMA half-life"
        )
    bt = data.settings.backtest
    if len(common) <= bt.burn_in_months + 24:
        warns.append(
            f"only {len(common)} months of history for a {bt.burn_in_months}m "
            f"burn-in - the backtest will be skipped"
        )

    # holdings vs schedule
    w = data.holdings.weights
    for i, a in enumerate(u):
        if w[i] < a.lb - 1e-6 or w[i] > a.ub + 1e-6:
            warns.append(f"current weight of '{a.name}' {w[i]:.2%} is outside [{a.lb:.2%}, {a.ub:.2%}]")
    sched = data.schedule
    if len(sched):
        fut = sched[sched["date"] > data.asof]
        if len(fut) < len(sched):
            rep.notes.append(
                f"{len(sched) - len(fut)} scheduled call(s) on/before the as-of "
                f"date {data.asof:%Y-%m-%d} are treated as paid"
            )
        by = fut.groupby("asset")["amount"].sum()
        for n in u.illiquid_names:
            und = float(data.holdings.undrawn.get(n, 0.0))
            sch = float(by.get(n, 0.0))
            if abs(und - sch) > 1.0:
                warns.append(
                    f"'{n}': undrawn {und:,.0f} vs scheduled future calls "
                    f"{sch:,.0f} - only the schedule is pre-funded"
                )
        no_pacing = sorted(set(fut["asset"]) - set(data.pacing))
        if no_pacing:
            errs.append(f"commitment_future.csv names sleeves without pacing rows {no_pacing}")
        if not ladder:
            warns.append(
                "a call schedule exists but no asset has a ladder_bucket - nothing pre-funds the calls"
            )
    return rep
