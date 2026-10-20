"""Goal-ordered strategic asset allocation.

Three ranked goals - (1) return >= target, (2) income >= target,
(3) volatility inside [vol_lo, vol_hi] - and the order is binding.

Why a frontier scan: ``vol <= vol_hi`` is convex but ``vol >= vol_lo`` is
reverse-convex, so the band cannot be one convex constraint. Instead the
optimiser solves a convex problem at each target vol on a grid spanning the
band and then SELECTS the point that best meets the goals in order.

Inside every grid point a strict three-stage lexicographic solve runs:

  A. maximise expected return                    (goal 1 sees no competition)
  B. minimise the income shortfall s.t. return >= min(target, max_A)
  C. re-maximise return s.t. the stage-B shortfall (no return left behind)

Each stage is compiled ONCE as a parametrised cvxpy problem (target vol,
return floor and shortfall cap are Parameters) and re-solved across the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from ..config import OptimizerConfig
from ..privates.ladder import Floor
from ..risk.covariance import align_to, nearest_psd
from ..universe import GroupConstraint, Universe
from .constraints import STRATEGIC, ConstraintSpec, max_attainable, solve_problem

_BAND_TOL = 1e-4  # a point within 1bp of vol_lo counts as inside the band


@dataclass
class Solution:
    weights: np.ndarray
    target_vol: float
    exp_return: float
    income: float
    vol: float
    income_shortfall: float
    turnover: float = float("nan")
    band_relaxed: bool = False  # vol target above vol_hi (goal 3 yielded)
    risk_binding: bool = True  # False -> frontier flat, risk budget unused
    status: str = "optimal"
    selection_rule: str = ""

    @property
    def below_band(self) -> bool:
        return bool(self.risk_binding is False and self.vol < self.target_vol - _BAND_TOL)

    def summary(self) -> dict:
        return dict(
            exp_return=self.exp_return,
            income=self.income,
            vol=self.vol,
            income_shortfall=self.income_shortfall,
            turnover=self.turnover,
            band_relaxed=self.band_relaxed,
            rule=self.selection_rule,
        )


@dataclass
class LadderShortfall:
    bucket: int
    need: float
    granted: float


@dataclass
class OptimizationResult:
    solution: Solution
    scan: List[Solution]
    floors: Sequence[Floor] = field(default_factory=list)
    ladder_shortfalls: List[LadderShortfall] = field(default_factory=list)
    clamped_groups: List[str] = field(default_factory=list)
    fallback: Optional[str] = None
    audit: Optional[pd.DataFrame] = None

    @property
    def weights(self) -> np.ndarray:
        return self.solution.weights

    @property
    def audit_failures(self) -> List[str]:
        if self.audit is None:
            return []
        bad = self.audit[~self.audit["pass"]]
        return [f"{r.check}: {r.value}" for r in bad.itertuples()]


class InfeasibleError(RuntimeError):
    """No feasible portfolio; the message names the conflicting constraints."""


class GoalOptimizer:
    """One decision date: universe + covariance + targets (+ ladder floors).

    Parameters
    ----------
    universe : Universe          bounds, groups, CMAs
    sigma    : DataFrame         annualised covariance (any label order)
    config   : OptimizerConfig   targets and solver settings
    floors   : list[Floor]       funding-ladder floors (already clamped or not)
    w_prev   : array             current weights (turnover, no-trade bands)
    mode     : 'strategic' | 'rebalance'
    """

    def __init__(
        self,
        universe: Universe,
        sigma: pd.DataFrame,
        config: OptimizerConfig,
        floors: Sequence[Floor] = (),
        w_prev: Optional[np.ndarray] = None,
        mode: str = STRATEGIC,
    ):
        self.universe = universe
        self.config = config
        self.sigma = nearest_psd(align_to(sigma, universe.names))
        self.spec = ConstraintSpec(universe, config, list(floors), w_prev, mode)
        self._chol = self._cholesky(self.sigma.values)
        self._stages: Optional[tuple] = None

    @staticmethod
    def _cholesky(S: np.ndarray) -> np.ndarray:
        jitter = 0.0
        for _ in range(6):
            try:
                return np.linalg.cholesky(S + jitter * np.eye(len(S)))
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10, 1e-12 * max(np.trace(S), 1e-12))
        raise np.linalg.LinAlgError("covariance is not positive definite")

    def vol_of(self, w: np.ndarray) -> float:
        return float(np.linalg.norm(self._chol.T @ w))

    def _build_stages(self):
        n, u, c = self.universe.n, self.universe, self.config
        w = cp.Variable(n, name="w")
        s_inc = cp.Variable(nonneg=True, name="income_shortfall")
        sig = cp.Parameter(nonneg=True, name="sigma_target")
        r_floor = cp.Parameter(name="return_floor")
        s_cap = cp.Parameter(nonneg=True, name="shortfall_cap")
        base = self.spec.build(w)
        base += [cp.norm(self._chol.T @ w, 2) <= sig, u.income @ w + s_inc >= c.target_income]
        if c.hard_return_floor:
            base.append(u.mu @ w >= c.target_return)
        if c.lexicographic:
            stages = dict(
                A=cp.Problem(cp.Maximize(u.mu @ w), base),
                B=cp.Problem(cp.Minimize(s_inc), base + [u.mu @ w >= r_floor]),
                C=cp.Problem(cp.Maximize(u.mu @ w), base + [u.mu @ w >= r_floor, s_inc <= s_cap]),
            )
        else:
            stages = dict(P=cp.Problem(cp.Maximize(u.mu @ w - c.income_penalty * s_inc), base))
        self._stages = (w, s_inc, sig, r_floor, s_cap, stages)

    def solve_at(self, sigma_target: float) -> Optional[Solution]:
        """Lexicographic solve at one target volatility; None if infeasible."""
        if self._stages is None:
            self._build_stages()
        assert self._stages is not None
        w, s_inc, sig, r_floor, s_cap, P = self._stages
        c, u = self.config, self.universe
        sig.value = float(sigma_target)
        solver = c.solver

        if c.lexicographic:
            if not solve_problem(P["A"], solver):
                return None
            w_a = np.asarray(w.value).ravel().copy()
            r_max = float(u.mu @ w_a)
            r_floor.value = (
                c.target_return if r_max >= c.target_return - c.return_tol else r_max - c.return_tol
            )
            if solve_problem(P["B"], solver):
                s_cap.value = max(float(s_inc.value), 0.0) + c.return_tol
                if solve_problem(P["C"], solver):
                    wv, status = np.asarray(w.value).ravel(), P["C"].status
                else:
                    solve_problem(P["B"], solver)
                    wv, status = np.asarray(w.value).ravel(), P["B"].status
            else:
                wv, status = w_a, P["A"].status
        else:
            if not solve_problem(P["P"], solver):
                return None
            wv, status = np.asarray(w.value).ravel(), P["P"].status
        return self._make_solution(wv, sigma_target, status)

    def _make_solution(self, wv: np.ndarray, sigma_target: float, status: str) -> Solution:
        c, u = self.config, self.universe
        wv = np.where(np.abs(wv) < 1e-8, 0.0, wv)
        vol = self.vol_of(wv)
        w_prev = self.spec.w_prev
        return Solution(
            weights=wv,
            target_vol=float(sigma_target),
            exp_return=float(u.mu @ wv),
            income=float(u.income @ wv),
            vol=vol,
            income_shortfall=float(max(0.0, c.target_income - u.income @ wv)),
            turnover=float(0.5 * np.abs(wv - w_prev).sum()) if w_prev is not None else np.nan,
            band_relaxed=bool(sigma_target > c.vol_hi + 1e-9),
            risk_binding=bool(vol >= sigma_target - _BAND_TOL),
            status=status,
        )

    def scan(self, grid: Optional[Sequence[float]] = None) -> List[Solution]:
        """Solve across the band; extend above it only if the return goal is
        unreachable inside and ``vol_relax_max`` allows (risk yields last)."""
        c = self.config
        grid = c.vol_grid if grid is None else grid
        out = [s for s in (self.solve_at(v) for v in grid) if s is not None]
        relax_to = float(c.vol_relax_max or 0.0)
        if relax_to > c.vol_hi + 1e-9 and (
            not out or max(s.exp_return for s in out) < c.target_return - 1e-6
        ):
            step = (c.vol_hi - c.vol_lo) / max(1, c.vol_steps - 1) or 0.0025
            extra = list(np.arange(c.vol_hi + step, relax_to + 1e-12, step))
            if not extra or extra[-1] < relax_to - 1e-9:
                extra.append(relax_to)
            for v in extra:
                s = self.solve_at(v)
                if s is not None:
                    out.append(s)
                    if s.exp_return >= c.target_return - 1e-6:
                        break
        return out

    def select(self, scan: List[Solution], tol: float = 1e-5) -> Solution:
        """Strict goal-ordered selection.

        1. return AND income met inside the band  -> lowest vol in the band
        2. return met inside the band             -> smallest income gap
        3. return only reachable above the band   -> lowest such vol
        4. return unreachable                     -> max return INSIDE the band
        Within each tier points that genuinely reach ``vol_lo`` are preferred
        (a flat frontier can leave every point below the band floor).
        """
        if not scan:
            raise InfeasibleError("no feasible point in the risk band")
        c = self.config
        in_band = [s for s in scan if not s.band_relaxed]
        relaxed = [s for s in scan if s.band_relaxed]

        def gap(s: Solution) -> float:  # distance below the band floor
            return max(0.0, c.vol_lo - s.vol - _BAND_TOL)

        ok = [s for s in in_band if s.exp_return >= c.target_return - tol]
        both = [s for s in ok if s.income_shortfall <= tol]
        if both:
            best, rule = min(both, key=lambda s: (gap(s), s.vol)), "return+income met in band -> min risk"
        elif ok:
            best, rule = (
                min(ok, key=lambda s: (s.income_shortfall, gap(s), s.vol)),
                "return met in band -> min income gap",
            )
        else:
            rx = [s for s in relaxed if s.exp_return >= c.target_return - tol]
            if rx:
                best = min(rx, key=lambda s: (s.income_shortfall > tol, s.vol))
                rule = f"return unreachable in band -> band relaxed to {best.vol:.2%}"
            else:
                pool = in_band or scan
                best = max(pool, key=lambda s: (s.exp_return, -s.income_shortfall))
                rule = "targets unreachable -> max return, then min income gap"
        if gap(best) > 0:
            rule += f" (frontier flat: vol {best.vol:.2%} below band floor)"
        return replace(best, selection_rule=rule)

    def min_attainable_vol(self) -> float:
        w = cp.Variable(self.universe.n)
        p = cp.Problem(cp.Minimize(cp.norm(self._chol.T @ w, 2)), self.spec.build(w))
        return float(p.value) if solve_problem(p, self.config.solver) else float("nan")

    def _relaxed_band_solution(self) -> Optional[Solution]:
        """Band infeasible under the current constraints: scan from the minimum
        attainable vol upward and take the LOWEST vol that meets return+income,
        else return, else the minimum-vol point itself (de-risk)."""
        c = self.config
        v_min = self.min_attainable_vol()
        if not np.isfinite(v_min):
            return None
        hi = max(0.40, 2.0 * c.vol_hi, v_min + 0.01)
        grid = np.linspace(v_min + 1e-6, hi, max(2 * c.vol_steps, 21))
        sc = sorted((s for s in (self.solve_at(v) for v in grid) if s is not None), key=lambda s: s.vol)
        if not sc:
            return None
        tol = 1e-5
        for rule, ok in (
            (
                "return+income met",
                lambda s: s.exp_return >= c.target_return - tol and s.income_shortfall <= tol,
            ),
            ("return met", lambda s: s.exp_return >= c.target_return - tol),
        ):
            hit = [s for s in sc if ok(s)]
            if hit:
                return replace(
                    hit[0],
                    band_relaxed=True,
                    selection_rule=f"band infeasible -> lowest vol with {rule} ({hit[0].vol:.2%})",
                )
        return replace(
            sc[0],
            band_relaxed=True,
            selection_rule=f"band infeasible, targets unreachable -> min vol ({sc[0].vol:.2%})",
        )

    def min_vol_fallback(self) -> Optional[Solution]:
        """No feasible point in the band (e.g. a pinned private book after a
        dislocation, or a current book far outside policy under a turnover cap).

        1. keep every constraint, let the band (goal 3) yield minimally;
        2. if the constraint set itself is infeasible, relax the turnover cap
           stepwise (a per-step friction, never a policy limit) and solve
           normally. Every relaxation is stamped on ``selection_rule``.
        """
        sol = self._relaxed_band_solution()
        if sol is not None:
            return sol
        c = self.config
        if c.turnover_limit is None:
            return None
        for mult in (1.5, 2.0, 3.0, None):
            tl = None if mult is None else min(1.0, c.turnover_limit * mult)
            opt = GoalOptimizer(
                self.universe,
                self.sigma,
                replace(c, turnover_limit=tl),
                self.spec.floors,
                self.spec.w_prev,
                self.spec.mode,
            )
            label = "OFF" if tl is None else f"{tl:.0%}"
            sc = opt.scan()
            if sc:
                best = opt.select(sc)
                return replace(
                    best, selection_rule=f"{best.selection_rule} [turnover cap relaxed to {label}]"
                )
            relaxed = opt._relaxed_band_solution()
            if relaxed is not None:
                return replace(
                    relaxed, selection_rule=f"{relaxed.selection_rule} [turnover cap relaxed to {label}]"
                )
        return None

    def diagnose(self) -> dict:
        """Attainable envelope under the SAME constraints as the scan."""
        u, c = self.universe, self.config
        w = cp.Variable(u.n)
        base = self.spec.build(w)
        vol = cp.norm(self._chol.T @ w, 2)

        def val(obj, extra=()):
            p = cp.Problem(obj, base + list(extra))
            return float(p.value) if solve_problem(p, c.solver) else float("nan")

        d = dict(
            max_return=val(cp.Maximize(u.mu @ w)),
            max_income=val(cp.Maximize(u.income @ w)),
            min_vol=val(cp.Minimize(vol)),
            max_return_at_vol_hi=val(cp.Maximize(u.mu @ w), [vol <= c.vol_hi]),
            max_income_at_vol_hi=val(cp.Maximize(u.income @ w), [vol <= c.vol_hi]),
        )
        r_hi = d["max_return_at_vol_hi"]
        if np.isnan(r_hi) or r_hi < c.target_return:
            d["min_vol_for_return_target"] = float("nan")
        else:
            lo, hi = (d["min_vol"] if np.isfinite(d["min_vol"]) else 0.0), c.vol_hi
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                v = val(cp.Maximize(u.mu @ w), [vol <= mid])
                lo, hi = (lo, mid) if (np.isfinite(v) and v >= c.target_return) else (mid, hi)
            d["min_vol_for_return_target"] = hi
        d["return_target_feasible"] = bool(np.isfinite(r_hi) and r_hi >= c.target_return - 1e-9)
        d["income_target_feasible"] = bool(
            np.isfinite(d["max_income_at_vol_hi"]) and d["max_income_at_vol_hi"] >= c.target_income - 1e-9
        )
        return d

    def infeasibility_report(self) -> str:
        try:
            diag = self.diagnose()
            lines = [
                f"  {k}: {v:.4%}" if isinstance(v, float) and np.isfinite(v) else f"  {k}: {v}"
                for k, v in diag.items()
            ]
        except Exception as exc:  # diagnostics never mask the cause
            lines = [f"  diagnostics failed: {exc}"]
        probs = self.universe.structural_problems()
        for f in self.spec.floors:
            cap = float(sum(self.universe[n].ub for n in f.assets))
            if f.level > cap + 1e-9:
                probs.append(
                    f"ladder bucket {f.bucket} ({'+'.join(f.assets)}): floor {f.level:.2%} > caps {cap:.2%}"
                )
        txt = ("\n  structural conflicts:\n    - " + "\n    - ".join(probs)) if probs else ""
        return "\n".join(lines) + txt

    def audit(self, sol: Solution) -> pd.DataFrame:
        ceiling = max(self.config.vol_hi, sol.vol) if sol.band_relaxed else self.config.vol_hi
        return self.spec.audit(sol.weights, self.sigma, ceiling)

    def run(self, allow_fallback: bool = False) -> OptimizationResult:
        """Clamp the ladder floors to what policy allows, scan, select, audit."""
        probs = [p for p in self.universe.structural_problems() if "unknown assets" in p or "non-finite" in p]
        if probs:
            raise ValueError("setup validation failed:\n  - " + "\n  - ".join(probs))
        floors, shortfalls = clamp_ladder_floors(self.spec)
        opt = (
            self
            if floors is self.spec.floors
            else GoalOptimizer(
                self.universe, self.sigma, self.config, floors, self.spec.w_prev, self.spec.mode
            )
        )
        scan = opt.scan()
        fallback = None
        if scan:
            sol = opt.select(scan)
        else:
            fb = opt.min_vol_fallback() if allow_fallback else None
            if fb is None:
                raise InfeasibleError(
                    "no feasible portfolio in the risk band:\n" + opt.infeasibility_report()
                )
            sol, fallback, scan = fb, fb.selection_rule, [fb]
        return OptimizationResult(sol, scan, floors, shortfalls, fallback=fallback, audit=opt.audit(sol))


def clamp_ladder_floors(spec: ConstraintSpec) -> Tuple[Sequence[Floor], List[LadderShortfall]]:
    """Clamp each ladder floor to the weight its bucket can hold under policy
    (bounds and groups - trading frictions are relaxed by the fallback instead),
    front buckets first (year-1 money has priority). Returns the floors to
    impose and the recorded shortfalls."""
    if not spec.floors:
        return spec.floors, []
    granted: List[Floor] = []
    shortfalls: List[LadderShortfall] = []
    changed = False
    for f in spec.floors:
        sub = ConstraintSpec(spec.universe, spec.config, granted, spec.w_prev, spec.mode)
        cap = max_attainable(sub, f.selector)
        if not np.isfinite(cap):
            cap = 0.0  # base set infeasible: impose nothing
        if f.level > cap - 1e-9:
            eff = max(0.0, cap - 1e-6)
            shortfalls.append(LadderShortfall(f.bucket, f.level, eff))
            granted.append(replace(f, level=eff))
            changed = True
        else:
            granted.append(f)
    return (granted if changed else spec.floors), shortfalls


def clamp_group_floors(universe: Universe, config: OptimizerConfig) -> Tuple[Universe, List[str]]:
    """A group floor that the (pinned) universe cannot reach makes every solve
    infeasible - lower it to the attainable maximum and report it."""
    groups, notes = [], []
    for g in universe.groups:
        others = universe.with_groups([x for x in universe.groups if x is not g])
        cap = max_attainable(ConstraintSpec(others, config), universe.selector(g.members))
        lo = g.lo
        if np.isfinite(cap) and lo > cap - 1e-9:
            notes.append(f"group '{g.name}' floor {lo:.2%} unreachable (max {cap:.2%}) - lowered")
            lo = max(0.0, cap - 1e-6)
        groups.append(GroupConstraint(g.name, g.members, lo, g.hi))
    return universe.with_groups(groups), notes
