"""Static charts (PNG) for the report folder and the README.

One visual system: a fixed categorical order (colour follows the entity,
never its rank), recessive axes and gridlines, thin marks, direct labels
where they help, one y-axis per chart.

Figures are built with matplotlib's object-oriented ``Figure`` API inside a
temporary style context: importing or using this module never changes the
caller's matplotlib backend or global rcParams (safe in Jupyter).
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict, List, Mapping, Optional, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.transforms import ScaledTranslation

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9.5,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
}


def _styled(fn):
    """Apply STYLE only while ``fn`` builds and saves its figure."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(cast(Any, STYLE)):
            return fn(*args, **kwargs)

    return wrapper


def _subplots(*args, **kwargs):
    """A detached Figure with axes - no pyplot, no global figure registry."""
    figsize = kwargs.pop("figsize", None)
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(*args, **kwargs)


def colour_map(keys: List[str]) -> Dict[str, str]:
    """Stable entity -> colour assignment (fixed order; > 8 folds to grey)."""
    return {k: (SERIES[i] if i < len(SERIES) else MUTED) for i, k in enumerate(keys)}


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def _pct(ax, axis="y", decimals=0):
    fmt = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


@_styled
def allocation_by_class(
    weights: pd.DataFrame, classes: Mapping[str, str], path: str, title: str = "Allocation by asset class"
) -> str:
    """Horizontal 100% bars, one per portfolio (columns of ``weights``)."""
    by = weights.groupby(pd.Series(classes)).sum()
    order = list(dict.fromkeys(classes.values()))
    by = by.reindex([c for c in order if c in by.index])
    cmap = colour_map(list(by.index))
    fig, ax = _subplots(figsize=(8.5, 0.75 * len(by.columns) + 1.6))
    left = np.zeros(len(by.columns))
    for cls, row in by.iterrows():
        vals = row.values
        ax.barh(
            by.columns,
            vals,
            left=left,
            color=cmap[cls],
            edgecolor=SURFACE,
            linewidth=2,
            height=0.55,
            label=cls,
        )
        for i, v in enumerate(vals):
            if v >= 0.06:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(ncol=min(len(by), 6), loc="upper left", bbox_to_anchor=(0, -0.12))
    return _save(fig, path)


@_styled
def transition_path(
    weights: pd.DataFrame,
    classes: Mapping[str, str],
    path: str,
    title: str = "Projected allocation at each capital-call date",
) -> str:
    """Stacked bars by asset class across dates (rows of ``weights``)."""
    by = weights.T.groupby(pd.Series(classes)).sum()
    order = [c for c in dict.fromkeys(classes.values()) if c in by.index]
    by = by.reindex(order)
    cmap = colour_map(order)
    fig, ax = _subplots(figsize=(8.5, 4.2))
    x = np.arange(by.shape[1])
    bottom = np.zeros(len(x))
    for cls, row in by.iterrows():
        ax.bar(
            x,
            row.values,
            bottom=bottom,
            color=cmap[cls],
            edgecolor=SURFACE,
            linewidth=2,
            width=0.62,
            label=cls,
        )
        bottom += row.values
    ax.set_xticks(x, [str(c)[:7] for c in by.columns])
    ax.set_ylim(0, 1)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(ncol=min(len(order), 6), loc="upper left", bbox_to_anchor=(0, -0.08))
    return _save(fig, path)


@_styled
def frontiers(
    curves: Mapping[str, pd.DataFrame],
    points: Mapping[str, tuple],
    path: str,
    target_return: Optional[float] = None,
    band: Optional[tuple] = None,
    title: str = "Efficient frontiers",
) -> str:
    fig, ax = _subplots(figsize=(8.5, 4.8))
    cmap = colour_map(list(curves))
    if band:
        ax.axvspan(band[0], band[1], color=GRID, alpha=0.5, lw=0)
        ax.text(
            (band[0] + band[1]) / 2,
            0.98,
            "risk band",
            color=MUTED,
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )
    for k, f in curves.items():
        if len(f):
            ax.plot(f["vol"], f["exp_return"], color=cmap[k], label=k.replace("_", " "))
    if target_return is not None:
        ax.axhline(target_return, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(ax.get_xlim()[0], target_return, " return target", color=INK2, va="bottom", fontsize=8)
    for name, (v, r) in points.items():
        ax.plot(v, r, "o", ms=8, color=INK, mec=SURFACE, mew=2)
        ax.annotate(name, (v, r), xytext=(7, -3), textcoords="offset points", fontsize=8.5)
    _pct(ax, "x")
    _pct(ax, "y", 1)
    ax.set_xlabel("Volatility (annualised)")
    ax.set_ylabel("Expected return")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, path)


@_styled
def backtest_growth(
    returns: pd.DataFrame, path: str, title: str = "Walk-forward backtest: growth of 1"
) -> str:
    fig, (ax, ax2) = _subplots(
        2, 1, figsize=(8.5, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.08)
    )
    cmap = colour_map(list(returns.columns))
    ends = {}
    for c in returns.columns:
        cum = (1 + returns[c].fillna(0)).cumprod()
        ax.plot(
            cum.index, cum.values, color=cmap[c], label=c.replace("_", " "), lw=2.2 if c == "process" else 1.6
        )
        ends[c] = (cum.index[-1], float(cum.iloc[-1]))
        dd = cum / np.maximum(cum.cummax(), 1.0) - 1
        ax2.plot(dd.index, dd.values, color=cmap[c], lw=1.2)
    # end-of-line values, nudged apart so close finishes stay legible
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    shift = ax.transData + ScaledTranslation(4 / 72, 0, fig.dpi_scale_trans)
    placed: List[float] = []
    for _c, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        ylab = max(y, placed[-1] + gap) if placed else y
        placed.append(ylab)
        ax.text(x, ylab, f"{y:.2f}", transform=shift, va="center", fontsize=8, color=INK2, clip_on=False)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.set_ylabel("Drawdown")
    _pct(ax2)
    return _save(fig, path)


@_styled
def risk_vs_weight(table: pd.DataFrame, path: str, title: str) -> str:
    """Grouped bars per asset class: share of capital vs share of risk."""
    fig, ax = _subplots(figsize=(8.5, 4.0))
    x = np.arange(len(table))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, table["weight"], w, color=SERIES[0], label="share of capital")
    ax.bar(x + w / 2 + 0.01, table["pct_of_risk"], w, color=SERIES[1], label="share of risk")
    ax.set_xticks(x, table.index, rotation=0)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, path)


@_styled
def pm_cashflows(
    annual: pd.DataFrame,
    path: str,
    scale: float = 1e6,
    title: str = "Private markets: projected calls and distributions",
) -> str:
    fig, ax = _subplots(figsize=(8.5, 4.2))
    x = annual.index.astype(int)
    ax.bar(x, -annual["call"] / scale, color=SERIES[1], width=0.6, label="capital calls")
    ax.bar(x, annual["distribution"] / scale, color=SERIES[0], width=0.6, label="distributions")
    ax.plot(
        x,
        annual["cum_net_cashflow"] / scale,
        color=INK,
        lw=1.6,
        marker="o",
        ms=4,
        label="cumulative net cash flow",
    )
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}m"))
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    return _save(fig, path)


@_styled
def wealth_fan(
    pct: pd.DataFrame,
    path: str,
    start_value: float,
    scale: float = 1e6,
    target_path: Optional[pd.Series] = None,
    title: str = "Wealth projection (Monte Carlo)",
) -> str:
    fig, ax = _subplots(figsize=(8.5, 4.2))
    x = pct.index
    ax.fill_between(
        x, pct["p5"] / scale, pct["p95"] / scale, color="#cde2fb", lw=0, label="5th-95th percentile"
    )
    ax.fill_between(
        x, pct["p25"] / scale, pct["p75"] / scale, color="#86b6ef", lw=0, label="25th-75th percentile"
    )
    ax.plot(x, pct["p50"] / scale, color=SERIES[0], label="median")
    if target_path is not None:
        ax.plot(
            target_path.index,
            target_path.values / scale,
            color=INK2,
            lw=1,
            ls=(0, (4, 3)),
            label="return target path",
        )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}m"))
    ax.set_xlabel("Years from today")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, path)


@_styled
def stress_bars(pnl: pd.Series, path: str, title: str) -> str:
    s = pnl.sort_values()
    fig, ax = _subplots(figsize=(8.5, 0.32 * len(s) + 1.4))
    colours = [CRITICAL if v < 0 else GOOD for v in s.values]
    ax.barh(s.index, s.values, color=colours, height=0.6)
    for i, v in enumerate(s.values):
        ax.text(
            v,
            i,
            f" {v:+.1%} " if v >= 0 else f" {v:+.1%} ",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=INK2,
        )
    ax.axvline(0, color=AXIS, lw=0.8)
    lo, hi = min(s.min(), 0.0), max(s.max(), 0.0)
    pad = 0.18 * (hi - lo or 0.01)
    ax.set_xlim(lo - pad, hi + pad)
    _pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    return _save(fig, path)
