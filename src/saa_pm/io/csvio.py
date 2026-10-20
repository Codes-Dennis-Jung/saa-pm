"""Locale-tolerant CSV reading.

Every input file goes through :func:`read_csv`, which

1. sniffs the field separator from a whitelist (``, ; tab |``) by voting
   across the first lines (pandas' ``sep=None`` sniffer can pick a letter);
2. strips the UTF-8 BOM that Windows Excel prepends;
3. decides **once per file** whether ``,`` is the decimal mark (German
   locale) or a thousands separator (US locale) and, for German files,
   rewrites every numeric token to US form (``'3,67%' -> '3.67%'``,
   ``'1.114,07' -> '1114.07'``). Dates, names and free text are untouched
   because only tokens that *fully* match a number pattern are rewritten.

Downstream of :func:`read_csv` there is exactly one numeric convention.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import List

import numpy as np
import pandas as pd

from .._logging import logger

_SEP_CANDIDATES = [",", ";", "\t", "|"]
_NA_TOKENS = {"", "na", "nan", "none", "n/a"}

# A token that is unambiguously a German-formatted number. '31.03.2016' does
# NOT match (groups after the dots are not 3 digits), so dates pass through.
_DE_NUM = re.compile(r"[-+]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?\s*%?")


def sniff_separator(path: str) -> str:
    """Pick the field separator from a whitelist by voting across lines.

    Ties (a description cell containing commas inside a ';' file) are broken
    by consistency: a real separator appears equally often on every row.
    """
    lines: List[str] = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                lines.append(t)
                if len(lines) >= 6:
                    break
    name = os.path.basename(path)
    if not lines:
        raise ValueError(f"{name} is empty")
    counts = {c: lines[0].count(c) for c in _SEP_CANDIDATES}
    best = max(counts.values())
    if best == 0:
        raise ValueError(
            f"{name}: no field separator (, ; tab |) in the header "
            f"line {lines[0][:60]!r} - re-save the file as CSV."
        )
    tied = [c for c, n in counts.items() if n == best]
    for c in tied:
        if len(tied) > 1 and len({ln.count(c) for ln in lines}) == 1:
            return c
    return tied[0]


def _de_to_us_cell(v):
    """'3,67%' -> '3.67%', '1.114,07' -> '1114.07', '0,25|0,50' -> '0.25|0.50'."""
    if not isinstance(v, str):
        return v

    def one(tok: str) -> str:
        t = tok.strip()
        return t.replace(".", "").replace(",", ".") if _DE_NUM.fullmatch(t) else tok

    return "|".join(one(p) for p in v.split("|")) if "|" in v else one(v)


def detect_decimal_style(frame: pd.DataFrame, sep_hint: str = ",") -> str:
    """Return ``'de'`` (decimal comma) or ``'us'`` (decimal dot) for a file.

    Votes are cast only by *decisive* cells; '1,500' alone is ambiguous. With
    no decisive cell the separator decides: ';' implies a decimal-comma locale.
    """
    de = us = 0
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        for v in s.dropna().astype(str).str.strip():
            for t in v.split("|"):
                has_c, has_d = "," in t, "." in t
                if has_c and has_d:
                    de, us = (de + 1, us) if t.rfind(",") > t.rfind(".") else (de, us + 1)
                elif has_c:
                    if t.count(",") > 1:
                        us += 1
                    elif len(t.split(",")[1]) != 3:
                        de += 1
                elif has_d:
                    if t.count(".") > 1:
                        de += 1
                    elif len(t.split(".")[1]) != 3:
                        us += 1
    if de != us:
        return "de" if de > us else "us"
    return "de" if sep_hint == ";" else "us"


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV regardless of the locale that wrote it (see module doc)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing input file: {path}")
    sep = sniff_separator(path)
    df = pd.read_csv(
        path, sep=sep, engine="python", comment="#", skipinitialspace=True, encoding="utf-8-sig", dtype=str
    )
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    # drop fully empty trailing columns written by Excel (';;' at line ends)
    df = df.loc[:, [c for c in df.columns if not (c.startswith("Unnamed") and df[c].isna().all())]]
    if df.shape[1] == 1 and len(df):
        raise ValueError(
            f"{os.path.basename(path)} parsed as a single column "
            f"('{df.columns[0]}') with separator {sep!r}. Re-save as CSV."
        )
    df = df.dropna(how="all")
    if detect_decimal_style(df, sep) == "de":
        df = df.apply(lambda c: c.map(_de_to_us_cell))
    df.attrs["sep"] = sep
    return df


def to_float(x, default: float = np.nan) -> float:
    """Parse a scalar: 0.034, '3.4%', '3,4', '1,500,000', '1.114,07', "1'500"."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    if s.lower() in _NA_TOKENS:
        return default
    pct = s.endswith("%")
    s = s.rstrip("%").replace(" ", "").replace("\xa0", "").replace("'", "").replace("’", "")
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif has_c:
        s = s.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError as exc:
        raise ValueError(f"cannot parse {x!r} as a number") from exc
    return v / 100.0 if pct else v


def to_numeric_column(col: pd.Series) -> pd.Series:
    """Vectorised numeric parse of an already US-normalised column."""
    s = col.astype(str).str.strip()
    s = s.where(~s.str.lower().isin(_NA_TOKENS))
    pct = s.str.endswith("%", na=False)
    s = s.str.rstrip("%").str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    return out.where(~pct, out / 100.0)


def to_bool(x, default: bool = True) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    s = str(x).strip().lower()
    if s in _NA_TOKENS:
        return default
    return s in ("1", "true", "t", "yes", "y", "wahr")


def to_str(x, default: str = "") -> str:
    """Blank cells arrive as NaN, and NaN is truthy - never ``str(x or d)``."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    s = str(x).strip()
    return s if s else default


def to_list(x) -> List[str]:
    """Split a '|'-separated (or ';'-separated) member list."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [p.strip() for p in str(x).replace(";", "|").split("|") if p.strip()]


def as_rate(v: float, where: str, notes: List[str] | None = None) -> float:
    """Rates live in [0, 1]; a value > 1 is read as a percent (4.2 -> 0.042).

    Only route known rate fields through this - never counts or bps.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)) or abs(v) <= 1.0:
        return v
    msg = f"{where}: {v:g} > 1 for a rate - read as {v:g}% = {v / 100.0:g}"
    if notes is not None:
        notes.append(msg)
    else:
        logger.warning(msg)
    return v / 100.0


def parse_dates(orig: pd.Series, where: str) -> pd.Series:
    """Parse dates trying both day-first and month-first orientations.

    Preference: fewer failures, then the monotonic reading, then (full tie,
    every day <= 12 in a dotted file) the European day-first reading.
    Unparseable dates raise instead of silently deleting rows.
    """
    orig = orig.astype(str).str.strip()
    iso = orig.str.fullmatch(r"\d{4}-\d{2}-\d{2}.*").all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if iso:
            parsed = pd.to_datetime(orig, errors="coerce", format="mixed")
        else:
            p_us = pd.to_datetime(orig, dayfirst=False, errors="coerce", format="mixed")
            p_eu = pd.to_datetime(orig, dayfirst=True, errors="coerce", format="mixed")
            n_us, n_eu = int(p_us.isna().sum()), int(p_eu.isna().sum())
            if n_us != n_eu:
                parsed = p_us if n_us < n_eu else p_eu
            else:
                mono_us = bool(p_us.dropna().is_monotonic_increasing)
                mono_eu = bool(p_eu.dropna().is_monotonic_increasing)
                if mono_eu != mono_us:
                    parsed = p_eu if mono_eu else p_us
                else:
                    dotted = len(orig) > 0 and "." in str(orig.iloc[0])
                    parsed = p_eu if dotted else p_us
    bad = int(parsed.isna().sum())
    if bad:
        raise ValueError(f"{where}: {bad} unparseable date(s) (first: {orig[parsed.isna()].iloc[0]!r})")
    return parsed
