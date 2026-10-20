"""Package logging.

The library logs to the ``saa_pm`` logger and stays silent by default (a
``NullHandler``), as a library should. ``SAAStudy(verbose=True)`` and the CLI
attach a plain console handler; configure ``logging.getLogger("saa_pm")``
yourself for anything else.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("saa_pm")
logger.addHandler(logging.NullHandler())

_CONSOLE_FLAG = "_saa_pm_console"


def enable_console_logging(level: int = logging.INFO) -> None:
    """Print saa_pm messages to stdout (idempotent)."""
    if not any(getattr(h, _CONSOLE_FLAG, False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        setattr(h, _CONSOLE_FLAG, True)
        logger.addHandler(h)
    logger.setLevel(level)
