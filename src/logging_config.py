"""Centralised logging configuration.

Call ``configure_logging()`` once at application startup (e.g. from
``app/streamlit_app.py``). Every module then uses
``log = logging.getLogger(__name__)`` with no manual handler setup.

``configure_logging`` is idempotent — safe to call multiple times; only
the first call installs handlers.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install a stderr handler with a compact timestamped format.

    Third-party libraries that are chatty at INFO (urllib3, yfinance) are
    pinned to WARNING so our own logs stay readable.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    _CONFIGURED = True
