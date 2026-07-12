"""Logging configuration for the AB Test Analyzer.

Call ``setup_logging()`` once at application startup (in ``main.py``)
to configure the root logger with a consistent format.
"""

from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for console output.

    Args:
        level: Minimum severity level to capture.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if called more than once.
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with the given *name*.

    Usage:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
