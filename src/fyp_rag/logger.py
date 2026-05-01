"""Single source of truth for logger configuration."""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def get_logger(name: str = "fyp_rag", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide configured logger."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("fyp_rag")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True

    return logging.getLogger(name)
