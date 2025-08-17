"""Utilities for discovering experiment sessions.

The functions in this module scan a base data directory for session folders.
Sessions are expected to be stored in subdirectories of ``data`` with names
that begin with an experiment type followed by an underscore, e.g.::

    headfixed_2025-06-01
    freelymoving_2025-06-02

The module exposes two convenience functions:

``list_sessions``
    Return all available session folder names.
``list_sessions_by_type``
    Return only the session names matching a given experiment type.

The base data directory can be overridden with the environment variable
``EHC_DATA_DIR``. If the directory does not exist, an empty list is returned.
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, List

# Base directory containing all recorded sessions.  Users may override this by
# setting the ``EHC_DATA_DIR`` environment variable.
_DATA_DIR = Path(os.environ.get("EHC_DATA_DIR", Path(__file__).resolve().parent / "data"))


def _session_dirs(base: Path | None = None) -> Iterable[Path]:
    """Yield session directories located under ``base``.

    Parameters
    ----------
    base:
        Optional path to search. When omitted, the module level ``_DATA_DIR``
        is used. The directory is created lazily if it does not exist.
    """
    root = base or _DATA_DIR
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def list_sessions() -> List[str]:
    """Return the names of all available sessions.

    Returns
    -------
    list of str
        Sorted list of session directory names. If the data directory is
        missing, an empty list is returned.
    """
    return sorted(p.name for p in _session_dirs())


def list_sessions_by_type(experiment_type: str) -> List[str]:
    """Return sessions whose name matches ``experiment_type``.

    Parameters
    ----------
    experiment_type:
        Prefix identifying an experiment class. Session names are expected to
        have the form ``"{experiment_type}_<details>"``. The comparison is
        case-insensitive.

    Returns
    -------
    list of str
        Sorted list of session names beginning with ``experiment_type``.
    """
    prefix = f"{experiment_type.lower()}_"
    return sorted(
        name
        for name in list_sessions()
        if name.lower().startswith(prefix)
    )


__all__ = ["list_sessions", "list_sessions_by_type"]
