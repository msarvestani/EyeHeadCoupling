"""Utilities for loading session configuration data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import os
import yaml


@dataclass
class SessionConfig:
    """Configuration for a data collection session.

    Parameters are stored in the ``params`` dictionary and can be
    accessed as attributes.  For example, if ``params`` contains a key
    ``"data_path"`` then ``config.data_path`` will return that value.
    """

    session_id: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        try:
            return self.params[item]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(item) from exc


def load_session(session_id: str) -> SessionConfig:
    """Load the configuration for ``session_id``.

    Parameters
    ----------
    session_id:
        Identifier of the session to load.

    Returns
    -------
    SessionConfig
        The configuration for the requested session.

    Raises
    ------
    KeyError
        If ``session_id`` is not present in ``session_manifest.yml``.
    """

    manifest_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "session_manifest.yml"
    )

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest: Dict[str, Any] = yaml.safe_load(fh) or {}

    # The manifest may either contain a top-level ``sessions`` key or map
    # session identifiers directly to their configuration.  Support both
    # layouts for flexibility.
    sessions: Dict[str, Any] = manifest.get("sessions", manifest)

    try:
        data = sessions[session_id] or {}
    except KeyError as exc:
        raise KeyError(f"Unknown session id: {session_id}") from exc

    return SessionConfig(session_id=session_id, params=data)


_DATA_DIR = Path(
    os.environ.get(
        "EHC_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "data"
    )
)


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
        name for name in list_sessions() if name.lower().startswith(prefix)
    )


__all__ = [
    "SessionConfig",
    "load_session",
    "list_sessions",
    "list_sessions_by_type",
]
