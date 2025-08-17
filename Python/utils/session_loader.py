"""Utilities for loading session configuration data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

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

    manifest_path = Path(__file__).resolve().parent.parent / "session_manifest.yml"

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


__all__ = ["SessionConfig", "load_session"]
