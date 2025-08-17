"""Example driver for running analyses across multiple sessions.

This script shows how to leverage the ``session_loader`` helpers to iterate
through available sessions. Replace the ``analyze_session`` function with the
actual analysis routine for a single recording session.
"""
from __future__ import annotations

from utils.session_loader import list_sessions, list_sessions_by_type


def analyze_session(session_name: str) -> None:
    """Placeholder for per-session analysis.

    Parameters
    ----------
    session_name:
        Name of the session directory to process.
    """
    # In a real project, this function would load data and perform analysis.
    print(f"Analyzing session {session_name}")


def analyze_all_sessions(experiment_type: str | None = None) -> None:
    """Run analysis on all sessions, optionally filtered by type."""
    sessions = (
        list_sessions_by_type(experiment_type)
        if experiment_type is not None
        else list_sessions()
    )
    for session in sessions:
        analyze_session(session)


if __name__ == "__main__":
    analyze_all_sessions()
