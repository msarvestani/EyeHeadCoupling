"""Run prosaccade analysis across multiple sessions.

This script selects sessions from ``data/session_manifest.yml`` based on the
requested experiment type and executes the full prosaccade analysis pipeline
for each one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis import prosaccade_session
from analysis.prosaccade_session import main

assert main is prosaccade_session.main


def _sessions_by_type(experiment_type: str) -> list[str]:
    """Return session IDs matching ``experiment_type`` from the manifest."""
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    sessions = manifest.get("sessions", {})
    return [
        session_id
        for session_id, meta in sessions.items()
        if meta.get("experiment_type") == experiment_type
    ]


def analyze_all_sessions(experiment_type: str = "fixation") -> None:
    """Run prosaccade analysis on all sessions of ``experiment_type``."""
    for session_id in _sessions_by_type(experiment_type):
        prosaccade_session.main(session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis across sessions filtered by experiment type"
    )
    parser.add_argument(
        "--experiment-type",
        default="fixation",
        help="Experiment type to process",
    )
    args = parser.parse_args()
    analyze_all_sessions(args.experiment_type)

