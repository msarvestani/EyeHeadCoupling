"""Run prosaccade analysis across multiple sessions.

This script selects sessions from ``data/session_manifest.yml`` based on the
requested experiment type and executes the full prosaccade analysis pipeline
for each one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis import prosaccade_session
from analysis.prosaccade_session import main
from utils.session_loader import list_sessions_from_manifest

assert main is prosaccade_session.main


def analyze_all_sessions(experiment_type: str = "fixation") -> None:
    """Run prosaccade analysis on all sessions of ``experiment_type``."""
    for session_id in list_sessions_from_manifest(experiment_type):
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

