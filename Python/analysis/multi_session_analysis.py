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

import pandas as pd

from analysis import prosaccade_session
from analysis.prosaccade_session import main
from utils.session_loader import list_sessions_from_manifest

assert main is prosaccade_session.main


def analyze_all_sessions(experiment_type: str = "fixation") -> pd.DataFrame:
    """Run prosaccade analysis on all sessions of ``experiment_type``.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all processed sessions.  If no sessions
        are found, an empty :class:`~pandas.DataFrame` is returned.
    """
    tables: list[pd.DataFrame] = []
    for session_id in list_sessions_from_manifest(experiment_type):
        session_df = prosaccade_session.main(session_id)
        tables.append(session_df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


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
    aggregated = analyze_all_sessions(args.experiment_type)
    aggregated.to_csv("multi_session_results.csv", index=False)

