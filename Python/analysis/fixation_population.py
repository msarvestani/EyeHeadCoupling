"""Run fixation analysis across multiple sessions.

This script selects sessions from ``data/session_manifest.yml`` based on the
requested experiment type and executes the full fixation analysis pipeline
for each one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from analysis import fixation_session
from analysis.fixation_session import main
from utils.session_loader import list_sessions_from_manifest

assert main is fixation_session.main


def analyze_all_sessions(experiment_type: str = "fixation") -> pd.DataFrame:
    """Run fixation analysis on all sessions of ``experiment_type``.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all processed sessions. If no sessions
        are found, an empty :class:`~pandas.DataFrame` is returned.
    """
    tables: list[pd.DataFrame] = []
    for session_id in list_sessions_from_manifest(experiment_type):
        session_df = fixation_session.main(session_id)
        tables.append(session_df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis across sessions filtered by experiment type",
    )
    parser.add_argument(
        "--experiment-type",
        default="fixation",
        help="Experiment type to process",
    )
    args = parser.parse_args()
    aggregated = analyze_all_sessions(args.experiment_type)
    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "data" / "session_manifest.yml"
    try:
        with manifest_path.open() as f:
            manifest = yaml.safe_load(f) or {}
    except FileNotFoundError:
        manifest = {}
    results_root = Path(manifest.get("results_root", root_dir))
    results_root.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(
        results_root / "fixation_population_results.csv", index=False
    )