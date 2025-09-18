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
import numpy as np
import yaml

import matplotlib.pyplot as plt


from analysis import prosaccade_session
from analysis.prosaccade_session import main
from utils.session_loader import list_sessions_from_manifest,load_session

assert main is prosaccade_session.main


def analyze_all_sessions(experiment_type: str = "prosaccade") -> pd.DataFrame:
    """Run prosaccade analysis on all sessions of ``experiment_type``.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all processed sessions. If no sessions
        are found, an empty :class:`~pandas.DataFrame` is returned.
    """
    tables: list[pd.DataFrame] = []
    left_angle_all = []
    right_angle_all = []
    left_angle_all_with_dates = {}  ## Dictionaries to hold the left and right angles with dates. TODO: Use these instead of the lists later
    right_angle_all_with_dates = {}
    
    for session_id in list_sessions_from_manifest(
        experiment_type, match_prefix=True
    ):
        session_df,left_angle,right_angle = prosaccade_session.main(session_id)
        tables.append(session_df)
        left_angle_all.append(left_angle)
        right_angle_all.append(right_angle)
        date_str = session_df["session_date"].iloc[0] if "session_date" in session_df.columns else "unknown_date"
        left_angle_all_with_dates[date_str] = left_angle
        right_angle_all_with_dates[date_str] = right_angle

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True), left_angle_all, right_angle_all, left_angle_all_with_dates, right_angle_all_with_dates

def plot_prosaccade_trends_from_dictionary(left_angle_dict: dict, right_angle_dict: dict, experiment_type: str = "prosaccade") -> None:

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # First sort the dictionary by date
    sorted_dates = sorted(left_angle_dict.keys())
    left_angles_sorted = [left_angle_dict[date] for date in sorted_dates]
    right_angles_sorted = [right_angle_dict[date] for date in sorted_dates]
    saccade_percentage_left_list = []
    saccade_percentage_right_list = []
    reward_angle = 35  # degrees
    for i, date in enumerate(sorted_dates):
        left_angles = left_angle_dict[date]
        right_angles = right_angle_dict[date]
        if experiment_type == "prosaccade":
            saccade_percentage_left = np.sum(np.abs(left_angles) <= np.deg2rad(reward_angle)) / len(left_angles) * 100
            saccade_percentage_right = np.sum(np.abs(right_angles) >= np.deg2rad(180-reward_angle)) / len(right_angles) * 100
        elif experiment_type == "antisaccade":
            saccade_percentage_left = np.sum(np.abs(left_angles) >= np.deg2rad(180-reward_angle)) / len(left_angles) * 100
            saccade_percentage_right = np.sum(np.abs(right_angles) <= np.deg2rad(reward_angle)) / len(right_angles) * 100

       # Plot the saccade percentages
        saccade_percentage_left_list.append(saccade_percentage_left)
        saccade_percentage_right_list.append(saccade_percentage_right)

    # Plot the sacccade percentage across sessions
    ax.plot(range(len(sorted_dates)), saccade_percentage_left_list, marker='o', color='b', label='Left Eye')
    ax.plot(range(len(sorted_dates)), saccade_percentage_right_list, marker='o', color='r', label='Right Eye')
    # Set plot labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Saccade Percentage (%)")
    ax.set_title(f"{experiment_type} Saccade Percentages Over Time")
    ax.set_xticks(range(len(sorted_dates)))
    ax.set_xticklabels(sorted_dates, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Save the plot
    fig.savefig(results_root / f"{experiment_type}_saccade_percentage_trends.png")
    fig.savefig(results_root / f"{experiment_type}_saccade_percentage_trends.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis across sessions filtered by experiment type",
    )
    parser.add_argument(
        "--experiment-type",
        default="prosaccade",
        help="Experiment type to process",
    )
    args = parser.parse_args()
    aggregated, left_angle_all, right_angle_all, left_angle_all_with_dates, right_angle_all_with_dates = analyze_all_sessions(args.experiment_type)
    root_dir = Path(__file__).resolve().parents[2]
        
    manifest_path = root_dir / "data" / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    results_root = Path(manifest.get("results_root") or root_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    ### Plot the left right angle results
    from  eyehead.analysis import plot_left_right_angle
    left_angle_all = np.concatenate(left_angle_all)
    right_angle_all = np.concatenate(right_angle_all)
    plot_left_right_angle(left_angle_all, right_angle_all, 35, sessionname=f"{args.experiment_type}_population", resultdir=results_root,experiment_type=args.experiment_type)
    plot_prosaccade_trends_from_dictionary(left_angle_all_with_dates, right_angle_all_with_dates, experiment_type=args.experiment_type)
    aggregated.to_csv(
        results_root / f"{args.experiment_type}_population_results.csv", index=False
    )