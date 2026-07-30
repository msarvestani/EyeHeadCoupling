"""Run prosaccade analysis across multiple sessions.

This script selects sessions from ``session_manifest.yml`` based on the
requested experiment type and executes the full prosaccade analysis pipeline
for each one.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt


from analysis import prosaccade_session
from analysis.prosaccade_session import main
from utils.session_loader import load_session, list_sessions_from_manifest

assert main is prosaccade_session.main


def analyze_all_sessions(
    experiment_type: str | None = "prosaccade",
    animal_name: str | None = None,
):
    """Run prosaccade analysis on sessions that match the provided filters.

    Parameters
    ----------
    experiment_type:
        Experiment type used to select sessions from the manifest. When ``None``
        all experiment types are considered.
    animal_name:
        Optional animal name used to further restrict the manifest lookup.

    Returns
    -------
    tuple
        A tuple containing the aggregated session table, lists of left and
        right eye angles, dictionaries keyed by session date, and a set of the
        unique animal names that were processed.
    """

    tables: list[pd.DataFrame] = []
    left_angle_all = []
    right_angle_all = []
    left_angle_all_with_dates = {}
    right_angle_all_with_dates = {}
    processed_animals: set[str] = set()

    for session_id in list_sessions_from_manifest(
        experiment_type,
        match_prefix=True,
        animal_name=animal_name,
    ):
        session_df, left_angle, right_angle = prosaccade_session.main(session_id)
        session_cfg = load_session(session_id)

        session_df = session_df.copy()
        session_df["animal_name"] = session_cfg.animal_name
        if session_cfg.animal_name:
            processed_animals.add(session_cfg.animal_name)

        tables.append(session_df)
        left_angle_all.append(left_angle)
        right_angle_all.append(right_angle)
        date_str = (
            session_df["session_date"].iloc[0]
            if "session_date" in session_df.columns
            else "unknown_date"
        )
        left_angle_all_with_dates[date_str] = left_angle
        right_angle_all_with_dates[date_str] = right_angle

    if not tables:
        return (
            pd.DataFrame(),
            left_angle_all,
            right_angle_all,
            left_angle_all_with_dates,
            right_angle_all_with_dates,
            processed_animals,
        )

    return (
        pd.concat(tables, ignore_index=True),
        left_angle_all,
        right_angle_all,
        left_angle_all_with_dates,
        right_angle_all_with_dates,
        processed_animals,
    )

def plot_prosaccade_trends_from_dictionary(
    left_angle_dict: dict,
    right_angle_dict: dict,
    experiment_type: str = "prosaccade",
    animal_label: str | None = None,
) -> None:

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
    title = f"{experiment_type} Saccade Percentages Over Time"
    label_text = (str(animal_label).strip() if animal_label is not None else "")
    animal_suffix = ""
    if label_text:
        title = f"{title} – {label_text}"
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label_text).strip("_")
        if safe_label:
            animal_suffix = f"_{safe_label}"
    ax.set_xticks(range(len(sorted_dates)))
    ax.set_xticklabels(sorted_dates, rotation=45)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Save the plot
    fig.savefig(results_root / f"{experiment_type}_saccade_percentage_trends{animal_suffix}.png")
    fig.savefig(results_root / f"{experiment_type}_saccade_percentage_trends{animal_suffix}.svg")


# Usage: python Python/analysis/prosaccade_population.py --experiment-type prosaccade [--animal-name ANIMAL_NAME]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis across sessions filtered by experiment type",
    )
    parser.add_argument(
        "--experiment-type",
        default="prosaccade",
        help="Experiment type to process",
    )
    parser.add_argument(
        "--animal-name",
        default=None,
        help="Optional animal name to filter sessions",
    )
    args = parser.parse_args()
    (
        aggregated,
        left_angle_all,
        right_angle_all,
        left_angle_all_with_dates,
        right_angle_all_with_dates,
        processed_animals,
    ) = analyze_all_sessions(
        args.experiment_type,
        animal_name=args.animal_name,
    )
    root_dir = Path(__file__).resolve().parents[2]
        
    manifest_path = root_dir / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    results_root = Path(manifest.get("results_root") or root_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    ### Plot the left right angle results
    from eyehead.analysis import plot_left_right_angle
    left_angle_all = np.concatenate(left_angle_all)
    right_angle_all = np.concatenate(right_angle_all)
    animal_label = None
    if isinstance(aggregated, pd.DataFrame) and not aggregated.empty and "session_id" in aggregated:
        session_ids = aggregated["session_id"].dropna().unique()
        animal_names: list[str] = []
        for session_id in session_ids:
            try:
                session_cfg = load_session(session_id)
            except KeyError:
                continue
            if session_cfg.animal_name:
                animal_names.append(session_cfg.animal_name)
        if animal_names:
            # Preserve manifest order while removing duplicates
            unique_animals = list(dict.fromkeys(animal_names))
            animal_label = ", ".join(unique_animals)

    plot_left_right_angle(
        left_angle_all,
        right_angle_all,
        35,
        sessionname="population",
        resultdir=results_root,
        experiment_type=args.experiment_type,
        animal_name=animal_label,
    )
    plot_prosaccade_trends_from_dictionary(
        left_angle_all_with_dates,
        right_angle_all_with_dates,
        experiment_type=args.experiment_type,
        animal_label=animal_label,
    )
    aggregated.to_csv(
        results_root / f"{args.experiment_type}_population_results.csv", index=False
    )
