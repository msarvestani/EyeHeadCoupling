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

import matplotlib.pyplot as plt
import numpy as np
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
    for session_id in list_sessions_from_manifest(
        experiment_type, match_prefix=True
    ):
        session_df = fixation_session.main(session_id)
        tables.append(session_df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_metric_trends(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot fixation metrics across sessions with a consistent colour scheme.

    Sessions are ordered by recording date and assigned colours from a
    sequential colormap.  For each metric, both fixation and random values are
    plotted using the same session colour but different markers.

    Parameters
    ----------
    df:
        Aggregated fixation metrics for all sessions.
    save_dir:
        Directory where the plot images will be written.
    """

    if df.empty:
        return

    data = df.copy()
    data["session_date"] = pd.to_datetime(data["session_date"], errors="coerce")
    data.sort_values("session_date", inplace=True)

    order = np.arange(len(data))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(data)))

    metrics = [
        (
            "mean_step_fix",
            "mean_step_fix_sem",
            "mean_step_rand",
            "mean_step_rand_sem",
            "Mean step (deg)",
            "fixation_mean_step_trend.png",
        ),
        (
            "mean_speed_fix",
            "mean_speed_fix_sem",
            "mean_speed_rand",
            "mean_speed_rand_sem",
            "Mean speed (deg/s)",
            "fixation_mean_speed_trend.png",
        ),
        (
            "net_drift_fix",
            "net_drift_fix_sem",
            "net_drift_rand",
            "net_drift_rand_sem",
            "Net drift (deg)",
            "fixation_net_drift_trend.png",
        ),
    ]

    for fix_col, fix_sem_col, rand_col, rand_sem_col, ylabel, fname in metrics:
        if fix_col not in data or rand_col not in data:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, color in enumerate(colors):
            ax.errorbar(
                order[i],
                data[fix_col].iloc[i],
                yerr=data.get(fix_sem_col, pd.Series([0])).iloc[i],
                fmt="o",
                color=color,
                markersize=4,
                capsize=3,
            )
            ax.errorbar(
                order[i],
                data[rand_col].iloc[i],
                yerr=data.get(rand_sem_col, pd.Series([0])).iloc[i],
                fmt="x",
                color=color,
                markersize=4,
                capsize=3,
            )

        # Connect sessions with dashed lines for fixation and random conditions
        ax.plot(
            order,
            data[fix_col],
            linestyle="--",
            color="tab:blue",
            label="Fixation",
        )
        ax.plot(
            order,
            data[rand_col],
            linestyle="--",
            color="tab:orange",
            label="Random",
        )
        ax.set_xlabel("Session (earlier → later)")
        ax.set_ylabel(ylabel)

        norm = plt.Normalize(vmin=0, vmax=len(data) - 1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Session order")
        ax.legend()

        fig.tight_layout()
        fig.savefig(save_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


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
    plot_metric_trends(aggregated, results_root)
