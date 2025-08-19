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
    for session_id in list_sessions_from_manifest(experiment_type):
        session_df = fixation_session.main(session_id)
        tables.append(session_df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_net_drift_trend(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot net drift across sessions using a colour gradient.

    Sessions are ordered by their recording date and assigned colours from a
    sequential colormap so that later sessions appear in a different hue than
    earlier ones. The resulting figure is saved into ``save_dir``.

    Parameters
    ----------
    df:
        Aggregated fixation metrics for all sessions.
    save_dir:
        Directory where the plot image will be written.
    """

    if df.empty or "net_drift_fix" not in df:
        return

    data = df.copy()
    data["session_date"] = pd.to_datetime(data["session_date"], errors="coerce")
    data.sort_values("session_date", inplace=True)

    order = np.arange(len(data))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(data)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(order, data["net_drift_fix"], c=colors, s=40)
    ax.set_xlabel("Session (earlier → later)")
    ax.set_ylabel("Net drift during fixation (deg)")

    norm = plt.Normalize(vmin=0, vmax=len(data) - 1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Session order")

    fig.tight_layout()
    fig.savefig(save_dir / "fixation_net_drift_trend.png", dpi=300, bbox_inches="tight")
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
    plot_net_drift_trend(aggregated, results_root)