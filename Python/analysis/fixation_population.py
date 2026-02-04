"""Run fixation analysis across multiple sessions.

This script selects sessions from ``session_manifest.yml`` based on the
requested experiment type and executes the full fixation analysis pipeline
for each one.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
import numpy as np
import pandas as pd
import yaml

from analysis import fixation_session
from analysis.fixation_session import main
from utils.session_loader import list_sessions_from_manifest, load_session

assert main is fixation_session.main


def _animal_suffix(animal_name: str | None) -> str:
    """Return a filesystem-friendly suffix for ``animal_name``."""

    if not animal_name:
        return ""

    safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", animal_name.strip())
    safe_name = safe_name.strip("_")
    return f"_{safe_name}" if safe_name else ""


def analyze_all_sessions(
    experiment_type: str = "fixation", animal_name: str | None = None
) -> pd.DataFrame:
    """Run fixation analysis on all sessions of ``experiment_type``.

    Parameters
    ----------
    experiment_type:
        Experiment type used to select sessions from the manifest.
    animal_name:
        Optional animal name used to further restrict the manifest lookup and
        annotate aggregated artefacts.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all processed sessions. If no sessions
        are found, an empty :class:`~pandas.DataFrame` is returned.
    """
    tables: list[pd.DataFrame] = []
    for session_id in list_sessions_from_manifest(
        experiment_type, match_prefix=True, animal_name=animal_name
    ):
        session_df = fixation_session.main(session_id)

        if "animal_name" not in session_df.columns or session_df["animal_name"].isna().all():
            session_cfg = load_session(session_id)
            session_df = session_df.copy()
            session_df["animal_name"] = session_cfg.animal_name

        missing_cols = [
            col
            for col in ("total_trials", "valid_trial_fraction")
            if col not in session_df.columns
        ]
        if missing_cols:
            session_df = session_df.copy()
            for col in missing_cols:
                session_df[col] = np.nan

        tables.append(session_df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_metric_trends(
    df: pd.DataFrame,
    save_dir: Path,
    *,
    animal_name: str | None = None,
    max_interval_s: float | None = None,
) -> None:
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
    animal_name:
        Optional animal name used to annotate saved plots and filenames.
    max_interval_s:
        Optional maximum interval (in seconds) used when pairing cue and go
        events. When provided, it is included in the plot titles for context.
    """

    if df.empty:
        return

    data = df.copy()
    data["session_date"] = pd.to_datetime(data["session_date"], errors="coerce")
    data.sort_values("session_date", inplace=True)

    order = np.arange(len(data))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(data)))

    if "valid_trial_fraction" in data.columns:
        valid_fractions = pd.to_numeric(
            data["valid_trial_fraction"], errors="coerce"
        )
    else:
        valid_fractions = pd.Series(np.nan, index=data.index, dtype=float)

    if "total_trials" in data.columns:
        total_trials = pd.to_numeric(data["total_trials"], errors="coerce")
    else:
        total_trials = pd.Series(np.nan, index=data.index, dtype=float)

    metrics = [
        (
            "mean_step_fix",
            "mean_step_fix_sem",
            "mean_step_rand",
            "mean_step_rand_sem",
            "Mean step (deg)",
            "fixation_mean_step_trend",
        ),
        (
            "mean_speed_fix",
            "mean_speed_fix_sem",
            "mean_speed_rand",
            "mean_speed_rand_sem",
            "Mean speed (deg/s)",
            "fixation_mean_speed_trend",
        ),
        (
            "net_drift_fix",
            "net_drift_fix_sem",
            "net_drift_rand",
            "net_drift_rand_sem",
            "Net drift (deg)",
            "fixation_net_drift_trend",
        ),
    ]

    suffix = _animal_suffix(animal_name)

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
                #color=color,
                color = 'darkblue',
                markersize=4,
                capsize=3,
            )
            ax.errorbar(
                order[i],
                data[rand_col].iloc[i],
                yerr=data.get(rand_sem_col, pd.Series([0])).iloc[i],
                fmt="x",
                #color=color,
                color = 'orange',
                markersize=4,
                capsize=3,
            )

            valid_label = "n/a"
            valid_value = valid_fractions.iloc[i]
            if pd.notna(valid_value):
                valid_label = f"{valid_value * 100:.0f}%"

            total_label = "n/a"
            total_value = total_trials.iloc[i]
            if pd.notna(total_value) and np.isfinite(total_value):
                total_float = float(total_value)
                if total_float.is_integer():
                    total_label = f"{int(total_float)}"
                else:
                    total_label = f"{total_float:.0f}"

            session_label = f"{valid_label} ({total_label})"

            fix_y = data[fix_col].iloc[i]
            if pd.notna(fix_y):
                ax.annotate(
                    session_label,

                    xy=(order[i], fix_y),
                    xytext=(-6, 6),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7,
                    color="dimgray",
                )

            rand_y = data[rand_col].iloc[i]
            if pd.notna(rand_y):
                # Get session date and format it
                session_date = data["session_date"].iloc[i]
                date_label = ""
                if pd.notna(session_date):
                    date_label = session_date.strftime("%Y-%m-%d") + "\n"

                ax.annotate(
                    date_label + session_label,
                    xy=(order[i], rand_y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    ha="left",
                    fontsize=7,
                    color="dimgray",
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
        title_suffix = f" ({animal_name})" if animal_name else ""
        interval_suffix = (
            f" – max Δt <{max_interval_s:.1f} s" if max_interval_s is not None else ""
        )
        validity_suffix = ""
        if "valid_trial_fraction" in data.columns:
            valid_series = pd.to_numeric(data["valid_trial_fraction"], errors="coerce")
            if valid_series.notna().any():
                mean_pct = float(valid_series.mean() * 100.0)
                validity_suffix = f" – mean valid trials {mean_pct:.0f}%"
        ax.set_title(
            f"{ylabel} by session{title_suffix}{interval_suffix}{validity_suffix}"
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
        fig.savefig(save_dir / f"{fname}{suffix}.png", bbox_inches="tight")
        plt.close(fig)


# Usage: python Python/analysis/fixation_population.py --animal-name Paris
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis across sessions filtered by experiment type",
    )
    parser.add_argument(
        "--experiment-type",
        default="fixation",
        help="Experiment type to process",
    )
    parser.add_argument(
        "--animal-name",
        default=None,
        help="Optional animal name to filter sessions",
    )
    args = parser.parse_args()
    aggregated = analyze_all_sessions(
        args.experiment_type, animal_name=args.animal_name
    )
    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "session_manifest.yml"
    try:
        with manifest_path.open() as f:
            manifest = yaml.safe_load(f) or {}
    except FileNotFoundError:
        manifest = {}
    raw_max_interval = manifest.get("max_interval_s")
    try:
        max_interval_s = float(raw_max_interval) if raw_max_interval is not None else None
    except (TypeError, ValueError):
        max_interval_s = None
    results_root = Path(manifest.get("results_root", root_dir))
    results_root.mkdir(parents=True, exist_ok=True)
    suffix = _animal_suffix(args.animal_name)
    #aggregated.to_csv(
        #results_root / f"fixation_population_results{suffix}.csv", index=False
    #)
    plot_metric_trends(
        aggregated,
        results_root,
        animal_name=args.animal_name,
        max_interval_s=max_interval_s,
    )
