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


def _animal_prefix(animal_name: str | None) -> str:
    """Return a filesystem-friendly prefix for ``animal_name``."""

    if not animal_name:
        return ""

    safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", animal_name.strip())
    safe_name = safe_name.strip("_")
    return f"{safe_name}_" if safe_name else ""


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
        experiment_type, match_prefix=False, animal_name=animal_name
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


def plot_active_stabilization(
    df: pd.DataFrame,
    save_dir: Path,
    *,
    animal_name: str | None = None,
) -> None:
    """Plot active_stabilization across sessions, one point per session."""
    if df.empty or "active_stabilization" not in df.columns:
        return

    data = df.copy()
    data["session_date"] = pd.to_datetime(data["session_date"], errors="coerce")
    data.sort_values("session_date", inplace=True, ignore_index=True)

    n_sessions = len(data)
    x_pos = np.arange(n_sessions)
    session_numbers = x_pos + 1  # 1-based labels

    metric = pd.to_numeric(data["active_stabilization"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, n_sessions * 1.1), 4))

    ax.plot(x_pos, metric, linestyle="--", color="0.75", linewidth=1, zorder=1)
    ax.scatter(x_pos, metric, color="steelblue", s=60, zorder=2)

    for i in range(n_sessions):
        y = metric[i]
        if not np.isfinite(y):
            continue
        date_val = data["session_date"].iloc[i]
        label = date_val.strftime("%Y-%m-%d") if pd.notna(date_val) else ""
        ax.annotate(
            label,
            xy=(x_pos[i], y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="dimgray",
            rotation=45,
        )

    ax.axhline(0, color="0.4", linestyle=":", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(session_numbers, fontsize=8)
    ax.set_xlabel("Session number")
    ax.set_ylabel("Active stabilization\n(cue_suppression × selection_bias²)")
    title = "Active stabilization across sessions"
    if animal_name:
        title += f" ({animal_name})"
    ax.set_title(title)

    fig.tight_layout()
    animal_id = data["animal_id"].dropna().iloc[0] if "animal_id" in data.columns and not data["animal_id"].dropna().empty else animal_name
    prefix = _animal_prefix(animal_id)
    for ext in ("png", "svg"):
        fig.savefig(save_dir / f"{prefix}active_stabilization_trend.{ext}", bbox_inches="tight")
    plt.show()
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
    raw_max_interval = manifest.get("max_interval_fixations")
    try:
        max_interval_fixations = float(raw_max_interval) if raw_max_interval is not None else None
    except (TypeError, ValueError):
        max_interval_fixations = None
    results_root = Path(manifest.get("results_root", root_dir))
    results_root.mkdir(parents=True, exist_ok=True)
    plot_active_stabilization(
        aggregated,
        results_root,
        animal_name=args.animal_name,
    )
