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
    experiment_type: str = "fixation",
    animal_name: str | None = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """Run fixation analysis on all sessions of ``experiment_type``.

    Parameters
    ----------
    experiment_type:
        Experiment type used to select sessions from the manifest.
    animal_name:
        Optional animal name used to further restrict the manifest lookup and
        annotate aggregated artefacts.
    show_plots:
        When ``True`` per-session figures are displayed interactively.
        Defaults to ``False`` so population runs don't require clicking through
        windows. Figures are always saved regardless of this flag.

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
        session_df = fixation_session.main(session_id, show_plots=show_plots)

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
    show_plots: bool = False,
    fname_stem: str = "fixation_trend",
    title: str = "Fixation without visual feedback across sessions",
) -> None:
    """Plot active_stabilization across sessions, one point per session.

    When the DataFrame contains multiple animals, each is plotted in a
    different color with its own session numbering and a legend.
    """
    if df.empty or "active_stabilization" not in df.columns:
        return

    data = df.copy()
    data["session_date"] = pd.to_datetime(data["session_date"], errors="coerce")
    data.sort_values(["animal_id", "session_date"], inplace=True, ignore_index=True)

    id_col = "animal_id" if "animal_id" in data.columns else "animal_name"
    animals = data[id_col].dropna().unique()
    multi = len(animals) > 1

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(animals), 1)))
    color_map = {a: colors[i] for i, a in enumerate(animals)}

    max_sessions = max(len(data[data[id_col] == a]) for a in animals)
    fig, ax = plt.subplots(figsize=(max(6, max_sessions * 1.1), 4))

    for animal in animals:
        adf = data[data[id_col] == animal].reset_index(drop=True)
        n = len(adf)
        x_pos = np.arange(n)
        metric = pd.to_numeric(adf["active_stabilization"], errors="coerce").to_numpy()
        color = color_map[animal]

        ax.plot(x_pos, metric, linestyle="--", color=color, linewidth=1, alpha=0.7, zorder=1)
        ax.scatter(x_pos, metric, color=color, s=60, zorder=2,
                   label=str(animal) if multi else None)

        # Annotate points with session dates, rotated to avoid overlap. This is a bit hacky but it works reasonably well for a small number of sessions.
        # for i in range(n):
        #     y = metric[i]
        #     if not np.isfinite(y):
        #         continue
        #     date_val = adf["session_date"].iloc[i]
        #     label = date_val.strftime("%Y-%m-%d") if pd.notna(date_val) else ""
        #     ax.annotate(
        #         label,
        #         xy=(x_pos[i], y),
        #         xytext=(0, 8),
        #         textcoords="offset points",
        #         ha="center",
        #         va="bottom",
        #         fontsize=7,
        #         color=color,
        #         rotation=45,
        #     )

    ax.axhline(0, color="0.4", linestyle=":", linewidth=0.8)
    ax.set_xticks(np.arange(max_sessions))
    ax.set_xticklabels(np.arange(1, max_sessions + 1), fontsize=8)
    ax.set_xlabel("Session number")
    ax.set_ylabel("Active stabilization\n(cue_suppression × selection_bias²)")
    full_title = title
    if animal_name:
        full_title += f" ({animal_name})"
    ax.set_title(full_title)
    if multi:
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    animal_id = "_".join(str(a) for a in animals) if multi else (
        data[id_col].dropna().iloc[0] if not data[id_col].dropna().empty else animal_name
    )
    prefix = _animal_prefix(animal_id)
    for ext in ("png", "svg"):
        fig.savefig(save_dir / f"{prefix}{fname_stem}.{ext}", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


# Usage: python Python/analysis/fixation_population.py --animal-name Paris
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fixation and fixation_learning analysis across sessions",
    )
    parser.add_argument(
        "--animal-name",
        nargs="+",
        default=None,
        help="One or more animal names to include (e.g. --animal-name Paris Apollo)",
    )
    parser.add_argument(
        "--experiment-type",
        nargs="+",
        default=None,
        choices=["fixation", "fixation_learning"],
        help="Which experiment type(s) to process (default: both)",
    )
    parser.add_argument(
        "--show-session-plots",
        action="store_true",
        default=False,
        help="Display per-session figures interactively (always saved; hidden by default)",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "session_manifest.yml"
    try:
        with manifest_path.open() as f:
            manifest = yaml.safe_load(f) or {}
    except FileNotFoundError:
        manifest = {}
    results_root = Path(manifest.get("results_root", root_dir))
    results_root.mkdir(parents=True, exist_ok=True)

    animal_names = args.animal_name or [None]
    title_name = " & ".join(animal_names) if animal_names != [None] else None

    all_experiment_configs = {
        "fixation": (
            "fixation_trend",
            "Fixation without visual feedback across sessions",
        ),
        "fixation_learning": (
            "fixation_learning_trend",
            "Fixation learning across sessions",
        ),
    }
    requested = args.experiment_type or list(all_experiment_configs.keys())
    experiment_configs = [
        (exp_type, *all_experiment_configs[exp_type]) for exp_type in requested
    ]

    for exp_type, fname_stem, plot_title in experiment_configs:
        frames = [
            analyze_all_sessions(
                exp_type, animal_name=name, show_plots=args.show_session_plots
            )
            for name in animal_names
        ]
        aggregated = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if aggregated.empty:
            print(f"No sessions found for experiment_type='{exp_type}', skipping.")
            continue
        plot_active_stabilization(
            aggregated,
            results_root,
            animal_name=title_name,
            show_plots=args.show_session_plots,
            fname_stem=fname_stem,
            title=plot_title,
        )
