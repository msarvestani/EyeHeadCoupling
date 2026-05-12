"""Population-level psychometric curve for fixation-with-feedback sessions.

Aggregates per-session psychometric data (success rate vs target diameter)
across all sessions for one animal, producing:
  - thin per-session curves
  - thick mean ± SEM population curve

Sessions must be registered in session_manifest.yml with::

    experiment_type: fixation_feedback

Usage
-----
    python Python/analysis/fixationfeedback_population.py --animal-name Paris
    python Python/analysis/fixationfeedback_population.py --animal-name Paris --results /path/to/out
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import list_sessions_from_manifest, load_session
from fixationfeedback_session import load_fixation_feedback_data


# ---------------------------------------------------------------------------
# Per-session psychometric helper
# ---------------------------------------------------------------------------

def compute_session_psychometric(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> dict[float, tuple[float, int]]:
    """Return {diameter: (success_rate_0_to_1, n_trials)} for one session.

    Merges eot_df and target_df by row position (one row = one trial).
    Diameters are rounded to 3 decimal places to avoid float key collisions.
    """
    if "trial_success" not in eot_df.columns or "diameter" not in target_df.columns:
        return {}

    n = min(len(eot_df), len(target_df))
    combined = pd.DataFrame({
        "trial_success": eot_df["trial_success"].iloc[:n].values,
        "diameter": target_df["diameter"].iloc[:n].values,
    })

    result: dict[float, tuple[float, int]] = {}
    for diam, group in combined.groupby("diameter"):
        diam_key = round(float(diam), 3)
        n_trials = len(group)
        n_success = int((group["trial_success"] == 2).sum())
        result[diam_key] = (n_success / n_trials if n_trials > 0 else 0.0, n_trials)
    return result


# ---------------------------------------------------------------------------
# Population plot
# ---------------------------------------------------------------------------

def plot_population_psychometric(
    session_records: list[dict],
    animal_id: str = "",
    animal_name: str = "",
    results_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> plt.Figure:
    """Plot per-session curves + population mean ± SEM for one animal.

    Parameters
    ----------
    session_records:
        List of dicts, one per session::

            {
                "session_id": str,
                "date": str,          # YYYY-MM-DD
                "psychometric": {diameter: (success_rate, n_trials), ...}
            }

    animal_id:
        Short animal ID used for the filename prefix.
    animal_name:
        Human-readable animal name used in the plot title.
    results_dir:
        Directory to save the figure.  No file is written when ``None``.
    show_plots:
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Collect all diameters present across sessions
    all_diameters: set[float] = set()
    for rec in session_records:
        all_diameters.update(rec["psychometric"].keys())
    diameters = np.array(sorted(all_diameters))

    if len(diameters) == 0:
        print("Warning: no diameter data found across sessions — skipping population plot")
        return None

    # Build (n_sessions × n_diameters) rate matrix; NaN where session lacked a diameter
    n_sessions = len(session_records)
    rate_matrix = np.full((n_sessions, len(diameters)), np.nan)
    for s_idx, rec in enumerate(session_records):
        for d_idx, d in enumerate(diameters):
            if d in rec["psychometric"]:
                rate, _ = rec["psychometric"][d]
                rate_matrix[s_idx, d_idx] = rate * 100  # → percentage

    # Population stats (only over sessions that tested each diameter)
    with np.errstate(all="ignore"):
        pop_mean = np.nanmean(rate_matrix, axis=0)
        pop_n = np.sum(~np.isnan(rate_matrix), axis=0)
        pop_sem = np.nanstd(rate_matrix, axis=0, ddof=1) / np.sqrt(pop_n)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- individual session curves ---
    colors = cm.Blues(np.linspace(0.35, 0.75, n_sessions))
    for s_idx, rec in enumerate(session_records):
        sess_rates = rate_matrix[s_idx]
        valid = ~np.isnan(sess_rates)
        if valid.sum() < 2:
            continue
        label = rec.get("date", rec["session_id"])
        ax.plot(diameters[valid], sess_rates[valid],
                "o--", color=colors[s_idx], markersize=5, linewidth=1,
                alpha=0.6, label=label)

    # --- population mean ± SEM ---
    valid_pop = ~np.isnan(pop_mean)
    ax.errorbar(
        diameters[valid_pop], pop_mean[valid_pop], yerr=pop_sem[valid_pop],
        fmt="o-", color="navy", ecolor="navy",
        markersize=10, linewidth=2.5, capsize=5, capthick=2,
        label="Population mean ± SEM", zorder=5,
    )

    # --- n=X sessions annotations ---
    for d, mean_val, sem_val, n in zip(
        diameters[valid_pop], pop_mean[valid_pop], pop_sem[valid_pop], pop_n[valid_pop]
    ):
        ax.text(d, mean_val + sem_val + 3, f"n={int(n)}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="navy")

    # Reference lines
    ax.axhline(100, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(0,   color="red",   linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Target Diameter", fontsize=14, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=14, fontweight="bold")

    title = "Psychometric Curve: Success Rate vs Target Diameter"
    label_parts = [p for p in (animal_name, animal_id) if p]
    if label_parts:
        title += f"\n{label_parts[0]} — {n_sessions} session{'s' if n_sessions != 1 else ''}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_ylim(-5, 110)
    x_pad = (diameters.max() - diameters.min()) * 0.08 if len(diameters) > 1 else 0.1
    ax.set_xlim(diameters.min() - x_pad, diameters.max() + x_pad)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.7)

    plt.tight_layout()

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}psychometric_central_fixation_population.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved population plot to {results_dir / filename}")

    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Main aggregation entry point
# ---------------------------------------------------------------------------

def analyze_animal(
    animal_name: str,
    results_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> pd.DataFrame:
    """Load all fixation_feedback sessions for one animal and plot population curve.

    Sessions must have ``experiment_type: fixation_feedback`` in the manifest.

    Parameters
    ----------
    animal_name:
        Animal name as recorded in the manifest (e.g. ``"Paris"``).
    results_dir:
        Directory to save the population figure.  When ``None`` the manifest's
        ``results_root`` is used if available, otherwise the current directory.
    show_plots:
        Whether to display plots interactively.

    Returns
    -------
    pd.DataFrame
        One row per session with session-level summary stats.
    """
    session_ids = list_sessions_from_manifest(
        "fixation_feedback", animal_name=animal_name
    )

    if not session_ids:
        print(f"No fixation_feedback sessions found for animal '{animal_name}' in manifest.")
        return pd.DataFrame()

    print(f"Found {len(session_ids)} session(s) for '{animal_name}':")
    for sid in session_ids:
        print(f"  {sid}")

    # Resolve results_dir from manifest if not provided
    if results_dir is None:
        manifest_path = Path(__file__).resolve().parents[2] / "session_manifest.yml"
        try:
            with manifest_path.open() as f:
                manifest = yaml.safe_load(f) or {}
            results_root = manifest.get("results_root")
            results_dir = Path(results_root) if results_root else Path.cwd()
        except FileNotFoundError:
            results_dir = Path.cwd()

    session_records: list[dict] = []
    summary_rows: list[pd.DataFrame] = []
    animal_id = ""

    for session_id in session_ids:
        config = load_session(session_id)
        if config.folder_path is None:
            print(f"  Skipping {session_id}: no folder_path in manifest")
            continue

        if not config.folder_path.exists():
            print(f"  Skipping {session_id}: folder not found at {config.folder_path}")
            continue

        print(f"\nLoading {session_id}...")
        try:
            eot_df, target_df = load_fixation_feedback_data(config.folder_path)
        except Exception as exc:
            print(f"  Error loading {session_id}: {exc}")
            continue

        if not animal_id and config.animal_id:
            animal_id = config.animal_id

        psychometric = compute_session_psychometric(eot_df, target_df)

        # Parse date from folder name
        folder_name = config.folder_path.name
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", folder_name)
        date_str = date_match.group() if date_match else ""

        session_records.append({
            "session_id": session_id,
            "date": date_str,
            "psychometric": psychometric,
        })

        n_trials = len(eot_df)
        n_success = int((eot_df["trial_success"] == 2).sum()) if "trial_success" in eot_df.columns else 0
        summary_rows.append(pd.DataFrame({
            "session_id": [session_id],
            "animal_id": [animal_id],
            "animal_name": [animal_name],
            "session_date": [date_str],
            "n_trials": [n_trials],
            "n_success": [n_success],
            "success_rate": [n_success / n_trials if n_trials > 0 else float("nan")],
        }))

    if not session_records:
        print("No sessions could be loaded — no population plot generated.")
        return pd.DataFrame()

    # Sort by date before plotting so legend is chronological
    session_records.sort(key=lambda r: r["date"])

    plot_population_psychometric(
        session_records,
        animal_id=animal_id,
        animal_name=animal_name,
        results_dir=results_dir,
        show_plots=show_plots,
    )

    return pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Population psychometric curve for fixation-with-feedback sessions"
    )
    parser.add_argument(
        "--animal-name", required=True,
        help="Animal name as in session_manifest.yml (e.g. Paris)",
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Directory to save the population plot (default: manifest results_root)",
    )
    parser.add_argument(
        "--no-show", dest="show_plots", action="store_false", default=True,
        help="Suppress interactive plot display",
    )
    args = parser.parse_args()

    analyze_animal(
        animal_name=args.animal_name,
        results_dir=Path(args.results) if args.results else None,
        show_plots=args.show_plots,
    )
