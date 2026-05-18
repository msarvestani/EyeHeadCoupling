"""Analyse a single fixation-with-feedback session (psychometric curve).

Usage
-----
Via manifest session ID::

    python Python/analysis/fixationfeedback_session.py Tsh001_2026-01-08T13_07_23

Via direct folder path::

    python Python/analysis/fixationfeedback_session.py --folder /path/to/folder --animal Tsh001
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session
from eyehead.io import clean_csv
from fixation_session import plot_psychometric_central_fixation, bonsai_to_deg
from prosaccade_feedback_session import (
    extract_trial_trajectories,
    identify_and_filter_failed_trials,
    detect_fixations,
    FIXATION_MIN_DURATION,
    FIXATION_MAX_MOVEMENT,
)


def load_fixation_feedback_data(folder_path: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Load end-of-trial, eye position, and target CSVs from a fixation feedback session folder.

    Returns
    -------
    (eot_df, eye_df, target_df)
        eot_df  – one row per trial, always has 'trial_success'; has 'diameter'
                  when recorded there.
        eye_df  – frame-by-frame eye/cursor position (green_x, green_y), or None if
                  no vstim_go file is found.
        target_df – one row per trial, always has 'diameter' and 'visible'.
    """
    csv_files = list(folder_path.glob("*.csv"))

    endoftrial_file = None
    target_file = None
    vstim_go_file = None
    for f in csv_files:
        name = f.name.lower()
        if "endoftrial" in name:
            endoftrial_file = f
        elif "vstim_go" in name:
            vstim_go_file = f
        elif "vstim_cue" in name or "target" in name:
            target_file = f

    if endoftrial_file is None:
        raise FileNotFoundError(f"Could not find endoftrial CSV in {folder_path}")

    # --- end-of-trial ---
    print(f"\nLoading {endoftrial_file.name}...")
    eot_arr = np.genfromtxt(clean_csv(str(endoftrial_file)), delimiter=",", skip_header=1, dtype=float)
    if eot_arr.ndim == 1:
        eot_arr = eot_arr.reshape(1, -1)
    n = eot_arr.shape[1]
    if n >= 7:
        eot_df = pd.DataFrame(eot_arr, columns=["frame", "timestamp", "trial_success",
                                                  "trial_number", "green_x", "green_y", "diameter"])
    elif n == 6:
        eot_df = pd.DataFrame(eot_arr, columns=["frame", "timestamp", "trial_success",
                                                  "green_x", "green_y", "diameter"])
    elif n >= 3:
        eot_df = pd.DataFrame({"frame": eot_arr[:, 0],
                                "timestamp": eot_arr[:, 1],
                                "trial_success": eot_arr[:, 2]})
    else:
        raise ValueError(f"Unexpected endoftrial column count: {n}")
    eot_df["frame"] = eot_df["frame"].astype(int)
    eot_df["trial_success"] = eot_df["trial_success"].astype(int)
    print(f"  Loaded {len(eot_df)} end-of-trial events")

    # --- eye / cursor position (vstim_go) ---
    eye_df = None
    if vstim_go_file is not None:
        print(f"\nLoading {vstim_go_file.name}...")
        eye_arr = np.genfromtxt(clean_csv(str(vstim_go_file)), delimiter=",", skip_header=1, dtype=float)
        if eye_arr.ndim == 1:
            eye_arr = eye_arr.reshape(1, -1)
        n_eye = eye_arr.shape[1]
        eye_df = pd.DataFrame({
            "frame": eye_arr[:, 0],
            "timestamp": eye_arr[:, 1],
            "green_x": eye_arr[:, 2],
            "green_y": eye_arr[:, 3],
            "diameter": eye_arr[:, 4] if n_eye >= 5 else 0.2,
        })
        eye_df["frame"] = eye_df["frame"].astype(int)
        print(f"  Loaded {len(eye_df)} eye position rows")

    # --- target (diameter source) ---
    if target_file is not None:
        print(f"\nLoading {target_file.name}...")
        t_arr = np.genfromtxt(clean_csv(str(target_file)), delimiter=",", skip_header=1, dtype=float)
        if t_arr.ndim == 1:
            t_arr = t_arr.reshape(1, -1)
        nt = t_arr.shape[1]
        col_names = ["frame", "timestamp", "target_x", "target_y", "diameter", "visible"]
        target_df = pd.DataFrame(t_arr, columns=col_names[:nt])
        target_df["diameter"] = target_df["diameter"].astype(float)
        # One row per trial – drop duplicate frames
        if "frame" in target_df.columns:
            target_df = target_df.drop_duplicates(subset=["frame"], keep="first").reset_index(drop=True)
        # Ensure visible column exists
        if "visible" not in target_df.columns:
            target_df["visible"] = 1
        print(f"  Loaded {len(target_df)} target rows")
    elif "diameter" in eot_df.columns:
        # Diameter recorded in endoftrial file – use it directly
        target_df = eot_df[["diameter"]].copy().reset_index(drop=True)
        target_df["visible"] = 1
    else:
        raise FileNotFoundError(
            f"No target CSV and no 'diameter' column in endoftrial – cannot build psychometric curve. "
            f"Folder: {folder_path}"
        )

    return eot_df, eye_df, target_df



def compute_success_trial_times(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> np.ndarray:
    """Return array of trial durations (s) for successful trials only.

    Returns empty array when required columns are unavailable.
    """
    if (
        "trial_success" not in eot_df.columns
        or "timestamp" not in eot_df.columns
        or "timestamp" not in target_df.columns
    ):
        return np.array([], dtype=float)

    n = min(len(eot_df), len(target_df))
    eot_ts = pd.to_numeric(eot_df["timestamp"], errors="coerce").to_numpy()
    tgt_ts = pd.to_numeric(target_df["timestamp"], errors="coerce").to_numpy()
    durations = eot_ts[:n] - tgt_ts[:n]

    success_mask = eot_df["trial_success"].iloc[:n].values == 2
    return durations[success_mask & ~np.isnan(durations)]


def compute_trial_times_by_diameter(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> dict[float, tuple[float, float, int]]:
    """Return {diameter_deg: (mean_s, sd_s, n)} for successful trials grouped by diameter.

    Returns empty dict when timestamps or diameter data are unavailable.
    """
    if (
        "trial_success" not in eot_df.columns
        or "timestamp" not in eot_df.columns
        or "timestamp" not in target_df.columns
        or "diameter" not in target_df.columns
    ):
        return {}

    n = min(len(eot_df), len(target_df))
    eot_ts = pd.to_numeric(eot_df["timestamp"], errors="coerce").to_numpy()
    tgt_ts = pd.to_numeric(target_df["timestamp"], errors="coerce").to_numpy()
    durations = eot_ts[:n] - tgt_ts[:n]

    combined = pd.DataFrame({
        "trial_success": eot_df["trial_success"].iloc[:n].values,
        "diameter": target_df["diameter"].iloc[:n].values,
        "duration": durations,
    })
    success = combined[combined["trial_success"] == 2]

    result: dict[float, tuple[float, float, int]] = {}
    for diam, group in success.groupby("diameter"):
        diam_key = round(float(bonsai_to_deg(diam)), 3)
        times = group["duration"].dropna().values
        if len(times) == 0:
            continue
        result[diam_key] = (
            float(np.mean(times)),
            float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            len(times),
        )
    return result


def plot_trial_time_session(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
    results_dir: Path,
    animal_id: str = "",
    date_str: str = "",
    show_plots: bool = True,
) -> Optional[plt.Figure]:
    """Bar chart of mean successful trial time per target diameter for one session."""
    times_by_diam = compute_trial_times_by_diameter(eot_df, target_df)
    if not times_by_diam:
        return None

    diameters = sorted(times_by_diam.keys())
    means = [times_by_diam[d][0] for d in diameters]
    sds = [times_by_diam[d][1] for d in diameters]
    ns = [times_by_diam[d][2] for d in diameters]

    fig, ax = plt.subplots(figsize=(max(5, len(diameters) * 1.2), 4))
    ax.bar(range(len(diameters)), means, yerr=sds, color="steelblue",
           edgecolor="white", alpha=0.8, capsize=4)
    ax.set_xticks(range(len(diameters)))
    ax.set_xticklabels([f"{d:.1f}°\n(n={n})" for d, n in zip(diameters, ns)], fontsize=9)
    ax.set_xlabel("Target diameter (°)")
    ax.set_ylabel("Mean trial time (s)")
    title = "Successful trial time by target diameter"
    if animal_id:
        title += f" – {animal_id}"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    fig.tight_layout()

    prefix = f"{animal_id}_" if animal_id else ""
    date_tag = f"_{date_str}" if date_str else ""
    for ext in ("png", "svg"):
        fig.savefig(results_dir / f"{prefix}trial_time_by_diameter{date_tag}.{ext}", bbox_inches="tight")

    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def compute_fixation_variance_by_diameter(trials: list[dict]) -> dict[float, float]:
    """Return {diameter: var_x} of fixation centerpoints, keyed by target diameter."""
    from collections import defaultdict
    centerpoints_by_diam: dict[float, list[float]] = defaultdict(list)
    for trial in trials:
        if not trial.get('has_eye_data', False):
            continue
        diameter = trial.get('target_diameter')
        if diameter is None:
            continue
        eye_x = np.array(trial['eye_x'])
        eye_y = np.array(trial['eye_y'])
        eye_times = np.array(trial.get('eye_times', np.arange(len(eye_x))))
        fixations = detect_fixations(eye_x, eye_y, eye_times, FIXATION_MIN_DURATION, FIXATION_MAX_MOVEMENT)
        if fixations:
            start, end, *_ = fixations[-1]
            centerpoints_by_diam[diameter].append(float(np.mean(eye_x[start:end])))
    return {
        diam: float(np.var(xs)) if len(xs) > 1 else float('nan')
        for diam, xs in centerpoints_by_diam.items()
    }


def plot_trajectories_by_diameter(
    trials: list[dict],
    results_dir: Optional[Path] = None,
    animal_id: Optional[str] = None,
    session_date: str = "",
    session_time: Optional[str] = None,
    min_fixation_duration: float = FIXATION_MIN_DURATION,
    max_fixation_movement: float = FIXATION_MAX_MOVEMENT,
    show_plots: bool = True,
) -> plt.Figure:
    """Plot fixation points grouped by target diameter.

    Creates a figure with subplots, one for each unique target diameter.
    Each subplot shows fixation points with successful trials in green and
    failed trials in red.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries containing eye trajectories.
    results_dir : Path, optional
        Directory to save the figure.
    animal_id : str, optional
        Animal identifier for filename.
    session_date : str, optional
        Session date for title (format: YYYY-MM-DD).
    session_time : str, optional
        Session time for title (format: HH:MM).
    min_fixation_duration : float
        Minimum fixation duration in seconds.
    max_fixation_movement : float
        Maximum movement threshold for fixation.
    show_plots : bool
        Whether to call plt.show() on both figures.

    Returns
    -------
    matplotlib.figure.Figure
        The generated scatter figure.
    """
    # Group trials by diameter
    diameter_trials = {}
    for trial in trials:
        if not trial.get('has_eye_data', False):
            continue
        diameter = trial.get('target_diameter', None)
        if diameter is None:
            continue
        if diameter not in diameter_trials:
            diameter_trials[diameter] = []
        diameter_trials[diameter].append(trial)

    if len(diameter_trials) == 0:
        print("Warning: No trials with eye data and diameter information found")
        return None

    # Sort diameters for consistent plotting
    sorted_diameters = sorted(diameter_trials.keys())
    n_diameters = len(sorted_diameters)

    # Create subplots arranged in a grid
    n_cols = min(3, n_diameters)
    n_rows = int(np.ceil(n_diameters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Make axes always iterable
    if n_diameters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Collect variance data for separate plot
    variance_data = {'diameters': [], 'variances': [], 'percent_correct': [], 'iti_variances': []}

    # Plot fixations for each diameter
    for idx, diameter in enumerate(sorted_diameters):
        ax = axes[idx]
        trial_list = diameter_trials[diameter]

        n_success = 0
        n_failed = 0
        target_x = None
        target_y = None

        # Store all centerpoints for variance calculation
        all_centerpoints = []
        iti_centerpoints = []

        # Collect fixation points from all trials
        for trial in trial_list:
            eye_x = np.array(trial['eye_x'])
            eye_y = np.array(trial['eye_y'])
            eye_times = np.array(trial.get('eye_times', np.arange(len(eye_x))))

            if target_x is None:
                target_x = trial['target_x']
                target_y = trial['target_y']

            is_failed = trial.get('trial_failed', False)

            if is_failed:
                n_failed += 1
                color = 'red'
                marker = 'x'
            else:
                n_success += 1
                color = 'green'
                marker = 'o'

            # Detect fixations
            fixations = detect_fixations(eye_x, eye_y, eye_times,
                                         min_fixation_duration, max_fixation_movement)

            # Plot only the last fixation
            if len(fixations) > 0:
                start, end, duration, span = fixations[-1]
                fix_x = eye_x[start:end]
                fix_y = eye_y[start:end]
                ax.plot(fix_x, fix_y, marker, color=color, markersize=6, alpha=0.8)

                # Calculate centerpoint of this fixation
                centerpoint_x = np.mean(fix_x)
                centerpoint_y = np.mean(fix_y)

                # Store centerpoint
                all_centerpoints.append([centerpoint_x, centerpoint_y])

            # # Plot inter-trial fixations in purple (keep comment, no block)

        # Calculate variance of all centerpoints using var_x only
        if len(all_centerpoints) > 0:
            all_centerpoints_arr = np.array(all_centerpoints)
            var_x = np.var(all_centerpoints_arr[:, 0])
            centerpoint_var = var_x
        else:
            centerpoint_var = np.nan

        # Calculate variance of inter-trial centerpoints
        if len(iti_centerpoints) > 0:
            iti_centerpoints_arr = np.array(iti_centerpoints)
            iti_var_x = np.var(iti_centerpoints_arr[:, 0])
            iti_var = iti_var_x
        else:
            iti_var = np.nan

        # Store data for variance plot
        variance_data['diameters'].append(diameter)
        variance_data['variances'].append(centerpoint_var)
        variance_data['iti_variances'].append(iti_var)
        pct_correct = (n_success / (n_success + n_failed)) * 100 if (n_success + n_failed) > 0 else 0
        variance_data['percent_correct'].append(pct_correct)

        # Draw target circle
        circle = Circle((target_x, target_y), diameter / 2,
                         fill=False, edgecolor='blue', linewidth=2.5, linestyle='--')
        ax.add_patch(circle)

        # Mark target center
        ax.plot(target_x, target_y, 'b+', markersize=15, markeredgewidth=3)

        # Set axis limits
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

        # Labels and title
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)

        # Build title WITHOUT variance information
        title_text = f'Diameter: {diameter:.3f}\n(n={n_success + n_failed}: {n_success} success, {n_failed} failed)'

        ax.set_title(title_text, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add legend to first subplot only
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=8, label='Success fixations'),
                Line2D([0], [0], marker='x', color='red', markersize=8,
                       linewidth=2, label='Failed fixations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                       markersize=8, label='Inter-trial fixations'),
                Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Target'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Hide extra subplots if we have more subplots than diameters
    for idx in range(n_diameters, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    title = 'Fixation Points by Target Diameter'
    if animal_id:
        title += f'\n{animal_id}'
    if session_date:
        title += f' - {session_date}'
        if session_time:
            title += f' @ {session_time}'
    elif session_time:
        title += f' - {session_time}'

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        date_suffix = f"_{session_date}" if session_date else ""
        filename = f"{prefix}fixations_by_diameter{date_suffix}.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"Saved fixations by diameter to {results_dir / filename}")

    if show_plots:
        plt.show()
    plt.close(fig)

    # Create variance vs diameter plot
    fig_var, ax_var = plt.subplots(figsize=(10, 7))

    # Plot trial fixation variance (solid blue line)
    ax_var.plot(variance_data['diameters'], variance_data['variances'], 'o-',
                linewidth=2, markersize=10, color='steelblue',
                markerfacecolor='lightblue', markeredgecolor='steelblue',
                markeredgewidth=2, label='Trial fixations')

    # Plot inter-trial fixation variance (dashed purple line)
    valid_iti = [(d, v) for d, v in zip(variance_data['diameters'], variance_data['iti_variances']) if not np.isnan(v)]
    if valid_iti:
        iti_diameters, iti_variances = zip(*valid_iti)
        ax_var.plot(iti_diameters, iti_variances, 'o--',
                    linewidth=2, markersize=10, color='purple',
                    markerfacecolor='lavender', markeredgecolor='purple',
                    markeredgewidth=2, label='Inter-trial fixations')

    # Add % correct labels
    for d, v, pct in zip(variance_data['diameters'], variance_data['variances'], variance_data['percent_correct']):
        if not np.isnan(v):
            ax_var.annotate(f'{pct:.1f}%', xy=(d, v), xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax_var.set_xlabel('Target Diameter', fontsize=12, fontweight='bold')
    ax_var.set_ylabel('Fixation Centerpoint X Variance (Var(X))', fontsize=12, fontweight='bold')
    var_title = 'Fixation Centerpoint Variance vs Target Diameter'
    if animal_id:
        var_title += f'\n{animal_id}'
    if session_date:
        var_title += f' - {session_date}'
        if session_time:
            var_title += f' @ {session_time}'
    elif session_time:
        var_title += f' - {session_time}'
    ax_var.set_title(var_title, fontsize=14, fontweight='bold')
    ax_var.grid(True, alpha=0.3)
    ax_var.legend(loc='best', fontsize=11)

    if results_dir:
        prefix = f"{animal_id}_" if animal_id else ""
        date_suffix = f"_{session_date}" if session_date else ""
        filename_var = f"{prefix}fixation_variance_by_diameter{date_suffix}.png"
        fig_var.savefig(results_dir / filename_var, dpi=150, bbox_inches='tight')
        print(f"Saved fixation variance by diameter to {results_dir / filename_var}")

    if show_plots:
        plt.show()
    plt.close(fig_var)

    return fig


def analyze_session(
    folder_path: str | Path,
    results_dir: Optional[str | Path] = None,
    animal_id: str = "",
    show_plots: bool = True,
) -> pd.DataFrame:
    """Run fixation feedback analysis for one session folder.

    Returns
    -------
    pd.DataFrame
        One-row summary with trial counts and overall success rate.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if results_dir is None:
        results_dir = folder_path / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {folder_path}")
    print(f"Results:   {results_dir}")

    folder_name = folder_path.name

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", folder_name)
    date_str = date_match.group() if date_match else ""

    session_time = None
    time_match = re.search(r"\d{4}-\d{2}-\d{2}T(\d{2})_(\d{2})_\d{2}", folder_name)
    if time_match:
        session_time = f"{int(time_match.group(1)):02d}:{int(time_match.group(2)):02d}"

    if not animal_id:
        m = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}", folder_name)
        if m:
            animal_id = m.group(1)

    eot_df, eye_df, target_df = load_fixation_feedback_data(folder_path)

    print("\nGenerating psychometric curve...")
    fig = plot_psychometric_central_fixation(
        eot_df, target_df, results_dir, animal_id, date_str, session_time
    )
    if fig is not None:
        if show_plots:
            plt.show()
        plt.close(fig)

    print("\nGenerating trial time by diameter plot...")
    plot_trial_time_session(
        eot_df, target_df, results_dir, animal_id, date_str, show_plots=show_plots
    )

    if eye_df is not None:
        print("\nGenerating trajectory plots by diameter...")
        successful_indices = identify_and_filter_failed_trials(target_df, eot_df, exclude_failed=False)
        trials = extract_trial_trajectories(eot_df, eye_df, target_df, successful_indices)
        plot_trajectories_by_diameter(
            trials, results_dir, animal_id, date_str, session_time, show_plots=show_plots
        )

    return pd.DataFrame({
        "session_id": [folder_name],
        "animal_id": [animal_id],
        "session_date": [date_str],
        "session_time": [session_time],
    })


def main(session_id: str, show_plots: bool = True) -> pd.DataFrame:
    """Run analysis for a manifest session ID."""
    config = load_session(session_id)
    if config.folder_path is None:
        raise ValueError(f"Session '{session_id}' has no folder_path in manifest")
    return analyze_session(
        config.folder_path,
        results_dir=config.results_dir,
        animal_id=config.animal_id or "",
        show_plots=show_plots,
    )


def plot_population(session_dfs: list[pd.DataFrame]) -> plt.Figure:
    """Plot overall success rate across sessions.

    Parameters
    ----------
    session_dfs:
        List of DataFrames returned by :func:`analyze_session` or :func:`main`,
        one per session.

    Returns
    -------
    matplotlib.figure.Figure
    """
    combined = pd.concat(session_dfs, ignore_index=True)
    combined = combined.sort_values(["animal_id", "session_date", "session_time"])

    fig, ax = plt.subplots(figsize=(max(8, len(combined) * 0.8), 5))
    labels = combined.apply(
        lambda r: f"{r['animal_id']}\n{r['session_date']}" if r["session_date"]
        else str(r["session_id"]), axis=1
    )
    ax.bar(range(len(combined)), combined["success_rate"] * 100, color="steelblue", edgecolor="navy")
    ax.set_xticks(range(len(combined)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Overall Success Rate (%)", fontsize=13, fontweight="bold")
    ax.set_title("Fixation Feedback – Population Summary", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse a fixation-with-feedback session (psychometric curve)"
    )
    parser.add_argument("session_id", nargs="?",
                        help="Session identifier from session_manifest.yml")
    parser.add_argument("--folder", type=str,
                        help="Direct path to data folder (alternative to session_id)")
    parser.add_argument("--animal", type=str, default="",
                        help="Animal ID (used in --folder mode if not parseable from folder name)")
    parser.add_argument("--results", type=str,
                        help="Results directory (for --folder mode)")
    parser.add_argument("--no-show", dest="show_plots", action="store_false", default=True,
                        help="Suppress interactive plot display")
    args = parser.parse_args()

    if args.folder:
        analyze_session(args.folder, args.results, args.animal, show_plots=args.show_plots)
    elif args.session_id:
        main(args.session_id, show_plots=args.show_plots)
    else:
        parser.print_help()
