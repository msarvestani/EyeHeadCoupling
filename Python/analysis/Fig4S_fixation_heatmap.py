"""Plot eye-cursor heatmaps from multiple sessions in vertical subplots for comparison.

This script creates a single figure with one heatmap subplot per session
(stacked vertically), to allow easy comparison of eye position density
across sessions run on the same day.

Which sessions are plotted is controlled by the ``SESSION_FOLDERS`` constant
below rather than by command-line arguments. To plot a different set of
sessions, either edit ``SESSION_FOLDERS`` directly or uncomment one of the
alternative session sets already listed there.

Usage
-----
    python Fig4S_heatmap_v_target_session.py

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'   # keep text as text, not vector outlines
plt.rcParams['font.family'] = 'Arial'   # or another font AI/your system actually has installed

# Add the repo's Python folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import functions from prosaccade_feedback_session
from analysis.prosaccade_feedback_session import (
    load_feedback_data,
    identify_and_filter_failed_trials,
    extract_trial_trajectories
)

# ---------------------------------------------------------------------------
# Session folders to plot
# ---------------------------------------------------------------------------

# Paris, 2026-01-13
SESSION_FOLDERS = [
    r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2026-01-13T13_26_58",
    r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2026-01-13T13_00_15",
    r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2026-01-13T13_13_35",
]
# ---------------------------------------------------------------------------
# Other run settings
# ---------------------------------------------------------------------------
ANIMAL_ID = "Tsh001"
RESULTS_DIR: Optional[Path] = None  # e.g. Path("results") to save the figure to disk
SHOW_PLOT = True

def plot_session_heatmap(ax, trials: list[dict], title: Optional[str] = None, show_xlabel: bool = False, show_ylabel: bool = True):
    """Plot a single heatmap on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    trials : list of dict
        List of trial data dictionaries
    title : str, optional
        Title for this subplot (omit to leave the subplot untitled)
    show_xlabel : bool
        Whether to show the x-axis label (typically only for bottom subplot)
    show_ylabel : bool
        Whether to show the y-axis label (typically only for the middle subplot)
    """
    # Collect all eye positions from all trials
    all_x = []
    all_y = []
    for trial in trials:
        all_x.extend(trial['eye_x'])
        all_y.extend(trial['eye_y'])

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Create 2D histogram
    bins = 10  # Number of bins in each dimension
    h, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[-1.7, 1.7], [-1, 1]])

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, extent=extent, origin='lower', cmap='hot', aspect='auto', interpolation='bilinear', vmin=0, vmax=350)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of Samples')

        # Overlay target positions
    for trial in trials:
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                              edgecolor='cyan', linewidth=0.5, linestyle='-', alpha=0.1)
        ax.add_patch(target_circle)

    if show_xlabel:
        ax.set_xlabel('Monitor x', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Monitor y', fontsize=12)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')


def plot_multi_session_heatmaps(session_folders: list[Path], animal_ids: list[str],
                                results_dir: Optional[Path] = None) -> plt.Figure:
    """Plot heatmaps for multiple sessions in a grid: one row per session,
    two columns comparing all trials (left) against successful trials only
    (right).

    Parameters
    ----------
    session_folders : list of Path
        List of paths to session folders
    animal_ids : list of str
        List of animal IDs for each session
    results_dir : Path, optional
        Directory to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    n_sessions = len(session_folders)
    middle_row = n_sessions // 2

    # Create figure with one row per session, two columns (all / successful trials)
    fig, axes = plt.subplots(n_sessions, 2, figsize=(10, 3 * n_sessions))
    axes = np.atleast_2d(axes)

    # Process each session
    for idx, (folder_path, animal_id) in enumerate(zip(session_folders, animal_ids)):
        print(f"\nProcessing session {idx + 1}/{n_sessions}: {folder_path}")

        # Load data
        eot_df, eye_df, target_df_all = load_feedback_data(folder_path, animal_id)

        # Filter failed trials
        _, failed_indices, successful_indices = identify_and_filter_failed_trials(
            target_df_all, eot_df, exclude_failed=False
        )

        # Extract trial trajectories
        trials_all = extract_trial_trajectories(eot_df, eye_df, target_df_all,
                                                successful_indices=successful_indices)
        trials_successful = [t for t in trials_all if not t.get('trial_failed', False) and t.get('has_eye_data', True)]

        print(f"  {len(trials_all)} total trials ({len(failed_indices)} failed), "
              f"{len(trials_successful)} successful trials")

        show_xlabel = (idx == n_sessions - 1)  # Only show x-label on bottom row
        show_ylabel = (idx == middle_row)  # Only show y-label on middle row, left column

        for col, (trials_for_plot, column_title) in enumerate([
            (trials_all, "All Trials"),
            (trials_successful, "Successful Trials"),
        ]):
            ax = axes[idx, col]
            title = column_title if idx == 0 else None

            if len(trials_for_plot) == 0:
                print(f"  Warning: No valid trials found for session {idx + 1}, column '{column_title}'")
                if title:
                    ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
                continue

            plot_session_heatmap(ax, trials_for_plot, title=title,
                                show_xlabel=show_xlabel,
                                show_ylabel=show_ylabel and col == 0)

    # Overall title
    fig.suptitle('Eye Position Heatmaps', fontsize=12, fontweight='bold', y=0.97)

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        base_filename = "multi_session_heatmap_comparison"
        png_path = results_dir / f"{base_filename}.png"
        svg_path = results_dir / f"{base_filename}.svg"
        fig.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"\nSaved multi-session heatmap to {png_path}")
        print(f"Saved multi-session heatmap to {svg_path}")

    return fig

def main():
    """Run the multi-session heatmap comparison on the hardwired session folders.

    Edit ``SESSION_FOLDERS`` and the other constants near the top of this
    file to change which sessions are plotted and how.
    """
    session_folders = [Path(f) for f in SESSION_FOLDERS]

    # Validate that all folders exist
    for folder in session_folders:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder}")

    animal_ids = [ANIMAL_ID] * len(session_folders)

    # Create the plot
    fig = plot_multi_session_heatmaps(
        session_folders=session_folders,
        animal_ids=animal_ids,
        results_dir=RESULTS_DIR,
    )

    if SHOW_PLOT:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()