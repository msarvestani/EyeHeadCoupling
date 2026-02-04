"""Plot eye-cursor heatmaps from multiple sessions in vertical subplots for comparison.

This script creates a single figure with 3 vertical subplots, one for each session,
to allow easy comparison of eye position density across different sessions.

This script specifically applies to 3 sessions run on Paris on 2026-01-13

Usage:   python heatmap_v_target_session.py
X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2026-01-13T13_26_58
X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2026-01-13T13_13_35
X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2026-01-13T13_00_15       
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add the repo's Python folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import functions from prosaccade_feedback_session
from analysis.prosaccade_feedback_session import (
    load_feedback_data,
    identify_and_filter_failed_trials,
    extract_trial_trajectories
)


def plot_session_heatmap(ax, trials: list[dict], title: str, show_xlabel: bool = False):
    """Plot a single heatmap on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    trials : list of dict
        List of trial data dictionaries
    title : str
        Title for this subplot
    show_xlabel : bool
        Whether to show the x-axis label (typically only for bottom subplot)
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
    bins = 50  # Number of bins in each dimension
    h, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[-1.7, 1.7], [-1, 1]])

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, extent=extent, origin='lower', cmap='hot', aspect='auto', interpolation='bilinear')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of Samples')

    # Overlay target positions
    for i, trial in enumerate(trials):
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                              edgecolor='cyan', linewidth=2, linestyle='-', alpha=0.7)
        ax.add_patch(target_circle)

    if show_xlabel:
        ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')


def plot_multi_session_heatmaps(session_folders: list[Path], animal_ids: list[str],
                                session_labels: Optional[list[str]] = None,
                                include_failed_trials: bool = False,
                                results_dir: Optional[Path] = None) -> plt.Figure:
    """Plot heatmaps for multiple sessions in vertical subplots.

    Parameters
    ----------
    session_folders : list of Path
        List of paths to session folders
    animal_ids : list of str
        List of animal IDs for each session
    session_labels : list of str, optional
        Custom labels for each session (defaults to folder names)
    include_failed_trials : bool
        Whether to include failed trials in the heatmap (default: False)
    results_dir : Path, optional
        Directory to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    n_sessions = len(session_folders)

    if session_labels is None:
        session_labels = [f.name for f in session_folders]

    # Create figure with vertical subplots
    fig, axes = plt.subplots(n_sessions, 1, figsize=(10, 6 * n_sessions))

    # Handle case where there's only one subplot (axes won't be a list)
    if n_sessions == 1:
        axes = [axes]

    # Process each session
    for idx, (folder_path, animal_id, label) in enumerate(zip(session_folders, animal_ids, session_labels)):
        print(f"\nProcessing session {idx + 1}/{n_sessions}: {folder_path}")

        # Load data
        eot_df, eye_df, target_df_all = load_feedback_data(folder_path, animal_id)

        # Filter failed trials
        target_df_successful, failed_indices, successful_indices = identify_and_filter_failed_trials(
            target_df_all, eot_df, exclude_failed=True
        )

        # Extract trial trajectories
        trials_all = extract_trial_trajectories(eot_df, eye_df, target_df_all,
                                                successful_indices=successful_indices)

        # Decide which trials to use
        if include_failed_trials:
            trials_for_plot = trials_all
            print(f"  Using ALL {len(trials_all)} trials (including {len(failed_indices)} failed)")
        else:
            trials_for_plot = [t for t in trials_all if not t.get('trial_failed', False) and t.get('has_eye_data', True)]
            print(f"  Using {len(trials_for_plot)} successful trials")

        if len(trials_for_plot) == 0:
            print(f"  Warning: No valid trials found for session {idx + 1}")
            continue

        # Plot on the corresponding axis
        show_xlabel = (idx == n_sessions - 1)  # Only show x-label on bottom subplot
        plot_session_heatmap(axes[idx], trials_for_plot, label, show_xlabel=show_xlabel)

    # Overall title
    fig.suptitle('Eye Position Density Heatmaps - Multi-Session Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = "multi_session_heatmap_comparison.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved multi-session heatmap to {results_dir / filename}")

    return fig


def main():
    """Main function to run the multi-session heatmap comparison."""
    parser = argparse.ArgumentParser(
        description="Plot eye-cursor heatmaps from multiple sessions in vertical subplots for comparison"
    )
    parser.add_argument("folders", nargs='+', help="Paths to session folders (3 or more)")
    parser.add_argument("--animal-ids", nargs='+', default=None,
                       help="Animal IDs for each session (default: Tsh001 for all)")
    parser.add_argument("--labels", nargs='+', default=None,
                       help="Custom labels for each session (default: folder names)")
    parser.add_argument("--include-failed-trials", action='store_true', default=False,
                       help="Include failed trials in the heatmap (default: False)")
    parser.add_argument("--results", type=str, help="Results directory to save the figure")
    parser.add_argument("--no-show", action='store_true', default=False,
                       help="Don't display the plot (only save)")

    args = parser.parse_args()

    # Convert folder paths to Path objects
    session_folders = [Path(f) for f in args.folders]

    # Validate that all folders exist
    for folder in session_folders:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder}")

    # Set animal IDs (default to Tsh001 for all sessions)
    if args.animal_ids is None:
        animal_ids = ["Tsh001"] * len(session_folders)
    else:
        animal_ids = args.animal_ids
        if len(animal_ids) != len(session_folders):
            raise ValueError(
                f"Number of animal IDs ({len(animal_ids)}) must match "
                f"number of folders ({len(session_folders)})"
            )

    # Set labels
    session_labels = args.labels
    if session_labels is not None and len(session_labels) != len(session_folders):
        raise ValueError(
            f"Number of labels ({len(session_labels)}) must match "
            f"number of folders ({len(session_folders)})"
        )

    # Set results directory
    results_dir = Path(args.results) if args.results else None

    # Create the plot
    fig = plot_multi_session_heatmaps(
        session_folders=session_folders,
        animal_ids=animal_ids,
        session_labels=session_labels,
        include_failed_trials=args.include_failed_trials,
        results_dir=results_dir
    )

    # Show plot unless --no-show flag is set
    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
