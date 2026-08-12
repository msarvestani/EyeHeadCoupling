"""Draw one session's eye-position density heatmap on a single axis.

Pools the eye-position samples from every trial in ``trials`` into a
2D histogram over the monitor's stimulus-unit coordinate space, then
overlays a single circle at the mean target position, sized to the
mean target diameter across those same trials.

Parameters
----------
ax : matplotlib.axes.Axes
    Axis to draw on. Modified in place; nothing is returned.
trials : list of dict
    Trial dictionaries as produced by
    ``analysis.prosaccade_feedback_session.extract_trial_trajectories``.
    Each trial must provide ``eye_x``/``eye_y`` (arrays of eye
    position samples for that trial) and
    ``target_x``/``target_y``/``target_diameter`` (that trial's
    target). May be empty — the histogram is then all zero and the
    target circle is skipped.
title : str, optional
    Subplot title (e.g. "Successful Trials"). Omit to leave the
    subplot untitled — used so only the top row of the multi-session
    grid gets a column header.
show_xlabel : bool
    Whether to draw the x-axis label ("Monitor x"). Typically set
    only for the bottom row of the grid.
show_ylabel : bool
    Whether to draw the y-axis label ("Monitor y"). Typically set for
    only one subplot (the middle row's left column) to avoid
    repeating it on every panel.

Returns
-------
None
    The heatmap, colorbar, target circle, and axis labels/limits are
    all drawn directly onto ``ax``.
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
RESULTS_DIR: Optional[Path] = Path(r"X:\Analysis\EyeHeadCoupling\Fig4S_fixation_heatmap")  # e.g. Path("results") to save the figure to disk
SHOW_PLOT = True

def plot_session_heatmap(ax, trials: list[dict], title: Optional[str] = None, show_xlabel: bool = False, show_ylabel: bool = True,
                          target_x: Optional[float] = None, target_y: Optional[float] = None, target_diameter: Optional[float] = None):
    """Draw one session's eye-position density heatmap on a single axis.

    Pools the eye-position samples from every trial in ``trials`` into a
    2D histogram over the monitor's stimulus-unit coordinate space, then
    overlays a single circle at ``(target_x, target_y)`` sized to
    ``target_diameter``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on. Modified in place; nothing is returned.
    trials : list of dict
        Trial dictionaries as produced by
        ``analysis.prosaccade_feedback_session.extract_trial_trajectories``.
        Each trial must provide ``eye_x``/``eye_y`` (arrays of eye
        position samples for that trial). May be empty — the histogram is
        then all zero.
    title : str, optional
        Subplot title (e.g. "Successful Trials"). Omit to leave the
        subplot untitled — used so only the top row of the multi-session
        grid gets a column header.
    show_xlabel : bool
        Whether to draw the x-axis label ("Monitor x"). Typically set
        only for the bottom row of the grid.
    show_ylabel : bool
        Whether to draw the y-axis label ("Monitor y"). Typically set for
        only one subplot (the middle row's left column) to avoid
        repeating it on every panel.
    target_x, target_y, target_diameter : float, optional
        Center and diameter of the single target circle to overlay.
        Passed in by the caller (see :func:`plot_multi_session_heatmaps`)
        as the mean across *all* of a session's trials, so the successful
        and failed columns for the same session show the identical
        target circle rather than each computing its own average from
        just its own trial subset. Omit any of the three to skip drawing
        the circle.

    Returns
    -------
    None
        The heatmap, colorbar, target circle, and axis labels/limits are
        all drawn directly onto ``ax``.
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

    # Overlay the shared target circle (same for both the successful and
    # failed columns of this session)
    if target_x is not None and target_y is not None and target_diameter is not None:
        target_circle = Circle((target_x, target_y), radius=target_diameter / 2.0, fill=False,
                              edgecolor='cyan', linewidth=1.0, linestyle='-', alpha=1.0)
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
    """Build the full session-by-outcome grid of fixation heatmaps.

    For each session folder, loads the feedback-task CSVs and splits
    trials into successful vs. failed (via each trial's ``trial_failed``/
    ``has_eye_data`` flags — see
    ``analysis.prosaccade_feedback_session.extract_trial_trajectories``),
    then plots each subset as its own heatmap panel (see
    :func:`plot_session_heatmap`). The resulting figure has one row per
    session and two columns: successful trials (left) and failed trials
    (right), so behavior can be compared both across sessions (down a
    column) and against itself within a session (across a row). Both
    columns of a given row overlay the *same* target circle — the mean
    target position/diameter across that session's trials, regardless of
    outcome — so any difference between the two panels reflects eye
    behavior, not a shifted reference target.

    Parameters
    ----------
    session_folders : list of Path
        Session folders to load, in top-to-bottom row order. Each is
        passed to
        ``analysis.prosaccade_feedback_session.load_feedback_data``.
    animal_ids : list of str
        Animal ID for each session, same length/order as
        ``session_folders``.
    results_dir : Path, optional
        If given, the figure is saved here as both
        ``multi_session_heatmap_comparison.png`` (raster, 150 dpi) and
        ``multi_session_heatmap_comparison.svg`` (vector, with editable
        text — see the ``svg.fonttype``/``font.family`` rcParams set at
        the top of this file). The directory is created if it doesn't
        exist. If omitted, the figure is only returned, not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure, not yet shown or closed — the caller
        (:func:`main`) is responsible for ``plt.show()``/``plt.close()``.
    """
    n_sessions = len(session_folders)
    middle_row = n_sessions // 2

    # Create figure with one row per session, two columns (successful / failed trials)
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
        trials_failed = [t for t in trials_all if t.get('trial_failed', False)]

        print(f"  {len(trials_all)} total trials: {len(trials_successful)} successful, "
              f"{len(trials_failed)} failed")

        # Shared target circle for this session — same mean position/size
        # is drawn on both the successful and failed columns
        if trials_all:
            mean_target_x = np.mean([t['target_x'] for t in trials_all])
            mean_target_y = np.mean([t['target_y'] for t in trials_all])
            mean_target_diameter = np.mean([t['target_diameter'] for t in trials_all])
        else:
            mean_target_x = mean_target_y = mean_target_diameter = None

        show_xlabel = (idx == n_sessions - 1)  # Only show x-label on bottom row
        show_ylabel = (idx == middle_row)  # Only show y-label on middle row, left column

        for col, (trials_for_plot, column_title) in enumerate([
            (trials_successful, "Successful Trials"),
            (trials_failed, "Failed Trials"),
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
                                show_ylabel=show_ylabel and col == 0,
                                target_x=mean_target_x, target_y=mean_target_y,
                                target_diameter=mean_target_diameter)

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