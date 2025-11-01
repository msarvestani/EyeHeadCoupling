"""Analysis script for saccade feedback task with visual feedback.

This script analyzes data from a saccade task where:
- The animal's eye movements map to a green dot on the monitor
- A blue target dot appears at some location
- The animal is rewarded when the green dot (eye position) touches the blue dot (target)
- After a delay, a new trial starts

The script produces:
1. Trajectory plots showing eye position paths relative to target position
2. Time-to-target analysis showing trial durations
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Put the repo's "Python" folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session
from eyehead.io import clean_csv


def load_feedback_data(folder_path: Path, animal_id: str = "Tsh001") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV files for saccade feedback analysis.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing the CSV files
    animal_id : str
        Animal identifier prefix for the files (default: "Tsh001")

    Returns
    -------
    tuple of (end_of_trial_df, eye_position_df, target_position_df)
        DataFrames containing the loaded and cleaned data
    """
    # Find files matching the pattern
    csv_files = list(folder_path.glob("*.csv"))

    # Find endoftrial file
    endoftrial_file = None
    vstim_go_file = None
    vstim_cue_file = None

    for f in csv_files:
        fname = f.name.lower()
        if "endoftrial" in fname:
            endoftrial_file = f
        elif "vstim_go" in fname:
            vstim_go_file = f
        elif "vstim_cue" in fname:
            vstim_cue_file = f

    if endoftrial_file is None:
        raise FileNotFoundError(f"Could not find endoftrial file in {folder_path}")
    if vstim_go_file is None:
        raise FileNotFoundError(f"Could not find vstim_go file in {folder_path}")
    if vstim_cue_file is None:
        raise FileNotFoundError(f"Could not find vstim_cue file in {folder_path}")

    # Load end of trial data using standard approach
    # Columns: Frame, timestamp, trial_number, green_dot_x, green_dot_y, diameter
    try:
        print(f"\nLoading {endoftrial_file.name}...")
        cleaned = clean_csv(str(endoftrial_file))
        eot_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1)

        eot_df = pd.DataFrame(eot_arr, columns=['frame', 'timestamp', 'trial_number', 'green_x', 'green_y', 'diameter'])
        eot_df['frame'] = eot_df['frame'].astype(int)
        eot_df['trial_number'] = eot_df['trial_number'].astype(int)

        print(f"  Loaded {len(eot_df)} end-of-trial events")
    except Exception as e:
        raise ValueError(f"Error loading end of trial file {endoftrial_file}: {e}")

    # Load eye position / green dot position data using standard approach
    # Columns: Frame, timestamp, placeholder, green_dot_x, green_dot_y, diameter
    # Note: This file has duplicates that need to be cleaned
    try:
        print(f"\nLoading {vstim_go_file.name}...")
        cleaned = clean_csv(str(vstim_go_file))
        eye_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1)

        eye_df = pd.DataFrame(eye_arr, columns=['frame', 'timestamp', 'placeholder', 'green_x', 'green_y', 'diameter'])
        eye_df['frame'] = eye_df['frame'].astype(int)

        # Remove duplicate frame entries - keep only the first occurrence
        before_dedup = len(eye_df)
        eye_df = eye_df.drop_duplicates(subset=['frame'], keep='first')
        after_dedup = len(eye_df)
        if before_dedup != after_dedup:
            print(f"  Removed {before_dedup - after_dedup} duplicate frame entries")

        eye_df = eye_df.sort_values('frame').reset_index(drop=True)
        print(f"  Loaded {len(eye_df)} eye position samples (after deduplication)")
    except Exception as e:
        raise ValueError(f"Error loading vstim_go file {vstim_go_file}: {e}")

    # Load target position / blue dot position data using standard approach
    # Columns: Frame, timestamp, target_x, target_y, diameter
    try:
        print(f"\nLoading {vstim_cue_file.name}...")
        cleaned = clean_csv(str(vstim_cue_file))
        target_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1)

        target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y', 'diameter'])
        target_df['frame'] = target_df['frame'].astype(int)

        print(f"  Loaded {len(target_df)} target position samples")
    except Exception as e:
        raise ValueError(f"Error loading vstim_cue file {vstim_cue_file}: {e}")

    print(f"\nData loaded successfully!")
    print(f"  Frame range: {eye_df['frame'].min()} to {eye_df['frame'].max()}")
    print(f"  Timestamp range: {eot_df['timestamp'].min():.2f} to {eot_df['timestamp'].max():.2f}")
    print(f"  First target at frame {target_df.iloc[0]['frame']}: ({target_df.iloc[0]['target_x']:.1f}, {target_df.iloc[0]['target_y']:.1f})")
    print(f"  First trial ends at frame {eot_df.iloc[0]['frame']}, timestamp {eot_df.iloc[0]['timestamp']:.2f}")
    if len(eot_df) > 1:
        duration_example = eot_df.iloc[1]['timestamp'] - eot_df.iloc[0]['timestamp']
        print(f"  Example: Trial 1 to Trial 2 timestamp diff = {duration_example:.2f} (should be in seconds)")

    return eot_df, eye_df, target_df


def extract_trial_trajectories(eot_df: pd.DataFrame, eye_df: pd.DataFrame,
                                target_df: pd.DataFrame) -> list[dict]:
    """Extract eye position trajectories for each trial.

    Parameters
    ----------
    eot_df : pd.DataFrame
        End of trial data
    eye_df : pd.DataFrame
        Eye position data (cleaned, no duplicates)
    target_df : pd.DataFrame
        Target position data

    Returns
    -------
    list of dict
        List of trial dictionaries containing trajectory and metadata
    """
    trials = []
    n_trials = len(eot_df)

    for i in range(n_trials):
        trial_num = eot_df.iloc[i]['trial_number']
        end_frame = eot_df.iloc[i]['frame']
        end_time = eot_df.iloc[i]['timestamp']

        # Get start frame - either from previous trial end or from beginning
        if i > 0:
            start_frame = eot_df.iloc[i-1]['frame']
            start_time = eot_df.iloc[i-1]['timestamp']
        else:
            start_frame = 0
            start_time = eye_df.iloc[0]['timestamp'] if len(eye_df) > 0 else 0

        # Find target position for this trial
        # Target should appear at or just after the start of the trial
        target_mask = (target_df['frame'] >= start_frame) & (target_df['frame'] <= end_frame)
        target_samples = target_df[target_mask]

        if len(target_samples) > 0:
            # Use the first target position for this trial
            target_x = target_samples.iloc[0]['target_x']
            target_y = target_samples.iloc[0]['target_y']
            target_diameter = target_samples.iloc[0]['diameter']
        else:
            # If no target found, skip this trial
            print(f"Warning: No target found for trial {trial_num}, skipping")
            continue

        # Extract eye position trajectory for this trial
        eye_mask = (eye_df['frame'] > start_frame) & (eye_df['frame'] <= end_frame)
        eye_trajectory = eye_df[eye_mask]

        if len(eye_trajectory) == 0:
            print(f"Warning: No eye data for trial {trial_num}, skipping")
            continue

        # Drop any rows with NA values in position data
        eye_trajectory = eye_trajectory.dropna(subset=['green_x', 'green_y', 'timestamp'])

        if len(eye_trajectory) == 0:
            print(f"Warning: No valid eye position data for trial {trial_num}, skipping")
            continue

        # Calculate path length (cumulative distance along trajectory)
        start_eye_x = eye_trajectory['green_x'].values[0]
        start_eye_y = eye_trajectory['green_y'].values[0]

        if len(eye_trajectory) > 1:
            dx = np.diff(eye_trajectory['green_x'].values)
            dy = np.diff(eye_trajectory['green_y'].values)
            segment_lengths = np.sqrt(dx**2 + dy**2)
            path_length = np.sum(segment_lengths)

            # Calculate straight-line distance from start to target
            straight_line_distance = np.sqrt((target_x - start_eye_x)**2 + (target_y - start_eye_y)**2)

            # Path efficiency: ratio of straight-line to actual path (closer to 1.0 is more efficient)
            if path_length > 0:
                path_efficiency = straight_line_distance / path_length
            else:
                path_efficiency = 0.0

            # Initial movement direction: angle between initial movement and ideal vector
            # Use first 5 samples (or fewer if trial is short) to determine initial direction
            n_samples = min(5, len(eye_trajectory))
            initial_dx = eye_trajectory['green_x'].values[n_samples-1] - start_eye_x
            initial_dy = eye_trajectory['green_y'].values[n_samples-1] - start_eye_y

            # Ideal vector from start to target
            ideal_dx = target_x - start_eye_x
            ideal_dy = target_y - start_eye_y

            # Calculate angle between vectors using dot product
            # angle = arccos(dot(v1, v2) / (|v1| * |v2|))
            initial_mag = np.sqrt(initial_dx**2 + initial_dy**2)
            ideal_mag = np.sqrt(ideal_dx**2 + ideal_dy**2)

            if initial_mag > 0 and ideal_mag > 0:
                cos_angle = (initial_dx * ideal_dx + initial_dy * ideal_dy) / (initial_mag * ideal_mag)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                initial_direction_error = np.degrees(np.arccos(cos_angle))
            else:
                initial_direction_error = np.nan
        else:
            path_length = 0.0
            path_efficiency = 0.0
            straight_line_distance = 0.0
            initial_direction_error = np.nan

        trial_data = {
            'trial_number': trial_num,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'target_x': target_x,
            'target_y': target_y,
            'target_diameter': target_diameter,
            'start_eye_x': start_eye_x,
            'start_eye_y': start_eye_y,
            'eye_x': eye_trajectory['green_x'].values,
            'eye_y': eye_trajectory['green_y'].values,
            'eye_times': eye_trajectory['timestamp'].values,
            'path_length': path_length,
            'straight_line_distance': straight_line_distance,
            'path_efficiency': path_efficiency,
            'initial_direction_error': initial_direction_error,
        }

        trials.append(trial_data)

    print(f"\nExtracted {len(trials)} valid trials out of {n_trials} total")
    if len(trials) > 0:
        print(f"  First trial duration: {trials[0]['duration']:.2f} (units: check if seconds or frames)")
        if len(trials) > 1:
            print(f"  Second trial duration: {trials[1]['duration']:.2f}")
        print(f"  Mean trial duration: {np.mean([t['duration'] for t in trials]):.2f}")

        print(f"\n  Starting eye positions:")
        for i, trial in enumerate(trials[:5]):  # Show first 5 trials
            print(f"    Trial {trial['trial_number']}: ({trial['start_eye_x']:.3f}, {trial['start_eye_y']:.3f})")
        if len(trials) > 5:
            print(f"    ... (showing first 5 of {len(trials)} trials)")

        path_lengths = [t['path_length'] for t in trials]
        print(f"\n  Path length statistics:")
        print(f"    Mean: {np.mean(path_lengths):.3f}")
        print(f"    Median: {np.median(path_lengths):.3f}")
        print(f"    Range: {np.min(path_lengths):.3f} - {np.max(path_lengths):.3f}")

        efficiencies = [t['path_efficiency'] for t in trials]
        print(f"\n  Path efficiency statistics (1.0 = perfectly direct):")
        print(f"    Mean: {np.mean(efficiencies):.3f}")
        print(f"    Median: {np.median(efficiencies):.3f}")

        dir_errors = [t['initial_direction_error'] for t in trials if not np.isnan(t['initial_direction_error'])]
        if dir_errors:
            print(f"\n  Initial direction error (degrees from ideal):")
            print(f"    Mean: {np.mean(dir_errors):.1f}°")
            print(f"    Median: {np.median(dir_errors):.1f}°")

    return trials


def plot_trajectories(trials: list[dict], results_dir: Optional[Path] = None,
                      animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot eye position trajectories and target positions in absolute coordinates.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color map for trials - use coolwarm to show temporal progression (blue=early, red=late)
    cmap = plt.cm.coolwarm
    n_trials = len(trials)

    # Plot each trial
    for i, trial in enumerate(trials):
        # Use absolute eye positions
        eye_x = trial['eye_x']
        eye_y = trial['eye_y']

        # Plot trajectory as scatter points to visualize density
        color = cmap(i / max(1, n_trials - 1))
        ax.scatter(eye_x, eye_y, alpha=0.5, s=10, color=color,
                   label=f"Trial {trial['trial_number']}" if n_trials <= 20 else None)

        # Mark start and end points with different markers
        ax.plot(eye_x[0], eye_y[0], 'o', color=color, markersize=8, alpha=0.9,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot(eye_x[-1], eye_y[-1], 's', color=color, markersize=8, alpha=0.9,
                markeredgecolor='white', markeredgewidth=1)

        # Draw target position as black circle at actual position with actual diameter
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                              edgecolor='black', linewidth=2.5, linestyle='-',
                              label='Target' if i == 0 else None)
        ax.add_patch(target_circle)

        # Add smaller filled circle at target center
        ax.plot(target_x, target_y, 'ko', markersize=4,
               label='Target Center' if i == 0 else None)

    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)

    title = 'Eye Position Trajectories to Target'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)

    # Set axis limits to -1 to 1 for both axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # Use set_aspect instead of axis('equal') to preserve the limits
    ax.set_aspect('equal', adjustable='box')

    # Add legend if not too many trials
    if n_trials <= 20:
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    else:
        # Just show a colorbar for trial number
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_trials))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Trial Number')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_trajectories.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_trajectories.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved trajectory plot to {results_dir / filename}")

    return fig


def plot_density_heatmap(trials: list[dict], results_dir: Optional[Path] = None,
                         animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot 2D histogram heatmap showing density of eye positions across all trials.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

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
    h, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[-1, 1], [-1, 1]])

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

    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)

    title = 'Eye Position Density Heatmap'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_heatmap.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_heatmap.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved heatmap plot to {results_dir / filename}")

    return fig


def plot_time_to_target(trials: list[dict], results_dir: Optional[Path] = None,
                        animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot time from trial onset to trial end (time to reach target).

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    trial_numbers = [t['trial_number'] for t in trials]
    durations = [t['duration'] for t in trials]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Duration vs trial number
    ax1.plot(trial_numbers, durations, 'o-', linewidth=2, markersize=8,
            color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue',
            markeredgewidth=1.5)
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Time to Target (seconds)', fontsize=12)

    title = 'Time to Reach Target Across Trials'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add mean line
    mean_duration = np.mean(durations)
    ax1.axhline(mean_duration, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_duration:.2f}s')
    ax1.legend(fontsize=10)

    # Plot 2: Histogram of durations
    ax2.hist(durations, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Time to Target (seconds)', fontsize=12)
    ax2.set_ylabel('Number of Trials', fontsize=12)
    ax2.set_title('Distribution of Trial Durations', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    std_duration = np.std(durations)
    median_duration = np.median(durations)
    stats_text = f'Mean: {mean_duration:.2f}s\nMedian: {median_duration:.2f}s\nStd: {std_duration:.2f}s\nN: {len(durations)}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_time_to_target.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_time_to_target.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved time-to-target plot to {results_dir / filename}")

    return fig


def plot_path_length(trials: list[dict], results_dir: Optional[Path] = None,
                     animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot trajectory path length by trial.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    trial_numbers = [t['trial_number'] for t in trials]
    path_lengths = [t['path_length'] for t in trials]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Path length vs trial number
    ax1.plot(trial_numbers, path_lengths, 'o-', linewidth=2, markersize=8,
            color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue',
            markeredgewidth=1.5)
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Path Length (stimulus units)', fontsize=12)

    title = 'Trajectory Path Length Across Trials'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add mean line
    mean_path = np.mean(path_lengths)
    ax1.axhline(mean_path, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_path:.3f}')
    ax1.legend(fontsize=10)

    # Plot 2: Histogram of path lengths
    ax2.hist(path_lengths, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Path Length (stimulus units)', fontsize=12)
    ax2.set_ylabel('Number of Trials', fontsize=12)
    ax2.set_title('Distribution of Path Lengths', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    std_path = np.std(path_lengths)
    median_path = np.median(path_lengths)
    stats_text = f'Mean: {mean_path:.3f}\nMedian: {median_path:.3f}\nStd: {std_path:.3f}\nN: {len(path_lengths)}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_path_length.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_path_length.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved path length plot to {results_dir / filename}")

    return fig


def plot_learning_metrics(trials: list[dict], results_dir: Optional[Path] = None,
                          animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot learning metrics: path efficiency and initial direction error across trials.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    trial_numbers = [t['trial_number'] for t in trials]
    efficiencies = [t['path_efficiency'] for t in trials]
    dir_errors = [t['initial_direction_error'] for t in trials]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Color trials by progression (blue=early, red=late)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(trials)))

    # Plot 1: Path Efficiency (higher = more direct, better learning)
    ax1.scatter(trial_numbers, efficiencies, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Path Efficiency (straight-line / actual path)', fontsize=12)

    title = 'Path Efficiency Across Trials'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (1.0)')

    # Add trend line
    z = np.polyfit(trial_numbers, efficiencies, 1)
    p = np.poly1d(z)
    ax1.plot(trial_numbers, p(trial_numbers), "k--", linewidth=2, alpha=0.7, label=f'Trend: {z[0]:+.4f}/trial')
    ax1.legend(fontsize=10)

    # Add mean line
    mean_eff = np.mean(efficiencies)
    ax1.axhline(mean_eff, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Plot 2: Initial Direction Error (lower = better aiming, better learning)
    valid_indices = [i for i, err in enumerate(dir_errors) if not np.isnan(err)]
    valid_trial_nums = [trial_numbers[i] for i in valid_indices]
    valid_errors = [dir_errors[i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]

    ax2.scatter(valid_trial_nums, valid_errors, c=valid_colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Initial Direction Error (degrees)', fontsize=12)
    ax2.set_title('Initial Movement Direction Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (0°)')

    # Add trend line
    if len(valid_trial_nums) > 1:
        z2 = np.polyfit(valid_trial_nums, valid_errors, 1)
        p2 = np.poly1d(z2)
        ax2.plot(valid_trial_nums, p2(valid_trial_nums), "k--", linewidth=2, alpha=0.7, label=f'Trend: {z2[0]:+.3f}°/trial')
        ax2.legend(fontsize=10)

    # Add mean line
    mean_err = np.mean(valid_errors)
    ax2.axhline(mean_err, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_learning_metrics.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_learning_metrics.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved learning metrics plot to {results_dir / filename}")

    return fig


def plot_spatial_bias_by_phase(trials: list[dict], results_dir: Optional[Path] = None,
                                animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot spatial heatmaps split by trial phase (early/middle/late).

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    n_trials = len(trials)
    # Split into thirds
    early_end = n_trials // 3
    middle_end = 2 * n_trials // 3

    early_trials = trials[:early_end]
    middle_trials = trials[early_end:middle_end]
    late_trials = trials[middle_end:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    bins = 50
    vmax = None  # Will be set to max across all phases for consistent colorbar

    # Function to collect all positions from a trial subset
    def collect_positions(trial_list):
        all_x = []
        all_y = []
        for trial in trial_list:
            all_x.extend(trial['eye_x'])
            all_y.extend(trial['eye_y'])
        return np.array(all_x), np.array(all_y)

    # First pass: determine vmax for consistent color scale
    all_counts = []
    for trial_subset in [early_trials, middle_trials, late_trials]:
        if len(trial_subset) > 0:
            x, y = collect_positions(trial_subset)
            h, _, _ = np.histogram2d(x, y, bins=bins, range=[[-1, 1], [-1, 1]])
            all_counts.append(h.max())
    vmax = max(all_counts) if all_counts else 1

    # Plot each phase
    phases = [
        (early_trials, 'Early Trials', f'Trials 1-{early_end}'),
        (middle_trials, 'Middle Trials', f'Trials {early_end+1}-{middle_end}'),
        (late_trials, 'Late Trials', f'Trials {middle_end+1}-{n_trials}')
    ]

    for ax, (trial_subset, phase_name, trial_range) in zip(axes, phases):
        if len(trial_subset) == 0:
            ax.text(0.5, 0.5, 'No trials', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            continue

        # Collect positions
        x, y = collect_positions(trial_subset)

        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-1, 1], [-1, 1]])

        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(h.T, extent=extent, origin='lower', cmap='hot',
                       aspect='auto', interpolation='bilinear', vmin=0, vmax=vmax)

        # Overlay target positions
        for trial in trial_subset:
            target_x = trial['target_x']
            target_y = trial['target_y']
            target_radius = trial['target_diameter'] / 2.0
            target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                                  edgecolor='cyan', linewidth=1.5, linestyle='-', alpha=0.6)
            ax.add_patch(target_circle)

        ax.set_xlabel('Horizontal Position', fontsize=10)
        ax.set_ylabel('Vertical Position', fontsize=10)
        ax.set_title(f'{phase_name}\n({trial_range})', fontsize=11, fontweight='bold')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Number of Samples', fraction=0.046, pad=0.04)

    # Overall title
    title = 'Spatial Distribution by Trial Phase'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_spatial_bias.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_spatial_bias.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved spatial bias plot to {results_dir / filename}")

    return fig


def calculate_trial_metrics_for_target(trial: dict, target_x: float, target_y: float) -> dict:
    """Calculate metrics for a trial given a specific target position.

    This allows us to compute metrics for shuffled trial-target pairings.

    Parameters
    ----------
    trial : dict
        Trial data dictionary containing trajectory
    target_x : float
        Target x position
    target_y : float
        Target y position

    Returns
    -------
    dict
        Metrics: path_efficiency, initial_direction_error, final_distance, duration
    """
    start_eye_x = trial['eye_x'][0]
    start_eye_y = trial['eye_y'][0]
    end_eye_x = trial['eye_x'][-1]
    end_eye_y = trial['eye_y'][-1]

    # Straight-line distance from start to target
    straight_line_distance = np.sqrt((target_x - start_eye_x)**2 + (target_y - start_eye_y)**2)

    # Path length (already calculated in trial)
    path_length = trial['path_length']

    # Path efficiency
    if path_length > 0:
        path_efficiency = straight_line_distance / path_length
    else:
        path_efficiency = 0.0

    # Initial direction error
    if len(trial['eye_x']) > 1:
        n_samples = min(5, len(trial['eye_x']))
        initial_dx = trial['eye_x'][n_samples-1] - start_eye_x
        initial_dy = trial['eye_y'][n_samples-1] - start_eye_y

        ideal_dx = target_x - start_eye_x
        ideal_dy = target_y - start_eye_y

        initial_mag = np.sqrt(initial_dx**2 + initial_dy**2)
        ideal_mag = np.sqrt(ideal_dx**2 + ideal_dy**2)

        if initial_mag > 0 and ideal_mag > 0:
            cos_angle = (initial_dx * ideal_dx + initial_dy * ideal_dy) / (initial_mag * ideal_mag)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            initial_direction_error = np.degrees(np.arccos(cos_angle))
        else:
            initial_direction_error = np.nan
    else:
        initial_direction_error = np.nan

    # Final distance to target
    final_distance = np.sqrt((end_eye_x - target_x)**2 + (end_eye_y - target_y)**2)

    return {
        'path_efficiency': path_efficiency,
        'initial_direction_error': initial_direction_error,
        'final_distance': final_distance,
        'duration': trial['duration']
    }


def shuffle_control_analysis(trials: list[dict], n_shuffles: int = 1000, seed: int = 42) -> dict:
    """Perform shuffle control analysis to test for voluntary control.

    Shuffles trial-target pairings and compares real metrics to shuffled distributions.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    n_shuffles : int
        Number of shuffle iterations (default: 1000)
    seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    dict
        Contains real metrics, shuffled distributions, and p-values
    """
    np.random.seed(seed)

    n_trials = len(trials)

    # Extract real target positions
    real_targets = [(t['target_x'], t['target_y']) for t in trials]

    # Calculate real metrics
    real_metrics = {
        'path_efficiency': [],
        'initial_direction_error': [],
        'final_distance': [],
        'duration': []
    }

    for trial in trials:
        metrics = calculate_trial_metrics_for_target(trial, trial['target_x'], trial['target_y'])
        for key in real_metrics:
            if not np.isnan(metrics[key]):
                real_metrics[key].append(metrics[key])

    # Initialize shuffled distributions
    shuffled_distributions = {
        'path_efficiency': [],
        'initial_direction_error': [],
        'final_distance': []
    }

    print(f"\nRunning shuffle control analysis ({n_shuffles} iterations)...")

    # Perform shuffles
    for shuffle_idx in range(n_shuffles):
        if (shuffle_idx + 1) % 100 == 0:
            print(f"  Completed {shuffle_idx + 1}/{n_shuffles} shuffles...")

        # Shuffle targets
        shuffled_targets = real_targets.copy()
        np.random.shuffle(shuffled_targets)

        # Calculate metrics for this shuffle
        shuffle_efficiencies = []
        shuffle_dir_errors = []
        shuffle_distances = []

        for trial, (target_x, target_y) in zip(trials, shuffled_targets):
            metrics = calculate_trial_metrics_for_target(trial, target_x, target_y)
            if not np.isnan(metrics['path_efficiency']):
                shuffle_efficiencies.append(metrics['path_efficiency'])
            if not np.isnan(metrics['initial_direction_error']):
                shuffle_dir_errors.append(metrics['initial_direction_error'])
            if not np.isnan(metrics['final_distance']):
                shuffle_distances.append(metrics['final_distance'])

        # Store mean for this shuffle
        shuffled_distributions['path_efficiency'].append(np.mean(shuffle_efficiencies))
        shuffled_distributions['initial_direction_error'].append(np.mean(shuffle_dir_errors))
        shuffled_distributions['final_distance'].append(np.mean(shuffle_distances))

    # Calculate p-values (one-tailed tests)
    real_mean_efficiency = np.mean(real_metrics['path_efficiency'])
    real_mean_dir_error = np.mean(real_metrics['initial_direction_error'])
    real_mean_distance = np.mean(real_metrics['final_distance'])

    # P-value: proportion of shuffles with efficiency >= real (real should be higher)
    p_efficiency = np.mean(np.array(shuffled_distributions['path_efficiency']) >= real_mean_efficiency)

    # P-value: proportion of shuffles with dir_error <= real (real should be lower)
    p_dir_error = np.mean(np.array(shuffled_distributions['initial_direction_error']) <= real_mean_dir_error)

    # P-value: proportion of shuffles with distance <= real (real should be lower)
    p_distance = np.mean(np.array(shuffled_distributions['final_distance']) <= real_mean_distance)

    results = {
        'real_metrics': real_metrics,
        'real_means': {
            'path_efficiency': real_mean_efficiency,
            'initial_direction_error': real_mean_dir_error,
            'final_distance': real_mean_distance
        },
        'shuffled_distributions': shuffled_distributions,
        'p_values': {
            'path_efficiency': p_efficiency,
            'initial_direction_error': p_dir_error,
            'final_distance': p_distance
        },
        'n_shuffles': n_shuffles
    }

    print(f"\nShuffle control results:")
    print(f"  Path Efficiency: Real={real_mean_efficiency:.3f}, p={p_efficiency:.4f}")
    print(f"  Direction Error: Real={real_mean_dir_error:.1f}°, p={p_dir_error:.4f}")
    print(f"  Final Distance:  Real={real_mean_distance:.3f}, p={p_distance:.4f}")

    return results


def plot_shuffle_control(shuffle_results: dict, results_dir: Optional[Path] = None,
                         animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot shuffle control analysis results.

    Parameters
    ----------
    shuffle_results : dict
        Results from shuffle_control_analysis()
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    real_means = shuffle_results['real_means']
    shuffled = shuffle_results['shuffled_distributions']
    p_values = shuffle_results['p_values']
    n_shuffles = shuffle_results['n_shuffles']

    # Plot 1: Path Efficiency
    ax = axes[0]
    ax.hist(shuffled['path_efficiency'], bins=50, color='gray', alpha=0.6, edgecolor='black', label='Shuffled')
    ax.axvline(real_means['path_efficiency'], color='red', linewidth=3, label=f"Real (p={p_values['path_efficiency']:.4f})")
    ax.set_xlabel('Mean Path Efficiency', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Path Efficiency\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentile text
    percentile = np.mean(np.array(shuffled['path_efficiency']) < real_means['path_efficiency']) * 100
    ax.text(0.05, 0.95, f'Real > {percentile:.1f}% of shuffles', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Plot 2: Initial Direction Error
    ax = axes[1]
    ax.hist(shuffled['initial_direction_error'], bins=50, color='gray', alpha=0.6, edgecolor='black', label='Shuffled')
    ax.axvline(real_means['initial_direction_error'], color='red', linewidth=3, label=f"Real (p={p_values['initial_direction_error']:.4f})")
    ax.set_xlabel('Mean Direction Error (degrees)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Initial Direction Error\n(Lower = Better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentile text
    percentile = np.mean(np.array(shuffled['initial_direction_error']) > real_means['initial_direction_error']) * 100
    ax.text(0.05, 0.95, f'Real < {percentile:.1f}% of shuffles', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Plot 3: Final Distance to Target
    ax = axes[2]
    ax.hist(shuffled['final_distance'], bins=50, color='gray', alpha=0.6, edgecolor='black', label='Shuffled')
    ax.axvline(real_means['final_distance'], color='red', linewidth=3, label=f"Real (p={p_values['final_distance']:.4f})")
    ax.set_xlabel('Mean Final Distance to Target', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Final Distance to Target\n(Lower = Better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentile text
    percentile = np.mean(np.array(shuffled['final_distance']) > real_means['final_distance']) * 100
    ax.text(0.05, 0.95, f'Real < {percentile:.1f}% of shuffles', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Overall title
    title = f'Shuffle Control Analysis (n={n_shuffles} shuffles)\nReal vs Shuffled Trial-Target Pairings'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_shuffle_control.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_shuffle_control.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved shuffle control plot to {results_dir / filename}")

    return fig


def shuffle_control_by_target_side(trials: list[dict], left_x: float = -0.7, right_x: float = 0.7,
                                   tolerance: float = 0.1, n_shuffles: int = 1000, seed: int = 42,
                                   results_dir: Optional[Path] = None,
                                   animal_id: Optional[str] = None, session_date: str = "") -> dict:
    """Run shuffle control analysis separately for all trials, left targets, and right targets.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    left_x : float
        Expected x position for left targets (default: -0.7)
    right_x : float
        Expected x position for right targets (default: +0.7)
    tolerance : float
        Tolerance for matching target positions (default: 0.1)
    n_shuffles : int
        Number of shuffle iterations (default: 1000)
    seed : int
        Random seed for reproducibility (default: 42)
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    dict
        Contains results for all, left, and right trials
    """
    # Classify trials by target side
    left_trials = []
    right_trials = []

    for trial in trials:
        target_x = trial['target_x']
        if abs(target_x - left_x) < tolerance:
            left_trials.append(trial)
        elif abs(target_x - right_x) < tolerance:
            right_trials.append(trial)

    n_left = len(left_trials)
    n_right = len(right_trials)

    print(f"\nRunning shuffle control by target side...")
    print(f"  All trials: {len(trials)}")
    print(f"  Left trials (x ≈ {left_x}): {n_left}")
    print(f"  Right trials (x ≈ {right_x}): {n_right}")

    # Run shuffle control for all trials
    print(f"\n  Running shuffle control for all trials...")
    all_results = shuffle_control_analysis(trials, n_shuffles=n_shuffles, seed=seed)

    # Run shuffle control for left trials
    left_results = None
    if n_left > 0:
        print(f"\n  Running shuffle control for left trials only...")
        left_results = shuffle_control_analysis(left_trials, n_shuffles=n_shuffles, seed=seed+1)
    else:
        print(f"\n  Skipping left trials (not enough trials)")

    # Run shuffle control for right trials
    right_results = None
    if n_right > 0:
        print(f"\n  Running shuffle control for right trials only...")
        right_results = shuffle_control_analysis(right_trials, n_shuffles=n_shuffles, seed=seed+2)
    else:
        print(f"\n  Skipping right trials (not enough trials)")

    # Create comprehensive visualization
    fig = plot_shuffle_control_comparison(all_results, left_results, right_results,
                                          n_left, n_right,
                                          results_dir=results_dir,
                                          animal_id=animal_id,
                                          session_date=session_date)

    results = {
        'all': all_results,
        'left': left_results,
        'right': right_results,
        'n_left': n_left,
        'n_right': n_right,
        'figure': fig
    }

    return results


def plot_shuffle_control_comparison(all_results: dict, left_results: Optional[dict],
                                    right_results: Optional[dict],
                                    n_left: int, n_right: int,
                                    results_dir: Optional[Path] = None,
                                    animal_id: Optional[str] = None,
                                    session_date: str = "") -> plt.Figure:
    """Plot comparison of shuffle control results across all, left, and right trials.

    Parameters
    ----------
    all_results : dict
        Shuffle control results for all trials
    left_results : dict, optional
        Shuffle control results for left trials
    right_results : dict, optional
        Shuffle control results for right trials
    n_left : int
        Number of left trials
    n_right : int
        Number of right trials
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create figure with 3 rows (metrics) x 3 columns (all/left/right)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Define metrics and their properties
    metrics = [
        ('path_efficiency', 'Path Efficiency', 'Higher = Better', True),  # True = higher is better
        ('initial_direction_error', 'Direction Error (°)', 'Lower = Better', False),
        ('final_distance', 'Final Distance', 'Lower = Better', False)
    ]

    results_list = [
        (all_results, f'All Trials (n={len(all_results["real_metrics"]["path_efficiency"])})', 'steelblue'),
        (left_results, f'Left Targets (n={n_left})', 'darkred'),
        (right_results, f'Right Targets (n={n_right})', 'darkgreen')
    ]

    for row_idx, (metric_key, metric_label, direction_label, higher_better) in enumerate(metrics):
        for col_idx, (results, title, color) in enumerate(results_list):
            ax = axes[row_idx, col_idx]

            if results is None:
                # No data for this condition
                ax.text(0.5, 0.5, 'Insufficient\ntrials', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xlabel(metric_label, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            real_mean = results['real_means'][metric_key]
            shuffled_dist = results['shuffled_distributions'][metric_key]
            p_value = results['p_values'][metric_key]

            # Plot histogram of shuffled distribution
            ax.hist(shuffled_dist, bins=40, color='lightgray', alpha=0.7,
                   edgecolor='black', linewidth=0.5, label='Shuffled')

            # Plot real value
            ax.axvline(real_mean, color=color, linewidth=3,
                      label=f'Real (p={p_value:.4f})')

            # Calculate percentile
            if higher_better:
                percentile = np.mean(np.array(shuffled_dist) < real_mean) * 100
                percentile_text = f'Real > {percentile:.1f}%\nof shuffles'
            else:
                percentile = np.mean(np.array(shuffled_dist) > real_mean) * 100
                percentile_text = f'Real < {percentile:.1f}%\nof shuffles'

            # Add percentile box
            box_color = 'lightgreen' if p_value < 0.05 else 'yellow'
            ax.text(0.05, 0.95, percentile_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.6))

            # Formatting
            if col_idx == 0:
                ax.set_ylabel('Count', fontsize=10)
            ax.set_xlabel(metric_label, fontsize=9)

            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')

            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2, axis='y')

    # Overall title
    title = 'Shuffle Control Analysis by Target Side\n'
    title += f'Real vs Shuffled Trial-Target Pairings (n={all_results["n_shuffles"]} shuffles)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_shuffle_by_side.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_shuffle_by_side.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"\nSaved shuffle control by side plot to {results_dir / filename}")

    return fig


def compare_left_right_performance(trials: list[dict], left_x: float = -0.7, right_x: float = 0.7,
                                   tolerance: float = 0.1, results_dir: Optional[Path] = None,
                                   animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Compare performance metrics for left vs right target trials.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    left_x : float
        Expected x position for left targets (default: -0.7)
    right_x : float
        Expected x position for right targets (default: +0.7)
    tolerance : float
        Tolerance for matching target positions (default: 0.1)
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    tuple of (fig, stats_dict)
        Figure and dictionary containing statistics and test results
    """
    from scipy import stats as scipy_stats

    # Classify trials as left or right based on target_x position
    left_trials = []
    right_trials = []
    other_trials = []

    for trial in trials:
        target_x = trial['target_x']
        if abs(target_x - left_x) < tolerance:
            left_trials.append(trial)
        elif abs(target_x - right_x) < tolerance:
            right_trials.append(trial)
        else:
            other_trials.append(trial)

    n_left = len(left_trials)
    n_right = len(right_trials)
    n_other = len(other_trials)

    print(f"\nLeft/Right Target Analysis:")
    print(f"  Left trials (x ≈ {left_x}): {n_left}")
    print(f"  Right trials (x ≈ {right_x}): {n_right}")
    print(f"  Other positions: {n_other}")

    if n_left == 0 or n_right == 0:
        print("Warning: Not enough trials for left/right comparison")
        return None, None

    # Extract metrics for each side
    def extract_metrics(trial_list):
        durations = [t['duration'] for t in trial_list]
        path_lengths = [t['path_length'] for t in trial_list]
        efficiencies = [t['path_efficiency'] for t in trial_list]
        dir_errors = [t['initial_direction_error'] for t in trial_list if not np.isnan(t['initial_direction_error'])]
        return {
            'durations': durations,
            'path_lengths': path_lengths,
            'efficiencies': efficiencies,
            'dir_errors': dir_errors
        }

    left_metrics = extract_metrics(left_trials)
    right_metrics = extract_metrics(right_trials)

    # Statistical tests (Mann-Whitney U test - non-parametric)
    duration_stat, duration_p = scipy_stats.mannwhitneyu(
        left_metrics['durations'], right_metrics['durations'], alternative='two-sided'
    )
    length_stat, length_p = scipy_stats.mannwhitneyu(
        left_metrics['path_lengths'], right_metrics['path_lengths'], alternative='two-sided'
    )
    eff_stat, eff_p = scipy_stats.mannwhitneyu(
        left_metrics['efficiencies'], right_metrics['efficiencies'], alternative='two-sided'
    )

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time to Target
    ax = axes[0, 0]
    positions = [1, 2]
    box_data = [left_metrics['durations'], right_metrics['durations']]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_ylabel('Time to Target (s)', fontsize=12)
    ax.set_title(f'Time to Target\np = {duration_p:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add means as points
    ax.plot(1, np.mean(left_metrics['durations']), 'ro', markersize=10, label='Mean')
    ax.plot(2, np.mean(right_metrics['durations']), 'ro', markersize=10)

    # Add sample sizes
    ax.text(1, ax.get_ylim()[0], f'n={n_left}', ha='center', va='top', fontsize=9)
    ax.text(2, ax.get_ylim()[0], f'n={n_right}', ha='center', va='top', fontsize=9)

    # Plot 2: Path Length
    ax = axes[0, 1]
    box_data = [left_metrics['path_lengths'], right_metrics['path_lengths']]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_ylabel('Path Length', fontsize=12)
    ax.set_title(f'Path Length\np = {length_p:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax.plot(1, np.mean(left_metrics['path_lengths']), 'ro', markersize=10, label='Mean')
    ax.plot(2, np.mean(right_metrics['path_lengths']), 'ro', markersize=10)

    # Plot 3: Path Efficiency
    ax = axes[1, 0]
    box_data = [left_metrics['efficiencies'], right_metrics['efficiencies']]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightyellow', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_ylabel('Path Efficiency', fontsize=12)
    ax.set_title(f'Path Efficiency\np = {eff_p:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax.plot(1, np.mean(left_metrics['efficiencies']), 'ro', markersize=10, label='Mean')
    ax.plot(2, np.mean(right_metrics['efficiencies']), 'ro', markersize=10)

    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Left', 'Right', 'p-value'],
        ['', f'(n={n_left})', f'(n={n_right})', ''],
        ['Duration (s)',
         f'{np.mean(left_metrics["durations"]):.2f}±{np.std(left_metrics["durations"]):.2f}',
         f'{np.mean(right_metrics["durations"]):.2f}±{np.std(right_metrics["durations"]):.2f}',
         f'{duration_p:.4f}'],
        ['Path Length',
         f'{np.mean(left_metrics["path_lengths"]):.3f}±{np.std(left_metrics["path_lengths"]):.3f}',
         f'{np.mean(right_metrics["path_lengths"]):.3f}±{np.std(right_metrics["path_lengths"]):.3f}',
         f'{length_p:.4f}'],
        ['Path Efficiency',
         f'{np.mean(left_metrics["efficiencies"]):.3f}±{np.std(left_metrics["efficiencies"]):.3f}',
         f'{np.mean(right_metrics["efficiencies"]):.3f}±{np.std(right_metrics["efficiencies"]):.3f}',
         f'{eff_p:.4f}'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight significant p-values
    for i, p_val in enumerate([duration_p, length_p, eff_p], start=2):
        if p_val < 0.05:
            table[(i, 3)].set_facecolor('#ffcccc')
            table[(i, 3)].set_text_props(weight='bold')

    ax.set_title('Summary Statistics\n(Mann-Whitney U Test)', fontsize=12, fontweight='bold')

    # Overall title
    title = 'Left vs Right Target Performance'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_left_vs_right.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_left_vs_right.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved left vs right comparison to {results_dir / filename}")

    # Compile statistics dictionary
    stats_dict = {
        'n_left': n_left,
        'n_right': n_right,
        'n_other': n_other,
        'left_metrics': left_metrics,
        'right_metrics': right_metrics,
        'p_values': {
            'duration': duration_p,
            'path_length': length_p,
            'path_efficiency': eff_p
        }
    }

    # Print summary
    print(f"\n  Duration: Left={np.mean(left_metrics['durations']):.2f}s, Right={np.mean(right_metrics['durations']):.2f}s, p={duration_p:.4f}")
    print(f"  Path Length: Left={np.mean(left_metrics['path_lengths']):.3f}, Right={np.mean(right_metrics['path_lengths']):.3f}, p={length_p:.4f}")
    print(f"  Path Efficiency: Left={np.mean(left_metrics['efficiencies']):.3f}, Right={np.mean(right_metrics['efficiencies']):.3f}, p={eff_p:.4f}")

    if duration_p < 0.05:
        print(f"  *** Significant difference in duration (p < 0.05)")
    if length_p < 0.05:
        print(f"  *** Significant difference in path length (p < 0.05)")
    if eff_p < 0.05:
        print(f"  *** Significant difference in efficiency (p < 0.05)")

    return fig, stats_dict


def _clean_path(path_str: str | Path) -> str:
    """Clean path string by removing Python string literal syntax if present.

    Handles cases where user accidentally includes r' or ' from Python syntax.
    """
    if isinstance(path_str, Path):
        return str(path_str)

    path_str = str(path_str).strip()

    # Remove r' prefix and trailing ' if present (raw string literal syntax)
    if path_str.startswith("r'") and path_str.endswith("'"):
        path_str = path_str[2:-1]
    elif path_str.startswith('r"') and path_str.endswith('"'):
        path_str = path_str[2:-1]
    # Remove regular quotes
    elif path_str.startswith("'") and path_str.endswith("'"):
        path_str = path_str[1:-1]
    elif path_str.startswith('"') and path_str.endswith('"'):
        path_str = path_str[1:-1]

    return path_str


def analyze_folder(folder_path: str | Path, results_dir: Optional[str | Path] = None,
                   animal_id: str = "Tsh001", show_plots: bool = True) -> pd.DataFrame:
    """Run saccade feedback analysis directly on a folder (without session manifest).

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder containing the CSV files
    results_dir : str or Path, optional
        Directory to save results (defaults to folder_path/results)
    animal_id : str
        Animal identifier (default: "Tsh001")
    show_plots : bool
        Whether to display plots (default: True)

    Returns
    -------
    pd.DataFrame
        Summary statistics for the session
    """
    # Clean the path in case user included Python string syntax
    folder_path = Path(_clean_path(folder_path))

    # Validate that the folder exists
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}\n"
            f"Please check the path and try again."
        )
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    if results_dir is None:
        results_dir = folder_path / "results"
    else:
        results_dir = Path(_clean_path(results_dir))

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing folder: {folder_path}")
    print(f"Results directory: {results_dir}")

    # Try to extract date from folder name
    import re
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', str(folder_path))
    date_str = date_match.group() if date_match else ""

    # Load the three CSV files
    eot_df, eye_df, target_df = load_feedback_data(folder_path, animal_id)

    # Extract trial trajectories
    trials = extract_trial_trajectories(eot_df, eye_df, target_df)

    if len(trials) == 0:
        print("No valid trials found, exiting")
        return pd.DataFrame()

    # Generate plots
    print("\nGenerating trajectory plot...")
    fig_traj = plot_trajectories(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_traj)

    print("\nGenerating density heatmap...")
    fig_heat = plot_density_heatmap(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_heat)

    print("\nGenerating time-to-target plot...")
    fig_time = plot_time_to_target(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_time)

    print("\nGenerating path length plot...")
    fig_path = plot_path_length(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_path)

    print("\nGenerating learning metrics plot...")
    fig_learn = plot_learning_metrics(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_learn)

    print("\nGenerating spatial bias by phase plot...")
    fig_spatial = plot_spatial_bias_by_phase(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_spatial)

    print("\nRunning shuffle control analysis (voluntary control test)...")
    # Original shuffle control for all trials
    shuffle_results = shuffle_control_analysis(trials, n_shuffles=1000, seed=42)
    fig_shuffle = plot_shuffle_control(shuffle_results, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_shuffle)

    # Shuffle control split by target side (all, left, right)
    print("\nRunning shuffle control by target side (all/left/right)...")
    shuffle_by_side = shuffle_control_by_target_side(trials, left_x=-0.7, right_x=0.7,
                                                      n_shuffles=1000, seed=42,
                                                      results_dir=results_dir,
                                                      animal_id=animal_id,
                                                      session_date=date_str)
    if shuffle_by_side['figure'] is not None:
        if show_plots:
            plt.show()
        plt.close(shuffle_by_side['figure'])

    print("\nRunning left vs right target comparison...")
    fig_lr, lr_stats = compare_left_right_performance(trials, left_x=-0.7, right_x=0.7,
                                                       results_dir=results_dir,
                                                       animal_id=animal_id,
                                                       session_date=date_str)
    if fig_lr is not None:
        if show_plots:
            plt.show()
        plt.close(fig_lr)

    # Create summary DataFrame
    durations = [t['duration'] for t in trials]
    path_lengths = [t['path_length'] for t in trials]
    efficiencies = [t['path_efficiency'] for t in trials]
    dir_errors = [t['initial_direction_error'] for t in trials if not np.isnan(t['initial_direction_error'])]

    df = pd.DataFrame({
        'folder_path': [str(folder_path)],
        'animal_id': [animal_id],
        'session_date': [date_str],
        'n_trials': [len(trials)],
        'mean_duration': [np.mean(durations)],
        'median_duration': [np.median(durations)],
        'std_duration': [np.std(durations)],
        'min_duration': [np.min(durations)],
        'max_duration': [np.max(durations)],
        'mean_path_length': [np.mean(path_lengths)],
        'median_path_length': [np.median(path_lengths)],
        'std_path_length': [np.std(path_lengths)],
        'mean_path_efficiency': [np.mean(efficiencies)],
        'median_path_efficiency': [np.median(efficiencies)],
        'mean_initial_dir_error': [np.mean(dir_errors) if dir_errors else np.nan],
        'median_initial_dir_error': [np.median(dir_errors) if dir_errors else np.nan],
    })

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Folder: {folder_path}")
    print(f"Animal: {animal_id}")
    print(f"Date: {date_str}")
    print(f"Valid trials: {len(trials)}")
    print(f"\nTime to Target:")
    print(f"  Mean: {np.mean(durations):.2f} ± {np.std(durations):.2f} s")
    print(f"  Median: {np.median(durations):.2f} s")
    print(f"  Range: {np.min(durations):.2f} - {np.max(durations):.2f} s")
    print(f"\nPath Length:")
    print(f"  Mean: {np.mean(path_lengths):.3f} ± {np.std(path_lengths):.3f}")
    print(f"  Median: {np.median(path_lengths):.3f}")
    print(f"  Range: {np.min(path_lengths):.3f} - {np.max(path_lengths):.3f}")
    print(f"\nPath Efficiency (1.0 = perfectly direct):")
    print(f"  Mean: {np.mean(efficiencies):.3f} ± {np.std(efficiencies):.3f}")
    print(f"  Median: {np.median(efficiencies):.3f}")
    if dir_errors:
        print(f"\nInitial Direction Error:")
        print(f"  Mean: {np.mean(dir_errors):.1f}° ± {np.std(dir_errors):.1f}°")
        print(f"  Median: {np.median(dir_errors):.1f}°")
    print("="*60)

    return df


def main(session_id: str) -> pd.DataFrame:
    """Run the saccade feedback analysis pipeline for ``session_id``.

    Parameters
    ----------
    session_id : str
        Identifier of the session to analyse.

    Returns
    -------
    pd.DataFrame
        Summary statistics for the session
    """
    config = load_session(session_id)

    folder_path = config.folder_path
    results_dir = config.results_dir
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session path: {folder_path}")
    print(f"Results directory: {results_dir}")

    # Get animal ID and date
    animal_id = config.animal_id
    date_str = config.params.get("date", "")

    # Load the three CSV files
    eot_df, eye_df, target_df = load_feedback_data(folder_path, animal_id or "Tsh001")

    # Extract trial trajectories
    trials = extract_trial_trajectories(eot_df, eye_df, target_df)

    if len(trials) == 0:
        print("No valid trials found, exiting")
        return pd.DataFrame()

    # Generate plots
    print("\nGenerating trajectory plot...")
    fig_traj = plot_trajectories(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_traj)

    print("\nGenerating density heatmap...")
    fig_heat = plot_density_heatmap(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_heat)

    print("\nGenerating time-to-target plot...")
    fig_time = plot_time_to_target(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_time)

    print("\nGenerating path length plot...")
    fig_path = plot_path_length(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_path)

    print("\nGenerating learning metrics plot...")
    fig_learn = plot_learning_metrics(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_learn)

    print("\nGenerating spatial bias by phase plot...")
    fig_spatial = plot_spatial_bias_by_phase(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_spatial)

    print("\nRunning shuffle control analysis (voluntary control test)...")
    # Original shuffle control for all trials
    shuffle_results = shuffle_control_analysis(trials, n_shuffles=1000, seed=42)
    fig_shuffle = plot_shuffle_control(shuffle_results, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_shuffle)

    # Shuffle control split by target side (all, left, right)
    print("\nRunning shuffle control by target side (all/left/right)...")
    shuffle_by_side = shuffle_control_by_target_side(trials, left_x=-0.7, right_x=0.7,
                                                      n_shuffles=1000, seed=42,
                                                      results_dir=results_dir,
                                                      animal_id=animal_id,
                                                      session_date=date_str)
    if shuffle_by_side['figure'] is not None:
        plt.show()
        plt.close(shuffle_by_side['figure'])

    print("\nRunning left vs right target comparison...")
    fig_lr, lr_stats = compare_left_right_performance(trials, left_x=-0.7, right_x=0.7,
                                                       results_dir=results_dir,
                                                       animal_id=animal_id,
                                                       session_date=date_str)
    if fig_lr is not None:
        plt.show()
        plt.close(fig_lr)

    # Create summary DataFrame
    durations = [t['duration'] for t in trials]
    path_lengths = [t['path_length'] for t in trials]
    efficiencies = [t['path_efficiency'] for t in trials]
    dir_errors = [t['initial_direction_error'] for t in trials if not np.isnan(t['initial_direction_error'])]

    df = pd.DataFrame({
        'session_id': [session_id],
        'animal_id': [animal_id],
        'session_date': [date_str],
        'n_trials': [len(trials)],
        'mean_duration': [np.mean(durations)],
        'median_duration': [np.median(durations)],
        'std_duration': [np.std(durations)],
        'min_duration': [np.min(durations)],
        'max_duration': [np.max(durations)],
        'mean_path_length': [np.mean(path_lengths)],
        'median_path_length': [np.median(path_lengths)],
        'std_path_length': [np.std(path_lengths)],
        'mean_path_efficiency': [np.mean(efficiencies)],
        'median_path_efficiency': [np.median(efficiencies)],
        'mean_initial_dir_error': [np.mean(dir_errors) if dir_errors else np.nan],
        'median_initial_dir_error': [np.median(dir_errors) if dir_errors else np.nan],
    })

    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Session: {session_id}")
    print(f"Animal: {animal_id}")
    print(f"Date: {date_str}")
    print(f"Valid trials: {len(trials)}")
    print(f"\nTime to Target:")
    print(f"  Mean: {np.mean(durations):.2f} ± {np.std(durations):.2f} s")
    print(f"  Median: {np.median(durations):.2f} s")
    print(f"  Range: {np.min(durations):.2f} - {np.max(durations):.2f} s")
    print(f"\nPath Length:")
    print(f"  Mean: {np.mean(path_lengths):.3f} ± {np.std(path_lengths):.3f}")
    print(f"  Median: {np.median(path_lengths):.3f}")
    print(f"  Range: {np.min(path_lengths):.3f} - {np.max(path_lengths):.3f}")
    print(f"\nPath Efficiency (1.0 = perfectly direct):")
    print(f"  Mean: {np.mean(efficiencies):.3f} ± {np.std(efficiencies):.3f}")
    print(f"  Median: {np.median(efficiencies):.3f}")
    if dir_errors:
        print(f"\nInitial Direction Error:")
        print(f"  Mean: {np.mean(dir_errors):.1f}° ± {np.std(dir_errors):.1f}°")
        print(f"  Median: {np.median(dir_errors):.1f}°")
    print("="*60)

    return df


# Usage Examples:
# 1. With session manifest:
#    python Clean/Python/analysis/prosaccade_feedback_session.py SESSION_ID
#
# 2. Direct folder (Linux/Mac):
#    python Clean/Python/analysis/prosaccade_feedback_session.py --folder /path/to/data --animal Tsh001
#
# 3. Direct folder (Windows - no quotes needed on command line):
#    python Clean/Python/analysis/prosaccade_feedback_session.py --folder X:\path\to\data --animal Tsh001
#
# Note: On command line, DO NOT use Python string syntax like r'...' or '...'
#       Just provide the path directly without quotes (unless path has spaces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse a saccade feedback session with visual feedback"
    )
    parser.add_argument("session_id", nargs='?', help="Session identifier from session_manifest.yml")
    parser.add_argument("--folder", type=str, help="Direct path to data folder (alternative to session_id)")
    parser.add_argument("--animal", type=str, default="Tsh001", help="Animal ID (for --folder mode)")
    parser.add_argument("--results", type=str, help="Results directory (for --folder mode)")
    args = parser.parse_args()

    if args.folder:
        # Direct folder analysis mode
        analyze_folder(args.folder, args.results, args.animal, show_plots=True)
    elif args.session_id:
        # Session manifest mode
        main(args.session_id)
    else:
        parser.error("Either session_id or --folder must be provided")
