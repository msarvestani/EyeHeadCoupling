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

        # Check number of columns
        if eot_arr.ndim == 1:
            eot_arr = eot_arr.reshape(1, -1)

        n_cols = eot_arr.shape[1]
        print(f"  Detected {n_cols} columns in endoftrial file")

        if n_cols == 6:
            eot_df = pd.DataFrame(eot_arr, columns=['frame', 'timestamp', 'trial_number', 'green_x', 'green_y', 'diameter'])
        elif n_cols == 5:
            # Older format without diameter column
            eot_df = pd.DataFrame(eot_arr, columns=['frame', 'timestamp', 'trial_number', 'green_x', 'green_y'])
            eot_df['diameter'] = 0.2  # Default diameter if not present
            print(f"  Warning: 'diameter' column not found, using default value of 0.2")
        else:
            raise ValueError(f"Unexpected number of columns: {n_cols}. Expected 5 or 6.")

        eot_df['frame'] = eot_df['frame'].astype(int)
        eot_df['trial_number'] = eot_df['trial_number'].astype(int)

        print(f"  Loaded {len(eot_df)} end-of-trial events")
    except Exception as e:
        raise ValueError(f"Error loading end of trial file {endoftrial_file}: {e}")

    # Load eye position / green dot position data using standard approach
    # Read from end: last column is ignored, -2 is y, -3 is x
    # This generalizes across files with 5, 6, or more columns
    try:
        print(f"\nLoading {vstim_go_file.name}...")
        cleaned = clean_csv(str(vstim_go_file))
        eye_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1)

        # Check number of columns
        if eye_arr.ndim == 1:
            eye_arr = eye_arr.reshape(1, -1)

        n_cols = eye_arr.shape[1]
        print(f"  Detected {n_cols} columns in vstim_go file")

        if n_cols < 4:
            raise ValueError(f"Too few columns: {n_cols}. Expected at least 4 (frame, timestamp, x, y)")

        # Extract columns: first 2 are frame & timestamp, last is ignored, -2 is y, -3 is x
        eye_df = pd.DataFrame({
            'frame': eye_arr[:, 0],
            'timestamp': eye_arr[:, 1],
            'green_x': eye_arr[:, -3],
            'green_y': eye_arr[:, -2],
        })
        eye_df['diameter'] = 0.2  # Default diameter

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

        # Check number of columns
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(1, -1)

        n_cols = target_arr.shape[1]
        print(f"  Detected {n_cols} columns in vstim_cue file")

        if n_cols == 5:
            target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y', 'diameter'])
        elif n_cols == 4:
            # Older format without diameter column
            target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y'])
            target_df['diameter'] = 0.5  # Default target diameter if not present
            print(f"  Warning: 'diameter' column not found, using default value of 0.5")
        else:
            raise ValueError(f"Unexpected number of columns: {n_cols}. Expected 4 or 5.")

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

        # Find target position for this trial
        # The trial starts when the target appears (vstim_cue), not at previous trial end
        # Search for target between previous trial end and current trial end
        if i > 0:
            search_start_frame = eot_df.iloc[i-1]['frame']
        else:
            search_start_frame = 0

        target_mask = (target_df['frame'] > search_start_frame) & (target_df['frame'] <= end_frame)
        target_samples = target_df[target_mask]

        if len(target_samples) > 0:
            # Use the first target position for this trial
            target_x = target_samples.iloc[0]['target_x']
            target_y = target_samples.iloc[0]['target_y']
            target_diameter = target_samples.iloc[0]['diameter']
            # FIXED: Trial starts when target appears, not at previous trial end
            start_frame = target_samples.iloc[0]['frame']
            start_time = target_samples.iloc[0]['timestamp']
        else:
            # If no target found, skip this trial
            print(f"Warning: No target found for trial {trial_num}, skipping")
            continue

        # Extract eye position trajectory for this trial
        # FIXED: Now starts from target onset, excluding inter-trial interval
        eye_mask = (eye_df['frame'] >= start_frame) & (eye_df['frame'] <= end_frame)
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

        # Plot trajectory as lines
        color = cmap(i / max(1, n_trials - 1))
        ax.plot(eye_x, eye_y, '-', color=color, alpha=0.6, linewidth=1.5,
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


def plot_trajectories_by_direction(trials: list[dict], results_dir: Optional[Path] = None,
                                   animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot eye position trajectories with different colors for left vs right targets.

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
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate trials by target direction
    left_trials = [t for t in trials if t['target_x'] < 0]
    right_trials = [t for t in trials if t['target_x'] >= 0]

    # Colors for left and right
    left_color = 'blue'
    right_color = 'red'

    # Plot left trials
    for trial in left_trials:
        eye_x = trial['eye_x']
        eye_y = trial['eye_y']

        ax.plot(eye_x, eye_y, '-', color=left_color, alpha=0.5, linewidth=1.5)

        # Mark start and end points
        ax.plot(eye_x[0], eye_y[0], 'o', color=left_color, markersize=6, alpha=0.7,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot(eye_x[-1], eye_y[-1], 's', color=left_color, markersize=6, alpha=0.7,
                markeredgecolor='white', markeredgewidth=1)

    # Plot right trials
    for trial in right_trials:
        eye_x = trial['eye_x']
        eye_y = trial['eye_y']

        ax.plot(eye_x, eye_y, '-', color=right_color, alpha=0.5, linewidth=1.5)

        # Mark start and end points
        ax.plot(eye_x[0], eye_y[0], 'o', color=right_color, markersize=6, alpha=0.7,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot(eye_x[-1], eye_y[-1], 's', color=right_color, markersize=6, alpha=0.7,
                markeredgecolor='white', markeredgewidth=1)

    # Draw targets
    targets_drawn = set()
    for trial in trials:
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        key = (round(target_x, 2), round(target_y, 2))

        if key not in targets_drawn:
            target_circle = Circle((target_x, target_y), radius=target_radius,
                                  fill=False, edgecolor='black', linewidth=2.5, linestyle='-')
            ax.add_patch(target_circle)
            ax.plot(target_x, target_y, 'ko', markersize=4)
            targets_drawn.add(key)

    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)

    title = 'Eye Position Trajectories by Target Direction'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=left_color, linewidth=2, label=f'Left targets (n={len(left_trials)})'),
        Line2D([0], [0], color=right_color, linewidth=2, label=f'Right targets (n={len(right_trials)})'),
        Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=8,
               markeredgecolor='white', markeredgewidth=1, label='Start'),
        Line2D([0], [0], marker='s', color='gray', linewidth=0, markersize=8,
               markeredgecolor='white', markeredgewidth=1, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_trajectories_by_direction.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_trajectories_by_direction.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved trajectory by direction plot to {results_dir / filename}")

    return fig


def plot_trajectories_by_time(trials: list[dict], results_dir: Optional[Path] = None,
                                animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot eye position trajectories colored by temporal progression within each trial.

    Each trajectory is divided into quartiles (0-25%, 25-50%, 50-75%, 75-100% of trial duration)
    with different colors to show if movements diverge early for left vs right targets.

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

    # Define colors for temporal quartiles - using viridis-like progression (purple -> green -> yellow)
    quartile_colors = ['#440154', '#31688e', '#35b779', '#fde724']  # purple -> teal -> green -> yellow
    quartile_labels = ['0-25%', '25-50%', '50-75%', '75-100%']

    # Plot each trial
    for trial in trials:
        eye_x = trial['eye_x']
        eye_y = trial['eye_y']
        n_samples = len(eye_x)

        if n_samples < 4:
            # Not enough samples to divide into quartiles, just plot as scatter
            ax.scatter(eye_x, eye_y, c='gray', alpha=0.2, s=15, edgecolors='none')
            continue

        # Divide trajectory into quartiles
        quartile_size = n_samples / 4.0

        for q in range(4):
            # Get indices for this quartile
            start_idx = int(q * quartile_size)
            end_idx = int((q + 1) * quartile_size) if q < 3 else n_samples

            if end_idx > start_idx:
                # Plot this segment as scatter points
                x_segment = eye_x[start_idx:end_idx]
                y_segment = eye_y[start_idx:end_idx]

                ax.scatter(x_segment, y_segment, c=quartile_colors[q],
                          alpha=0.4, s=20, edgecolors='none')

        # Mark start point
        ax.plot(eye_x[0], eye_y[0], 'o', color=quartile_colors[0], markersize=10,
               markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)

        # Mark end point
        ax.plot(eye_x[-1], eye_y[-1], 's', color=quartile_colors[3], markersize=10,
               markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)

        # Draw target position
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                              edgecolor='black', linewidth=2.5, linestyle='-', alpha=0.8)
        ax.add_patch(target_circle)
        ax.plot(target_x, target_y, 'ko', markersize=4)

    # Create custom legend for quartiles
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=quartile_colors[i],
                             markersize=10, label=quartile_labels[i],
                             linestyle='None') for i in range(4)]
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=quartile_colors[0],
                                 markeredgecolor='white', markeredgewidth=1.5,
                                 markersize=10,
                                 label='Trial start', linestyle='None'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                 markerfacecolor=quartile_colors[3],
                                 markeredgecolor='white', markeredgewidth=1.5,
                                 markersize=10,
                                 label='Trial end', linestyle='None'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             title='Trial Time', framealpha=0.9)

    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)

    title = 'Eye Position Trajectories Colored by Time\n(Purple=early, Yellow=late within each trial)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_trajectories_by_time.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_trajectories_by_time.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved trajectory by time plot to {results_dir / filename}")

    return fig


def interactive_trajectories(trials: list[dict], animal_id: Optional[str] = None,
                            session_date: str = ""):
    """Interactive plot showing trajectories one trial at a time. Press spacebar to advance.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    animal_id : str, optional
        Animal identifier for title
    session_date : str, optional
        Session date for title
    """
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color map for trials
    cmap = plt.cm.coolwarm
    n_trials = len(trials)

    # Set up the plot
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)

    title = 'Eye Position Trajectories - Interactive (Press SPACE for next trial)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Pre-draw all targets (static) in black
    target_circles = {}  # Store target circles by position
    for trial in trials:
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        key = (round(target_x, 2), round(target_y, 2))

        if key not in target_circles:
            target_circle = Circle((target_x, target_y), radius=target_radius,
                                  fill=False, edgecolor='black', linewidth=2,
                                  linestyle='-', alpha=0.5)
            ax.add_patch(target_circle)
            ax.plot(target_x, target_y, 'ko', markersize=3, alpha=0.5)
            target_circles[key] = (target_circle, target_x, target_y)

    # Storage for current trial elements (will be cleared each time)
    trial_lines = []
    current_target_highlight = []
    current_trial_idx = [0]  # Use list to modify in nested function

    # Text showing progress
    progress_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    def plot_trial(trial_idx):
        """Plot a single trial"""
        nonlocal trial_lines, current_target_highlight

        if trial_idx >= n_trials:
            progress_text.set_text(f'All {n_trials} trials shown!\n(Close window to continue)')
            fig.canvas.draw()
            return

        # Clear previous trial's trajectory
        for line in trial_lines:
            line.remove()
        trial_lines = []

        # Clear previous target highlight
        for item in current_target_highlight:
            item.remove()
        current_target_highlight = []

        trial = trials[trial_idx]
        eye_x = trial['eye_x']
        eye_y = trial['eye_y']
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        color = cmap(trial_idx / max(1, n_trials - 1))

        # Highlight current target in green
        target_circle_green = Circle((target_x, target_y), radius=target_radius,
                                     fill=True, facecolor='green', edgecolor='darkgreen',
                                     linewidth=3, alpha=0.3)
        ax.add_patch(target_circle_green)
        target_dot_green, = ax.plot(target_x, target_y, 'go', markersize=8,
                                    markeredgecolor='darkgreen', markeredgewidth=2)
        current_target_highlight = [target_circle_green, target_dot_green]

        # Plot trajectory
        line, = ax.plot(eye_x, eye_y, '-', color=color, linewidth=2, alpha=0.7)
        start, = ax.plot(eye_x[0], eye_y[0], 'o', color=color,
                        markersize=10, markeredgecolor='white',
                        markeredgewidth=2, alpha=0.9, label='Start')
        end, = ax.plot(eye_x[-1], eye_y[-1], 's', color=color,
                      markersize=10, markeredgecolor='white',
                      markeredgewidth=2, alpha=0.9, label='End')

        trial_lines.extend([line, start, end])

        # Update progress text
        target_dir = 'Left' if trial['target_x'] < 0 else 'Right'
        progress_text.set_text(
            f"Trial {trial['trial_number']} (showing {trial_idx + 1}/{n_trials})\n"
            f"Target: {target_dir}\n"
            f"Duration: {trial['duration']:.3f}s\n"
            f"Efficiency: {trial['path_efficiency']:.2f}\n\n"
            f"Press SPACE for next"
        )

        # Add legend (only on first trial to avoid duplicates)
        if trial_idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        fig.canvas.draw()

    def on_key(event):
        """Handle key press events"""
        if event.key == ' ':  # Spacebar
            current_trial_idx[0] += 1
            plot_trial(current_trial_idx[0])

    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show first trial
    plot_trial(0)

    plt.show()


def animate_trajectories(trials: list[dict], results_dir: Optional[Path] = None,
                        animal_id: Optional[str] = None, session_date: str = "",
                        fps: int = 30, points_per_frame: int = 2) -> str:
    """Create an animation showing trajectories building over time, one trial at a time.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    results_dir : Path, optional
        Directory to save the animation
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title
    fps : int
        Frames per second for animation (default: 30)
    points_per_frame : int
        Number of points to add per frame when building trajectory (default: 2)

    Returns
    -------
    str
        Path to saved animation file
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color map for trials
    cmap = plt.cm.coolwarm
    n_trials = len(trials)

    # Set up the plot
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=12)

    title = 'Eye Position Trajectories - Animated'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Pre-draw all targets (they don't animate)
    for trial in trials:
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        target_circle = Circle((target_x, target_y), radius=target_radius,
                              fill=False, edgecolor='black', linewidth=2,
                              linestyle='-', alpha=0.5)
        ax.add_patch(target_circle)
        ax.plot(target_x, target_y, 'ko', markersize=3, alpha=0.5)

    # Storage for completed trials (will persist across frames)
    completed_lines = []
    completed_markers = []

    # Current trial line and points (updated each frame)
    current_line, = ax.plot([], [], '-', linewidth=1.5, alpha=0.8)
    current_start, = ax.plot([], [], 'o', markersize=8, markeredgecolor='white',
                             markeredgewidth=1, alpha=0.9)
    current_end, = ax.plot([], [], 's', markersize=8, markeredgecolor='white',
                           markeredgewidth=1, alpha=0.9)

    # Text showing progress
    progress_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Calculate total number of frames needed
    total_points = sum(len(trial['eye_x']) for trial in trials)
    total_frames = int(np.ceil(total_points / points_per_frame))

    # Keep track of where we are
    current_trial_idx = 0
    current_point_idx = 0

    def init():
        """Initialize animation"""
        current_line.set_data([], [])
        current_start.set_data([], [])
        current_end.set_data([], [])
        progress_text.set_text('')
        return [current_line, current_start, current_end, progress_text]

    def animate(frame):
        """Animation function called for each frame"""
        nonlocal current_trial_idx, current_point_idx, completed_lines, completed_markers

        # Check if we've finished all trials
        if current_trial_idx >= n_trials:
            return [current_line, current_start, current_end, progress_text]

        trial = trials[current_trial_idx]
        eye_x = trial['eye_x']
        eye_y = trial['eye_y']
        color = cmap(current_trial_idx / max(1, n_trials - 1))

        # Add points to current trajectory
        end_idx = min(current_point_idx + points_per_frame, len(eye_x))

        # Update current line
        current_line.set_data(eye_x[:end_idx], eye_y[:end_idx])
        current_line.set_color(color)

        # Update start marker
        current_start.set_data([eye_x[0]], [eye_y[0]])
        current_start.set_color(color)

        # Update end marker if we're at the end of this trial
        if end_idx == len(eye_x):
            current_end.set_data([eye_x[-1]], [eye_y[-1]])
            current_end.set_color(color)
        else:
            current_end.set_data([], [])

        # Update progress text
        progress_text.set_text(f'Trial {current_trial_idx + 1}/{n_trials}\n' +
                              f'Point {end_idx}/{len(eye_x)}')

        # Check if current trial is complete
        if end_idx >= len(eye_x):
            # Save this trial as a completed line
            completed_line, = ax.plot(eye_x, eye_y, '-', color=color,
                                     linewidth=1.5, alpha=0.6)
            completed_start, = ax.plot(eye_x[0], eye_y[0], 'o', color=color,
                                      markersize=8, markeredgecolor='white',
                                      markeredgewidth=1, alpha=0.9)
            completed_end, = ax.plot(eye_x[-1], eye_y[-1], 's', color=color,
                                    markersize=8, markeredgecolor='white',
                                    markeredgewidth=1, alpha=0.9)

            completed_lines.extend([completed_line, completed_start, completed_end])

            # Move to next trial
            current_trial_idx += 1
            current_point_idx = 0

            # Clear current line for next trial
            current_line.set_data([], [])
            current_start.set_data([], [])
            current_end.set_data([], [])
        else:
            current_point_idx = end_idx

        return [current_line, current_start, current_end, progress_text] + completed_lines

    # Create animation
    print(f"\nCreating animation ({total_frames} frames at {fps} fps)...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=total_frames, interval=1000/fps,
                                  blit=True, repeat=False)

    # Save animation
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_trajectories_animated.mp4"
        filepath = results_dir / filename

        print(f"Saving animation to {filepath}")
        print("(This may take a minute...)")

        # Try to save as mp4 (requires ffmpeg)
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(str(filepath), writer=writer, dpi=100)
            print(f"Saved animation to {filepath}")
            plt.close(fig)
            return str(filepath)
        except Exception as e:
            print(f"Could not save as mp4 (ffmpeg may not be installed): {e}")
            print("Trying to save as gif instead...")

            # Fall back to gif
            filename_gif = f"{prefix}saccade_feedback_trajectories_animated.gif"
            filepath_gif = results_dir / filename_gif
            try:
                anim.save(str(filepath_gif), writer='pillow', fps=fps, dpi=80)
                print(f"Saved animation as gif to {filepath_gif}")
                plt.close(fig)
                return str(filepath_gif)
            except Exception as e2:
                print(f"Could not save animation: {e2}")
                print("Please install ffmpeg or pillow to save animations")
                plt.close(fig)
                return None
    else:
        plt.close(fig)
        return None


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
        'initial_direction_error': []
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

        for trial, (target_x, target_y) in zip(trials, shuffled_targets):
            metrics = calculate_trial_metrics_for_target(trial, target_x, target_y)
            if not np.isnan(metrics['path_efficiency']):
                shuffle_efficiencies.append(metrics['path_efficiency'])
            if not np.isnan(metrics['initial_direction_error']):
                shuffle_dir_errors.append(metrics['initial_direction_error'])

        # Store mean for this shuffle
        shuffled_distributions['path_efficiency'].append(np.mean(shuffle_efficiencies))
        shuffled_distributions['initial_direction_error'].append(np.mean(shuffle_dir_errors))

    # Calculate p-values (one-tailed tests)
    real_mean_efficiency = np.mean(real_metrics['path_efficiency'])
    real_mean_dir_error = np.mean(real_metrics['initial_direction_error'])

    # P-value: proportion of shuffles with efficiency >= real (real should be higher)
    p_efficiency = np.mean(np.array(shuffled_distributions['path_efficiency']) >= real_mean_efficiency)

    # P-value: proportion of shuffles with dir_error <= real (real should be lower)
    p_dir_error = np.mean(np.array(shuffled_distributions['initial_direction_error']) <= real_mean_dir_error)

    results = {
        'real_metrics': real_metrics,
        'real_means': {
            'path_efficiency': real_mean_efficiency,
            'initial_direction_error': real_mean_dir_error
        },
        'shuffled_distributions': shuffled_distributions,
        'p_values': {
            'path_efficiency': p_efficiency,
            'initial_direction_error': p_dir_error
        },
        'n_shuffles': n_shuffles
    }

    print(f"\nShuffle control results:")
    print(f"  Path Efficiency: Real={real_mean_efficiency:.3f}, p={p_efficiency:.4f}")
    print(f"  Direction Error: Real={real_mean_dir_error:.1f}°, p={p_dir_error:.4f}")

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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


def test_initial_direction_correlation(trials: list[dict], results_dir: Optional[Path] = None,
                                        animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Test #2: Initial Direction Correlation - do initial movements point toward targets?

    Voluntary movements should show strong correlation between initial movement direction
    and the actual direction to the target. Random movements would show no correlation.

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
    tuple of (fig, stats_dict)
        Figure and dictionary containing correlation statistics
    """
    from scipy import stats as scipy_stats

    # Calculate angles for each trial
    target_angles = []
    initial_angles = []

    for trial in trials:
        start_x = trial['eye_x'][0]
        start_y = trial['eye_y'][0]
        target_x = trial['target_x']
        target_y = trial['target_y']

        # Angle to target (in degrees, 0 = right, 90 = up)
        target_angle = np.degrees(np.arctan2(target_y - start_y, target_x - start_x))

        # Initial movement angle (using first 5 samples)
        if len(trial['eye_x']) >= 5:
            n_samples = 5
            initial_x = trial['eye_x'][n_samples-1]
            initial_y = trial['eye_y'][n_samples-1]
            initial_angle = np.degrees(np.arctan2(initial_y - start_y, initial_x - start_x))

            target_angles.append(target_angle)
            initial_angles.append(initial_angle)

    target_angles = np.array(target_angles)
    initial_angles = np.array(initial_angles)

    # FIXED: Handle circular statistics for left/right targets properly
    # Left targets are at ~±180° (wraps around), right targets at ~0°
    # Unwrap angles so left targets are consistently at 180° (not split between -180 and +180)
    # This prevents artificial splitting of the same target direction

    # For angles near ±180°, convert to +180° for consistency
    target_angles = np.where(target_angles < -90, target_angles + 360, target_angles)
    initial_angles = np.where(initial_angles < -90, initial_angles + 360, initial_angles)

    # Now angles are in range [-90, 270] approximately
    # Right targets: ~0°
    # Left targets: ~180° (not split between -180 and +180)

    # Calculate correlation
    r, p_value = scipy_stats.pearsonr(target_angles, initial_angles)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter plot with regression line
    ax1.scatter(target_angles, initial_angles, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Determine plot range based on actual data
    all_angles = np.concatenate([target_angles, initial_angles])
    angle_min = max(-100, all_angles.min() - 10)
    angle_max = min(280, all_angles.max() + 10)

    # Add diagonal line (perfect correlation)
    ax1.plot([angle_min, angle_max], [angle_min, angle_max], 'g--', linewidth=2, alpha=0.5,
             label='Perfect correlation')

    # Add reference lines for left (180°) and right (0°) targets
    ax1.axvline(0, color='blue', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(180, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axhline(0, color='blue', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axhline(180, color='red', linestyle=':', linewidth=1.5, alpha=0.6)

    # Set limits
    ax1.set_xlim(angle_min, angle_max)
    ax1.set_ylim(angle_min, angle_max)

    # Add regression line
    z = np.polyfit(target_angles, initial_angles, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(target_angles.min(), target_angles.max(), 100)
    ax1.plot(x_fit, p(x_fit), 'r-', linewidth=2, label=f'Actual fit (r={r:.3f})')

    ax1.set_xlabel('Target Direction (degrees)\n0°=right, 180°=left', fontsize=12)
    ax1.set_ylabel('Initial Movement Direction (degrees)', fontsize=12)
    ax1.set_title(f'Initial Direction Correlation\nr = {r:.3f}, p = {p_value:.4e}',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Add interpretation box
    if r > 0.7 and p_value < 0.05:
        interpretation = 'VOLUNTARY\n(strong correlation)'
        box_color = 'lightgreen'
    elif r > 0.4 and p_value < 0.05:
        interpretation = 'Likely voluntary\n(moderate correlation)'
        box_color = 'yellow'
    else:
        interpretation = 'Random or weak\n(low correlation)'
        box_color = 'lightcoral'

    ax1.text(0.05, 0.95, interpretation, transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    # Plot 2: Residuals (angular errors)
    angular_errors = initial_angles - target_angles
    # Normalize to [-180, 180]
    angular_errors = (angular_errors + 180) % 360 - 180

    ax2.hist(angular_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect aiming')
    ax2.axvline(np.mean(angular_errors), color='orange', linestyle='-', linewidth=2,
                label=f'Mean error: {np.mean(angular_errors):.1f}°')
    ax2.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Distribution of Aiming Errors\nStd = {np.std(angular_errors):.1f}°',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Overall title
    title = 'Test #2: Initial Direction Correlation (Voluntary Control Test)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Save figure
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_test2_direction_correlation.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_test2_direction_correlation.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved initial direction correlation plot to {results_dir / filename}")

    stats_dict = {
        'r': r,
        'p_value': p_value,
        'mean_angular_error': np.mean(angular_errors),
        'std_angular_error': np.std(angular_errors),
        'n_trials': len(target_angles)
    }

    print(f"\nTest #2: Initial Direction Correlation")
    print(f"  Correlation: r = {r:.3f}, p = {p_value:.4e}")
    print(f"  Mean angular error: {np.mean(angular_errors):.1f}° ± {np.std(angular_errors):.1f}°")
    print(f"  Interpretation: {'VOLUNTARY' if r > 0.7 and p_value < 0.05 else 'Random or weak'}")

    return fig, stats_dict


def test_trial_to_trial_adaptation(trials: list[dict], results_dir: Optional[Path] = None,
                                   animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Test #3: Trial-to-Trial Learning/Adaptation

    Voluntary behavior should show trial-to-trial correlations in performance,
    either through error correction or learning consistency. Random movements
    would show no correlation between consecutive trials.

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
    tuple of (fig, stats_dict)
        Figure and dictionary containing auto-correlation statistics
    """
    from scipy import stats as scipy_stats

    # Extract metrics
    durations = np.array([t['duration'] for t in trials])
    efficiencies = np.array([t['path_efficiency'] for t in trials])
    dir_errors = np.array([t['initial_direction_error'] for t in trials])

    # Calculate consecutive differences (trial N+1 - trial N)
    duration_diffs = np.diff(durations)
    efficiency_diffs = np.diff(efficiencies)

    # Calculate auto-correlation (correlation between trial N and trial N+1)
    duration_autocorr, duration_p = scipy_stats.pearsonr(durations[:-1], durations[1:])
    efficiency_autocorr, efficiency_p = scipy_stats.pearsonr(efficiencies[:-1], efficiencies[1:])

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Duration auto-correlation
    ax = axes[0, 0]
    ax.scatter(durations[:-1], durations[1:], alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Add diagonal line (perfect persistence)
    lim_min = min(durations.min(), durations.min())
    lim_max = max(durations.max(), durations.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=2, alpha=0.5,
            label='Perfect persistence')

    # Add regression line
    z = np.polyfit(durations[:-1], durations[1:], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(durations.min(), durations.max(), 100)
    ax.plot(x_fit, p(x_fit), 'r-', linewidth=2, label=f'r={duration_autocorr:.3f}')

    ax.set_xlabel('Trial N Duration (s)', fontsize=11)
    ax.set_ylabel('Trial N+1 Duration (s)', fontsize=11)
    ax.set_title(f'Duration Auto-correlation\nr = {duration_autocorr:.3f}, p = {duration_p:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Efficiency auto-correlation
    ax = axes[0, 1]
    ax.scatter(efficiencies[:-1], efficiencies[1:], alpha=0.6, s=60,
               edgecolors='black', linewidth=0.5, color='orange')

    lim_min = efficiencies.min()
    lim_max = efficiencies.max()
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=2, alpha=0.5,
            label='Perfect persistence')

    z = np.polyfit(efficiencies[:-1], efficiencies[1:], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(efficiencies.min(), efficiencies.max(), 100)
    ax.plot(x_fit, p(x_fit), 'r-', linewidth=2, label=f'r={efficiency_autocorr:.3f}')

    ax.set_xlabel('Trial N Efficiency', fontsize=11)
    ax.set_ylabel('Trial N+1 Efficiency', fontsize=11)
    ax.set_title(f'Efficiency Auto-correlation\nr = {efficiency_autocorr:.3f}, p = {efficiency_p:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Duration changes over trials
    ax = axes[1, 0]
    trial_nums = np.arange(1, len(trials))
    ax.plot(trial_nums, duration_diffs, 'o-', alpha=0.6, markersize=4)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No change')
    ax.set_xlabel('Trial Number', fontsize=11)
    ax.set_ylabel('Duration Change (trial N+1 - N)', fontsize=11)
    ax.set_title('Trial-to-Trial Duration Changes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Moving average of efficiency (learning curve)
    ax = axes[1, 1]
    window = min(10, len(efficiencies) // 3)  # Adaptive window size
    if window >= 3:
        moving_avg = np.convolve(efficiencies, np.ones(window)/window, mode='valid')
        ax.plot(range(len(efficiencies)), efficiencies, 'o', alpha=0.3, markersize=4,
                color='lightblue', label='Individual trials')
        ax.plot(range(window-1, len(efficiencies)), moving_avg, 'b-', linewidth=3,
                label=f'{window}-trial moving avg')

        # Add trend line
        trial_indices = np.arange(len(efficiencies))
        z = np.polyfit(trial_indices, efficiencies, 1)
        p = np.poly1d(z)
        ax.plot(trial_indices, p(trial_indices), 'r--', linewidth=2,
                label=f'Trend: {z[0]:+.4f}/trial')
    else:
        ax.plot(range(len(efficiencies)), efficiencies, 'o-', alpha=0.6, markersize=6)

    ax.set_xlabel('Trial Number', fontsize=11)
    ax.set_ylabel('Path Efficiency', fontsize=11)
    ax.set_title('Learning Curve (Efficiency)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overall interpretation
    interpretation_lines = []
    if abs(duration_autocorr) > 0.3 and duration_p < 0.05:
        interpretation_lines.append(f"✓ Duration shows trial-to-trial correlation (r={duration_autocorr:.3f})")
    if abs(efficiency_autocorr) > 0.3 and efficiency_p < 0.05:
        interpretation_lines.append(f"✓ Efficiency shows trial-to-trial correlation (r={efficiency_autocorr:.3f})")

    if len(interpretation_lines) > 0:
        interpretation = "VOLUNTARY behavior:\n" + "\n".join(interpretation_lines)
        box_color = 'lightgreen'
    else:
        interpretation = "Weak or no trial-to-trial correlation\n(consistent with random movements)"
        box_color = 'lightcoral'

    fig.text(0.5, 0.01, interpretation, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    # Overall title
    title = 'Test #3: Trial-to-Trial Adaptation (Voluntary Control Test)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_test3_trial_adaptation.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_test3_trial_adaptation.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved trial-to-trial adaptation plot to {results_dir / filename}")

    stats_dict = {
        'duration_autocorr': duration_autocorr,
        'duration_p': duration_p,
        'efficiency_autocorr': efficiency_autocorr,
        'efficiency_p': efficiency_p,
        'n_trials': len(trials)
    }

    print(f"\nTest #3: Trial-to-Trial Adaptation")
    print(f"  Duration auto-correlation: r = {duration_autocorr:.3f}, p = {duration_p:.4f}")
    print(f"  Efficiency auto-correlation: r = {efficiency_autocorr:.3f}, p = {efficiency_p:.4f}")

    return fig, stats_dict


def test_speed_accuracy_tradeoff(trials: list[dict], results_dir: Optional[Path] = None,
                                 animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Test #6: Speed-Accuracy Tradeoff

    Voluntary movements typically show a speed-accuracy tradeoff: slower trials
    are more efficient/accurate. Random movements would show no such relationship.

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
    tuple of (fig, stats_dict)
        Figure and dictionary containing correlation statistics
    """
    from scipy import stats as scipy_stats

    # Extract metrics
    durations = []
    efficiencies = []

    for trial in trials:
        durations.append(trial['duration'])
        efficiencies.append(trial['path_efficiency'])

    durations = np.array(durations)
    efficiencies = np.array(efficiencies)

    # Calculate correlation
    r_eff, p_eff = scipy_stats.pearsonr(durations, efficiencies)

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Color points by trial number to show temporal progression
    trial_nums = np.arange(len(trials))
    scatter = ax.scatter(durations, efficiencies, alpha=0.6, s=80, c=trial_nums,
                         cmap='coolwarm', edgecolors='black', linewidth=0.5)

    # Add regression line
    z = np.polyfit(durations, efficiencies, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(durations.min(), durations.max(), 100)
    ax.plot(x_fit, p(x_fit), 'r-', linewidth=3, label=f'r={r_eff:.3f}')

    ax.set_xlabel('Trial Duration (s)', fontsize=13)
    ax.set_ylabel('Path Efficiency', fontsize=13)
    ax.set_title(f'Speed vs Efficiency\nr = {r_eff:.3f}, p = {p_eff:.4f}',
                  fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add colorbar for trial progression
    cbar = plt.colorbar(scatter, ax=ax, label='Trial Number')

    # Add interpretation
    if r_eff > 0.3 and p_eff < 0.05:
        interpretation = 'VOLUNTARY\n(slower → more efficient)'
        box_color = 'lightgreen'
    elif r_eff < -0.3 and p_eff < 0.05:
        interpretation = 'VOLUNTARY\n(faster → more efficient)'
        box_color = 'lightgreen'
    else:
        interpretation = 'No clear tradeoff'
        box_color = 'yellow'

    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    # Overall title
    title = 'Test #6: Speed-Accuracy Tradeoff (Voluntary Control Test)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save figure
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_test6_speed_accuracy.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_test6_speed_accuracy.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved speed-accuracy tradeoff plot to {results_dir / filename}")

    stats_dict = {
        'r_duration_efficiency': r_eff,
        'p_duration_efficiency': p_eff,
        'n_trials': len(trials)
    }

    print(f"\nTest #6: Speed-Accuracy Tradeoff")
    print(f"  Duration vs Efficiency: r = {r_eff:.3f}, p = {p_eff:.4f}")

    return fig, stats_dict


def test_reaction_time_consistency(trials: list[dict], movement_threshold: float = 0.01,
                                   results_dir: Optional[Path] = None,
                                   animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Test #7: Reaction Time Consistency

    Voluntary movements should have consistent reaction times (latency from trial
    start to first movement). Random movements would have high variance or no
    consistent pattern.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    movement_threshold : float
        Distance threshold for detecting movement onset (default: 0.01)
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    tuple of (fig, stats_dict)
        Figure and dictionary containing reaction time statistics
    """
    from scipy import stats as scipy_stats

    # Calculate reaction times for each trial
    reaction_times = []
    reaction_distances = []

    for trial in trials:
        start_x = trial['eye_x'][0]
        start_y = trial['eye_y'][0]
        times = trial['eye_times'] - trial['eye_times'][0]  # Relative to trial start

        # Find first movement (when cumulative distance exceeds threshold)
        for i in range(1, len(trial['eye_x'])):
            dist = np.sqrt((trial['eye_x'][i] - start_x)**2 + (trial['eye_y'][i] - start_y)**2)
            if dist > movement_threshold:
                reaction_times.append(times[i])
                reaction_distances.append(dist)
                break
        else:
            # No movement detected - use full duration
            reaction_times.append(trial['duration'])
            reaction_distances.append(0)

    reaction_times = np.array(reaction_times)

    # Calculate statistics
    mean_rt = np.mean(reaction_times)
    std_rt = np.std(reaction_times)
    cv = std_rt / mean_rt if mean_rt > 0 else np.inf  # Coefficient of variation

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Histogram of reaction times
    ax = axes[0, 0]
    n, bins, patches = ax.hist(reaction_times, bins=25, color='steelblue', alpha=0.7,
                                edgecolor='black')
    ax.axvline(mean_rt, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_rt:.3f}s')
    ax.axvline(mean_rt - std_rt, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(mean_rt + std_rt, color='orange', linestyle=':', linewidth=2, alpha=0.7,
               label=f'±1 SD: {std_rt:.3f}s')

    ax.set_xlabel('Reaction Time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Reaction Time Distribution\nMean={mean_rt:.3f}s, SD={std_rt:.3f}s, CV={cv:.2f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation
    if cv < 0.3:
        interpretation = 'VOLUNTARY\n(consistent RT, low CV)'
        box_color = 'lightgreen'
    elif cv < 0.5:
        interpretation = 'Moderate consistency\n(medium CV)'
        box_color = 'yellow'
    else:
        interpretation = 'High variability\n(inconsistent, high CV)'
        box_color = 'lightcoral'

    ax.text(0.95, 0.95, interpretation, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    # Plot 2: Reaction time across trials (check for learning/fatigue)
    ax = axes[0, 1]
    trial_nums = np.arange(1, len(reaction_times) + 1)
    ax.plot(trial_nums, reaction_times, 'o', alpha=0.5, markersize=6)

    # Add trend line
    z = np.polyfit(trial_nums, reaction_times, 1)
    p = np.poly1d(z)
    ax.plot(trial_nums, p(trial_nums), 'r-', linewidth=2,
            label=f'Trend: {z[0]:+.5f}s/trial')
    ax.axhline(mean_rt, color='green', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Mean: {mean_rt:.3f}s')

    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Reaction Time (s)', fontsize=12)
    ax.set_title('Reaction Time Across Session', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Q-Q plot (test for normality)
    ax = axes[1, 0]
    scipy_stats.probplot(reaction_times, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Test)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Shapiro-Wilk test for normality
    if len(reaction_times) >= 3:
        shapiro_stat, shapiro_p = scipy_stats.shapiro(reaction_times)
        ax.text(0.05, 0.95, f'Shapiro-Wilk p={shapiro_p:.4f}\n' +
                ('Normal' if shapiro_p > 0.05 else 'Non-normal'),
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Cumulative distribution
    ax = axes[1, 1]
    sorted_rt = np.sort(reaction_times)
    cumulative = np.arange(1, len(sorted_rt) + 1) / len(sorted_rt)
    ax.plot(sorted_rt, cumulative, 'b-', linewidth=2)
    ax.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.3f}s')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(np.median(reaction_times), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(reaction_times):.3f}s')

    ax.set_xlabel('Reaction Time (s)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Overall title
    title = 'Test #7: Reaction Time Consistency (Voluntary Control Test)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    fig.suptitle(title, fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Save figure
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_test7_reaction_time.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        filename_svg = f"{prefix}saccade_feedback_test7_reaction_time.svg"
        fig.savefig(results_dir / filename_svg, bbox_inches='tight')
        print(f"Saved reaction time consistency plot to {results_dir / filename}")

    stats_dict = {
        'mean_rt': mean_rt,
        'std_rt': std_rt,
        'cv': cv,
        'median_rt': np.median(reaction_times),
        'n_trials': len(reaction_times)
    }

    print(f"\nTest #7: Reaction Time Consistency")
    print(f"  Mean RT: {mean_rt:.3f}s ± {std_rt:.3f}s")
    print(f"  Coefficient of Variation: {cv:.2f}")
    print(f"  Interpretation: {'VOLUNTARY (consistent)' if cv < 0.3 else 'High variability'}")

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

    print("\nGenerating trajectory plot by direction (left vs right)...")
    fig_traj_dir = plot_trajectories_by_direction(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_traj_dir)

    print("\nGenerating trajectory plot colored by time...")
    fig_traj_time = plot_trajectories_by_time(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_traj_time)

    print("\nShowing interactive trajectory viewer...")
    print("(Press SPACE to advance to next trial)")
    if show_plots:
        interactive_trajectories(trials, animal_id=animal_id, session_date=date_str)

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
    shuffle_results = shuffle_control_analysis(trials, n_shuffles=1000, seed=42)
    fig_shuffle = plot_shuffle_control(shuffle_results, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_shuffle)

    print("\nRunning left vs right target comparison...")
    fig_lr, lr_stats = compare_left_right_performance(trials, left_x=-0.7, right_x=0.7,
                                                       results_dir=results_dir,
                                                       animal_id=animal_id,
                                                       session_date=date_str)
    if fig_lr is not None:
        if show_plots:
            plt.show()
        plt.close(fig_lr)

    print("\nRunning additional voluntary control tests...")

    # Test #2: Initial Direction Correlation
    fig_test2, stats_test2 = test_initial_direction_correlation(trials, results_dir=results_dir,
                                                                 animal_id=animal_id,
                                                                 session_date=date_str)
    if show_plots:
        plt.show()
    plt.close(fig_test2)

    # Test #3: Trial-to-Trial Adaptation
    fig_test3, stats_test3 = test_trial_to_trial_adaptation(trials, results_dir=results_dir,
                                                             animal_id=animal_id,
                                                             session_date=date_str)
    if show_plots:
        plt.show()
    plt.close(fig_test3)

    # Test #6: Speed-Accuracy Tradeoff
    fig_test6, stats_test6 = test_speed_accuracy_tradeoff(trials, results_dir=results_dir,
                                                           animal_id=animal_id,
                                                           session_date=date_str)
    if show_plots:
        plt.show()
    plt.close(fig_test6)

    # Test #7: Reaction Time Consistency
    fig_test7, stats_test7 = test_reaction_time_consistency(trials, results_dir=results_dir,
                                                             animal_id=animal_id,
                                                             session_date=date_str)
    if show_plots:
        plt.show()
    plt.close(fig_test7)

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

    print("\nGenerating trajectory plot by direction (left vs right)...")
    fig_traj_dir = plot_trajectories_by_direction(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_traj_dir)

    print("\nGenerating trajectory plot colored by time...")
    fig_traj_time = plot_trajectories_by_time(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_traj_time)

    print("\nShowing interactive trajectory viewer...")
    print("(Press SPACE to advance to next trial)")
    interactive_trajectories(trials, animal_id=animal_id, session_date=date_str)

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
    shuffle_results = shuffle_control_analysis(trials, n_shuffles=1000, seed=42)
    fig_shuffle = plot_shuffle_control(shuffle_results, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_shuffle)

    print("\nRunning left vs right target comparison...")
    fig_lr, lr_stats = compare_left_right_performance(trials, left_x=-0.7, right_x=0.7,
                                                       results_dir=results_dir,
                                                       animal_id=animal_id,
                                                       session_date=date_str)
    if fig_lr is not None:
        plt.show()
        plt.close(fig_lr)

    print("\nRunning additional voluntary control tests...")

    # Test #2: Initial Direction Correlation
    fig_test2, stats_test2 = test_initial_direction_correlation(trials, results_dir=results_dir,
                                                                 animal_id=animal_id,
                                                                 session_date=date_str)
    plt.show()
    plt.close(fig_test2)

    # Test #3: Trial-to-Trial Adaptation
    fig_test3, stats_test3 = test_trial_to_trial_adaptation(trials, results_dir=results_dir,
                                                             animal_id=animal_id,
                                                             session_date=date_str)
    plt.show()
    plt.close(fig_test3)

    # Test #6: Speed-Accuracy Tradeoff
    fig_test6, stats_test6 = test_speed_accuracy_tradeoff(trials, results_dir=results_dir,
                                                           animal_id=animal_id,
                                                           session_date=date_str)
    plt.show()
    plt.close(fig_test6)

    # Test #7: Reaction Time Consistency
    fig_test7, stats_test7 = test_reaction_time_consistency(trials, results_dir=results_dir,
                                                             animal_id=animal_id,
                                                             session_date=date_str)
    plt.show()
    plt.close(fig_test7)

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
