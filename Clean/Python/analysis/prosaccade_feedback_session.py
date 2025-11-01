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

    # Load end of trial data
    # Columns: Frame, timestamp, trial_number, green_dot_x, green_dot_y, diameter
    eot_df = pd.read_csv(endoftrial_file, header=None,
                         names=['frame', 'timestamp', 'trial_number', 'green_x', 'green_y', 'diameter'])

    # Load eye position / green dot position data
    # Columns: Frame, timestamp, placeholder, green_dot_x, green_dot_y, diameter
    # Note: This file has duplicates that need to be cleaned
    eye_df = pd.read_csv(vstim_go_file, header=None,
                         names=['frame', 'timestamp', 'placeholder', 'green_x', 'green_y', 'diameter'])

    # Remove duplicate frame entries - keep only the first occurrence
    eye_df = eye_df.drop_duplicates(subset=['frame'], keep='first')
    eye_df = eye_df.sort_values('frame').reset_index(drop=True)

    # Load target position / blue dot position data
    # Columns: Frame, timestamp, target_x, target_y, diameter
    target_df = pd.read_csv(vstim_cue_file, header=None,
                            names=['frame', 'timestamp', 'target_x', 'target_y', 'diameter'])

    print(f"Loaded data from {folder_path}")
    print(f"  End of trial events: {len(eot_df)}")
    print(f"  Eye position samples: {len(eye_df)} (after deduplication)")
    print(f"  Target position samples: {len(target_df)}")

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

        trial_data = {
            'trial_number': trial_num,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'target_x': target_x,
            'target_y': target_y,
            'eye_x': eye_trajectory['green_x'].values,
            'eye_y': eye_trajectory['green_y'].values,
            'eye_times': eye_trajectory['timestamp'].values,
        }

        trials.append(trial_data)

    print(f"Extracted {len(trials)} valid trials out of {n_trials} total")
    return trials


def plot_trajectories(trials: list[dict], results_dir: Optional[Path] = None,
                      animal_id: Optional[str] = None, session_date: str = "") -> plt.Figure:
    """Plot eye position trajectories relative to target position.

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

    # Color map for trials
    cmap = plt.cm.viridis
    n_trials = len(trials)

    for i, trial in enumerate(trials):
        # Normalize positions relative to target (target at origin)
        rel_x = trial['eye_x'] - trial['target_x']
        rel_y = trial['eye_y'] - trial['target_y']

        # Plot trajectory with color indicating trial number
        color = cmap(i / max(1, n_trials - 1))
        ax.plot(rel_x, rel_y, alpha=0.6, linewidth=1.5, color=color,
                label=f"Trial {trial['trial_number']}" if n_trials <= 20 else None)

        # Mark start and end points
        ax.plot(rel_x[0], rel_y[0], 'o', color=color, markersize=6, alpha=0.8)
        ax.plot(rel_x[-1], rel_y[-1], 's', color=color, markersize=6, alpha=0.8)

    # Draw target position as black circle at origin
    target_circle = Circle((0, 0), radius=20, fill=False, edgecolor='black',
                          linewidth=2.5, linestyle='-', label='Target')
    ax.add_patch(target_circle)

    # Add smaller filled circle at center
    ax.plot(0, 0, 'ko', markersize=8, label='Target Center')

    ax.set_xlabel('Horizontal Position Relative to Target (stimulus units)', fontsize=12)
    ax.set_ylabel('Vertical Position Relative to Target (stimulus units)', fontsize=12)

    title = 'Eye Position Trajectories to Target'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.axis('equal')

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
    folder_path = Path(folder_path)
    if results_dir is None:
        results_dir = folder_path / "results"
    else:
        results_dir = Path(results_dir)

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

    print("\nGenerating time-to-target plot...")
    fig_time = plot_time_to_target(trials, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_time)

    # Create summary DataFrame
    durations = [t['duration'] for t in trials]

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
    })

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Folder: {folder_path}")
    print(f"Animal: {animal_id}")
    print(f"Date: {date_str}")
    print(f"Valid trials: {len(trials)}")
    print(f"Mean time to target: {np.mean(durations):.2f} ± {np.std(durations):.2f} s")
    print(f"Median time to target: {np.median(durations):.2f} s")
    print(f"Range: {np.min(durations):.2f} - {np.max(durations):.2f} s")
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

    print("\nGenerating time-to-target plot...")
    fig_time = plot_time_to_target(trials, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_time)

    # Create summary DataFrame
    durations = [t['duration'] for t in trials]

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
    })

    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Session: {session_id}")
    print(f"Animal: {animal_id}")
    print(f"Date: {date_str}")
    print(f"Valid trials: {len(trials)}")
    print(f"Mean time to target: {np.mean(durations):.2f} ± {np.std(durations):.2f} s")
    print(f"Median time to target: {np.median(durations):.2f} s")
    print(f"Range: {np.min(durations):.2f} - {np.max(durations):.2f} s")
    print("="*60)

    return df


# Usage:
# 1. With session manifest: python Clean/Python/analysis/prosaccade_feedback_session.py SESSION_ID
# 2. Direct folder: python Clean/Python/analysis/prosaccade_feedback_session.py --folder /path/to/data
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
