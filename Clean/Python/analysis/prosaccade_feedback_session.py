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
import csv
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
        eot_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1, dtype=float)

        # Check number of columns
        if eot_arr.ndim == 1:
            eot_arr = eot_arr.reshape(1, -1)

        n_cols = eot_arr.shape[1]
     

        # Handle different column formats
        # Third column (index 2) is always trial_success (2=success, 1=failed)
        if n_cols >= 3:
            # Modern format: frame, timestamp, trial_success, [optional trial_number, green_x, green_y, diameter]
            if n_cols == 7:
                eot_df = pd.DataFrame(eot_arr, columns=['frame', 'timestamp', 'trial_success', 'trial_number', 'green_x', 'green_y', 'diameter'])
            elif n_cols == 6:
                eot_df = pd.DataFrame(eot_arr, columns=['frame', 'timestamp', 'trial_success', 'green_x', 'green_y', 'diameter'])
                eot_df['diameter'] = 0.2  # Default diameter if not present
            else:
                # Flexible: just use first 2 and third column
                eot_df = pd.DataFrame({
                    'frame': eot_arr[:, 0],
                    'timestamp': eot_arr[:, 1],
                    'trial_success': eot_arr[:, 2]
                })
                print(f"  Using flexible column layout: frame, timestamp, trial_success, ...")
        else:
            raise ValueError(f"Unexpected number of columns: {n_cols}. Expected at least 3.")

        eot_df['frame'] = eot_df['frame'].astype(int)
        eot_df['trial_success'] = eot_df['trial_success'].astype(int)
        if 'trial_number' in eot_df.columns:
            eot_df['trial_number'] = eot_df['trial_number'].astype(int)

        print(f"  Loaded {len(eot_df)} end-of-trial events")
        if 'trial_success' in eot_df.columns:
            n_success = (eot_df['trial_success'] == 2).sum()
            n_failed = (eot_df['trial_success'] != 2).sum()
            print(f"  Trial success indicators: {n_success} successful, {n_failed} failed")
    except Exception as e:
        raise ValueError(f"Error loading end of trial file {endoftrial_file}: {e}")

    # Load eye position / green dot position data using standard approach
    # Read from end: last column is ignored, -2 is y, -3 is x
    # This generalizes across files with 5, 6, or more columns
    try:
        print(f"\nLoading {vstim_go_file.name}...")
        cleaned = clean_csv(str(vstim_go_file))
        eye_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1, dtype=float)

        # Check number of columns
        if eye_arr.ndim == 1:
            eye_arr = eye_arr.reshape(1, -1)

        n_cols = eye_arr.shape[1]
        print(f"  Detected {n_cols} columns in vstim_go file")

        if n_cols < 4:
            raise ValueError(f"Too few columns: {n_cols}. Expected at least 4 (frame, timestamp, x, y)")

        # Extract columns: frame, timestamp, x, y, diameter (5th column)
        # Note: Using positive indexing for diameter to get 5th column (index 4)
        eye_df = pd.DataFrame({
            'frame': eye_arr[:, 0],
            'timestamp': eye_arr[:, 1],
            'green_x': eye_arr[:, 2],
            'green_y': eye_arr[:, 3],
            'diameter': eye_arr[:, 4] if n_cols >= 5 else 0.2,
        })
        eye_df['frame'] = eye_df['frame'].astype(int)

        # Remove duplicate frame entries - keep only the first occurrence
        #before_dedup = len(eye_df)
        #eye_df = eye_df.drop_duplicates(subset=['frame'], keep='first')
        #after_dedup = len(eye_df)
        #if before_dedup != after_dedup:
            #print(f"  Removed {before_dedup - after_dedup} duplicate frame entries")

       # eye_df = eye_df.sort_values('frame').reset_index(drop=True)
        #print(f"  Loaded {len(eye_df)} eye position samples (after deduplication)")
    except Exception as e:
        raise ValueError(f"Error loading vstim_go file {vstim_go_file}: {e}")

    # Load target position / blue dot position data using standard approach
    # Columns: Frame, timestamp, target_x, target_y, diameter, [visible]
    try:
        print(f"\nLoading {vstim_cue_file.name}...")
        cleaned = clean_csv(str(vstim_cue_file))
        target_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1, dtype=float)

        # Check number of columns
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(1, -1)

        n_cols = target_arr.shape[1]
        print(f"  Detected {n_cols} columns in vstim_cue file")

        if n_cols == 6:
            # New format with visibility column (0 = invisible, 1 = visible)
            target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y', 'diameter', 'visible'])
            print(f"  Target visibility column detected")
        elif n_cols == 5:
            target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y', 'diameter'])
            target_df['visible'] = 1  # Default to visible if column not present
            print(f"  Warning: 'visible' column not found, assuming all targets are visible")
        else:
            raise ValueError(f"Unexpected number of columns: {n_cols}. Expected 4, 5, or 6.")

        target_df['frame'] = target_df['frame'].astype(int)
        target_df['visible'] = target_df['visible'].astype(int)
        target_df['diameter'] = target_df['diameter'].astype(float)

        # Detect and remove duplicate entries
        original_len = len(target_df)
        duplicates = target_df.duplicated(subset=['frame'], keep='first')
        n_duplicates = duplicates.sum()

        if n_duplicates > 0:
            print(f"  Warning: Found {n_duplicates} duplicate entries in vstim_cue (based on frame number)")
            # Show some examples of duplicates
            dup_frames = target_df[duplicates]['frame'].values[:5]  # Show up to 5 examples
            print(f"    Example duplicate frames: {dup_frames}")
            # Remove duplicates, keeping the first occurrence
            target_df = target_df[~duplicates].reset_index(drop=True)
            print(f"    Removed {n_duplicates} duplicates, kept first occurrence of each")

        print(f"  Loaded {len(target_df)} target position samples (after removing duplicates)")
        n_invisible = len(target_df[target_df['visible'] == 0])
        if n_invisible > 0:
            print(f"  Found {n_invisible} invisible targets")
    except Exception as e:
        raise ValueError(f"Error loading vstim_cue file {vstim_cue_file}: {e}")


    if len(eot_df) > 1:
        duration_example = eot_df.iloc[1]['timestamp'] - eot_df.iloc[0]['timestamp']


    return eot_df, eye_df, target_df


def identify_and_filter_failed_trials(target_df: pd.DataFrame, eot_df: pd.DataFrame,
                                      exclude_failed: bool = True) -> Tuple[pd.DataFrame, list, list]:
    """Identify failed trials using the trial_success column in end_of_trial data.

    The trial_success column in eot_df indicates: 1=success, 0=failed.
    Each eot_df row corresponds to one trial (both successful and failed).

    Parameters
    ----------
    target_df : pd.DataFrame
        Target/cue position data (all trials including failed ones)
    eot_df : pd.DataFrame
        End of trial data (all trials with trial_success indicator)
    exclude_failed : bool
        If True, filter out failed trials from target_df. If False, keep all trials.

    Returns
    -------
    filtered_target_df : pd.DataFrame
        Target dataframe with failed trials removed (if exclude_failed=True)
    failed_indices : list
        List of indices of failed trials (in original target_df indexing)
    successful_indices : list
        List of indices of successful trials (in original target_df indexing)
    """
    if len(target_df) == 0 or len(eot_df) == 0:
        print("\nWarning: Empty target or end-of-trial data, cannot identify failed trials")
        return target_df, [], []

    # Check if trial_success column exists
    if 'trial_success' not in eot_df.columns:
        print("\nWarning: trial_success column not found in eot_df, assuming all trials successful")
        successful_indices = list(range(len(target_df)))
        failed_indices = []
    else:
        # Match target_df (cue events) with eot_df by timestamp
        # Assuming they are in chronological order and 1:1 correspondence
        target_df = target_df.sort_values('timestamp').reset_index(drop=True)
        eot_df = eot_df.sort_values('timestamp').reset_index(drop=True)

        # Get trial success flags from eot_df
        # eot_df should have one entry per trial (including failed ones)
        trial_success_flags = eot_df['trial_success'].values


        # Identify successful and failed indices
        successful_indices = []
        failed_indices = []

        n_trials = min(len(target_df), len(eot_df))
        for idx in range(n_trials):
            if idx < len(trial_success_flags) and trial_success_flags[idx] == 2:
                successful_indices.append(idx)
            else:
                failed_indices.append(idx)

    # Report trial statistics
    n_total = len(target_df)
    n_success = len(successful_indices)
    n_failed = len(failed_indices)

    print(f"\n{'='*60}")
    print(f"Trial Summary (from endoftrial trial_success column):")
    print(f"  Total trials (from cue events): {n_total}")
    pct_success = 100*n_success/n_total if n_total > 0 else 0
    pct_failed = 100*n_failed/n_total if n_total > 0 else 0
    print(f"  Successful trials: {n_success} ({pct_success:.1f}%)")
    print(f"  Failed trials: {n_failed} ({pct_failed:.1f}%)")
    if n_failed > 0:
        print(f"  Failed trial indices: {failed_indices}")
    print(f"  exclude_failed_trials: {exclude_failed}")
    if exclude_failed:
        print(f"  → Only successful trials will be included in analysis")
    else:
        print(f"  → All trials (including failed) will be included in analysis")
    print(f"{'='*60}\n")

    # Filter if requested
    if exclude_failed and n_success > 0:
        filtered_target_df = target_df.iloc[successful_indices].reset_index(drop=True)
        print(f"Filtered target_df from {len(target_df)} to {len(filtered_target_df)} trials")
        return filtered_target_df, failed_indices, successful_indices
    else:
        return target_df, failed_indices, successful_indices


def extract_trial_trajectories(eot_df: pd.DataFrame, eye_df: pd.DataFrame,
                                target_df: pd.DataFrame,
                                successful_indices: Optional[list] = None) -> list[dict]:
    """Extract eye position trajectories for each trial.

    Trial timing is calculated from vstim_cue (target_df):
    - Each trial starts at the timestamp/frame from vstim_cue
    - Inter-trial interval (ITI) is inferred by finding the minimum difference
      between consecutive vstim_cue timestamps, rounded down to nearest whole second
    - Trial end: trial_end(i) = trial_start(i+1) - ITI
    - Last trial uses end_of_trial data if available

    Parameters
    ----------
    eot_df : pd.DataFrame
        End of trial data (used only for last trial if available)
    eye_df : pd.DataFrame
        Eye position data (cleaned, no duplicates)
    target_df : pd.DataFrame
        Target position data from vstim_cue

    Returns
    -------
    list of dict
        List of trial dictionaries containing trajectory and metadata
    """
    trials = []
    n_trials = len(target_df)

    # Calculate inter-trial interval (ITI) from vstim_cue timestamps
    # ITI = minimum difference between consecutive target presentations, rounded down
    if n_trials > 1:
        time_diffs = np.diff(target_df['timestamp'].values)
        min_diff = np.min(time_diffs)
        ITI = np.floor(min_diff)  # Round down to nearest whole second
        print(f"\nCalculated ITI (inter-trial interval): {ITI:.0f} seconds")
        print(f"  (from minimum difference in vstim_cue: {min_diff:.3f}s)")
    else:
        ITI = 0
        print(f"\nWarning: Only 1 trial found, ITI set to 0")

    for i in range(n_trials):
        # Use original trial number if available, otherwise sequential numbering
        if 'original_trial_number' in target_df.columns:
            trial_num = int(target_df.iloc[i]['original_trial_number'])
        else:
            trial_num = i + 1

        # Trial starts at target onset (vstim_cue)
        target_x = target_df.iloc[i]['target_x']
        target_y = target_df.iloc[i]['target_y']
        target_diameter = target_df.iloc[i]['diameter']
        target_visible = target_df.iloc[i]['visible']
        start_frame = target_df.iloc[i]['frame']
        start_time = target_df.iloc[i]['timestamp']

        # Calculate trial end time using end_of_trial data
        # Since target_df has been filtered to only successful trials,
        # it should align 1:1 with eot_df
        if i < len(eot_df):
            end_frame = int(eot_df.iloc[i]['frame'])
            end_time = eot_df.iloc[i]['timestamp']
        else:
            # Fallback if eot_df doesn't have this trial (shouldn't happen after filtering)
            print(f"Warning: No end_of_trial data for trial {trial_num}, using next trial start - ITI")
            # if i < n_trials - 1:
            #     next_start_time = target_df.iloc[i+1]['timestamp']
            #     next_start_frame = target_df.iloc[i+1]['frame']
            #     end_time = next_start_time - ITI
            #     if start_time != end_time:
            #         frame_rate = (next_start_frame - start_frame) / (next_start_time - start_time)
            #         end_frame = int(start_frame + (end_time - start_time) * frame_rate)
            #     else:
            #         end_frame = start_frame
            # else:
            #     end_time = start_time + ITI
            #     end_frame = start_frame + 1000

        # Extract eye position trajectory for this trial
        # FIXED: Now starts from target onset, excluding inter-trial interval
        eye_mask = (eye_df['frame'] >= start_frame) & (eye_df['frame'] <= end_frame)
        eye_trajectory = eye_df[eye_mask]

        # Drop any rows with NA values in position data
        eye_trajectory = eye_trajectory.dropna(subset=['green_x', 'green_y', 'timestamp'])

        # Handle trials with no eye data - create placeholder instead of skipping
        has_eye_data = len(eye_trajectory) > 0

        if not has_eye_data:
            print(f"Warning: No eye data for trial {trial_num}, creating placeholder")
            # Create placeholder values for trials with no eye data
            start_eye_x = np.nan
            start_eye_y = np.nan
            final_eye_x = np.nan
            final_eye_y = np.nan
            final_eye_frame = np.nan
            eye_duration = 0.0
            path_length = 0.0
            path_efficiency = 0.0
            straight_line_distance = 0.0
            initial_direction_error = np.nan
            eye_times_raw = np.array([])
            eye_x_full = np.array([])
            eye_y_full = np.array([])
            eye_times_full = np.array([])
            eye_frames_full = np.array([])
            eye_start_time = start_time
            eye_end_time = end_time
            cursor_diameter = 0.2  # Default cursor diameter
        else:
            # Calculate path length (cumulative distance along trajectory)
            start_eye_x = eye_trajectory['green_x'].values[0]
            start_eye_y = eye_trajectory['green_y'].values[0]
            eye_times_raw = eye_trajectory['timestamp'].values
            eye_start_time = eye_times_raw[0]
            eye_end_time = eye_times_raw[-1]
            eye_duration = eye_end_time - eye_start_time

            # Extract cursor diameter (use first value from eye_trajectory)
            cursor_diameter = eye_trajectory['diameter'].values[0]

            # Extract frame numbers from eye_trajectory
            eye_frames_raw = eye_trajectory['frame'].values

            # OPTION 2: Use the next row after the last position within trial window
            # Get the last row within the trial window
            last_within_trial_idx = eye_trajectory.index[-1]
            last_within_trial_frame = eye_trajectory['frame'].values[-1]

            # Find this index in the full eye_df and get the next row
            eye_df_position = eye_df.index.get_loc(last_within_trial_idx)

            if eye_df_position + 1 < len(eye_df):
                # Get the next row after the last position within trial
                next_row = eye_df.iloc[eye_df_position + 0]
                final_eye_x = next_row['green_x']
                final_eye_y = next_row['green_y']
                final_eye_frame = int(next_row['frame'])
            else:
                # If there's no next row, use the last position within trial
                final_eye_x = eye_trajectory['green_x'].values[-1]
                final_eye_y = eye_trajectory['green_y'].values[-1]
                final_eye_frame = int(eye_trajectory['frame'].values[-1])


            # Append the final position to trajectory arrays for continuous plotting
            # This ensures no gap/jump between trajectory and final position marker
            eye_x_full = np.append(eye_trajectory['green_x'].values, final_eye_x)
            eye_y_full = np.append(eye_trajectory['green_y'].values, final_eye_y)
            # Also append timestamp and frame for the final position
            if eye_df_position + 1 < len(eye_df):
                final_timestamp = eye_df.iloc[eye_df_position + 1]['timestamp']
                eye_times_full = np.append(eye_times_raw, final_timestamp)
                eye_frames_full = np.append(eye_frames_raw, final_eye_frame)
            else:
                eye_times_full = eye_times_raw
                eye_frames_full = eye_frames_raw

        if has_eye_data and len(eye_trajectory) > 1:
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
                cos_angle = (initial_dx * ideal_dx + initial_dy * initial_dy) / (initial_mag * ideal_mag)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                initial_direction_error = np.degrees(np.arccos(cos_angle))
            else:
                initial_direction_error = np.nan
        elif has_eye_data:
            # Single point trajectory
            path_length = 0.0
            path_efficiency = 0.0
            straight_line_distance = 0.0
            initial_direction_error = np.nan

        # Sanity check: trial duration should not exceed 15 seconds (timeout)
        if has_eye_data and eye_duration > 15.0:
            print(f"WARNING: Trial {trial_num} has duration {eye_duration:.2f}s (> 15s timeout)")
            print(f"  start_time={start_time:.2f}, end_time={end_time:.2f}, duration={end_time-start_time:.2f}s")
            print(f"  eye_start_time={eye_start_time:.2f}, eye_end_time={eye_end_time:.2f}, eye_duration={eye_duration:.2f}s")

        # Determine if this trial was successful or failed
        # If successful_indices is provided, check if current index is in it
        trial_failed = False
        if successful_indices is not None:
            trial_failed = i not in successful_indices

        trial_data = {
            'trial_number': trial_num,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': eye_duration,  # Use eye trajectory duration for consistency
            'target_x': target_x,
            'target_y': target_y,
            'target_diameter': target_diameter,
            'target_visible': target_visible,
            'cursor_diameter': cursor_diameter,
            'start_eye_x': start_eye_x,
            'start_eye_y': start_eye_y,
            'final_eye_x': final_eye_x,
            'final_eye_y': final_eye_y,
            'final_eye_frame': final_eye_frame,
            'eye_x': eye_x_full if has_eye_data else np.array([]),
            'eye_y': eye_y_full if has_eye_data else np.array([]),
            'eye_times': eye_times_full if has_eye_data else np.array([]),
            'eye_frames': eye_frames_full if has_eye_data else np.array([]),
            'eye_start_time': eye_start_time,  # For relative time calculations
            'path_length': path_length,
            'straight_line_distance': straight_line_distance,
            'path_efficiency': path_efficiency,
            'initial_direction_error': initial_direction_error,
            'trial_failed': trial_failed,
            'has_eye_data': has_eye_data,
        }

        trials.append(trial_data)

    print(f"\nExtracted {len(trials)} valid trials out of {n_trials} total")
    if len(trials) > 0:
        print(f"  First trial duration: {trials[0]['duration']:.2f}s")
        if len(trials) > 1:
            print(f"  Second trial duration: {trials[1]['duration']:.2f}s")
        print(f"  Mean trial duration: {np.mean([t['duration'] for t in trials]):.2f}s")

        print(f"\n  Starting eye positions:")
        for i, trial in enumerate(trials[:5]):  # Show first 5 trials
            print(f"    Trial {trial['trial_number']}: ({trial['start_eye_x']:.3f}, {trial['start_eye_y']:.3f})")
        if len(trials) > 5:
            print(f"    ... (showing first 5 of {len(trials)} trials)")

        print(f"\n  Final eye positions (from row after last position within trial):")
        for i, trial in enumerate(trials[:5]):  # Show first 5 trials
            if not np.isnan(trial['final_eye_x']):
                print(f"    Trial {trial['trial_number']}: ({trial['final_eye_x']:.3f}, {trial['final_eye_y']:.3f}) at frame {trial['final_eye_frame']}")
            else:
                print(f"    Trial {trial['trial_number']}: No final position data")
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
    ax.set_xlim(-1.7, 1.7)
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

    # Pre-draw all targets (static) in gray - neutral visibility
    # We'll show correct visibility when highlighting the active trial's target
    target_circles = {}  # Store target circles by position
    for trial in trials:
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        key = (round(target_x, 2), round(target_y, 2))

        if key not in target_circles:
            # Draw all targets as gray circles (no visibility indication yet)
            target_circle = Circle((target_x, target_y), radius=target_radius,
                                  fill=False, edgecolor='gray', linewidth=1.5,
                                  linestyle='-', alpha=0.3)
            ax.add_patch(target_circle)
            ax.plot(target_x, target_y, 'o', color='gray', markersize=3, alpha=0.3)
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
        target_visible = trial.get('target_visible', 1)
        trial_failed = trial.get('trial_failed', False)

        # Color scheme: RED for failed trials, regular colormap for successful trials
        if trial_failed:
            color = 'red'
            target_color = 'red'
            target_edge_color = 'darkred'
        else:
            color = cmap(trial_idx / max(1, n_trials - 1))
            target_color = 'green'
            target_edge_color = 'darkgreen'

        # Highlight current target with CORRECT visibility for this trial
        # Use solid line for visible, dashed for invisible
        linestyle = '-' if target_visible else '--'
        target_circle_green = Circle((target_x, target_y), radius=target_radius,
                                     fill=True, facecolor=target_color, edgecolor=target_edge_color,
                                     linewidth=3, linestyle=linestyle, alpha=0.3)
        ax.add_patch(target_circle_green)

        # Draw center marker - filled for visible, hollow for invisible
        if target_visible:
            target_dot_green, = ax.plot(target_x, target_y, 'o', color=target_color, markersize=8,
                                        markeredgecolor=target_edge_color, markeredgewidth=2)
        else:
            target_dot_green, = ax.plot(target_x, target_y, 'o', color=target_color, markersize=8,
                                        markerfacecolor='none', markeredgecolor=target_edge_color,
                                        markeredgewidth=2)
        current_target_highlight = [target_circle_green, target_dot_green]

        # Plot trajectory (only if eye data exists)
        has_eye_data = trial.get('has_eye_data', True)
        if has_eye_data and len(eye_x) > 0:
            line, = ax.plot(eye_x, eye_y, '-', color=color, linewidth=2, alpha=0.7)

            start, = ax.plot(eye_x[0], eye_y[0], 'o', color=color,
                            markersize=10, markeredgecolor='white',
                            markeredgewidth=2, alpha=0.9, label='Start')
            # Draw end position using calculated final_eye_x/y (from vstim_go next row)
            # NOT the last trajectory point
            final_x = trial.get('final_eye_x', eye_x[-1])
            final_y = trial.get('final_eye_y', eye_y[-1])
            end_circle = Circle((final_x, final_y), radius=0.1, fill=True,
                               facecolor=color, edgecolor='white', linewidth=2, alpha=0.9,
                               label='End' if trial_idx == 0 else None)
            ax.add_patch(end_circle)

            trial_lines.extend([line, start, end_circle])

        # Update progress text
        target_dir = 'Left' if trial['target_x'] < 0 else 'Right'
        trial_status = " [FAILED]" if trial_failed else ""
        no_data_status = " [NO EYE DATA]" if not has_eye_data else ""
        progress_text.set_text(
            f"Trial {trial['trial_number']}{trial_status}{no_data_status} (showing {trial_idx + 1}/{n_trials})\n"
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

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}saccade_feedback_heatmap.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
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
        print(f"Saved time-to-target plot to {results_dir / filename}")

    return fig


def detect_fixations(eye_x: np.ndarray, eye_y: np.ndarray, eye_times: np.ndarray,
                     min_fixation_duration: float = 0.65,
                     max_movement: float = 0.1) -> list[tuple]:
    """Detect fixation windows based on frame-to-frame movement velocity.

    A fixation is a period where consecutive frame-to-frame movements are below
    the max_movement threshold and the total duration meets the minimum.

    Parameters
    ----------
    eye_x : np.ndarray
        X positions of eye trajectory
    eye_y : np.ndarray
        Y positions of eye trajectory
    eye_times : np.ndarray
        Timestamps for each position
    min_fixation_duration : float
        Minimum duration (seconds) for a valid fixation (default: 0.65)
    max_movement : float
        Maximum frame-to-frame movement for fixation (default: 0.1 units)

    Returns
    -------
    list of tuples
        Each tuple is (start_idx, end_idx, duration, span) where:
        - start_idx: Starting index in the arrays
        - end_idx: Ending index (exclusive)
        - duration: Total time from start to end
        - span: Spatial extent of the fixation
    """
    if len(eye_x) < 2:
        return []

    n = len(eye_x)
    fixations = []
    i = 0

    while i < n:
        # Try to extend a fixation starting at point i
        j = i + 1

        # Extend while frame-to-frame movement is below threshold
        while j < n:
            # Calculate movement from point j-1 to point j
            dx = eye_x[j] - eye_x[j-1]
            dy = eye_y[j] - eye_y[j-1]
            movement = np.sqrt(dx**2 + dy**2)

            if movement < max_movement:
                j += 1  # Include point j in the fixation
            else:
                break  # Movement too large, stop before point j

        # Now we have a potential fixation from index i to j (exclusive end)
        # This includes points [i, i+1, ..., j-1]

        if j > i + 1:  # At least 2 points
            duration = eye_times[j-1] - eye_times[i]
            if duration >= min_fixation_duration:
                # Valid fixation! Calculate span for informational purposes
                fix_x = eye_x[i:j]
                fix_y = eye_y[i:j]
                x_range = np.max(fix_x) - np.min(fix_x)
                y_range = np.max(fix_y) - np.min(fix_y)
                span = np.sqrt(x_range**2 + y_range**2)
                fixations.append((i, j, duration, span))
                i = j  # Start next search after this fixation
            else:
                i += 1  # Duration too short, try next starting point
        else:
            i += 1  # No valid extension, try next starting point

    return fixations


def calculate_trial_success_from_fixations(eye_x: np.ndarray, eye_y: np.ndarray,
                                          eye_times: np.ndarray,
                                          target_x: float, target_y: float,
                                          contact_threshold: float,
                                          min_fixation_duration: float = 0.65,
                                          max_movement: float = 0.1) -> tuple[bool, float]:
    """Determine trial success based on the last fixation.

    Success criterion:
    - Detect fixations using frame-to-frame movement threshold
    - Use ONLY the LAST fixation (since each trial should have only one fixation)
    - Check if this last fixation ENDS on target (last point within contact_threshold)
    - Success if the last fixation ends on target and duration >= min_fixation_duration

    NOTE: In prosaccade trials, there should ideally be only one fixation:
    - A fixation inside the target should end the trial with success
    - A fixation outside the target should end the trial with failure
    - If multiple fixations are detected, we use only the last one

    Parameters
    ----------
    eye_x, eye_y, eye_times : np.ndarray
        Eye trajectory data
    target_x, target_y : float
        Target position
    contact_threshold : float
        Distance threshold for being "on target" (target_radius + cursor_radius)
    min_fixation_duration : float
        Minimum fixation duration for success (default: 0.65s)
    max_movement : float
        Maximum movement for fixation detection (default: 0.1 units)

    Returns
    -------
    tuple of (success, fixation_duration)
        success : bool
            Whether trial succeeded
        fixation_duration : float
            Duration of last fixation if it ends on target (0.0 if none or off-target)
    """
    # Detect all fixations
    fixations = detect_fixations(eye_x, eye_y, eye_times,
                                 min_fixation_duration, max_movement)

    # Use only the LAST fixation
    if len(fixations) == 0:
        return False, 0.0

    # Get the last fixation
    start_idx, end_idx, duration, span = fixations[-1]

    # Check if the last fixation ends on target
    final_x = eye_x[end_idx - 1]
    final_y = eye_y[end_idx - 1]
    dist = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)

    if dist <= contact_threshold:
        # Last fixation ends on target, success if duration meets requirement
        success = (duration >= min_fixation_duration)
        return success, duration
    else:
        # Last fixation does not end on target
        return False, 0.0


def calculate_chance_level(trials: list[dict], n_shuffles: int = 10000,
                           target_filter: Optional[callable] = None,
                           min_fixation_duration: float = 0.65,
                           max_movement: float = 0.1,
                           results_dir: Optional[Path] = None) -> float:
    """Calculate chance level success rate by shuffling target positions.

    Shuffles target positions randomly across trials and calculates what the
    success rate would be if targets were at those shuffled positions, using
    the actual fixation-based success criterion.

    For subset analysis (e.g., left targets only):
    - Filters trials to get specific eye trajectories (e.g., from left-target trials)
    - But shuffles among ALL target positions from the full dataset
    - This asks: "For these specific trials, what if targets were randomly placed anywhere?"

    Success criterion (same as actual trials):
    - Detects fixations when frame-to-frame movement < max_movement
    - A fixation counts if it ENDS on the shuffled target
    - Success if any fixation ending on target has duration >= min_fixation_duration

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries with eye trajectory and target information
    n_shuffles : int
        Number of random shuffles to perform (default: 10000)
    target_filter : callable, optional
        Optional function to filter which trials' eye trajectories to use.
        Note: Shuffles among ALL target positions regardless of filter.
    min_fixation_duration : float
        Minimum fixation duration required for success (default: 0.65 seconds)
    max_movement : float
        Maximum frame-to-frame movement for fixation detection (default: 0.1 units)

    Returns
    -------
    float
        Average success rate across all shuffles (as fraction, not percentage)
    """
    if not trials or len(trials) == 0:
        return 0.0

    # IMPORTANT: Extract ALL target positions BEFORE filtering
    # This ensures we shuffle among all possible positions, not just the filtered subset
    all_target_positions = np.array([(t['target_x'], t['target_y']) for t in trials])
    all_target_diameters = np.array([t['target_diameter'] for t in trials])
    all_cursor_diameters = np.array([t.get('cursor_diameter', 0.2) for t in trials])

    # Remove any duplicate positions and their associated sizes for shuffling pool
    # (we only need unique positions to shuffle from)
    unique_positions = []
    unique_diameters = []
    unique_cursor_diams = []
    seen = set()
    for i, (tx, ty) in enumerate(all_target_positions):
        key = (round(tx, 3), round(ty, 3))
        if key not in seen:
            seen.add(key)
            unique_positions.append((tx, ty))
            unique_diameters.append(all_target_diameters[i])
            unique_cursor_diams.append(all_cursor_diameters[i])

    shuffle_pool_positions = np.array(unique_positions)
    shuffle_pool_diameters = np.array(unique_diameters)
    n_unique_positions = len(shuffle_pool_positions)

    # NOW apply filter to get specific trials (eye trajectories) to test
    filtered_trials = trials
    if target_filter is not None:
        filtered_trials = [t for t in trials if target_filter(t)]

    if len(filtered_trials) == 0:
        return 0.0

    # Filter out trials without eye data
    valid_trials = []
    for trial in filtered_trials:
        if trial.get('has_eye_data', False):
            eye_x = np.array(trial.get('eye_x', []))
            eye_y = np.array(trial.get('eye_y', []))
            eye_times = np.array(trial.get('eye_times', []))
            cursor_diam = trial.get('cursor_diameter', 0.2)
            if len(eye_x) > 0 and len(eye_times) > 0:
                valid_trials.append({
                    'eye_x': eye_x,
                    'eye_y': eye_y,
                    'eye_times': eye_times,
                    'cursor_diameter': cursor_diam,
                    'target_x': trial['target_x'],
                    'target_y': trial['target_y'],
                    'target_diameter': trial['target_diameter']
                })

    if len(valid_trials) == 0:
        return 0.0

    n_valid = len(valid_trials)
    success_rates = []

    # Only write CSV for all trials (no filter applied)
    write_csv = (target_filter is None and results_dir is not None)
    csv_writer = None
    csvfile = None

    if write_csv:
        csv_path = results_dir / 'chance_level_trials.csv'
        csvfile = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['trial_number', 'fixation_ended_on', 'actual_target', 'shuffled_target'])

    for shuffle_idx in range(n_shuffles):
        # Shuffle among ALL unique target positions, not just filtered ones
        # Sample WITH replacement to match number of trials
        random_indices = np.random.choice(n_unique_positions, size=n_valid, replace=True)
        shuffled_targets = shuffle_pool_positions[random_indices]

        # Calculate success for this shuffle by comparing target positions
        n_success = 0
        for i in range(n_valid):
            # Get actual and shuffled target positions
            actual_target_x = valid_trials[i]['target_x']
            actual_target_y = valid_trials[i]['target_y']
            shuffled_x, shuffled_y = shuffled_targets[i]

            # Determine sides
            actual_target_side = 'left' if actual_target_x < 0 else 'right'
            shuffled_target_side = 'left' if shuffled_x < 0 else 'right'

            # Success if shuffled target matches actual target position
            if actual_target_side == shuffled_target_side:
                n_success += 1

            # Only write to CSV for the first shuffle and when enabled
            if write_csv and shuffle_idx == 0:
                eye_x = valid_trials[i]['eye_x']
                eye_y = valid_trials[i]['eye_y']
                eye_times = valid_trials[i]['eye_times']
                cursor_radius = valid_trials[i]['cursor_diameter'] / 2.0

                # Determine where the last fixation ended by detecting fixations
                fixations = detect_fixations(eye_x, eye_y, eye_times,
                                            min_fixation_duration, max_movement)

                if len(fixations) > 0:
                    # Get the last fixation's ending position
                    start_idx, end_idx, duration, span = fixations[-1]
                    final_x = eye_x[end_idx - 1]
                    final_y = eye_y[end_idx - 1]

                    # Check which target the last fixation ended on
                    # Find all left and right target positions from shuffle pool
                    left_targets = [(tx, ty, shuffle_pool_diameters[idx])
                                   for idx, (tx, ty) in enumerate(shuffle_pool_positions) if tx < 0]
                    right_targets = [(tx, ty, shuffle_pool_diameters[idx])
                                    for idx, (tx, ty) in enumerate(shuffle_pool_positions) if tx >= 0]

                    # Check if fixation ended on left target
                    on_left = False
                    if left_targets:
                        left_x, left_y, left_diam = left_targets[0]
                        left_dist = np.sqrt((final_x - left_x)**2 + (final_y - left_y)**2)
                        left_threshold = (left_diam / 2.0) + cursor_radius
                        on_left = (left_dist <= left_threshold)

                    # Check if fixation ended on right target
                    on_right = False
                    if right_targets:
                        right_x, right_y, right_diam = right_targets[0]
                        right_dist = np.sqrt((final_x - right_x)**2 + (final_y - right_y)**2)
                        right_threshold = (right_diam / 2.0) + cursor_radius
                        on_right = (right_dist <= right_threshold)

                    # Determine which side the fixation ended on
                    if on_left and not on_right:
                        fixation_side = 'left'
                    elif on_right and not on_left:
                        fixation_side = 'right'
                    elif on_left and on_right:
                        # If on both (shouldn't happen), use closest
                        fixation_side = 'left' if left_dist < right_dist else 'right'
                    else:
                        # Not on either target, use position
                        fixation_side = 'left' if final_x < 0 else 'right'
                else:
                    # No fixations detected, use last position
                    fixation_side = 'left' if eye_x[-1] < 0 else 'right'

                # Write trial data
                csv_writer.writerow([i + 1, fixation_side, actual_target_side, shuffled_target_side])

        # Calculate success rate for this shuffle
        success_rate = n_success / n_valid if n_valid > 0 else 0.0
        success_rates.append(success_rate)

    if csvfile:
        csvfile.close()

    # Return average success rate across all shuffles
    return np.mean(success_rates)


def plot_trial_success(eot_df: pd.DataFrame, results_dir: Optional[Path] = None,
                       animal_id: Optional[str] = None, session_date: str = "",
                       trials: Optional[list[dict]] = None) -> plt.Figure:
    """Plot trial success vs failure summary, independent of --include-failed-trials flag.

    Creates a figure with:
    - Top: Bar chart showing fraction of successful vs failed trials
    - Bottom: Time-series showing trial success/failure for each trial

    Parameters
    ----------
    eot_df : pd.DataFrame
        End-of-trial dataframe containing all trials with 'trial_success' column
        (2 = success, other values = failed)
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
    if 'trial_success' not in eot_df.columns:
        print("Warning: trial_success column not found in eot_df, cannot plot trial success")
        return None

    # Get trial success values (2 = success, other = failed)
    trial_success = eot_df['trial_success'].values
    n_trials = len(trial_success)
    trial_numbers = np.arange(1, n_trials + 1)

    # Calculate success/failure counts
    is_success = trial_success == 2
    n_success = np.sum(is_success)
    n_failed = n_trials - n_success
    pct_success = 100 * n_success / n_trials if n_trials > 0 else 0
    pct_failed = 100 * n_failed / n_trials if n_trials > 0 else 0

    # Calculate chance level if trials data is provided
    chance_level = None
    if trials is not None and len(trials) > 0:
        print("  Calculating chance level (1000 shuffles)...")
        chance_level = calculate_chance_level(trials, n_shuffles=1000, results_dir=results_dir)
        print(f"  Chance level: {100*chance_level:.1f}%")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.5])

    # --- Top plot: Bar chart showing fraction of success vs failure ---
    if chance_level is not None:
        categories = ['Success', 'Failed', 'Chance']
        counts = [n_success, n_failed, chance_level * n_trials]
        percentages = [pct_success, pct_failed, 100 * chance_level]
        colors = ['forestgreen', 'firebrick', 'gray']
    else:
        categories = ['Success', 'Failed']
        counts = [n_success, n_failed]
        percentages = [pct_success, pct_failed]
        colors = ['forestgreen', 'firebrick']

    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        if chance_level is not None and bar == bars[-1]:  # Chance bar
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Number of Trials', fontsize=12)
    title = 'Trial Success Rate (All Trials)'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.2)  # Leave room for labels
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Bottom plot: Time-series of trial success/failure ---
    # Plot each trial as a colored point/bar
    success_trials = trial_numbers[is_success]
    failed_trials = trial_numbers[~is_success]

    # Use scatter plot with different colors
    ax2.scatter(success_trials, np.ones(len(success_trials)),
                c='forestgreen', s=50, marker='o', label=f'Success (n={n_success})',
                edgecolors='darkgreen', linewidths=0.5)
    ax2.scatter(failed_trials, np.zeros(len(failed_trials)),
                c='firebrick', s=50, marker='x', label=f'Failed (n={n_failed})',
                linewidths=2)

    # Add connecting lines showing the sequence
    for i in range(n_trials):
        y_val = 1 if is_success[i] else 0
        color = 'forestgreen' if is_success[i] else 'firebrick'
        ax2.vlines(trial_numbers[i], 0.5, y_val, colors=color, alpha=0.3, linewidth=1)

    # Add horizontal reference lines
    ax2.axhline(1, color='forestgreen', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axhline(0, color='firebrick', linestyle='--', alpha=0.3, linewidth=1)

    # Calculate and plot running success rate
    ax2_twin = ax2.twinx()
    running_success = np.cumsum(is_success) / trial_numbers * 100
    ax2_twin.plot(trial_numbers, running_success, 'b-', linewidth=2, alpha=0.7,
                  label='Running success rate')

    ax2_twin.set_ylabel('Running Success Rate (%)', fontsize=11, color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')
    ax2_twin.set_ylim(0, 105)
    ax2_twin.legend(loc='upper right', fontsize=9)

    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Trial Outcome', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Failed', 'Success'])
    ax2.set_xlim(0, n_trials + 1)
    ax2.set_ylim(-0.3, 1.5)
    ax2.set_title('Trial Outcomes Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}trial_success_summary.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"Saved trial success summary to {results_dir / filename}")

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
        print(f"Saved path length plot to {results_dir / filename}")

    return fig


def plot_final_positions_by_target(trials: list[dict], min_duration: float = 0.01, max_duration: float = 15.0,
                                   results_dir: Optional[Path] = None, animal_id: Optional[str] = None,
                                   session_date: str = "") -> plt.Figure:
    """Plot final cursor positions grouped by target position.

    Shows the last sample position for each trial, grouped by target location.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    min_duration : float
        Minimum trial duration in seconds (default: 0.1)
    max_duration : float
        Maximum trial duration in seconds (default: 10.0)
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
    from collections import defaultdict

    # Filter trials by duration
    filtered_trials = [t for t in trials if min_duration <= t['duration'] <= max_duration]
    n_excluded = len(trials) - len(filtered_trials)

    print(f"\nFinal Position Analysis:")
    print(f"  Total trials: {len(trials)}")
    print(f"  Excluded trials (duration < {min_duration}s or > {max_duration}s): {n_excluded}")
    print(f"  Trials included in analysis: {len(filtered_trials)}")

    if len(filtered_trials) == 0:
        print("  Warning: No trials left after filtering!")
        return None

    # Group trials by target position only (ignore visibility)
    target_groups = defaultdict(list)
    for t in filtered_trials:
        # Get final position (use calculated final_eye_x/y from vstim_go)
        final_x = t.get('final_eye_x', t['eye_x'][-1] if len(t['eye_x']) > 0 else np.nan)
        final_y = t.get('final_eye_y', t['eye_y'][-1] if len(t['eye_y']) > 0 else np.nan)

        # Key: (target_x, target_y) - NO visibility
        target_key = (round(t['target_x'], 2), round(t['target_y'], 2))
        target_groups[target_key].append({
            'final_x': final_x,
            'final_y': final_y,
            'target_x': t['target_x'],
            'target_y': t['target_y'],
            'target_diameter': t['target_diameter']
        })

    # Sort groups by position
    sorted_groups = sorted(target_groups.keys(), key=lambda k: (k[0], k[1]))

    print(f"  Detected {len(sorted_groups)} unique target positions:")
    for target_key in sorted_groups:
        tx, ty = target_key
        n_trials = len(target_groups[target_key])
        print(f"    Target ({tx:+.2f}, {ty:+.2f}): {n_trials} trials")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use different colors for each unique target position
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))

    # Plot each target position
    for idx, target_key in enumerate(sorted_groups):
        tx, ty = target_key
        trials_data = target_groups[target_key]

        # Extract final positions
        final_xs = [d['final_x'] for d in trials_data]
        final_ys = [d['final_y'] for d in trials_data]

        # Calculate mean
        mean_x = np.mean(final_xs)
        mean_y = np.mean(final_ys)

        color = colors[idx]
        label = f"Target ({tx:+.1f}, {ty:+.1f}) (n={len(trials_data)})"

        # Plot individual trial endpoints
        ax.scatter(final_xs, final_ys, alpha=0.4, color=color, s=30, label=label)

        # Plot mean as larger marker
        ax.scatter([mean_x], [mean_y], color=color, s=300, marker='*',
                  edgecolors='black', linewidths=2, zorder=10)

        # Draw target circle at actual position
        target_radius = trials_data[0]['target_diameter'] / 2.0

        circle = Circle((tx, ty), radius=target_radius, fill=False,
                       edgecolor=color, linewidth=2.5, linestyle='-',
                       alpha=0.7)
        ax.add_patch(circle)

        # Add small marker at target center
        ax.plot(tx, ty, 'o', color=color, markersize=5, markeredgecolor='black',
               markeredgewidth=0.5)

        print(f"    Mean final position for ({tx:+.2f}, {ty:+.2f}): "
              f"({mean_x:.3f}, {mean_y:.3f})")

    ax.set_xlabel('Horizontal Position (stimulus units)', fontsize=14)
    ax.set_ylabel('Vertical Position (stimulus units)', fontsize=14)

    title = 'Final Cursor Positions by Target Type'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    title += f'\n(Last sample position, filtered: {min_duration}s ≤ duration ≤ {max_duration}s, N={len(filtered_trials)})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}final_positions_by_target.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved final positions plot to {results_dir / filename}")

    return fig


    # Calculate average position during time window for each trial
    def get_avg_position_in_window(trial, window_start, window_end):
        """Calculate average eye position during specified time window."""
        # Use eye trajectory's own timestamps for consistency
        trial_times = trial['eye_times'] - trial['eye_start_time']  # Relative to eye trajectory start
        mask = (trial_times >= window_start) & (trial_times <= window_end)

        if np.sum(mask) == 0:
            # No data in window, return NaN
            return np.nan, np.nan

        avg_x = np.mean(trial['eye_x'][mask])
        avg_y = np.mean(trial['eye_y'][mask])
        return avg_x, avg_y

    # Extract average positions for left trials
    left_positions = [get_avg_position_in_window(t, window_start, window_end) for t in left_trials]
    left_avg_x = np.array([pos[0] for pos in left_positions])
    left_avg_y = np.array([pos[1] for pos in left_positions])

    # Remove trials with NaN (not enough data in window)
    valid_left = ~(np.isnan(left_avg_x) | np.isnan(left_avg_y))
    left_avg_x = left_avg_x[valid_left]
    left_avg_y = left_avg_y[valid_left]
    n_valid_left = len(left_avg_x)

    # Extract average positions for right trials
    right_positions = [get_avg_position_in_window(t, window_start, window_end) for t in right_trials]
    right_avg_x = np.array([pos[0] for pos in right_positions])
    right_avg_y = np.array([pos[1] for pos in right_positions])

    # Remove trials with NaN
    valid_right = ~(np.isnan(right_avg_x) | np.isnan(right_avg_y))
    right_avg_x = right_avg_x[valid_right]
    right_avg_y = right_avg_y[valid_right]
    n_valid_right = len(right_avg_x)

    print(f"  Left target trials with valid data in window: {n_valid_left}/{len(left_trials)}")
    print(f"  Right target trials with valid data in window: {n_valid_right}/{len(right_trials)}")

    if n_valid_left == 0 or n_valid_right == 0:
        print("  Warning: Not enough trials with data in the time window!")
        return None, None

    # Statistical tests (Mann-Whitney U test, non-parametric)
    stat_x, p_x = scipy_stats.mannwhitneyu(left_avg_x, right_avg_x, alternative='two-sided')
    stat_y, p_y = scipy_stats.mannwhitneyu(left_avg_y, right_avg_y, alternative='two-sided')

    # Calculate summary statistics
    stats_dict = {
        'left': {
            'n': n_valid_left,
            'avg_x_mean': np.mean(left_avg_x),
            'avg_x_std': np.std(left_avg_x),
            'avg_y_mean': np.mean(left_avg_y),
            'avg_y_std': np.std(left_avg_y),
        },
        'right': {
            'n': n_valid_right,
            'avg_x_mean': np.mean(right_avg_x),
            'avg_x_std': np.std(right_avg_x),
            'avg_y_mean': np.mean(right_avg_y),
            'avg_y_std': np.std(right_avg_y),
        },
        'tests': {
            'x_statistic': stat_x,
            'x_pvalue': p_x,
            'y_statistic': stat_y,
            'y_pvalue': p_y,
        },
        'time_window': time_window,
    }

    print(f"\n  Left trials - Starting position ({window_start}-{window_end}s):")
    print(f"    X: {stats_dict['left']['avg_x_mean']:.3f} ± {stats_dict['left']['avg_x_std']:.3f}")
    print(f"    Y: {stats_dict['left']['avg_y_mean']:.3f} ± {stats_dict['left']['avg_y_std']:.3f}")
    print(f"  Right trials - Starting position ({window_start}-{window_end}s):")
    print(f"    X: {stats_dict['right']['avg_x_mean']:.3f} ± {stats_dict['right']['avg_x_std']:.3f}")
    print(f"    Y: {stats_dict['right']['avg_y_mean']:.3f} ± {stats_dict['right']['avg_y_std']:.3f}")
    print(f"\n  Mann-Whitney U test:")
    print(f"    X-position: U={stat_x:.1f}, p={p_x:.4f} {'***' if p_x < 0.001 else '**' if p_x < 0.01 else '*' if p_x < 0.05 else 'ns'}")
    print(f"    Y-position: U={stat_y:.1f}, p={p_y:.4f} {'***' if p_y < 0.001 else '**' if p_y < 0.01 else '*' if p_y < 0.05 else 'ns'}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get labels for positions
    pos1_label = f"Pos ({sorted_positions[0][0]:+.1f}, {sorted_positions[0][1]:+.1f})"
    if len(sorted_positions) == 2:
        pos2_label = f"Pos ({sorted_positions[1][0]:+.1f}, {sorted_positions[1][1]:+.1f})"
    else:
        pos2_label = f"Other {len(sorted_positions)-1} positions"

    # Use consistent colors
    color1 = 'steelblue'
    color2 = 'coral'

    # Plot 1: X-position distributions
    ax = axes[0, 0]
    ax.hist(left_avg_x, bins=20, alpha=0.6, color=color1, label=f'{pos1_label} (n={n_valid_left})')
    ax.hist(right_avg_x, bins=20, alpha=0.6, color=color2, label=f'{pos2_label} (n={n_valid_right})')
    ax.axvline(np.mean(left_avg_x), color=color1, linestyle='--', linewidth=2, label=f'Pos1 mean: {np.mean(left_avg_x):.3f}')
    ax.axvline(np.mean(right_avg_x), color=color2, linestyle='--', linewidth=2, label=f'Pos2+ mean: {np.mean(right_avg_x):.3f}')
    ax.set_xlabel(f'Avg X Position ({window_start}-{window_end}s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'X-Position Distribution\np = {p_x:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Y-position distributions
    ax = axes[0, 1]
    ax.hist(left_avg_y, bins=20, alpha=0.6, color=color1, label=f'{pos1_label} (n={n_valid_left})')
    ax.hist(right_avg_y, bins=20, alpha=0.6, color=color2, label=f'{pos2_label} (n={n_valid_right})')
    ax.axvline(np.mean(left_avg_y), color=color1, linestyle='--', linewidth=2, label=f'Pos1 mean: {np.mean(left_avg_y):.3f}')
    ax.axvline(np.mean(right_avg_y), color=color2, linestyle='--', linewidth=2, label=f'Pos2+ mean: {np.mean(right_avg_y):.3f}')
    ax.set_xlabel(f'Avg Y Position ({window_start}-{window_end}s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Y-Position Distribution\np = {p_y:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: 2D scatter of average positions
    ax = axes[1, 0]
    # Draw actual target positions for all unique positions
    from matplotlib.patches import Circle
    colors_for_targets = plt.cm.tab10(np.linspace(0, 1, len(sorted_positions)))
    for idx, pos in enumerate(sorted_positions):
        target_x, target_y = pos
        color = colors_for_targets[idx]
        circle = Circle((target_x, target_y), radius=0.05, fill=False, edgecolor=color,
                       linewidth=2, linestyle='--', alpha=0.6,
                       label=f'Target ({target_x:+.1f}, {target_y:+.1f})')
        ax.add_patch(circle)

    ax.scatter(left_avg_x, left_avg_y, alpha=0.5, color=color1, s=30, label=f'{pos1_label} trials')
    ax.scatter(right_avg_x, right_avg_y, alpha=0.5, color=color2, s=30, label=f'{pos2_label} trials')
    # Plot means as larger markers
    ax.scatter([np.mean(left_avg_x)], [np.mean(left_avg_y)], color=color1, s=200,
               marker='*', edgecolors='black', linewidths=2, label=f'{pos1_label} mean', zorder=10)
    ax.scatter([np.mean(right_avg_x)], [np.mean(right_avg_y)], color=color2, s=200,
               marker='*', edgecolors='black', linewidths=2, label=f'{pos2_label} mean', zorder=10)
    ax.set_xlabel(f'Avg X Position ({window_start}-{window_end}s)', fontsize=12)
    ax.set_ylabel(f'Avg Y Position ({window_start}-{window_end}s)', fontsize=12)
    ax.set_title('Average Positions (2D)', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Shorten labels for table if needed
    table_pos1 = pos1_label if len(pos1_label) < 15 else f"Pos1"
    table_pos2 = pos2_label if len(pos2_label) < 15 else f"Pos2+"

    table_data = [
        ['Metric', table_pos1, table_pos2, 'p-value'],
        ['N trials', f"{n_valid_left}", f"{n_valid_right}", ''],
        ['X position', f"{stats_dict['left']['avg_x_mean']:.3f} ± {stats_dict['left']['avg_x_std']:.3f}",
         f"{stats_dict['right']['avg_x_mean']:.3f} ± {stats_dict['right']['avg_x_std']:.3f}",
         f"{p_x:.4f} {'***' if p_x < 0.001 else '**' if p_x < 0.01 else '*' if p_x < 0.05 else 'ns'}"],
        ['Y position', f"{stats_dict['left']['avg_y_mean']:.3f} ± {stats_dict['left']['avg_y_std']:.3f}",
         f"{stats_dict['right']['avg_y_mean']:.3f} ± {stats_dict['right']['avg_y_std']:.3f}",
         f"{p_y:.4f} {'***' if p_y < 0.001 else '**' if p_y < 0.01 else '*' if p_x < 0.05 else 'ns'}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Statistics\n(Mann-Whitney U Test)', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    if len(sorted_positions) == 2:
        title = f'Starting Position Bias Analysis (avg {window_start}-{window_end}s): Position 1 vs Position 2'
    else:
        title = f'Starting Position Bias Analysis (avg {window_start}-{window_end}s): {len(sorted_positions)} Positions'
    if animal_id:
        title += f' - {animal_id}'
    if session_date:
        title += f' ({session_date})'
    title += f'\n(Trials filtered: {min_duration}s ≤ duration ≤ {max_duration}s, N valid={n_valid_left + n_valid_right})'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure if results directory provided
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        filename = f"{prefix}starting_position_bias.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved starting position bias plot to {results_dir / filename}")

    return fig, stats_dict


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
        # Count successes and failures
        successes = sum(1 for t in trial_list if not t.get('trial_failed', False))
        failures = sum(1 for t in trial_list if t.get('trial_failed', False))
        success_rate = successes / len(trial_list) if len(trial_list) > 0 else 0
        return {
            'durations': durations,
            'path_lengths': path_lengths,
            'efficiencies': efficiencies,
            'dir_errors': dir_errors,
            'successes': successes,
            'failures': failures,
            'success_rate': success_rate
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

    # Fisher's exact test for success rates
    # Create contingency table: [[left_success, left_fail], [right_success, right_fail]]
    contingency_table = [
        [left_metrics['successes'], left_metrics['failures']],
        [right_metrics['successes'], right_metrics['failures']]
    ]
    success_odds_ratio, success_p = scipy_stats.fisher_exact(contingency_table)

    # Calculate chance levels for left and right separately
    print("  Calculating chance level for left targets (1000 shuffles)...")
    left_chance = calculate_chance_level(trials, n_shuffles=1000,
                                         target_filter=lambda t: abs(t['target_x'] - left_x) < tolerance,
                                         results_dir=results_dir)
    print(f"  Left chance level: {100*left_chance:.1f}%")

    print("  Calculating chance level for right targets (1000 shuffles)...")
    right_chance = calculate_chance_level(trials, n_shuffles=1000,
                                          target_filter=lambda t: abs(t['target_x'] - right_x) < tolerance,
                                          results_dir=results_dir)
    print(f"  Right chance level: {100*right_chance:.1f}%")

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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

    # Plot 4: Success/Failure Rates
    ax = axes[0, 2]
    # Bar chart showing success and failure rates, plus chance levels
    x_pos = np.array([0.7, 1.0, 1.3, 1.7, 2.0, 2.3])
    success_counts = [left_metrics['successes'], left_metrics['failures'], left_chance * n_left,
                     right_metrics['successes'], right_metrics['failures'], right_chance * n_right]
    colors = ['forestgreen', 'firebrick', 'gray', 'forestgreen', 'firebrick', 'gray']
    bars = ax.bar(x_pos, success_counts, width=0.25, color=colors, edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, success_counts)):
        height = bar.get_height()
        # For chance bars, just show percentage
        if i == 2 or i == 5:
            pct = 100 * left_chance if i == 2 else 100 * right_chance
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add overall percentage labels for actual performance
    left_success_pct = 100 * left_metrics['success_rate']
    right_success_pct = 100 * right_metrics['success_rate']
    ax.text(1.0, max(success_counts) * 0.4, f'{left_success_pct:.1f}%\nsuccess',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(2.0, max(success_counts) * 0.4, f'{right_success_pct:.1f}%\nsuccess',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_ylabel('Trial Count', fontsize=12)
    ax.set_title(f'Success/Failure Rates\np = {success_p:.4f}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(success_counts) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([bars[0], bars[1], bars[2]], ['Success', 'Failure', 'Chance'], loc='upper right', fontsize=9)

    # Plot 5: Summary statistics table
    ax = axes[1, 2]
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
        ['Success Rate',
         f'{100*left_metrics["success_rate"]:.1f}% ({left_metrics["successes"]}/{n_left})',
         f'{100*right_metrics["success_rate"]:.1f}% ({right_metrics["successes"]}/{n_right})',
         f'{success_p:.4f}'],
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
    for i, p_val in enumerate([duration_p, length_p, eff_p, success_p], start=2):
        if p_val < 0.05:
            table[(i, 3)].set_facecolor('#ffcccc')
            table[(i, 3)].set_text_props(weight='bold')

    ax.set_title('Summary Statistics\n(Mann-Whitney U & Fisher Exact)', fontsize=12, fontweight='bold')

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
            'path_efficiency': eff_p,
            'success_rate': success_p
        }
    }

    # Print summary
    print(f"\n  Duration: Left={np.mean(left_metrics['durations']):.2f}s, Right={np.mean(right_metrics['durations']):.2f}s, p={duration_p:.4f}")
    print(f"  Path Length: Left={np.mean(left_metrics['path_lengths']):.3f}, Right={np.mean(right_metrics['path_lengths']):.3f}, p={length_p:.4f}")
    print(f"  Path Efficiency: Left={np.mean(left_metrics['efficiencies']):.3f}, Right={np.mean(right_metrics['efficiencies']):.3f}, p={eff_p:.4f}")
    print(f"  Success Rate: Left={100*left_metrics['success_rate']:.1f}% ({left_metrics['successes']}/{n_left}), Right={100*right_metrics['success_rate']:.1f}% ({right_metrics['successes']}/{n_right}), p={success_p:.4f}")

    if duration_p < 0.05:
        print(f"  *** Significant difference in duration (p < 0.05)")
    if length_p < 0.05:
        print(f"  *** Significant difference in path length (p < 0.05)")
    if eff_p < 0.05:
        print(f"  *** Significant difference in efficiency (p < 0.05)")
    if success_p < 0.05:
        print(f"  *** Significant difference in success rate (p < 0.05)")

    return fig, stats_dict


def compare_visible_invisible_performance(trials: list[dict], results_dir: Optional[Path] = None,
                                          animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Compare performance metrics for visible vs invisible target trials.

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
        Figure and dictionary containing statistics and test results
    """
    from scipy import stats as scipy_stats

    # Classify trials by target visibility
    visible_trials = [t for t in trials if t.get('target_visible', 1) == 1]
    invisible_trials = [t for t in trials if t.get('target_visible', 1) == 0]

    n_visible = len(visible_trials)
    n_invisible = len(invisible_trials)

    print(f"\nVisible/Invisible Target Analysis:")
    print(f"  Visible trials: {n_visible}")
    print(f"  Invisible trials: {n_invisible}")

    if n_visible == 0 or n_invisible == 0:
        print("Warning: Not enough trials for visible/invisible comparison")
        return None, None

    # Extract metrics for each group
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

    visible_metrics = extract_metrics(visible_trials)
    invisible_metrics = extract_metrics(invisible_trials)

    # Statistical tests (Mann-Whitney U test - non-parametric)
    duration_stat, duration_p = scipy_stats.mannwhitneyu(
        visible_metrics['durations'], invisible_metrics['durations'], alternative='two-sided'
    )
    length_stat, length_p = scipy_stats.mannwhitneyu(
        visible_metrics['path_lengths'], invisible_metrics['path_lengths'], alternative='two-sided'
    )
    eff_stat, eff_p = scipy_stats.mannwhitneyu(
        visible_metrics['efficiencies'], invisible_metrics['efficiencies'], alternative='two-sided'
    )

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time to Target
    ax = axes[0, 0]
    positions = [1, 2]
    box_data = [visible_metrics['durations'], invisible_metrics['durations']]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Visible', 'Invisible'])
    ax.set_ylabel('Time to Target (s)', fontsize=12)
    ax.set_title(f'Time to Target\np = {duration_p:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add means as points
    ax.plot(1, np.mean(visible_metrics['durations']), 'ro', markersize=10, label='Mean')
    ax.plot(2, np.mean(invisible_metrics['durations']), 'ro', markersize=10)

    # Add sample sizes
    ax.text(1, ax.get_ylim()[0], f'n={n_visible}', ha='center', va='top', fontsize=9)
    ax.text(2, ax.get_ylim()[0], f'n={n_invisible}', ha='center', va='top', fontsize=9)

    # Plot 2: Path Length
    ax = axes[0, 1]
    box_data = [visible_metrics['path_lengths'], invisible_metrics['path_lengths']]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Visible', 'Invisible'])
    ax.set_ylabel('Path Length', fontsize=12)
    ax.set_title(f'Path Length\np = {length_p:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax.plot(1, np.mean(visible_metrics['path_lengths']), 'ro', markersize=10, label='Mean')
    ax.plot(2, np.mean(invisible_metrics['path_lengths']), 'ro', markersize=10)

    # Plot 3: Path Efficiency
    ax = axes[1, 0]
    box_data = [visible_metrics['efficiencies'], invisible_metrics['efficiencies']]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightyellow', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Visible', 'Invisible'])
    ax.set_ylabel('Path Efficiency', fontsize=12)
    ax.set_title(f'Path Efficiency\np = {eff_p:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax.plot(1, np.mean(visible_metrics['efficiencies']), 'ro', markersize=10, label='Mean')
    ax.plot(2, np.mean(invisible_metrics['efficiencies']), 'ro', markersize=10)

    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Visible', 'Invisible', 'p-value'],
        ['', f'(n={n_visible})', f'(n={n_invisible})', ''],
        ['Duration (s)',
         f'{np.mean(visible_metrics["durations"]):.2f}±{np.std(visible_metrics["durations"]):.2f}',
         f'{np.mean(invisible_metrics["durations"]):.2f}±{np.std(invisible_metrics["durations"]):.2f}',
         f'{duration_p:.4f}'],
        ['Path Length',
         f'{np.mean(visible_metrics["path_lengths"]):.3f}±{np.std(visible_metrics["path_lengths"]):.3f}',
         f'{np.mean(invisible_metrics["path_lengths"]):.3f}±{np.std(invisible_metrics["path_lengths"]):.3f}',
         f'{length_p:.4f}'],
        ['Path Efficiency',
         f'{np.mean(visible_metrics["efficiencies"]):.3f}±{np.std(visible_metrics["efficiencies"]):.3f}',
         f'{np.mean(invisible_metrics["efficiencies"]):.3f}±{np.std(invisible_metrics["efficiencies"]):.3f}',
         f'{eff_p:.4f}'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight significant p-values
    for i, p_val in enumerate([duration_p, length_p, eff_p], start=2):
        if p_val < 0.05:
            table[(i, 3)].set_facecolor('#ffcccc')
            table[(i, 3)].set_text_props(weight='bold')

    ax.set_title('Summary Statistics\n(Mann-Whitney U Test)', fontsize=12, fontweight='bold')

    # Overall title
    title = 'Visible vs Invisible Target Performance'
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
        filename = f"{prefix}saccade_feedback_visible_vs_invisible.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"Saved visible vs invisible comparison to {results_dir / filename}")

    # Compile statistics dictionary
    stats_dict = {
        'n_visible': n_visible,
        'n_invisible': n_invisible,
        'visible_metrics': visible_metrics,
        'invisible_metrics': invisible_metrics,
        'p_values': {
            'duration': duration_p,
            'path_length': length_p,
            'path_efficiency': eff_p,
        },
        'statistics': {
            'duration': duration_stat,
            'path_length': length_stat,
            'path_efficiency': eff_stat,
        }
    }

    # Print summary
    print(f"\n  Duration: Visible={np.mean(visible_metrics['durations']):.2f}s, Invisible={np.mean(invisible_metrics['durations']):.2f}s, p={duration_p:.4f}")
    print(f"  Path Length: Visible={np.mean(visible_metrics['path_lengths']):.3f}, Invisible={np.mean(invisible_metrics['path_lengths']):.3f}, p={length_p:.4f}")
    print(f"  Path Efficiency: Visible={np.mean(visible_metrics['efficiencies']):.3f}, Invisible={np.mean(invisible_metrics['efficiencies']):.3f}, p={eff_p:.4f}")

    if duration_p < 0.05:
        print(f"  *** Significant difference in duration (p < 0.05)")
    if length_p < 0.05:
        print(f"  *** Significant difference in path length (p < 0.05)")
    if eff_p < 0.05:
        print(f"  *** Significant difference in efficiency (p < 0.05)")

    return fig, stats_dict


def plot_visible_invisible_detailed_stats(trials: list[dict], results_dir: Optional[Path] = None,
                                          animal_id: Optional[str] = None, session_date: str = "") -> tuple:
    """Plot detailed statistics comparing visible vs invisible targets.

    Shows:
    - Initial direction error distributions
    - Success/failure counts
    - Direction error boxplots

    This function is standalone and can be easily removed without affecting other analyses.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries (should include both successful and failed trials)
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title

    Returns
    -------
    tuple of (fig, stats_dict)
        Figure and dictionary containing statistics
    """
    from scipy import stats as scipy_stats

    # Classify trials by target visibility
    visible_trials = [t for t in trials if t.get('target_visible', 1) == 1]
    invisible_trials = [t for t in trials if t.get('target_visible', 1) == 0]

    n_visible = len(visible_trials)
    n_invisible = len(invisible_trials)

    print(f"\nDetailed Visible/Invisible Target Analysis:")
    print(f"  Visible trials: {n_visible}")
    print(f"  Invisible trials: {n_invisible}")

    if n_visible == 0 or n_invisible == 0:
        print("Warning: Not enough trials for visible/invisible detailed comparison")
        return None, None

    # Count success/failure for each group
    visible_success = sum(1 for t in visible_trials if not t.get('trial_failed', False))
    visible_failed = n_visible - visible_success
    invisible_success = sum(1 for t in invisible_trials if not t.get('trial_failed', False))
    invisible_failed = n_invisible - invisible_success

    # Extract initial direction errors (only for successful trials with valid data)
    visible_dir_errors = [t['initial_direction_error'] for t in visible_trials
                         if not t.get('trial_failed', False) and not np.isnan(t.get('initial_direction_error', np.nan))]
    invisible_dir_errors = [t['initial_direction_error'] for t in invisible_trials
                           if not t.get('trial_failed', False) and not np.isnan(t.get('initial_direction_error', np.nan))]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Visible vs Invisible Target Analysis - {animal_id} - {session_date}',
                fontsize=14, fontweight='bold')

    # Plot 1: Success/Failure counts (stacked bar chart)
    ax = axes[0, 0]
    categories = ['Visible', 'Invisible']
    success_counts = [visible_success, invisible_success]
    failed_counts = [visible_failed, invisible_failed]

    x_pos = np.arange(len(categories))
    width = 0.6

    bars1 = ax.bar(x_pos, success_counts, width, label='Success', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos, failed_counts, width, bottom=success_counts, label='Failed', color='red', alpha=0.7)

    ax.set_ylabel('Number of Trials', fontsize=12)
    ax.set_title('Trial Success/Failure Counts', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for i, (s, f) in enumerate(zip(success_counts, failed_counts)):
        ax.text(i, s/2, str(s), ha='center', va='center', fontweight='bold', fontsize=11)
        ax.text(i, s + f/2, str(f), ha='center', va='center', fontweight='bold', fontsize=11)
        ax.text(i, s + f + 1, f'n={s+f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Success rates (percentage)
    ax = axes[0, 1]
    visible_success_rate = 100 * visible_success / n_visible if n_visible > 0 else 0
    invisible_success_rate = 100 * invisible_success / n_invisible if n_invisible > 0 else 0

    bars = ax.bar(x_pos, [visible_success_rate, invisible_success_rate], width,
                  color=['green', 'darkgreen'], alpha=0.7)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Trial Success Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (rate, bar) in enumerate(zip([visible_success_rate, invisible_success_rate], bars)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 3: Initial direction error boxplots (successful trials only)
    ax = axes[1, 0]
    if len(visible_dir_errors) > 0 and len(invisible_dir_errors) > 0:
        positions = [1, 2]
        box_data = [visible_dir_errors, invisible_dir_errors]
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', edgecolor='black'),
                       medianprops=dict(color='darkred', linewidth=2))

        # Statistical test
        dir_stat, dir_p = scipy_stats.mannwhitneyu(visible_dir_errors, invisible_dir_errors,
                                                   alternative='two-sided')

        ax.set_xticks(positions)
        ax.set_xticklabels(['Visible', 'Invisible'])
        ax.set_ylabel('Initial Direction Error (degrees)', fontsize=12)
        ax.set_title(f'Initial Direction Error (Successful Trials)\np = {dir_p:.4f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add means as points
        ax.plot(1, np.mean(visible_dir_errors), 'ro', markersize=10, label='Mean')
        ax.plot(2, np.mean(invisible_dir_errors), 'ro', markersize=10)

        # Add sample sizes
        ax.text(1, ax.get_ylim()[0], f'n={len(visible_dir_errors)}', ha='center', va='top', fontsize=9)
        ax.text(2, ax.get_ylim()[0], f'n={len(invisible_dir_errors)}', ha='center', va='top', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor direction error analysis',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        dir_p = np.nan

    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Visible', 'Invisible'],
        ['Total Trials', f'{n_visible}', f'{n_invisible}'],
        ['Successful', f'{visible_success} ({visible_success_rate:.1f}%)',
         f'{invisible_success} ({invisible_success_rate:.1f}%)'],
        ['Failed', f'{visible_failed} ({100-visible_success_rate:.1f}%)',
         f'{invisible_failed} ({100-invisible_success_rate:.1f}%)'],
        ['', '', ''],
        ['Direction Error', 'Mean ± SD', ''],
    ]

    if len(visible_dir_errors) > 0:
        table_data.append(['  Visible',
                          f'{np.mean(visible_dir_errors):.1f}° ± {np.std(visible_dir_errors):.1f}°', ''])
    if len(invisible_dir_errors) > 0:
        table_data.append(['  Invisible',
                          f'{np.mean(invisible_dir_errors):.1f}° ± {np.std(invisible_dir_errors):.1f}°', ''])

    if len(visible_dir_errors) > 0 and len(invisible_dir_errors) > 0:
        table_data.append(['  p-value', f'{dir_p:.4f}', ''])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()

    # Save figure
    if results_dir:
        filename = f"{animal_id}_{session_date}_visible_invisible_detailed_stats.png"
        save_path = results_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved detailed visible/invisible stats to {save_path}")

    # Compile statistics dictionary
    stats_dict = {
        'n_visible': n_visible,
        'n_invisible': n_invisible,
        'visible_success': visible_success,
        'visible_failed': visible_failed,
        'invisible_success': invisible_success,
        'invisible_failed': invisible_failed,
        'visible_success_rate': visible_success_rate,
        'invisible_success_rate': invisible_success_rate,
        'visible_dir_errors': visible_dir_errors,
        'invisible_dir_errors': invisible_dir_errors,
    }

    if len(visible_dir_errors) > 0 and len(invisible_dir_errors) > 0:
        stats_dict['dir_error_p_value'] = dir_p
        stats_dict['dir_error_statistic'] = dir_stat

    # Print summary
    print(f"\n  Success Rate: Visible={visible_success_rate:.1f}%, Invisible={invisible_success_rate:.1f}%")
    if len(visible_dir_errors) > 0 and len(invisible_dir_errors) > 0:
        print(f"  Direction Error: Visible={np.mean(visible_dir_errors):.1f}°, Invisible={np.mean(invisible_dir_errors):.1f}°, p={dir_p:.4f}")
        if dir_p < 0.05:
            print(f"  *** Significant difference in direction error (p < 0.05)")

    return fig, stats_dict


def plot_heatmaps_by_position_and_visibility(trials: list[dict], results_dir: Optional[Path] = None,
                                              animal_id: Optional[str] = None, session_date: str = "",
                                              left_threshold: float = 0.0, right_threshold: float = 0.0) -> plt.Figure:
    """Plot 4 heatmaps showing eye position density for left/right × visible/invisible targets.

    This function is standalone and can be easily removed without affecting other analyses.

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
    left_threshold : float
        X-coordinate threshold - targets with x < this are considered "left" (default: 0.0)
    right_threshold : float
        X-coordinate threshold - targets with x > this are considered "right" (default: 0.0)

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure with 4 subplots
    """
    # Filter trials into 4 groups
    left_visible = [t for t in trials if t.get('target_x', 0) < left_threshold and t.get('target_visible', 1) == 1]
    right_visible = [t for t in trials if t.get('target_x', 0) > right_threshold and t.get('target_visible', 1) == 1]
    left_invisible = [t for t in trials if t.get('target_x', 0) < left_threshold and t.get('target_visible', 1) == 0]
    right_invisible = [t for t in trials if t.get('target_x', 0) > right_threshold and t.get('target_visible', 1) == 0]

    print(f"\nHeatmap breakdown:")
    print(f"  Left Visible: {len(left_visible)} trials")
    print(f"  Right Visible: {len(right_visible)} trials")
    print(f"  Left Invisible: {len(left_invisible)} trials")
    print(f"  Right Invisible: {len(right_invisible)} trials")

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Eye Position Heatmaps by Target Position & Visibility - {animal_id} - {session_date}',
                fontsize=14, fontweight='bold')

    # Helper function to create a heatmap for a specific group
    def create_heatmap(ax, trial_group, title, group_name):
        if len(trial_group) == 0:
            ax.text(0.5, 0.5, f'No {group_name} trials', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=12, fontweight='bold')
            return

        # Collect all eye positions from this group
        all_x = []
        all_y = []
        for trial in trial_group:
            if len(trial['eye_x']) > 0:  # Only include trials with eye data
                all_x.extend(trial['eye_x'])
                all_y.extend(trial['eye_y'])

        if len(all_x) == 0:
            ax.text(0.5, 0.5, f'No eye data for\n{group_name} trials', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=12, fontweight='bold')
            return

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Create 2D histogram
        bins = 40  # Number of bins in each dimension
        h, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[-1.7, 1.7], [-1, 1]])

        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(h.T, extent=extent, origin='lower', cmap='hot', aspect='auto', interpolation='bilinear')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Number of Samples')

        # Overlay target positions
        for trial in trial_group:
            target_x = trial['target_x']
            target_y = trial['target_y']
            target_radius = trial['target_diameter'] / 2.0
            # Use different colors for visible vs invisible
            if trial.get('target_visible', 1) == 1:
                target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                                      edgecolor='cyan', linewidth=2, linestyle='-', alpha=0.7)
            else:
                target_circle = Circle((target_x, target_y), radius=target_radius, fill=False,
                                      edgecolor='lime', linewidth=2, linestyle='--', alpha=0.7)
            ax.add_patch(target_circle)

        ax.set_xlabel('Horizontal Position', fontsize=11)
        ax.set_ylabel('Vertical Position', fontsize=11)
        ax.set_title(f'{title}\n(n={len(trial_group)} trials, {len(all_x)} samples)',
                    fontsize=12, fontweight='bold')
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')

    # Create each heatmap
    create_heatmap(axes[0, 0], left_visible, 'Left Visible', 'left visible')
    create_heatmap(axes[0, 1], right_visible, 'Right Visible', 'right visible')
    create_heatmap(axes[1, 0], left_invisible, 'Left Invisible', 'left invisible')
    create_heatmap(axes[1, 1], right_invisible, 'Right Invisible', 'right invisible')

    plt.tight_layout()

    # Save figure
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{animal_id}_{session_date}_heatmaps_position_visibility.png"
        save_path = results_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved position×visibility heatmaps to {save_path}")

    return fig



# Fixation detection parameters - shared across analysis functions
FIXATION_MIN_DURATION = 0.65  # seconds
FIXATION_MAX_MOVEMENT = 0.1  # stimulus units
def interactive_fixation_viewer(trials: list[dict], animal_id: Optional[str] = None,
                                 session_date: str = "", 
                                 min_duration: float = FIXATION_MIN_DURATION,
                                 max_movement: float = FIXATION_MAX_MOVEMENT):
    """Interactive viewer showing detected fixations for each trial.

    Detects periods where eyes moved less than max_movement units for at least
    min_duration seconds, and highlights those points on the trajectory.

    Press SPACE to advance to next trial.

    This function is standalone and can be easily removed without affecting other analyses.

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries
    animal_id : str, optional
        Animal identifier for title
    session_date : str, optional
        Session date for title
    min_duration : float
        Minimum fixation duration in seconds (default: 0.45)
    max_movement : float
        Maximum movement threshold for fixation in stimulus units (default: 0.15)
    """
    if len(trials) == 0:
        print("No trials to display")
        return

    # Filter to trials with eye data
    trials_with_data = [t for t in trials if len(t.get('eye_x', [])) > 0]
    if len(trials_with_data) == 0:
        print("No trials with eye tracking data")
        return

    print(f"Interactive Fixation Viewer: {len(trials_with_data)} trials")
    print(f"Fixation criteria: ≥{min_duration}s duration, frame-to-frame movement <{max_movement} units")
    print("Press SPACE to advance, ESC or 'q' to quit")

    fig, ax = plt.subplots(figsize=(12, 10))
    current_trial_idx = [0]  # Use list to allow modification in nested function

    def plot_trial(idx):
        ax.clear()
        trial = trials_with_data[idx]

        eye_x = np.array(trial['eye_x'])
        eye_y = np.array(trial['eye_y'])
        eye_times = np.array(trial.get('eye_times', np.arange(len(eye_x))))
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_diameter = trial['target_diameter']
        target_radius = target_diameter / 2.0
        cursor_radius = trial.get('cursor_diameter', 0.2) / 2.0
        contact_threshold = target_radius + cursor_radius
        trial_num = trial.get('trial_number', idx + 1)
        trial_end_time = trial.get('end_time', eye_times[-1] if len(eye_times) > 0 else 0)

        is_failed = trial.get('trial_failed', False)
        target_visible = trial.get('target_visible', 1)

        # Detect fixations (uses shared module-level function)
        fixations = detect_fixations(eye_x, eye_y, eye_times, min_duration, max_movement)

        # Plot full trajectory (lighter)
        if is_failed:
            ax.plot(eye_x, eye_y, 'r-', linewidth=1.5, alpha=0.3, label='Eye trajectory (FAILED)', zorder=1)
        else:
            ax.plot(eye_x, eye_y, 'b-', linewidth=1.5, alpha=0.3, label='Eye trajectory', zorder=1)

        # Plot non-fixation points (small dots)
        fixation_mask = np.zeros(len(eye_x), dtype=bool)
        for start, end, duration, span in fixations:
            fixation_mask[start:end] = True

        non_fixation_x = eye_x[~fixation_mask]
        non_fixation_y = eye_y[~fixation_mask]
        if len(non_fixation_x) > 0:
            ax.plot(non_fixation_x, non_fixation_y, 'o', color='gray',
                   markersize=4, alpha=0.5, label='Non-fixation', zorder=2)

        # Highlight fixation points with colormap (early = purple/blue, late = yellow)
        # Use colormap that doesn't include red
        cmap = plt.cm.viridis  # viridis goes from purple to yellow, no red
        n_fixations = len(fixations)
        n_missed = 0

        for fix_idx, (start, end, duration, span) in enumerate(fixations):
            fix_x = eye_x[start:end]
            fix_y = eye_y[start:end]
            fix_times = eye_times[start:end]

            # Calculate distances from target for all points in fixation
            fix_distances = np.sqrt((fix_x - target_x)**2 + (fix_y - target_y)**2)
            all_points_within_target = np.all(fix_distances <= contact_threshold)

            # Calculate time from end of fixation to end of trial
            fixation_end_time = fix_times[-1]
            time_to_trial_end = trial_end_time - fixation_end_time

            # Check if this is a potential missed detection
            is_potential_missed = (all_points_within_target and time_to_trial_end > 0.5)

            if is_potential_missed:
                # Color potentially missed fixations in RED
                color = 'red'
                n_missed += 1
                label = f'Fixation {fix_idx+1} - POTENTIALLY MISSED ({duration:.2f}s, span={span:.4f})'
            else:
                # Use colormap for other fixations (avoiding red)
                color_val = fix_idx / max(1, n_fixations - 1) if n_fixations > 1 else 0.5
                color = cmap(color_val)
                label = f'Fixation {fix_idx+1} ({duration:.2f}s, span={span:.4f})'

            # Plot fixation points (large, bright)
            ax.plot(fix_x, fix_y, 'o', color=color, markersize=10, alpha=0.8,
                   label=label, zorder=4)

            # Calculate and plot fixation center
            fix_center_x = np.mean(fix_x)
            fix_center_y = np.mean(fix_y)
            ax.plot(fix_center_x, fix_center_y, 'x', color=color, markersize=15,
                   markeredgewidth=3, zorder=5)

            # Add text label for potentially missed fixations
            if is_potential_missed:
                ax.text(fix_center_x, fix_center_y + 0.1, 'MISSED?',
                       color='red', fontsize=12, fontweight='bold',
                       ha='center', va='bottom', zorder=6)

        # Plot start position
        start_x = eye_x[0]
        start_y = eye_y[0]
        ax.plot(start_x, start_y, 'go', markersize=12, label='Start', zorder=3)

        # Plot target
        target_circle = Circle((target_x, target_y), radius=target_radius,
                              fill=False, edgecolor='red', linewidth=2, linestyle='-', label='Target')
        ax.add_patch(target_circle)
        ax.plot(target_x, target_y, 'r*', markersize=15, zorder=5)

        # Plot contact threshold circle (target + cursor radius)
        contact_circle = Circle((target_x, target_y), radius=contact_threshold,
                               fill=False, edgecolor='orange', linewidth=2, linestyle='--',
                               label=f'Contact threshold ({contact_threshold:.4f})', alpha=0.7)
        ax.add_patch(contact_circle)

        # Plot final position
        final_x = trial.get('final_eye_x', eye_x[-1])
        final_y = trial.get('final_eye_y', eye_y[-1])
        ax.plot(final_x, final_y, 'ks', markersize=10, label='Final position', zorder=3)

        ax.set_xlabel('Horizontal Position', fontsize=12)
        ax.set_ylabel('Vertical Position', fontsize=12)

        # Title with trial info, visibility, success/failure, and fixation count
        visibility_str = 'VISIBLE' if target_visible == 1 else 'INVISIBLE'
        success_str = 'FAILED' if is_failed else 'SUCCESS'
        title = f'Trial {trial_num} (showing {idx + 1}/{len(trials_with_data)}) - Target: {visibility_str} - Status: {success_str}\n'
        title += f'{len(fixations)} fixation(s) detected'
        if n_missed > 0:
            title += f' ({n_missed} potentially missed)'
        title += f'\nContact threshold: {contact_threshold:.4f} = target_r({target_radius:.4f}) + cursor_r({cursor_radius:.4f})'

        if animal_id or session_date:
            title = f'{animal_id} {session_date}\n{title}'

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')

        fig.canvas.draw()

    def on_key(event):
        if event.key == ' ':  # Space bar
            current_trial_idx[0] = (current_trial_idx[0] + 1) % len(trials_with_data)
            plot_trial(current_trial_idx[0])
        elif event.key in ['escape', 'q']:
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Plot first trial
    plot_trial(0)
    plt.show()


def save_detailed_fixation_data(trials: list[dict], results_dir: Optional[Path] = None,
                                  animal_id: Optional[str] = None, session_date: str = "",
                                  vstim_go_fixation_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Save detailed frame-by-frame fixation data for all trials.

    For each detected fixation, saves all individual data points including:
    - Trial number
    - Fixation number (within that trial)
    - Frame number (absolute frame number from vstim_go CSV)
    - Eye position (x, y)
    - Distance from target
    - Time (if available)

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries (used for trial-level metadata like trial_failed)
    results_dir : Path, optional
        Directory to save CSV file. If None, doesn't save to disk.
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for filename
    vstim_go_fixation_df : pd.DataFrame, optional
        DataFrame from create_vstim_go_fixation_csv containing fixation detection.
        If provided, uses this for fixation detection (recommended for consistency).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: trial_number, fixation_number, frame_number,
        eye_x, eye_y, distance_from_target, time_sec, target_x, target_y,
        target_radius, cursor_radius, contact_threshold, target_visible, trial_failed,
        fixation_duration, fixation_span, all_points_within_target, etc.

    Notes
    -----
    A fixation point is considered "on target" when the eye-to-target distance is
    <= (target_radius + cursor_radius), accounting for both target and cursor sizes.
    """
    import pandas as pd

    # Build trial metadata lookup from trials list
    trial_metadata = {}
    for trial in trials:
        trial_num = trial.get('trial_number', 0)
        trial_metadata[trial_num] = {
            'trial_failed': trial.get('trial_failed', False),
            'trial_duration': trial.get('duration', 0),
            'trial_end_time': trial.get('end_time', 0),
            'target_visible': trial.get('target_visible', 1),
        }

    if vstim_go_fixation_df is None or len(vstim_go_fixation_df) == 0:
        print("No vstim_go_fixation_df provided - cannot generate detailed fixation data")
        return pd.DataFrame()

    # Filter to only fixation frames during trials
    fix_df = vstim_go_fixation_df[
        (vstim_go_fixation_df['in_fixation'] == True) &
        (vstim_go_fixation_df['trial_number'] > 0)
    ].copy()

    if len(fix_df) == 0:
        print("No fixation frames found in vstim_go_fixation_df")
        return pd.DataFrame()

    print(f"Processing fixation data from vstim_go_fixation_df...")
    print(f"Found {len(fix_df)} fixation frames across {fix_df['trial_number'].nunique()} trials")

    # Collect all data points for all fixations
    all_fixation_data = []

    # Process each trial
    for trial_num in sorted(fix_df['trial_number'].unique()):
        trial_fix = fix_df[fix_df['trial_number'] == trial_num].copy()

        # Get trial metadata
        meta = trial_metadata.get(trial_num, {
            'trial_failed': False,
            'trial_duration': 0,
            'trial_end_time': trial_fix['timestamp'].max() if len(trial_fix) > 0 else 0,
            'target_visible': 1,
        })

        # Process each fixation in this trial
        for fix_id in sorted(trial_fix['fixation_id'].unique()):
            if fix_id == 0:  # Skip non-fixation frames
                continue

            fixation_frames = trial_fix[trial_fix['fixation_id'] == fix_id].sort_values('frame')

            if len(fixation_frames) == 0:
                continue

            # Calculate fixation-level metrics
            fix_x = fixation_frames['eye_x'].values
            fix_y = fixation_frames['eye_y'].values
            fix_times = fixation_frames['timestamp'].values
            fix_frames = fixation_frames['frame'].values

            # Fixation duration and span
            duration = fix_times[-1] - fix_times[0]
            x_range = np.max(fix_x) - np.min(fix_x)
            y_range = np.max(fix_y) - np.min(fix_y)
            span = np.sqrt(x_range**2 + y_range**2)

            # Target info (should be same for all frames in trial)
            target_x = fixation_frames['target_x'].iloc[0]
            target_y = fixation_frames['target_y'].iloc[0]
            target_radius = fixation_frames['target_radius'].iloc[0]
            cursor_radius = fixation_frames['cursor_radius'].iloc[0]
            contact_threshold = fixation_frames['contact_threshold'].iloc[0]

            # Distance metrics
            fix_distances = fixation_frames['distance_from_target'].values
            max_dist_in_fixation = np.max(fix_distances)
            min_dist_in_fixation = np.min(fix_distances)

            # Check if ALL points are within contact threshold
            all_points_within_target = fixation_frames['within_contact_threshold'].all()

            # Time from end of fixation to end of trial
            fixation_end_time = fix_times[-1]
            trial_end_time = meta['trial_end_time']
            if trial_end_time == 0:
                # Estimate from last frame in trial
                trial_frames = vstim_go_fixation_df[vstim_go_fixation_df['trial_number'] == trial_num]
                trial_end_time = trial_frames['timestamp'].max() if len(trial_frames) > 0 else fixation_end_time
            time_to_trial_end = trial_end_time - fixation_end_time

            # Potential missed detection
            is_potential_missed_detection = (all_points_within_target and time_to_trial_end > 0.5)

            # Add each frame in this fixation
            for idx, (_, row) in enumerate(fixation_frames.iterrows()):
                all_fixation_data.append({
                    'trial_number': trial_num,
                    'fixation_number': fix_id,
                    'frame_number': int(row['frame']),
                    'point_index_in_fixation': idx,
                    'eye_x': row['eye_x'],
                    'eye_y': row['eye_y'],
                    'distance_from_target': row['distance_from_target'],
                    'point_within_target': row['within_contact_threshold'],
                    'frame_to_frame_movement': row['frame_to_frame_movement'],
                    'time_sec': row['timestamp'],
                    'time_since_fixation_start': row['timestamp'] - fix_times[0],
                    'time_until_fixation_end': fixation_end_time - row['timestamp'],
                    'target_x': target_x,
                    'target_y': target_y,
                    'target_radius': target_radius,
                    'cursor_radius': cursor_radius,
                    'contact_threshold': contact_threshold,
                    'target_visible': meta['target_visible'],
                    'trial_failed': meta['trial_failed'],
                    'trial_duration': meta['trial_duration'],
                    'fixation_duration': duration,
                    'fixation_span': span,
                    'fixation_start_time': fix_times[0],
                    'fixation_end_time': fixation_end_time,
                    'all_points_within_target': all_points_within_target,
                    'max_distance_from_target_in_fixation': max_dist_in_fixation,
                    'min_distance_from_target_in_fixation': min_dist_in_fixation,
                    'time_from_fixation_end_to_trial_end': time_to_trial_end,
                    'is_potential_missed_detection': is_potential_missed_detection
                })

    # Create DataFrame
    df = pd.DataFrame(all_fixation_data)

    # Count unique fixations
    if len(df) > 0:
        n_fixations = df.groupby(['trial_number', 'fixation_number']).ngroups
        n_trials = df['trial_number'].nunique()
    else:
        n_fixations = 0
        n_trials = 0

    print(f"Found {n_fixations} fixations across {n_trials} trials")
    print(f"Total data points: {len(df)}")

    # Summary of on-target fixations and potential missed detections
    if len(df) > 0:
        # Group by trial and fixation to get unique fixations (not individual frames)
        fixation_summary = df.groupby(['trial_number', 'fixation_number']).first()

        n_on_target = fixation_summary['all_points_within_target'].sum()
        n_missed = fixation_summary['is_potential_missed_detection'].sum()

        print(f"\nFixation Analysis:")
        print(f"  Fixations with all points within target: {n_on_target}/{len(fixation_summary)} ({100*n_on_target/len(fixation_summary):.1f}%)")
        print(f"  Potential missed detections (on-target but trial continued >0.5s): {n_missed}/{n_on_target} ({100*n_missed/n_on_target:.1f}% of on-target)" if n_on_target > 0 else f"  Potential missed detections: {n_missed}")

        if n_missed > 0:
            missed_fixations = fixation_summary[fixation_summary['is_potential_missed_detection']]
            print(f"\n  Missed detection details:")
            print(f"    Mean time from fixation end to trial end: {missed_fixations['time_from_fixation_end_to_trial_end'].mean():.2f}s")
            print(f"    Trials with missed detections: {missed_fixations.index.get_level_values('trial_number').nunique()}")

    # Save to CSV if results_dir is provided
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{animal_id}_detailed_fixation_data.csv" if animal_id else "detailed_fixation_data.csv"
        if session_date:
            filename = f"{animal_id}_{session_date}_detailed_fixation_data.csv"

        filepath = results_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved detailed fixation data to: {filepath}")

    return df


def calculate_and_validate_trial_success(trials: list[dict], eot_df: pd.DataFrame,
                                         min_fixation_duration: float = 0.65,
                                         max_movement: float = 0.1) -> pd.DataFrame:
    """Calculate trial success from fixation data and compare to actual trial success.

    For each trial, this function:
    1. Detects fixations using frame-to-frame movement threshold (same as interactive viewer)
    2. Uses ONLY the LAST fixation (since each prosaccade trial should have only one fixation)
    3. Checks if the last fixation ENDS on target (within contact_threshold)
    4. Determines if the last fixation meets the minimum duration requirement
    5. Compares to actual trial_success from task

    Success criteria:
    - Fixation detected when consecutive frame-to-frame movements < max_movement
    - Uses ONLY the LAST fixation detected
    - Fixation must END on target (last point within contact_threshold: target_radius + cursor_radius)
    - Fixation duration must be >= min_fixation_duration (default: 0.65s)

    NOTE: In prosaccade trials, there should ideally be only one fixation:
    - A fixation inside the target should end the trial with success
    - A fixation outside the target should end the trial with failure
    - If multiple fixations are detected, we use only the last one

    Parameters
    ----------
    trials : list of dict
        List of trial data dictionaries with eye position and target information
    eot_df : pd.DataFrame
        End-of-trial dataframe with actual trial_success column (2=success, !=2=failed)
    min_fixation_duration : float
        Minimum fixation duration required for success (default: 0.65 seconds)
    max_movement : float
        Maximum frame-to-frame movement for fixation detection (default: 0.1 units)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: trial_number, actual_success, calculated_success,
        match, max_fixation_duration, explanation
    """
    print(f"\n{'='*80}")
    print(f"TRIAL SUCCESS VALIDATION")
    print(f"{'='*80}")
    print(f"Calculating trial success from fixation data and comparing to actual results...")
    print(f"Success criterion: LAST fixation (≥{min_fixation_duration}s) that ENDS on target")
    print(f"Fixation detection: frame-to-frame movement < {max_movement} units")
    print(f"  (Using ONLY the last fixation, counting TOTAL fixation duration)")
    print()

    results = []

    # Debug: Print info about first few trials
    debug_first_n = 5

    for trial_idx, trial in enumerate(trials):
        trial_num = trial.get('trial_number', -1)

        debug_this_trial = (trial_idx < debug_first_n)

        # Get actual trial success from eot_df
        if trial_num <= len(eot_df):
            actual_success_code = eot_df.iloc[trial_num - 1]['trial_success']
            actual_success = (actual_success_code == 2)  # 2 = success
        else:
            actual_success_code = -1
            actual_success = None


        # Skip trials without eye data
        if not trial.get('has_eye_data', False):
            results.append({
                'trial_number': trial_num,
                'actual_success': actual_success,
                'actual_success_code': actual_success_code,
                'calculated_success': None,
                'match': None,
                'max_fixation_duration': 0.0,
                'max_fixation_info': '',
                'explanation': 'No eye data',
                'target_x': trial.get('target_x', np.nan),
                'target_y': trial.get('target_y', np.nan),
                'contact_threshold': np.nan,
                'num_fixations_detected': 0,
                'num_on_target_fixations': 0,
            })
            continue

        # Extract eye position data
        eye_x = np.array(trial.get('eye_x', []))
        eye_y = np.array(trial.get('eye_y', []))
        eye_times = np.array(trial.get('eye_times', []))

        if len(eye_x) == 0 or len(eye_times) == 0:
            results.append({
                'trial_number': trial_num,
                'actual_success': actual_success,
                'actual_success_code': actual_success_code,
                'calculated_success': None,
                'match': None,
                'max_fixation_duration': 0.0,
                'max_fixation_info': '',
                'explanation': 'Empty eye trajectory',
                'target_x': trial.get('target_x', np.nan),
                'target_y': trial.get('target_y', np.nan),
                'contact_threshold': np.nan,
                'num_fixations_detected': 0,
                'num_on_target_fixations': 0,
            })
            continue

        # Get target information
        target_x = trial['target_x']
        target_y = trial['target_y']
        target_radius = trial['target_diameter'] / 2.0
        cursor_radius = trial.get('cursor_diameter', 0.2) / 2.0
        contact_threshold = target_radius + cursor_radius


        # Step 1: Detect all fixations using frame-to-frame movement (uses shared function)
        fixations = detect_fixations(eye_x, eye_y, eye_times, min_fixation_duration, max_movement)

        # Step 2: Use ONLY the LAST fixation
        # In prosaccade trials, there should ideally be only one fixation:
        # - A fixation inside the target ends the trial with success
        # - A fixation outside the target ends the trial with failure
        # If multiple fixations are detected, we use only the last one
        max_fixation_duration = 0.0
        max_fixation_info = ""
        on_target_fixations = []

        if len(fixations) > 0:
            # Get the LAST fixation only
            start_idx, end_idx, duration, span = fixations[-1]

            # Get eye positions for the last fixation
            fix_x = eye_x[start_idx:end_idx]
            fix_y = eye_y[start_idx:end_idx]
            fix_times = eye_times[start_idx:end_idx]

            # Calculate distances from target for all points in fixation
            fix_distances = np.sqrt((fix_x - target_x)**2 + (fix_y - target_y)**2)
            within_target = fix_distances <= contact_threshold

            # Check if the last fixation ENDS on target
            ends_on_target = within_target[-1]

            if ends_on_target:
                # Last fixation ends on target, use its total duration
                on_target_fixations.append((start_idx, end_idx, duration, span))
                max_fixation_duration = duration
                max_fixation_info = f"{duration:.3f}s total fixation (ends on target at t={fix_times[-1]:.2f}s)"

        # Determine calculated success (last fixation ends on target with sufficient duration)
        calculated_success = (max_fixation_duration >= min_fixation_duration)


        # Check if it matches actual success
        if actual_success is None:
            match = None
            explanation = f"No actual success data; calculated: {calculated_success}"
        else:
            match = (calculated_success == actual_success)
            if match:
                explanation = f"✓ Match: max_fixation={max_fixation_duration:.3f}s"
            else:
                if calculated_success and not actual_success:
                    explanation = f"✗ MISMATCH: Calculated SUCCESS (fixation={max_fixation_duration:.3f}s) but actual FAILED"
                else:
                    explanation = f"✗ MISMATCH: Calculated FAILED (max_fixation={max_fixation_duration:.3f}s) but actual SUCCESS"

        results.append({
            'trial_number': trial_num,
            'actual_success': actual_success,
            'actual_success_code': actual_success_code,
            'calculated_success': calculated_success,
            'match': match,
            'max_fixation_duration': max_fixation_duration,
            'max_fixation_info': max_fixation_info,
            'explanation': explanation,
            'target_x': target_x,
            'target_y': target_y,
            'contact_threshold': contact_threshold,
            'num_fixations_detected': len(fixations),
            'num_on_target_fixations': len(on_target_fixations),
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print(f"\nResults for {len(df)} trials:")
    print(f"-" * 80)

    if df['match'].notna().any():
        n_match = df['match'].sum()
        n_total = df['match'].notna().sum()
        n_mismatch = n_total - n_match

        print(f"  Matches:     {n_match}/{n_total} ({100*n_match/n_total:.1f}%)")
        print(f"  Mismatches:  {n_mismatch}/{n_total} ({100*n_mismatch/n_total:.1f}%)")
        print()

        # Show success breakdown (excluding None values)
        n_actual_success = df['actual_success'].fillna(False).sum()
        n_actual_failed = df[df['actual_success'] == False].shape[0]  # Count False values only
        n_calc_success = df['calculated_success'].fillna(False).sum()
        n_calc_failed = df[df['calculated_success'] == False].shape[0]  # Count False values only

        print(f"  Actual:      {int(n_actual_success)} success, {int(n_actual_failed)} failed")
        print(f"  Calculated:  {int(n_calc_success)} success, {int(n_calc_failed)} failed")
        print()

        # Show mismatches in detail (limit to first 10 for readability)
        if n_mismatch > 0:
            print(f"\n{'!'*80}")
            print(f"MISMATCHES DETECTED ({int(n_mismatch)} trials):")
            print(f"{'!'*80}")
            mismatches = df[df['match'] == False].copy()

            # Show first 10 mismatches in detail
            n_show = min(10, len(mismatches))
            print(f"\nShowing first {n_show} of {len(mismatches)} mismatches:")

            for idx, (_, row) in enumerate(mismatches.head(n_show).iterrows()):
                print(f"\n  Trial {int(row['trial_number'])}:")
                print(f"    Actual: {'SUCCESS' if row['actual_success'] else 'FAILED'} (code={row['actual_success_code']})")
                print(f"    Calculated: {'SUCCESS' if row['calculated_success'] else 'FAILED'}")
                print(f"    Fixations detected: {int(row['num_fixations_detected'])} total, {int(row['num_on_target_fixations'])} on target")
                print(f"    Max on-target fixation: {row['max_fixation_duration']:.3f}s (threshold: {min_fixation_duration}s)")
                if row['max_fixation_info']:
                    print(f"    {row['max_fixation_info']}")
                print(f"    Contact threshold: {row['contact_threshold']:.4f}")
                print(f"    Target position: ({row['target_x']:.3f}, {row['target_y']:.3f})")
        else:
            print(f"\n{'✓'*80}")
            print(f"ALL TRIALS MATCH! Calculated success matches actual success perfectly.")
            print(f"{'✓'*80}")

    print(f"\n{' '*80}")

    return df


def create_vstim_go_fixation_csv(folder_path: Path, results_dir: Optional[Path] = None,
                                   animal_id: Optional[str] = None, session_date: str = "",
                                   min_duration: float = FIXATION_MIN_DURATION,
                                   max_movement: float = FIXATION_MAX_MOVEMENT) -> pd.DataFrame:
    """Create frame-by-frame CSV from vstim_go with fixation detection and target metrics.

    This function processes ALL frames from vstim_go (not just fixation points) and adds:
    - Frame-to-frame movement (velocity)
    - Which trial the frame belongs to
    - Target information for that trial
    - Whether the frame is part of a detected fixation
    - Distance from target and contact threshold
    - Whether the frame is within contact threshold

    Parameters
    ----------
    folder_path : Path
        Path to session folder containing vstim_go, vstim_cue, endoftrial CSVs
    results_dir : Path, optional
        Directory to save output CSV
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for filename
    min_duration : float
        Minimum fixation duration in seconds (default: 0.5)
    max_movement : float
        Maximum frame-to-frame movement for fixation (default: 0.12)

    Returns
    -------
    pd.DataFrame
        Frame-by-frame data with fixation detection and target metrics
    """
    import pandas as pd

    # Load the CSV files
    folder_path = Path(folder_path)
    files = list(folder_path.glob("*.csv"))

    vstim_go_file = None
    vstim_cue_file = None
    eot_file = None

    for f in files:
        fname = f.name.lower()
        if "vstim_go" in fname:
            vstim_go_file = f
        elif "vstim_cue" in fname:
            vstim_cue_file = f
        elif "endoftrial" in fname or "end_of_trial" in fname:
            eot_file = f

    if vstim_go_file is None:
        raise FileNotFoundError(f"Could not find vstim_go file in {folder_path}")
    if vstim_cue_file is None:
        raise FileNotFoundError(f"Could not find vstim_cue file in {folder_path}")
    if eot_file is None:
        raise FileNotFoundError(f"Could not find endoftrial file in {folder_path}")

    # Load vstim_go
    print(f"Loading {vstim_go_file.name}...")
    cleaned = clean_csv(str(vstim_go_file))
    eye_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1, dtype=float)
    if eye_arr.ndim == 1:
        eye_arr = eye_arr.reshape(1, -1)

    eye_df = pd.DataFrame({
        'frame': eye_arr[:, 0].astype(int),
        'timestamp': eye_arr[:, 1],
        'eye_x': eye_arr[:, 2],
        'eye_y': eye_arr[:, 3],
        'cursor_diameter': eye_arr[:, 4] if eye_arr.shape[1] >= 5 else 0.2,
    })

    # Load vstim_cue
    print(f"Loading {vstim_cue_file.name}...")
    cleaned = clean_csv(str(vstim_cue_file))
    target_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1, dtype=float)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(1, -1)

    n_cols = target_arr.shape[1]
    if n_cols == 6:
        target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y', 'target_diameter', 'visible'])
    elif n_cols == 5:
        target_df = pd.DataFrame(target_arr, columns=['frame', 'timestamp', 'target_x', 'target_y', 'target_diameter'])
        target_df['visible'] = 1
    else:
        raise ValueError(f"Unexpected number of columns in vstim_cue: {n_cols}")

    target_df['frame'] = target_df['frame'].astype(int)
    target_df = target_df.drop_duplicates(subset=['frame'], keep='first')

    # Load endoftrial
    print(f"Loading {eot_file.name}...")
    cleaned = clean_csv(str(eot_file))
    eot_arr = np.genfromtxt(cleaned, delimiter=",", skip_header=1, dtype=float)
    if eot_arr.ndim == 1:
        eot_arr = eot_arr.reshape(1, -1)

    # Handle variable number of columns in endoftrial (only need first 2)
    n_cols_eot = eot_arr.shape[1]
    if n_cols_eot >= 2:
        eot_df = pd.DataFrame({
            'frame': eot_arr[:, 0].astype(int),
            'timestamp': eot_arr[:, 1]
        })
    else:
        raise ValueError(f"endoftrial file has too few columns: {n_cols_eot}")

    # Match each vstim_go frame to a trial
    print("Matching frames to trials...")
    eye_df['trial_number'] = 0
    eye_df['target_x'] = np.nan
    eye_df['target_y'] = np.nan
    eye_df['target_diameter'] = np.nan
    eye_df['target_visible'] = 0
    eye_df['trial_start_frame'] = 0
    eye_df['trial_end_frame'] = 0

    for trial_idx in range(len(target_df)):
        start_frame = target_df.iloc[trial_idx]['frame']
        end_frame = eot_df.iloc[trial_idx]['frame'] if trial_idx < len(eot_df) else eye_df['frame'].max()

        mask = (eye_df['frame'] >= start_frame) & (eye_df['frame'] <= end_frame)
        eye_df.loc[mask, 'trial_number'] = trial_idx + 1
        eye_df.loc[mask, 'target_x'] = target_df.iloc[trial_idx]['target_x']
        eye_df.loc[mask, 'target_y'] = target_df.iloc[trial_idx]['target_y']
        eye_df.loc[mask, 'target_diameter'] = target_df.iloc[trial_idx]['target_diameter']
        eye_df.loc[mask, 'target_visible'] = target_df.iloc[trial_idx]['visible']
        eye_df.loc[mask, 'trial_start_frame'] = start_frame
        eye_df.loc[mask, 'trial_end_frame'] = end_frame

    # Calculate frame-to-frame movement
    print("Calculating frame-to-frame movement...")
    eye_df['frame_to_frame_movement'] = 0.0
    eye_df.loc[1:, 'frame_to_frame_movement'] = np.sqrt(
        np.diff(eye_df['eye_x'])**2 + np.diff(eye_df['eye_y'])**2
    )

    # Calculate distance from target and contact threshold
    print("Calculating distance from target...")
    eye_df['target_radius'] = eye_df['target_diameter'] / 2.0
    eye_df['cursor_radius'] = eye_df['cursor_diameter'] / 2.0
    eye_df['contact_threshold'] = eye_df['target_radius'] + eye_df['cursor_radius']
    eye_df['distance_from_target'] = np.sqrt(
        (eye_df['eye_x'] - eye_df['target_x'])**2 +
        (eye_df['eye_y'] - eye_df['target_y'])**2
    )
    eye_df['within_contact_threshold'] = eye_df['distance_from_target'] <= eye_df['contact_threshold']

    # Initialize fixation columns
    eye_df['in_fixation'] = False
    eye_df['fixation_id'] = 0
    eye_df['point_index_in_fixation'] = 0
    eye_df['time_since_fixation_start'] = 0.0

    # Detect fixations per trial
    print("Detecting fixations...")

    def detect_fixations_for_trial(trial_mask):
        """Detect fixations for frames in a single trial."""
        trial_data = eye_df[trial_mask].copy()
        if len(trial_data) < 2:
            return []

        eye_x = trial_data['eye_x'].values
        eye_y = trial_data['eye_y'].values
        eye_times = trial_data['timestamp'].values
        indices = trial_data.index.values

        fixations = []  # List of (start_idx, end_idx, start_time)

        i = 0
        n = len(eye_x)
        while i < n:
            j = i + 1
            while j < n:
                dx = eye_x[j] - eye_x[j-1]
                dy = eye_y[j] - eye_y[j-1]
                movement = np.sqrt(dx**2 + dy**2)

                if movement < max_movement:
                    j += 1
                else:
                    break

            if j > i + 1:
                duration = eye_times[j-1] - eye_times[i]
                if duration >= min_duration:
                    fixations.append((indices[i], indices[j-1], eye_times[i]))
                    i = j
                else:
                    i += 1
            else:
                i += 1

        return fixations

    # Process each trial (including inter-trial intervals where trial_number=0)
    for trial_num in eye_df['trial_number'].unique():
        trial_mask = eye_df['trial_number'] == trial_num
        fixations = detect_fixations_for_trial(trial_mask)

        for fix_id, (start_idx, end_idx, start_time) in enumerate(fixations, start=1):
            # Mark frames as in fixation
            fix_mask = (eye_df.index >= start_idx) & (eye_df.index <= end_idx)
            eye_df.loc[fix_mask, 'in_fixation'] = True
            eye_df.loc[fix_mask, 'fixation_id'] = fix_id

            # Calculate point index and time since fixation start
            for idx, (row_idx, row) in enumerate(eye_df[fix_mask].iterrows()):
                eye_df.at[row_idx, 'point_index_in_fixation'] = idx
                eye_df.at[row_idx, 'time_since_fixation_start'] = row['timestamp'] - start_time

    n_trials = eye_df['trial_number'].nunique()
    has_iti = 0 in eye_df['trial_number'].values
    trial_desc = f"{n_trials - 1} trials + inter-trial intervals" if has_iti else f"{n_trials} trials"
    print(f"Processed {len(eye_df)} frames across {trial_desc}")
    print(f"Detected {eye_df['in_fixation'].sum()} frames in fixations (including inter-trial intervals)")

    # Save to CSV
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{animal_id}_vstim_go_fixation.csv" if animal_id else "vstim_go_fixation.csv"
        if session_date:
            filename = f"{animal_id}_{session_date}_vstim_go_fixation.csv"

        filepath = results_dir / filename
        eye_df.to_csv(filepath, index=False)
        print(f"Saved vstim_go_fixation CSV to: {filepath}")

    return eye_df



def export_eye_positions_csv(trials: list[dict], eye_df: pd.DataFrame,
                              results_dir: Path, animal_id: str, date_str: str,
                              ITI: float) -> Path:
    """Export eye positions for all trials to CSV, including ITI periods.

    Parameters
    ----------
    trials : list of dict
        List of trial dictionaries with start/end times
    eye_df : pd.DataFrame
        Complete eye tracking data
    results_dir : Path
        Directory to save CSV
    animal_id : str
        Animal identifier
    date_str : str
        Session date
    ITI : float
        Inter-trial interval duration in seconds

    Returns
    -------
    Path
        Path to saved CSV file
    """
    csv_rows = []

    for trial_idx, trial in enumerate(trials):
        trial_num = trial['trial_number']
        start_frame = trial['start_frame']
        end_frame = trial['end_frame']
        start_time = trial['start_time']
        end_time = trial['end_time']
        trial_failed = trial.get('trial_failed', False)
        has_eye_data = trial.get('has_eye_data', True)

        # Get trial eye data
        trial_mask = (eye_df['frame'] >= start_frame) & (eye_df['frame'] <= end_frame)
        trial_eye_data = eye_df[trial_mask].dropna(subset=['green_x', 'green_y', 'timestamp'])

        # Add trial data rows
        for _, row in trial_eye_data.iterrows():
            csv_rows.append({
                'trial_number': trial_num,
                'trial_failed': trial_failed,
                'frame': int(row['frame']),
                'timestamp': row['timestamp'],
                'eye_x': row['green_x'],
                'eye_y': row['green_y'],
                'is_iti': False,
                'target_x': trial['target_x'],
                'target_y': trial['target_y']
            })

        # Add ITI data (from trial end to next trial start)
        if trial_idx < len(trials) - 1:
            next_trial = trials[trial_idx + 1]
            next_start_frame = next_trial['start_frame']
            next_start_time = next_trial['start_time']

            # ITI eye data
            iti_mask = (eye_df['frame'] > end_frame) & (eye_df['frame'] < next_start_frame)
            iti_eye_data = eye_df[iti_mask].dropna(subset=['green_x', 'green_y', 'timestamp'])

            for _, row in iti_eye_data.iterrows():
                csv_rows.append({
                    'trial_number': trial_num,  # Associate ITI with the trial that just ended
                    'trial_failed': trial_failed,
                    'frame': int(row['frame']),
                    'timestamp': row['timestamp'],
                    'eye_x': row['green_x'],
                    'eye_y': row['green_y'],
                    'is_iti': True,
                    'target_x': np.nan,  # No target during ITI
                    'target_y': np.nan
                })

    # Create DataFrame and save
    df = pd.DataFrame(csv_rows)

    filename = f"{animal_id}_{date_str}_eye_positions_with_iti.csv"
    csv_path = results_dir / filename
    df.to_csv(csv_path, index=False)

    print(f"\n Saved eye positions CSV to: {csv_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Trial rows: {(~df['is_iti']).sum()}")
    print(f"  ITI rows: {df['is_iti'].sum()}")

    return csv_path


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
                   animal_id: str = "Tsh001", show_plots: bool = True,
                   trial_min_duration: float = 0.01, trial_max_duration: float = 15.0,
                   show_failed_in_viewer: bool = False,
                   include_failed_trials: bool = False) -> pd.DataFrame:
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
    trial_min_duration : float
        Minimum trial duration for position bias analyses (default: 0.1 seconds)
    trial_max_duration : float
        Maximum trial duration for position bias analyses (default: 10.0 seconds)
    show_failed_in_viewer : bool
        Whether to show failed trials (in red) in the interactive viewer only (default: False)
    include_failed_trials : bool
        Whether to include failed trials in ALL plots and analyses (default: False)
        When True, all plots will show both successful and failed trials

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
    eot_df, eye_df, target_df_all = load_feedback_data(folder_path, animal_id)

    # Add original trial numbers (1-indexed) to preserve through filtering
    target_df_all['original_trial_number'] = range(1, len(target_df_all) + 1)

    # Always identify and filter failed trials for clean analysis
    target_df_successful, failed_indices, successful_indices = identify_and_filter_failed_trials(
        target_df_all, eot_df, exclude_failed=True
    )

    # Extract ALL trial trajectories for the interactive viewer (includes trials with no eye data)
    # eot_df now contains ALL trials (successful + failed) with trial_success column
    print("\nExtracting ALL trials for interactive viewer...")
    print(f"  Using eot_df with {len(eot_df)} trials (matches target_df_all with {len(target_df_all)} trials)")

    # Extract all trials (will include placeholders for trials with no eye data)
    trials_all = extract_trial_trajectories(eot_df, eye_df, target_df_all,
                                            successful_indices=successful_indices)

    # Separate successful trials for analysis
    trials_successful = [t for t in trials_all if not t.get('trial_failed', False) and t.get('has_eye_data', True)]

    # Decide which trials to use for analysis based on include_failed_trials parameter
    if include_failed_trials:
        trials_for_analysis = trials_all
        print(f"\nUsing ALL {len(trials_all)} trials for analysis (including {len(failed_indices)} failed trials)")
    else:
        trials_for_analysis = trials_successful
        print(f"\nUsing only {len(trials_successful)} successful trials for analysis")

    if len(trials_for_analysis) == 0:
        print("No valid trials found, exiting")
        return pd.DataFrame()

    # Validate trial success calculation from fixation data
    validation_df = calculate_and_validate_trial_success(trials_all, eot_df, min_fixation_duration=0.65)

    # Save validation results to CSV
    if results_dir is not None:
        validation_filename = f"{animal_id}_{date_str}_trial_success_validation.csv" if animal_id and date_str else "trial_success_validation.csv"
        validation_filepath = results_dir / validation_filename
        validation_df.to_csv(validation_filepath, index=False)
        print(f"Saved validation results to: {validation_filepath}")

    # Plot trial success summary (uses ALL trials, independent of --include-failed-trials flag)
    print("\nGenerating trial success summary plot...")
    fig_success = plot_trial_success(eot_df, results_dir, animal_id, date_str, trials=trials_all)
    if fig_success is not None:
        if show_plots:
            plt.show()
        plt.close(fig_success)

    # Interactive viewer: show all trials or just successful ones
    print("\nShowing interactive trajectory viewer...")
    if show_failed_in_viewer:
        print(f"  Showing ALL {len(trials_all)} trials (including {len(failed_indices)} failed trials in RED)")
        trials_for_viewer = trials_all
    else:
        print(f"  Showing only successful trials with eye data ({len(trials_successful)} trials)")
        trials_for_viewer = trials_for_analysis
    print("(Press SPACE to advance to next trial)")
    if show_plots:
        interactive_trajectories(trials_for_viewer, animal_id=animal_id, session_date=date_str)

    print("\nGenerating density heatmap...")
    fig_heat = plot_density_heatmap(trials_for_analysis, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_heat)

    print("\nGenerating time-to-target plot...")
    fig_time = plot_time_to_target(trials_for_analysis, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_time)

    print("\nGenerating path length plot...")
    fig_path = plot_path_length(trials_for_analysis, results_dir, animal_id, date_str)
    if show_plots:
        plt.show()
    plt.close(fig_path)

    print("\nRunning left vs right target comparison...")
    # Note: Always use trials_all for success/failure stats, regardless of --include-failed-trials flag
    # Auto-detect left and right target positions from the data
    unique_x_positions = sorted(list(set([t['target_x'] for t in trials_all])))
    print(f"  Unique target X positions found: {unique_x_positions}")

    # Use the leftmost and rightmost positions if we have at least 2
    if len(unique_x_positions) >= 2:
        detected_left_x = unique_x_positions[0]
        detected_right_x = unique_x_positions[-1]
        print(f"  Using left={detected_left_x:.2f}, right={detected_right_x:.2f}")
        print(f"  (Using all {len(trials_all)} trials including failures for success rate calculation)")
        fig_lr, lr_stats = compare_left_right_performance(trials_all,
                                                           left_x=detected_left_x,
                                                           right_x=detected_right_x,
                                                           results_dir=results_dir,
                                                           animal_id=animal_id,
                                                           session_date=date_str)
    else:
        print(f"  Not enough unique X positions for left/right comparison")
        fig_lr, lr_stats = None, None

    if fig_lr is not None:
        if show_plots:
            plt.show()
        plt.close(fig_lr)

    print("\nRunning visible vs invisible target comparison...")
    fig_vis, vis_stats = compare_visible_invisible_performance(trials_for_analysis, results_dir=results_dir,
                                                                animal_id=animal_id,
                                                                session_date=date_str)
    if fig_vis is not None:
        if show_plots:
            plt.show()
        plt.close(fig_vis)

    # NEW: Detailed visible/invisible statistics (including success/failure and direction errors)
    print("\nGenerating detailed visible vs invisible target statistics...")
    fig_vis_detailed, vis_detailed_stats = plot_visible_invisible_detailed_stats(trials_all, results_dir=results_dir,
                                                                                  animal_id=animal_id,
                                                                                  session_date=date_str)
    if fig_vis_detailed is not None:
        if show_plots:
            plt.show()
        plt.close(fig_vis_detailed)

    # NEW: Interactive fixation viewer
    print("\nShowing interactive fixation viewer...")
    print("  (Shows detected fixation periods for each trial - press SPACE to advance)")
    if show_plots:
        interactive_fixation_viewer(trials_for_analysis, animal_id=animal_id, session_date=date_str)

    # NEW: Create vstim_go_fixation CSV (single source of truth for fixation detection)
    vstim_go_fixation_df = create_vstim_go_fixation_csv(folder_path, results_dir=results_dir,
                                                         animal_id=animal_id,
                                                         session_date=date_str)

    # NEW: Save detailed fixation data (uses vstim_go_fixation_df for consistency)
    print("\nSaving detailed fixation data...")
    detailed_fixation_df = save_detailed_fixation_data(trials_for_analysis, results_dir=results_dir,
                                                        animal_id=animal_id,
                                                        session_date=date_str,
                                                        vstim_go_fixation_df=vstim_go_fixation_df)



    print("\nPlotting final positions by target type...")
    fig_final_pos = plot_final_positions_by_target(trials_for_analysis, min_duration=trial_min_duration, max_duration=trial_max_duration,
                                                    results_dir=results_dir,
                                                    animal_id=animal_id,
                                                    session_date=date_str)
    if fig_final_pos is not None:
        if show_plots:
            plt.show()
        plt.close(fig_final_pos)


    # Create summary DataFrame
    durations = [t['duration'] for t in trials_for_analysis]
    path_lengths = [t['path_length'] for t in trials_for_analysis]
    efficiencies = [t['path_efficiency'] for t in trials_for_analysis]
    dir_errors = [t['initial_direction_error'] for t in trials_for_analysis if not np.isnan(t['initial_direction_error'])]

    df = pd.DataFrame({
        'folder_path': [str(folder_path)],
        'animal_id': [animal_id],
        'session_date': [date_str],
        'n_trials': [len(trials_for_analysis)],
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
    print(f"Valid trials: {len(trials_for_analysis)}")

    return df


def main(session_id: str, trial_min_duration: float = 0.01, trial_max_duration: float = 15.0,
         show_failed_in_viewer: bool = False,
         include_failed_trials: bool = False) -> pd.DataFrame:
    """Run the saccade feedback analysis pipeline for ``session_id``.

    Parameters
    ----------
    session_id : str
        Identifier of the session to analyse.
    trial_min_duration : float
        Minimum trial duration for position bias analyses (default: 0.1 seconds)
    trial_max_duration : float
        Maximum trial duration for position bias analyses (default: 10.0 seconds)
    show_failed_in_viewer : bool
        Whether to show failed trials (in red) in the interactive viewer only (default: False)
    include_failed_trials : bool
        Whether to include failed trials in ALL plots and analyses (default: False)
        When True, all plots will show both successful and failed trials

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
    eot_df, eye_df, target_df_all = load_feedback_data(folder_path, animal_id or "Tsh001")

    # Add original trial numbers (1-indexed) to preserve through filtering
    target_df_all['original_trial_number'] = range(1, len(target_df_all) + 1)

    # Always identify and filter failed trials for clean analysis
    target_df_successful, failed_indices, successful_indices = identify_and_filter_failed_trials(
        target_df_all, eot_df, exclude_failed=True
    )

    # Extract ALL trial trajectories for the interactive viewer (includes trials with no eye data)
    # eot_df now contains ALL trials (successful + failed) with trial_success column
    print("\nExtracting ALL trials for interactive viewer...")
    print(f"  Using eot_df with {len(eot_df)} trials (matches target_df_all with {len(target_df_all)} trials)")

    # Extract all trials (will include placeholders for trials with no eye data)
    trials_all = extract_trial_trajectories(eot_df, eye_df, target_df_all,
                                            successful_indices=successful_indices)

    # Separate successful trials for analysis
    trials_successful = [t for t in trials_all if not t.get('trial_failed', False) and t.get('has_eye_data', True)]

    # Decide which trials to use for analysis based on include_failed_trials parameter
    if include_failed_trials:
        trials_for_analysis = trials_all
        print(f"\nUsing ALL {len(trials_all)} trials for analysis (including {len(failed_indices)} failed trials)")
    else:
        trials_for_analysis = trials_successful
        print(f"\nUsing only {len(trials_successful)} successful trials for analysis")

    if len(trials_for_analysis) == 0:
        print("No valid trials found, exiting")
        return pd.DataFrame()

    # Validate trial success calculation from fixation data
    validation_df = calculate_and_validate_trial_success(trials_all, eot_df, min_fixation_duration=0.65)

    # Save validation results to CSV
    if results_dir is not None:
        validation_filename = f"{animal_id}_{date_str}_trial_success_validation.csv" if animal_id and date_str else "trial_success_validation.csv"
        validation_filepath = results_dir / validation_filename
        validation_df.to_csv(validation_filepath, index=False)
        print(f"Saved validation results to: {validation_filepath}")

    # Plot trial success summary (uses ALL trials, independent of --include-failed-trials flag)
    print("\nGenerating trial success summary plot...")
    fig_success = plot_trial_success(eot_df, results_dir, animal_id, date_str, trials=trials_all)
    if fig_success is not None:
        plt.show()
        plt.close(fig_success)


    print("\nRunning left vs right target comparison...")
    # Note: Always use trials_all for success/failure stats, regardless of --include-failed-trials flag
    # Auto-detect left and right target positions from the data
    unique_x_positions = sorted(list(set([t['target_x'] for t in trials_all])))
    print(f"  Unique target X positions found: {unique_x_positions}")

    # Use the leftmost and rightmost positions if we have at least 2
    if len(unique_x_positions) >= 2:
        detected_left_x = unique_x_positions[0]
        detected_right_x = unique_x_positions[-1]
        print(f"  Using left={detected_left_x:.2f}, right={detected_right_x:.2f}")
        print(f"  (Using all {len(trials_all)} trials including failures for success rate calculation)")
        fig_lr, lr_stats = compare_left_right_performance(trials_all,
                                                           left_x=detected_left_x,
                                                           right_x=detected_right_x,
                                                           results_dir=results_dir,
                                                           animal_id=animal_id,
                                                           session_date=date_str)
    else:
        print(f"  Not enough unique X positions for left/right comparison")
        fig_lr, lr_stats = None, None

    if fig_lr is not None:
        plt.show()
        plt.close(fig_lr)


    # Generate plots
    print("\nGenerating trajectory plot...")
    fig_traj = plot_trajectories(trials_for_analysis, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_traj)

    print("\nGenerating trajectory plot by direction (left vs right)...")
    fig_traj_dir = plot_trajectories_by_direction(trials_for_analysis, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_traj_dir)

    # Interactive viewer: show all trials or just successful ones
    print("\nShowing interactive trajectory viewer...")
    if show_failed_in_viewer:
        print(f"  Showing ALL {len(trials_all)} trials (including {len(failed_indices)} failed trials in RED)")
        trials_for_viewer = trials_all
    else:
        print(f"  Showing only successful trials with eye data ({len(trials_successful)} trials)")
        trials_for_viewer = trials_for_analysis
    print("(Press SPACE to advance to next trial)")
    interactive_trajectories(trials_for_viewer, animal_id=animal_id, session_date=date_str)

    print("\nGenerating density heatmap...")
    fig_heat = plot_density_heatmap(trials_for_analysis, results_dir, animal_id, date_str)
    plt.show()
    plt.close(fig_heat)

    # print("\nGenerating time-to-target plot...")
    # fig_time = plot_time_to_target(trials_for_analysis, results_dir, animal_id, date_str)
    # plt.show()
    # plt.close(fig_time)

    # print("\nGenerating path length plot...")
    # fig_path = plot_path_length(trials_for_analysis, results_dir, animal_id, date_str)
    # plt.show()
    # plt.close(fig_path)



    # print("\nRunning visible vs invisible target comparison...")
    # fig_vis, vis_stats = compare_visible_invisible_performance(trials_for_analysis, results_dir=results_dir,
    #                                                             animal_id=animal_id,
    #                                                             session_date=date_str)
    # if fig_vis is not None:
    #     plt.show()
    #     plt.close(fig_vis)

    # # NEW: Detailed visible/invisible statistics (including success/failure and direction errors)
    # print("\nGenerating detailed visible vs invisible target statistics...")
    # fig_vis_detailed, vis_detailed_stats = plot_visible_invisible_detailed_stats(trials_all, results_dir=results_dir,
    #                                                                               animal_id=animal_id,
    #                                                                               session_date=date_str)
    # if fig_vis_detailed is not None:
    #     plt.show()
    #     plt.close(fig_vis_detailed)

    # # NEW: Heatmaps broken down by position and visibility
    # print("\nGenerating heatmaps by position and visibility...")
    # fig_heatmaps = plot_heatmaps_by_position_and_visibility(trials_for_analysis, results_dir=results_dir,
    #                                                          animal_id=animal_id,
    #                                                          session_date=date_str)
    # plt.show()
    # plt.close(fig_heatmaps)

    # # NEW: Test for voluntary targeted movement
    # print("\nTesting for voluntary targeted movement...")
    # fig_voluntary, voluntary_stats = test_voluntary_targeted_movement(trials_for_analysis, results_dir=results_dir,
    #                                                                    animal_id=animal_id,
    #                                                                    session_date=date_str)
    # if fig_voluntary is not None:
    #     plt.show()
    #     plt.close(fig_voluntary)

    # # NEW: Test for left/right targeted movement (simplified binary test)
    # print("\nTesting for left/right targeted movement (binary classification)...")
    # fig_lr_test, lr_test_stats = test_left_right_targeted_movement(trials_for_analysis, results_dir=results_dir,
    #                                                                 animal_id=animal_id,
    #                                                                 session_date=date_str)
    # if fig_lr_test is not None:
    #     plt.show()
    #     plt.close(fig_lr_test)

    # # NEW: Interactive initial direction viewer
    # print("\nShowing interactive initial direction viewer...")
    # print("  (Shows initial direction vectors for each trial - press SPACE to advance)")
    # interactive_initial_direction_viewer(trials_for_analysis, animal_id=animal_id, session_date=date_str)

    # NEW: Interactive fixation viewer
    print("\nShowing interactive fixation viewer...")
    print("  (Shows detected fixation periods for each trial - press SPACE to advance)")
    interactive_fixation_viewer(trials_for_analysis, animal_id=animal_id, session_date=date_str)

    print("\nPlotting final positions by target type...")
    fig_final_pos = plot_final_positions_by_target(trials_for_analysis, min_duration=trial_min_duration, max_duration=trial_max_duration,
                                                    results_dir=results_dir,
                                                    animal_id=animal_id,
                                                    session_date=date_str)
    if fig_final_pos is not None:
        plt.show()
        plt.close(fig_final_pos)

    # Export eye positions CSV with ITI markers
    print("\nExporting eye positions to CSV...")
    # Calculate ITI from target_df_all
    if len(target_df_all) > 1:
        time_diffs = np.diff(target_df_all['timestamp'].values)
        min_diff = np.min(time_diffs)
        ITI = np.floor(min_diff)
    else:
        ITI = 0
    export_eye_positions_csv(trials_all, eye_df, results_dir, animal_id, date_str, ITI)

    # Create summary DataFrame
    durations = [t['duration'] for t in trials_for_analysis]
    path_lengths = [t['path_length'] for t in trials_for_analysis]
    efficiencies = [t['path_efficiency'] for t in trials_for_analysis]
    dir_errors = [t['initial_direction_error'] for t in trials_for_analysis if not np.isnan(t['initial_direction_error'])]

    df = pd.DataFrame({
        'session_id': [session_id],
        'animal_id': [animal_id],
        'session_date': [date_str],
        'n_trials': [len(trials_for_analysis)],
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
    print(f"Valid trials: {len(trials_for_analysis)}")

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
    parser.add_argument("--show-failed-in-viewer", dest='show_failed_in_viewer', action='store_true', default=False,
                        help="Show failed trials (in RED) in interactive viewer only (default: False)")
    parser.add_argument("--include-failed-trials", dest='include_failed_trials', action='store_true', default=False,
                        help="Include failed trials in ALL plots and analyses (default: False)")
    args = parser.parse_args()

    if args.folder:
        # Direct folder analysis mode
        analyze_folder(args.folder, args.results, args.animal, show_plots=True,
                      show_failed_in_viewer=args.show_failed_in_viewer,
                      include_failed_trials=args.include_failed_trials)
    elif args.session_id:
        # Session manifest mode
        main(args.session_id, show_failed_in_viewer=args.show_failed_in_viewer,
             include_failed_trials=args.include_failed_trials)
    else:
        parser.error("Either session_id or --folder must be provided")
