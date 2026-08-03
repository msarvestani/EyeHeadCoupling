"""Pre-analysis cleanup: derive per-trial timing and fixed task parameters from a raw session folder.

Scans a single Bonsai session folder for its "go", "endoftrial", and "cue" log
files, aligns them into a per-trial table (trial start/end frame, target
duration, outcome), and estimates session-level constants (reward window,
cue duration, inter-trial interval). Writes two outputs into that same
folder: `session_info.csv` (the per-trial table) and `fixed_parameters.png`
(a rendered table of the fixed parameters).

How to run:
    conda env create -f Python/EyeHeadCoupling.yml   # first time only
    conda activate EyeHeadCoupling
    python Python/analysis/preanalysis_cleanup.py /path/to/session_folder

The session folder must contain exactly one file whose name (case-insensitive)
contains each of "go", "endoftrial", and "cue".
"""
from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.plotting import table 
import matplotlib.pyplot as plt

# Put the repo's “Python” folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session
from eyehead.io import clean_csv



def extract_session_info(folder_path):
    """
    Extract session information from the folder path.
    """

    print(f"Scanning folder: {folder_path}")
    print(f"Found {len(os.listdir(folder_path))} files")

    fixed_parameters = {}
    df = pd.DataFrame(columns=['trial_start', 'trial_end', 'target_duration', 'trial_outcome'])

    # Scan the folder for specific files
    for f in os.listdir(folder_path):
        f_lower = f.lower()
        full_path = os.path.join(folder_path, f)
        if 'go' in f_lower:
            go_file = full_path
        if 'endoftrial' in f_lower:
            end_of_trial_file = full_path
        if 'cue' in f_lower:
            cue_file = full_path

    ## The go file contains the start frame of the target and the direction of the target. 
    go_data = np.genfromtxt(clean_csv(go_file), delimiter=',', skip_header=1, dtype=np.float64)
    [go_frame, go_time, go_direction_x,go_direction_y] = go_data[:, 0], go_data[:, 1], go_data[:, 2], go_data[:,3]
    go_frame = go_frame.astype(int)  # Convert go_frame to integer type

    ## The end of trial file contains the frame number marking the end of the trial. This can be two different things: i) For successful trials,
    ## it means when the animal has made first correct saccade to the target ii) For unsuccessful trials, it means the maximum time allowed for the target has elapsed.
    ## Additionally, the file contains time of the end of trial, the direction of the target, the direction of the eye movement, and the result of the trial (true=success)
    end_of_trial_data = np.genfromtxt(clean_csv(end_of_trial_file), delimiter=',', skip_header=1, dtype=np.float64)
    [end_of_trial_frame, end_of_trial_ts, trial_stim_direction, trial_eye_movement_direction,trial_success] = end_of_trial_data[:, 0], end_of_trial_data[:, 1], end_of_trial_data[:, 2], end_of_trial_data[:, 3], end_of_trial_data[:, 4]
    end_of_trial_frame = end_of_trial_frame.astype(int)  # Convert end_of_trial_frame to integer type

    ## the cue file contains all frame numbers when the visual cue was present. (Redundant just use the first frame of the cue for each trial) 
    cue_data = np.genfromtxt(clean_csv(cue_file), delimiter=',', skip_header=1, dtype=np.float64)
    [cue_frame,cue_time] = cue_data[:, 0], cue_data[:, 1]
    cue_frame = cue_frame.astype(int)  # Convert cue_frame to integer type
    diff = np.diff(cue_frame)
    threshold = 10  # Define a threshold for detecting gaps
    start_indices = np.concatenate(([0], np.where(diff > threshold)[0] + 1))
    cue_frame = cue_frame[start_indices]  
    cue_time = cue_time[start_indices]

    ## If the length of the go_frame, end_of_trial_frame, and cue_frame are not equal, then we need to trim them to the same length (min of the three)
    min_length = min(len(go_frame), len(end_of_trial_frame), len(cue_frame))
    go_frame = go_frame[:min_length]
    end_of_trial_frame = end_of_trial_frame[:min_length]
    cue_frame = cue_frame[:min_length]

    
    ## Create the dataframe with trial start, trial end, target duration, and trial outcome
    trial_duration = end_of_trial_frame - go_frame
    df['trial_start'] = cue_frame
    df['trial_end'] = end_of_trial_frame
    df['target_duration'] = trial_duration
    df['trial_outcome'] = trial_success

    # Extract fixed parameters 
    reward_window = []
    for dur, out in zip(trial_duration, trial_success):
        if out == 0:
            reward_window.append(dur)
    reward_window = np.array(reward_window)
    fixed_parameters['reward_window'] = np.median(reward_window)
    

    cue_durations = go_frame - cue_frame
    fixed_parameters['cue_duration'] = np.median(cue_durations)

    iti = []
    for prev_end, next_start in zip(end_of_trial_frame[:-1], cue_frame[1:]):
        iti.append(next_start - prev_end)

    iti = np.array(iti)
    fixed_parameters['iti'] = np.median(iti)



    return df, fixed_parameters


def main():
    parser = argparse.ArgumentParser(description="Extract session information from a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing session files.")
    args = parser.parse_args()

    df, fixed_parameters = extract_session_info(args.folder)
    ## Create a csv in the same folder from the df
    output_csv_path = os.path.join(args.folder, "session_info.csv")
    df.to_csv(output_csv_path, index=False)

    ## Create a png in the same folder from the fixed parameters
    output_png_path = os.path.join(args.folder, "fixed_parameters.png")
    df = pd.DataFrame(
    list(fixed_parameters.items()),
    columns=["Parameter", "Value(frames)"]
)
    df["Value(frames)"] = df["Value(frames)"].astype(float)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis("off")

    tbl = table(ax, df, loc="center", cellLoc="center", colWidths=[0.6, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    plt.savefig(output_png_path, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()