"""Pre-analysis cleanup: derive per-trial timing and fixed task parameters from a raw session folder.

Loads a session's go/end_of_trial/cue data using the shared session-loading
utilities (`load_session_or_path` + `load_session_data`, the same loaders
used by the other analysis scripts), then derives a per-trial timing table
and three fixed task parameters: reward_window, cue_duration, and iti
(inter-trial interval), all in frames. Writes two outputs into the session
folder: `session_info.csv` (the per-trial table) and `fixed_parameters.png`
(a rendered table of the fixed parameters).

How to run:
    conda env create -f Python/EyeHeadCoupling.yml   # first time only
    conda activate EyeHeadCoupling
    python Python/analysis/preanalysis_cleanup.py /path/to/session_folder

The folder argument may be a session ID already present in
`session_manifest.yml`, or a direct path to a raw Bonsai session folder.
Note: for a folder not yet in the manifest, `load_session_or_path` still
needs to infer `ttl_freq`/`calibration_factor` by looking up another
manifest entry for the same animal_id — if no such entry exists yet (e.g.
this is the very first session recorded for a new animal), it raises a
clear ValueError asking you to add a manifest entry first.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt

# Put the repo's "Python" folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session_or_path
from eyehead import load_session_data


def extract_session_info(folder_path):
    """Load a session and compute its per-trial timing table and fixed parameters.

    Returns
    -------
    df : pandas.DataFrame
        Columns: trial_start, trial_end, target_duration, trial_outcome.
    fixed_parameters : dict
        Median reward_window, cue_duration, and iti, all in frames.
    """
    print(f"Loading session data from: {folder_path}")
    config = load_session_or_path(str(folder_path))
    data = load_session_data(config)

    trial_duration = data.end_of_trial_frame - data.go_frame
    df = pd.DataFrame({
        'trial_start': data.cue_frame,
        'trial_end': data.end_of_trial_frame,
        'target_duration': trial_duration,
        'trial_outcome': data.trial_success,
    })

    fixed_parameters = {
        'reward_window': np.median(trial_duration[data.trial_success == 0]),
        'cue_duration': np.median(data.go_frame - data.cue_frame),
        'iti': np.median(data.cue_frame[1:] - data.end_of_trial_frame[:-1]),
    }

    return df, fixed_parameters


def main():
    parser = argparse.ArgumentParser(description="Extract session information from a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing session files.")
    args = parser.parse_args()

    df, fixed_parameters = extract_session_info(args.folder)
    output_csv_path = os.path.join(args.folder, "session_info.csv")
    df.to_csv(output_csv_path, index=False)

    params_df = pd.DataFrame(
        list(fixed_parameters.items()),
        columns=["Parameter", "Value(frames)"],
    )
    params_df["Value(frames)"] = params_df["Value(frames)"].astype(float)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis("off")

    tbl = table(ax, params_df, loc="center", cellLoc="center", colWidths=[0.6, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    output_png_path = os.path.join(args.folder, "fixed_parameters.png")
    plt.savefig(output_png_path, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()