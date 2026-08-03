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

    python Python/analysis/preanalysis_cleanup.py Tsh001_2025-07-23T16_20_03   # manifest session ID
    python Python/analysis/preanalysis_cleanup.py /path/to/session_folder       # or a direct path

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


def extract_session_info(config):
    """Compute the per-trial timing table and fixed parameters for an already-loaded session.

    Parameters
    ----------
    config : utils.session_loader.SessionConfig
        Resolved session configuration (from `load_session_or_path`).

    Returns
    -------
    df : pandas.DataFrame
        Columns: trial_start, trial_end, target_duration, trial_outcome.
    fixed_parameters : dict
        Median reward_window, cue_duration, and iti, all in frames.
    """
    data = load_session_data(config)

    trial_duration = data.end_of_trial_frame - data.go_frame
    df = pd.DataFrame({
        'trial_start': data.cue_frame,
        'trial_end': data.end_of_trial_frame,
        'target_duration': trial_duration,
        'trial_outcome': data.trial_success,
    })

    # Fixed task parameters (medians, in frames), derived from trial timing:
    #   reward_window : max time the target stayed on for failed trials —
    #       trial_success == 0 marks a miss/timeout, so target_duration on
    #       those trials approximates the response window the animal was given.
    #   cue_duration  : time from cue onset to go/target onset.
    #   iti           : inter-trial interval — gap between one trial's end
    #       and the next trial's cue onset.
    fixed_parameters = {
        'reward_window': np.median(trial_duration[data.trial_success == 0]),
        'cue_duration': np.median(data.go_frame - data.cue_frame),
        'iti': np.median(data.cue_frame[1:] - data.end_of_trial_frame[:-1]),
    }

    return df, fixed_parameters


def main():
    parser = argparse.ArgumentParser(description="Extract session information from a folder.")
    parser.add_argument("folder", type=str, help="Session ID (from session_manifest.yml) or a direct path to a session folder.")
    args = parser.parse_args()

    config = load_session_or_path(args.folder)
    output_dir = Path(config.folder_path)
    print(f"Loading session data from: {output_dir}")

    df, fixed_parameters = extract_session_info(config)
    output_csv_path = output_dir / "session_info.csv"
    df.to_csv(output_csv_path, index=False)

    params_df = pd.DataFrame(
        list(fixed_parameters.items()),
        columns=["Parameter", "Value(frames)"],
    )
    params_df["Value(frames)"] = params_df["Value(frames)"].round(2)
    params_df["Value(seconds)"] = (params_df["Value(frames)"] / config.ttl_freq).round(2)

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    tbl = table(ax, params_df, loc="center", cellLoc="center", colWidths=[0.4, 0.3, 0.3])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    output_png_path = output_dir / "fixed_parameters.png"
    plt.savefig(output_png_path, bbox_inches="tight", dpi=300)
    plt.show()
    #plt.close()


if __name__ == "__main__":
    main()
