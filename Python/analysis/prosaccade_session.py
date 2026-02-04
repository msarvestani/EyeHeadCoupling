from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Put the repo's “Python” folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session

from eyehead import (
    SaccadeConfig,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    organize_stims,
    sort_saccades,
    get_session_date_from_path,
)


def main(session_id: str) -> pd.DataFrame:
    """Run the full analysis pipeline for ``session_id``.

    Parameters
    ----------
    session_id:
        Identifier of the session to analyse.
    """
    config = load_session(session_id)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    folder_path = config.folder_path
    results_dir = config.results_dir
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

    # The rest of the analysis would operate on ``folder_path`` and
    # save any generated figures into ``results_dir``.  For now we simply
    # report the resolved paths so that the script remains functional
    # even when the full analysis pipeline is not available.
    print(f"Session path: {folder_path}")
    print(f"Results directory: {results_dir}")

    date_str = config.params.get("date")
    if not date_str and folder_path is not None:
        try:
            date_str = get_session_date_from_path(str(folder_path)).strftime("%Y-%m-%d")
        except Exception:
            date_str = ""

    data = load_session_data(config)
    eye_pos_cal = calibrate_eye_position(data, config)

    saccade_cfg = SaccadeConfig(**config.params["saccade_config"])

    saccades, fig_saccades, _ = detect_saccades(
        eye_pos_cal,
        data.eye_frame,
        saccade_cfg,
        config,
        data=data,
        plot=False,
    )
    if fig_saccades is not None:
        plt.close(fig_saccades)
    indices = saccades["saccade_indices_xy"]
    saccade_frames = saccades.get("saccade_frames_xy", [])
    print(f"Detected {len(indices)} saccades")
    saccades["stim_frames"], stim_type = organize_stims(
        data.go_frame,
        go_dir_x=data.go_direction_x,
        go_dir_y=data.go_direction_y,
    )
    df = pd.DataFrame(
        {
            "session_id": [session_id] * len(indices),
            "session_date": [date_str] * len(indices),
            "saccade_frame_xy": saccade_frames,
            "saccade_index_xy": indices,

        }
    )
    sorted_data,left_angle,right_angle,fig_sorted, _ = sort_saccades(config, saccade_cfg, saccades, stim_type=stim_type, plot=True)
    if fig_sorted is not None:
        plt.close(fig_sorted)
    df = pd.DataFrame(
        {
            "session_id": [session_id] * len(indices),
            "session_date": [date_str] * len(indices),
            "saccade_frame_xy": saccade_frames,
            "saccade_index_xy": indices,

        }
    )
    return df,left_angle,right_angle


# Usage: python Python/analysis/prosaccade_session.py SESSION_ID
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml")
    args = parser.parse_args()
    main(args.session_id)

