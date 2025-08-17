from __future__ import annotations

import argparse

from eyehead import (
    SaccadeConfig,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    organize_stims,
    sort_plot_saccades,
)
from utils.session_loader import load_session


def main(session_id: str) -> None:
    """Run the full analysis pipeline for ``session_id``."""
    config = load_session(session_id)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    data = load_session_data(config)
    eye_pos_cal = calibrate_eye_position(data, config)

    saccade_cfg = SaccadeConfig(
        saccade_threshold=1.0,
        saccade_threshold_torsion=1.5,
        blink_threshold=10.0,
        blink_detection=1,
        saccade_win=0.7,
    )

    saccades = detect_saccades(
        eye_pos_cal,
        data.eye_frame,
        saccade_cfg,
        config,
        data=data,
    )
    saccades["stim_frames"], stim_type = organize_stims(
        data.go_frame,
        go_dir_x=data.go_direction_x,
        go_dir_y=data.go_direction_y,
    )
    sort_plot_saccades(config, saccade_cfg, saccades, stim_type=stim_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml")
    args = parser.parse_args()
    main(args.session_id)
