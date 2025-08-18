from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Put the repo's "Python" folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session

from eyehead import (
    SaccadeConfig,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    plot_eye_fixations_between_cue_and_go_by_trial,
    quantify_fixation_stability_vs_random,
)


def main(session_id: str) -> pd.DataFrame:
    """Run fixation analysis for ``session_id``."""
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

    trial_success = data.trial_success or np.array([])
    eye_position_during_fixation = []
    eye_position_during_fixation_success = []
    for i, gf in enumerate((data.go_frame or [])[: len(trial_success)]):
        idx = np.where(data.eye_frame < gf)[0]
        if idx.size < 7:
            continue
        last_idx = idx[-7:-1]
        eye_pos = saccades["eye_pos"][last_idx, :2]
        mean_pos = np.mean(eye_pos, axis=0)
        eye_position_during_fixation.append(mean_pos)
        if trial_success[i] == 1:
            eye_position_during_fixation_success.append(mean_pos)

    eye_position_during_fixation = np.asarray(eye_position_during_fixation)
    eye_position_during_fixation_success = np.asarray(eye_position_during_fixation_success)

    eye_pos_all = saccades["eye_pos"][:, :2]
    spread_fixation = np.std(eye_position_during_fixation, axis=0)
    spread_all = np.std(eye_pos_all, axis=0)
    ratio_spread = spread_fixation / spread_all

    fig_spread = plt.figure(figsize=(8, 6))
    plt.scatter(eye_pos_all[:, 0], eye_pos_all[:, 1], color="red", alpha=0.1, label="All Eye Positions")
    if eye_position_during_fixation.size:
        plt.scatter(
            eye_position_during_fixation[:, 0],
            eye_position_during_fixation[:, 1],
            color="blue",
            alpha=0.4,
            label="Eye Positions During Fixation",
        )
    if eye_position_during_fixation_success.size:
        plt.scatter(
            eye_position_during_fixation_success[:, 0],
            eye_position_during_fixation_success[:, 1],
            color="green",
            alpha=0.5,
            label="Eye Positions During Fixation (Successful Trials)",
        )
    plt.xlabel("X Position (deg)")
    plt.ylabel("Y Position (deg)")
    plt.title("Eye Positions in the orbit during the whole session and during fixation")
    plt.legend()
    plt.grid()
    fname_spread = f"{config.session_name}_{config.eye_name}_eye_position_spread.png"
    fig_spread.savefig(config.results_dir / fname_spread, dpi=300, bbox_inches="tight")
    plt.close(fig_spread)

    pairs_cf, pairs_gf, pairs_ct, pairs_gt, pairs_dt, valid_trials, fig_pairs, _ = (
        plot_eye_fixations_between_cue_and_go_by_trial(
            eye_frame=data.eye_frame,
            eye_pos=saccades["eye_pos"],
            eye_timestamp=data.eye_timestamp,
            cue_frame=data.cue_frame,
            cue_time=data.cue_time,
            go_frame=data.go_frame,
            go_time=data.go_time,
            max_interval_s=1,
            results_dir=config.results_dir,
            session_name=config.session_name,
            eye_name=config.eye_name,
        )
    )
    plt.close(fig_pairs)

    stats = quantify_fixation_stability_vs_random(
        eye_timestamp=data.eye_timestamp,
        eye_pos=saccades["eye_pos"],
        pairs_ct=pairs_ct,
        pairs_gt=pairs_gt,
        valid_trials=valid_trials,
        plot=True,
        rng_seed=0,
    )

    if stats and stats.get("figure") is not None:
        fig = stats["figure"]
        fname = f"{config.session_name}_{config.eye_name}_fixation_vs_random.png"
        fig.savefig(config.results_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = stats["summary"] if stats else {}
    ms_fix, _, _ = summary.get("mean_step_fix_mean±sem", (np.nan, np.nan, 0))
    ms_rnd, _, _ = summary.get("mean_step_rand_mean±sem", (np.nan, np.nan, 0))
    sp_fix, _ = summary.get("mean_speed_fix_mean±sem", (np.nan, np.nan))
    sp_rnd, _ = summary.get("mean_speed_rand_mean±sem", (np.nan, np.nan))
    dr_fix, _ = summary.get("net_drift_fix_mean±sem", (np.nan, np.nan))
    dr_rnd, _ = summary.get("net_drift_rand_mean±sem", (np.nan, np.nan))

    df = pd.DataFrame(
        {
            "session_id": [session_id],
            "ratio_spread_x": [ratio_spread[0] if ratio_spread.size else np.nan],
            "ratio_spread_y": [ratio_spread[1] if ratio_spread.size > 1 else np.nan],
            "mean_step_fix": [ms_fix],
            "mean_step_rand": [ms_rnd],
            "mean_speed_fix": [sp_fix],
            "mean_speed_rand": [sp_rnd],
            "net_drift_fix": [dr_fix],
            "net_drift_rand": [dr_rnd],
            "valid_trials": [int(valid_trials.sum()) if stats else 0],
        }
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session for fixation metrics")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml")
    args = parser.parse_args()
    main(args.session_id)
