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
    get_session_date_from_path,
)



def main(session_id: str) -> pd.DataFrame:
    """Run fixation analysis for ``session_id``."""
    config = load_session(session_id)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    date_str = config.params.get("date")
    if not date_str and config.folder_path is not None:
        try:
            date_str = get_session_date_from_path(str(config.folder_path)).strftime("%Y-%m-%d")
        except Exception:
            date_str = ""

    data = load_session_data(config)
    eye_pos_cal = calibrate_eye_position(data, config)

    saccade_cfg = SaccadeConfig(**config.params["saccade_config"])

    saccades, fig_saccades, ax_saccades = detect_saccades(
        eye_pos_cal,
        data.eye_frame,
        saccade_cfg,
        config,
        data=data,
        plot=True,
    )
    plt.show()
    if fig_saccades is not None:
        plt.close(fig_saccades)


    ( pairs_cf,pairs_gf,pairs_ct,pairs_gt,
        pairs_dt,valid_trials,fig_pairs,_,
    ) = plot_eye_fixations_between_cue_and_go_by_trial(
        eye_frame=data.eye_frame,
        eye_pos=saccades["eye_pos"],
        eye_timestamp=data.eye_timestamp,
        cue_frame=data.cue_frame,
        cue_time=data.cue_time,
        go_frame=data.go_frame,
        go_time=data.go_time,
        max_interval_s=1,
        results_dir=config.results_dir,
        animal_id=config.animal_id,
        eye_name=config.eye_name,
        plot=True,
    )
    plt.show()
    if fig_pairs is not None:
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
        fname = f"{config.animal_id}_{config.eye_name}_fixation_vs_random.png"
        fig.savefig(config.results_dir / fname, dpi=300, bbox_inches="tight")
        plt.show()
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
            "session_date": [date_str],
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

