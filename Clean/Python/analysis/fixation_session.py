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
from eyehead.analysis import _filename_with_animal



def main(session_id: str, animal_name: str | None = None) -> pd.DataFrame:
    """Run fixation analysis for ``session_id``.

    Parameters
    ----------
    session_id:
        Identifier of the session to analyse.
    animal_name:
        Optional override for the animal label associated with the session.
        When provided, this value is stored on the session configuration so
        that downstream helpers incorporate it into generated artefacts.
    """
    config = load_session(session_id)
    if animal_name is not None:
        config.animal_name = animal_name or None
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
        plot=False,
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
        animal_name=config.animal_name,
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
        eye_part = (config.eye_name or "Eye").replace(" ", "")
        id_part = str(config.animal_id).strip() if config.animal_id is not None else ""
        stem_parts = [part for part in (id_part, eye_part, "fixation_vs_random") if part]
        stem = "_".join(stem_parts) if stem_parts else "fixation_vs_random"

        base_png = f"{stem}.png"
        base_svg = f"{stem}.svg"
        animal_label = config.animal_name or config.animal_id
        fname_png = _filename_with_animal(base_png, animal_label)
        fname_svg = _filename_with_animal(base_svg, animal_label)

        fig.savefig(config.results_dir / fname_png, bbox_inches="tight")
        fig.savefig(config.results_dir / fname_svg, bbox_inches="tight")

        plt.show()
        plt.close(fig)

    summary = stats["summary"] if stats else {}
    ms_fix, se_fix, _ = summary.get("mean_step_fix_mean±sem", (np.nan, np.nan, 0))
    ms_rnd, se_rnd, _ = summary.get("mean_step_rand_mean±sem", (np.nan, np.nan, 0))
    sp_fix, se_spf = summary.get("mean_speed_fix_mean±sem", (np.nan, np.nan))
    sp_rnd, se_spr = summary.get("mean_speed_rand_mean±sem", (np.nan, np.nan))
    dr_fix, se_drf = summary.get("net_drift_fix_mean±sem", (np.nan, np.nan))
    dr_rnd, se_drr = summary.get("net_drift_rand_mean±sem", (np.nan, np.nan))

    df = pd.DataFrame(
        {
            "session_id": [session_id],
            "animal_name": [config.animal_name],
            "session_date": [date_str],
            "mean_step_fix": [ms_fix],
            "mean_step_fix_sem": [se_fix],
            "mean_step_rand": [ms_rnd],
            "mean_step_rand_sem": [se_rnd],
            "mean_speed_fix": [sp_fix],
            "mean_speed_fix_sem": [se_spf],
            "mean_speed_rand": [sp_rnd],
            "mean_speed_rand_sem": [se_spr],
            "net_drift_fix": [dr_fix],
            "net_drift_fix_sem": [se_drf],
            "net_drift_rand": [dr_rnd],
            "net_drift_rand_sem": [se_drr],
            "valid_trials": [int(valid_trials.sum()) if stats else 0],
        }
    )
    return df


# Usage: python Clean/Python/analysis/fixation_session.py SESSION_ID
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session for fixation metrics")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml")
    parser.add_argument(
        "--animal-name",
        dest="animal_name",
        default=None,
        help="Optional override for the animal label used in generated artefacts.",
    )
    args = parser.parse_args()
    main(args.session_id, animal_name=args.animal_name)

