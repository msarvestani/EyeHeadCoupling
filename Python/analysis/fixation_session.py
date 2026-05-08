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


def _compute_path_bins(
    eye_ts: np.ndarray,
    xy: np.ndarray,
    cue_ts: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_path, sem_path) arrays over bins for the given cue times."""
    n_bins = len(bin_edges) - 1
    n_trials = len(cue_ts)
    trial_paths = np.full((n_trials, n_bins), np.nan)
    for t_idx, ct in enumerate(cue_ts):
        for b_idx in range(n_bins):
            t0 = ct + bin_edges[b_idx]
            t1 = ct + bin_edges[b_idx + 1]
            a = np.searchsorted(eye_ts, t0, side="left")
            b = np.searchsorted(eye_ts, t1, side="right")
            if b - a < 2:
                continue
            seg = xy[a:b]
            diffs = np.diff(seg, axis=0)
            trial_paths[t_idx, b_idx] = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    n_ok = np.sum(~np.isnan(trial_paths), axis=0)
    mean_path = np.nanmean(trial_paths, axis=0)
    with np.errstate(invalid="ignore"):
        sem_path = np.nanstd(trial_paths, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))
    sem_path[n_ok < 2] = np.nan
    return mean_path, sem_path


def _draw_path_panel(
    ax: plt.Axes,
    bin_centers: np.ndarray,
    mean_path: np.ndarray,
    sem_path: np.ndarray,
    title: str,
    color: str,
    bin_s: float,
) -> None:
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1.2, label="Cue onset")
    ax.fill_between(
        bin_centers,
        mean_path - sem_path,
        mean_path + sem_path,
        alpha=0.25,
        color=color,
    )
    ax.plot(bin_centers, mean_path, color=color, linewidth=1.5)
    ax.set_ylabel("Mean path length (deg)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def plot_pericue_path_length(
    eye_timestamp: np.ndarray,
    eye_pos: np.ndarray,
    cue_times: np.ndarray,
    *,
    valid_trials: np.ndarray | None = None,
    pre_s: float = 1.5,
    post_s: float = 5.0,
    bin_s: float = 0.25,
) -> plt.Figure:
    """Plot mean eye path length in bins aligned to cue onset.

    Three stacked panels show valid trials (top), invalid trials (middle),
    and all trials (bottom). When valid_trials is None all panels show all
    trials with equivalent content.
    """
    eye_ts = np.asarray(eye_timestamp).ravel()
    xy = np.asarray(eye_pos)[:, :2]
    cue_ts_all = np.asarray(cue_times).ravel()

    bin_edges = np.arange(-pre_s, post_s + bin_s / 2, bin_s)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if valid_trials is not None:
        mask = np.asarray(valid_trials, dtype=bool)
        cue_ts_valid = cue_ts_all[mask]
        cue_ts_invalid = cue_ts_all[~mask]
    else:
        cue_ts_valid = cue_ts_all
        cue_ts_invalid = cue_ts_all

    total = len(cue_ts_all)
    n_valid = len(cue_ts_valid)
    n_invalid = len(cue_ts_invalid)
    pct_valid = 100.0 * n_valid / total if total > 0 else 0.0
    pct_invalid = 100.0 * n_invalid / total if total > 0 else 0.0

    mean_valid, sem_valid = _compute_path_bins(eye_ts, xy, cue_ts_valid, bin_edges)
    mean_invalid, sem_invalid = _compute_path_bins(eye_ts, xy, cue_ts_invalid, bin_edges)
    mean_all, sem_all = _compute_path_bins(eye_ts, xy, cue_ts_all, bin_edges)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    _draw_path_panel(
        axes[0], bin_centers, mean_valid, sem_valid,
        f"Valid trials  (n={n_valid}, {pct_valid:.1f}%,  {bin_s:.2f} s bins)",
        "steelblue", bin_s,
    )
    _draw_path_panel(
        axes[1], bin_centers, mean_invalid, sem_invalid,
        f"Invalid trials  (n={n_invalid}, {pct_invalid:.1f}%,  {bin_s:.2f} s bins)",
        "tomato", bin_s,
    )
    _draw_path_panel(
        axes[2], bin_centers, mean_all, sem_all,
        f"All trials  (n={total},  {bin_s:.2f} s bins)",
        "dimgray", bin_s,
    )

    axes[2].set_xlabel("Time relative to cue (s)")
    fig.suptitle("Eye path length around cue onset")
    #fig.tight_layout()
    return fig


def main(session_id: str) -> pd.DataFrame:
    """Run fixation analysis for ``session_id``.

    Parameters
    ----------
    session_id:
        Identifier of the session to analyse.
    """
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
        plot=False,
    )
    if fig_saccades is not None:
        plt.show()
        plt.close(fig_saccades)


    max_interval_fixations = float(config.params.get("max_interval_fixations", 1.0))

    (
        pairs_cf,
        pairs_gf,
        pairs_ct,
        pairs_gt,
        pairs_dt,
        valid_trials,
        fig_pairs,
        _,
        fig_interval,
    ) = plot_eye_fixations_between_cue_and_go_by_trial(
        eye_frame=data.eye_frame,
        eye_pos=saccades["eye_pos"],
        eye_timestamp=data.eye_timestamp,
        cue_frame=data.cue_frame,
        cue_time=data.cue_time,
        go_frame=data.go_frame,
        go_time=data.go_time,
        max_interval_fixations=max_interval_fixations,
        results_dir=config.results_dir,
        animal_id=config.animal_id,
        eye_name=config.eye_name,
        animal_name=config.animal_name,
        plot=True,
    )
    plt.show()
    for fig in (fig_pairs, fig_interval):
        if fig is not None:
            plt.close(fig)

    total_trials = int(valid_trials.size)
    valid_count = int(valid_trials.sum())
    valid_fraction = (
        valid_count / total_trials if total_trials > 0 else np.nan
    )
    total_trials_value = total_trials if total_trials > 0 else np.nan

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

    fig_pericue = plot_pericue_path_length(
        eye_timestamp=data.eye_timestamp,
        eye_pos=saccades["eye_pos"],
        cue_times=data.cue_time,
        valid_trials=valid_trials,
    )
    eye_part = (config.eye_name or "Eye").replace(" ", "")
    id_part = str(config.animal_id).strip() if config.animal_id is not None else ""
    animal_label = config.animal_name or config.animal_id
    stem_parts = [part for part in (id_part, eye_part, "pericue_path_length") if part]
    stem = "_".join(stem_parts) if stem_parts else "pericue_path_length"
    for ext in ("png", "svg"):
        fname = _filename_with_animal(f"{stem}.{ext}", animal_label)
        fig_pericue.savefig(config.results_dir / fname, bbox_inches="tight")
    plt.show()
    plt.close(fig_pericue)

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
            "valid_trials": [valid_count],
            "total_trials": [total_trials_value],
            "valid_trial_fraction": [valid_fraction],
        }
    )
    return df


# Usage: python Python/analysis/fixation_session.py SESSION_ID
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session for fixation metrics")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml")
    args = parser.parse_args()
    main(args.session_id)

