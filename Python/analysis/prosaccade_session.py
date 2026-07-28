from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Put the repo's “Python” folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session_or_path

from eyehead import (
    SaccadeConfig,
    SessionData,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    organize_stims,
    sort_saccades,
    get_session_date_from_path,
)

def compute_saccade_psth(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_width: float = 0.02,
    mask: Optional[np.ndarray] = None,
    n_boot: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Target-aligned saccade rate (Hz), with a bootstrap confidence band.

    Parameters
    ----------
    data : SessionData
        Must provide ``go_frame`` (target-onset frame numbers).
    saccades : dict
        Output of :func:`eyehead.detect_saccades`; must contain
        ``saccade_frames_xy``.
    config : SessionConfig
        Must provide ``ttl_freq`` (frames per second), used to convert frame
        counts to seconds, matching the convention used elsewhere in this
        repo (e.g. :func:`eyehead.sort_saccades`).
    window : (float, float)
        Time window relative to target onset, in seconds.
    bin_width : float
        Histogram bin width, in seconds.
    mask : boolean array, optional
        Selects a subset of target onsets (e.g. horizontal-target trials
        only). Defaults to all target onsets in ``data.go_frame``.
    n_boot : int
        Trial-resampling bootstrap iterations for the CI band; set to 0 to
        skip the bootstrap (``ci`` then just repeats ``rate``).

    Returns
    -------
    bin_centers, rate, ci, n_trials
        ``ci`` is a ``(2, n_bins)`` array of (lower, upper) 95% bootstrap
        bounds on the rate.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    if mask is not None:
        go_frame = go_frame[mask]
    n_trials = len(go_frame)

    saccade_times = saccades["saccade_frames_xy"] / ttl_freq
    edges = np.arange(window[0], window[1] + bin_width, bin_width)
    bin_centers = edges[:-1] + bin_width / 2

    def _rate_for(frames: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(bin_centers))
        for f in frames:
            rel = saccade_times - f / ttl_freq
            in_window = rel[(rel >= window[0]) & (rel < window[1])]
            counts += np.histogram(in_window, bins=edges)[0]
        return counts / (len(frames) * bin_width) if len(frames) else counts

    rate = _rate_for(go_frame)

    if n_trials and n_boot:
        rng = np.random.default_rng()
        boot_rates = np.empty((n_boot, len(bin_centers)))
        for b in range(n_boot):
            resampled = rng.choice(go_frame, size=n_trials, replace=True)
            boot_rates[b] = _rate_for(resampled)
        ci = np.percentile(boot_rates, [2.5, 97.5], axis=0)
    else:
        ci = np.tile(rate, (2, 1))

    return bin_centers, rate, ci, n_trials


def find_first_saccade_per_trial(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    max_latency: float = 1.0,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Latency and target-congruency of the first saccade after each target onset.

    A trial's first saccade is congruent when its horizontal direction (sign
    of ``eye_vel``'s x-component at the saccade index) matches the sign of
    that trial's ``go_direction_x``.

    Returns
    -------
    latencies : ndarray
        Time (s) from target onset to each trial's first saccade. Trials with
        no saccade within ``max_latency`` are omitted.
    congruent : ndarray of bool
        Whether that first saccade went toward the target.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]

    latencies, congruent = [], []
    for f, gdx in zip(go_frame, go_dir_x):
        rel = (saccade_frames - f) / ttl_freq
        valid = rel > 0
        if not np.any(valid):
            continue
        rel_valid = rel[valid]
        first = np.argmin(rel_valid)
        if rel_valid[first] > max_latency:
            continue
        idx_first = saccade_indices[valid][first]
        latencies.append(rel_valid[first])
        congruent.append(np.sign(dx[idx_first]) == np.sign(gdx))

    return np.array(latencies), np.array(congruent, dtype=bool)


def fraction_toward_target_by_latency(
    latencies: np.ndarray,
    congruent: np.ndarray,
    window_span: Tuple[float, float] = (0.0, 0.8),
    win_width: float = 0.15,
    step: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fraction of first saccades toward the target, in sliding latency windows."""
    centers = np.arange(window_span[0], window_span[1] + step, step)
    fraction = np.full(len(centers), np.nan)
    n_per_window = np.zeros(len(centers), dtype=int)

    for i, c in enumerate(centers):
        lo, hi = c - win_width / 2, c + win_width / 2
        sel = (latencies >= lo) & (latencies < hi)
        n_per_window[i] = np.count_nonzero(sel)
        if n_per_window[i] > 0:
            fraction[i] = np.mean(congruent[sel])

    return centers, fraction, n_per_window


def plot_psth_and_congruency(
    psth_centers: np.ndarray,
    psth_rate: np.ndarray,
    psth_ci: np.ndarray,
    n_trials: int,
    latency_centers: np.ndarray,
    fraction_toward: np.ndarray,
    n_per_window: np.ndarray,
    config,
    show_plots: bool = True,
) -> plt.Figure:
    """Two-panel figure: target-aligned rate PSTH, and accuracy-by-latency."""
    fig, (ax_rate, ax_frac) = plt.subplots(1, 2, figsize=(10, 4))

    ax_rate.fill_between(psth_centers, psth_ci[0], psth_ci[1], color="tab:blue", alpha=0.25, lw=0)
    ax_rate.plot(psth_centers, psth_rate, color="tab:blue", lw=1.4)
    ax_rate.axvline(0, color="k", lw=0.8)
    ax_rate.set_xlabel("Time from target onset (s)")
    ax_rate.set_ylabel("Saccade rate (Hz)")
    ax_rate.set_title(f"Target-aligned saccade rate (n={n_trials} trials)")

    valid = n_per_window > 0
    ax_frac.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax_frac.plot(latency_centers[valid], fraction_toward[valid], "-o", color="tab:green", ms=3)
    ax_frac.set_ylim(0, 1)
    ax_frac.set_xlabel("Window centre, time from target (s)")
    ax_frac.set_ylabel("Fraction of first saccades toward target")
    ax_frac.set_title("Saccade accuracy vs. latency")

    fig.suptitle(f"{config.animal_name or ''} {config.session_name}".strip())
    fig.tight_layout()
    fig.savefig(config.results_dir / f"{config.session_name}_psth_congruency.png",
                dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def main(session_id: str) -> pd.DataFrame:
    """Run the full analysis pipeline for ``session_id``.

    Parameters
    ----------
    session_id:
        Identifier of the session to analyse, either a manifest session ID
        or a direct path to a session folder.
    """
    config = load_session_or_path(session_id)
    session_id = config.session_id
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

    mask_horizontal = data.go_direction_x != 0
    psth_centers, psth_rate, psth_ci, n_trials_psth = compute_saccade_psth(
        data, saccades, config, mask=mask_horizontal
    )
    latencies, congruent = find_first_saccade_per_trial(
        data, saccades, config, mask=mask_horizontal
    )
    latency_centers, fraction_toward, n_per_window = fraction_toward_target_by_latency(
        latencies, congruent
    )
    plot_psth_and_congruency(
        psth_centers, psth_rate, psth_ci, n_trials_psth,
        latency_centers, fraction_toward, n_per_window,
        config,
    )


    df = pd.DataFrame(
        {
            "session_id": [session_id] * len(indices),
            "session_date": [date_str] * len(indices),
            "saccade_frame_xy": saccade_frames,
            "saccade_index_xy": indices,

        }
    )
    return df,left_angle,right_angle


# Usage: python Python/analysis/prosaccade_session.py SESSION_ID_OR_PATH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml, or a direct path to a session folder")
    args = parser.parse_args()
    main(args.session_id)

