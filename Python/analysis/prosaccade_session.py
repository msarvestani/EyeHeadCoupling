from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

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
    window: Tuple[float, float] = (-1.5, 1.5),
    bin_width: float = 0.1,
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
    max_latency: float = 1.5,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq
    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]
        end_frame = end_frame[mask]

    go_dir_x = -np.array(go_dir_x, dtype=np.float64) #fixing target direction mapping

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]


    latencies, congruent = [], []
    for f, gdx, end_f in zip(go_frame, go_dir_x, end_frame):
        # only saccades after target onset and before this trial actually ends
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        first = np.argmin(rel_valid)
        if rel_valid[first] > max_latency:
            continue
        idx_first = saccade_indices[valid][first]
        latencies.append(rel_valid[first])
        congruent.append(np.sign(dx[idx_first]) == np.sign(gdx))

    return np.array(latencies), np.array(congruent, dtype=bool)


def session_max_trial_duration(
    data: SessionData,
    config,
    mask: Optional[np.ndarray] = None,
    default: float = 1.5,
) -> float:
    """Maximum trial duration for the session, in seconds, derived from data.

    Uses the longest interval from target onset (``go_frame``) to trial end
    (``end_of_trial_frame``). Trials in which the animal never made the
    trial-ending saccade run to the task's time-out, so this maximum recovers
    the response-window ceiling rather than a lone outlier. It is meant to be
    computed once per session and used as the common upper time bound across
    the PSTH, first-saccade latency, fraction-toward, and congruency analyses
    so they all share the same window.

    Falls back to ``default`` seconds if end-of-trial timing is unavailable.
    """
    if data.end_of_trial_frame is None:
        warnings.warn(
            "No end_of_trial data; using default max trial duration of "
            f"{default}s for analysis windows."
        )
        return float(default)
    dur = (
        np.asarray(data.end_of_trial_frame) - np.asarray(data.go_frame)
    ) / config.ttl_freq
    if mask is not None:
        dur = dur[mask]
    dur = dur[np.isfinite(dur) & (dur > 0)]
    if dur.size == 0:
        return float(default)
    return float(np.max(dur))


def first_saccade_indices_by_direction(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    max_latency: float = 1.5,
    frames_key: str = "saccade_frames_xy",
    indices_key: str = "saccade_indices_xy",
) -> Dict[str, np.ndarray]:
    """First-saccade eye indices per trial, grouped by stimulus direction.

    Selects, for every trial, the *first* saccade in the same reward window
    used by :func:`find_first_saccade_per_trial` /
    :func:`analyze_latency_by_outcome` — i.e. between target onset
    (``go_frame``) and that trial's actual end (``end_of_trial_frame``),
    capped at ``max_latency`` — and returns the corresponding index into the
    eye-position arrays. This is the same saccade the online task scored for
    reward.

    ``frames_key`` / ``indices_key`` select which saccade stream to search:
    the translational stream (``saccade_frames_xy`` / ``saccade_indices_xy``,
    the default) or the torsional stream (``saccade_frames_theta`` /
    ``saccade_indices_theta``). Returns an empty dict if that stream is absent
    or empty.

    Trials are grouped into direction labels exactly as :func:`organize_stims`
    groups them, so the returned dict can be dropped straight into
    :func:`eyehead.sort_saccades` in place of its fixed-window search.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    go_dir_y = data.go_direction_y
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq

    saccade_frames = saccades.get(frames_key)
    saccade_indices = saccades.get(indices_key)
    if saccade_frames is None or saccade_indices is None or len(saccade_frames) == 0:
        return {}
    saccade_frames = np.asarray(saccade_frames)
    saccade_indices = np.asarray(saccade_indices)

    first_idx = np.full(len(go_frame), -1, dtype=int)
    for i, (f, end_f) in enumerate(zip(go_frame, end_frame)):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        first = np.argmin(rel_valid)
        if rel_valid[first] > max_latency:
            continue
        first_idx[i] = saccade_indices[valid][first]

    have = first_idx >= 0
    groups: Dict[str, np.ndarray] = {}
    if go_dir_x is not None and np.any(np.asarray(go_dir_x) != 0):
        go_dir_x = np.asarray(go_dir_x)
        groups["Left"] = first_idx[have & (go_dir_x < 0)]
        groups["Right"] = first_idx[have & (go_dir_x > 0)]
    if go_dir_y is not None and np.any(np.asarray(go_dir_y) != 0):
        go_dir_y = np.asarray(go_dir_y)
        groups["Down"] = first_idx[have & (go_dir_y < 0)]
        groups["Up"] = first_idx[have & (go_dir_y > 0)]
    if not groups:
        groups["All"] = first_idx[have]
    return groups


def find_precue_saccade_per_trial(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    window: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Latency and target-congruency of the last saccade before target onset.

    Serves as a pre-cue/pre-target control: since the target hasn't appeared
    yet, any lateral bias in this saccade's direction can't reflect
    target-directed behaviour, so congruency here should sit near chance.

    Returns
    -------
    latencies : ndarray
        Time (s), negative, from target onset to each trial's last saccade
        in the preceding ``window`` seconds. Trials with none are omitted.
    congruent : ndarray of bool
        Whether that saccade's direction happened to match the (not-yet-seen)
        target's direction.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]

    go_dir_x = -np.array(go_dir_x, dtype=np.float64)  # fixing target direction mapping

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]

    latencies, congruent = [], []
    for f, gdx in zip(go_frame, go_dir_x):
        rel = (saccade_frames - f) / ttl_freq
        valid = (rel < 0) & (rel >= -window)
        if not np.any(valid):
            continue
        rel_valid = rel[valid]
        last = np.argmax(rel_valid)  # closest to (but before) target onset
        idx_last = saccade_indices[valid][last]
        latencies.append(rel_valid[last])
        congruent.append(np.sign(dx[idx_last]) == np.sign(gdx))

    return np.array(latencies), np.array(congruent, dtype=bool)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (center - margin, center + margin)


def congruency_in_window(
    latencies: np.ndarray,
    congruent: np.ndarray,
    window: Tuple[float, float] = (0.15, 0.45),
) -> Tuple[float, int, float, float]:
    """Fraction congruent (with Wilson 95% CI) for saccades in a fixed latency window."""
    sel = (latencies >= window[0]) & (latencies < window[1])
    n = int(np.count_nonzero(sel))
    if n == 0:
        return np.nan, 0, np.nan, np.nan
    frac = float(np.mean(congruent[sel]))
    ci_lo, ci_hi = wilson_ci(int(np.sum(congruent[sel])), n)
    return frac, n, ci_lo, ci_hi

def fraction_toward_target_by_latency(
    latencies: np.ndarray,
    congruent: np.ndarray,
    window_span: Tuple[float, float] = (0.2, 1.0),
    win_width: float = 0.3,
    step: float = 0.05,
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


def analyze_latency_by_outcome(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    mask: Optional[np.ndarray] = None,
    max_latency: float = 1.5,
) -> Dict[str, np.ndarray]:
    """Per-trial first-saccade latency, paired with two independent "correct" labels.

    For every trial, finds the first saccade between target onset and that
    trial's actual end (``data.end_of_trial_frame``) and records its latency,
    together with:

    - ``congruent``: whether that saccade's direction matched the target's
      (same definition as :func:`find_first_saccade_per_trial`).
    - ``trial_success``: the independently-logged outcome from
      ``data.trial_success`` (the ``end_of_trial`` CSV's success column), or
      NaN if that file didn't load for this session.

    Trials with no saccade between target onset and trial end are excluded
    from the returned arrays but counted in ``n_no_saccade``.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    end_frame = data.end_of_trial_frame

    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq

    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]
        end_frame = end_frame[mask]


    go_dir_x = -np.array(go_dir_x, dtype=np.float64)  # fixing target direction mapping

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]

    latencies, congruent = [], []
    n_no_saccade = 0
    for f, gdx, end_f in zip(go_frame, go_dir_x, end_frame):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            n_no_saccade += 1
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        first = np.argmin(rel_valid)
        if rel_valid[first] > max_latency:
            n_no_saccade += 1
            continue
        idx_first = saccade_indices[valid][first]
        latencies.append(rel_valid[first])
        congruent.append(np.sign(dx[idx_first]) == np.sign(gdx))

    return {
        "latencies": np.array(latencies),
        "congruent": np.array(congruent, dtype=bool),
        "n_no_saccade": n_no_saccade,
        "n_total": len(go_frame),
    }


def plot_latency_by_outcome(
    result: Dict[str, np.ndarray],
    config,
    bins: int = 20,
    show_plots: bool = True,
) -> plt.Figure:

    """First-saccade latency split by saccade-target congruency.

    Top: overlaid histograms of latency for congruent ("correct", toward the
    target) vs. incongruent ("incorrect") first saccades. Bottom: empirical
    CDFs of those same two distributions, using the matching colour scheme."""
    latencies = result["latencies"]
    congruent = result["congruent"]
    n_no_saccade = result["n_no_saccade"]

    correct = latencies[congruent]
    incorrect = latencies[~congruent]
    n_correct = correct.size
    n_incorrect = incorrect.size

    fig, (ax_hist, ax_cdf) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    ax_hist.hist(correct, bins=bins, alpha=0.6, color="tab:green", label="correct")
    ax_hist.hist(incorrect, bins=bins, alpha=0.6, color="tab:red", label="incorrect")
    ax_hist.set_ylabel("Trial count")
    ax_hist.set_title(
        f"Correct = saccade toward target\n{n_correct} correct, {n_incorrect} incorrect"
    )
    ax_hist.legend(fontsize=8)

    def _cdf(ax, values, color, label):
        if values.size == 0:
            return
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", color=color, label=label)

    _cdf(ax_cdf, correct, "tab:green", "correct")
    _cdf(ax_cdf, incorrect, "tab:red", "incorrect")
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel("First-saccade latency (s)")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("Latency CDF")
    ax_cdf.legend(fontsize=8)

    fig.suptitle(
        f"{config.animal_name or ''} {config.session_name}  \u2014  "
        f"{result['n_total']} trials, {len(latencies)} with a saccade, "
        f"{n_no_saccade} excluded (no saccade)"
    )
    fig.tight_layout()
    fig.savefig(config.results_dir / f"{config.session_name}_latency_by_outcome.png",
                dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig

def plot_psth_and_congruency(
    psth_centers: np.ndarray,
    psth_rate: np.ndarray,
    psth_ci: np.ndarray,
    n_trials: int,
    latency_centers: np.ndarray,
    fraction_toward: np.ndarray,
    n_per_window: np.ndarray,
    frac: float,
    n_window: int,
    ci_lo: float,
    ci_hi: float,
    precue_frac: float,
    precue_n: int,
    config,
    window: Tuple[float, float] = (0.15, 0.45),
    show_plots: bool = True,
) -> plt.Figure:
    """Three-panel figure: target-aligned rate PSTH, accuracy-by-latency, and windowed congruency vs. pre-cue control."""
    fig, (ax_rate, ax_frac, ax_summary) = plt.subplots(1, 3, figsize=(14, 4))

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

    if n_window > 0:
        ax_summary.errorbar(
            frac, 0,
            xerr=[[frac - ci_lo], [ci_hi - frac]],
            fmt="D", color="tab:red", ecolor="tab:red", capsize=4, ms=9,
        )
        ax_summary.text(max(ci_hi, frac) + 0.03, 0, f"{frac:.0%}  (n={n_window})",
                         va="center", fontsize=9, color="dimgray")

    if n_window > 0:
        ax_summary.errorbar(
            frac, 0,
            xerr=[[frac - ci_lo], [ci_hi - frac]],
            fmt="D", color="tab:red", ecolor="tab:red", capsize=4, ms=9,
        )
        ax_summary.text(max(ci_hi, frac) + 0.03, 0.15, f"{frac:.0%}  (n={n_window})",
                         va="center", fontsize=9, color="tab:red")
    if precue_n > 0 and not np.isnan(precue_frac):
        ax_summary.plot(precue_frac, 0, "o", mfc="none", mec="gray", ms=9)
        ax_summary.text(precue_frac, -0.15, f"pre-cue: {precue_frac:.0%} (n={precue_n})",
                         va="center", ha="center", fontsize=8, color="dimgray")
    ax_summary.axvline(0.5, color="gray", ls="--", lw=1)
    ax_summary.set_xlim(0.0, 1.15)
    ax_summary.set_ylim(-0.5, 0.5)
    ax_summary.set_yticks([])
    ax_summary.set_xlabel(f"Fraction toward target, {window[0]:.2f}\u2013{window[1]:.2f} s")
    ax_summary.set_title("Congruency vs. pre-cue control")
    
    ax_summary.axvline(0.5, color="gray", ls="--", lw=1)
    ax_summary.set_xlim(0.0, 1.15)
    ax_summary.set_yticks([])
    ax_summary.set_xlabel(f"Fraction toward target, {window[0]:.2f}\u2013{window[1]:.2f} s")
    ax_summary.set_title("Congruency vs. pre-cue control")

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
    mask_horizontal = data.go_direction_x != 0

    # Single response window used as the common upper time bound across all
    # of the analyses below so they share one window. This is the configured
    # ``saccade_win`` (the same value shown in the first figure's title), i.e.
    # how long after target onset a first/response saccade is considered.
    # (``session_max_trial_duration(data, config, mask=mask_horizontal)`` is
    # available if you'd rather derive it per session from end_of_trial.)
    max_trial_time = saccade_cfg.saccade_win

    first_saccades = first_saccade_indices_by_direction(
        data, saccades, config, max_latency=max_trial_time
    )
    first_saccades_theta = first_saccade_indices_by_direction(
        data, saccades, config, max_latency=max_trial_time,
        frames_key="saccade_frames_theta", indices_key="saccade_indices_theta",
    )
    sorted_data,left_angle,right_angle,fig_sorted, _ = sort_saccades(
        config, saccade_cfg, saccades, stim_type=stim_type,
        first_saccade_indices=first_saccades,
        first_saccade_indices_theta=first_saccades_theta or None,
        plot=True,
    )
    if fig_sorted is not None:
        plt.close(fig_sorted)

    psth_centers, psth_rate, psth_ci, n_trials_psth = compute_saccade_psth(
        data, saccades, config, window=(-max_trial_time, max_trial_time),
        mask=mask_horizontal,
    )
    latencies, congruent = find_first_saccade_per_trial(
        data, saccades, config, max_latency=max_trial_time, mask=mask_horizontal
    )
    latency_centers, fraction_toward, n_per_window = fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0.2, max_trial_time)
    )

    window = (0.15, 0.45)
    frac, n_window, ci_lo, ci_hi = congruency_in_window(latencies, congruent, window=window)
    precue_latencies, precue_congruent = find_precue_saccade_per_trial(
        data, saccades, config, mask=mask_horizontal
    )
    precue_frac = float(np.mean(precue_congruent)) if len(precue_congruent) else np.nan
    precue_n = len(precue_congruent)

    plot_psth_and_congruency(
        psth_centers, psth_rate, psth_ci, n_trials_psth,
        latency_centers, fraction_toward, n_per_window,
        frac, n_window, ci_lo, ci_hi, precue_frac, precue_n,
        config, window=window,
    )

    latency_outcome = analyze_latency_by_outcome(
        data, saccades, config, mask=mask_horizontal, max_latency=max_trial_time
    )
    plot_latency_by_outcome(latency_outcome, config)


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

