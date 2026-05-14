from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

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
    get_session_date_from_path,
)
from eyehead.analysis import _filename_with_animal

# Bonsai coordinate system: monitor width spans -1.7 to 1.7 units (total 3.4)
# Viewing distance equals monitor width (both 47 cm), so the 47s cancel:
#   visual_deg = 2 * arctan(d_bonsai / (2 * 3.4))
_BONSAI_WIDTH_RANGE = 3.4


def bonsai_to_deg(d: float | np.ndarray) -> float | np.ndarray:
    """Convert a diameter in Bonsai units to visual degrees."""
    return np.degrees(2 * np.arctan(np.asarray(d) / (2 * _BONSAI_WIDTH_RANGE)))


def _compute_path_bins(
    eye_ts: np.ndarray,
    xy: np.ndarray,
    cue_ts: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_path, sem_path) arrays over bins for the given cue times.

    Each trial's path is baseline-corrected by subtracting the value in the
    bin nearest to cue onset (t=0) before averaging, so the mean is 0 at
    cue onset and the SEM reflects the corrected variability.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    baseline_idx = int(np.argmin(np.abs(bin_centers)))
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
    # Subtract each trial's cue-onset bin so every trial is baseline-corrected
    # before averaging; trials missing that bin become fully NaN.
    trial_paths -= trial_paths[:, baseline_idx: baseline_idx + 1]
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
    ax.set_ylabel("Path length change from cue onset (deg)")
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

    bin_edges = np.arange(-pre_s - bin_s / 2, post_s + bin_s, bin_s)
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


def plot_pericue_pre_post_summary(
    eye_timestamp: np.ndarray,
    eye_pos: np.ndarray,
    cue_times: np.ndarray,
    *,
    valid_trials: np.ndarray | None = None,
    pre_window: tuple[float, float] = (-1.0, -0.25),
    post_window: tuple[float, float] = (0.25, 1.0),
) -> plt.Figure:
    """Bar plot comparing pre-cue vs post-cue path length for valid and invalid trials.

    Returns
    -------
    fig : plt.Figure
    metrics : dict
        pre_valid_mean, post_valid_mean, pre_invalid_mean, post_invalid_mean,
        active_stabilization = (pre_valid - post_valid) × pre_valid / pre_invalid²
            Equivalent to cue_suppression × selection_bias².  Squaring the
            selection_bias term means a session that catches naturally still
            periods (pre_valid << pre_invalid) is penalised more aggressively
            than a linear product would allow.
    """
    from scipy import stats as scipy_stats

    eye_ts = np.asarray(eye_timestamp).ravel()
    xy = np.asarray(eye_pos)[:, :2]
    cue_ts = np.asarray(cue_times).ravel()
    mask = (
        np.ones(len(cue_ts), dtype=bool)
        if valid_trials is None
        else np.asarray(valid_trials, dtype=bool)
    )

    def _window_path(ct, t0, t1):
        a = np.searchsorted(eye_ts, ct + t0, side="left")
        b = np.searchsorted(eye_ts, ct + t1, side="right")
        if b - a < 2:
            return np.nan
        seg = xy[a:b]
        return float(np.sum(np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1))))

    pre = np.array([_window_path(ct, *pre_window) for ct in cue_ts])
    post = np.array([_window_path(ct, *post_window) for ct in cue_ts])

    pre_v, post_v = pre[mask], post[mask]
    pre_i, post_i = pre[~mask], post[~mask]

    def _paired_p(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 4:
            return np.nan
        return float(scipy_stats.wilcoxon(a[ok], b[ok]).pvalue)

    def _indep_p(a, b):
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 4 or len(b) < 4:
            return np.nan
        return float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)

    p_valid = _paired_p(pre_v, post_v)
    p_invalid = _paired_p(pre_i, post_i)
    p_pre = _indep_p(pre_v, pre_i)

    def _mean_sem(arr):
        ok = arr[~np.isnan(arr)]
        if len(ok) == 0:
            return np.nan, np.nan
        return float(np.mean(ok)), float(np.std(ok, ddof=1) / np.sqrt(len(ok)))

    def _stars(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    mv_pre, sv_pre = _mean_sem(pre_v)
    mv_post, sv_post = _mean_sem(post_v)
    mi_pre, si_pre = _mean_sem(pre_i)
    mi_post, si_post = _mean_sem(post_i)

    # cue_suppression × selection_bias²
    # = (pre_valid - post_valid)/pre_valid × (pre_valid/pre_invalid)²
    # = (pre_valid - post_valid) × pre_valid / pre_invalid²
    # Squaring selection_bias penalises selection bias more aggressively than
    # a linear term so a biased session can't outscore a genuine one.
    active_stabilization = (
        (mv_pre - mv_post) * mv_pre / (mi_pre ** 2)
        if np.isfinite(mi_pre) and np.isfinite(mv_pre) and np.isfinite(mv_post) and mi_pre > 0
        else np.nan
    )

    metrics = {
        "pre_valid_mean": mv_pre,
        "post_valid_mean": mv_post,
        "pre_invalid_mean": mi_pre,
        "post_invalid_mean": mi_post,
        "active_stabilization": active_stabilization,
    }

    # x positions: valid group centred at 0.5, invalid group at 3.0
    xs = [0.0, 1.0, 2.5, 3.5]
    means = [mv_pre, mv_post, mi_pre, mi_post]
    sems = [sv_pre, sv_post, si_pre, si_post]
    colors = ["#7fb3d3", "#1a6fa8", "#f5a08a", "#c0392b"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for x, m, s, c in zip(xs, means, sems, colors):
        ax.bar(x, m, yerr=s, capsize=5, color=c, width=0.75,
               error_kw=dict(lw=1.5, ecolor="0.2"))

    ax.set_xticks(xs)
    ax.set_xticklabels(["Pre-cue", "Post-cue", "Pre-cue", "Post-cue"], fontsize=10)
    ax.set_ylabel("Mean path length (deg)")

    ax.text(0.5, -0.15, "Valid trials", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold",
            color="#1a6fa8")
    ax.text(3.0, -0.15, "Invalid trials", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold",
            color="#c0392b")

    finite_vals = [m + s for m, s in zip(means, sems) if not np.isnan(m + s)]
    if not finite_vals:
        return fig, {}
    y_max = max(finite_vals)
    gap = y_max * 0.08

    def _bracket(x1, x2, y, label):
        ax.plot([x1, x1, x2, x2], [y, y + gap * 0.4, y + gap * 0.4, y],
                lw=1.2, color="0.25")
        ax.text((x1 + x2) / 2, y + gap * 0.5, label,
                ha="center", va="bottom", fontsize=10)

    y1 = y_max + gap
    y2 = y_max + gap
    y3 = y_max + gap * 2.5

    _bracket(xs[0], xs[1], y1, _stars(p_valid))
    _bracket(xs[2], xs[3], y2, _stars(p_invalid))
    _bracket(xs[0], xs[2], y3, f"pre: {_stars(p_pre)}")

    ax.set_ylim(bottom=0, top=y3 + gap * 2)
    ax.set_title(
        "Pre- vs post-cue path length by trial outcome\n"
        f"valid n={mask.sum()}, invalid n={(~mask).sum()}  |  "
        f"fixation={fixation:.3f}"
    )
    fig.tight_layout()
    return fig, metrics


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

    # --- pre/post summary (control figure — remove this block to drop it) ---
    fig_prepost, prepost_metrics = plot_pericue_pre_post_summary(
        eye_timestamp=data.eye_timestamp,
        eye_pos=saccades["eye_pos"],
        cue_times=data.cue_time,
        valid_trials=valid_trials,
    )
    stem_parts = [part for part in (id_part, eye_part, "pericue_pre_post") if part]
    stem = "_".join(stem_parts) if stem_parts else "pericue_pre_post"
    for ext in ("png", "svg"):
        fname = _filename_with_animal(f"{stem}.{ext}", animal_label)
        fig_prepost.savefig(config.results_dir / fname, bbox_inches="tight")
    plt.show()
    plt.close(fig_prepost)
    # --- end control figure ---

    df = pd.DataFrame(
        {
            "session_id": [session_id],
            "animal_id": [config.animal_id],
            "animal_name": [config.animal_name],
            "session_date": [date_str],
            "valid_trials": [valid_count],
            "total_trials": [total_trials_value],
            "valid_trial_fraction": [valid_fraction],
            "pre_valid_mean": [prepost_metrics["pre_valid_mean"]],
            "post_valid_mean": [prepost_metrics["post_valid_mean"]],
            "pre_invalid_mean": [prepost_metrics["pre_invalid_mean"]],
            "post_invalid_mean": [prepost_metrics["post_invalid_mean"]],
            "active_stabilization": [prepost_metrics["active_stabilization"]],
        }
    )
    return df


def plot_psychometric_central_fixation(eot_df: pd.DataFrame, target_df: pd.DataFrame, results_dir: Optional[Path] = None,
                                       animal_id: Optional[str] = None, session_date: str = "",
                                       session_time: Optional[str] = None,
                                       random_walk_chance: Optional[dict] = None) -> plt.Figure:
    """Plot psychometric curve showing success rate as a function of target diameter.

    Creates a plot with:
    - Success rate (%) on y-axis
    - Target diameter on x-axis
    - Error bars showing binomial standard error
    - Number of trials annotated for each diameter
    - Optional: Random walk chance performance per diameter

    Parameters
    ----------
    eot_df : pd.DataFrame
        End-of-trial dataframe containing 'trial_success' (2=success) and 'diameter' columns
    target_df : pd.DataFrame
        Target dataframe containing 'diameter' column
    results_dir : Path, optional
        Directory to save the figure
    animal_id : str, optional
        Animal identifier for filename
    session_date : str, optional
        Session date for title (format: YYYY-MM-DD)
    session_time : str, optional
        Session time for title (format: HH:MM)
    random_walk_chance : dict, optional
        Results from calculate_random_walk_chance_performance() containing
        'by_diameter' key with diameter -> chance rate mapping

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    if 'trial_success' not in eot_df.columns:
        print("Warning: trial_success column not found in eot_df, cannot plot psychometric curve")
        return None
    if 'diameter' not in target_df.columns:
        print("Warning: diameter column not found in target_df, cannot plot psychometric curve")
        return None

    if len(eot_df) != len(target_df):
        print(f"Warning: eot_df has {len(eot_df)} rows but target_df has {len(target_df)} rows")
        min_len = min(len(eot_df), len(target_df))
        combined_df = pd.DataFrame({
            'trial_success': eot_df['trial_success'].iloc[:min_len].values,
            'diameter': target_df['diameter'].iloc[:min_len].values
        })
    else:
        combined_df = pd.DataFrame({
            'trial_success': eot_df['trial_success'].values,
            'diameter': target_df['diameter'].values
        })

    combined_df['diameter'] = bonsai_to_deg(combined_df['diameter'].values)

    diameter_groups = combined_df.groupby('diameter')

    diameters = []
    success_rates = []
    error_bars = []
    n_trials_per_diameter = []

    for diameter, group in diameter_groups:
        n_trials = len(group)
        n_success = np.sum(group['trial_success'] == 2)
        success_rate = n_success / n_trials if n_trials > 0 else 0

        if n_trials > 0:
            std_error = np.sqrt(success_rate * (1 - success_rate) / n_trials)
        else:
            std_error = 0

        diameters.append(diameter)
        success_rates.append(success_rate * 100)
        error_bars.append(std_error * 100)
        n_trials_per_diameter.append(n_trials)

    sorted_indices = np.argsort(diameters)
    diameters = np.array(diameters)[sorted_indices]
    success_rates = np.array(success_rates)[sorted_indices]
    error_bars = np.array(error_bars)[sorted_indices]
    n_trials_per_diameter = np.array(n_trials_per_diameter)[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(diameters, success_rates, yerr=error_bars,
                fmt='o-', markersize=10, linewidth=2, capsize=5, capthick=2,
                color='steelblue', ecolor='darkblue', label='Success Rate')

    if random_walk_chance is not None and 'by_diameter' in random_walk_chance:
        chance_by_diameter = {
            bonsai_to_deg(k): v for k, v in random_walk_chance['by_diameter'].items()
        }
        chance_se_by_diameter = {
            bonsai_to_deg(k): v
            for k, v in random_walk_chance.get('by_diameter_se', {}).items()
        }

        chance_rates = []
        chance_errors = []
        chance_diameters = []
        for d in diameters:
            if d in chance_by_diameter:
                chance_diameters.append(d)
                chance_rates.append(chance_by_diameter[d] * 100)
                se = chance_se_by_diameter.get(d, 0.0) * 100
                chance_errors.append(se)

        if len(chance_rates) > 0:
            ax.errorbar(chance_diameters, chance_rates, yerr=chance_errors,
                       fmt='o--', markersize=8, linewidth=2, capsize=5, capthick=2,
                       color='gray', ecolor='darkgray',
                       markeredgecolor='black', markeredgewidth=1,
                       label='Random Walk Chance', alpha=0.8)

    for i, (d, sr, n) in enumerate(zip(diameters, success_rates, n_trials_per_diameter)):
        text_y = sr + error_bars[i] + 3
        ax.text(d, text_y, f'n={n}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='darkblue')

    ax.axhline(100, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Target Diameter (°)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')

    title = 'Psychometric Curve: Success Rate vs Target Diameter'
    if animal_id:
        title += f'\n{animal_id}'
    if session_date:
        title += f' - {session_date}'
        if session_time:
            title += f' @ {session_time}'
    elif session_time:
        title += f' - {session_time}'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    margin = (diameters.max() - diameters.min()) * 0.1 + 0.5
    ax.set_xlim(diameters.min() - margin, diameters.max() + margin)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{animal_id}_" if animal_id else ""
        date_suffix = f"_{session_date}" if session_date else ""
        filename = f"{prefix}psychometric_central_fixation{date_suffix}.png"
        fig.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
        print(f"Saved psychometric curve to {results_dir / filename}")

    return fig


# Usage: python Python/analysis/fixation_session.py SESSION_ID
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session for fixation metrics")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml")
    args = parser.parse_args()
    main(args.session_id)

