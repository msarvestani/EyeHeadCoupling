from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle

# Put the repo's "Python" folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session

from eyehead import (
    SaccadeConfig,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
)


def plot_eye_fixations_between_cue_and_go_by_trial(
    eye_frame: np.ndarray,
    eye_pos: np.ndarray,
    eye_timestamp: np.ndarray,
    cue_frame: np.ndarray,
    cue_time: np.ndarray,
    go_frame: np.ndarray,
    go_time: np.ndarray,
    max_interval_s: float = 1.0,
    color_all: str = "0.85",
    s_all: int = 2,
    alpha_all: float = 0.25,
    s_subset: int = 5,
    alpha_subset: float = 0.9,
    cmap_name: str = "tab20",
    results_dir: Optional[Path] = None,
    session_name: Optional[str] = None,
    eye_name: str = "Eye",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, plt.Figure, plt.Axes]:
    """Plot eye positions between cue and go for each trial.

    Returns paired cue/go frames and times, their deltas, a boolean mask of
    valid trials, and the generated figure/axes.
    """
    eye_ts = np.asarray(eye_timestamp).ravel()
    eye_x = np.asarray(eye_pos[:, 0]).ravel()
    eye_y = np.asarray(eye_pos[:, 1]).ravel()
    cue_frame = np.asarray(cue_frame).astype(int).ravel()
    cue_time = np.asarray(cue_time).astype(float).ravel()
    go_frame = np.asarray(go_frame).astype(int).ravel()
    go_time = np.asarray(go_time).astype(float).ravel()

    ci = np.argsort(cue_time)
    cue_time, cue_frame = cue_time[ci], cue_frame[ci]
    gi = np.argsort(go_time)
    go_time, go_frame = go_time[gi], go_frame[gi]

    cue_time_on, cue_frame_on = cue_time, cue_frame
    go_time_on, go_frame_on = go_time, go_frame

    pairs_ct: list[float] = []
    pairs_gt: list[float] = []
    pairs_cf: list[int] = []
    pairs_gf: list[int] = []
    pairs_dt: list[float] = []
    gptr = 0
    for ct, cf in zip(cue_time_on, cue_frame_on):
        while gptr < len(go_time_on) and go_time_on[gptr] < ct:
            gptr += 1
        if gptr >= len(go_time_on):
            break
        dt = float(go_time_on[gptr] - ct)
        pairs_ct.append(ct)
        pairs_gt.append(go_time_on[gptr])
        pairs_cf.append(cf)
        pairs_gf.append(int(go_frame_on[gptr]))
        pairs_dt.append(dt)
        gptr += 1

    pairs_ct = np.asarray(pairs_ct)
    pairs_gt = np.asarray(pairs_gt)
    pairs_cf = np.asarray(pairs_cf, dtype=int)
    pairs_gf = np.asarray(pairs_gf, dtype=int)
    pairs_dt = np.asarray(pairs_dt)

    valid_trials = (pairs_dt >= 0) & (pairs_dt < max_interval_s)

    cmap = cm.get_cmap(cmap_name)
    base_colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    color_cycle = cycle(base_colors)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(eye_x, eye_y, s=s_all, c=color_all, alpha=alpha_all, label="All eye centers")

    legend_handles = []
    trial_num = 0
    for ct, gt, ok, dt in zip(pairs_ct, pairs_gt, valid_trials, pairs_dt):
        if not ok:
            continue
        a = np.searchsorted(eye_ts, min(ct, gt), side="left")
        b = np.searchsorted(eye_ts, max(ct, gt), side="right")
        if b <= a:
            continue
        col = next(color_cycle)
        h = ax.scatter(
            eye_x[a:b],
            eye_y[a:b],
            s=s_subset,
            c=[col],
            alpha=alpha_subset,
            label=f"Trial {trial_num} (Δt={dt:.2f}s)",
        )
        legend_handles.append(h)
        trial_num += 1

    ax.set_aspect("equal")
    ax.set_xlabel("Eye center X (deg)")
    ax.set_ylabel("Eye center Y (deg)")
    ax.set_title(
        f"Eye centers: all vs. time-paired cue→go windows (<{max_interval_s:.1f}s)"
    )

    if len(legend_handles) > 10:
        ax.legend(
            [ax.collections[0], *legend_handles[:10]],
            ["All eye centers", *[lh.get_label() for lh in legend_handles[:10]]],
            frameon=False,
            loc="best",
        )
    else:
        ax.legend(frameon=False, loc="best")

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        fname = f"{session_name or 'session'}_{(eye_name or 'Eye').replace(' ', '')}_cue_go_timepaired.png"
        fig.savefig(results_dir / fname, dpi=300, bbox_inches="tight")

    print(f"Raw cues: {cue_time.size} → deduped: {cue_time_on.size}")
    print(f"Raw gos : {go_time.size}  → deduped: {go_time_on.size}")
    if pairs_dt.size:
        print(
            "Paired trials: {} | dt min/median/max = {:.3f} / {:.3f} / {:.3f} s".format(
                pairs_dt.size, np.nanmin(pairs_dt), np.nanmedian(pairs_dt), np.nanmax(pairs_dt)
            )
        )
        print(f"Passing (<{max_interval_s:.2f}s): {valid_trials.sum()} trials")

    return pairs_cf, pairs_gf, pairs_ct, pairs_gt, pairs_dt, valid_trials, fig, ax


def quantify_fixation_stability_vs_random(
    eye_timestamp: np.ndarray,
    eye_pos: np.ndarray,
    pairs_ct: np.ndarray,
    pairs_gt: np.ndarray,
    valid_trials: np.ndarray,
    *,
    plot: bool = True,
    rng_seed: int = 0,
) -> Optional[Dict[str, object]]:
    """Compare eye stability during fixation windows to random times."""
    ts = np.asarray(eye_timestamp, dtype=float).ravel()
    x = np.asarray(eye_pos[:, 0]).ravel()
    y = np.asarray(eye_pos[:, 1]).ravel()
    if not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts)
        ts, x, y = ts[order], x[order], y[order]

    ct = np.asarray(pairs_ct, dtype=float).ravel()
    gt = np.asarray(pairs_gt, dtype=float).ravel()
    ok = np.asarray(valid_trials, dtype=bool).ravel()

    fix_windows = [(c, g) for c, g, v in zip(ct, gt, ok) if v and (g > c)]
    if len(fix_windows) == 0:
        print("No valid fixation windows. Nothing to compute.")
        return None

    fix_windows = sorted(fix_windows, key=lambda w: w[0])
    merged: list[list[float]] = []
    for s, e in fix_windows:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    fix_windows = [(s, e) for s, e in merged]

    def window_metrics(t0: float, t1: float) -> Tuple[float, float, float]:
        a = np.searchsorted(ts, t0, side="left")
        b = np.searchsorted(ts, t1, side="right")
        if b - a < 2:
            return np.nan, np.nan, np.nan
        dx = np.diff(x[a:b])
        dy = np.diff(y[a:b])
        dt = np.diff(ts[a:b])
        m = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt > 0)
        if not np.any(m):
            return np.nan, np.nan, np.nan
        step_disp = np.hypot(dx[m], dy[m])
        speed = step_disp / dt[m]
        drift = np.hypot(x[b - 1] - x[a], y[b - 1] - y[a])
        return float(step_disp.mean()), float(speed.mean()), float(drift)

    orig_fix_windows = [(c, g) for c, g, v in zip(ct, gt, ok) if v and (g > c)]
    fix_len = np.array([g - c for c, g in orig_fix_windows], dtype=float)

    fix_mean_step = np.empty(len(orig_fix_windows))
    fix_mean_speed = np.empty(len(orig_fix_windows))
    fix_drift = np.empty(len(orig_fix_windows))
    for i, (c, g) in enumerate(orig_fix_windows):
        fix_mean_step[i], fix_mean_speed[i], fix_drift[i] = window_metrics(c, g)

    session_start, session_end = float(ts[0]), float(ts[-1])
    allowed = []
    cursor = session_start
    for s, e in fix_windows:
        if s > cursor:
            allowed.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < session_end:
        allowed.append((cursor, session_end))

    rng = np.random.default_rng(rng_seed)

    def sample_random_window(duration: float) -> Optional[Tuple[float, float]]:
        candidates = [(a, b) for (a, b) in allowed if (b - a) >= duration]
        if not candidates:
            return None
        a, b = candidates[rng.integers(0, len(candidates))]
        start = float(a) + rng.random() * float((b - a) - duration)
        return start, start + duration

    rnd_mean_step = np.empty(len(orig_fix_windows))
    rnd_mean_speed = np.empty(len(orig_fix_windows))
    rnd_drift = np.empty(len(orig_fix_windows))

    for i, L in enumerate(fix_len):
        rw = sample_random_window(L)
        if rw is None:
            rnd_mean_step[i] = rnd_mean_speed[i] = rnd_drift[i] = np.nan
        else:
            rnd_mean_step[i], rnd_mean_speed[i], rnd_drift[i] = window_metrics(*rw)

    def nice_stats(arr: np.ndarray) -> Tuple[float, float, int]:
        arr = np.asarray(arr, dtype=float)
        m = np.isfinite(arr)
        if not m.any():
            return np.nan, np.nan, 0
        vals = arr[m]
        return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size)), int(vals.size)

    ms_fix, se_fix, n_fix = nice_stats(fix_mean_step)
    ms_rnd, se_rnd, n_rnd = nice_stats(rnd_mean_step)
    sp_fix, se_spf, _ = nice_stats(fix_mean_speed)
    sp_rnd, se_spr, _ = nice_stats(rnd_mean_speed)
    dr_fix, se_drf, _ = nice_stats(fix_drift)
    dr_rnd, se_drr, _ = nice_stats(rnd_drift)

    print("=== Stability summary (mean ± s.e.m.) ===")
    print(
        f"Mean step displacement (deg):  fix {ms_fix:.3f} ± {se_fix:.3f}   vs   rand {ms_rnd:.3f} ± {se_rnd:.3f}  (n={n_fix} pairs)"
    )
    print(
        f"Mean speed (deg/s):            fix {sp_fix:.3f} ± {se_spf:.3f}   vs   rand {sp_rnd:.3f} ± {se_spr:.3f}"
    )
    print(
        f"Net drift (deg):               fix {dr_fix:.3f} ± {se_drf:.3f}   vs   rand {dr_rnd:.3f} ± {se_drr:.3f}"
    )

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        pairs = [
            ("Mean step (deg)", fix_mean_step, rnd_mean_step),
            ("Mean speed (deg/s)", fix_mean_speed, rnd_mean_speed),
            ("Net drift (deg)", fix_drift, rnd_drift),
        ]
        for ax, (title, a, b) in zip(axes, pairs):
            m = np.isfinite(a) & np.isfinite(b)
            ax.scatter(a[m], b[m], s=10, alpha=0.6)
            lo = np.nanmin(np.concatenate([a[m], b[m]]))
            hi = np.nanmax(np.concatenate([a[m], b[m]]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, alpha=0.5)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            ax.set_xlabel("Fixation")
            ax.set_ylabel("Random")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle("Fixation vs. random windows (paired, equal duration)")
    else:
        fig = None

    return {
        "fix_mean_step_px": fix_mean_step,
        "rnd_mean_step_px": rnd_mean_step,
        "fix_mean_speed_px_s": fix_mean_speed,
        "rnd_mean_speed_px_s": rnd_mean_speed,
        "fix_net_drift_px": fix_drift,
        "rnd_net_drift_px": rnd_drift,
        "summary": {
            "mean_step_fix_mean±sem": (ms_fix, se_fix, n_fix),
            "mean_step_rand_mean±sem": (ms_rnd, se_rnd, n_rnd),
            "mean_speed_fix_mean±sem": (sp_fix, se_spf),
            "mean_speed_rand_mean±sem": (sp_rnd, se_spr),
            "net_drift_fix_mean±sem": (dr_fix, se_drf),
            "net_drift_rand_mean±sem": (dr_rnd, se_drr),
        },
        "figure": fig,
    }


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
