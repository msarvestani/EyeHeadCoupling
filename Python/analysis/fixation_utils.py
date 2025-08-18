from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle


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
    """Pair cue and go events and visualise eye position traces.

    Parameters
    ----------
    eye_frame : array-like
        Frame numbers for each eye sample.
    eye_pos : array-like, shape (N, 2)
        Eye centre coordinates in degrees.
    eye_timestamp : array-like
        Timestamps for ``eye_pos``.
    cue_frame, cue_time, go_frame, go_time : array-like
        Frame numbers and timestamps of cue and go events.
    max_interval_s : float, default 1.0
        Maximum allowed cue→go interval for a trial to be considered valid.
    color_all, s_all, alpha_all :
        Matplotlib styling for all eye samples.
    s_subset, alpha_subset :
        Styling for samples within valid trials.
    cmap_name : str, default "tab20"
        Colormap used to colour individual trials.
    results_dir : Path, optional
        If given, save the generated figure here.
    session_name : str, optional
        Used in the saved filename when ``results_dir`` is provided.
    eye_name : str, default "Eye"
        Label used in the saved filename.

    Returns
    -------
    pairs_cf, pairs_gf : ndarray of int
        Paired cue and go frame indices.
    pairs_ct, pairs_gt : ndarray of float
        Paired cue and go timestamps.
    pairs_dt : ndarray of float
        Time difference between paired events (go - cue).
    valid_trials : ndarray of bool
        Mask indicating which pairs fall within ``max_interval_s``.
    fig, ax : Figure and Axes
        Handles to the generated scatter plot.
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
    ax.set_title("Eye positions between cue and go")
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize="small", loc="upper right")

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        fname = f"{session_name or 'session'}_{(eye_name or 'Eye').replace(' ', '')}_cue_go_timepaired.png"
        fig.savefig(results_dir / fname, dpi=300, bbox_inches="tight")

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
    """Compare eye stability during fixation windows to random times.

    Parameters
    ----------
    eye_timestamp : array-like
        Sample timestamps in seconds.
    eye_pos : array-like, shape (N, 2)
        Eye centre coordinates.
    pairs_ct, pairs_gt : array-like
        Cue and go timestamps for each paired trial.
    valid_trials : array-like of bool
        Mask indicating which trials to use for the analysis.
    plot : bool, default ``True``
        If ``True`` a figure summarising the comparison is included in the
        returned dictionary.
    rng_seed : int, default ``0``
        Seed used when sampling random time windows.

    Returns
    -------
    dict or None
        ``None`` is returned when no valid fixation windows are available.
        Otherwise a dictionary containing per-window metrics and a summary of
        fixation vs random statistics is returned.  When ``plot`` is ``True``
        a ``figure`` entry with a :class:`matplotlib.figure.Figure` is added.
    """
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
