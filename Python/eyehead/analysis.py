from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.signal import medfilt
from itertools import cycle

from utils.session_loader import SessionConfig

from .filters import interpolate_nans
from .plotting import vector_to_rgb, plot_angle_distribution, plot_linear_histogram
from .io import SessionData


@dataclass
class SaccadeConfig:
    """Configuration parameters for :func:`detect_saccades`."""

    saccade_threshold: float
    saccade_threshold_torsion: float
    blink_threshold: float = 10.0
    blink_detection: int = 1
    saccade_win: float = 0.7


def calibrate_eye_position(data: SessionData, config: SessionConfig) -> np.ndarray:
    """Calibrate eye position using eyelid markers and gaze samples."""
    if data.origin_of_eye_coordinate is None or data.ellipse_center_xy is None:
        raise ValueError(
            "Missing eye marker data: origin_of_eye_coordinate or ellipse_center_xy"
        )

    oc = data.origin_of_eye_coordinate
    ec = data.ellipse_center_xy

    marker1_x, marker1_y = oc[:, 2], oc[:, 3]
    marker2_x, marker2_y = oc[:, 4], oc[:, 5]
    gaze_x, gaze_y = ec[:, 2], ec[:, 3]

    eye_origin = np.column_stack(((marker1_x + marker2_x) / 2.0,
                                  (marker1_y + marker2_y) / 2.0))
    eye_camera = np.column_stack((gaze_x - eye_origin[:, 0],
                                  gaze_y - eye_origin[:, 1])).astype(np.float64, copy=False)

    eye_camera[:, 0] = medfilt(eye_camera[:, 0], kernel_size=3)
    eye_camera[:, 1] = medfilt(eye_camera[:, 1], kernel_size=3)

    cal = np.asarray(config.calibration_factor, dtype=np.float64)
    if cal.ndim == 0:
        fx = fy = float(cal)
    elif cal.shape == (2,):
        fx, fy = float(cal[0]), float(cal[1])
    else:
        raise ValueError("calibration_factor must be scalar or length-2 sequence")

    eye_camera[:, 0] /= fx
    eye_camera[:, 1] /= fy
    eye_camera[:, 1] *= -1

    return eye_camera


def detect_saccades(
    eye_pos_cal: np.ndarray,
    eye_frames: np.ndarray,
    saccade_config: SaccadeConfig,
    config: SessionConfig,
    data: SessionData | None = None,
) -> Dict[str, np.ndarray]:
    """Detect saccades from eye tracking data."""
    torsion_angle = None
    vd_axis_lx = vd_axis_ly = vd_axis_rx = vd_axis_ry = None
    if data is not None:
        if data.torsion is not None:
            torsion_angle = data.torsion[:, 2]
        if data.vdaxis is not None:
            vd_axis_lx, vd_axis_ly = data.vdaxis[:, 2], data.vdaxis[:, 3]
            vd_axis_rx, vd_axis_ry = data.vdaxis[:, 4], data.vdaxis[:, 5]

    dx = np.ediff1d(eye_pos_cal[:, 0], to_begin=0)
    dy = np.ediff1d(eye_pos_cal[:, 1], to_begin=0)
    xy_speed = np.sqrt(dx ** 2 + dy ** 2)

    xy_mask = xy_speed >= saccade_config.saccade_threshold

    if torsion_angle is not None:
        torsion_angle = interpolate_nans(np.asarray(torsion_angle, dtype=np.float64))
        dtheta = np.ediff1d(torsion_angle, to_begin=0)
        torsion_speed = np.abs(dtheta)
        thresh = (
            saccade_config.saccade_threshold_torsion
            if saccade_config.saccade_threshold_torsion is not None
            else np.inf
        )
        torsion_mask = torsion_speed >= thresh
    else:
        torsion_speed = np.zeros_like(xy_speed)
        dtheta = torsion_speed
        torsion_mask = np.zeros_like(xy_mask, dtype=bool)

    saccade_indices_xy = np.where(xy_mask)[0]
    saccade_frames_xy = eye_frames[saccade_indices_xy]

    saccade_indices_theta = np.where(torsion_mask)[0]
    saccade_frames_theta = eye_frames[saccade_indices_theta]

    if torsion_angle is not None:
        eye_pos = np.column_stack([eye_pos_cal, torsion_angle])
        eye_vel = np.column_stack([dx, dy, dtheta])
    else:
        eye_pos = eye_pos_cal
        eye_vel = np.column_stack([dx, dy])

    if (
        saccade_config.blink_detection
        and vd_axis_lx is not None
        and vd_axis_ly is not None
        and vd_axis_rx is not None
        and vd_axis_ry is not None
    ):
        vd_axis_left = np.vstack([vd_axis_lx, vd_axis_ly]).T
        vd_axis_right = np.vstack([vd_axis_rx, vd_axis_ry]).T
        vd_axis_d = np.linalg.norm(vd_axis_right - vd_axis_left, axis=1)
        vd_axis_vel = np.gradient(vd_axis_d)
        blink_indices = np.where(np.abs(vd_axis_vel) > saccade_config.blink_threshold)[0]
        mask = ~np.isin(saccade_indices_xy, blink_indices)
        saccade_indices_xy = saccade_indices_xy[mask]
        saccade_frames_xy = eye_frames[saccade_indices_xy]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    frames = np.arange(len(xy_speed))
    ax.plot(frames, xy_speed, linewidth=0.8, label="Speed (°/frame)")
    ax.scatter(saccade_indices_xy, xy_speed[saccade_indices_xy], color="tab:red", s=12, label="Saccade idx")
    ax.axhline(
        saccade_config.saccade_threshold,
        color="tab:orange",
        linestyle="--",
        label=f"Threshold = {saccade_config.saccade_threshold}",
    )
    ax.set_ylabel("Speed (° / frame)")
    ax.set_title("Instantaneous XY speed with detected saccade frames")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2.plot(frames, torsion_speed, linewidth=0.8, label="Torsion Speed (°/frame)")
    ax2.scatter(
        saccade_indices_theta,
        torsion_speed[saccade_indices_theta],
        color="tab:purple",
        s=12,
        label="Torsion idx",
    )
    ax2.axhline(
        saccade_config.saccade_threshold_torsion,
        color="tab:purple",
        linestyle="--",
        label=f"Threshold = {saccade_config.saccade_threshold_torsion}",
    )
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Torsion Speed (° / frame)")
    ax2.set_title("Instantaneous torsion speed with detected torsional saccades")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    side_tag = f"_{config.camera_side}" if config.camera_side else ""
    session_folder = Path(config.folder_path).name if config.folder_path else config.session_name
    prob_fname = f"{session_folder}{side_tag}_saccades.png"
    fig.savefig(config.results_dir / prob_fname, dpi=300, bbox_inches="tight")

    return {
        "eye_pos": eye_pos,
        "eye_vel": eye_vel,
        "saccade_indices_xy": saccade_indices_xy,
        "saccade_frames_xy": saccade_frames_xy,
        "saccade_indices_theta": saccade_indices_theta,
        "saccade_frames_theta": saccade_frames_theta,
    }


def organize_stims(
    go_frame: np.ndarray,
    go_dir_x: Optional[np.ndarray] = None,
    go_dir_y: Optional[np.ndarray] = None,
) -> tuple[Dict[str, np.ndarray], str]:
    """Group stimulus frames by direction and return the stimulus type."""
    has_lr = go_dir_x is not None and np.any(go_dir_x != 0)
    has_ud = go_dir_y is not None and np.any(go_dir_y != 0)

    direction_sets: Dict[str, np.ndarray] = {}
    if has_lr:
        direction_sets["Left"] = go_dir_x < 0
        direction_sets["Right"] = go_dir_x > 0
    if has_ud:
        direction_sets["Down"] = go_dir_y < 0
        direction_sets["Up"] = go_dir_y > 0
    if not direction_sets:
        direction_sets["All"] = np.full(len(go_frame), True)

    stim_frames = {lab: go_frame[mask] for lab, mask in direction_sets.items()}

    if has_lr and has_ud:
        stim_type = "Interleaved"
    elif has_lr:
        stim_type = "LR"
    elif has_ud:
        stim_type = "UD"
    else:
        stim_type = "None"

    return stim_frames, stim_type


def sort_plot_saccades(
    config: SessionConfig,
    saccade_config: SaccadeConfig,
    saccades: Dict[str, np.ndarray],
    stim_type: str = "None",
) -> None:
    """Sort saccades by stimulus and generate summary plots."""
    saccade_window_frames = saccade_config.saccade_win * config.ttl_freq
    session_path = config.folder_path
    eye_name = config.eye_name

    eye_pos = saccades["eye_pos"].copy()
    mask = np.isfinite(eye_pos[:, 0]) & np.isfinite(eye_pos[:, 1])
    if not mask.any():
        warnings.warn("No finite eye positions; skipping plot")
        return
    dropped = np.count_nonzero(~mask)
    if dropped:
        warnings.warn(f"Dropped {dropped} samples with non-finite eye positions")
    x_mean = np.nanmean(eye_pos[mask, 0])
    y_mean = np.nanmean(eye_pos[mask, 1])
    eye_pos[:, 0] -= x_mean
    eye_pos[:, 1] -= y_mean
    eye_pos_diff = saccades["eye_vel"]
    saccade_indices_xy = saccades["saccade_indices_xy"]
    saccade_frames_xy = saccades["saccade_frames_xy"]
    saccade_indices_theta = saccades["saccade_indices_theta"]
    saccade_frames_theta = saccades["saccade_frames_theta"]
    stim_frames = saccades["stim_frames"]
    session_name = os.path.basename(str(session_path).replace("\\", "/"))

    mask_xy = mask[saccade_indices_xy]
    saccade_indices_xy = saccade_indices_xy[mask_xy]
    saccade_frames_xy = saccade_frames_xy[mask_xy]

    if saccade_indices_theta is not None and len(saccade_indices_theta) > 0:
        saccade_indices_theta = np.array(saccade_indices_theta, dtype=int)
        mask_theta = mask[saccade_indices_theta]
        saccade_indices_theta = saccade_indices_theta[mask_theta]
        saccade_frames_theta = saccade_frames_theta[mask_theta]
        if saccade_indices_theta.size == 0:
            saccade_indices_theta = None
            saccade_frames_theta = None
    else:
        saccade_indices_theta = None
        saccade_frames_theta = None

    if eye_pos_diff.shape[1] == 3:
        dx, dy, dtheta = eye_pos_diff[:, 0], eye_pos_diff[:, 1], eye_pos_diff[:, 2]
        x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
        torsion_present = True
    else:
        dx, dy = eye_pos_diff[:, 0], eye_pos_diff[:, 1]
        x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
        dtheta = None

        torsion_present = False

    x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
    if torsion_present and saccade_indices_theta is not None:
        t_all = eye_pos[saccade_indices_theta, 2]

    pad = 0.10
    max_abs_x = np.nanmax(np.abs(eye_pos[mask, 0]))
    max_abs_y = np.nanmax(np.abs(eye_pos[mask, 1]))
    X_LIM = (-max_abs_x * (1 + pad), max_abs_x * (1 + pad))
    Y_LIM = (-max_abs_y * (1 + pad), max_abs_y * (1 + pad))
    abs_all = np.hypot(dx[saccade_indices_xy], dy[saccade_indices_xy])
    max_abs = np.nanmax(abs_all)


    angle_all = np.arctan2(dy[saccade_indices_xy], dx[saccade_indices_xy])
    n_all = len(saccade_indices_xy)

    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2])
    ax_quiver = fig.add_subplot(gs[:, 0])
    ax_polar = fig.add_subplot(gs[0, 1], polar=True)
    ax_linear = fig.add_subplot(gs[1, 1])

    ax_quiver.set_xlim(*X_LIM)
    ax_quiver.set_ylim(*Y_LIM)
    ax_quiver.set_xlabel("X (°)")
    ax_quiver.set_ylabel("Y (°)")
    ax_quiver.set_title(
        f"{session_name}\n" +
        f"All translational saccades ({n_all}) — {eye_name}  (stim: {stim_type})\n" +
        f"saccade_thresh = {saccade_config.saccade_threshold}, saccade_win = {saccade_config.saccade_win}s\n" +
        f"blink_thresh = {saccade_config.blink_threshold}, blink_detection = {saccade_config.blink_detection}s\n"
    )

    cols = np.array([vector_to_rgb(a) for a in angle_all])
    ax_quiver.quiver(
        x_all,
        y_all,
        dx[saccade_indices_xy],
        dy[saccade_indices_xy],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=cols,
        alpha=0.5,
    )

    plot_angle_distribution(angle_all, ax_polar)
    plot_linear_histogram(angle_all, ax_linear)
    plt.tight_layout()

    all_fname = f"{session_name}_{eye_name}_ALL_{stim_type}.png"
    fig.savefig(config.results_dir / all_fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    plot_window = np.arange(0, saccade_window_frames, 1)

    for label, frames in stim_frames.items():
        if label == "All":
            continue
        idx_buf = []
        sorted_pairs_xy = sorted(zip(saccade_frames_xy, saccade_indices_xy))
        for f in frames:
            lower_bound = max(f + plot_window[0], 0)
            upper_bound = min(f + plot_window[-1], saccade_frames_xy.max())
            for sf, idx in sorted_pairs_xy:
                if sf < lower_bound:
                    continue
                elif sf <= upper_bound:
                    idx_buf.append(idx)
                    break
                else:
                    break
        idx_use = np.array(idx_buf, dtype=int)
        if idx_use.size == 0:
            continue
        ang = np.arctan2(dy[idx_use], dx[idx_use])
        n_cond = len(idx_use)

        fig = plt.figure(figsize=(9, 5))
        gs = gridspec.GridSpec(3, 2, width_ratios=[3, 2])
        ax_q = fig.add_subplot(gs[:, 0])
        ax_p = fig.add_subplot(gs[0, 1], polar=True)
        ax_l = fig.add_subplot(gs[1, 1])
        ax_t = fig.add_subplot(gs[2, 1]) if torsion_present else None

        ax_q.set_xlim(*X_LIM)
        ax_q.set_ylim(*Y_LIM)
        ax_q.set_xlabel("X (°)")
        ax_q.set_ylabel("Y (°)")
        ax_q.set_title(f"{session_name}\n{eye_name} — {label} (n={n_cond})")

        cols = np.array([vector_to_rgb(a) for a in ang])
        ax_q.quiver(
            eye_pos[idx_use, 0],
            eye_pos[idx_use, 1],
            dx[idx_use],
            dy[idx_use],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=cols,
            alpha=0.5,
        )

        plot_angle_distribution(ang, ax_p)
        plot_linear_histogram(ang, ax_l)

        if torsion_present and saccade_indices_theta is not None:
            idx_buf_torsion: list[int] = []
            sorted_pairs_theta = sorted(
                zip(saccade_frames_theta, saccade_indices_theta)
            )
            for f in frames:
                lower_bound = max(f + plot_window[0], 0)
                upper_bound = min(f + plot_window[-1], saccade_frames_theta.max())
                for sf, idx in sorted_pairs_theta:
                    if sf < lower_bound:
                        continue
                    elif sf <= upper_bound:
                        idx_buf_torsion.append(idx)
                        break
                    else:
                        break
            idx_use_t = np.array(idx_buf_torsion, dtype=int)
            for i in idx_use_t:
                x, y = eye_pos[i, 0], eye_pos[i, 1]
                arrow = FancyArrowPatch(
                    (x, y),
                    (x, y),
                    connectionstyle=f"arc3,rad={0.3 * np.sign(dtheta[i])}",
                    mutation_scale=10 * abs(dtheta[i]),
                    color="purple",
                    linewidth=1.5,
                )
                ax_q.add_patch(arrow)
            ax_t.hist(
                dtheta[idx_use_t],
                bins=20,
                color="purple",
                alpha=0.5,
                edgecolor="k",
            )
            ax_t.set_xlabel("Δθ (deg/frame)")
            ax_t.set_ylabel("Count")
            ax_t.set_xlim(-15, 15)

        fig.tight_layout()
        cond_fname = f"{session_name}_{eye_name}_{label}_{stim_type}.png"
        fig.savefig(config.results_dir / cond_fname, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


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
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    plt.Figure,
    plt.Axes,
]:
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


__all__ = [
    "SaccadeConfig",
    "calibrate_eye_position",
    "detect_saccades",
    "organize_stims",
    "sort_plot_saccades",
    "plot_eye_fixations_between_cue_and_go_by_trial",
    "quantify_fixation_stability_vs_random",
]
