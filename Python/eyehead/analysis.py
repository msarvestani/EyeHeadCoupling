from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.signal import medfilt

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
    prob_fname = f"{config.session_name}{side_tag}_saccades.png"
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

    eye_pos = saccades["eye_pos"]
    eye_pos_diff = saccades["eye_vel"]
    saccade_indices_xy = saccades["saccade_indices_xy"]
    saccade_frames_xy = saccades["saccade_frames_xy"]
    saccade_indices_theta = saccades["saccade_indices_theta"]
    saccade_frames_theta = saccades["saccade_frames_theta"]
    stim_frames = saccades["stim_frames"]
    session_name = os.path.basename(str(session_path).replace("\\", "/"))

    if saccade_indices_theta is not None and len(saccade_indices_theta) > 0:
        saccade_indices_theta = np.array(saccade_indices_theta, dtype=int)
        t_all = eye_pos[saccade_indices_theta, 2]
    else:
        t_all = None

    if eye_pos_diff.shape[1] == 3:
        dx, dy, _ = eye_pos_diff[:, 0], eye_pos_diff[:, 1], eye_pos_diff[:, 2]
        x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
        t_all = eye_pos[saccade_indices_theta, 2] if saccade_indices_theta is not None else None
        torsion_present = True
    else:
        dx, dy = eye_pos_diff[:, 0], eye_pos_diff[:, 1]
        x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
        t_all = None
        torsion_present = False

    pad = 0.10
    rngX = x_all.max() - x_all.min()
    rngY = y_all.max() - y_all.min()
    X_LIM = (x_all.min() - pad * rngX, x_all.max() + pad * rngX)
    Y_LIM = (y_all.min() - pad * rngY, y_all.max() + pad * rngY)
    abs_all = np.hypot(dx[saccade_indices_xy], dy[saccade_indices_xy])
    max_abs = abs_all.max()

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

    cols = np.array([vector_to_rgb(a, m, max_abs) for a, m in zip(angle_all, abs_all)])
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
        mag = np.hypot(dx[idx_use], dy[idx_use])
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

        cols = np.array([vector_to_rgb(a, m, max_abs) for a, m in zip(ang, mag)])
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

        if torsion_present:
            idx_buf_torsion: list[int] = []
            sorted_pairs_theta = sorted(
                zip(saccade_frames_theta, range(len(saccade_indices_theta)))
            )
            for f in frames:
                lower_bound = max(f + plot_window[0], 0)
                upper_bound = min(f + plot_window[-1], saccade_frames_theta.max())
                for sf, pos in sorted_pairs_theta:
                    if sf < lower_bound:
                        continue
                    elif sf <= upper_bound:
                        idx_buf_torsion.append(pos)
                        break
                    else:
                        break
            idx_use_t = np.array(idx_buf_torsion, dtype=int)
            ax_t.hist(t_all[idx_use_t], bins=18, color="b", alpha=0.5, edgecolor="k")
            ax_t.set_xlabel("Δθ (°)")
            ax_t.set_ylabel("Count")

        fig.tight_layout()
        cond_fname = f"{session_name}_{eye_name}_{label}_{stim_type}.png"
        fig.savefig(config.results_dir / cond_fname, dpi=300, bbox_inches="tight")


__all__ = [
    "SaccadeConfig",
    "calibrate_eye_position",
    "detect_saccades",
    "organize_stims",
    "sort_plot_saccades",
]
