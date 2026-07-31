from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.signal import medfilt
from scipy.stats import circmean, circstd
from scipy.ndimage import binary_dilation, binary_fill_holes, label
from scipy.stats import gaussian_kde,vonmises
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


def _normalise_animal_tag(animal_name: str | None) -> str:
    """Return a filesystem-friendly tag for ``animal_name``."""

    text = (animal_name or "").strip()
    if not text:
        return "unknown"
    safe = re.sub(r"[^0-9A-Za-z]+", "_", text)
    safe = safe.strip("_")
    return safe or "unknown"


def _label_with_animal(label: str | None, animal_name: str | None) -> str:
    """Prefix ``label`` with the normalised animal tag."""

    tag = _normalise_animal_tag(animal_name)
    base = (label or "").strip()
    if base.startswith(f"{tag}_"):
        return base
    return f"{tag}_{base}" if base else tag


def _filename_with_animal(base_filename: str, animal_name: str | None) -> str:
    """Add the normalised animal tag to ``base_filename`` before the suffix."""

    stem, suffix = os.path.splitext(base_filename)
    stem_with_tag = _label_with_animal(stem, animal_name)
    return f"{stem_with_tag}{suffix}"


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
    plot: bool = False,
) -> Tuple[Dict[str, np.ndarray], Optional[plt.Figure], Optional[plt.Axes]]:
    """Detect saccades from eye tracking data.

    Parameters
    ----------
    eye_pos_cal:
        Calibrated eye position samples.
    eye_frames:
        Corresponding frame numbers for ``eye_pos_cal``.
    saccade_config:
        Configuration for saccade detection.
    config:
        Session configuration object.
    data:
        Optional :class:`~eyehead.io.SessionData` with torsion and VD axis
        information.
    plot:
        If ``True`` return the generated Matplotlib figure and axes in addition
        to the detected saccades.

    Returns
    -------
    saccades : dict
        Dictionary with detected saccade metrics.
    fig : Figure or None
        Generated figure if ``plot=True``, otherwise ``None``.
    ax : Axes or None
        Primary axes of the diagnostic plot when ``plot=True``; ``None``
        otherwise.
    """
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

    saccades = {
        "eye_pos": eye_pos,
        "eye_vel": eye_vel,
        "saccade_indices_xy": saccade_indices_xy,
        "saccade_frames_xy": saccade_frames_xy,
        "saccade_indices_theta": saccade_indices_theta,
        "saccade_frames_theta": saccade_frames_theta,
    }

    fig = ax = None
    if plot:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        frames = np.arange(len(xy_speed))
        ax.plot(frames, xy_speed, linewidth=0.8, label="Speed (°/frame)")
        ax.scatter(
            saccade_indices_xy,
            xy_speed[saccade_indices_xy],
            color="tab:red",
            s=12,
            label="Saccade idx",
        )
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

        side_tag = f"_{config.camera_side}" if config.camera_side else ""
        session_folder = (
            Path(config.folder_path).name if config.folder_path else config.session_name
        )
        base_fname = f"{session_folder}{side_tag}_saccades.png"
        prob_fname = _filename_with_animal(base_fname, config.animal_name)
        fig.savefig(config.results_dir / prob_fname, dpi=300, bbox_inches="tight")

    return saccades, fig, ax


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


#### Dirty fixes TODO
def plot_left_right_angle(left_angle,right_angle,reward_angle=35,sessionname=None,resultdir=None,experiment_type="prosaccade",animal_name=None):
        fig, (ax_polar_left, ax_polar_right) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(15, 6))
        context_parts = []
        if sessionname:
            context_parts.append(str(sessionname))
        if experiment_type:
            context_parts.append(str(experiment_type))
        context_label = " | ".join(context_parts)
        animal_label = str(animal_name).strip() if animal_name is not None else ""
        if animal_label:
            if context_label:
                suptitle = f"{context_label} – Animal: {animal_label}"
            else:
                suptitle = f"Animal: {animal_label}"
            fig.suptitle(suptitle)
        elif context_label:
            fig.suptitle(context_label)
        counts_left, bins_left = np.histogram(left_angle, bins=18, range=(-np.pi, np.pi))
        counts_right, bins_right = np.histogram(right_angle, bins=18, range=(-np.pi, np.pi))
    # Normalize the histograms
        counts_left = counts_left / np.size(left_angle)
        counts_right = counts_right / np.size(right_angle)
        ax_polar_left.bar( 
            bins_left[:-1],
            counts_left,
            width=np.diff(bins_left),
            align="edge",
            #bottom=0.0,
            color='green',
            alpha=0.5,
            label='Left Trials'
        )
        ax_polar_right.bar(
            bins_right[:-1],
            counts_right,
            width=np.diff(bins_right),
            align="edge",
            #bottom=0.0,
            color='pink',
            alpha=0.5,
            label='Right Trials'
        )
## Plot the von mises fit for both left and right angles

        

        ax_polar_left.set_yticklabels([])
        ax_polar_right.set_yticklabels([])
        ax_polar_left.yaxis.grid(False)
        ax_polar_right.yaxis.grid(False)
        ax_polar_left.set_thetagrids([0,90,180,270], labels=['0°','90°','180°','270°'])
        ax_polar_right.set_thetagrids([0,90,180,270], labels=['0°','90°','180°','270°'])

        # Plot the reward zone
        #reward_angle = reward_angle  # This should be extracted from the config
        if experiment_type == "prosaccade":
            reward_zone_left = np.deg2rad(np.arange(180 - reward_angle, 180 + reward_angle, 1))
            reward_zone_right = np.deg2rad(np.arange(-reward_angle, reward_angle, 1))
        elif experiment_type == "antisaccade":
            reward_zone_left = np.deg2rad(np.arange(-reward_angle, reward_angle, 1))
            reward_zone_right = np.deg2rad(np.arange(180 - reward_angle, 180 + reward_angle, 1))
        ax_polar_left.fill_between(
            reward_zone_left,
            0,
            np.max(counts_left),
            color="yellow",
            alpha=0.15,
            label="Reward Zone",
        )
        ax_polar_right.fill_between(
            reward_zone_right,
            0,
            np.max(counts_right),
            color="yellow",
            alpha=0.15,
            label="Reward Zone",
        )

        # Plot the circular mean
        mean_left_angle = circmean(left_angle, high=np.pi, low=-np.pi)
        mean_left_angle_in_deg = np.rad2deg(mean_left_angle)
        if mean_left_angle_in_deg < 0:
            mean_left_angle_in_deg += 360
        mean_right_angle = circmean(right_angle, high=np.pi, low=-np.pi)
        mean_right_angle_in_deg = np.rad2deg(mean_right_angle)
        if mean_right_angle_in_deg < 0:
            mean_right_angle_in_deg += 360
        std_left_angle = circstd(left_angle, high=np.pi, low=-np.pi)
        std_right_angle = circstd(right_angle, high=np.pi, low=-np.pi)
        ax_polar_left.plot(
            [mean_left_angle, mean_left_angle],
            [0, np.max(counts_left)],
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_left_angle_in_deg:.1f}° ± {np.rad2deg(std_left_angle):.1f}°"
        )
        ax_polar_right.plot(
            [mean_right_angle, mean_right_angle],
            [0, np.max(counts_right)],
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_right_angle_in_deg:.1f}° ± {np.rad2deg(std_right_angle):.1f}°"
        )

        ax_polar_left.legend(loc='upper right', fontsize='small')
        ax_polar_right.legend(loc='upper right', fontsize='small')

        ## print the saccade percentage on the plot
        if experiment_type == "prosaccade":
            saccade_percentage_left = np.sum(np.abs(left_angle) >= np.deg2rad(180-reward_angle)) / len(left_angle) * 100
            saccade_percentage_right = np.sum(np.abs(right_angle) <= np.deg2rad(reward_angle)) / len(right_angle) * 100
        elif experiment_type == "antisaccade":
            saccade_percentage_left = np.sum(np.abs(left_angle) <= np.deg2rad(reward_angle)) / len(left_angle) * 100
            saccade_percentage_right = np.sum(np.abs(right_angle) >= np.deg2rad(180-reward_angle)) / len(right_angle) * 100

        ax_polar_left.text(
            0.5,
            0.9 * np.max(counts_left),
            f"Saccades in the rewarded direction: {saccade_percentage_left:.1f}%",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax_polar_left.transAxes,
            fontsize='small',
            color='black'
        )
        ax_polar_right.text(
            0.5,
            0.9 * np.max(counts_right),
            f"Saccades in the rewarded direction: {saccade_percentage_right:.1f}%",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax_polar_right.transAxes,
            fontsize='small',
            color='black'
        )

        # # Smooth KDE for left angles
        # theta_dense = np.linspace(-np.pi, np.pi, 400)
        # kde_left = gaussian_kde(left_angle, bw_method=0.15)  # adjust bw_method for smoothness
        # density_left = kde_left(theta_dense)
        # density_left_scaled = density_left * np.max(counts_left) / np.max(density_left)

        # theta_closed = np.append(theta_dense, theta_dense[0])
        # density_closed = np.append(density_left_scaled, density_left_scaled[0])

        # ax_polar_left.plot(
        #     theta_closed,
        #     density_closed,
        #     color="darkgreen",
        #     linewidth=2,
        #     label="KDE Smooth Curve"
        # )
        # # Smooth KDE for right angles
        # kde_right = gaussian_kde(right_angle, bw_method=0.15)
        # density_right = kde_right(theta_dense)
        # density_right_scaled = density_right * np.max(counts_right) / np.max(density_right)

        # theta_closed = np.append(theta_dense, theta_dense[0])
        # density_closed = np.append(density_right_scaled, density_right_scaled[0])

        # ax_polar_right.plot(
        #     theta_closed,
        #     density_closed,
        #     color='purple',
        #     linewidth=2,
        #     label='KDE Smooth Curve'
        # )

        ## Plot the von mises kde for left and right angles
        kappa = 12
        theta_dense = np.linspace(-np.pi, np.pi, 200)
        kernels_left = np.array([vonmises.pdf(theta_dense, kappa, loc=a) for a in left_angle])
        density_left = kernels_left.sum(axis=0)
        density_left_scaled = density_left * np.max(counts_left) / np.max(density_left)
        # Wrap around for circular plot
        theta_left_closed = np.append(theta_dense, theta_dense[0])
        density_left_closed = np.append(density_left_scaled, density_left_scaled[0])
        ax_polar_left.plot(
            theta_left_closed,
            density_left_closed,
            color="darkgreen",
            linewidth=2,
            label="Von Mises KDE"
        )
        kernels_right = np.array([vonmises.pdf(theta_dense, kappa, loc=a) for a in right_angle])
        density_right = kernels_right.sum(axis=0)
        density_right_scaled = density_right * np.max(counts_right) / np.max(density_right)
        # Wrap around for circular plot
        theta_right_closed = np.append(theta_dense, theta_dense[0])
        density_right_closed = np.append(density_right_scaled, density_right_scaled[0])
        ax_polar_right.plot(
            theta_right_closed,
            density_right_closed,
            color="purple",
            linewidth=2,
            label="Von Mises KDE"
        )


        # bin_centers_left = (bins_left[:-1] + bins_left[1:]) / 2
        # theta_left_closed = np.append(bin_centers_left, bin_centers_left[0])
        # counts_left_closed = np.append(counts_left, counts_left[0])
        # ax_polar_left.plot(
        #     theta_left_closed,
        #     counts_left_closed,
        #     color='darkgreen',
        #     linewidth=2,
        #     label='Histogram Curve'
        # )

        # # For RIGHT angles: Plot line connecting histogram bins (with wrap around)
        # bin_centers_right = (bins_right[:-1] + bins_right[1:]) / 2
        # theta_right_closed = np.append(bin_centers_right, bin_centers_right[0])
        # counts_right_closed = np.append(counts_right, counts_right[0])
        # ax_polar_right.plot(
        #     theta_right_closed,
        #     counts_right_closed,
        #     color='purple',
        #     linewidth=2,
        #     label='Histogram Curve'
        # )
        # ax_polar_left.set_ylim(0, np.max([np.max(counts_left), 0.4]))
        # ax_polar_right.set_ylim(0, np.max([np.max(counts_right), 0.4]))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if resultdir is None:
            raise ValueError("resultdir must be provided to save the plot")
        resultdir_path = Path(resultdir)

        session_label = str(sessionname).strip() if sessionname else ""
        experiment_label = str(experiment_type).strip() if experiment_type else ""
        session_label_clean = re.sub(r"\s+", "_", session_label).strip("_")
        experiment_label_clean = re.sub(r"\s+", "_", experiment_label).strip("_")

        base_parts = []
        if session_label_clean:
            base_parts.append(session_label_clean)
            session_label_lower = session_label_clean.lower()
        else:
            session_label_lower = ""

        if experiment_label_clean and experiment_label_clean.lower() not in session_label_lower:
            base_parts.append(experiment_label_clean)

        base_parts.append("left_right")
        base_stem = "_".join(part for part in base_parts if part) or "left_right"

        cond_fname_png = _filename_with_animal(f"{base_stem}.png", animal_name)
        cond_fname_svg = _filename_with_animal(f"{base_stem}.svg", animal_name)
        fig.savefig(resultdir_path / cond_fname_png, dpi=300, bbox_inches="tight")
        fig.savefig(resultdir_path / cond_fname_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def sort_saccades(
    config: SessionConfig,
    saccade_config: SaccadeConfig,
    saccades: Dict[str, np.ndarray],
    first_saccade_indices: Dict[str, np.ndarray],
    stim_type: str = "None",
    first_saccade_indices_theta: Optional[Dict[str, np.ndarray]] = None,
    plot: bool = False,
) -> Dict[str, np.ndarray] | Tuple[Dict[str, np.ndarray], plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Sort saccades by stimulus and optionally plot summaries.

    Parameters
    ----------
    config, saccade_config, saccades, stim_type :
        Same as in the original :func:`sort_plot_saccades`.
    first_saccade_indices : dict
        Mapping of stimulus-direction label (``"Left"``, ``"Right"``, ...) to
        the eye-position indices of each trial's first saccade in the online
        reward window (see
        :func:`prosaccade_session.first_saccade_indices_by_direction`). When
        provided, the per-condition translational figures plot exactly these
        first saccades instead of searching a fixed ``saccade_win`` window, so
        they match the latency/congruency analysis. The session-wide "All"
        figure is unaffected and still shows every detected saccade.
    first_saccade_indices_theta : dict, optional
        Same as ``first_saccade_indices`` but for the torsional stream
        (indices into ``saccade_indices_theta``); used for the per-condition
        torsion overlay/histogram so torsion is drawn from the same reward
        window. Falls back to the fixed ``saccade_win`` search when omitted.
    plot : bool, optional
        When ``True`` the function generates the same diagnostic plots as
        before and returns them alongside the sorted saccade indices.

    Returns
    -------
    sorted_data : dict
        Dictionary mapping stimulus labels to the indices of saccades
        occurring within the configured window.
    fig, axes : Figure and tuple of Axes
        Only returned when ``plot=True``.  ``axes`` contains the main quiver,
        polar and linear histogram axes of the summary plot.
    """
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
    session_name_with_animal = _label_with_animal(
        config.session_name or session_name, config.animal_name
    )

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

    sorted_data: Dict[str, np.ndarray] = {"All": saccade_indices_xy}

    if plot:
        fig_all = plt.figure(figsize=(11, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2])
        ax_quiver = fig_all.add_subplot(gs[:, 0])
        ax_polar = fig_all.add_subplot(gs[0, 1], polar=True)
        ax_linear = fig_all.add_subplot(gs[1, 1])

        ax_quiver.set_xlim(*X_LIM)
        ax_quiver.set_ylim(*Y_LIM)
        ax_quiver.set_xlabel("X (°)")
        ax_quiver.set_ylabel("Y (°)")
        ax_quiver.set_title(
            f"{session_name}\n"
            + f"ALL saccades in session ({n_all}) — not trial-aligned — {eye_name}  (stim: {stim_type})\n"
            + f"saccade_thresh = {saccade_config.saccade_threshold}, "
            + f"blink_thresh = {saccade_config.blink_threshold}, "
            f"blink_detection = {saccade_config.blink_detection}s\n"
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

        base_all_fname = f"{session_name}_{eye_name}_ALL_{stim_type}.png"
        all_fname = _filename_with_animal(base_all_fname, config.animal_name)
        fig_all.savefig(config.results_dir / all_fname, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        fig_all = None
        ax_quiver = ax_polar = ax_linear = None

    for label, frames in stim_frames.items():
        if label == "All":
            continue

        idx_use = np.asarray(first_saccade_indices.get(label, []), dtype=int)
        if idx_use.size:
            idx_use = idx_use[mask[idx_use]]
        if idx_use.size == 0:
            continue
        sorted_data[label] = idx_use
        ang = np.arctan2(dy[idx_use], dx[idx_use])
        n_cond = len(idx_use)

        if plot:
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
            _win_note = "first saccade per trial, target onset → end-of-trial"
            ax_q.set_title(
                f"{session_name}\n{eye_name} — {label} (n={n_cond})\n{_win_note}"
            )

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

            if torsion_present and saccade_indices_theta is not None and first_saccade_indices_theta is not None:
                idx_use_t = np.asarray(first_saccade_indices_theta.get(label, []), dtype=int)
                
                if idx_use_t.size:
                    idx_use_t = idx_use_t[mask[idx_use_t]]

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
            base_cond_fname = f"{config.session_name}_{eye_name}_{label}_{stim_type}.png"
            cond_fname = _filename_with_animal(base_cond_fname, config.animal_name)
            fig.savefig(config.results_dir / cond_fname, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)

    if plot:
        has_left_right = "Left" in sorted_data and "Right" in sorted_data
        left_angle = (
            np.arctan2(dy[sorted_data["Left"]], dx[sorted_data["Left"]])
            if "Left" in sorted_data else np.array([])
        )
        right_angle = (
            np.arctan2(dy[sorted_data["Right"]], dx[sorted_data["Right"]])
            if "Right" in sorted_data else np.array([])
        )
        if has_left_right:
            reward_contingency = getattr(config, "reward_contingency", None) or {}
            reward_angle = reward_contingency.get("reward_angle", 35)
            plot_left_right_angle(
                left_angle,
                right_angle,
                reward_angle=reward_angle,
                sessionname=session_name_with_animal,
                resultdir=config.results_dir,
                experiment_type=getattr(config, "experiment_type", None) or "prosaccade",
            )

        return sorted_data, left_angle, right_angle, fig_all, (ax_quiver, ax_polar, ax_linear)
    return sorted_data






def plot_fixation_intervals_by_trial(
    pairs_dt: np.ndarray,
    valid_trials: np.ndarray,
    max_interval_fixations: float,
    results_dir: Optional[Path] = None,
    animal_id: Optional[str] = None,
    eye_name: str = "Eye",
    animal_name: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualise cue→go intervals for each paired trial.

    Parameters
    ----------
    pairs_dt : array-like
        Cue→go intervals in seconds ordered by cue time.
    valid_trials : array-like of bool
        Mask indicating which trials satisfy ``max_interval_fixations``.
    max_interval_fixations : float
        Maximum allowed interval used to determine ``valid_trials``.
    results_dir : Path, optional
        If provided, the generated figure is saved in this directory.
    animal_id, eye_name, animal_name : str, optional
        Metadata used to build the saved filename when ``results_dir`` is
        provided.

    Returns
    -------
    fig, ax : :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
        Handles to the created interval plot.
    """

    intervals = np.asarray(pairs_dt, dtype=float).ravel()
    valid = np.asarray(valid_trials, dtype=bool).ravel()
    if valid.size != intervals.size:
        valid = np.zeros(intervals.size, dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Cue→go interval (s): \nTime until fixation and going to next trial")

    total_trials = int(intervals.size)
    valid_count = int(valid.sum()) if total_trials > 0 else 0

    if total_trials == 0:
        ax.set_title(
            "Cue→Go intervals "
            f"(max={max_interval_fixations:.2f}s)\nNo paired trials available"
        )
        fig.tight_layout()
        return fig, ax

    trial_indices = np.arange(total_trials)
    invalid = ~valid

    if np.any(valid):
        ax.scatter(
            trial_indices[valid],
            intervals[valid],
            color="tab:green",
            marker="o",
            s=40,
            label="Valid",
        )
    if np.any(invalid):
        ax.scatter(
            trial_indices[invalid],
            intervals[invalid],
            color="tab:red",
            marker="x",
            s=50,
            label="Invalid",
        )

    ax.axhline(
        max_interval_fixations,
        color="0.3",
        linestyle="--",
        linewidth=1.2,
        label=f"Max interval ({max_interval_fixations:.2f}s)",
    )

    ax.set_xlim(-0.5, total_trials - 0.5 if total_trials > 0 else 0.5)

    if total_trials > 0:
        valid_fraction = (valid_count / total_trials) * 100.0
        summary = f"Valid trials: {valid_count}/{total_trials} ({valid_fraction:.0f}%)"
    else:
        summary = "Valid trials: 0/0"

    ax.set_title(
        "Cue→Go intervals "
        f"(max={max_interval_fixations:.2f}s)\n{summary}"
    )

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    fig.tight_layout()

    if results_dir is not None and total_trials > 0:
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)

        eye_part = (eye_name or "Eye").replace(" ", "")
        id_part = str(animal_id).strip() if animal_id is not None else ""
        stem_parts = [part for part in (id_part, eye_part, "cue_go_intervals") if part]
        stem = "_".join(stem_parts) if stem_parts else "cue_go_intervals"
        base_fname = f"{stem}.png"
        fname = _filename_with_animal(base_fname, animal_name or animal_id)
        fig.savefig(results_dir / fname, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_eye_fixations_between_cue_and_go_by_trial(
    eye_frame: np.ndarray,
    eye_pos: np.ndarray,
    eye_timestamp: np.ndarray,
    cue_frame: np.ndarray,
    cue_time: np.ndarray,
    go_frame: np.ndarray,
    go_time: np.ndarray,
    max_interval_fixations: float = 1.0,
    color_all: str = "0.85",
    s_all: int = 2,
    alpha_all: float = 0.25,
    s_subset: int = 5,
    alpha_subset: float = 0.9,
    cmap_name: str = "tab20",
    results_dir: Optional[Path] = None,
    animal_id: Optional[str] = None,
    eye_name: str = "Eye",
    animal_name: Optional[str] = None,
    *,
    plot: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[plt.Figure],
    Optional[plt.Axes],
    Optional[plt.Figure],
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
    max_interval_fixations : float, default 1.0
        Maximum allowed cue→go interval for a trial to be considered valid.
    color_all, s_all, alpha_all :
        Matplotlib styling for all eye samples.
    s_subset, alpha_subset :
        Styling for samples within valid trials.
    cmap_name : str, default "tab20"
        Colormap used to colour individual trials.
    results_dir : Path, optional
        If given, save the generated figure here.
    animal_id : str, optional
        Included in the saved filename when ``results_dir`` is provided.
    eye_name : str, default "Eye"
        Label used in the saved filename.
    animal_name : str, optional
        Animal identifier whose normalised form is added to saved filenames.
    plot : bool, default ``False``
        When ``True``, generate a scatter plot and return figure and axes
        handles.

    Returns
    -------
    pairs_cf, pairs_gf : ndarray of int
        Paired cue and go frame indices.
    pairs_ct, pairs_gt : ndarray of float
        Paired cue and go timestamps.
    pairs_dt : ndarray of float
        Time difference between paired events (go - cue).
    valid_trials : ndarray of bool
        Mask indicating which pairs fall within ``max_interval_fixations``.
    fig, ax : Figure and Axes, optional
        Handles to the generated scatter plot when ``plot`` is ``True``;
        otherwise ``None``.
    interval_fig : Figure, optional
        Figure showing cue→go intervals for each trial when ``plot`` is
        ``True``; otherwise ``None``.
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

    valid_trials = (pairs_dt >= 0) & (pairs_dt < max_interval_fixations)
    total_trials = valid_trials.size
    valid_count = int(valid_trials.sum())
    valid_fraction = valid_count / total_trials if total_trials > 0 else np.nan

    interval_fig = None

    if plot:
        cmap = cm.get_cmap(cmap_name)
        base_colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        color_cycle = cycle(base_colors)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            eye_x,
            eye_y,
            s=s_all,
            c=color_all,
            alpha=alpha_all,
            label="All eye centers",
        )

        trial_num = 0
        for ct, gt, ok, dt in zip(pairs_ct, pairs_gt, valid_trials, pairs_dt):
            if not ok:
                continue
            a = np.searchsorted(eye_ts, min(ct, gt), side="left")
            b = np.searchsorted(eye_ts, max(ct, gt), side="right")
            if b <= a:
                continue
            col = next(color_cycle)
            ax.scatter(
                eye_x[a:b],
                eye_y[a:b],
                s=s_subset,
                c=[col],
                alpha=alpha_subset,
                label=f"Trial {trial_num} (Δt={dt:.2f}s)",
            )
            trial_num += 1

        ax.set_aspect("equal")
        ax.set_xlabel("Eye center X (deg)")
        ax.set_ylabel("Eye center Y (deg)")
        if total_trials > 0:
            ratio_text = f"{valid_count}/{total_trials} ({valid_fraction * 100:.0f}%)"
        else:
            ratio_text = "0/0 (n/a)"
        ax.set_title(
            "Eye positions between cue and go "
            f"(<{max_interval_fixations:.2f}s)\nValid trials: {ratio_text}"
        )

        interval_fig, _ = plot_fixation_intervals_by_trial(
            pairs_dt=pairs_dt,
            valid_trials=valid_trials,
            max_interval_fixations=max_interval_fixations,
            results_dir=results_dir,
            animal_id=animal_id,
            eye_name=eye_name,
            animal_name=animal_name,
        )

        if results_dir is not None:
            results_dir = Path(results_dir)
            results_dir.mkdir(exist_ok=True, parents=True)

            id_part = str(animal_id).strip() if animal_id is not None else ""
            eye_part = (eye_name or "Eye").replace(" ", "")
            stem_parts = [part for part in (id_part, eye_part, "cue_go_timepaired") if part]
            stem = "_".join(stem_parts) if stem_parts else "cue_go_timepaired"
            base_fname = f"{stem}.png"

            fname = _filename_with_animal(base_fname, animal_name or animal_id)
            fig.savefig(results_dir / fname, dpi=300, bbox_inches="tight")
    else:
        fig = ax = None

    return (
        pairs_cf,
        pairs_gf,
        pairs_ct,
        pairs_gt,
        pairs_dt,
        valid_trials,
        fig,
        ax,
        interval_fig,
    )


def quantify_fixation_stability_vs_random(
    eye_timestamp: np.ndarray,
    eye_pos: np.ndarray,
    pairs_ct: np.ndarray,
    pairs_gt: np.ndarray,
    valid_trials: np.ndarray,
    *,
    plot: bool = False,
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
    plot : bool, default ``False``
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
        
        ############################################### calculate step, speed, and drift
        step_disp = np.hypot(dx[m], dy[m]) # mean euclidean distance between consecutive points, for all frames in window
        speed = step_disp / dt[m] #divided by window size to get mean speed
        drift = np.hypot(x[b - 1] - x[a], y[b - 1] - y[a]) #net displacement between start and end of window
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

def extract_eye_position_from_dlc(
    dlc_data: pd.DataFrame,
    cal: float,
    fps: float,
    pupil_likelihood_thresh: float = 0.95,
    eye_likelihood_thresh: float = 0.80,
    medfilt_kernel: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract eye position time series from a DLC CSV dataframe.

    Parameters
    ----------
    dlc_data                 : pd.DataFrame
        Loaded DLC CSV (skiprows=2).
    cal                      : float
        Pixels per degree calibration factor.
    fps                      : float
        Camera frame rate in Hz.
    pupil_likelihood_thresh  : float, optional
        Minimum likelihood for pupil points (default 0.95).
    eye_likelihood_thresh    : float, optional
        Minimum likelihood for eye corner points (default 0.80).
    medfilt_kernel           : int, optional
        Median filter kernel size in samples (default 5).

    Returns
    -------
    eye_position     : (N, 3) ndarray
        Columns are ``[x_deg, y_deg, torsion_deg]``.
    time             : (N,) ndarray
        Time vector in seconds.
    eye_axes_lengths : (N, 2) ndarray
        Columns are ``[NT_distance_px, VD_distance_px]``.
    pupil_area       : (N,) ndarray
        Pupil area in pixels squared (interpolated across blinks).
    """
    import cv2  # local import: OpenCV is only needed for DLC pupil fitting

    # --- Extract keypoints ---
    pupil_points = dlc_data[
        ['x.4',  'y.4',  'x.5',  'y.5',  'x.6',  'y.6',  'x.7',  'y.7',
         'x.8',  'y.8',  'x.9',  'y.9',  'x.10', 'y.10', 'x.11', 'y.11']
    ].values

    pupil_points_likelihood = dlc_data[
        ['likelihood.4',  'likelihood.5',  'likelihood.6',  'likelihood.7',
         'likelihood.8',  'likelihood.9',  'likelihood.10', 'likelihood.11']
    ].values

    eye_points               = dlc_data[['x',    'y',    'x.2',  'y.2' ]].values
    eye_points_likelihood    = dlc_data[['likelihood', 'likelihood.2']].values
    eye_points_VD            = dlc_data[['x.1',  'y.1',  'x.3',  'y.3' ]].values
    eye_points_VD_likelihood = dlc_data[['likelihood.1', 'likelihood.3']].values

    n_frames = pupil_points.shape[0]

    # --- Pupil centres via ellipse fit ---
    pupil_centers = []
    pupil_area    = []
    for i in range(n_frames):
        likelihoods = pupil_points_likelihood[i]
        pts         = pupil_points[i].reshape(-1, 2).astype(np.float32)
        good_mask   = likelihoods >= pupil_likelihood_thresh
        good_pts    = pts[good_mask]

        if len(good_pts) < 6:
            pupil_centers.append([np.nan, np.nan, np.nan])
            pupil_area.append(np.nan)
            continue
        try:
            ellipse = cv2.fitEllipseDirect(good_pts)
            pupil_centers.append([ellipse[0][0], ellipse[0][1], ellipse[2]])
            pupil_area.append(np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2))
        except cv2.error:
            pupil_centers.append([np.nan, np.nan, np.nan])
            pupil_area.append(np.nan)

    pupil_centers = np.array(pupil_centers, dtype=float)
    pupil_area    = np.array(pupil_area,    dtype=float)

    # --- Eye centres ---
    eye_centers     = []
    eye_NT_distance = []
    for i in range(n_frames):
        pts         = eye_points[i].reshape(-1, 2).astype(np.float32)
        likelihoods = eye_points_likelihood[i]
        if np.any(pts == 0) or np.any(likelihoods < eye_likelihood_thresh):
            eye_centers.append([np.nan, np.nan])
            eye_NT_distance.append(np.nan)
            continue
        eye_centers.append(pts.mean(axis=0))
        eye_NT_distance.append(np.linalg.norm(pts[0] - pts[1]))

    eye_centers     = np.array(eye_centers,     dtype=float)
    eye_NT_distance = (pd.Series(eye_NT_distance, dtype=float)
                         .interpolate().bfill().ffill().values)

    # --- VD distance for blink detection ---
    eye_VD_distance = []
    for i in range(n_frames):
        pts         = eye_points_VD[i].reshape(-1, 2).astype(np.float32)
        likelihoods = eye_points_VD_likelihood[i]
        if np.any(pts == 0) or np.any(likelihoods < eye_likelihood_thresh):
            eye_VD_distance.append(np.nan)
            continue
        eye_VD_distance.append(np.linalg.norm(pts[0] - pts[1]))

    eye_VD_distance = (pd.Series(eye_VD_distance, dtype=float)
                         .interpolate().bfill().ffill().values)

    eye_axes_lengths = np.vstack((eye_NT_distance, eye_VD_distance)).T

    # --- Blink detection and interpolation ---
    eye_VD_velocity    = np.abs(np.gradient(eye_VD_distance))
    blink_thresh       = np.percentile(eye_VD_velocity, 99.99)
    blink_mask         = eye_VD_velocity > blink_thresh
    blink_window       = int(0.2 * fps)                     # 200 ms window
    blink_mask_dilated = binary_dilation(blink_mask, iterations=blink_window)

    # Remove components that are too long to be genuine blinks
    labeled, n_components = label(blink_mask_dilated)
    max_blink_frames      = int(0.4 * fps)                  # 400 ms maximum
    for i in range(1, n_components + 1):
        if np.sum(labeled == i) > max_blink_frames:
            blink_mask_dilated[labeled == i] = False

    for idx in np.where(blink_mask_dilated)[0]:
        s = max(0,            idx - 10)
        e = min(n_frames - 1, idx + 10)
        pupil_centers[s:e+1] = np.nan
        eye_centers[s:e+1]   = np.nan

    for j in range(2):
        pupil_centers[:, j] = (pd.Series(pupil_centers[:, j])
                                 .interpolate().bfill().ffill())
        eye_centers[:, j]   = (pd.Series(eye_centers[:, j])
                                 .interpolate().bfill().ffill())
    pupil_centers[:, 2] = (pd.Series(pupil_centers[:, 2])
                              .interpolate().bfill().ffill())
    pupil_area = (pd.Series(pupil_area)
                    .interpolate().bfill().ffill().values)

    # --- Eye position in degrees ---
    eye_positions           = pupil_centers[:, :2] - eye_centers
    eye_position_in_degrees = eye_positions / cal
    eye_position_in_degrees = np.hstack(
        (eye_position_in_degrees, pupil_centers[:, 2:3])
    )

    # --- Median filter ---
    for j in range(3):
        eye_position_in_degrees[:, j] = medfilt(
            eye_position_in_degrees[:, j], kernel_size=medfilt_kernel
        )

    # --- Time vector ---
    time = np.arange(n_frames) / fps

    return (eye_position_in_degrees,
            time,
            eye_axes_lengths,
            pupil_area)

def identify_saccades_1d(
    position: np.ndarray,
    time: np.ndarray,
    min_duration_ms: float = 20,
    max_duration_ms: float = 50,
    mad_multiplier: float = 6.0,
    mad_max_iter: int = 20,
    mad_tolerance: float = 1e-3,
    min_amplitude: float = 1.0,
    min_peak_ratio: float = 2.0,
    dominant_direction_ratio: float = 0.9,
    *,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Optional[plt.Figure]]:
    """
    1D saccade detection using MAD-based velocity thresholding.

    Detects saccades as continuous periods where speed exceeds a
    data-driven threshold estimated iteratively from the MAD of the
    baseline speed distribution.

    Parameters
    ----------
    position        : (N,) position in deg
    time            : (N,) time in seconds
    min_duration_ms : float, minimum saccade duration in ms (default 20)
    max_duration_ms : float, maximum saccade duration in ms (default 50)
    mad_multiplier  : float, threshold = median + k * MAD (default 6)
    mad_max_iter    : int,   maximum MAD iterations (default 20)
    mad_tolerance   : float, convergence tolerance (default 1e-3)
    min_amplitude   : float, minimum saccade amplitude in deg (default 1.0)
    min_peak_ratio  : float, minimum ratio of peak speed to threshold (default 2.0)
    dominant_direction_ratio : float, minimum ratio of dominant direction
                               samples (default 0.9)
    plot            : bool, if True a figure with two vertically stacked
                      subplots (position and speed) on a shared time axis is
                      returned in the output tuple; saccade epochs are shaded
                      in both panels and annotated with per-saccade metrics
                      (default False)

    Returns
    -------
    saccades            : (K,)   index of peak speed per saccade
    saccade_windows     : (K, 2) [start, end] indices of threshold crossing
    saccade_durations   : (K,)   duration in samples
    saccade_amplitudes  : (K,)   amplitude (abs position change start→end)
    saccade_peak_speeds : (K,)   peak speed on raw velocity
    velocity_threshold  : float  converged MAD threshold
    figure              : matplotlib.figure.Figure or None
                          Present only when ``plot=True``.
    """

    def _interp_threshold_crossing(speed, pos, time, idx, threshold, side):
        """
        Linearly interpolate position and time at the exact threshold crossing.

        Parameters
        ----------
        side : 'rising'  → crossing between idx-1 and idx  (bout start)
            'falling' → crossing between idx and idx+1   (bout end)
        """
        if side == 'rising':
            i0, i1 = idx - 1, idx
        else:
            i0, i1 = idx, idx + 1

        # Clamp to valid range
        i0 = max(i0, 0)
        i1 = min(i1, len(speed) - 1)

        s0, s1 = speed[i0], speed[i1]
        denom = s1 - s0
        if abs(denom) < 1e-12:          # flat → no interpolation possible
            return pos[idx], time[idx]

        alpha = (threshold - s0) / denom   # fractional position in [0, 1]
        t_cross = time[i0] + alpha * (time[i1] - time[i0])
        p_cross = pos[i0]  + alpha * (pos[i1]  - pos[i0])
        return p_cross, t_cross






    pos   = np.asarray(position).ravel()
    time  = np.asarray(time).ravel()
    vel   = np.gradient(pos, time)

    fs    = 1.0 / np.median(np.diff(time))
    n     = len(time)
    speed = np.abs(vel)

    min_dur_samples = min_duration_ms / 1000.0 * fs
    max_dur_samples = max_duration_ms / 1000.0 * fs

    # ----------------------------------------------------------------
    # MAD-based iterative threshold estimation
    # ----------------------------------------------------------------
    baseline_mask      = np.ones(n, dtype=bool)
    velocity_threshold = np.inf
    med, mad           = 0.0, 0.0

    for iteration in range(mad_max_iter):
        baseline_speed = speed[baseline_mask]

        med = np.median(baseline_speed)
        mad = np.median(np.abs(baseline_speed - med))

        new_threshold = med + mad_multiplier * mad

        if abs(new_threshold - velocity_threshold) < mad_tolerance:
            velocity_threshold = new_threshold
            break

        velocity_threshold = new_threshold
        baseline_mask      = speed <= velocity_threshold

        if baseline_mask.sum() < 0.1 * n:
            break

    # ----------------------------------------------------------------
    # Threshold crossing detection
    # ----------------------------------------------------------------
    above_threshold = speed > velocity_threshold
    padded          = np.diff(np.concatenate([[0], above_threshold.astype(int), [0]]))
    bout_starts     = np.where(padded ==  1)[0]
    bout_ends       = np.where(padded == -1)[0]-1

    saccades            = []
    saccade_windows     = []
    saccade_durations   = []
    saccade_amplitudes  = []
    saccade_peak_speeds = []
    saccade_interp_times = []  

    for s, e in zip(bout_starts, bout_ends):
        p_start, t_start = _interp_threshold_crossing(
            speed, pos, time, s, velocity_threshold, side='rising')
        p_end,   t_end   = _interp_threshold_crossing(
            speed, pos, time, e, velocity_threshold, side='falling')

        amp = abs(p_end - p_start)
        dur_interp_s = t_end - t_start          # seconds
        dur = dur_interp_s * fs                 # fractional samples, for thresholding
        #dur = e - s + 1

        if not (min_dur_samples <= dur <= max_dur_samples):
            continue

        peak_local  = np.argmax(speed[s:e+1])
        peak_idx    = s + peak_local
        peak_speed  = speed[peak_idx]
        #amp         = abs(pos[e] - pos[s])
        vel_in_bout = vel[s:e+1]

        if amp < min_amplitude:
            continue
        if peak_speed < min_peak_ratio * velocity_threshold:
            continue

        pos_samples = np.sum(vel_in_bout > 0)
        neg_samples = np.sum(vel_in_bout < 0)
        total       = len(vel_in_bout)

        if max(pos_samples, neg_samples) / total < dominant_direction_ratio:
            continue

        saccades.append(peak_idx)
        saccade_windows.append((s, e))
        saccade_interp_times.append((t_start, t_end)) 
        saccade_durations.append(dur)
        saccade_amplitudes.append(amp)
        saccade_peak_speeds.append(peak_speed)

    saccades            = np.array(saccades,            dtype=int)
    saccade_windows     = np.array(saccade_windows,     dtype=int)
    saccade_durations   = np.array(saccade_durations,   dtype=float)
    saccade_amplitudes  = np.array(saccade_amplitudes,  dtype=float)
    saccade_peak_speeds = np.array(saccade_peak_speeds, dtype=float)
    saccade_interp_times = np.array(saccade_interp_times, dtype=float)
    # ----------------------------------------------------------------
    # Optional figure: position + speed with shared time axis
    # ----------------------------------------------------------------
    if plot:
        fig, (ax_pos, ax_spd) = plt.subplots(
            2, 1, figsize=(14, 6),
            sharex=True, constrained_layout=True,
        )

        # --- position panel ---
        ax_pos.plot(time, pos, color="steelblue", linewidth=0.8, label="Position")
        ax_pos.set_ylabel("Position (deg)")

        # --- speed panel ---
        ax_spd.plot(time, vel, 'k.', linewidth=0.8, label="Speed",)
        ax_spd.axhline(
            velocity_threshold,
            color="tomato", linewidth=1.2, linestyle="--",
            label=f"Threshold ({velocity_threshold:.1f} deg/s)",
        )
        ax_spd.axhline(
            -velocity_threshold,
            color="tomato", linewidth=1.2, linestyle="--",
            #label=f"Threshold ({-velocity_threshold:.1f} deg/s)",
        )
        ax_spd.set_ylabel("Velocity (deg/s)")
        ax_spd.set_xlabel("Time (s)")

        # --- per-saccade shading ---- 
        dur_ms = saccade_durations / fs * 1000.0   # samples → ms

        for i, (t_s, t_e) in enumerate(saccade_interp_times):  # ← interpolated times
            sacc_lbl = "Saccade" if i == 0 else None

            for ax in (ax_pos, ax_spd):
                ax.axvspan(t_s, t_e, color="gold", alpha=0.35,
                        linewidth=0, label=sacc_lbl)
                sacc_lbl = None

            

        ax_pos.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax_spd.legend(loc="upper right", fontsize=8, framealpha=0.7)
        fig.suptitle(
            f"Saccade detection  —  {len(saccades)} saccade(s) found  |  "
            f"threshold = {velocity_threshold:.1f} deg/s",
            fontsize=10,
        )
    else:
        fig = None

    return (saccades,
            saccade_windows,
            saccade_durations,
            saccade_amplitudes,
            saccade_peak_speeds,
            float(velocity_threshold),
            fig)

__all__ = [
    "SaccadeConfig",
    "calibrate_eye_position",
    "detect_saccades",
    "organize_stims",
    "sort_saccades",
    "plot_fixation_intervals_by_trial",
    "plot_eye_fixations_between_cue_and_go_by_trial",
]
