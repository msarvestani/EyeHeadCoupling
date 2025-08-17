
from __future__ import annotations


import sys
import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import ShortTimeFFT, butter, hilbert, sosfiltfilt, medfilt
from scipy.signal.windows import gaussian
import scipy.stats as stats
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA
from io import StringIO
from matplotlib import gridspec
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import re
from datetime import datetime
from matplotlib.patches import FancyArrowPatch
from itertools import cycle
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib
from dataclasses import dataclass
from typing import Optional, Union



@dataclass
class SaccadeConfig:
    """Configuration parameters for :func:`detect_saccades`."""
    saccade_threshold: float
    saccade_threshold_torsion: float
    blink_threshold: float = 10.0
    blink_detection: int = 1
    saccade_win: float = 0.7

@dataclass
class SessionConfig:
    """Configuration parameters for a session."""
    session_name: str
    results_dir: Path
    calibration_factor: float
    camera_side: Optional[str] = None
    eye_name: Optional[str] = None
    ttl_freq: Optional[float] = None
    folder_path: Optional[str] = None
    # Add other session-related parameters as needed

def get_session_date_from_path(path):
    match = re.search(r"\d{4}-\d{2}-\d{2}", path)
    if match:
        return datetime.strptime(match.group(), "%Y-%m-%d")
    else:
        raise ValueError("No valid date (YYYY-MM-DD) found in path")

def determine_camera_side(path, cutoff_date_str="2025-06-30"):
    session_date = get_session_date_from_path(path)
    cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")
    return "L" if session_date >= cutoff_date else "R"



# Prompt the user to select a folder
def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory()  # Open the file selection dialog
    return directory


# Prompt the user to open a file 
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open the file selection dialog
    return file_path

def choose_option(option1, option2, option3, option4):
    result = {}

    def select(choice):
        result['value'] = choice
        root.destroy()

    root = tk.Tk()
    root.title("Choose the type of visual stim")

    tk.Label(root, text="Please choose the type of visual stim:").pack(pady=10)
    tk.Button(root, text=option1, width=12, command=lambda: select(option1)).pack(side='left', padx=10, pady=10)
    tk.Button(root, text=option2, width=12, command=lambda: select(option2)).pack(side='left', padx=10, pady=10)
    tk.Button(root, text=option3, width=12, command=lambda: select(option3)).pack(side='left', padx=10, pady=10)
    tk.Button(root, text=option4, width=12, command=lambda: select(option4)).pack(side='left', padx=10, pady=10)

    # Manual event loop, blocks until window is destroyed
    while not result.get('value'):
        root.update()

    return result['value']

def remove_parentheses_chars(line):
    # Remove only '(' and ')' characters
    return line.replace('(', '').replace(')', '').replace('True', '1').replace('False', '0')


def clean_csv(filename):
    with open(filename, 'r') as f:
        lines = [remove_parentheses_chars(line) for line in f]
    # Join lines and create a file-like object
        cleaned = StringIO(''.join(lines))
        return cleaned


# Butterworth filter to remove high frequency noise
def butter_noncausal(signal, fs, cutoff_freq=1, order=4):
    sos = butter(order, cutoff_freq/(fs/2), btype='low', output='sos')  # 50 Hz cutoff frequency
    return sosfiltfilt(sos, signal)   


def interpolate_nans(arr):
    nans = np.isnan(arr)
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])

    return arr

def rotation_matrix(angle_rad):
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                     [np.sin(angle_rad), np.cos(angle_rad)]])

def vector_to_rgb(angle, absolute): ##Got it from https://stackoverflow.com/questions/19576495/color-matplotlib-quiver-field-according-to-magnitude-and-direction
    global max_abs

    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    # return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
    #                                      absolute / max_abs, 
    #                                      absolute / max_abs))
    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                         1, 
                                         1))

def plot_angle_distribution(angle, ax_polar, num_bins=18):
    """
    Plots a normalized polar histogram of angles.

    Parameters:
        angle (np.ndarray): array of saccade angles in radians
        ax_polar (matplotlib.axes._subplots.PolarAxesSubplot): the polar subplot to draw on
        num_bins (int): number of histogram bins
    """
    angle_2pi = np.where(angle < 0, angle + 2 * np.pi, angle)
    counts, bin_edges = np.histogram(angle_2pi, bins=num_bins, range=(0, 2 * np.pi))
    counts = counts / np.size(angle_2pi)  # Normalize
    width = np.diff(bin_edges)

    bars = ax_polar.bar(bin_edges[:-1], counts, width=width, align='edge', color='b', alpha=0.5, edgecolor='k')
    ax_polar.set_title("Normalized angle distribution")
    ax_polar.set_yticklabels([])

def plot_linear_histogram(angles, ax, num_bins=18):
    ang_deg = np.degrees(angles)
    ang_deg = np.mod(ang_deg, 360)
    counts, bins = np.histogram(ang_deg, bins=num_bins, range=(0, 360))
    counts = counts / ang_deg.size
    ax.bar(bins[:-1], counts, width=np.diff(bins), color="b", alpha=0.5, edgecolor="k")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Normalised count")
    ax.set_title("Linear angle histogram")    

def calibrate_eye_position(marker1_x, marker1_y, marker2_x, marker2_y,
        gaze_x, gaze_y,
        SessionConfig,
        ):
    
    """Detect saccades from eye tracking data.

    Parameters
    ----------   
        marker1_x, marker1_y, marker2_x, marker2_y : array_like
        Coordinates of the eyelid markers.
    gaze_x, gaze_y : array_like
        Gaze position from Bonsai.
    config : :class:`SaccadeDetectionConfig`
        Parameters controlling the detection.
    """

    # 1. eye-centred coordinates → degrees
    eye_origin = np.column_stack(((marker1_x + marker2_x) / 2.0,
                                  (marker1_y + marker2_y) / 2.0))
    eye_camera = np.column_stack((gaze_x - eye_origin[:, 0],
                                  gaze_y - eye_origin[:, 1])).astype(np.float64, copy=False)

    # small denoise
    eye_camera[:, 0] = medfilt(eye_camera[:, 0], kernel_size=3)
    eye_camera[:, 1] = medfilt(eye_camera[:, 1], kernel_size=3)

    # read in 1 or 2 calibration factors
    cal = np.asarray(SessionConfig.calibration_factor, dtype=np.float64)
    if cal.ndim == 0:
        fx = fy = float(cal)
    elif cal.shape == (2,):
        fx, fy = float(cal[0]), float(cal[1])
    else:
        raise ValueError("calibration_factor must be scalar or length-2 sequence")

    eye_camera[:, 0] /= fx
    eye_camera[:, 1] /= fy

    eye_camera_cal = eye_camera
    return eye_camera_cal

def detect_saccades(
    eye_pos_cal,
    eye_frames,
    SaccadeConfig,
    SessionConfig,
    vd_axis_lx=None, vd_axis_ly=None, vd_axis_rx=None, vd_axis_ry=None,
    torsion_angle=None,
):
    """Detect saccades from eye tracking data.

    Parameters
    ----------

    vd_axis_lx, vd_axis_ly, vd_axis_rx, vd_axis_ry : array_like, optional
        Vertical displacement axis of the eyelids, used for blink detection.
    torsion_angle : array_like, optional
        Torsion angle of the eye.
    """

    # 2. instantaneous velocity  →  speed
    dx = np.ediff1d(eye_pos_cal[:, 0], to_begin=0)
    dy = np.ediff1d(eye_pos_cal[:, 1], to_begin=0)
    xy_speed = np.sqrt(dx**2 + dy**2)

    xy_mask = xy_speed >= SaccadeConfig.saccade_threshold

    # 3. torsional velocity
    if torsion_angle is not None:
        torsion_angle = interpolate_nans(np.asarray(torsion_angle, dtype=np.float64))
        dtheta = np.ediff1d(torsion_angle, to_begin=0)
        torsion_speed = np.abs(dtheta)
        thresh = (SaccadeConfig.saccade_threshold_torsion
                  if SaccadeConfig.saccade_threshold_torsion is not None else np.inf)
        torsion_mask = torsion_speed >= thresh
    else:
        torsion_speed = np.zeros_like(xy_speed)
        dtheta = torsion_speed
        torsion_mask = np.zeros_like(xy_mask, dtype=bool)

    # 4. Detect saccade indices
    saccade_indices_xy = np.where(xy_mask)[0]
    saccade_frames_xy = eye_frames[saccade_indices_xy]

    saccade_indices_theta = np.where(torsion_mask)[0]
    saccade_frames_theta = eye_frames[saccade_indices_theta]

    # 5. Package eye positions and velocity into output

    if torsion_angle is not None:
        eye_pos = np.column_stack([eye_pos_cal, torsion_angle])
        eye_vel = np.column_stack([dx, dy, dtheta])
    else:
        eye_pos = eye_pos_cal
        eye_vel = np.column_stack([dx, dy])

    # 6. Optional blink removal
    if (SaccadeConfig.blink_detection and vd_axis_lx is not None and vd_axis_ly is not None
            and vd_axis_rx is not None and vd_axis_ry is not None):
        vd_axis_left = np.vstack([vd_axis_lx, vd_axis_ly]).T
        vd_axis_right = np.vstack([vd_axis_rx, vd_axis_ry]).T
        vd_axis_d = np.linalg.norm(vd_axis_right - vd_axis_left, axis=1)
        vd_axis_vel = np.gradient(vd_axis_d)
        blink_indices = np.where(np.abs(vd_axis_vel) > SaccadeConfig.blink_threshold)[0]
        mask = ~np.isin(saccade_indices_xy, blink_indices)
        saccade_indices_xy = saccade_indices_xy[mask]
        saccade_frames_xy = eye_frames[saccade_indices_xy]



    # Plot saccade and threshold to make sure it's detected
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    frames = np.arange(len(xy_speed))

    # ─── Plot XY (translational) saccades ───
    ax.plot(frames, xy_speed, linewidth=0.8, label='Speed (°/frame)')
    ax.scatter(saccade_indices_xy, xy_speed[saccade_indices_xy],
            color='tab:red', s=12, label='Saccade idx')
    ax.axhline(SaccadeConfig.saccade_threshold, color='tab:orange',
            linestyle='--', label=f'Threshold = {SaccadeConfig.saccade_threshold}')
    ax.set_ylabel('Speed (° / frame)')
    ax.set_title('Instantaneous XY speed with detected saccade frames')
    ax.legend()
    ax.grid(alpha=.3)

    # ─── Plot torsional saccades ───
    ax2.plot(frames, torsion_speed, linewidth=0.8, label='Torsion Speed (°/frame)')
    ax2.scatter(saccade_indices_theta, torsion_speed[saccade_indices_theta],
                color='tab:purple', s=12, label='Torsion idx')
    ax2.axhline(SaccadeConfig.saccade_threshold_torsion, color='tab:purple',
                linestyle='--', label=f'Threshold = {SaccadeConfig.saccade_threshold_torsion}')
    ax2.set_xlabel('Frame number')
    ax2.set_ylabel('Torsion Speed (° / frame)')
    ax2.set_title('Instantaneous torsion speed with detected torsional saccades')
    ax2.legend()
    ax2.grid(alpha=.3)

    plt.tight_layout()
    plt.show()


    # save alongside other figures
    prob_fname = f"{SessionConfig.session_name}_saccades.png"
    fig.savefig(SessionConfig.results_dir / prob_fname, dpi=300, bbox_inches='tight')


    return {
        "eye_pos": eye_pos,
        "eye_vel": eye_vel,
        "saccade_indices_xy": saccade_indices_xy,
        "saccade_frames_xy": saccade_frames_xy,
        "saccade_indices_theta": saccade_indices_theta,
        "saccade_frames_theta": saccade_frames_theta,
    }


def organize_stims(
    go_frame, 
    go_dir_x = None,
    go_dir_y = None):
    
    has_lr = go_dir_x is not None and np.any(go_dir_x != 0)
    has_ud = go_dir_y is not None and np.any(go_dir_y != 0)

    direction_sets = {}

    if has_lr:
        direction_sets["Left"]  = go_dir_x < 0
        direction_sets["Right"] = go_dir_x > 0
    if has_ud:
        direction_sets["Down"] = go_dir_y < 0
        direction_sets["Up"]   = go_dir_y > 0
    if not direction_sets:
        direction_sets["All"] = np.full(len(go_frame), True)

    stim_frames = {lab: go_frame[mask] for lab, mask in direction_sets.items()}

    # Return the inferred stim type too
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
    SessionConfig,
    SaccadeConfig,
    saccades,
    stim_type='None',

):
    saccade_window_frames = SaccadeConfig.saccade_win*SessionConfig.ttl_freq
    session_path = SessionConfig.folder_path
    eye_name = SessionConfig.eye_name

    eye_pos = saccades["eye_pos"]
    eye_pos_diff = saccades["eye_vel"]
    saccade_indices_xy = saccades["saccade_indices_xy"]
    saccade_frames_xy= saccades["saccade_frames_xy"]
    saccade_indices_theta= saccades["saccade_indices_theta"]
    saccade_frames_theta = saccades["saccade_frames_theta"]
    stim_frames = saccades["stim_frames"]
    session_name = os.path.basename(session_path.rstrip("/\\"))

    # ───────── global axis limits (all saccades) ─────────

    if saccade_indices_theta is not None and len(saccade_indices_theta) > 0:
        saccade_indices_theta = np.array(saccade_indices_theta, dtype=int)
        t_all = eye_pos[saccade_indices_theta, 2]
    else:
        t_all = None

    if eye_pos_diff.shape[1] == 3:
        dx, dy, dtheta = eye_pos_diff[:, 0], eye_pos_diff[:, 1], eye_pos_diff[:, 2]
        x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
        # Extract torsion angles and convert to degrees
        t_all = eye_pos[saccade_indices_theta,2] if saccade_indices_theta is not None else None
        torsion_present = True
    else:
        dx, dy = eye_pos_diff[:, 0], eye_pos_diff[:, 1]
        x_all, y_all = eye_pos[saccade_indices_xy, 0], eye_pos[saccade_indices_xy, 1]
        t_all = None
        dtheta = None
        saccade_indices_theta = None
        saccade_frames_theta = None
        torsion_present = False


    pad   = 0.10
    rngX  = x_all.max() - x_all.min()
    rngY  = y_all.max() - y_all.min()
    X_LIM = (x_all.min() - pad*rngX, x_all.max() + pad*rngX)
    Y_LIM = (y_all.min() - pad*rngY, y_all.max() + pad*rngY)
    max_abs = np.max(np.hypot(dx[saccade_indices_xy], dy[saccade_indices_xy]))
    
    #calculate angles for all translational saccades
    angle_all = np.arctan2(dy[saccade_indices_xy],
                            dx[saccade_indices_xy])
    n_all = len(saccade_indices_xy)

    # ───────── master figure (ALL saccades) ─────────
    fig = plt.figure(figsize=(11, 6))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[3, 2])
    ax_quiver = fig.add_subplot(gs[:, 0])
    ax_polar  = fig.add_subplot(gs[0, 1], polar=True)
    ax_linear = fig.add_subplot(gs[1, 1])

    ax_quiver.set_xlim(*X_LIM); ax_quiver.set_ylim(*Y_LIM)
    ax_quiver.set_xlabel('X (°)'); ax_quiver.set_ylabel('Y (°)')
    ax_quiver.set_title(
        f"{session_name}\n"
        f"All translational saccades ({n_all}) — {eye_name}  (stim: {stim_type})\n"
        f"saccade_thresh = {SaccadeConfig.saccade_threshold}, saccade_win = {SaccadeConfig.saccade_win}s\n"
        f"blink_thresh = {SaccadeConfig.blink_threshold}, blink_detection = {SaccadeConfig.blink_detection}s\n"
        )

    cols = np.array([vector_to_rgb(a, max_abs) for a in angle_all])
    ax_quiver.quiver(x_all, y_all,
                        dx[saccade_indices_xy],
                        dy[saccade_indices_xy],
                        angles='xy', scale_units='xy', scale=1,
                        color=cols, alpha=.5)


    plot_angle_distribution(angle_all, ax_polar)
    plot_linear_histogram(angle_all, ax_linear)
    plt.tight_layout()

    # save master figure
    all_fname = f"{session_name}_{eye_name}_ALL_{stim_type}.png"
    fig.savefig(SessionConfig.results_dir / all_fname, dpi=300, bbox_inches='tight')


    # Determine the overall frame range [0, last_frame]
    last_frame = int(saccade_frames_xy.max())
    clipped_any = False
    plot_window = np.arange(0,saccade_window_frames,1)

    # ───────── one figure per stimulus label (skip "All") ─────────
    for label, frames in stim_frames.items():
        if label == "All":
            continue

        # gather 1st saccades within ±plot_window around each stim

        idx_buf = []  # buffer to collect saccade indices for this label

        # sort saccade frames to ensure they are in order
        sorted_pairs_xy = sorted(zip(saccade_frames_xy, saccade_indices_xy))

        for f in frames:

            lower_bound = max(f + plot_window[0], 0)
            upper_bound = min(f + plot_window[-1], saccade_frames_xy.max())

            for sf, idx in sorted_pairs_xy:
                if sf < lower_bound:
                    continue
                elif sf <= upper_bound:
                    idx_buf.append(idx)   # first valid saccade
                    break                 # only take the first one
                else:
                    break                 # skip to next stim

        idx_use = np.array(idx_buf, dtype=int)
        if idx_use.size == 0:
            continue

        ang = np.arctan2(dy[idx_use],
                            dx[idx_use])
        n_cond = len(idx_use)

        fig = plt.figure(figsize=(9, 5))
        gs  = gridspec.GridSpec(3, 2, width_ratios=[3, 2])
        ax_q = fig.add_subplot(gs[:, 0])
        ax_p = fig.add_subplot(gs[0, 1], polar=True)
        ax_l = fig.add_subplot(gs[1, 1])
        ax_t = fig.add_subplot(gs[2, 1]) if torsion_present else None

        ax_q.set_xlim(*X_LIM); ax_q.set_ylim(*Y_LIM)
        ax_q.set_xlabel('X (°)'); ax_q.set_ylabel('Y (°)')
        ax_q.set_title(f"{session_name}\n{eye_name} — {label} (n={n_cond})")

        cols = np.array([vector_to_rgb(a, max_abs) for a in ang])
        ax_q.quiver(eye_pos[idx_use, 0], eye_pos[idx_use, 1],
                    dx[idx_use], dy[idx_use],
                    angles='xy', scale_units='xy', scale=1,
                    color=cols, alpha=.5)

        plot_angle_distribution(ang, ax_p)
        plot_linear_histogram(ang, ax_l)

        if torsion_present:
            # Plot histogram of dtheta only for torsional saccades within window
            idx_buf_torsion = []

            # sort torsion saccade frames
            sorted_pairs_theta = sorted(zip(saccade_frames_theta, saccade_indices_theta))

            for f in frames:
                lower_bound = max(f + plot_window[0], 0)
                upper_bound = min(f + plot_window[-1], saccade_frames_theta.max())

                for sf, idx in sorted_pairs_theta:
                    if sf < lower_bound:
                        continue
                    elif sf <= upper_bound:
                        idx_buf_torsion.append(idx)
                        # break  # first torsional saccade
                    else:
                        break

            idx_torsion_use = np.array(idx_buf_torsion, dtype=int)
            if idx_torsion_use.size > 0:
                dtheta_torsion = dtheta[idx_torsion_use]
                ax_t.hist(dtheta_torsion, bins=20, color='purple', alpha=0.5, edgecolor='k')
                ax_t.set_title("Torsion angle distribution")
                ax_t.set_xlabel("deg/frame")
                ax_t.set_ylabel("Count")
                ax_t.set_xlim(-15, 15)

                # Add curved arrows for each torsional saccade
                for i in idx_torsion_use:
                    x0, y0 = eye_pos[i, 0], eye_pos[i, 1]
                    rotation_magnitude = np.abs(dtheta[i])
                    #print(f"Rotation magnitude for index {i}: {rotation_magnitude}")
                    curvature = -0.3 * np.sign(dtheta[i])  # direction of rotatio
                    arrow = FancyArrowPatch(
                        posA=(x0 - 0.7, y0-1),
                        posB=(x0 + 0.7, y0),
                        connectionstyle=f"arc3,rad={curvature}",
                        color='purple',
                        arrowstyle='->',
                        mutation_scale=10 + 2 * rotation_magnitude,  # scale by magnitude
                        linewidth=1.0,
                        alpha=0.8
                    )
                    ax_q.add_patch(arrow)

        fig.tight_layout()
        fname = f"{session_name}_{eye_name}_{label.replace('/','-')}.png"
        fig.savefig(SessionConfig.results_dir / fname, dpi=300, bbox_inches='tight')


def plot_eye_fixations_between_cue_and_go_by_trial(
    eye_frame, eye_pos, eye_timestamp,
    cue_frame, cue_time, go_frame, go_time,
    max_interval_s=1.0,
    color_all='0.85', s_all=2, alpha_all=0.25,
    s_subset=5,  alpha_subset=0.9,
    cmap_name='tab20',
    results_dir=None, session_name=None, eye_name='Eye'
):
    # ---- coerce to arrays
    eye_ts     = np.asarray(eye_timestamp).ravel()
    eye_x      = np.asarray(eye_pos[:, 0]).ravel()
    eye_y      = np.asarray(eye_pos[:, 1]).ravel()
    cue_frame  = np.asarray(cue_frame).astype(int).ravel()
    cue_time   = np.asarray(cue_time).astype(float).ravel()
    go_frame   = np.asarray(go_frame).astype(int).ravel()
    go_time    = np.asarray(go_time).astype(float).ravel()

    # ---- sort cues & gos by time (carry frames alongside)
    ci = np.argsort(cue_time); cue_time, cue_frame = cue_time[ci], cue_frame[ci]
    gi = np.argsort(go_time);  go_time,  go_frame  = go_time[gi],  go_frame[gi]

    # ---- dedupe by time gaps (keeps first in each contiguous run)
    cue_time_on, cue_frame_on = cue_time, cue_frame
    go_time_on, go_frame_on = go_time, go_frame

    # ---- one-to-one time-based pairing: for each cue, take the NEXT go
    pairs_ct, pairs_gt, pairs_cf, pairs_gf, pairs_dt = [], [], [], [], []
    gptr = 0
    for ct, cf in zip(cue_time_on, cue_frame_on):
        while gptr < len(go_time_on) and go_time_on[gptr] < ct:
            gptr += 1
        if gptr >= len(go_time_on):
            break
        dt = float(go_time_on[gptr] - ct)
        pairs_ct.append(ct);  pairs_gt.append(go_time_on[gptr])
        pairs_cf.append(cf);  pairs_gf.append(int(go_frame_on[gptr]))
        pairs_dt.append(dt)
        gptr += 1  # consume this GO so it’s one-to-one

    pairs_ct = np.asarray(pairs_ct); pairs_gt = np.asarray(pairs_gt)
    pairs_cf = np.asarray(pairs_cf, dtype=int); pairs_gf = np.asarray(pairs_gf, dtype=int)
    pairs_dt = np.asarray(pairs_dt, dtype=float)

    # ---- filter by Δt window (seconds)
    valid_trials = (pairs_dt >= 0) & (pairs_dt < max_interval_s)

    # ---- plotting (use time to find eye samples; safer than frames)
    cmap = cm.get_cmap(cmap_name)
    base_colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    color_cycle = cycle(base_colors)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(eye_x, eye_y, s=s_all, c=color_all, alpha=alpha_all, label='All eye centers')

    legend_handles = []
    trial_num = 0
    for ct, gt, ok, dt in zip(pairs_ct, pairs_gt, valid_trials, pairs_dt):
        if not ok:
            continue
        a = np.searchsorted(eye_ts, min(ct, gt), side='left')
        b = np.searchsorted(eye_ts, max(ct, gt), side='right')
        if b <= a:
            continue
        col = next(color_cycle)
        h = ax.scatter(eye_x[a:b], eye_y[a:b], s=s_subset, c=[col], alpha=alpha_subset,
                       label=f'Trial {trial_num} (Δt={dt:.2f}s)')
        legend_handles.append(h)
        trial_num += 1

    ax.set_aspect('equal')
    ax.set_xlabel('Eye center X (deg)')
    ax.set_ylabel('Eye center Y (deg)')
    ax.set_title(f'Eye centers: all vs. time-paired cue→go windows (<{max_interval_s:.1f}s)')

    # Trim legend clutter
    if len(legend_handles) > 10:
        ax.legend([ax.collections[0], *legend_handles[:10]],
                  ['All eye centers', *[lh.get_label() for lh in legend_handles[:10]]],
                  frameon=False, loc='best')
    else:
        ax.legend(frameon=False, loc='best')

    # optional save
    if results_dir is not None:
        results_dir = Path(results_dir); results_dir.mkdir(exist_ok=True, parents=True)
        fname = f"{session_name or 'session'}_{(eye_name or 'Eye').replace(' ', '')}_cue_go_timepaired.png"
        fig.savefig(results_dir / fname, dpi=300, bbox_inches='tight')

    # ---- diagnostics so you can sanity-check counts & thresholds
    print(f"Raw cues: {cue_time.size} → deduped: {cue_time_on.size}")
    print(f"Raw gos : {go_time.size}  → deduped: {go_time_on.size}")
    if pairs_dt.size:
        print(f"Paired trials: {pairs_dt.size} | dt min/median/max = "
              f"{np.nanmin(pairs_dt):.3f} / {np.nanmedian(pairs_dt):.3f} / {np.nanmax(pairs_dt):.3f} s")
        print(f"Passing (<{max_interval_s:.2f}s): {valid_trials.sum()} trials")

    return (pairs_cf, pairs_gf, pairs_ct, pairs_gt, pairs_dt, valid_trials,fig,ax)

def quantify_fixation_stability_vs_random(
    eye_timestamp, eye_pos,
    pairs_ct, pairs_gt, valid_trials,
    plot=True,
    rng_seed=0
):
    """
    Compare eye stability during fixation windows (cue->go for valid trials)
    to random, equal-duration windows drawn from the rest of the session.

    Returns a dict with arrays of per-window metrics and high-level means.
    Metrics per window:
      - mean_step_disp_px  : mean Euclidean step size (px)
      - mean_speed_px_s    : mean |velocity| (px/s)
      - net_drift_px       : |last - first| (px)
    """

    # --- coerce & sort eye samples by time
    ts = np.asarray(eye_timestamp, dtype=float).ravel()
    x  = np.asarray(eye_pos[:, 0]).ravel()
    y  = np.asarray(eye_pos[:, 1]).ravel()
    if not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts)
        ts, x, y = ts[order], x[order], y[order]

    # --- build fixation windows from valid cue->go pairs
    ct = np.asarray(pairs_ct, dtype=float).ravel()
    gt = np.asarray(pairs_gt, dtype=float).ravel()
    ok = np.asarray(valid_trials, dtype=bool).ravel()

    fix_windows = [(c, g) for c, g, v in zip(ct, gt, ok) if v and (g > c)]
    if len(fix_windows) == 0:
        print("No valid fixation windows. Nothing to compute.")
        return None

    # Merge/clean fixation windows (ensure sorted, non-overlapping)
    fix_windows = sorted(fix_windows, key=lambda w: w[0])
    merged = []
    for s, e in fix_windows:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)  # merge overlaps
    fix_windows = [(s, e) for s, e in merged]

    # --- helper: compute metrics inside [t0, t1]
    def window_metrics(t0, t1):
        a = np.searchsorted(ts, t0, side='left')
        b = np.searchsorted(ts, t1, side='right')
        if b - a < 2:
            return np.nan, np.nan, np.nan
        dx = np.diff(x[a:b])
        dy = np.diff(y[a:b])
        dt = np.diff(ts[a:b])

        # valid finite steps with positive dt
        m = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt > 0)
        if not np.any(m):
            return np.nan, np.nan, np.nan

        step_disp = np.hypot(dx[m], dy[m])              # pixels
        speed     = step_disp / dt[m]                    # px/s
        drift     = np.hypot(x[b-1] - x[a], y[b-1] - y[a])  # pixels

        return float(step_disp.mean()), float(speed.mean()), float(drift)

    # --- compute fixation metrics per *original* (unmerged) window
    # (We’ll compare one random window per fixation with the same duration)
    orig_fix_windows = [(c, g) for c, g, v in zip(ct, gt, ok) if v and (g > c)]
    fix_len = np.array([g - c for c, g in orig_fix_windows], dtype=float)

    fix_mean_step = np.empty(len(orig_fix_windows))
    fix_mean_speed = np.empty(len(orig_fix_windows))
    fix_drift = np.empty(len(orig_fix_windows))
    for i, (c, g) in enumerate(orig_fix_windows):
        fix_mean_step[i], fix_mean_speed[i], fix_drift[i] = window_metrics(c, g)

    # --- build allowed (non-fixation) intervals across the whole session
    session_start, session_end = float(ts[0]), float(ts[-1])
    # complement of merged fixation windows
    allowed = []
    cursor = session_start
    for s, e in fix_windows:
        if s > cursor:
            allowed.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < session_end:
        allowed.append((cursor, session_end))

    # convenience: function to draw a random start for a given duration
    rng = np.random.default_rng(rng_seed)
    def sample_random_window(duration):
        # find allowed intervals that can fit this duration
        candidates = [(a, b) for (a, b) in allowed if (b - a) >= duration]
        if not candidates:
            return None  # cannot fit (rare)
        a, b = candidates[rng.integers(0, len(candidates))]
        start = float(a) + rng.random() * float((b - a) - duration)
        return (start, start + duration)

    # --- draw one random window per fixation (equal duration) and compute metrics
    rnd_mean_step = np.empty(len(orig_fix_windows))
    rnd_mean_speed = np.empty(len(orig_fix_windows))
    rnd_drift = np.empty(len(orig_fix_windows))

    for i, L in enumerate(fix_len):
        rw = sample_random_window(L)
        if rw is None:
            rnd_mean_step[i] = rnd_mean_speed[i] = rnd_drift[i] = np.nan
        else:
            rnd_mean_step[i], rnd_mean_speed[i], rnd_drift[i] = window_metrics(*rw)

    # --- summarize (ignore NaNs)
    def nice_stats(arr):
        arr = np.asarray(arr, dtype=float)
        m = np.isfinite(arr)
        if not m.any():
            return np.nan, np.nan, 0
        vals = arr[m]
        return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size)), int(vals.size)

    ms_fix,  se_fix,  n_fix  = nice_stats(fix_mean_step)
    ms_rnd,  se_rnd,  n_rnd  = nice_stats(rnd_mean_step)
    sp_fix,  se_spf,  _      = nice_stats(fix_mean_speed)
    sp_rnd,  se_spr,  _      = nice_stats(rnd_mean_speed)
    dr_fix,  se_drf,  _      = nice_stats(fix_drift)
    dr_rnd,  se_drr,  _      = nice_stats(rnd_drift)

    print("=== Stability summary (mean ± s.e.m.) ===")
    print(f"Mean step displacement (px):  fix {ms_fix:.3f} ± {se_fix:.3f}   vs   rand {ms_rnd:.3f} ± {se_rnd:.3f}  (n={n_fix} pairs)")
    print(f"Mean speed (px/s):            fix {sp_fix:.3f} ± {se_spf:.3f}   vs   rand {sp_rnd:.3f} ± {se_spr:.3f}")
    print(f"Net drift (px):               fix {dr_fix:.3f} ± {se_drf:.3f}   vs   rand {dr_rnd:.3f} ± {se_drr:.3f}")

    # --- optional quick plot
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
            # y=x reference line
            lo = np.nanmin(np.concatenate([a[m], b[m]]))
            hi = np.nanmax(np.concatenate([a[m], b[m]]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Fixation"); ax.set_ylabel("Random")
            ax.set_title(title)
            ax.set_aspect('equal', adjustable='box')
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

