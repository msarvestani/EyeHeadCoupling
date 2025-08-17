from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.signal import medfilt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in *arr* in-place."""
    arr = np.asarray(arr, dtype=float)
    nans = np.isnan(arr)
    if np.any(nans):
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SaccadeDetectionConfig:
    """Configuration parameters for :func:`detect_saccades`.

    Attributes
    ----------
    calibration_factor:
        Pixel-per-degree calibration (scalar or ``(fx, fy)`` sequence).
    blink_velocity_threshold:
        Threshold on eyelid velocity used to identify blinks.
    saccade_threshold:
        Velocity threshold (deg/frame) for detecting translational saccades.
    blink_detection:
        Whether to remove saccades occurring during blinks.
    saccade_threshold_torsion:
        Optional velocity threshold for detecting torsional saccades.
    """

    calibration_factor: Union[float, np.ndarray]
    blink_velocity_threshold: float
    saccade_threshold: float
    blink_detection: int = 0
    saccade_threshold_torsion: Optional[float] = None


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def detect_saccades(
    marker1_x, marker1_y, marker2_x, marker2_y,
    gaze_x, gaze_y,
    eye_frames,
    config: SaccadeDetectionConfig,
    vd_axis_lx=None, vd_axis_ly=None, vd_axis_rx=None, vd_axis_ry=None,
    torsion_angle=None,
):
    """Detect saccades from eye tracking data.

    Parameters
    ----------
    marker1_x, marker1_y, marker2_x, marker2_y : array_like
        Coordinates of the eyelid markers.
    gaze_x, gaze_y : array_like
        Gaze position from Bonsai.
    eye_frames : array_like
        Frame numbers associated with ``gaze_x``/``gaze_y``.
    config : :class:`SaccadeDetectionConfig`
        Parameters controlling the detection.
    vd_axis_lx, vd_axis_ly, vd_axis_rx, vd_axis_ry : array_like, optional
        Vertical displacement axis of the eyelids, used for blink detection.
    torsion_angle : array_like, optional
        Torsion angle of the eye.
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
    cal = np.asarray(config.calibration_factor, dtype=np.float64)
    if cal.ndim == 0:
        fx = fy = float(cal)
    elif cal.shape == (2,):
        fx, fy = float(cal[0]), float(cal[1])
    else:
        raise ValueError("calibration_factor must be scalar or length-2 sequence")

    eye_camera[:, 0] /= fx
    eye_camera[:, 1] /= fy

    # 2. instantaneous velocity → speed
    dx = np.ediff1d(eye_camera[:, 0], to_begin=0)
    dy = np.ediff1d(eye_camera[:, 1], to_begin=0)
    xy_speed = np.sqrt(dx**2 + dy**2)

    xy_mask = xy_speed >= config.saccade_threshold

    # 3. torsional velocity
    if torsion_angle is not None:
        torsion_angle = interpolate_nans(np.asarray(torsion_angle, dtype=np.float64))
        dtheta = np.ediff1d(torsion_angle, to_begin=0)
        torsion_speed = np.abs(dtheta)
        thresh = (config.saccade_threshold_torsion
                  if config.saccade_threshold_torsion is not None else np.inf)
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
        eye_pos = np.column_stack([eye_camera, torsion_angle])
        eye_vel = np.column_stack([dx, dy, dtheta])
    else:
        eye_pos = eye_camera
        eye_vel = np.column_stack([dx, dy])

    # 6. Optional blink removal
    if (config.blink_detection and vd_axis_lx is not None and vd_axis_ly is not None
            and vd_axis_rx is not None and vd_axis_ry is not None):
        vd_axis_left = np.vstack([vd_axis_lx, vd_axis_ly]).T
        vd_axis_right = np.vstack([vd_axis_rx, vd_axis_ry]).T
        vd_axis_d = np.linalg.norm(vd_axis_right - vd_axis_left, axis=1)
        vd_axis_vel = np.gradient(vd_axis_d)
        blink_indices = np.where(np.abs(vd_axis_vel) > config.blink_velocity_threshold)[0]
        mask = ~np.isin(saccade_indices_xy, blink_indices)
        saccade_indices_xy = saccade_indices_xy[mask]
        saccade_frames_xy = eye_frames[saccade_indices_xy]

    return {
        "eye_pos": eye_pos,
        "eye_vel": eye_vel,
        "saccade_indices_xy": saccade_indices_xy,
        "saccade_frames_xy": saccade_frames_xy,
        "saccade_indices_theta": saccade_indices_theta,
        "saccade_frames_theta": saccade_frames_theta,
    }

