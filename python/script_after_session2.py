"""Example script demonstrating use of :func:`detect_saccades`.

The script constructs a single ``SaccadeDetectionConfig`` instance and
passes it to ``detect_saccades``.  The data used here are tiny dummy
arrays purely for illustration; replace them with real measurements in a
real analysis pipeline.
"""

import numpy as np

from eyehead.functions import SaccadeDetectionConfig, detect_saccades

# Dummy data ---------------------------------------------------------------
l_x = np.array([0.0, 1.0, 2.0])
l_y = np.array([0.0, 0.0, 0.0])
r_x = np.array([1.0, 2.0, 3.0])
r_y = np.array([0.0, 0.0, 0.0])

eye_x = np.array([0.5, 1.5, 2.5])
eye_y = np.array([0.0, 0.1, 0.0])
eye_frame = np.array([0, 1, 2])

vd_lx = vd_ly = vd_rx = vd_ry = np.zeros(3)
torsion = np.zeros(3)

# Configuration ------------------------------------------------------------
config = SaccadeDetectionConfig(
    calibration_factor=1.0,
    blink_velocity_threshold=30.0,
    saccade_threshold=10.0,
    blink_detection=0,
    saccade_threshold_torsion=None,
)

# Call ---------------------------------------------------------------------
saccades = detect_saccades(
    l_x, l_y, r_x, r_y,
    eye_x, eye_y,
    eye_frame,
    config,
    vd_axis_lx=vd_lx, vd_axis_ly=vd_ly,
    vd_axis_rx=vd_rx, vd_axis_ry=vd_ry,
    torsion_angle=torsion,
)

print("Detected", len(saccades["saccade_indices_xy"]), "saccades")

