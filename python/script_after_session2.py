
from eyehead.functions import *

import sys
import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import ShortTimeFFT,butter,hilbert,sosfiltfilt,medfilt
from scipy.signal.windows import gaussian
import scipy.stats as stats
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA
import os
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


# Configuration ------------------------------------------------------------
config = SaccadeDetectionConfig(
    calibration_factor=3.76,
    blink_velocity_threshold=10.0,
    saccade_threshold=1.0,
    blink_detection=1,
    saccade_threshold_torsion=None,
)

cal = 3.76  # Calibration factor for the pixels to degrees
ttl_freq = 60  # TTL frequency in Hz

# Parameters for nlink and saccade detection
blink_detection = 1
blink_thresh= 10
saccade_thresh= 1.0
torsion_velocity_thresh = 1.5
saccade_win=0.7    # Window size for saccade detection in seconds

#folder_path = select_folder() #this won't work if you're running jupyter lab in browser, so hard coding


###################################### Paris

#First day when we started doing interleaved stim.
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-09T13_09_02\\" # interleaved

#Second day where she licked a lot!
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-11T12_50_45\\" #no stim
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-11T13_02_29\\" # interleaved
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-11T13_15_39\\" #  interleaved

#Not much licking on this day
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-13T13_07_55\\" #interleaved
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-13T13_25_04\\" #no-stim session

#Motivated on this day, but first day where juice was only given for saccades
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-16T12_13_51\\" #interleaved

#second day juice was given for saccades only but she was stressed 
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-17T12_50_28\\" #interleaved

#third day juce was given for saccades only, but she was not thirsty
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-18T12_19_39\\" #interleaved

#fourth day, but she didn't pay attention the whole time
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-19T12_17_52\\" #interleaved


folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-06-30T15_20_56\\" #just L/R


#this is after ratnadeep fixed a bunch of issues with the code! and data looks great!!
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-02T15_14_29\\" #just L/R


#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-07T14_49_12\\" #just L/R
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-07T15_12_22\\" #just L/R

#good session!
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-08T15_04_12\\" #just L/R

#good session! first day we tried U/D stim
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-09T15_29_43\\" #just L/R
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-09T15_49_30\\" #L/R and U/D 

#tough day, she was stressed and didn't pay attention
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-10T15_36_39\\" # U/D --bad
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-10T15_47_04\\" #L/R and U/D 
######################################## Bayleaf
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\Rat22_Bayleaf_server\Rat022_2025-06-10T16_23_02\\"   #no stim
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\Rat22_Bayleaf_server\Rat022_2025-06-10T16_10_21\\"   #no stim

# pretty bad performance this day, used wrong head bar
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-11T14_22_34\\" #L/R and U/D 
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-11T14_49_16\\" #L/R and U/D 


#came in on a sunday to do a session, figuring out she'd be motivated and calm
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-13T17_38_53\\" #L/R and U/D 
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-13T17_46_15\\" #L/R and U/D, moved u/down closer
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-13T17_52_47\\" #U/D only
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-13T18_19_05\\" #L/R only

#good long session today
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T14_58_27\\" #L/R only
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_04_46\\" #L/R only
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_11_17\\" #interleaved
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_16_32\\" #interleaved
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_26_38\\" #Up/Down
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_39_13\\" #Up / 2 levels
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_49_51\\" #Up / Down 2 levels
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-14T15_56_54\\" #Up / Down 1 levels

#good long session today, where we looked at torsion online
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-15T16_16_03\\" #interleaved
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-15T16_36_23\\" #U/D only
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-15T16_45_37\\" #L/R


################################################### #torsion training
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-17T15_32_42\\" #interleaved


folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-21T15_08_33\\" #Up/down
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-21T15_45_48\\" #Up/down
#testing monitor positions to see if down torsion gets better
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-22T15_02_57\\" #Up/down
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-22T15_25_56\\" #Up/down

##################################################### fixation training ####################
# #first day where we started punishing for not fixating during blue spot
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-30T15_29_31\\" #interleaved
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-30T15_39_26\\" #interleaved
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-30T15_47_15\\" #interleaved

# # #second day where we punish for not fixating during blue spot
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-31T15_23_59\\" #interleaved
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-07-31T15_56_37\\" #interleaved

#third day where we punish for not fixating during blue spot
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-01T15_15_48\\" #interleaved
folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-06T16_40_40\\" #interleaved


###################################################################### anti-saccade training
#folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-11T16_40_24\\" 
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-12T15_04_14\\"
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-13T14_48_16\\" #interleaved
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-13T15_02_30\\" #interleaved
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-14T15_06_12\\" #interleaved
# folder_path = r"X:\Experimental_Data\EyeHeadCoupling_RatTS_server\TSh01_Paris_server\Tsh001_2025-08-14T15_12_24\\" #interleaved



results_dir = Path(folder_path) / "Results\\"
results_dir.mkdir(exist_ok=True)

session_name = os.path.basename(folder_path.rstrip("/\\"))

# determine which camera is used based on the folder name and cut off date
camera_side = determine_camera_side(folder_path)
eye_name = f"{camera_side} Eye"  # e.g. "Left camera" or "Right camera"
print(f"Using camera side: {camera_side}")



print(f"Scanning folder: {folder_path}")
print(f"Found {len(os.listdir(folder_path))} files")

# Scan the folder for specific files
for f in os.listdir(folder_path):
    f_lower = f.lower()
    full_path = os.path.join(folder_path, f)

    if 'imu' in f_lower:
        IMU_file = full_path
    if 'camera' in f_lower:
        camera_file = full_path
    if 'go' in f_lower:
        go_file = full_path
    if f"ellipse_center_XY_{camera_side}".lower() in f_lower:
        ellipse_center_XY_file = full_path
    if f"origin_of_eyecoordinate_{camera_side}".lower() in f_lower:
        origin_of_eye_coordinate_file = full_path
    if f"vdaxis_{camera_side}".lower() in f_lower:
        vdaxis_file = full_path
    if f"torsion_{camera_side}".lower() in f_lower:
        torsion_file = full_path
    if 'endoftrial' in f_lower:
        end_of_trial_file = full_path
    if 'cue' in f_lower:
        cue_file = full_path

## Read the camera data and map between camera TTL (for saccades) and Bonsai TTLs (for frames)
camera_data = np.genfromtxt(camera_file, delimiter=',', skip_header=1, dtype=np.float64)
[bonsai_frame, bonsai_time] = camera_data[:, 0], camera_data[:, 1]
bonsai_frame = bonsai_frame.astype(int)  # Convert bonsai_frame to integer type


### Read the go file for the start of stim in the trial 
new_go_data_format=0
go_data = np.genfromtxt(clean_csv(go_file), delimiter=',', skip_header=1, dtype=np.float64)
if go_data.shape[1]>3:
    new_go_data_format = 1
    [go_frame, go_time, go_direction_x,go_direction_y] = go_data[:, 0], go_data[:, 1], go_data[:, 2], go_data[:,3]
else:
    [go_frame, go_time, go_direction] = go_data[:, 0], go_data[:, 1], go_data[:, 2]
go_frame = go_frame.astype(int)  # Convert go_frame to integer type

### Read the ellipse center XY file  
ellipse_center_XY_data = np.genfromtxt(clean_csv(ellipse_center_XY_file), delimiter=',', skip_header=1, dtype=np.float64)
[eye_frame,eye_timestamp,eye_x,eye_y] = ellipse_center_XY_data[:, 0], ellipse_center_XY_data[:, 1], ellipse_center_XY_data[:, 2], ellipse_center_XY_data[:, 3]
eye_frame = eye_frame.astype(int)  # Convert eye_frame to integer type
eye_x = interpolate_nans(eye_x)  # Interpolate NaN values in eye_x
eye_y = -1*interpolate_nans(eye_y)  # Interpolate NaN values in eye_y

### Read the origin of eye coordinate file
origin_of_eye_coordinate_data = np.genfromtxt(clean_csv(origin_of_eye_coordinate_file), delimiter=',', skip_header=1, dtype=np.float64)
[origin_frame,o_ts,l_x,l_y,r_x,r_y] = origin_of_eye_coordinate_data[:, 0], origin_of_eye_coordinate_data[:, 1], origin_of_eye_coordinate_data[:, 2], origin_of_eye_coordinate_data[:, 3], origin_of_eye_coordinate_data[:, 4], origin_of_eye_coordinate_data[:, 5]
origin_frame = origin_frame.astype(int)  # Convert origin_frame_r to integer type
l_x = interpolate_nans(l_x)  # Interpolate NaN values in l_rx
r_x = interpolate_nans(r_x)  # Interpolate NaN values in r_rx 
l_y = interpolate_nans(l_y)  # Interpolate NaN values in l_ry
r_y = interpolate_nans(r_y)  # Interpolate NaN values in r_ry


## Read the torsion data - this is used for torsion detection
torsion_data = np.genfromtxt(clean_csv(torsion_file), delimiter=',', skip_header=1, dtype=np.float64)
[torsion_frame, torsion_ts, torsion] = torsion_data[:, 0], torsion_data[:, 1], torsion_data[:, 2]
torsion_frame = torsion_frame.astype(int)   # Convert torsion_frame to integer type
# Interpolate NaN values        
torsion = interpolate_nans(torsion)


## Read the vertical (VD) axis data - this is used for blink detection
vdaxis_data = np.genfromtxt(clean_csv(vdaxis_file),delimiter=',',skip_header=1,dtype=np.float64)
[vd_frame,vd_ts,vd_lx,vd_ly,vd_rx,vd_ry] = vdaxis_data[:,0],vdaxis_data[:,1],vdaxis_data[:,2],vdaxis_data[:,3],vdaxis_data[:,4],vdaxis_data[:,5]
vd_frame = vd_frame.astype(int)
# Interpolate NaN values
vd_lx = interpolate_nans(vd_lx)
vd_ly = interpolate_nans(vd_ly)
vd_rx = interpolate_nans(vd_rx)
vd_ry = interpolate_nans(vd_ry)

        
### Read the IMU data for the accelerometer and gyroscope
imu_data = np.genfromtxt(IMU_file, delimiter=',', skip_header=1, dtype=np.float64)
[imu_time,a_x,a_y,a_z,g_x,g_y,g_z,m_x,m_y,m_z] = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2], imu_data[:, 3], imu_data[:, 4], imu_data[:, 5], imu_data[:, 6], imu_data[:, 7], imu_data[:, 8], imu_data[:, 9]
imu_time = imu_time.astype(np.float64)  # Ensure imu_time is in float64 format
# Interpolate NaN values in IMU data
a_x = interpolate_nans(a_x)
a_y = interpolate_nans(a_y)
a_z = interpolate_nans(a_z)
g_x = interpolate_nans(g_x)
g_y = interpolate_nans(g_y)
g_z = interpolate_nans(g_z)
m_x = interpolate_nans(m_x)
m_y = interpolate_nans(m_y)
m_z = interpolate_nans(m_z)


### Read the endoftrial file- This file tells us when the trial ends, stim direction, eye movement direction, torsion angle, and whether the trial was successful
try:
    end_of_trial_data = np.genfromtxt(clean_csv(end_of_trial_file), delimiter=',', skip_header=1, dtype=np.float64)
    [end_of_trial_frame, end_of_trial_ts, trial_stim_direction, trial_eye_movement_direction, trial_torsion_angle, trial_success] = end_of_trial_data[:, 0], end_of_trial_data[:, 1], end_of_trial_data[:, 2], end_of_trial_data[:, 3], end_of_trial_data[:, 4], end_of_trial_data[:, 5]
    end_of_trial_frame = end_of_trial_frame.astype(int)  # Convert end_of_trial_frame to integer type
    # Interpolate NaN values in trial_torsion_angle
    trial_torsion_angle = interpolate_nans(trial_torsion_angle)
    trial_eye_movement_direction = interpolate_nans(trial_eye_movement_direction)
except IndexError:
    end_of_trial_data = np.genfromtxt(clean_csv(end_of_trial_file), delimiter=',', skip_header=1, dtype=np.float64)
    [end_of_trial_frame, end_of_trial_ts, trial_stim_direction, trial_eye_movement_direction, trial_success] = end_of_trial_data[:, 0], end_of_trial_data[:, 1], end_of_trial_data[:, 2], end_of_trial_data[:, 3], end_of_trial_data[:, 4]
    end_of_trial_frame = end_of_trial_frame.astype(int)  # Convert end_of_trial_frame to integer type

    trial_eye_movement_direction = interpolate_nans(trial_eye_movement_direction)
except ValueError:
    print("No end of trial data found. Skipping this step.")
for t in range(len(trial_success)):
    if trial_success[t] == 0:
        if trial_eye_movement_direction[t] != -1:
            trial_success[t] = -1   # Incorrect trial

# --- Read the cue file (every-frame logging) and keep only trial onsets ---
cue_data = np.genfromtxt(clean_csv(cue_file), delimiter=',', skip_header=1, dtype=np.float64)

cue_frame_raw     = cue_data[:, 0].astype(int)
cue_time_raw      = cue_data[:, 1].astype(float)
cue_direction_raw = cue_data[:, 2]  # keep dtype as-is (often int/float)

# Sort by time to be safe (carry frames/directions along)
order = np.argsort(cue_time_raw)
cue_time_raw      = cue_time_raw[order]
cue_frame_raw     = cue_frame_raw[order]
cue_direction_raw = cue_direction_raw[order]

# Define what counts as a 'new trial' gap between consecutive cue rows
TRIAL_GAP_S = 1.5  # <-- adjust to 2–3 if your inter-trial gap is longer

# Keep only the FIRST row after each large time jump (trial onset)
onset_idx = np.r_[0, np.where(np.diff(cue_time_raw) > TRIAL_GAP_S)[0] + 1]
cue_frame     = cue_frame_raw[onset_idx]
cue_time      = cue_time_raw[onset_idx]
cue_direction = cue_direction_raw[onset_idx]

print(f"Detected {cue_frame.size} cue onsets from {cue_frame_raw.size} cue rows (gap > {TRIAL_GAP_S}s).")

# --- Align lengths with GO events (1 line per trial) ---
if len(cue_frame) != len(go_frame):
    n = min(len(cue_frame), len(go_frame))
    if len(cue_frame) > len(go_frame):
        print(f"Warning: {len(cue_frame)} cue onsets but {len(go_frame)} GO rows; truncating cues to {n}.")
        cue_frame, cue_time, cue_direction = cue_frame[:n], cue_time[:n], cue_direction[:n]
    else:
        print(f"Warning: {len(cue_frame)} cue onsets but {len(go_frame)} GO rows; truncating GO to {n}.")
        go_frame, go_time = go_frame[:n], go_time[:n]

d_frames = np.diff(go_frame)    # successive differences (frames)
d_sec = d_frames / ttl_freq

fig =plt.figure(figsize=(8,3))
plt.plot(d_sec, marker='o')
plt.xlabel('Stimulus index')
plt.ylabel('Δtime (s) to next stim')
plt.title('Seconds between successive Go Stims')
plt.grid(alpha=.3)
plt.tight_layout()
plt.show()

# optional: save alongside other figures
prob_fname = f"{session_name}_{eye_name}_Stim_Interval.png"
fig.savefig(results_dir / prob_fname, dpi=300, bbox_inches='tight')

# --------------------------------------------------------------
# ONE analyse-&-plot call for the Eye saccades
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


saccades["stim_frames"], stim_type = organize_stims(
    go_frame,
    go_dir_x = go_direction_x,
    go_dir_y = go_direction_y,
)

sort_plot_saccades(
    saccades,
    saccade_window= saccade_win*ttl_freq,
    session_path = folder_path,
    stim_type    = stim_type,
    eye_name     = eye_name,

)

# ============================================================
# Timeline plot: saccades (grey) + colour-coded stimuli
# ============================================================
# -----------------------------------------------------------------
# Assumes these objects already exist in the workspace
# -----------------------------------------------------------------
# right["saccade_frames"]   – Bonsai frame IDs of all saccades
# go_frame                  – Bonsai frame IDs of stimuli
# go_dir_x, go_dir_y        – direction codes (can be None)
# ttl_freq                  – camera TTL rate (Hz)
# results_dir               – Path(folder_path) / "Results"
# -----------------------------------------------------------------

# 1) prep data ----------------------------------------------------
saccade_frames = np.asarray(saccades["saccade_frames_xy"], dtype=int)

# fallback arrays if dir arrays are missing
gx = go_dir_x if (go_dir_x is not None) else np.zeros_like(go_frame)
gy = go_dir_y if (go_dir_y is not None) else np.zeros_like(go_frame)


# ── 1-bis. count how many of each stimulus -----------------------------
eps = 1e-6                      # tolerance for “zero”
is_left   = (np.abs(gy) < eps) & (gx < -eps)
is_right  = (np.abs(gy) < eps) & (gx >  eps)
is_down   = (np.abs(gx) < eps) & (gy < -eps)
is_up     = (np.abs(gx) < eps) & (gy >  eps)

n_left, n_right = is_left.sum(),  is_right.sum()
n_down, n_up    = is_down.sum(),  is_up.sum()


# palette mapping
palette = {'L': 'green', 'R': 'pink', 'D': 'blue', 'U': 'red', 'NA': 'gray'}

# build colour list per stimulus
colors = []
for x, y in zip(gx, gy):
    if abs(y) > 1e-6:                       # Up / Down has priority
        colors.append(palette['U' if y > 0 else 'D'])
    elif abs(x) > 1e-6:                     # Left / Right
        colors.append(palette['R' if x > 0 else 'L'])
    else:
        colors.append(palette['NA'])

# sort frames & colours together
order      = np.argsort(go_frame)
t_stim     = go_frame[order] / ttl_freq
colors     = [colors[i] for i in order]

# convert saccade frames to seconds
t_sacc = np.sort(saccade_frames) / ttl_freq

# 2) plot ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 2.5))

# saccades: grey vertical ticks at y = 0
ax.vlines(t_sacc, -0.1, 0.1, colors='0.25', linewidth=1)

# stimuli: colour ticks at y = 1
ax.vlines(t_stim, 0.9, 1.1, colors=colors, linewidth=2)

# axes formatting
ax.set_yticks([0, 1])
ax.set_yticklabels(['Saccade', 'Stim'])
ax.set_xlabel('Time (s)')
ax.set_title('Timeline of saccades and stimuli')
ax.set_xlim(t_sacc.min() - 1, t_sacc.max() + 1)
ax.set_ylim(-0.5, 1.5)
ax.grid(axis='x', alpha=.3)

# legend
handles = [
    mlines.Line2D([], [], color='0.25', marker='|', ls='', markersize=10,
                  label='Saccade'),
    mlines.Line2D([], [], color='green', marker='|', ls='', markersize=10,
                  label=f'Stim Left  (n={n_left})'),
    mlines.Line2D([], [], color='pink',  marker='|', ls='', markersize=10,
                  label=f'Stim Right (n={n_right})'),
    mlines.Line2D([], [], color='blue',  marker='|', ls='', markersize=10,
                  label=f'Stim Down  (n={n_down})'),
    mlines.Line2D([], [], color='red',   marker='|', ls='', markersize=10,
                  label=f'Stim Up    (n={n_up})')
]
ax.legend(handles=handles, loc='upper right', ncol=5, fontsize=9, framealpha=.9)

plt.tight_layout()


# optional: save alongside other figures
prob_fname = f"{session_name}_{eye_name}_timeline_saccade_vs_stim.png"
fig.savefig(results_dir / prob_fname, dpi=300, bbox_inches='tight')

# ============================================================
#  Probability of ≥1 saccade within 0.5 s of each stimulus


# ----- parameters -------------------------------------------
win     = saccade_win                  # seconds after onset
w_frames = int(win * ttl_freq)  # convert to frames

# direction masks
gx = go_direction_x if go_direction_x is not None else np.zeros_like(go_frame)
gy = go_direction_y if go_direction_y is not None else np.zeros_like(go_frame)

dir_info = {
    'Left' :  (gx < -1e-6,  'green'),
    'Right':  (gx >  1e-6,  'pink'),
    'Down' :  (gy < -1e-6,  'blue'),
    'Up'   :  (gy >  1e-6,  'red')
}

labels, probs, colors = [], [], []

for label, (mask, col) in dir_info.items():
    stim_frames = go_frame[mask]
    n_stim      = len(stim_frames)
    if n_stim == 0:
        continue                                 # skip if this direction absent
    # check each stimulus: does ANY saccade happen within +win seconds?
    has_sacc = [( (saccade_frames >= f) & (saccade_frames <= f + w_frames) ).any()
                for f in stim_frames]
    prob = np.mean(has_sacc)                    # fraction of stimuli with ≥1 sac
    labels.append(label)
    probs.append(prob)
    colors.append(col)
    print(f"{label:5s}: {prob*100:5.1f}%  ({sum(has_sacc)}/{n_stim} stimuli)")

# ----- bar chart --------------------------------------------
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(labels, probs, color=colors, edgecolor='k')
ax.set_ylim(0, 1)
ax.set_ylabel(f"P(saccade within {win}s)")
ax.set_title(f"Probability of a saccade in first {win} s after stimulus")
ax.grid(axis='y', alpha=.3)
plt.tight_layout()

# ----- save (optional) --------------------------------------


# optional: save alongside other figures
prob_fname = f"{session_name}_{eye_name}_saccade_prob_within_{win*1000:.0f}ms.png"
fig.savefig(results_dir / prob_fname, dpi=300, bbox_inches='tight')

## For each go frame, look at the last available frame before the go frame in the eye position traces.
eye_position_during_fixation=[]
eye_position_during_fixation_success = []
## Basic sanity check regarding trials statistics collected from different sources
if (len(end_of_trial_frame) != len(go_frame)):
    print(f"Warning: Number of end of trial frames ({len(end_of_trial_frame)}) does not match number of go frames ({len(go_frame)}).")

for i, f in enumerate(go_frame[:len(trial_success)]):
    #last_frame_before_go = eye_frame[eye_frame < f][-31:-1]  # last 30 eye frame before stim
    #eye_pos = np.mean(saccades["eye_pos"][np.where(eye_frame == last_frame_before_go)[0]], axis=0)  # average eye position at that frame
    eye_pos = np.mean(saccades["eye_pos"][np.where(eye_frame <f)[0][-7:-1]], axis=0)  # average eye position at the last 30 frames before the go frame
    if len(eye_pos) == 0:
        print(f"Warning: No eye position data found for go frame {f}. Skipping this frame.")
        continue
    eye_position_during_fixation.append(eye_pos)
    if trial_success[i] == 1:
        eye_position_during_fixation_success.append(eye_pos)
eye_position_during_fixation = np.array(eye_position_during_fixation)
eye_position_during_fixation_success = np.array(eye_position_during_fixation_success)
# Ratio of the total spread of the eye positions during fixation to the total spread of all eye positions
eye_pos_all = saccades["eye_pos"]  
spread_fixation = np.std(eye_position_during_fixation, axis=0)
spread_all = np.std(eye_pos_all, axis=0)
ratio_spread = spread_fixation / spread_all
print(f"Ratio of spread during fixation to all eye positions: {ratio_spread}")


# Plot all the eye positions during the session first and then the eye positions during fixation
fig = plt.figure(figsize=(8, 6))
plt.scatter(saccades["eye_pos"][:, 0], saccades["eye_pos"][:, 1], color='red', alpha=0.1, label='All Eye Positions')
plt.scatter(eye_position_during_fixation[:, 0], eye_position_during_fixation[:, 1], color='blue', alpha=0.4, label='Eye Positions During Fixation')
plt.scatter(eye_position_during_fixation_success[:, 0], eye_position_during_fixation_success[:, 1], color='green', alpha=0.5, label='Eye Positions During Fixation (Successful Trials)')
plt.xlabel('X Position (deg)')
plt.ylabel('Y Position (deg)')
plt.title('Eye Positions in the orbit during the whole session and during fixation')
plt.legend()
plt.grid()
plt.show()

rng = np.random.default_rng(123)  # set a seed for reproducibility
cue_frame_jit = (cue_frame + rng.integers(0, 101, size=cue_frame.shape)).astype(int)
cue_time_jit  = cue_time + rng.uniform(0.0, 5.0, size=cue_time.shape)

(pairs_cf, pairs_gf, pairs_ct, pairs_gt, pairs_dt, valid_trials,fig,ax)= plot_eye_fixations_between_cue_and_go_by_trial(
    eye_frame=eye_frame, eye_pos=saccades["eye_pos"], eye_timestamp=eye_timestamp,
    cue_frame=cue_frame, cue_time=cue_time,
    #cue_frame=cue_frame_jit, cue_time=cue_time_jit,    
    go_frame=go_frame,  go_time=go_time,
    max_interval_s=1,
    results_dir=results_dir, session_name=session_name, eye_name=eye_name
)

# Now quantify stability vs random (and show a small paired scatter summary)
stats = quantify_fixation_stability_vs_random(
    eye_timestamp=eye_timestamp,
    eye_pos=saccades["eye_pos"],
    pairs_ct=pairs_ct, pairs_gt=pairs_gt,
    valid_trials=valid_trials,
    plot=True,     # set False if you only want numbers
    rng_seed=0
)

## This section is only for the fixation experiments. Might throw an error if the data is not from a fixation experiment.
fixation_experiment = False
#----------------------------------------------------------------------------------------------------------
## Divide the successful fixation trials based on how long it took the animal to reach fixation point. We calculate this from the difference between the cue frame and go_frame.
if fixation_experiment:
    fixation_time_theshold = 0.75  # seconds
    total_fixation_time_per_trial = go_time - cue_time
    #plt.hist(total_fixation_time_per_trial, bins=20, color='gray', alpha=0.7)
    mask_short_fixation = total_fixation_time_per_trial <= fixation_time_theshold
    mask_long_fixation = total_fixation_time_per_trial > fixation_time_theshold
    short_fixation_frames = []
    for i, f in enumerate(cue_frame[mask_short_fixation]):
        short_fixation_frames.append(eye_frame[np.where(eye_frame>=f)[0][0]:np.where(eye_frame<=go_frame[mask_short_fixation][i])[0][-1]])
    short_fixation_frames = np.array(short_fixation_frames, dtype=object)  # array of arrays
    short_fixation_eye_positions = []
    for frames in short_fixation_frames:
        positions = []
        for fr in frames:
            pos = saccades["eye_pos"][np.where(eye_frame == fr)[0]]
            if len(pos) > 0:
                positions.append(pos[0])
        positions = np.array(positions)
    # positions = positions - np.mean(positions, axis=0)  # center the positions around the mean
        short_fixation_eye_positions.append(positions)
    short_fixation_eye_positions = np.array(short_fixation_eye_positions, dtype=object)  # array of arrays
    long_fixation_frames = []
    for i, f in enumerate(cue_frame[mask_long_fixation]):
        long_fixation_frames.append(eye_frame[np.where(eye_frame>=f)[0][0]:np.where(eye_frame<=go_frame[mask_long_fixation][i])[0][-1]])
    long_fixation_frames = np.array(long_fixation_frames, dtype=object)  # array of arrays
    long_fixation_eye_positions = []    
    for frames in long_fixation_frames:
        positions = []
        for fr in frames:
            pos = saccades["eye_pos"][np.where(eye_frame == fr)[0]] 
            if len(pos) > 0:
                positions.append(pos[0])
        positions = np.array(positions)
    # positions = positions - np.mean(positions, axis=0)  # center the positions around the mean
        long_fixation_eye_positions.append(np.array(positions))
    long_fixation_eye_positions = np.array(long_fixation_eye_positions, dtype=object)  # array of arrays
    # Plot the eye positions during short and long fixation trials
    #fig = plt.figure(figsize=(12, 5))
    short_fixation_all_positions = np.vstack(short_fixation_eye_positions)
    long_fixation_all_positions = np.vstack(long_fixation_eye_positions)

    ## Let's look at some randomly selected windows from the whole sessions for control
    control_window_eye_positions = []
    control_window_size = int(fixation_time_theshold * ttl_freq)  # in frames
    control_window_start = np.random.choice



    #----------------------------------------------------------------------------------------------------------
    ## This commented section is for side-by-side plots
    #----------------------------------------------------------------------------------------------------------
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # ax1.scatter(short_fixation_all_positions[:, 0], short_fixation_all_positions[:, 1], color='blue', alpha=0.3)
    # ax1.set_title(f'Eye Positions During Short Fixation Trials (<= {fixation_time_theshold}s)')
    # ax1.set_xlabel('X Position (deg)')
    # ax1.set_ylabel('Y Position (deg)')  
    # ax1.grid()
    # ax2.scatter(long_fixation_all_positions[:, 0], long_fixation_all_positions[:, 1], color='orange', alpha=0.3)
    # ax2.set_title(f'Eye Positions During Long Fixation Trials (> {fixation_time_theshold}s)')
    # ax2.set_xlabel('X Position (deg)')
    # ax2.set_ylabel('Y Position (deg)')  
    # ax2.grid()
    # plt.show()
    #----------------------------------------------------------------------------------------------------------
    # This section is for overlayed plots
    #----------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(saccades["eye_pos"][:, 0], saccades["eye_pos"][:, 1], color='red', alpha=0.1, label='All Eye Positions')
    plt.scatter(long_fixation_all_positions[:, 0], long_fixation_all_positions[:, 1], color='orange', alpha=0.3, label=f'Long Fixation (> {fixation_time_theshold}s)')
    plt.scatter(short_fixation_all_positions[:, 0], short_fixation_all_positions[:, 1], color='blue', alpha=0.3, label=f'Short Fixation (<= {fixation_time_theshold}s)')
    plt.xlabel('X Position (deg)')
    plt.ylabel('Y Position (deg)')
    plt.title('Eye Positions During Short and Long Fixation Trials')
    plt.legend()
    plt.grid()
    plt.show()

## Plot the moving average of the trial success rate over a sliding window of 20 trials
window_size = 10
trial_success = np.array(trial_success, dtype=int)  # Ensure it's an integer array
moving_avg_success = np.convolve(trial_success, np.ones(window_size)/window_size, mode='valid')
fig = plt.figure(figsize=(10, 5))
plt.plot(moving_avg_success, color='blue', label='Moving Average Success Rate')
plt.xlabel('Trial Index')
plt.ylabel('Success Rate (Moving Average)')
plt.title(f'Moving Average of Trial Success Rate (Window Size: {window_size})')
plt.axhline(y=0.5, color='red', linestyle='--', label='50% Success Rate')
plt.legend()
plt.grid()
plt.tight_layout()

## First plot the percentage of the correct, incorrect, and missed trials
# end_of_trial_frame, end_of_trial_ts, trial_stim_direction, trial_eye_movement_direction, trial_torsion_angle, trial_success
num_trials = len(end_of_trial_frame)
num_correct = np.sum(trial_success == 1)
num_missed = np.sum(trial_success == 0)
num_incorrect = np.sum(trial_success == -1)
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Correct Trials', 'Missed Trials', 'Incorrect Trials']
sizes = [num_correct, num_missed, num_incorrect]
# Normalize sizes to percentages
# Percentages to be written on the bar chart
percentages = [f"{size/num_trials*100:.1f}%" for size in sizes]
colors = ['green', 'orange', 'red']
bars = ax.bar(labels, sizes, color=colors)
# Add percentage labels on top of the bars
for bar, percentage in zip(bars, percentages):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, percentage, ha='center', va='bottom')
ax.set_ylabel('Number of Trials')
ax.set_title('Trial Outcomes')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# Save the figure
trial_outcome_fname = f"{session_name}_{eye_name}_Trial_Outcomes.png"   
fig.savefig(results_dir / trial_outcome_fname, dpi=300, bbox_inches='tight')



## Plot the comparison of saccade magnitude and direction between correct, incorrect, and missed trials
saccade_speeds_correct_all = []
saccade_speeds_correct_latency_all = []
saccade_speeds_incorrect_before_correct_all = []
saccade_latency_incorrect_before_correct_all = []
saccade_speeds_incorrect_all = []
saccade_speeds_incorrect_latency_all = []
saccade_speeds_correct_first_saccade_all = []
saccade_latency_correct_first_saccade_all = []
for i, f in enumerate(go_frame[:len(end_of_trial_frame)]):  # Loop through each go frame
    if trial_success[i] == 1: ## Correct trial
        # Find the number of saccade between go frame and end of the trial frame
        saccade_indices = np.where((saccades['saccade_frames_xy'] >=f) & (saccades['saccade_frames_xy'] <= end_of_trial_frame[i]))[0]
        if len(saccade_indices) > 0:
            saccade_speeds_correct = np.linalg.norm(saccades['eye_vel'][saccades['saccade_indices_xy'][saccade_indices]], axis=1)
            saccade_directions_correct = np.rad2deg(np.arctan2(saccades['eye_vel'][saccades['saccade_indices_xy'][saccade_indices], 1], saccades['eye_vel'][saccades['saccade_indices_xy'][saccade_indices], 0]))
            saccade_speeds_correct_all.append(saccade_speeds_correct[-1])
            saccade_speeds_correct_latency_all.append((saccades['saccade_frames_xy'][saccade_indices[-1]] - f)/60.0)  # latency of the last saccade in the trial
            if len(saccade_indices) > 1:
                saccade_speeds_incorrect_before_correct_all.append(saccade_speeds_correct[0])  # first incorrect saccade before the correct one
                saccade_latency_incorrect_before_correct_all.append((saccades['saccade_frames_xy'][saccade_indices[0]] - f)/60.0)  # latency of the first saccade in the trial
                #print(f"Trial {i}: Correct trial with {len(saccade_indices)} saccades, first saccade speed: {saccade_speeds_correct[0]:.2f} deg/frame")
                #trial_success[i] = 2  # Mark this trial as succeincorrect before correct
            else:
                saccade_speeds_correct_first_saccade_all.append(saccade_speeds_correct[0])  # first saccade in the trial
                saccade_latency_correct_first_saccade_all.append((saccades['saccade_frames_xy'][saccade_indices[0]] - f)/60.0)  # latency of the first saccade in the trial
    elif trial_success[i] == -1: ## Incorrect trial
        # Find the number of saccade between go frame and end of the trial frame
        saccade_indices = np.where((saccades['saccade_frames_xy'] >= f) & (saccades['saccade_frames_xy'] <= end_of_trial_frame[i]))[0]
        if len(saccade_indices) > 0:
            saccade_speeds_incorrect = np.linalg.norm(saccades['eye_vel'][saccades['saccade_indices_xy'][saccade_indices]], axis=1)
            saccade_directions_incorrect = np.rad2deg(np.arctan2(saccades['eye_vel'][saccades['saccade_indices_xy'][saccade_indices], 1], saccades['eye_vel'][saccades['saccade_indices_xy'][saccade_indices], 0]))
            saccade_speeds_incorrect_all.append(saccade_speeds_incorrect[0])
            saccade_speeds_incorrect_latency_all.append((saccades['saccade_frames_xy'][saccade_indices[0]] - f)/60)  # latency of the first saccade in the trial
# saccade_speeds_correct_all = np.array(saccade_speeds_correct_all)
# saccade_speeds_correct_latency_all = np.array(saccade_speeds_correct_latency_all)
# saccade_speeds_incorrect_before_correct_all = np.array(saccade_speeds_incorrect_before_correct_all)
# saccade_speeds_incorrect_all = np.array(saccade_speeds_incorrect_all)
# saccade_speeds_incorrect_latency_all = np.array(saccade_speeds_incorrect_latency_all)
# saccade_speeds_correct_first_saccade_all = np.array(saccade_speeds_correct_first_saccade_all)
# saccade_latency_correct_first_saccade_all = np.array(saccade_latency_correct_first_saccade_all)
print(f"Number of correct trials with saccades: {len(saccade_speeds_correct_all)}, Number of incorrect trials with saccades: {len(saccade_speeds_incorrect_all)}")
print(f"Number of correct trials with both incorrect and correct saccades: {len(saccade_speeds_incorrect_before_correct_all)}")
print(f"Number of correct trials with only one correct saccade: {len(saccade_speeds_correct_first_saccade_all)}")
# Plot the box plot for saccade speeds for correct, incorrect, incorrect before correct, and correct first saccade trials
fig, ax = plt.subplots(figsize=(10, 6))
data = [saccade_speeds_correct_all, saccade_speeds_incorrect_all, saccade_speeds_incorrect_before_correct_all, saccade_speeds_correct_first_saccade_all]
labels = ['Correct Trials', 'Incorrect Trials', 'Incorrect Before Correct', 'Correct First Saccade']
ax.boxplot(data, labels=labels, patch_artist=True,
           boxprops=dict(facecolor='lightgreen', color='green'),    
              medianprops=dict(color='red'))    
ax.set_ylabel('Saccade Speed (deg/frame)')
ax.set_title('Comparison of Saccade Speeds by Trial Outcome')
#ax.grid(axis='y', alpha='0.3')
plt.tight_layout()
plt.show()
# Plot the box plot for saccade latencies for correct, incorrect, and incorrect before correct  and correct first saccade trials
fig, ax = plt.subplots(figsize=(10, 6))
data = [saccade_speeds_correct_latency_all, saccade_speeds_incorrect_latency_all, saccade_latency_incorrect_before_correct_all, saccade_latency_correct_first_saccade_all]
labels = ['Correct Trials', 'Incorrect Trials', 'Incorrect Before Correct', 'Correct First Saccade']
ax.boxplot(data, labels=labels, patch_artist=True,  
              boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='red'))
ax.set_ylabel('Saccade Latency (s)')
ax.set_title('Comparison of Saccade Latencies by Trial Outcome')
#ax.grid(axis='y', alpha='0.3')
plt.tight_layout()
plt.show()

