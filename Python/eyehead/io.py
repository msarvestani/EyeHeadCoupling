from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np

from utils.session_loader import SessionConfig


def get_session_date_from_path(path: str) -> datetime:
    """Extract the recording date from a session path."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", path)
    if match:
        return datetime.strptime(match.group(), "%Y-%m-%d")
    raise ValueError("No valid date (YYYY-MM-DD) found in path")


def determine_camera_side(path: str, cutoff_date_str: str = "2025-06-30") -> str:
    """Infer camera side ("L" or "R") from the session date."""
    session_date = get_session_date_from_path(path)
    cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")
    return "L" if session_date >= cutoff_date else "R"


def remove_parentheses_chars(line: str) -> str:
    """Remove parentheses and boolean strings from a line of text."""
    return line.replace("(", "").replace(")", "").replace("True", "1").replace("False", "0")


def clean_csv(filename: str) -> StringIO:
    """Return a file-like object with parentheses removed."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = [remove_parentheses_chars(line) for line in f]
    return StringIO("".join(lines))


@dataclass
class SessionData:
    """Container for arrays loaded from a Bonsai session."""

    camera: Optional[np.ndarray] = None
    go: Optional[np.ndarray] = None
    ellipse_center_xy: Optional[np.ndarray] = None
    origin_of_eye_coordinate: Optional[np.ndarray] = None
    torsion: Optional[np.ndarray] = None
    vdaxis: Optional[np.ndarray] = None
    imu: Optional[np.ndarray] = None
    end_of_trial: Optional[np.ndarray] = None
    cue: Optional[np.ndarray] = None

    bonsai_frame: Optional[np.ndarray] = None
    bonsai_time: Optional[np.ndarray] = None

    go_frame: Optional[np.ndarray] = None
    go_time: Optional[np.ndarray] = None
    go_direction_x: Optional[np.ndarray] = None
    go_direction_y: Optional[np.ndarray] = None
    go_direction: Optional[np.ndarray] = None

    eye_frame: Optional[np.ndarray] = None
    eye_timestamp: Optional[np.ndarray] = None
    eye_x: Optional[np.ndarray] = None
    eye_y: Optional[np.ndarray] = None

    origin_frame: Optional[np.ndarray] = None
    o_ts: Optional[np.ndarray] = None
    l_x: Optional[np.ndarray] = None
    l_y: Optional[np.ndarray] = None
    r_x: Optional[np.ndarray] = None
    r_y: Optional[np.ndarray] = None

    torsion_frame: Optional[np.ndarray] = None
    torsion_ts: Optional[np.ndarray] = None
    torsion_angle: Optional[np.ndarray] = None

    vd_frame: Optional[np.ndarray] = None
    vd_ts: Optional[np.ndarray] = None
    vd_lx: Optional[np.ndarray] = None
    vd_ly: Optional[np.ndarray] = None
    vd_rx: Optional[np.ndarray] = None
    vd_ry: Optional[np.ndarray] = None

    imu_time: Optional[np.ndarray] = None
    a_x: Optional[np.ndarray] = None
    a_y: Optional[np.ndarray] = None
    a_z: Optional[np.ndarray] = None
    g_x: Optional[np.ndarray] = None
    g_y: Optional[np.ndarray] = None
    g_z: Optional[np.ndarray] = None
    m_x: Optional[np.ndarray] = None
    m_y: Optional[np.ndarray] = None
    m_z: Optional[np.ndarray] = None

    end_of_trial_frame: Optional[np.ndarray] = None
    end_of_trial_ts: Optional[np.ndarray] = None
    trial_stim_direction: Optional[np.ndarray] = None
    trial_eye_movement_direction: Optional[np.ndarray] = None
    trial_torsion_angle: Optional[np.ndarray] = None
    trial_success: Optional[np.ndarray] = None

    cue_frame: Optional[np.ndarray] = None
    cue_time: Optional[np.ndarray] = None
    cue_direction: Optional[np.ndarray] = None


def load_session_data(config: SessionConfig) -> SessionData:
    """Load all Bonsai-generated CSV files for a session."""
    folder = Path(config.folder_path)
    data = SessionData()

    def _find_file(name: str, per_eye: bool) -> Optional[Path]:
        animal = (config.animal_id or "").lower()
        side = (config.camera_side or "").lower() if per_eye else ""
        for p in folder.glob("*.csv"):
            fname = p.name.lower()
            if animal and not fname.startswith(animal):
                continue
            if name.lower() not in fname:
                continue
            if side and f"_{side}" not in fname:
                continue
            return p
        return None

    def _load_csv(name: str, *, required: bool = False, per_eye: bool = False) -> Optional[np.ndarray]:
        file_path = _find_file(name, per_eye)
        if file_path is None:
            if required:
                raise FileNotFoundError(f"Required file matching '{name}' not found in {folder}")
            return None
        cleaned = clean_csv(str(file_path))
        return np.genfromtxt(cleaned, delimiter=",", skip_header=1)

    data.camera = _load_csv("camera")
    data.go = _load_csv("go")

    if data.go is not None:
        data.go_frame = data.go[:, 0].astype(int)
        data.go_time = data.go[:, 1]
        data.go_direction_x = data.go[:, 2]
        data.go_direction_y = data.go[:, 3]
        # Optional: combined direction if present
        if data.go.shape[1] > 4:
            data.go_direction = data.go[:, 4]

    data.ellipse_center_xy = _load_csv("ellipse_center_xy", required=True, per_eye=True)
    if data.ellipse_center_xy is not None:
        data.eye_frame = data.ellipse_center_xy[:, 0].astype(int)
        data.eye_timestamp = data.ellipse_center_xy[:, 1]
        data.eye_x = data.ellipse_center_xy[:, 2]
        data.eye_y = data.ellipse_center_xy[:, 3]

    data.origin_of_eye_coordinate = _load_csv(
        "origin_of_eyecoordinate", required=True, per_eye=True
    )
    data.torsion = _load_csv("torsion", per_eye=True)
    data.vdaxis = _load_csv("vdaxis", per_eye=True)
    data.imu = _load_csv("imu")
    data.end_of_trial = _load_csv("end_of_trial")
    data.cue = _load_csv("cue")

    return data


__all__ = [
    "SessionData",
    "load_session_data",
    "get_session_date_from_path",
    "determine_camera_side",
    "remove_parentheses_chars",
    "clean_csv",
]
