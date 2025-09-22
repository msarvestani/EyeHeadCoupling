"""Convenience imports for the :mod:`eyehead` package."""

from utils.session_loader import SessionConfig
from .analysis import (
    SaccadeConfig,
    calibrate_eye_position,
    detect_saccades,
    organize_stims,
    sort_saccades,
    plot_fixation_intervals_by_trial,
    plot_eye_fixations_between_cue_and_go_by_trial,
    quantify_fixation_stability_vs_random,
)
from .filters import butter_noncausal, interpolate_nans
from .io import (
    SessionData,
    load_session_data,
    get_session_date_from_path,
    determine_camera_side,
    remove_parentheses_chars,
    clean_csv,
)
from .plotting import (
    rotation_matrix,
    vector_to_rgb,
    plot_angle_distribution,
    plot_linear_histogram,
)
from .ui import select_folder, select_file, choose_option

__all__ = [
    "SessionConfig",
    "SaccadeConfig",
    "calibrate_eye_position",
    "detect_saccades",
    "organize_stims",
    "sort_saccades",
    "plot_fixation_intervals_by_trial",
    "plot_eye_fixations_between_cue_and_go_by_trial",
    "quantify_fixation_stability_vs_random",
    "butter_noncausal",
    "interpolate_nans",
    "SessionData",
    "load_session_data",
    "get_session_date_from_path",
    "determine_camera_side",
    "remove_parentheses_chars",
    "clean_csv",
    "rotation_matrix",
    "vector_to_rgb",
    "plot_angle_distribution",
    "plot_linear_histogram",
    "select_folder",
    "select_file",
    "choose_option",
]
