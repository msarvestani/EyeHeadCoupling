"""Plotting utilities with repository-wide style defaults.

This module loads matplotlib style settings from ``Python/style.mplstyle`` so
that all downstream figures share a consistent appearance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Apply repository style settings
_STYLE_PATH = Path(__file__).resolve().parent.parent / "style.mplstyle"
plt.style.use(_STYLE_PATH)


def rotation_matrix(angle_rad: float) -> np.ndarray:
    """Return a 2D rotation matrix for ``angle_rad`` radians."""
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                     [np.sin(angle_rad), np.cos(angle_rad)]])


def vector_to_rgb(angle: float) -> tuple[float, float, float]:
    """Map a vector angle to an RGB colour.

    The output colour encodes direction only; saturation and value are kept
    constant, so the magnitude of the vector does not influence the colour
    intensity.
    """
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return matplotlib.colors.hsv_to_rgb((angle / (2 * np.pi), 1.0, 1.0))


def plot_angle_distribution(angle: np.ndarray, ax_polar: plt.Axes, num_bins: int = 18) -> None:
    """Plot a normalized polar histogram of ``angle`` values."""
    angle_2pi = np.where(angle < 0, angle + 2 * np.pi, angle)
    counts, bin_edges = np.histogram(angle_2pi, bins=num_bins, range=(0, 2 * np.pi))
    counts = counts / np.size(angle_2pi)
    width = np.diff(bin_edges)
    ax_polar.bar(bin_edges[:-1], counts, width=width, align="edge", color="b", alpha=0.5, edgecolor="k")
    ax_polar.set_title("Normalized angle distribution")
    ax_polar.set_yticklabels([])
    ax_polar.yaxis.grid(False)


def plot_linear_histogram(angles: np.ndarray, ax: plt.Axes, num_bins: int = 18) -> None:
    """Plot a normalized Cartesian histogram of ``angles`` in degrees."""
    ang_deg = np.degrees(angles)
    ang_deg = np.mod(ang_deg, 360)
    counts, bins = np.histogram(ang_deg, bins=num_bins, range=(0, 360))
    counts = counts / ang_deg.size
    ax.bar(bins[:-1], counts, width=np.diff(bins), color="b", alpha=0.5, edgecolor="k")


__all__ = [
    "rotation_matrix",
    "vector_to_rgb",
    "plot_angle_distribution",
    "plot_linear_histogram",
]
