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
    ## Plot the von mises kde for the angles
    from scipy.stats import vonmises
    kappa = 12
    theta_dense = np.linspace(0, 2 * np.pi, 200)
    kernels = np.array([vonmises.pdf(theta_dense, kappa, loc=a) for a in angle_2pi])

    density = kernels.sum(axis=0)

    density_scaled = density * np.max(counts) / np.max(density)
    # Wrap around for circular plot
    theta_closed = np.append(theta_dense, theta_dense[0])
    density_closed = np.append(density_scaled, density_scaled[0])
    ax_polar.plot(theta_closed, density_closed, color="r", lw=2)
    ## Probability of looking right vs left in the 35 degree window (right is 35 degree to 0 and 325 to 360, left is 145 to 215)
    right_prob = np.sum(counts[(bin_edges[:-1] >= 0) & (bin_edges[:-1] < np.radians(35))]) + np.sum(counts[(bin_edges[:-1] >= np.radians(325)) & (bin_edges[:-1] < 2 * np.pi)])
    left_prob = np.sum(counts[(bin_edges[:-1] >= np.radians(145)) & (bin_edges[:-1] < np.radians(215))])
    ## Print on the plot
    ax_polar.text(0, 0.5, f"Right: {right_prob:.2f}\nLeft: {left_prob:.2f}", ha="center", va="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='k'))

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
