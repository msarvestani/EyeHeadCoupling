from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_noncausal(signal: np.ndarray, fs: float, cutoff_freq: float = 1, order: int = 4) -> np.ndarray:
    """Apply a zero-phase low-pass Butterworth filter."""
    sos = butter(order, cutoff_freq / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, signal)


def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Interpolate NaN values in ``arr`` using linear interpolation."""
    nans = np.isnan(arr)
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


__all__ = ["butter_noncausal", "interpolate_nans"]
