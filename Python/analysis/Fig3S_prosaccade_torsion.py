"""Supplementary per-session analysis: is torsion specific to Up/Down stimuli?

Companion to ``prosaccade_session.py``, but scores the *torsional* saccade
stream instead of the translational (xy) one, and — unlike
``prosaccade_session.py`` — does not discard any trials by direction. The
question this script answers is whether torsional eye movements are
preferentially evoked by vertical (Up/Down) targets, or occur just as
readily for horizontal (Left/Right) ones. There is no xy scoring anywhere
in this file.

Congruency rule (hardcoded, not read from the manifest)
---------------------------------------------------------
- Up target: a "congruent" torsional saccade is counterclockwise (CCW).
- Down target: a "congruent" torsional saccade is clockwise (CW).
- Left/Right targets have no natural torsion target, so congruency is
  undefined (``NaN``) for them — instead we report the raw rate and sign of
  any torsional saccade on those trials, as a no-target control condition
  to compare Up/Down's congruent rate against.
- A torsional saccade only counts toward congruency if its magnitude is at
  least ``TORSION_CONGRUENCE_THRESHOLD_DEG`` (on top of already having
  cleared the detection threshold, ``saccade_config.saccade_threshold_torsion``,
  during saccade detection itself).
- Scoring is still multi-attempt: an incongruent torsional saccade doesn't
  end the search — keep looking for the first congruent one in the trial's
  window, and if none was congruent, fall back to the last attempt
  (congruent=False). This is independent of the session's manifest
  ``scoring_mode``, which describes how the *xy* stream ended the actual
  behavioral trial, not how torsion is scored here.

This is independent of the reward contingency: the animal is not rewarded
based on torsion, so there is no manifest-configured torsion "reward_angle"
to check against — the congruency rule here is a fixed scientific criterion
for this analysis, not a description of what the rig scores.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import warnings

# Put the repo's "Python" folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session_or_path

from eyehead import (
    SaccadeConfig,
    SessionData,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    organize_stims,
    get_session_date_from_path,
)
from prosaccade_session import wilson_ci


# --- Hardcoded torsion congruency convention --------------------------------
# Sign of d(torsion_angle)/dframe that corresponds to counterclockwise (CCW)
# rotation under this rig's torsion-angle convention. CONFIRM against real
# data before trusting congruency labels — e.g. plot dtheta for a few known
# Up trials and check the sign matches your own visual read of CW/CCW.
CCW_SIGN = 1.0

UP_TARGET_SIGN = CCW_SIGN        # Up target -> congruent torsion is CCW
DOWN_TARGET_SIGN = -CCW_SIGN     # Down target -> congruent torsion is CW

# Minimum |dtheta| (deg) for a detected torsional saccade to count toward
# congruency, in addition to already having cleared saccade_threshold_torsion
# during detection.
TORSION_CONGRUENCE_THRESHOLD_DEG = 1.0

# Keep SVG text as real, editable text (not outlined paths) so exported
# figures can still be edited as text in Illustrator.
plt.rcParams["svg.fonttype"] = "none"


def _find_scored_torsion_saccade(rel_valid, idx_valid, dtheta, target_sign):
    """Pick this trial's scored torsional saccade, multi-attempt style.

    An incongruent torsional saccade doesn't end the search: this returns
    the first saccade in the window whose sign matches ``target_sign`` and
    whose magnitude is at least TORSION_CONGRUENCE_THRESHOLD_DEG. If no
    attempt in the window was congruent, it returns the *last* attempt
    instead, with congruent=False — its latency reflects how long the
    animal kept moving before the window closed.

    ``target_sign`` is ``None`` for Left/Right trials, which have no
    natural torsion target — congruency is then always False, so this
    always falls through to "last attempt, congruent=False". Callers
    should treat that as "not applicable" (relabel to NaN) rather than
    "wrong direction" for trials with no target.

    Returns
    -------
    idx : int
        Index into ``dtheta`` (and the theta-stream eye-position arrays)
        for the chosen saccade.
    latency : float
        Its time (s) since target onset.
    congruent : bool
        Whether it matched ``target_sign`` at >= threshold magnitude.
        Always False when ``target_sign`` is None.
    sign : float
        Sign of dtheta at the chosen saccade (+1 = CCW, -1 = CW), so
        Left/Right trials (no target) can still be reported by raw
        direction.
    """
    order = np.argsort(rel_valid)
    rel_sorted = rel_valid[order]
    idx_sorted = idx_valid[order]

    def _is_congruent(idx):
        if target_sign is None:
            return False
        return (
            np.sign(dtheta[idx]) == target_sign
            and abs(dtheta[idx]) >= TORSION_CONGRUENCE_THRESHOLD_DEG
        )

    for t, idx in zip(rel_sorted, idx_sorted):
        if _is_congruent(idx):
            return idx, float(t), True, float(np.sign(dtheta[idx]))

    idx_last = idx_sorted[-1]
    return idx_last, float(rel_sorted[-1]), False, float(np.sign(dtheta[idx_last]))

def find_torsion_saccade_per_trial(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    max_latency: float,
) -> Dict[str, np.ndarray]:
    """Per-trial scored torsional saccade, for every trial regardless of direction.

    Unlike the xy-only helpers in ``prosaccade_session.py``, no trial is
    dropped and no direction is masked out: every trial in
    ``data.go_frame`` gets one entry, so Left/Right and Up/Down trials can
    be compared on an equal footing (same denominators) in the
    Up/Down-specificity figure this script builds.

    Parameters
    ----------
    max_latency : float
        Trials whose scored torsional saccade would land later than this
        (seconds after target onset) are treated the same as "no torsional
        saccade" — same convention as ``max_trial_time`` in
        ``prosaccade_session.py``.

    Returns
    -------
    dict of ndarray, one entry per trial (in ``data.go_frame`` order):
        direction : "Left"/"Right"/"Up"/"Down" (this trial's go direction)
        target_sign : the hardcoded torsion target for that direction
            (UP_TARGET_SIGN / DOWN_TARGET_SIGN), NaN for Left/Right
        has_torsion : whether any torsional saccade was detected in the
            trial's window at all
        latency : scored saccade's latency (s); NaN where has_torsion is
            False
        sign : scored saccade's dtheta sign (+1 CCW, -1 CW); NaN where
            has_torsion is False
        congruent : 1.0/0.0 from :func:`_find_scored_torsion_saccade`;
            NaN wherever there's no defined target (Left/Right) or no
            torsional saccade was detected at all — "not applicable", not
            "wrong direction".
    """
    ttl_freq = config.ttl_freq
    go_frame = np.asarray(data.go_frame)
    go_dir_x = np.asarray(data.go_direction_x, dtype=np.float64)
    go_dir_y = np.asarray(data.go_direction_y, dtype=np.float64)
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq
    end_frame = np.asarray(end_frame)

    n_trials = len(go_frame)
    direction = np.full(n_trials, "", dtype=object)
    target_sign = np.full(n_trials, np.nan)
    direction[go_dir_x < 0] = "Left"
    direction[go_dir_x > 0] = "Right"
    direction[go_dir_y < 0] = "Down"
    direction[go_dir_y > 0] = "Up"
    target_sign[direction == "Up"] = UP_TARGET_SIGN
    target_sign[direction == "Down"] = DOWN_TARGET_SIGN

    saccade_frames = saccades["saccade_frames_theta"]
    saccade_indices = saccades["saccade_indices_theta"]
    dtheta = saccades["eye_vel"][:, 2]

    has_torsion = np.zeros(n_trials, dtype=bool)
    latency = np.full(n_trials, np.nan)
    sign = np.full(n_trials, np.nan)
    congruent = np.full(n_trials, np.nan)

    for i, (f, end_f, tsign) in enumerate(zip(go_frame, end_frame, target_sign)):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            continue
        target = tsign if np.isfinite(tsign) else None
        _, lat, is_congruent, sgn = _find_scored_torsion_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dtheta, target,
        )
        has_torsion[i] = True
        latency[i] = lat
        sign[i] = sgn
        congruent[i] = np.nan if target is None else float(is_congruent)

    return {
        "direction": direction,
        "target_sign": target_sign,
        "has_torsion": has_torsion,
        "latency": latency,
        "sign": sign,
        "congruent": congruent,
    }

def summarize_torsion_by_direction(per_trial: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-trial torsion scoring into one summary row per stimulus direction.

    For every direction (Left/Right/Up/Down), reports the raw CCW-rate and
    CW-rate: fraction of *all* trials in that direction (not just those
    with a detected torsional saccade) whose scored torsional saccade was
    CCW / CW, each with a Wilson 95% CI.

    For Up and Down specifically, also reports ``congruent_rate`` — the
    fraction that matched that direction's hardcoded torsion target (CCW
    for Up, CW for Down) at >= TORSION_CONGRUENCE_THRESHOLD_DEG, straight
    from :func:`_find_scored_torsion_saccade`'s ``congruent`` flag. By
    construction this equals the CCW-rate for Up and the CW-rate for Down,
    but is included explicitly so callers don't have to know that.

    Left/Right have no ``congruent_rate`` entry — there's no target to be
    congruent with. Their CCW/CW rates are the no-target control numbers to
    compare Up's/Down's congruent_rate against: if torsion is Up/Down-
    specific, Left/Right's rates should sit well below it.

    Returns
    -------
    dict keyed by direction label (only directions actually present in
    ``per_trial`` are included), each mapping to a dict with ``n_trials``,
    ``n_with_torsion``, ``ccw_rate``/``ccw_ci``, ``cw_rate``/``cw_ci``, and
    (Up/Down only) ``congruent_rate``/``congruent_ci``.
    """
    direction = per_trial["direction"]
    has_torsion = per_trial["has_torsion"]
    sign = per_trial["sign"]
    congruent = per_trial["congruent"]

    summary: Dict[str, Dict[str, float]] = {}
    for label in ["Left", "Right", "Up", "Down"]:
        sel = direction == label
        n = int(np.sum(sel))
        if n == 0:
            continue

        is_ccw = sel & has_torsion & (sign == CCW_SIGN)
        is_cw = sel & has_torsion & (sign == -CCW_SIGN)

        n_ccw = int(np.sum(is_ccw))
        n_cw = int(np.sum(is_cw))
        ccw_rate = n_ccw / n
        cw_rate = n_cw / n
        ccw_ci = wilson_ci(n_ccw, n)
        cw_ci = wilson_ci(n_cw, n)

        row = {
            "n_trials": n,
            "n_with_torsion": int(np.sum(sel & has_torsion)),
            "ccw_rate": ccw_rate,
            "ccw_ci": ccw_ci,
            "cw_rate": cw_rate,
            "cw_ci": cw_ci,
        }

        if label in ("Up", "Down"):
            n_congruent = int(np.nansum(congruent[sel]))
            row["congruent_rate"] = n_congruent / n
            row["congruent_ci"] = wilson_ci(n_congruent, n)

        summary[label] = row

    return summary


def _draw_torsion_strip(ax, per_trial: Dict[str, np.ndarray], order: list) -> None:
    """Draw the per-trial torsion-presence strip onto ``ax`` — one row per
    direction, a dot per trial with a detected torsional saccade (blue =
    CCW, orange = CW), nothing for trials without one."""
    direction = per_trial["direction"]
    has_torsion = per_trial["has_torsion"]
    sign = per_trial["sign"]

    for row, label in enumerate(order):
        sel = np.where(direction == label)[0]
        n = len(sel)
        n_with = int(np.sum(has_torsion[sel]))
        trial_idx = np.arange(n)

        row_has = has_torsion[sel]
        row_sign = sign[sel]

        is_ccw = row_has & (row_sign == CCW_SIGN)
        is_cw = row_has & (row_sign == -CCW_SIGN)

        ax.scatter(trial_idx[is_ccw], np.full(np.sum(is_ccw), row), color="tab:blue", s=25)
        ax.scatter(trial_idx[is_cw], np.full(np.sum(is_cw), row), color="tab:orange", s=25)

        ax.text(-1, row, f"{label} ({n_with}/{n})", ha="right", va="center", fontsize=9)

    ax.set_yticks([])
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("Trial index (within direction)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="tab:blue", linestyle="none", markersize=7, label="CCW"),
        Line2D([0], [0], marker="o", color="tab:orange", linestyle="none", markersize=7, label="CW"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)


def _draw_torsion_rate_bars(ax, summary: Dict[str, Dict[str, float]], order: list) -> None:
    """Draw grouped CCW/CW bars (fraction of *all* trials) onto ``ax``.

    Bars for a direction don't have to sum to 1 — the gap is the fraction
    of trials with no detected torsional saccade at all, which is exactly
    the point: this shows both (1) how much more often torsion occurs on
    Up/Down trials than Left/Right, and (2) that when it does occur on Up
    trials specifically, it's overwhelmingly CCW.
    """
    x = np.arange(len(order))
    width = 0.35

    ccw_rate = [summary[d]["ccw_rate"] for d in order]
    cw_rate = [summary[d]["cw_rate"] for d in order]
    ccw_err = [
        [summary[d]["ccw_rate"] - summary[d]["ccw_ci"][0] for d in order],
        [summary[d]["ccw_ci"][1] - summary[d]["ccw_rate"] for d in order],
    ]
    cw_err = [
        [summary[d]["cw_rate"] - summary[d]["cw_ci"][0] for d in order],
        [summary[d]["cw_ci"][1] - summary[d]["cw_rate"] for d in order],
    ]

    ax.bar(x - width / 2, ccw_rate, width, yerr=ccw_err, capsize=4, color="tab:blue", label="CCW")
    ax.bar(x + width / 2, cw_rate, width, yerr=cw_err, capsize=4, color="tab:orange", label="CW")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of all trials")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.legend(loc="upper right", fontsize=8)


def plot_torsion_presence_strip(
    per_trial: Dict[str, np.ndarray],
    title: str,
    save_path,
    show_plots: bool = True,
) -> plt.Figure:
    """Per-trial torsion-presence strip plot, standalone.

    One row per stimulus direction, a dot per trial with a detected
    torsional saccade (blue = CCW, orange = CW). See
    :func:`_draw_torsion_strip` for the drawing logic.
    """
    order = [d for d in ["Left", "Right", "Down", "Up"] if np.any(per_trial["direction"] == d)]

    fig, ax = plt.subplots(figsize=(10, 1.2 * len(order) + 1))
    _draw_torsion_strip(ax, per_trial, order)
    ax.set_title(title, fontsize=11, wrap=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(save_path).with_suffix(".svg"), bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def plot_torsion_rate_and_latency(
    per_trial: Dict[str, np.ndarray],
    summary: Dict[str, Dict[str, float]],
    title: str,
    save_path,
    bins: int = 10,
    show_plots: bool = True,
) -> plt.Figure:
    """Combined figure: CCW/CW rate bars (top) + per-direction latency histograms (bottom).

    Top panel is the grouped CCW/CW rate bar chart from
    :func:`summarize_torsion_by_direction` (fraction of *all* trials, per
    direction). Bottom row is one latency-histogram panel per direction,
    CCW vs. CW overlaid, for trials with a detected torsional saccade. The
    presence strip plot is intentionally not part of this figure — see
    :func:`plot_torsion_presence_strip`.
    """
    order = [d for d in ["Left", "Right", "Down", "Up"] if d in summary]
    n_dirs = len(order)

    direction = per_trial["direction"]
    has_torsion = per_trial["has_torsion"]
    sign = per_trial["sign"]
    latency = per_trial["latency"]

    fig = plt.figure(figsize=(1.5 * n_dirs, 5))
    gs = fig.add_gridspec(2, n_dirs, height_ratios=[3, 4], hspace=0.4, wspace=0.3)

    ax_bars = fig.add_subplot(gs[0, :])
    _draw_torsion_rate_bars(ax_bars, summary, order)

    all_latencies = latency[has_torsion]
    if all_latencies.size:
        bin_edges = np.linspace(np.nanmin(all_latencies), np.nanmax(all_latencies), bins + 1)
    else:
        bin_edges = bins

    axes_latency = []
    for i in range(n_dirs):
        ax = fig.add_subplot(gs[1, i], sharey=axes_latency[0] if axes_latency else None)
        axes_latency.append(ax)

    max_count = 0
    for ax, label in zip(axes_latency, order):
        sel = direction == label
        ccw_lat = latency[sel & has_torsion & (sign == CCW_SIGN)]
        cw_lat = latency[sel & has_torsion & (sign == -CCW_SIGN)]

        ccw_counts, _, _ = ax.hist(ccw_lat, bins=bin_edges, alpha=0.6, color="tab:blue", label=f"CCW (n={ccw_lat.size})")
        cw_counts, _, _ = ax.hist(cw_lat, bins=bin_edges, alpha=0.6, color="tab:orange", label=f"CW (n={cw_lat.size})")
        max_count = max(max_count, ccw_counts.max(), cw_counts.max())
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Latency (s)")
        #ax.legend(fontsize=8)

    axes_latency[0].set_ylabel("Trial count")
    axes_latency[0].set_ylim(0, max(10, 10 * np.ceil(max_count / 10)))
    axes_latency[0].yaxis.set_major_locator(MultipleLocator(10))
    for ax in axes_latency[1:]:
        ax.tick_params(labelleft=False)

    valid_frac_txt = ", ".join(
        f"{d}: {summary[d]['n_with_torsion']}/{summary[d]['n_trials']} "
        f"({summary[d]['n_with_torsion']/summary[d]['n_trials']:.0%}) valid"
        for d in order
    )
    fig.suptitle(f"{title}\n{valid_frac_txt}", fontsize=10, wrap=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(save_path).with_suffix(".svg"), bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def plot_saccade_trace_snippet(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    title: str,
    save_path,
    window_s: float = 20.0,
    start_time: Optional[float] = 340,
    show_plots: bool = True,
) -> plt.Figure:
    """Zoomed-in eye-position and velocity traces, with detections marked.

    Four rows sharing a time axis:
      1. x/y eye position (deg), overlaid
      2. xy saccade speed (deg/s)
      3. torsion (theta) eye position (deg)
      4. torsional saccade speed (deg/s)

    so each velocity panel sits directly under the position trace it was
    derived from — the position panels show the step that produces the
    speed spike below them. A red dot marks every detected saccade (xy or
    torsional) on both its position and speed panel.

    Parameters
    ----------
    start_time : float, optional
        Start of the window, in seconds from the start of the eye-tracking
        recording. Defaults to 1s before the first detected torsional
        saccade (or, if there is none, the first xy saccade), so the
        window reliably contains at least one detection instead of a
        possibly-empty arbitrary window.
    """
    ttl_freq = config.ttl_freq
    eye_frame = np.asarray(data.eye_frame)
    time_s = (eye_frame - eye_frame[0]) / ttl_freq

    eye_pos = saccades["eye_pos"]
    x_pos, y_pos, theta_pos = eye_pos[:, 0], eye_pos[:, 1], eye_pos[:, 2]

    dx = saccades["eye_vel"][:, 0] * ttl_freq
    dy = saccades["eye_vel"][:, 1] * ttl_freq
    xy_speed = np.hypot(dx, dy)
    torsion_speed = np.abs(saccades["eye_vel"][:, 2]) * ttl_freq

    saccade_indices_xy = np.asarray(saccades["saccade_indices_xy"], dtype=int)
    saccade_indices_theta = np.asarray(saccades["saccade_indices_theta"], dtype=int)

    if start_time is None:
        if saccade_indices_theta.size:
            start_time = max(0.0, time_s[saccade_indices_theta[0]] - 1.0)
        elif saccade_indices_xy.size:
            start_time = max(0.0, time_s[saccade_indices_xy[0]] - 1.0)
        else:
            start_time = 0.0

    end_time = start_time + window_s
    in_window = (time_s >= start_time) & (time_s < end_time)

    def _in_window(idx):
        return idx[(time_s[idx] >= start_time) & (time_s[idx] < end_time)]

    idx_xy_in_window = _in_window(saccade_indices_xy)
    idx_theta_in_window = _in_window(saccade_indices_theta)

    fig, (ax_pos_xy, ax_speed_xy, ax_pos_theta, ax_speed_theta) = plt.subplots(
        4, 1, figsize=(8, 11), sharex=True,
    )

    ax_pos_xy.plot(time_s[in_window], x_pos[in_window], color="tab:blue", lw=1, label="x")
    ax_pos_xy.plot(time_s[in_window], y_pos[in_window], color="tab:orange", lw=1, label="y")
    ax_pos_xy.scatter(time_s[idx_xy_in_window], x_pos[idx_xy_in_window], color="red", s=20, zorder=3)
    ax_pos_xy.scatter(time_s[idx_xy_in_window], y_pos[idx_xy_in_window], color="red", s=20, zorder=3,
                       label="Detected xy saccade")
    ax_pos_xy.set_ylabel("Eye position (deg)")
    ax_pos_xy.legend(fontsize=8, loc="upper right")

    ax_speed_xy.plot(time_s[in_window], xy_speed[in_window], color="0.3", lw=1)
    ax_speed_xy.scatter(time_s[idx_xy_in_window], xy_speed[idx_xy_in_window], color="red", s=25, zorder=3,
                         label="Detected xy saccade")
    ax_speed_xy.set_ylabel("Speed (deg/s)")
    ax_speed_xy.legend(fontsize=8, loc="upper right")

    ax_pos_theta.plot(time_s[in_window], theta_pos[in_window], color="tab:purple", lw=1, label="theta")
    ax_pos_theta.scatter(time_s[idx_theta_in_window], theta_pos[idx_theta_in_window], color="red", s=20, zorder=3,
                          label="Detected torsional saccade")
    ax_pos_theta.set_ylabel("Torsion position (deg)")
    ax_pos_theta.legend(fontsize=8, loc="upper right")

    ax_speed_theta.plot(time_s[in_window], torsion_speed[in_window], color="0.3", lw=1)
    ax_speed_theta.scatter(time_s[idx_theta_in_window], torsion_speed[idx_theta_in_window], color="red", s=25, zorder=3,
                            label="Detected torsional saccade")
    ax_speed_theta.set_xlabel("Time (s)")
    ax_speed_theta.set_ylabel("Torsional velocity (deg/s)")
    ax_speed_theta.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{title}\n{window_s:g}s eye-movement trace with saccade detections", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(save_path).with_suffix(".svg"), bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig

def main(session_id: str, show_plots: bool = True) -> Dict[str, object]:
    """Run the torsion-specificity pipeline for ``session_id``.

    Loads the session, detects saccades (xy is detected as a side effect of
    ``detect_saccades`` but never scored here — only the torsional stream
    is used), scores every trial's torsional saccade regardless of
    stimulus direction (:func:`find_torsion_saccade_per_trial`), and draws
    the per-trial presence/direction strip plot
    (:func:`plot_torsion_presence_by_direction`). Also computes (and
    prints, but does not yet plot) the per-direction rate summary
    (:func:`summarize_torsion_by_direction`), so you can look at the
    numbers alongside the strip plot before deciding on a final rate
    figure.
    """
    config = load_session_or_path(session_id)
    session_id = config.session_id
    config.results_dir = config.results_dir / "torsion"
    config.results_dir.mkdir(parents=True, exist_ok=True)

    date_str = config.params.get("date")
    if not date_str and config.folder_path is not None:
        try:
            date_str = get_session_date_from_path(str(config.folder_path)).strftime("%Y-%m-%d")
        except Exception:
            date_str = ""

    data = load_session_data(config)
    eye_pos_cal = calibrate_eye_position(data, config)
    saccade_cfg = SaccadeConfig(**config.params["saccade_config"])

    saccades, fig_saccades, _ = detect_saccades(
        eye_pos_cal, data.eye_frame, saccade_cfg, config, data=data, plot=True,
    )
    # if fig_saccades is not None:
    #     plt.close(fig_saccades)

    n_torsion_saccades = len(saccades.get("saccade_indices_theta", []))
    print(f"Detected {n_torsion_saccades} torsional saccades")

    reward_contingency = config.params.get("reward_contingency") or {}
    reward_window = reward_contingency.get("reward_window")
    if reward_window is None:
        raise ValueError(
            "No reward_window configured in reward_contingency for this session; "
            "add one to session_manifest.yml (global default or per-session override)."
        )
    max_trial_time = float(reward_window)

    per_trial = find_torsion_saccade_per_trial(
        data, saccades, config, max_latency=max_trial_time,
    )

    summary = summarize_torsion_by_direction(per_trial)
    session_title = f"{config.animal_name or ''} {config.session_name}".strip()
    print(f"\nTorsion summary — {session_title}")
    for label, row in summary.items():
        extra = f", congruent={row['congruent_rate']:.0%}" if "congruent_rate" in row else ""
        print(
            f"  {label}: {row['n_with_torsion']}/{row['n_trials']} trials with torsion "
            f"(CCW={row['ccw_rate']:.0%}, CW={row['cw_rate']:.0%}{extra})"
        )

    plot_torsion_presence_strip(
        per_trial,
        title=session_title,
        save_path=config.results_dir / f"{config.session_name}_torsion_strip.png",
        show_plots=show_plots,
    )

    plot_torsion_rate_and_latency(
        per_trial, summary,
        title=session_title,
        save_path=config.results_dir / f"{config.session_name}_torsion_rate_latency.png",
        show_plots=show_plots,
    )

    plot_saccade_trace_snippet(
        data, saccades, config,
        title=session_title,
        save_path=config.results_dir / f"{config.session_name}_saccade_trace_snippet.png",
        show_plots=show_plots,
    )

    return {
        "session_id": session_id,
        "session_date": date_str,
        "per_trial": per_trial,
        "summary": summary,
        "reward_window": max_trial_time,
    }


# Usage: python Python/analysis/prosaccade_torsion.py SESSION_ID_OR_PATH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse torsional saccades across all stimulus directions (Left/Right/Up/Down)"
    )
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml, or a direct path to a session folder")
    args = parser.parse_args()
    main(args.session_id)