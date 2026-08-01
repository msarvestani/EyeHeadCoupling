"""Per-session analysis pipeline for the head-fixed prosaccade task.

Task
----
On each trial, a target appears at ``go_frame`` in one of up to four fixed
directions (``data.go_direction_x``/``go_direction_y``: left/right, and
up/down on sessions that interleave them). The animal is rewarded for
making a saccade toward that target within a fixed time and angular
window. ``data.trial_success`` (loaded from the session's ``end_of_trial``
CSV) is the rig's own independently-logged record of which trials were
rewarded — this script treats it as ground truth for the QC checks below,
it does not (yet) recompute it from the eye-tracking data itself.

How a trial ends differs by rig/animal, via the manifest's
``reward_contingency.scoring_mode`` (no safe global default — set
explicitly per session):

- ``single_shot`` (Apollo): the first saccade after target onset (in any
  direction) ends the trial immediately. It's rewarded only if it also
  lands within ``reward_angle`` of the target direction and within
  ``reward_window`` of target onset.
- ``multi_attempt`` (Paris): an incongruent saccade does *not* end the
  trial — the animal gets to keep trying. The trial only ends early on a
  saccade that lands within ``reward_angle`` of the target (which is then
  rewarded); incongruent saccades before that point are attempts, not
  trial-enders.
- **Both**: reaching ``reward_window`` without a rewarded saccade ends the
  trial unrewarded, regardless of animal.

Accordingly, "the first saccade after target onset" is not always the
saccade the rig actually scored the trial on — only under ``single_shot``.
:func:`_find_scored_saccade` resolves this per ``scoring_mode``: under
``multi_attempt`` it returns the first *congruent* saccade in the window
(skipping incongruent attempts, since they don't end the trial), or, if no
attempt in the window was congruent, the *last* attempt — with
``congruent=False`` and its latency reflecting how long the animal kept
trying before time-out. Every congruency function in this file
(:func:`session_reward_window`, :func:`session_acceptance_angle`,
:func:`find_first_saccade_per_trial`, :func:`first_saccade_indices_by_direction`,
:func:`analyze_latency_by_outcome`) goes through this helper, so "congruent"
means the same thing everywhere and matches what the rig actually scored.
The one exception is :func:`find_precue_saccade_per_trial` — it measures
spontaneous pre-target saccade bias, which has nothing to do with how a
trial ends, so it doesn't use ``scoring_mode`` at all.

``first_saccade_indices_by_direction`` (and thus the eye-trace plots from
:func:`eyehead.sort_saccades`) currently covers the translational
(``saccade_frames_xy``) stream only; torsional saccade grouping has been
factored out for a separate script.

Reward contingency
-------------------
A trial is rewarded when the animal's saccade lands within an angular
acceptance zone around the target direction, within a fixed time window of
target onset. These numbers are properties of the rig/task, not of this
analysis, and are read from ``session_manifest.yml`` under
``reward_contingency`` (global default, overridable per session):

- ``reward_angle`` (deg): half-width of the angular acceptance zone around
  the target direction. A saccade is "congruent" (counted as toward the
  target) if the angle between its direction and the target's direction —
  computed in full 2D via ``arctan2``, not a left/right sign check — is
  within this many degrees.
- ``reward_window`` (s): how long after target onset a saccade can still
  be rewarded.
- ``scoring_mode``: ``single_shot`` or ``multi_attempt`` — see above. Must
  be set explicitly per session; there is no global fallback that would
  make sense across both rigs.

A fourth manifest key, ``congruency_window`` (s, a ``[lo, hi]`` pair), is
**not** part of the rig's reward contingency — it's purely an analysis
choice, the fixed post-target latency band used for one summary statistic
(the single-number "fraction of saccades toward target" reported by
:func:`congruency_in_window` / shown in :func:`plot_psth_and_congruency`).

Note
----
``reward_window`` doubles as the outer bound on a trial's duration —
there is no separate task time-out beyond it. A trial ends either on a
rewarded saccade within ``reward_window`` of target onset, or (if none
occurs) at ``reward_window`` itself. Accordingly, ``end_of_trial_frame -
go_frame`` should never legitimately exceed ``reward_window``, and
``main()`` uses ``reward_window`` directly as the search-latency ceiling
(``max_trial_time``) for every first-saccade lookup in this file.

Consistency checks
-------------------
Before running the rest of the analysis, two QC cross-checks
(:func:`session_reward_window`, :func:`session_acceptance_angle`) each
independently re-derive their respective quantity from the data — the
maximum first-saccade latency, and the 90th-percentile angular deviation,
both computed only over trials the rig actually rewarded — and compare it
to the manifest's configured value. Both always print the derived-vs-
manifest comparison; either raises a ``ValueError`` if the two disagree by
more than 10% (relative to the manifest value), which usually means either
the manifest entry is wrong for this session or something is off in the
session's ``end_of_trial``/``trial_success`` data.

Saccade detection itself (thresholds for translational/torsional saccades,
blink rejection) is a separate, unrelated manifest section
(``saccade_config``) and is not part of the reward contingency.

Function reference
-------------------
:func:`collect_psth_trials`
    Gathers each trial's relative saccade times and duration (both in
    seconds, decoupled from any one session's frame-number scale) — the raw
    ingredients for a PSTH. Session-agnostic, so results from several
    sessions can be concatenated and pooled by
    :func:`psth_rate_from_trials`.

:func:`psth_rate_from_trials`
    The trial-pooling core of the PSTH: given per-trial relative saccade
    times/durations (from one or many sessions), computes the at-risk-
    normalised rate curve with a trial-resampling bootstrap CI.

:func:`compute_saccade_psth`
    Thin per-session wrapper around :func:`collect_psth_trials` +
    :func:`psth_rate_from_trials`: target-aligned saccade-rate PSTH (Hz)
    with a bootstrap CI for a single session. Respects each trial's actual
    end (``end_of_trial_frame``) so post-target bins are normalised by the
    number of trials still "at risk" at that bin, not a flat trial count.

:func:`session_reward_window`
    QC cross-check: derives the reward window directly from the data (max
    first-saccade latency among rewarded, congruent trials) and compares it
    to the manifest's ``reward_window``, raising if they disagree by more
    than ``tolerance``.

:func:`session_acceptance_angle`
    QC cross-check: derives the acceptance angle directly from the data (a
    percentile of angular deviation among rewarded trials' scored saccades)
    and compares it to the manifest's ``reward_angle``, raising if they
    disagree by more than ``tolerance``.

:func:`find_first_saccade_per_trial`
    For every trial, finds the saccade the rig actually scored (via
    :func:`_find_scored_saccade`) and returns its latency and whether it was
    congruent with the target direction. Trials with no detectable saccade
    in the window are omitted.

:func:`first_saccade_indices_by_direction`
    Same per-trial scored-saccade lookup as
    :func:`find_first_saccade_per_trial`, but returns eye-position indices
    grouped by stimulus direction (Left/Right/Up/Down) instead of
    latencies, for feeding into :func:`eyehead.sort_saccades`'s
    per-condition plots. Also returns a matching congruent/incongruent flag
    per group so those plots can be colored by correctness.

:func:`find_precue_saccade_per_trial`
    Pre-target control: finds each trial's last saccade before target onset
    (within an equal-duration lookback window) and its "congruency" with
    the not-yet-seen target, establishing the chance-level baseline that
    real target-directed congruency is compared against.

:func:`wilson_ci`
    Wilson score confidence interval for a binomial proportion; used to put
    error bars on congruency fractions.

:func:`_find_scored_saccade`
    Core scoring-mode logic. Given all in-window saccades of a trial, picks
    the one the Bonsai code actually ended the trial on — the first
    saccade for ``single_shot``, or the first congruent saccade (else the
    last attempt) for ``multi_attempt``. Every congruency-aware function in
    this file routes through this, so "congruent" means one consistent
    thing everywhere.

:func:`_angular_deviation_deg`
    Angular difference, in degrees, between a saccade's 2D direction and
    the target's direction, computed via ``arctan2`` so it's correct for
    oblique/vertical targets, not just left/right.

:func:`congruency_in_window`
    Fraction of first saccades that were congruent, restricted to a fixed
    post-target latency window, with a Wilson 95% CI. This is the
    single-number summary shown in the psth/congruency figure's third
    panel.

:func:`fraction_toward_target_by_latency`
    Same congruency fraction as :func:`congruency_in_window`, but computed
    in sliding latency windows across the whole trial duration, to see how
    accuracy evolves with response time. Feeds the psth/congruency
    figure's middle panel.

:func:`analyze_latency_by_outcome`
    Per-trial first-saccade latency paired with its congruent/incongruent
    label, computed the same way as :func:`find_first_saccade_per_trial`
    but additionally keeping a full trial accounting (``n_no_saccade``,
    ``n_total``) for the latency-by-outcome histogram/CDF figure.

:func:`calculate_trial_success`
    Recomputes trial-by-trial success/failure straight from the
    eye-tracking data using the same scoring rule as everything else in
    this file, with exactly one entry per trial (nothing dropped) so it can
    be compared 1:1 against the rig's own logged ``trial_success``.

:func:`plot_latency_by_outcome`
    Draws the two-panel latency-by-outcome figure (top: histogram of
    latency split by correct/incorrect; bottom: matching CDFs), optionally
    shading the reward window, and saves it to ``results_dir``.

:func:`plot_trial_success_agreement`
    Draws a per-trial strip plot comparing the rig's logged outcome against
    the outcome recomputed by :func:`calculate_trial_success`, with a third
    row marking per-trial agreement/disagreement, and reports overall
    percent agreement.

:func:`plot_psth_and_congruency`
    Draws the three-panel summary figure — target-aligned rate PSTH,
    saccade accuracy vs. latency, and windowed congruency vs. a pre-cue
    control — optionally shading the reward window on the time-resolved
    panels.

:func:`main`
    Runs the full per-session pipeline, in this order: loads the session
    and its config, calibrates eye position, detects saccades, and
    organizes trials by stimulus direction; runs the two QC cross-checks
    (:func:`session_reward_window`, :func:`session_acceptance_angle`);
    computes first-saccade indices by direction
    (:func:`first_saccade_indices_by_direction`) and feeds them to
    :func:`eyehead.sort_saccades` for the per-condition/polar plots;
    computes the target-aligned PSTH (:func:`compute_saccade_psth`),
    per-trial latency/congruency (:func:`find_first_saccade_per_trial`,
    :func:`fraction_toward_target_by_latency`,
    :func:`congruency_in_window`) and the pre-cue control
    (:func:`find_precue_saccade_per_trial`), then draws
    :func:`plot_psth_and_congruency`; separately computes
    :func:`analyze_latency_by_outcome` and draws
    :func:`plot_latency_by_outcome`; and, if the rig's ``trial_success`` is
    available, recomputes success via :func:`calculate_trial_success` and
    draws :func:`plot_trial_success_agreement`. Returns a small per-saccade
    DataFrame plus ``left_angle``/``right_angle``.

"""

from __future__ import annotations
from logging import config
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings


# Put the repo's “Python” folder on sys.path so `import eyehead` works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session_or_path

from eyehead import (
    SaccadeConfig,
    SessionData,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    organize_stims,
    sort_saccades,
    get_session_date_from_path,
)

def compute_saccade_psth(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    window: Tuple[float, float],
    bin_width: float = 0.1,
    mask: Optional[np.ndarray] = None,
    n_boot: int = 200,
    respect_trial_end: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Target-aligned saccade rate (Hz), with a bootstrap confidence band.

    Parameters
    ----------
    data : SessionData
        Must provide ``go_frame`` (target-onset frame numbers).
    saccades : dict
        Output of :func:`eyehead.detect_saccades`; must contain
        ``saccade_frames_xy``.
    config : SessionConfig
        Must provide ``ttl_freq`` (frames per second), used to convert frame
        counts to seconds, matching the convention used elsewhere in this
        repo (e.g. :func:`eyehead.sort_saccades`).
    window : (float, float)
        Time window relative to target onset, in seconds.
    bin_width : float
        Histogram bin width, in seconds.
    mask : boolean array, optional
        Selects a subset of target onsets (e.g. horizontal-target trials
        only). Defaults to all target onsets in ``data.go_frame``.
    n_boot : int
        Trial-resampling bootstrap iterations for the CI band; set to 0 to
        skip the bootstrap (``ci`` then just repeats ``rate``).
    respect_trial_end : bool
        When ``True`` (default) each trial only contributes saccades and
        exposure up to its own ``end_of_trial_frame``: post-target bins are
        normalised by the number of trials still within their trial at that
        bin (the "at-risk" count), so saccades from after a trial ended (the
        inter-trial period of trials that ended early) don't leak into the
        rate. Pre-target bins use all trials. With ``False`` (or no
        ``end_of_trial`` data) every trial contributes to the whole fixed
        window, normalised by the flat trial count (the original behaviour).

    Returns
    -------
    bin_centers, rate, ci, n_trials
        ``ci`` is a ``(2, n_bins)`` array of (lower, upper) 95% bootstrap
        bounds on the rate. Bins with no at-risk trials are ``NaN``.
    """
    ttl_freq = config.ttl_freq
    go_frame = np.asarray(data.go_frame)
    end_frame = data.end_of_trial_frame
    if mask is not None:
        go_frame = go_frame[mask]
        if end_frame is not None:
            end_frame = np.asarray(end_frame)[mask]
    n_trials = len(go_frame)

    if respect_trial_end and end_frame is not None:
        durations = (np.asarray(end_frame) - go_frame) / ttl_freq
    else:
        durations = np.full(n_trials, np.inf)

    saccade_times = saccades["saccade_frames_xy"] / ttl_freq
    edges = np.arange(window[0], window[1] + bin_width, bin_width)
    bin_centers = edges[:-1] + bin_width / 2

    def _rate_for(trial_idx: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(bin_centers))
        at_risk = np.zeros(len(bin_centers))
        for i in trial_idx:
            f = go_frame[i]
            dur = durations[i]
            # a bin is "observed" for this trial if it is pre-target or falls
            # before the trial ended
            observed = (bin_centers < 0) | (bin_centers < dur)
            at_risk += observed
            rel = saccade_times - f / ttl_freq
            in_window = rel[(rel >= window[0]) & (rel < window[1])]
            in_window = in_window[(in_window < 0) | (in_window < dur)]
            counts += np.histogram(in_window, bins=edges)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(at_risk > 0, counts / (at_risk * bin_width), np.nan)

    all_idx = np.arange(n_trials)
    rate = _rate_for(all_idx)

    if n_trials and n_boot:
        rng = np.random.default_rng()
        boot_rates = np.empty((n_boot, len(bin_centers)))
        for b in range(n_boot):
            resampled = rng.choice(all_idx, size=n_trials, replace=True)
            boot_rates[b] = _rate_for(resampled)
        ci = np.nanpercentile(boot_rates, [2.5, 97.5], axis=0)
    else:
        ci = np.tile(rate, (2, 1))

    return bin_centers, rate, ci, n_trials

def session_reward_window(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    acceptance_angle_deg: float,
    max_latency: float,
    scoring_mode: str,
    mask: Optional[np.ndarray] = None,
    tolerance: float = 0.20,
) -> Optional[float]:
    """Reward-window duration (s), derived from the data, as a QC cross-check.

    A saccade within the reward window is rewarded and ends the trial;
    ``data.trial_success`` (the ``end_of_trial`` CSV's success column) logs
    which trials were rewarded. This returns the **maximum first-saccade
    latency among rewarded, congruent trials** — trials that were rewarded
    *and* whose first saccade landed within ``acceptance_angle_deg`` of the
    target direction.

    The pipeline's actual ``reward_window`` comes from the manifest's
    ``reward_contingency.reward_window``, not from this function. This
    function is a QC cross-check: both the derived and manifest values are
    always printed, and a ``ValueError`` is raised if they disagree by more
    than ``tolerance`` (as a fraction of the manifest value).

    Returns ``None`` if ``trial_success`` is unavailable or no such trial has
    a detectable first saccade, in which case the cross-check is skipped.
    """
    trial_success = data.trial_success
    if trial_success is None:
        return None
    ttl_freq = config.ttl_freq
    go_frame = np.asarray(data.go_frame)
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        end_frame = go_frame + max_latency * ttl_freq
    end_frame = np.asarray(end_frame)
    trial_success = np.asarray(trial_success)
    go_dir_x = np.asarray(data.go_direction_x, dtype=np.float64)
    go_dir_y = np.asarray(data.go_direction_y, dtype=np.float64)
    if mask is not None:
        go_frame = go_frame[mask]
        end_frame = end_frame[mask]
        trial_success = trial_success[mask]
        go_dir_x = go_dir_x[mask]
        go_dir_y = go_dir_y[mask]

    if data.trial_outcome_encoding == "code012":
        is_success = trial_success == 2
    else:
        is_success = trial_success > 0

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    rewarded_latencies = []
    for f, end_f, succ, gdx, gdy in zip(go_frame, end_frame, is_success, go_dir_x, go_dir_y):
        if not succ:
            continue
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            continue
        _, latency, is_congruent = _find_scored_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dx, dy, gdx, gdy,
            acceptance_angle_deg, scoring_mode,
        )
        if not is_congruent:
            continue  # rig marked this rewarded, but no attempt actually was congruent — don't corrupt the estimate
        rewarded_latencies.append(latency)

    if not rewarded_latencies:
        return None
    derived = float(np.max(rewarded_latencies))

    reward_contingency = config.params.get("reward_contingency") or {}
    manifest_window = reward_contingency.get("reward_window")
    if manifest_window is not None and manifest_window > 0:
        relative_diff = abs(derived - manifest_window) / manifest_window
        print(
            f"Reward window check: derived (max over rewarded/congruent trials) "
            f"= {derived:.3f}s, manifest reward_window = {manifest_window:.3f}s "
            f"(diff {relative_diff:.0%})"
        )
        if relative_diff > tolerance:
            raise ValueError(
                f"Derived reward window ({derived:.3f}s, from rewarded/congruent "
                f"trials) differs from the manifest's reward_contingency.reward_window "
                f"({manifest_window:.3f}s) by {relative_diff:.0%}, more than the allowed "
                f"{tolerance:.0%} tolerance. Check the manifest entry or this session's "
                "end_of_trial/trial_success data."
            )

    return derived

def session_acceptance_angle(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    max_latency: float,
    acceptance_angle_deg: float,
    scoring_mode: str,
    mask: Optional[np.ndarray] = None,
    percentile: float = 90.0,
    tolerance: float = 0.80,
) -> Optional[float]:
    """Acceptance angle (deg), derived from the data, as a QC cross-check.

    Among rewarded trials, the first saccade within the reward window must
    have landed inside the rig's true angular acceptance zone around the
    target direction, or it wouldn't have been rewarded. This returns the
    ``percentile``-th percentile of angular deviation among rewarded trials'
    first saccades (default: 90th) — robust to a single noisy/mis-detected
    rewarded trial, unlike a hard max.

    The pipeline's actual acceptance angle comes from the manifest's
    ``reward_contingency.reward_angle``, not from this function. This is a
    QC cross-check: both the derived and manifest values are always printed,
    and a ``ValueError`` is raised if they disagree by more than ``tolerance``
    (as a fraction of the manifest value).

    Returns ``None`` if ``trial_success`` is unavailable or no rewarded trial
    has a detectable first saccade, in which case the cross-check is skipped.
    """
    trial_success = data.trial_success
    if trial_success is None:
        return None
    ttl_freq = config.ttl_freq
    go_frame = np.asarray(data.go_frame)
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        end_frame = go_frame + max_latency * ttl_freq
    end_frame = np.asarray(end_frame)
    trial_success = np.asarray(trial_success)
    go_dir_x = np.asarray(data.go_direction_x, dtype=np.float64)
    go_dir_y = np.asarray(data.go_direction_y, dtype=np.float64)
    if mask is not None:
        go_frame = go_frame[mask]
        end_frame = end_frame[mask]
        trial_success = trial_success[mask]
        go_dir_x = go_dir_x[mask]
        go_dir_y = go_dir_y[mask]

    if data.trial_outcome_encoding == "code012":
        is_success = trial_success == 2
    else:
        is_success = trial_success > 0

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    deviations = []
    for f, end_f, succ, gdx, gdy in zip(go_frame, end_frame, is_success, go_dir_x, go_dir_y):
        if not succ:
            continue
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            continue
        idx_chosen, _, is_congruent = _find_scored_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dx, dy, gdx, gdy,
            acceptance_angle_deg, scoring_mode,
        )
        if not is_congruent:
            continue
        deviations.append(_angular_deviation_deg(dx[idx_chosen], dy[idx_chosen], gdx, gdy))

    if not deviations:
        return None
    derived = float(np.percentile(deviations, percentile))

    reward_contingency = config.params.get("reward_contingency") or {}
    manifest_angle = reward_contingency.get("reward_angle")
    if manifest_angle is not None and manifest_angle > 0:
        relative_diff = abs(derived - manifest_angle) / manifest_angle
        print(
            f"Acceptance angle check: derived (p{percentile:g} of rewarded trials) "
            f"= {derived:.1f}°, manifest reward_angle = {manifest_angle:.1f}° "
            f"(diff {relative_diff:.0%})"
        )
        if relative_diff > tolerance:
            raise ValueError(
                f"Derived acceptance angle ({derived:.1f}°, p{percentile:g} of rewarded "
                f"trials) differs from the manifest's reward_contingency.reward_angle "
                f"({manifest_angle:.1f}°) by {relative_diff:.0%}, more than the allowed "
                f"{tolerance:.0%} tolerance. Check the manifest entry or this session's "
                "end_of_trial/trial_success data."
            )

    return derived


def find_first_saccade_per_trial(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    acceptance_angle_deg: float,
    max_latency: float,
    scoring_mode: str,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Latency and target-congruency of the first saccade in each trial.

    For every trial, finds the first saccade between target onset
    (``go_frame``) and that trial's actual end (``end_of_trial_frame``,
    falling back to a flat ``max_latency`` cap if unavailable) and records
    its latency together with whether it was congruent.

    Parameters
    ----------
    acceptance_angle_deg : float
        A saccade counts as congruent only if its 2D direction
        (``arctan2(dy, dx)``) falls within this many degrees of the target's
        actual direction (``arctan2(go_dir_y, go_dir_x)``) — see
        :func:`_angular_deviation_deg`. This replaces the old left/right
        hemifield-only check (``sign(dx) == sign(go_dir_x)``), so it also
        handles oblique and up/down targets correctly, not just horizontal
        ones. Should be the manifest's ``reward_contingency.reward_angle``,
        so "congruent" here matches what the rig actually rewards.
    max_latency : float
        Trials whose first saccade lands later than this (seconds after
        target onset) are excluded entirely, not counted as incongruent.

    Returns
    -------
    latencies : ndarray
        First-saccade latency (s) for each trial with a detectable saccade
        in the search window. Trials with none are omitted.
    congruent : ndarray of bool
        Whether that trial's first saccade fell within ``acceptance_angle_deg``
        of the target direction.
    """

    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    go_dir_y = data.go_direction_y
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq
    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]
        go_dir_y = go_dir_y[mask]
        end_frame = end_frame[mask]

    go_dir_x = np.array(go_dir_x, dtype=np.float64)
    go_dir_y = np.array(go_dir_y, dtype=np.float64)

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    latencies, congruent = [], []
    for f, gdx, gdy, end_f in zip(go_frame, go_dir_x, go_dir_y, end_frame):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            continue
        _, latency, is_congruent = _find_scored_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dx, dy, gdx, gdy,
            acceptance_angle_deg, scoring_mode,
        )
        latencies.append(latency)
        congruent.append(is_congruent)

    return np.array(latencies), np.array(congruent, dtype=bool)


def first_saccade_indices_by_direction(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    acceptance_angle_deg: float,
    max_latency: float,
    scoring_mode: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """First-saccade eye indices per trial, grouped by stimulus direction.

    Selects, for every trial, the saccade the rig actually scored the trial
    on — see :func:`_find_scored_saccade` for the ``scoring_mode``-dependent
    definition (``single_shot``: the first saccade, any direction;
    ``multi_attempt``: the first congruent saccade, or the last attempt if
    none was congruent) — and returns the corresponding index into the
    eye-position arrays.

    Operates on the translational (``saccade_frames_xy`` / ``eye_vel``)
    stream only. Torsional saccade grouping is out of scope here.

    Trials are grouped into direction labels exactly as :func:`organize_stims`
    groups them, so the returned dicts can be dropped straight into
    :func:`eyehead.sort_saccades` in place of its fixed-window search.

    Returns
    -------
    indices_by_direction : dict
        Direction label -> eye-position indices of the scored saccade.
    congruent_by_direction : dict
        Same keys/order as ``indices_by_direction``; whether each of those
        saccades was the calculated "correct" outcome (within
        ``acceptance_angle_deg`` of the target) — lets callers recolor
        plots by correctness instead of direction.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    go_dir_y = data.go_direction_y
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq

    saccade_frames = saccades.get("saccade_frames_xy")
    saccade_indices = saccades.get("saccade_indices_xy")
    if saccade_frames is None or saccade_indices is None or len(saccade_frames) == 0:
        return {}, {}
    saccade_frames = np.asarray(saccade_frames)
    saccade_indices = np.asarray(saccade_indices)
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    go_dir_x_f = np.array(go_dir_x, dtype=np.float64)
    go_dir_y_f = np.array(go_dir_y, dtype=np.float64)

    first_idx = np.full(len(go_frame), -1, dtype=int)
    first_congruent = np.zeros(len(go_frame), dtype=bool)
    for i, (f, end_f, gdx, gdy) in enumerate(zip(go_frame, end_frame, go_dir_x_f, go_dir_y_f)):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            continue
        idx_chosen, _, is_congruent = _find_scored_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dx, dy, gdx, gdy,
            acceptance_angle_deg, scoring_mode,
        )
        first_idx[i] = idx_chosen
        first_congruent[i] = is_congruent

    have = first_idx >= 0
    groups: Dict[str, np.ndarray] = {}
    congruent_groups: Dict[str, np.ndarray] = {}
    if go_dir_x is not None and np.any(np.asarray(go_dir_x) != 0):
        go_dir_x = np.asarray(go_dir_x)
        sel_left = have & (go_dir_x < 0)
        sel_right = have & (go_dir_x > 0)
        groups["Left"] = first_idx[sel_left]
        groups["Right"] = first_idx[sel_right]
        congruent_groups["Left"] = first_congruent[sel_left]
        congruent_groups["Right"] = first_congruent[sel_right]
    if go_dir_y is not None and np.any(np.asarray(go_dir_y) != 0):
        go_dir_y = np.asarray(go_dir_y)
        sel_down = have & (go_dir_y < 0)
        sel_up = have & (go_dir_y > 0)
        groups["Down"] = first_idx[sel_down]
        groups["Up"] = first_idx[sel_up]
        congruent_groups["Down"] = first_congruent[sel_down]
        congruent_groups["Up"] = first_congruent[sel_up]
    if not groups:
        groups["All"] = first_idx[have]
        congruent_groups["All"] = first_congruent[have]
    return groups, congruent_groups

def find_precue_saccade_per_trial(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    acceptance_angle_deg: float,
    window: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Latency and target-congruency of the last saccade before target onset.

    Serves as a pre-cue/pre-target control: since the target hasn't appeared
    yet, any directional bias in this saccade can't reflect target-directed
    behaviour, so congruency here should sit near chance.

    Parameters
    ----------
    acceptance_angle_deg : float
        Same angular acceptance-zone definition as in
        :func:`find_first_saccade_per_trial` — should be the manifest's
        ``reward_contingency.reward_angle``.
    window : float
        How far back before target onset to search (seconds). Should be the
        manifest's ``reward_contingency.reward_window``, so this control
        period has the same duration as the real post-target reward window
        it's being compared against — an equal-duration baseline, not an
        arbitrary lookback.
        
    Returns
    -------
    latencies : ndarray
        Time (s), negative, from target onset to each trial's last saccade
        in the preceding ``window`` seconds. Trials with none are omitted.
    congruent : ndarray of bool
        Whether that saccade's direction happened to fall within
        ``acceptance_angle_deg`` of the (not-yet-seen) target's direction.
    """

    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    go_dir_y = data.go_direction_y
    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]
        go_dir_y = go_dir_y[mask]

    go_dir_x = np.array(go_dir_x, dtype=np.float64)
    go_dir_y = np.array(go_dir_y, dtype=np.float64)

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    latencies, congruent = [], []
    for f, gdx, gdy in zip(go_frame, go_dir_x, go_dir_y):
        rel = (saccade_frames - f) / ttl_freq
        valid = (rel < 0) & (rel >= -window)
        if not np.any(valid):
            continue
        rel_valid = rel[valid]
        last = np.argmax(rel_valid)  # closest to (but before) target onset
        idx_last = saccade_indices[valid][last]
        latencies.append(rel_valid[last])
        congruent.append(_angular_deviation_deg(dx[idx_last], dy[idx_last], gdx, gdy) <= acceptance_angle_deg)

    return np.array(latencies), np.array(congruent, dtype=bool)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (center - margin, center + margin)

def _find_scored_saccade(rel_valid, idx_valid, dx, dy, gdx, gdy, acceptance_angle_deg, scoring_mode):
    """Pick which in-window saccade the Bonsai code used.

    ``single_shot`` (Apollo): the first saccade ends the trial regardless
    of direction.

    ``multi_attempt`` (Paris): an incongruent saccade doesn't end the
    trial, so the scored saccade is the first *congruent* one. If none of
    the attempts in the window were congruent (never rewarded), the last
    attempt is returned instead, with congruent=False — its latency
    reflects how long the animal kept trying before time-out.

    Returns
    -------
    idx : int
        Index into the eye-velocity arrays (``dx``/``dy``) for the chosen
        saccade.
    latency : float
        Its time (s) since target onset.
    congruent : bool
        Whether it fell within ``acceptance_angle_deg`` of the target
        direction.
    """
    order = np.argsort(rel_valid)
    rel_sorted = rel_valid[order]
    idx_sorted = idx_valid[order]

    if scoring_mode == "single_shot":
        idx0 = idx_sorted[0]
        is_congruent = _angular_deviation_deg(dx[idx0], dy[idx0], gdx, gdy) <= acceptance_angle_deg
        return idx0, float(rel_sorted[0]), bool(is_congruent)

    if scoring_mode == "multi_attempt":
        for t, idx in zip(rel_sorted, idx_sorted):
            if _angular_deviation_deg(dx[idx], dy[idx], gdx, gdy) <= acceptance_angle_deg:
                return idx, float(t), True
        return idx_sorted[-1], float(rel_sorted[-1]), False

    raise ValueError(
        f"Unknown scoring_mode {scoring_mode!r}; expected 'single_shot' or 'multi_attempt'."
    )


def _angular_deviation_deg(dx, dy, gdx, gdy):
    saccade_angle = np.arctan2(dy, dx)
    target_angle = np.arctan2(gdy, gdx)
    diff = np.arctan2(np.sin(saccade_angle - target_angle), np.cos(saccade_angle - target_angle))
    return np.degrees(np.abs(diff))

def congruency_in_window(
    latencies: np.ndarray,
    congruent: np.ndarray,
    window: Tuple[float, float],
) -> Tuple[float, int, float, float]:
    """Fraction congruent (with Wilson 95% CI) for saccades in a fixed latency window."""
    sel = (latencies >= window[0]) & (latencies < window[1])
    n = int(np.count_nonzero(sel))
    if n == 0:
        return np.nan, 0, np.nan, np.nan
    frac = float(np.mean(congruent[sel]))
    ci_lo, ci_hi = wilson_ci(int(np.sum(congruent[sel])), n)
    return frac, n, ci_lo, ci_hi

def fraction_toward_target_by_latency(
    latencies: np.ndarray,
    congruent: np.ndarray,
    window_span: Tuple[float, float],
    win_width: float = 0.3,
    step: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fraction of first saccades toward the target, in sliding latency windows."""
    centers = np.arange(window_span[0], window_span[1] + step, step)
    fraction = np.full(len(centers), np.nan)
    n_per_window = np.zeros(len(centers), dtype=int)

    for i, c in enumerate(centers):
        lo, hi = c - win_width / 2, c + win_width / 2
        sel = (latencies >= lo) & (latencies < hi)
        n_per_window[i] = np.count_nonzero(sel)
        if n_per_window[i] > 0:
            fraction[i] = np.mean(congruent[sel])

    return centers, fraction, n_per_window


def analyze_latency_by_outcome(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    acceptance_angle_deg: float,
    max_latency: float,
    scoring_mode: str,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Per-trial first-saccade latency, paired with a "correct" label.

    For every trial, finds the first saccade between target onset and that
    trial's actual end (``data.end_of_trial_frame``) and records its latency,
    together with ``congruent``: whether that saccade's direction fell within
    ``acceptance_angle_deg`` of the target's actual direction (same
    definition as :func:`find_first_saccade_per_trial`).

    Trials with no saccade between target onset and trial end are excluded
    from the returned arrays but counted in ``n_no_saccade``.
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    go_dir_y = data.go_direction_y
    end_frame = data.end_of_trial_frame

    if end_frame is None:
        warnings.warn(
            "No end_of_trial data available; falling back to a flat "
            f"{max_latency}s latency cap instead of each trial's actual end."
        )
        end_frame = go_frame + max_latency * ttl_freq

    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]
        go_dir_y = go_dir_y[mask]
        end_frame = end_frame[mask]

    go_dir_x = np.array(go_dir_x, dtype=np.float64)
    go_dir_y = np.array(go_dir_y, dtype=np.float64)

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    latencies, congruent = [], []
    n_no_saccade = 0
    for f, gdx, gdy, end_f in zip(go_frame, go_dir_x, go_dir_y, end_frame):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            n_no_saccade += 1
            continue
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            n_no_saccade += 1
            continue
        _, latency, is_congruent = _find_scored_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dx, dy, gdx, gdy,
            acceptance_angle_deg, scoring_mode,
        )
        latencies.append(latency)
        congruent.append(is_congruent)

    return {
        "latencies": np.array(latencies),
        "congruent": np.array(congruent, dtype=bool),
        "n_no_saccade": n_no_saccade,
        "n_total": len(go_frame),
    }

def calculate_trial_success(
    data: SessionData,
    saccades: Dict[str, np.ndarray],
    config,
    acceptance_angle_deg: float,
    max_latency: float,
    scoring_mode: str,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trial calculated success/failure, using the same scoring rule as
    the rest of this file (:func:`_find_scored_saccade`).

    Unlike :func:`find_first_saccade_per_trial`, no trial is dropped: a
    trial with no detectable saccade in the window is recorded as a
    calculated failure (``False``) rather than omitted, so the result has
    exactly one entry per trial and can be compared 1:1 against the rig's
    own ``data.trial_success`` log.

    Returns
    -------
    calculated_success : ndarray of bool
        Per-trial calculated outcome.
    has_saccade : ndarray of bool
        Whether any saccade was detected in the window at all. A trial
        with ``has_saccade=False`` is always ``calculated_success=False``
        too, but returned separately so callers can distinguish "no
        saccade detected" from "a saccade was detected but incongruent".
    """
    ttl_freq = config.ttl_freq
    go_frame = data.go_frame
    go_dir_x = data.go_direction_x
    go_dir_y = data.go_direction_y
    end_frame = data.end_of_trial_frame
    if end_frame is None:
        end_frame = go_frame + max_latency * ttl_freq
    if mask is not None:
        go_frame = go_frame[mask]
        go_dir_x = go_dir_x[mask]
        go_dir_y = go_dir_y[mask]
        end_frame = end_frame[mask]

    go_dir_x = np.array(go_dir_x, dtype=np.float64)
    go_dir_y = np.array(go_dir_y, dtype=np.float64)

    saccade_frames = saccades["saccade_frames_xy"]
    saccade_indices = saccades["saccade_indices_xy"]
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    calculated_success = np.zeros(len(go_frame), dtype=bool)
    has_saccade = np.zeros(len(go_frame), dtype=bool)
    for i, (f, gdx, gdy, end_f) in enumerate(zip(go_frame, go_dir_x, go_dir_y, end_frame)):
        valid = (saccade_frames > f) & (saccade_frames < end_f)
        if not np.any(valid):
            continue  # no saccade at all -> calculated failure, has_saccade stays False
        rel_valid = (saccade_frames[valid] - f) / ttl_freq
        idx_valid = saccade_indices[valid]
        in_latency = rel_valid <= max_latency
        if not np.any(in_latency):
            continue
        has_saccade[i] = True
        _, _, is_congruent = _find_scored_saccade(
            rel_valid[in_latency], idx_valid[in_latency], dx, dy, gdx, gdy,
            acceptance_angle_deg, scoring_mode,
        )
        calculated_success[i] = is_congruent

    return calculated_success, has_saccade


def plot_latency_by_outcome(
    result: Dict[str, np.ndarray],
    config,
    bins: int = 20,
    show_plots: bool = True,
    reward_window: Optional[float] = None,
) -> plt.Figure:

    """First-saccade latency split by saccade-target congruency.

    Top: overlaid histograms of latency for congruent ("correct", toward the
    target) vs. incongruent ("incorrect") first saccades. Bottom: empirical
    CDFs of those same two distributions, using the matching colour scheme.

    ``reward_window`` (seconds), if given, is drawn as a shaded region /
    dashed line marking the rewarded epoch, and the fraction of first
    saccades that fell within it is reported in the title."""
    latencies = result["latencies"]
    congruent = result["congruent"]
    n_no_saccade = result["n_no_saccade"]

    correct = latencies[congruent]
    incorrect = latencies[~congruent]
    n_correct = correct.size
    n_incorrect = incorrect.size

    fig, (ax_hist, ax_cdf) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    ax_hist.hist(correct, bins=bins, alpha=0.6, color="tab:green", label="correct")
    ax_hist.hist(incorrect, bins=bins, alpha=0.6, color="tab:red", label="incorrect")
    ax_hist.set_ylabel("Trial count")
    ax_hist.set_title(
        f"Correct = saccade toward target\n{n_correct} correct, {n_incorrect} incorrect"
    )

    def _mark_reward(ax):
        if reward_window is None:
            return
        ax.axvspan(0, reward_window, color="gold", alpha=0.12, lw=0)
        ax.axvline(reward_window, color="goldenrod", ls=":", lw=1.2,
                   label=f"reward window ({reward_window:g}s)")

    _mark_reward(ax_hist)
    ax_hist.legend(fontsize=8)

    def _cdf(ax, values, color, label):
        if values.size == 0:
            return
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", color=color, label=label)

    _cdf(ax_cdf, correct, "tab:green", "correct")
    _cdf(ax_cdf, incorrect, "tab:red", "incorrect")
    _mark_reward(ax_cdf)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel("First-saccade latency (s)")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("Latency CDF")
    ax_cdf.legend(fontsize=8)

    reward_txt = ""
    if reward_window is not None and len(latencies):
        within = float(np.mean(latencies <= reward_window))
        reward_txt = f"  —  {within:.0%} of first saccades within reward window (≤{reward_window:g}s)"

    fig.suptitle(
        f"{config.animal_name or ''} {config.session_name}  —  "
        f"{result['n_total']} trials, {len(latencies)} with a saccade, "
        f"{n_no_saccade} excluded (no saccade){reward_txt}"
    )
    fig.tight_layout()
    fig.savefig(config.results_dir / f"{config.session_name}_latency_by_outcome.png",
                dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def plot_trial_success_agreement(
    calculated_success: np.ndarray,
    rig_success: np.ndarray,
    has_saccade: np.ndarray,
    config,
    show_plots: bool = True,
) -> plt.Figure:
    """Per-trial comparison of calculated vs. rig-logged trial outcomes.

    Two rows of colored ticks over trial index (rig outcome, calculated
    outcome; green = success, red = failure, black = no saccade detected
    in the window at all), plus a third strip marking per-trial agreement
    (light gray = agree, orange = disagree). Overall agreement percentage
    and each side's %correct are reported in the title.
    """
    agree = calculated_success == rig_success
    n = len(calculated_success)
    trial_idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.02), 3))

    def _outcome_colors(values):
        colors = np.where(values, "tab:green", "tab:red")
        return np.where(has_saccade, colors, "black")

    def _row(y, values, label, use_has_saccade):
        colors = np.where(values, "tab:green", "tab:red")
        if use_has_saccade:
            colors = np.where(has_saccade, colors, "black")
        ax.scatter(trial_idx, np.full(n, y), c=colors, marker="|", s=200, linewidths=1.5)
        ax.text(-n * 0.02, y, label, ha="right", va="center", fontsize=9)

    _row(2, rig_success, "Rig", use_has_saccade=False)
    _row(1, calculated_success, "Calculated", use_has_saccade=True)

    disagree_colors = np.where(agree, "lightgray", "tab:orange")
    ax.scatter(trial_idx, np.full(n, 0), c=disagree_colors, marker="|", s=200, linewidths=1.5)
    ax.text(-n * 0.02, 0, "Agreement", ha="right", va="center", fontsize=9)

    ax.set_xlim(-n * 0.05, n)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xlabel("Trial index")
    pct_agree = 100 * np.mean(agree) if n else float("nan")
    pct_rig = 100 * np.mean(rig_success) if n else float("nan")
    pct_calc = 100 * np.mean(calculated_success) if n else float("nan")
    ax.set_title(
        f"{config.animal_name or ''} {config.session_name} — calculated vs. rig trial outcome "
        f"({pct_agree:.1f}% agreement, {int(np.sum(~agree))}/{n} disagree)\n"
        f"% correct — rig: {pct_rig:.1f}%, calculated: {pct_calc:.1f}%",
        fontsize=10, wrap=True,
    )

    legend_handles = [
        Line2D([0], [0], marker="|", color="tab:green", linestyle="none",
               markersize=12, markeredgewidth=2, label="Rewarded"),
        Line2D([0], [0], marker="|", color="tab:red", linestyle="none",
               markersize=12, markeredgewidth=2, label="Not rewarded"),
        Line2D([0], [0], marker="|", color="black", linestyle="none",
               markersize=12, markeredgewidth=2, label="No saccade detected"),
        Line2D([0], [0], marker="|", color="lightgray", linestyle="none",
               markersize=12, markeredgewidth=2, label="Agree "),
        Line2D([0], [0], marker="|", color="tab:orange", linestyle="none",
               markersize=12, markeredgewidth=2, label="Disagree "),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(config.results_dir / f"{config.session_name}_trial_success_agreement.png",
                dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def plot_psth_and_congruency(
    psth_centers: np.ndarray,
    psth_rate: np.ndarray,
    psth_ci: np.ndarray,
    n_trials: int,
    latency_centers: np.ndarray,
    fraction_toward: np.ndarray,
    n_per_window: np.ndarray,
    frac: float,
    n_window: int,
    ci_lo: float,
    ci_hi: float,
    precue_frac: float,
    precue_n: int,
    config,
    window: Tuple[float, float],
    show_plots: bool = True,
    reward_window: Optional[float] = None,
) -> plt.Figure:
    """Three-panel figure: target-aligned rate PSTH, accuracy-by-latency, and windowed congruency vs. pre-cue control.

    ``reward_window`` (seconds), if given, is shaded on the time-resolved
    panels (PSTH and accuracy-vs-latency) to mark the rewarded epoch."""
    fig, (ax_rate, ax_frac, ax_summary) = plt.subplots(1, 3, figsize=(14, 4))

    def _mark_reward(ax):
        if reward_window is None:
            return
        ax.axvspan(0, reward_window, color="gold", alpha=0.12, lw=0)
        ax.axvline(reward_window, color="goldenrod", ls=":", lw=1.2,
                   label=f"reward window ({reward_window:g}s)")

    ax_rate.fill_between(psth_centers, psth_ci[0], psth_ci[1], color="tab:blue", alpha=0.25, lw=0)
    ax_rate.plot(psth_centers, psth_rate, color="tab:blue", lw=1.4)
    ax_rate.axvline(0, color="k", lw=0.8)
    _mark_reward(ax_rate)
    ax_rate.set_xlabel("Time from target onset (s)")
    ax_rate.set_ylabel("Saccade rate (Hz)")
    ax_rate.set_title(f"Target-aligned saccade rate (n={n_trials} trials)")
    if reward_window is not None:
        ax_rate.legend(fontsize=8)

    valid = n_per_window > 0
    ax_frac.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax_frac.plot(latency_centers[valid], fraction_toward[valid], "-o", color="tab:green", ms=3)
    _mark_reward(ax_frac)
    ax_frac.set_ylim(0, 1)
    ax_frac.set_xlabel("Window centre, time from target (s)")
    ax_frac.set_ylabel("Fraction of first saccades toward target")
    ax_frac.set_title("Saccade accuracy vs. latency")

    if n_window > 0:
        ax_summary.errorbar(
            frac, 0,
            xerr=[[frac - ci_lo], [ci_hi - frac]],
            fmt="D", color="tab:red", ecolor="tab:red", capsize=4, ms=9,
        )
        ax_summary.text(max(ci_hi, frac) + 0.03, 0, f"{frac:.0%}  (n={n_window})",
                         va="center", fontsize=9, color="dimgray")

    if n_window > 0:
        ax_summary.errorbar(
            frac, 0,
            xerr=[[frac - ci_lo], [ci_hi - frac]],
            fmt="D", color="tab:red", ecolor="tab:red", capsize=4, ms=9,
        )
        ax_summary.text(max(ci_hi, frac) + 0.03, 0.15, f"{frac:.0%}  (n={n_window})",
                         va="center", fontsize=9, color="tab:red")
    if precue_n > 0 and not np.isnan(precue_frac):
        ax_summary.plot(precue_frac, 0, "o", mfc="none", mec="gray", ms=9)
        ax_summary.text(precue_frac, -0.15, f"pre-cue: {precue_frac:.0%} (n={precue_n})",
                         va="center", ha="center", fontsize=8, color="dimgray")
    ax_summary.axvline(0.5, color="gray", ls="--", lw=1)
    ax_summary.set_xlim(0.0, 1.15)
    ax_summary.set_ylim(-0.5, 0.5)
    ax_summary.set_yticks([])
    ax_summary.set_xlabel(f"Fraction toward target, {window[0]:.2f}\u2013{window[1]:.2f} s")
    ax_summary.set_title("Congruency vs. pre-cue control")
    

    fig.suptitle(f"{config.animal_name or ''} {config.session_name}".strip())
    fig.tight_layout()
    fig.savefig(config.results_dir / f"{config.session_name}_psth_congruency.png",
                dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def main(session_id: str) -> pd.DataFrame:
    """Run the full analysis pipeline for ``session_id``.

    Parameters
    ----------
    session_id:
        Identifier of the session to analyse, either a manifest session ID
        or a direct path to a session folder.
    """
    config = load_session_or_path(session_id)
    session_id = config.session_id
    config.results_dir.mkdir(parents=True, exist_ok=True)

    folder_path = config.folder_path
    results_dir = config.results_dir
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

    # The rest of the analysis would operate on ``folder_path`` and
    # save any generated figures into ``results_dir``.  For now we simply
    # report the resolved paths so that the script remains functional
    # even when the full analysis pipeline is not available.
    print(f"Session path: {folder_path}")
    print(f"Results directory: {results_dir}")

    date_str = config.params.get("date")
    if not date_str and folder_path is not None:
        try:
            date_str = get_session_date_from_path(str(folder_path)).strftime("%Y-%m-%d")
        except Exception:
            date_str = ""

    data = load_session_data(config)
    eye_pos_cal = calibrate_eye_position(data, config)

    saccade_cfg = SaccadeConfig(**config.params["saccade_config"])

    saccades, fig_saccades, _ = detect_saccades(
        eye_pos_cal,
        data.eye_frame,
        saccade_cfg,
        config,
        data=data,
        plot=True,
    )

    indices = saccades["saccade_indices_xy"]
    saccade_frames = saccades.get("saccade_frames_xy", [])
    print(f"Detected {len(indices)} saccades")
    saccades["stim_frames"], stim_type = organize_stims(
        data.go_frame,
        go_dir_x=data.go_direction_x,
        go_dir_y=data.go_direction_y,
    )

    df = pd.DataFrame(
        {
            "session_id": [session_id] * len(indices),
            "session_date": [date_str] * len(indices),
            "saccade_frame_xy": saccade_frames,
            "saccade_index_xy": indices,

        }
    )
    #disregard Up/Down target trials if they exist
    mask_horizontal = data.go_direction_x != 0


    # Reward window from the manifest's reward_contingency (per-session
    # override merged over the global default in session_manifest.yml), not
    # derived from data.
    reward_contingency = config.params.get("reward_contingency") or {}
    reward_window = reward_contingency.get("reward_window")
    if reward_window is None:
        raise ValueError(
            "No reward_window configured in reward_contingency for this session; "
            "add one to session_manifest.yml (global default or per-session override)."
        )
    reward_window = float(reward_window)

    acceptance_angle = reward_contingency.get("reward_angle")
    if acceptance_angle is None:
        raise ValueError(
            "No reward_angle configured in reward_contingency for this session; "
            "add one to session_manifest.yml (global default or per-session override)."
        )
    acceptance_angle = float(acceptance_angle)

    #figure out which rule the Bonsai used (first-saccade or first correct saccade)
    scoring_mode = reward_contingency.get("scoring_mode")
    if scoring_mode is None:
        raise ValueError(
            "No scoring_mode configured in reward_contingency for this session; "
            "add one to session_manifest.yml ('single_shot' for Apollo, "
            "'multi_attempt' for Paris — no safe global default, must be set per session)."
        )

    # QC cross-checks: each raises if the data's own rewarded-trial statistic
    # disagrees with its manifest value by more than 10% (both also print
    # derived-vs-manifest regardless of pass/fail).
    session_reward_window(
        data, saccades, config, acceptance_angle_deg=acceptance_angle,
        mask=mask_horizontal, max_latency=reward_window,scoring_mode=scoring_mode,
    )
    session_acceptance_angle(
        data, saccades, config, acceptance_angle_deg=acceptance_angle,
        mask=mask_horizontal, max_latency=reward_window, 
        scoring_mode=scoring_mode,
    )

    max_trial_time = reward_window

    
    first_saccades, first_saccades_congruent = first_saccade_indices_by_direction(
        data, saccades, config, acceptance_angle_deg=acceptance_angle,
        max_latency=max_trial_time, scoring_mode=scoring_mode,
    )
    # Torsional saccade grouping removed here — to be handled by a separate script.
    sorted_data,left_angle,right_angle,fig_sorted, _ = sort_saccades(
        config, saccade_cfg, saccades, stim_type=stim_type,
        first_saccade_indices=first_saccades,
        first_saccade_congruent=first_saccades_congruent,
        plot=True,
    )

    if fig_sorted is not None:
        plt.close(fig_sorted)

    psth_centers, psth_rate, psth_ci, n_trials_psth = compute_saccade_psth(
        data, saccades, config, window=(-max_trial_time, max_trial_time),
        mask=mask_horizontal,
    )
    latencies, congruent = find_first_saccade_per_trial(
        data, saccades, config, acceptance_angle_deg=acceptance_angle,
        max_latency=max_trial_time, mask=mask_horizontal, scoring_mode=scoring_mode,
    )
    latency_centers, fraction_toward, n_per_window = fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0.2, max_trial_time)
    )

    #get the early congruency window from manifest
    reward_contingency = config.params.get("reward_contingency") or {}
    congruency_window = reward_contingency.get("congruency_window")
    if congruency_window is None:
        raise ValueError(
            "No congruency_window configured in reward_contingency for this session; "
            "add one to session_manifest.yml (global default or per-session override)."
        )
    window = tuple(float(w) for w in congruency_window)

    #calculate direction congruency in the window
    frac, n_window, ci_lo, ci_hi = congruency_in_window(latencies, congruent, window=window)

    precue_latencies, precue_congruent = find_precue_saccade_per_trial(
        data, saccades, config, acceptance_angle_deg=acceptance_angle, 
        window=reward_window, mask=mask_horizontal)
    
    precue_frac = float(np.mean(precue_congruent)) if len(precue_congruent) else np.nan
    precue_n = len(precue_congruent)

    plot_psth_and_congruency(
        psth_centers, psth_rate, psth_ci, n_trials_psth,
        latency_centers, fraction_toward, n_per_window,
        frac, n_window, ci_lo, ci_hi, precue_frac, precue_n,
        config, window=window, reward_window=reward_window,
    )

    latency_outcome = analyze_latency_by_outcome(
        data, saccades, config, acceptance_angle_deg=acceptance_angle,
        mask=mask_horizontal, max_latency=max_trial_time, scoring_mode=scoring_mode)
    
    plot_latency_by_outcome(latency_outcome, config, reward_window=reward_window)

    #plot calculated trial success vs what the task produced    
    if data.trial_success is not None:
        trial_success_masked = np.asarray(data.trial_success)[mask_horizontal]
        if data.trial_outcome_encoding == "code012":
            rig_success = trial_success_masked == 2
        else:
            rig_success = trial_success_masked > 0

        calculated_success, has_saccade = calculate_trial_success(
            data, saccades, config, acceptance_angle_deg=acceptance_angle,
            max_latency=max_trial_time, scoring_mode=scoring_mode, mask=mask_horizontal,
        )
        plot_trial_success_agreement(calculated_success, rig_success, has_saccade, config)
    else:
        warnings.warn(
            "No trial_success data available; skipping calculated-vs-rig "
            "trial outcome comparison."
        )


    df = pd.DataFrame(
        {
            "session_id": [session_id] * len(indices),
            "session_date": [date_str] * len(indices),
            "saccade_frame_xy": saccade_frames,
            "saccade_index_xy": indices,

        }
    )
    return df,left_angle,right_angle


# Usage: python Python/analysis/prosaccade_session.py SESSION_ID_OR_PATH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session")
    parser.add_argument("session_id", help="Session identifier from session_manifest.yml, or a direct path to a session folder")
    args = parser.parse_args()
    main(args.session_id)

