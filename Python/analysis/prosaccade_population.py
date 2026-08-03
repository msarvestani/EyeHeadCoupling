"""Population-level (multi-session) analysis for the head-fixed prosaccade task.

This script selects sessions from ``session_manifest.yml`` (via
:func:`utils.session_loader.list_sessions_from_manifest`), filtered by
experiment type and, optionally, one animal; runs the full per-session
pipeline (:func:`prosaccade_session.main`) on each; and pools the results
per animal to produce the same three summary figures
:func:`prosaccade_session.main` draws per session — latency-by-outcome,
target-aligned PSTH/congruency, and left/right polar saccade distributions
— but computed over every trial from every one of that animal's sessions at
once, plus one more set pooling every animal together.

Pooling
-------
Two different pooling strategies are used, matching how each underlying
statistic is computed in :mod:`prosaccade_session`:

- Per-trial arrays (latency/congruency, pre-cue latency/congruency, PSTH
  relative-saccade-time lists and durations, and the left/right polar
  angles) are concatenated directly across sessions —
  :func:`pool_animal_sessions` does this. This is exact: the same
  at-risk/duration-based normalisation
  :func:`prosaccade_session.psth_rate_from_trials` uses within one session
  works identically over trials pooled from several, since each trial still
  carries its own duration regardless of which session it came from.
- ``reward_window``/``congruency_window`` are single per-session values,
  not per-trial, so they can't be concatenated the same way — see below.

``pool_animal_sessions``'s return dict is deliberately the same shape as
one raw session's :func:`prosaccade_session.main` result, so it can be fed
right back into itself. That's how the all-animals-combined figure is
built: by pooling the already-pooled per-animal dicts (see
:func:`run_population_summary_plots`), with no separate combining logic.

Reward-window / congruency-window / reward-angle disagreement
-----------------------------------------------------------------
Sessions being pooled don't always share the same
``reward_contingency.reward_window`` (this does happen — see
``session_manifest.yml``; Apollo alone has both 1.0s- and 1.5s-window
sessions). :func:`pool_animal_sessions` uses the **max** across sessions for
the pooled PSTH's axis extent/shading, and warns on mismatch. This is safe:
each trial's own real duration (baked into ``psth_trial_durations``) still
bounds what it contributes to the at-risk normalisation regardless of the
axis, so a shorter-window session's trials aren't corrupted by the wider
axis — they just correctly show zero at-risk past their own window.
``congruency_window`` (the fixed post-target latency band for the
single-number "fraction toward target" summary) and ``reward_angle`` (the
acceptance-zone half-width shading the polar plot's reward zone — read from
each session's manifest, never a hardcoded default) are each a single value
applied once, not something a "widest across sessions" choice makes sense
for; the first session's value is used for both, and a mismatch warns.

Function reference
-------------------
:func:`pool_animal_sessions`
    Pools a list of per-session :func:`prosaccade_session.main` result
    dicts (typically one animal's sessions) into a single dict of the same
    shape: concatenates per-trial latency/congruency, pre-cue, PSTH, and
    polar-angle data, and resolves
    ``reward_window``/``congruency_window``/``reward_angle`` per the rules
    above.

:func:`analyze_all_sessions`
    Runs :func:`prosaccade_session.main` on every session matching
    ``experiment_type``/``animal_name`` from the manifest, grouping results
    by animal along the way. Returns the aggregated per-saccade table
    (``aggregated``, which feeds the CSV export in ``__main__``);
    ``animal_pooled``, a dict mapping each animal name to that animal's
    sessions pooled via :func:`pool_animal_sessions`, ready for the
    population plots; ``session_validity``, a list of per-session
    validity/accuracy/congruency stats (see :func:`plot_session_validity_summary`);
    and ``session_results``, a dict mapping every processed session's ID to
    its full :func:`prosaccade_session.main` result, so that data doesn't
    need to be recomputed by anything downstream (see
    :func:`save_population_cache`).

:func:`save_population_cache`
    Pickles ``session_results``/``animal_pooled``/``session_validity`` to
    ``<results_dir>/<experiment_type>_population_cache_<scope>.pkl``
    (``<scope>`` is the single animal's name for a filtered run, or
    ``all_animals`` otherwise), so other scripts — e.g. a composite figure
    script pulling one session's data plus the population summary — can
    build plots from this run without re-analyzing every session in the
    manifest themselves.

:func:`plot_population_summary`
    Draws the three population plots — target-aligned PSTH/congruency,
    latency-by-outcome, and left/right polar — for one pooled trial set
    (either one animal's pooled sessions, or several animals' pooled dicts
    pooled again for the all-animals-combined figure), by recomputing the
    session-level summary statistics from the pooled per-trial data and
    calling the same plotting functions :func:`prosaccade_session.main`
    uses per session, plus :func:`eyehead.analysis.plot_left_right_angle`
    (shaded using ``pooled``'s resolved ``reward_angle``, not a hardcoded
    default).

:func:`run_population_summary_plots`
    Driver: calls :func:`plot_population_summary` once per animal in
    ``animal_pooled``, then — only when more than one animal is present, to
    avoid a redundant duplicate of a single animal's own plots — once more
    for every animal pooled together via :func:`pool_animal_sessions`.

:func:`plot_session_validity_summary`
    Three-panel dot plot, grouped by animal: fraction of trials with a
    detected saccade, fraction of those that were correct, and windowed
    congruency vs. pre-cue control (reusing
    :func:`prosaccade_session.congruency_in_window`) — one green dot per
    session plus each animal's trial-weighted pooled average ± 95% Wilson
    CI in black. Filename includes the animal(s) plotted (single animal
    name, or ``all_animals``) so filtered and full runs never overwrite
    each other's output.

``__main__``
    Parses ``--experiment-type``/``--animal-name``/``--quiet-session-plots``,
    calls :func:`analyze_all_sessions`, writes the aggregated per-saccade
    table to CSV, calls :func:`save_population_cache`, then calls
    :func:`run_population_summary_plots` and
    :func:`plot_session_validity_summary` for the per-animal (+
    all-animals) summary figures.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt

from analysis import prosaccade_session
from analysis.prosaccade_session import main
from utils.session_loader import load_session, list_sessions_from_manifest
from eyehead.analysis import plot_left_right_angle

assert main is prosaccade_session.main

def pool_animal_sessions(results: list[dict]) -> dict:
    """Pool one animal's per-session :func:`prosaccade_session.main` results.

    Concatenates each session's per-trial latency/congruency, pre-cue
    saccades, PSTH per-trial ingredients, and polar-plot angles across every
    session in ``results``, so the three per-session population plots can be
    drawn once over all of an animal's pooled trials.

    ``reward_window`` is taken as the max across sessions (for axis extent /
    shading only — each trial's own duration, baked into
    ``psth_trial_durations``, still bounds what it contributes, so a
    shorter-window session isn't corrupted by the wider pooled axis); a
    warning is printed if sessions disagree. ``congruency_window`` is taken
    from the first session seen, since pooled latencies are compared against
    a single fixed window; a warning is printed if a later session's differs.

    Parameters
    ----------
    results : list of dict
        Each element is one session's return value from
        :func:`prosaccade_session.main`.

    Returns
    -------
    dict
        Same shape as one session's ``latency_outcome`` plus
        ``precue_latencies``/``precue_congruent``, ``left_angle``/
        ``right_angle``, ``psth_trial_rel_times``/``psth_trial_durations``,
        the resolved ``reward_window``/``congruency_window``, and
        ``n_sessions``.
    """
    latencies = np.concatenate([r["latency_outcome"]["latencies"] for r in results])
    congruent = np.concatenate([r["latency_outcome"]["congruent"] for r in results])
    n_no_saccade = sum(r["latency_outcome"]["n_no_saccade"] for r in results)
    n_total = sum(r["latency_outcome"]["n_total"] for r in results)

    precue_latencies = np.concatenate([r["precue_latencies"] for r in results])
    precue_congruent = np.concatenate([r["precue_congruent"] for r in results])

    left_angle = np.concatenate([r["left_angle"] for r in results])
    right_angle = np.concatenate([r["right_angle"] for r in results])

    psth_trial_rel_times = [t for r in results for t in r["psth_trial_rel_times"]]
    psth_trial_durations = np.concatenate([r["psth_trial_durations"] for r in results])

    reward_windows = sorted({r["reward_window"] for r in results})
    reward_window = max(reward_windows)
    if len(reward_windows) > 1:
        warnings.warn(
            f"Pooling sessions with different reward_window values {reward_windows}; "
            f"using the max ({reward_window}) for the population PSTH axis. Each "
            "trial's own duration still bounds what it contributes, so this only "
            "affects axis extent/shading."
        )

    congruency_windows = [tuple(r["congruency_window"]) for r in results]
    congruency_window = congruency_windows[0]
    if len(set(congruency_windows)) > 1:
        warnings.warn(
            f"Pooling sessions with different congruency_window values "
            f"{sorted(set(congruency_windows))}; using the first session's "
            f"({congruency_window}) for the pooled congruency summary."
        )

    reward_angles = [r["reward_angle"] for r in results]
    reward_angle = reward_angles[0]
    if len(set(reward_angles)) > 1:
        warnings.warn(
            f"Pooling sessions with different reward_angle values "
            f"{sorted(set(reward_angles))}; using the first session's "
            f"({reward_angle}) for the pooled polar reward-zone shading."
        )


    return {
        "latency_outcome": {
            "latencies": latencies,
            "congruent": congruent,
            "n_no_saccade": n_no_saccade,
            "n_total": n_total,
        },
        "precue_latencies": precue_latencies,
        "precue_congruent": precue_congruent,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "psth_trial_rel_times": psth_trial_rel_times,
        "psth_trial_durations": psth_trial_durations,
        "reward_window": reward_window,
        "congruency_window": congruency_window,
        "reward_angle": reward_angle,
        "n_sessions": len(results),
    }

def analyze_all_sessions(
    experiment_type: str | None = "prosaccade",
    animal_name: str | None = None,
    show_session_plots: bool = True,
):
    """Run prosaccade analysis on sessions that match the provided filters.

    Parameters
    ----------
    experiment_type:
        Experiment type used to select sessions from the manifest. When ``None``
        all experiment types are considered.
    animal_name:
        Optional animal name used to further restrict the manifest lookup.
    show_session_plots:
        Passed straight through to :func:`prosaccade_session.main` as
        ``show_plots`` for every session. Defaults to ``True`` (unchanged
        behaviour — every session's figures pop up as before). Set to
        ``False`` to still generate and save every per-session figure as
        normal but skip the interactive pop-up for all of them, e.g. when
        batch-processing many sessions and only the population-level
        figures should actually be shown.

    Returns
    -------
    tuple
        ``(aggregated, left_angle_all, right_angle_all, processed_animals,
        animal_pooled, session_validity, session_results)`` — the aggregated
        per-saccade session table, lists of left/right eye angles, the set
        of unique animal names processed, ``animal_pooled`` (a dict mapping
        each animal name to that animal's sessions pooled via
        :func:`pool_animal_sessions`, ready for the population plots),
        ``session_validity``: a list with one dict per session —
        ``{"animal_name", "session_id", "n_total", "n_valid",
        "fraction_valid", "fraction_correct", "window_frac", "window_n",
        "precue_frac", "precue_n"}`` — where ``fraction_valid`` is the
        fraction of trials with a detected first saccade in the window at
        all, ``fraction_correct`` is, of those valid trials, the fraction
        that were congruent (within ``reward_angle`` and ``reward_window``
        — see :func:`prosaccade_session.analyze_latency_by_outcome`), and
        ``window_frac``/``window_n`` are that session's own
        :func:`prosaccade_session.congruency_in_window` result (fraction
        congruent restricted to that session's configured
        ``congruency_window``) with ``precue_frac``/``precue_n`` the
        matching pre-cue control fraction — the same numbers plotted in the
        third panel of the per-session PSTH/congruency figure; and
        ``session_results``: a dict mapping every processed session's ID to
        its full :func:`prosaccade_session.main` result dict, so callers
        (e.g. :func:`save_population_cache`) can reuse any one session's
        already-computed data without re-running it.
    """

    tables: list[pd.DataFrame] = []
    left_angle_all = []
    right_angle_all = []
    processed_animals: set[str] = set()
    animal_session_results: dict[str, list[dict]] = defaultdict(list)
    session_validity: list[dict] = []
    session_results: dict[str, dict] = {}

    for session_id in list_sessions_from_manifest(
        experiment_type,
        match_prefix=True,
        animal_name=animal_name,
    ):
        result = prosaccade_session.main(session_id, show_plots=show_session_plots)
        session_results[session_id] = result
        session_cfg = load_session(session_id)

        session_df = result["df"].copy()
        session_df["animal_name"] = session_cfg.animal_name
        if session_cfg.animal_name:
            processed_animals.add(session_cfg.animal_name)
            animal_session_results[session_cfg.animal_name].append(result)

            lo = result["latency_outcome"]
            n_valid = lo["n_total"] - lo["n_no_saccade"]
            fraction_valid = n_valid / lo["n_total"] if lo["n_total"] else float("nan")
            fraction_correct = (
                float(np.mean(lo["congruent"])) if len(lo["congruent"]) else float("nan")
            )
            window_frac, window_n, _, _ = prosaccade_session.congruency_in_window(
                lo["latencies"], lo["congruent"], window=result["congruency_window"]
            )
            session_precue_congruent = result["precue_congruent"]
            precue_frac = (
                float(np.mean(session_precue_congruent))
                if len(session_precue_congruent) else float("nan")
            )
            precue_n = len(session_precue_congruent)
            session_validity.append({
                "animal_name": session_cfg.animal_name,
                "session_id": session_id,
                "n_total": lo["n_total"],
                "n_valid": n_valid,
                "fraction_valid": fraction_valid,
                "fraction_correct": fraction_correct,
                "window_frac": window_frac,
                "window_n": window_n,
                "precue_frac": precue_frac,
                "precue_n": precue_n,
            })

        tables.append(session_df)
        left_angle_all.append(result["left_angle"])
        right_angle_all.append(result["right_angle"])

    animal_pooled = {
        animal: pool_animal_sessions(results)
        for animal, results in animal_session_results.items()
    }

    if not tables:
        return (
            pd.DataFrame(),
            left_angle_all,
            right_angle_all,
            processed_animals,
            animal_pooled,
            session_validity,
            session_results,
        )

    return (
        pd.concat(tables, ignore_index=True),
        left_angle_all,
        right_angle_all,
        processed_animals,
        animal_pooled,
        session_validity,
        session_results,
    )

def save_population_cache(
    session_results: dict[str, dict],
    animal_pooled: dict[str, dict],
    session_validity: list[dict],
    results_dir: Path,
    experiment_type: str = "prosaccade",
) -> Path:
    """Pickle everything :func:`analyze_all_sessions` computed for this run,
    so other scripts (e.g. a composite figure script) can build plots from
    it without re-analyzing every session in the manifest.

    Saves ``{"session_results", "animal_pooled", "session_validity"}`` to
    ``<results_dir>/<experiment_type>_population_cache_<scope>.pkl``, where
    ``<scope>`` is the single animal's name when ``animal_pooled`` has one
    entry (e.g. a ``--animal-name``-filtered run), or ``all_animals``
    otherwise — mirroring :func:`plot_session_validity_summary`'s filename
    convention, so a filtered run's cache can never be silently mistaken
    for (or overwrite) a full, all-animals run's cache. Returns the path
    written.
    """
    animals_sorted = sorted(animal_pooled.keys())
    if len(animals_sorted) == 1:
        scope_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", animals_sorted[0]).strip("_") or "unknown"
    else:
        scope_tag = "all_animals"

    cache_path = results_dir / f"{experiment_type}_population_cache_{scope_tag}.pkl"
    payload = {
        "session_results": session_results,
        "animal_pooled": animal_pooled,
        "session_validity": session_validity,
    }
    with cache_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return cache_path

def plot_population_summary(
    pooled: dict,
    title: str,
    save_stem: str,
    results_dir: Path,
    experiment_type: str = "prosaccade",
    animal_name: str | None = None,
) -> None:
    """Draw the three population plots for one pooled trial set.

    ``pooled`` is the output of :func:`pool_animal_sessions` (either one
    animal's pooled sessions, or several animals' pooled dicts pooled again
    for an all-animals-combined figure). Recomputes the target-aligned PSTH,
    accuracy-vs-latency, and windowed-congruency-vs-pre-cue summaries from
    the pooled per-trial data (via
    :func:`prosaccade_session.psth_rate_from_trials`,
    :func:`prosaccade_session.fraction_toward_target_by_latency`, and
    :func:`prosaccade_session.congruency_in_window`), then draws the same
    three figures :func:`prosaccade_session.main` draws per session:
    :func:`prosaccade_session.plot_psth_and_congruency`,
    :func:`prosaccade_session.plot_latency_by_outcome`, and
    :func:`eyehead.analysis.plot_left_right_angle`.

    ``title`` is used as the figure title/suptitle; ``save_stem`` is the
    filename prefix (e.g. the animal name, or ``"all_animals"``) under
    ``results_dir``.
    """

    reward_window = pooled["reward_window"]
    reward_angle = pooled["reward_angle"]
    congruency_window = pooled["congruency_window"]
    latency_outcome = pooled["latency_outcome"]
    latencies = latency_outcome["latencies"]
    congruent = latency_outcome["congruent"]

    psth_centers, psth_rate, psth_ci, n_trials_psth = prosaccade_session.psth_rate_from_trials(
        pooled["psth_trial_rel_times"], pooled["psth_trial_durations"],
        window=(-reward_window, reward_window),
    )
    latency_centers, fraction_toward, n_per_window = prosaccade_session.fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0, reward_window)
    )
    frac, n_window, ci_lo, ci_hi = prosaccade_session.congruency_in_window(
        latencies, congruent, window=congruency_window
    )
    precue_congruent = pooled["precue_congruent"]
    precue_frac = float(np.mean(precue_congruent)) if len(precue_congruent) else np.nan
    precue_n = len(precue_congruent)

    plot_left_right_angle(
    pooled["left_angle"],
    pooled["right_angle"],
    reward_angle,
    sessionname=save_stem,
    resultdir=results_dir,
    experiment_type=experiment_type,
    animal_name=animal_name,
    )

    prosaccade_session.plot_psth_and_congruency(
        psth_centers, psth_rate, psth_ci, n_trials_psth,
        latency_centers, fraction_toward, n_per_window,
        frac, n_window, ci_lo, ci_hi, precue_frac, precue_n,
        title=title,
        save_path=results_dir / f"{save_stem}_psth_congruency.png",
        window=congruency_window, reward_window=reward_window,
    )

    prosaccade_session.plot_latency_by_outcome(
        latency_outcome,
        title=title,
        save_path=results_dir / f"{save_stem}_latency_by_outcome.png",
        reward_window=reward_window,
    )


def plot_session_validity_summary(
    session_validity: list[dict],
    animal_pooled: dict[str, dict],
    results_dir: Path,
    experiment_type: str = "prosaccade",
) -> None:
    """Three-panel dot plot: fraction of valid trials, fraction of those
    valid trials that were correct, and windowed congruency vs. pre-cue
    control — one dot per session grouped by animal, with each animal's
    trial-weighted pooled average (+ 95% Wilson CI) overlaid.

    "Valid" means a first saccade was detected in the reward window at all;
    "correct" means, of valid trials, the fraction whose saccade was
    congruent (within ``reward_angle`` and ``reward_window`` — see
    :func:`prosaccade_session.analyze_latency_by_outcome`). The third panel
    reuses the same numbers as the third panel of the per-session
    PSTH/congruency figure (:func:`prosaccade_session.congruency_in_window`
    restricted to each session's/animal's configured ``congruency_window``,
    plus the pre-cue control fraction) rather than recomputing anything new
    — it is NOT the same quantity as the second panel, since it further
    restricts to the (typically narrower) ``congruency_window`` instead of
    every valid trial regardless of latency. All three are taken directly
    from ``session_validity`` (see :func:`analyze_all_sessions`). Each
    animal's average is the trial-weighted pooled fraction computed from
    ``animal_pooled[animal]`` — consistent with how the rest of this script
    pools trials — not a simple mean of per-session fractions, so a session
    with more trials counts for more; its error bar is the 95% Wilson score
    interval, same CI method used elsewhere in this codebase for
    fraction-with-CI summaries.

    The saved filename includes the animal(s) actually plotted (the single
    animal's name when ``animal_pooled`` has one entry, e.g. from a
    ``--animal-name``-filtered run, or ``all_animals`` otherwise) so that
    per-animal and combined runs don't overwrite each other's output.
    """
    animals_sorted = sorted(animal_pooled.keys())
    if not animals_sorted:
        warnings.warn(
            "No animals to plot in plot_session_validity_summary; skipping."
        )
        return

    fig, (ax_valid, ax_correct, ax_window) = plt.subplots(
        1, 3, figsize=(max(9, 3.3 * len(animals_sorted)), 5)
    )
    rng = np.random.default_rng(0)

    for i, animal in enumerate(animals_sorted):
        sessions = [r for r in session_validity if r["animal_name"] == animal]
        jitter = rng.uniform(-0.15, 0.15, size=len(sessions))
        xs = np.full(len(sessions), i, dtype=float) + jitter
        valid_ys = [r["fraction_valid"] for r in sessions]
        correct_ys = [r["fraction_correct"] for r in sessions]
        window_ys = [r["window_frac"] for r in sessions]

        ax_valid.scatter(xs, valid_ys, color="tab:green", alpha=0.6, s=40, zorder=2,
                          label="session" if i == 0 else None)
        ax_correct.scatter(xs, correct_ys, color="tab:green", alpha=0.6, s=40, zorder=2,
                            label="session" if i == 0 else None)
        ax_window.scatter(xs, window_ys, color="tab:green", alpha=0.6, s=40, zorder=2,
                           label="session" if i == 0 else None)

        pooled = animal_pooled[animal]
        pooled_lo = pooled["latency_outcome"]
        pooled_n_total = pooled_lo["n_total"]
        pooled_n_valid = pooled_n_total - pooled_lo["n_no_saccade"]
        pooled_congruent = pooled_lo["congruent"]
        pooled_n_correct = int(np.sum(pooled_congruent))
        pooled_frac_valid = pooled_n_valid / pooled_n_total if pooled_n_total else np.nan
        pooled_frac_correct = (
            float(np.mean(pooled_congruent)) if len(pooled_congruent) else np.nan
        )

        valid_ci_lo, valid_ci_hi = prosaccade_session.wilson_ci(pooled_n_valid, pooled_n_total)
        correct_ci_lo, correct_ci_hi = prosaccade_session.wilson_ci(
            pooled_n_correct, len(pooled_congruent)
        )

        pooled_window_frac, pooled_window_n, window_ci_lo, window_ci_hi = (
            prosaccade_session.congruency_in_window(
                pooled_lo["latencies"], pooled_congruent, window=pooled["congruency_window"]
            )
        )
        pooled_precue_congruent = pooled["precue_congruent"]
        pooled_precue_frac = (
            float(np.mean(pooled_precue_congruent)) if len(pooled_precue_congruent) else np.nan
        )

        ax_valid.errorbar(
            [i], [pooled_frac_valid],
            yerr=[[max(0.0, pooled_frac_valid - valid_ci_lo)],
                  [max(0.0, valid_ci_hi - pooled_frac_valid)]],
            fmt="o", color="black", ecolor="black", capsize=4, ms=10, zorder=3,
            label="animal average (trial-weighted, 95% CI)" if i == 0 else None,
        )
        ax_correct.errorbar(
            [i], [pooled_frac_correct],
            yerr=[[max(0.0, pooled_frac_correct - correct_ci_lo)],
                  [max(0.0, correct_ci_hi - pooled_frac_correct)]],
            fmt="o", color="black", ecolor="black", capsize=4, ms=10, zorder=3,
            label="animal average (trial-weighted, 95% CI)" if i == 0 else None,
        )
        ax_window.errorbar(
            [i], [pooled_window_frac],
            yerr=[[max(0.0, pooled_window_frac - window_ci_lo)],
                  [max(0.0, window_ci_hi - pooled_window_frac)]],
            fmt="o", color="black", ecolor="black", capsize=4, ms=10, zorder=3,
            label="animal average (trial-weighted, 95% CI)" if i == 0 else None,
        )
        ax_window.plot(
            i, pooled_precue_frac, "o", mfc="none", mec="gray", ms=9, zorder=3,
            label="pre-cue control (pooled)" if i == 0 else None,
        )

    for ax, ylabel, title in (
        (ax_valid, "Fraction of trials with a detected saccade", "Trial validity"),
        (ax_correct, "Fraction of valid trials that were correct", "Trial accuracy (of valid trials)"),
        (ax_window, "Fraction toward target (congruency window)", "Congruency vs. pre-cue control"),
    ):
        ax.set_xticks(range(len(animals_sorted)))
        ax.set_xticklabels(animals_sorted, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(f"{experiment_type} — session validity/accuracy by animal", fontsize=12, wrap=True)
    fig.tight_layout()

    if len(animals_sorted) == 1:
        scope_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", animals_sorted[0]).strip("_") or "unknown"
    else:
        scope_tag = "all_animals"
    fname_stem = f"{experiment_type}_session_validity_summary_{scope_tag}"

    fig.savefig(results_dir / f"{fname_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(results_dir / f"{fname_stem}.svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def run_population_summary_plots(
    animal_pooled: dict[str, dict],
    results_dir: Path,
    experiment_type: str = "prosaccade",
) -> None:
    """Draw the three population plots for every animal, plus one combined set.

    Calls :func:`plot_population_summary` once per animal in
    ``animal_pooled``, then — only when more than one animal is present, to
    avoid a redundant duplicate of a single animal's own plots — once more
    for every animal pooled together (via :func:`pool_animal_sessions`
    applied to the already-pooled per-animal dicts, which share the same
    shape as a raw per-session result).
    """
    animal_names_sorted = sorted(animal_pooled.keys())
    if not animal_names_sorted:
        warnings.warn(
            f"No animals with pooled sessions found for experiment_type="
            f"{experiment_type!r}; skipping population summary plots."
        )
        return

    for animal in animal_names_sorted:
        pooled = animal_pooled[animal]
        safe_animal = re.sub(r"[^A-Za-z0-9_-]+", "_", animal).strip("_") or "unknown"
        plot_population_summary(
            pooled,
            title=f"{animal} — population ({pooled['n_sessions']} sessions)",
            save_stem=f"{experiment_type}_population_{safe_animal}",
            results_dir=results_dir,
            experiment_type=experiment_type,
            animal_name=animal,
        )

    if len(animal_pooled) > 1:
        combined_pooled = pool_animal_sessions(list(animal_pooled.values()))
        total_sessions = sum(p["n_sessions"] for p in animal_pooled.values())
        plot_population_summary(
            combined_pooled,
            title=f"All animals — population ({total_sessions} sessions across {len(animal_pooled)} animals)",
            save_stem=f"{experiment_type}_population_all_animals",
            results_dir=results_dir,
            experiment_type=experiment_type,
            animal_name=", ".join(animal_names_sorted),
        )
    else:
        print(
            "Only one animal in this run; skipping the all-animals-combined "
            "plot (it would just duplicate that animal's own)."
        )


# Usage: python Python/analysis/prosaccade_population.py --animal-name Paris will print all Paris session figures too
# Usage: python Python/analysis/prosaccade_population.py --animal-name Paris --quiet-session-plots will print only population
# Usage: python Python/analysis/prosaccade_population.py --animal-name --quiet-session-plots will run each session, for each animal, and for all

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run analysis across sessions filtered by experiment type",
    )
    parser.add_argument(
        "--experiment-type",
        default="prosaccade",
        help="Experiment type to process",
    )

    parser.add_argument(
        "--animal-name",
        default=None,
        help="Optional animal name to filter sessions",
    )
    parser.add_argument(
        "--quiet-session-plots",
        action="store_true",
        help=(
            "Still generate and save every per-session figure as normal, "
            "but don't pop them up interactively (no windows to close "
            "while batch-processing many sessions). Population-level "
            "figures always pop up regardless. Default: session plots do "
            "pop up, same as before."
        ),
    )
    args = parser.parse_args()
    (
        aggregated,
        left_angle_all,
        right_angle_all,
        processed_animals,
        animal_pooled,
        session_validity,
        session_results,
    ) = analyze_all_sessions(
        args.experiment_type,
        animal_name=args.animal_name,
        show_session_plots=not args.quiet_session_plots,
    )
    root_dir = Path(__file__).resolve().parents[2]

    manifest_path = root_dir / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    results_root = Path(manifest.get("results_root") or root_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    aggregated.to_csv(
        results_root / f"{args.experiment_type}_population_results.csv", index=False
    )

    ### Cache everything this run computed (per-session results, per-animal
    ### pooled data, session validity) so other scripts (e.g. a composite
    ### figure script) can build plots from it without re-analyzing every
    ### session in the manifest.
    cache_path = save_population_cache(
        session_results, animal_pooled, session_validity, results_root,
        experiment_type=args.experiment_type,
    )
    print(f"Saved population cache to {cache_path}")

    ### Per-animal (+ all-animals-combined) latency-by-outcome, psth/congruency,
    ### and left/right polar population plots.
    run_population_summary_plots(
        animal_pooled, results_root, experiment_type=args.experiment_type,
    )

    ### Per-session trial validity/accuracy, grouped by animal, with each
    ### animal's trial-weighted average overlaid.
    plot_session_validity_summary(
        session_validity, animal_pooled, results_root, experiment_type=args.experiment_type,
    )