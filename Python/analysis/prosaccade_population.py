"""Run prosaccade analysis across multiple sessions.

This script selects sessions from ``session_manifest.yml`` based on the
requested experiment type and executes the full prosaccade analysis pipeline
for each one.
"""
from __future__ import annotations

import argparse
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
        "n_sessions": len(results),
    }

def analyze_all_sessions(
    experiment_type: str | None = "prosaccade",
    animal_name: str | None = None,
):
    """Run prosaccade analysis on sessions that match the provided filters.

    Parameters
    ----------
    experiment_type:
        Experiment type used to select sessions from the manifest. When ``None``
        all experiment types are considered.
    animal_name:
        Optional animal name used to further restrict the manifest lookup.

    Returns
    -------
    tuple
        ``(aggregated, left_angle_all, right_angle_all,
        left_angle_all_with_dates, right_angle_all_with_dates,
        processed_animals, animal_pooled)`` — the aggregated per-saccade
        session table, lists of left/right eye angles, dictionaries keyed by
        session date, the set of unique animal names processed, and
        ``animal_pooled``: a dict mapping each animal name to that animal's
        sessions pooled via :func:`pool_animal_sessions`, ready for the
        population plots.
    """

    tables: list[pd.DataFrame] = []
    left_angle_all = []
    right_angle_all = []
    left_angle_all_with_dates = {}
    right_angle_all_with_dates = {}
    processed_animals: set[str] = set()
    animal_session_results: dict[str, list[dict]] = defaultdict(list)

    for session_id in list_sessions_from_manifest(
        experiment_type,
        match_prefix=True,
        animal_name=animal_name,
    ):
        result = prosaccade_session.main(session_id)
        session_cfg = load_session(session_id)

        session_df = result["df"].copy()
        session_df["animal_name"] = session_cfg.animal_name
        if session_cfg.animal_name:
            processed_animals.add(session_cfg.animal_name)
            animal_session_results[session_cfg.animal_name].append(result)

        tables.append(session_df)
        left_angle_all.append(result["left_angle"])
        right_angle_all.append(result["right_angle"])
        date_str = (
            session_df["session_date"].iloc[0]
            if "session_date" in session_df.columns
            else "unknown_date"
        )
        left_angle_all_with_dates[date_str] = result["left_angle"]
        right_angle_all_with_dates[date_str] = result["right_angle"]

    animal_pooled = {
        animal: pool_animal_sessions(results)
        for animal, results in animal_session_results.items()
    }

    if not tables:
        return (
            pd.DataFrame(),
            left_angle_all,
            right_angle_all,
            left_angle_all_with_dates,
            right_angle_all_with_dates,
            processed_animals,
            animal_pooled,
        )

    return (
        pd.concat(tables, ignore_index=True),
        left_angle_all,
        right_angle_all,
        left_angle_all_with_dates,
        right_angle_all_with_dates,
        processed_animals,
        animal_pooled,
    )

def plot_population_summary(
    pooled: dict,
    title: str,
    save_stem: str,
    results_dir: Path,
    experiment_type: str = "prosaccade",
    animal_name: str | None = None,
    reward_angle: float = 35.0,
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
    congruency_window = pooled["congruency_window"]
    latency_outcome = pooled["latency_outcome"]
    latencies = latency_outcome["latencies"]
    congruent = latency_outcome["congruent"]

    psth_centers, psth_rate, psth_ci, n_trials_psth = prosaccade_session.psth_rate_from_trials(
        pooled["psth_trial_rel_times"], pooled["psth_trial_durations"],
        window=(-reward_window, reward_window),
    )
    latency_centers, fraction_toward, n_per_window = prosaccade_session.fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0.2, reward_window)
    )
    frac, n_window, ci_lo, ci_hi = prosaccade_session.congruency_in_window(
        latencies, congruent, window=congruency_window
    )
    precue_congruent = pooled["precue_congruent"]
    precue_frac = float(np.mean(precue_congruent)) if len(precue_congruent) else np.nan
    precue_n = len(precue_congruent)

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


def plot_prosaccade_trends_from_dictionary(
    left_angle_dict: dict,
    right_angle_dict: dict,
    experiment_type: str = "prosaccade",
    animal_label: str | None = None,
) -> None:

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # First sort the dictionary by date
    sorted_dates = sorted(left_angle_dict.keys())
    left_angles_sorted = [left_angle_dict[date] for date in sorted_dates]
    right_angles_sorted = [right_angle_dict[date] for date in sorted_dates]
    saccade_percentage_left_list = []
    saccade_percentage_right_list = []
    reward_angle = 35  # degrees
    for i, date in enumerate(sorted_dates):
        left_angles = left_angle_dict[date]
        right_angles = right_angle_dict[date]
        if experiment_type == "prosaccade":
            saccade_percentage_left = np.sum(np.abs(left_angles) <= np.deg2rad(reward_angle)) / len(left_angles) * 100
            saccade_percentage_right = np.sum(np.abs(right_angles) >= np.deg2rad(180-reward_angle)) / len(right_angles) * 100
        elif experiment_type == "antisaccade":
            saccade_percentage_left = np.sum(np.abs(left_angles) >= np.deg2rad(180-reward_angle)) / len(left_angles) * 100
            saccade_percentage_right = np.sum(np.abs(right_angles) <= np.deg2rad(reward_angle)) / len(right_angles) * 100

       # Plot the saccade percentages
        saccade_percentage_left_list.append(saccade_percentage_left)
        saccade_percentage_right_list.append(saccade_percentage_right)

    # Plot the sacccade percentage across sessions
    ax.plot(range(len(sorted_dates)), saccade_percentage_left_list, marker='o', color='b', label='Left Eye')
    ax.plot(range(len(sorted_dates)), saccade_percentage_right_list, marker='o', color='r', label='Right Eye')
    # Set plot labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Saccade Percentage (%)")
    title = f"{experiment_type} Saccade Percentages Over Time"
    label_text = (str(animal_label).strip() if animal_label is not None else "")
    animal_suffix = ""
    if label_text:
        title = f"{title} – {label_text}"
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label_text).strip("_")
        if safe_label:
            animal_suffix = f"_{safe_label}"
    ax.set_xticks(range(len(sorted_dates)))
    ax.set_xticklabels(sorted_dates, rotation=45)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Save the plot
    fig.savefig(results_root / f"{experiment_type}_saccade_percentage_trends{animal_suffix}.png")
    fig.savefig(results_root / f"{experiment_type}_saccade_percentage_trends{animal_suffix}.svg")


# Usage: python Python/analysis/prosaccade_population.py --animal-name Paris
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
    args = parser.parse_args()

    (
        aggregated,
        left_angle_all,
        right_angle_all,
        left_angle_all_with_dates,
        right_angle_all_with_dates,
        processed_animals,
        animal_pooled,
    ) = analyze_all_sessions(
        args.experiment_type,
        animal_name=args.animal_name,
    )
    root_dir = Path(__file__).resolve().parents[2]
        
    manifest_path = root_dir / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    results_root = Path(manifest.get("results_root") or root_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    ### Plot the left right angle results
    from eyehead.analysis import plot_left_right_angle
    left_angle_all = np.concatenate(left_angle_all)
    right_angle_all = np.concatenate(right_angle_all)
    animal_label = None
    if isinstance(aggregated, pd.DataFrame) and not aggregated.empty and "session_id" in aggregated:
        session_ids = aggregated["session_id"].dropna().unique()
        animal_names: list[str] = []
        for session_id in session_ids:
            try:
                session_cfg = load_session(session_id)
            except KeyError:
                continue
            if session_cfg.animal_name:
                animal_names.append(session_cfg.animal_name)
        if animal_names:
            # Preserve manifest order while removing duplicates
            unique_animals = list(dict.fromkeys(animal_names))
            animal_label = ", ".join(unique_animals)

    plot_left_right_angle(
        left_angle_all,
        right_angle_all,
        35,
        sessionname="population",
        resultdir=results_root,
        experiment_type=args.experiment_type,
        animal_name=animal_label,
    )
    plot_prosaccade_trends_from_dictionary(
        left_angle_all_with_dates,
        right_angle_all_with_dates,
        experiment_type=args.experiment_type,
        animal_label=animal_label,
    )
    aggregated.to_csv(
        results_root / f"{args.experiment_type}_population_results.csv", index=False
    )

    ### Per-animal (+ all-animals-combined) latency-by-outcome, psth/congruency,
    ### and left/right polar population plots.
    run_population_summary_plots(
        animal_pooled, results_root, experiment_type=args.experiment_type,
    )
