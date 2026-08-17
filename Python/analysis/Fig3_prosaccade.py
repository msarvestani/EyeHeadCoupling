"""Standalone script: one composite multi-panel summary figure for the
prosaccade task, combining a single session's arrow/polar/latency plots
with the all-animals population summary.

Layout
------
- Panel A: blank placeholder for a hand-drawn task schematic (added
  separately, e.g. in Illustrator/PowerPoint).
- Panel B: one session's Left/Right arrow (quiver) plots on top, Left/Right
  polar saccade-angle histograms below.
- Panel C: that same session's latency-by-outcome histogram, latency CDF,
  and saccade-accuracy-vs-latency curve.
- Panel D: the population (every animal in the manifest, pooled together)
  latency-by-outcome histogram, accuracy-vs-latency curve, and the
  3-column session validity/accuracy summary (trial validity, trial
  accuracy, congruency vs. pre-cue control), grouped by animal.

Why this doesn't reuse the existing plotting functions wholesale
------------------------------------------------------------------
:func:`prosaccade_session.plot_latency_by_outcome`,
:func:`prosaccade_session.plot_psth_and_congruency`,
:func:`eyehead.analysis.plot_left_right_angle`, and
:func:`prosaccade_population.plot_session_validity_summary` each build and
save their own standalone figure, with their full annotation set (legends,
percentage-text, multi-line titles) — appropriate for a single-purpose
figure, too busy at this figure's smaller per-panel size. Rather than
change those (used throughout the rest of the pipeline), this script draws
deliberately simplified/condensed versions of the same panels (limited tick
counts, shared font sizing, no legends/percentage-text/multi-line titles)
sized to fit one composite page.

To keep those simplified panels from silently drifting out of sync with
the real ones, this script reuses as much of the actual pipeline as
possible rather than recomputing anything by hand:
- The DATA for every panel comes directly from the real pipeline entry
  points — :func:`prosaccade_session.main` (panels B/C) and
  :func:`prosaccade_population.analyze_all_sessions` /
  :func:`pool_animal_sessions` (panel D) — the exact same dicts the
  production plots consume, not a re-derivation.
- Every *statistic* drawn is computed with the same functions the real
  plots call internally: :func:`prosaccade_session.fraction_toward_target_by_latency`,
  :func:`prosaccade_session.congruency_in_window`,
  :func:`prosaccade_session.wilson_ci`.
- Panel B's arrows are drawn with the actual
  :func:`eyehead.analysis._draw_quiver_arrows` helper (the same one
  :func:`eyehead.analysis.sort_saccades` uses for its own per-condition
  figure) — only the per-trial arrow vectors feeding it are gathered
  independently (below), never the drawing itself.

The one place this script does re-derive data rather than read it from an
existing return value: Panel B's arrow-plot vectors (per-trial saccade
position/direction/congruency, split Left/Right) aren't returned by
:func:`prosaccade_session.main`, so :func:`_load_session_quiver_data`
re-runs just the detection/first-saccade-scoring steps independently,
mirroring ``main()``'s orchestration (masking, mean-centering) to get them.
If ``main()``'s own orchestration ever changes upstream of scoring, this
function needs updating to match — everything else in this script reads
data ``main()``/``analyze_all_sessions()`` already computed, so has no such
risk.

Running this script re-analyzes every session in the manifest for
``EXPERIMENT_TYPE`` (same cost as running ``prosaccade_population.py``
itself) plus one extra pass over ``SESSION_ID`` for the arrow-plot data.

Usage
-----
No CLI arguments -- edit ``SESSION_ID``/``EXPERIMENT_TYPE``/``OUTPUT_STEM``
near the top of this file directly, then run:

    python Python/analysis/prosaccade_summary_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import circmean, vonmises
import pickle
import sys
from datetime import datetime
from pathlib import Path

from analysis import prosaccade_session
from analysis import prosaccade_population as pp
from utils.session_loader import load_session_or_path
from eyehead import (
    SaccadeConfig,
    calibrate_eye_position,
    detect_saccades,
    load_session_data,
    organize_stims,
)
from eyehead.analysis import _draw_quiver_arrows

# ---------------------------------------------------------------------------
# Run configuration — edit these directly rather than passing CLI args.
# ---------------------------------------------------------------------------
SESSION_ID = "Tsh001_2025-07-17T15_32_42"  # Paris, 2025-07-17 -- panels B/C
EXPERIMENT_TYPE = "prosaccade"  # pooled across every animal in the manifest -- panel D
OUTPUT_STEM = None  # None -> <results_root>/<EXPERIMENT_TYPE>_summary_figure; or set an explicit Path/str

# ---------------------------------------------------------------------------
# Styling constants — tweak these to fit the final page/journal layout.
# ---------------------------------------------------------------------------
FONT_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans"]
FONT_SIZE_BASE = 8
FONT_SIZE_LABEL = 8
FONT_SIZE_TITLE = 9
FONT_SIZE_TICK = 7
FONT_SIZE_PANEL_LETTER = 13
N_TICKS = 3  # e.g. 0, 25, 50 instead of 0, 10, 20, 30, 40, 50

REWARD_COLOR = "gold"
CONGRUENCY_WINDOW_COLOR = "tab:purple"
CORRECT_COLOR = "tab:green"
INCORRECT_COLOR = "tab:red"


def _limit_ticks(ax, x=True, y=True, nbins=N_TICKS):
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.tick_params(labelsize=FONT_SIZE_TICK)


def _panel_letter(ax, letter):
    ax.text(
        -0.22, 1.15, letter, transform=ax.transAxes,
        fontsize=FONT_SIZE_PANEL_LETTER, fontweight="bold",
        va="bottom", ha="left",
    )


def _strip_box(ax, keep=("left", "bottom")):
    """Remove the axes' bounding-box spines except ``keep``, for a cleaner,
    less boxed-in look. Not applied to polar axes (Panel B's polar plots)
    since their outer circle is the actual radial-extent boundary, not
    decoration."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_population_cache(experiment_type: str) -> dict:
    """Load the ``{"session_results", "animal_pooled", "session_validity"}``
    cache written by :func:`prosaccade_population.save_population_cache`,
    instead of re-analyzing every session in the manifest.

    Always loads the ``all_animals``-scoped cache (i.e. from a full,
    unfiltered ``prosaccade_population.py`` run) since panel D needs every
    animal pooled together — a ``--animal-name``-filtered run's cache is a
    different file and is never picked up here by mistake.

    Raises a clear error, not a cryptic one, if the cache file doesn't
    exist yet.
    """
    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    results_root = Path(manifest.get("results_root") or root_dir)

    cache_path = results_root / f"{experiment_type}_population_cache_all_animals.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No population cache found at {cache_path}. Run "
            f"prosaccade_population.py (without --animal-name, so every "
            f"animal is included) first to generate it."
        )

    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    print(f"Loading population cache from {cache_path} (saved {mtime:%Y-%m-%d %H:%M})")
    with cache_path.open("rb") as fh:
        return pickle.load(fh)

def _load_session_quiver_data(session_id: str) -> dict:
    """Per-trial Left/Right arrow-plot ingredients for one session.

    Mirrors the same detection -> sorting steps
    :func:`prosaccade_session.main` runs (masking non-finite eye positions,
    mean-centering eye position, taking the scored first-saccade index per
    direction), but returns the raw per-trial vectors/labels instead of
    drawing anything — ``main()``'s return dict doesn't expose these.
    """
    config = load_session_or_path(session_id)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    data = load_session_data(config)
    eye_pos_cal = calibrate_eye_position(data, config)
    saccade_cfg = SaccadeConfig(**config.params["saccade_config"])

    saccades, _, _ = detect_saccades(
        eye_pos_cal, data.eye_frame, saccade_cfg, config, data=data, plot=False,
    )
    saccades["stim_frames"], _ = organize_stims(
        data.go_frame, go_dir_x=data.go_direction_x, go_dir_y=data.go_direction_y,
    )

    reward_contingency = config.params.get("reward_contingency") or {}
    reward_window = float(reward_contingency["reward_window"])
    acceptance_angle = float(reward_contingency["reward_angle"])
    scoring_mode = reward_contingency["scoring_mode"]

    first_indices, first_congruent = prosaccade_session.first_saccade_indices_by_direction(
        data, saccades, config, acceptance_angle_deg=acceptance_angle,
        max_latency=reward_window, scoring_mode=scoring_mode,
    )

    eye_pos = saccades["eye_pos"].copy()
    finite = np.isfinite(eye_pos[:, 0]) & np.isfinite(eye_pos[:, 1])
    eye_pos[:, 0] -= np.nanmean(eye_pos[finite, 0])
    eye_pos[:, 1] -= np.nanmean(eye_pos[finite, 1])
    dx = saccades["eye_vel"][:, 0]
    dy = saccades["eye_vel"][:, 1]

    quiver = {}
    for label in ("Left", "Right"):
        idx = np.asarray(first_indices.get(label, []), dtype=int)
        cong = np.asarray(first_congruent.get(label, []), dtype=bool)
        if idx.size:
            keep = finite[idx]
            idx = idx[keep]
            cong = cong[keep]
        quiver[label] = {
            "x": eye_pos[idx, 0], "y": eye_pos[idx, 1],
            "dx": dx[idx], "dy": dy[idx], "congruent": cong,
        }
    return quiver


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
def _draw_quiver_panel(ax, qd, label):
    """Draw one condition's arrows via the real
    :func:`eyehead.analysis._draw_quiver_arrows` helper (the same one
    :func:`eyehead.analysis.sort_saccades` uses for its own per-condition
    figure), so the arrow-drawing code itself is never duplicated — only
    axis cosmetics (limits, title, tick count) are specific to this
    composite figure."""
    extent = 1.0
    if qd["x"].size:
        extent = max(np.nanmax(np.abs(qd["x"])), np.nanmax(np.abs(qd["y"]))) * 1.5

    eye_pos = np.column_stack([qd["x"], qd["y"]])
    idx_use = np.arange(len(qd["x"]))
    _draw_quiver_arrows(
        ax, eye_pos, idx_use, qd["dx"], qd["dy"],
        congruent_use=qd["congruent"],
        xlim=(-13, 13), ylim=(-7, 7),
        title=f"{label} target",
    )
    ax.set_aspect("equal")
    ax.title.set_fontsize(FONT_SIZE_TITLE)
    ax.xaxis.label.set_fontsize(FONT_SIZE_LABEL)
    ax.yaxis.label.set_fontsize(FONT_SIZE_LABEL)
    _limit_ticks(ax)
    _strip_box(ax)


def _draw_polar_panel(ax, angle, reward_angle_deg, zone_center_deg, color, kappa=12):
    bins = 18
    counts, edges = np.histogram(angle, bins=bins, range=(-np.pi, np.pi))
    total = counts.sum()
    counts = counts / total if total else counts.astype(float)
    peak = counts.max() if counts.size else 1.0

    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=color, alpha=0.55, linewidth=0)

    zone = np.deg2rad(np.arange(zone_center_deg - reward_angle_deg,
                                 zone_center_deg + reward_angle_deg, 1))
    ax.fill_between(zone, 0, peak, color=REWARD_COLOR, alpha=0.15, linewidth=0)

    if angle.size:
        mean_angle = circmean(angle, high=np.pi, low=-np.pi)
        ax.plot([mean_angle, mean_angle], [0, peak], color="black", ls="--", lw=1)

        theta = np.linspace(-np.pi, np.pi, 200)
        kernels = np.array([vonmises.pdf(theta, kappa, loc=a) for a in angle])
        density = kernels.sum(axis=0)
        if density.max() > 0:
            density = density * peak / density.max()
        theta_closed = np.append(theta, theta[0])
        density_closed = np.append(density, density[0])
        ax.plot(theta_closed, density_closed, color="black", lw=1.1)

    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"],
                       fontsize=FONT_SIZE_TICK)
    ax.spines["polar"].set_visible(False) #get rid of visible circle outline

def _draw_latency_hist(ax, latencies, congruent, reward_window, congruency_window):
    correct = latencies[congruent]
    incorrect = latencies[~congruent]
    hi = max(reward_window, float(latencies.max())) if latencies.size else reward_window
    bins = np.linspace(0, hi, 21)
    ax.hist(correct, bins=bins, alpha=0.6, color=CORRECT_COLOR)
    ax.hist(incorrect, bins=bins, alpha=0.6, color=INCORRECT_COLOR)
    ax.axvspan(0, reward_window, color=REWARD_COLOR, alpha=0.10, lw=0)
    if congruency_window is not None:
        ax.axvspan(congruency_window[0], congruency_window[1], color=CONGRUENCY_WINDOW_COLOR, alpha=0.10, lw=0)
    
    ax.set_xlabel("Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Trial count", fontsize=FONT_SIZE_LABEL)
    _limit_ticks(ax)
    _strip_box(ax)


def _draw_latency_cdf(ax, latencies, congruent, reward_window, congruency_window):
    correct = np.sort(latencies[congruent])
    incorrect = np.sort(latencies[~congruent])
    if correct.size:
        ax.step(correct, np.arange(1, correct.size + 1) / correct.size,
                where="post", color=CORRECT_COLOR)
    if incorrect.size:
        ax.step(incorrect, np.arange(1, incorrect.size + 1) / incorrect.size,
                where="post", color=INCORRECT_COLOR)
    ax.axvspan(0, reward_window, color=REWARD_COLOR, alpha=0.10, lw=0)
    if congruency_window is not None:
        ax.axvspan(congruency_window[0], congruency_window[1],
                   color=CONGRUENCY_WINDOW_COLOR, alpha=0.10, lw=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Cumulative fraction", fontsize=FONT_SIZE_LABEL)
    _limit_ticks(ax, y=False)
    _strip_box(ax)


def _draw_accuracy_vs_latency(ax, latencies, congruent, reward_window, congruency_window):
    centers, frac, n_per_window = prosaccade_session.fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0, reward_window),
    )
    valid = n_per_window > 0
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.plot(centers[valid], frac[valid], "-o", color=CORRECT_COLOR, ms=3)
    ax.axvspan(0, reward_window, color=REWARD_COLOR, alpha=0.10, lw=0)
    if congruency_window is not None:
        ax.axvspan(congruency_window[0], congruency_window[1],
                   color=CONGRUENCY_WINDOW_COLOR, alpha=0.10, lw=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Fraction toward target", fontsize=FONT_SIZE_LABEL)
    _limit_ticks(ax, y=False)
    _strip_box(ax)


def _draw_validity_columns(axes, session_validity, animal_pooled):
    ax_valid, ax_correct, ax_window = axes
    animals_sorted = sorted(animal_pooled.keys())
    rng = np.random.default_rng(0)

    for i, animal in enumerate(animals_sorted):
        sessions = [r for r in session_validity if r["animal_name"] == animal]
        jitter = rng.uniform(-0.15, 0.15, size=len(sessions))
        xs = np.full(len(sessions), i, dtype=float) + jitter

        ax_valid.scatter(xs, [r["fraction_valid"] for r in sessions],
                          color=CORRECT_COLOR, s=16, alpha=0.6, zorder=2)
        ax_correct.scatter(xs, [r["fraction_correct"] for r in sessions],
                            color=CORRECT_COLOR, s=16, alpha=0.6, zorder=2)
        ax_window.scatter(xs, [r["window_frac"] for r in sessions],
                           color=CORRECT_COLOR, s=16, alpha=0.6, zorder=2)

        pooled = animal_pooled[animal]
        pooled_lo = pooled["latency_outcome"]
        n_total = pooled_lo["n_total"]
        n_valid = n_total - pooled_lo["n_no_saccade"]
        congruent = pooled_lo["congruent"]
        n_correct = int(np.sum(congruent))
        frac_valid = n_valid / n_total if n_total else np.nan
        frac_correct = float(np.mean(congruent)) if len(congruent) else np.nan
        valid_lo, valid_hi = prosaccade_session.wilson_ci(n_valid, n_total)
        correct_lo, correct_hi = prosaccade_session.wilson_ci(n_correct, len(congruent))
        window_frac, _, window_lo, window_hi = prosaccade_session.congruency_in_window(
            pooled_lo["latencies"], congruent, window=pooled["congruency_window"],
        )

        for ax, val, lo_, hi_ in (
            (ax_valid, frac_valid, valid_lo, valid_hi),
            (ax_correct, frac_correct, correct_lo, correct_hi),
            (ax_window, window_frac, window_lo, window_hi),
        ):
            ax.errorbar(
                [i], [val],
                yerr=[[max(0.0, val - lo_)], [max(0.0, hi_ - val)]],
                fmt="o", color="black", ecolor="black", capsize=3, ms=6, zorder=3,
            )

    for ax in axes:
        ax.set_xticks(range(len(animals_sorted)))
        ax.set_xticklabels(animals_sorted, rotation=30, ha="right", fontsize=FONT_SIZE_TICK)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Squished side by side: only the leftmost keeps a left spine/y-tick
    # labels (the shared 0-1 fraction scale); the other two drop theirs
    # entirely rather than repeat it three times in a cramped space.
    _strip_box(ax_valid, keep=("left", "bottom"))
    for ax in (ax_correct, ax_window):
        _strip_box(ax, keep=("bottom",))
        ax.tick_params(left=False, labelleft=False)

    ax_valid.set_ylabel("Fraction", fontsize=FONT_SIZE_LABEL)
    ax_valid.set_title("Validity", fontsize=FONT_SIZE_TITLE)
    ax_correct.set_title("Accuracy", fontsize=FONT_SIZE_TITLE)
    ax_window.set_title("Congruency vs.\npre-cue", fontsize=FONT_SIZE_TITLE)


# ---------------------------------------------------------------------------
# This section puts the actual figure together
# ---------------------------------------------------------------------------
def build_figure(session_id: str, experiment_type: str):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_SANS_SERIF
    plt.rcParams["font.size"] = FONT_SIZE_BASE
    # Matplotlib's SVG default ("path") outlines every glyph into vector
    # paths -- editable in Illustrator, but no longer real text (can't
    # select/retype it, change font, etc). "none" keeps actual <text>
    # elements referencing the font by name instead, so Illustrator imports
    # it as real, editable text -- as long as that font (Arial here) is
    # installed on whatever machine opens the SVG.
    plt.rcParams["svg.fonttype"] = "none"

    cache = _load_population_cache(experiment_type)
    session_results = cache["session_results"]
    animal_pooled = cache["animal_pooled"]
    session_validity = cache["session_validity"]

    if session_id not in session_results:
        raise KeyError(
            f"Session {session_id!r} isn't in the cached population run "
            f"({len(session_results)} sessions cached: "
            f"{sorted(session_results.keys())}). Either pick a cached "
            f"session or re-run prosaccade_population.py to refresh the "
            f"cache."
        )
    session_result = session_results[session_id]
    lo = session_result["latency_outcome"]

    # Not cached -- cheap (one session's detection, not the whole
    # manifest), so just computed directly each time. See
    # _load_session_quiver_data's docstring for why this can't be read from
    # session_result instead.
    quiver = _load_session_quiver_data(session_id)

    if not animal_pooled:
        raise ValueError(f"No animals found in the cached population run for experiment_type={experiment_type!r}.")
    if len(animal_pooled) > 1:
        pooled_all = pp.pool_animal_sessions(list(animal_pooled.values()))
    else:
        pooled_all = next(iter(animal_pooled.values()))
    pooled_lo = pooled_all["latency_outcome"]


    #setup the main figure size and layout
    fig = plt.figure(figsize=(11, 8.5))
    gs_main = fig.add_gridspec(3, 1, height_ratios=[0.9, 0.9, 0.9], hspace=0.4)

    # --- Row 1: Panel A (schematic placeholder) + Panel B (arrows/polar) ---
    gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[0.2, 2.5], wspace=0.2)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.axis("off")
    ax_a.add_patch(plt.Rectangle(
        (0.02, 0.02), 0.96, 0.96, fill=False, ls="--", lw=1, color="0.6",
        transform=ax_a.transAxes,
    ))
    ax_a.text(0.5, 0.5, "Panel A\n(task schematic)", ha="center", va="center",
              fontsize=FONT_SIZE_LABEL, color="0.5", transform=ax_a.transAxes)
    _panel_letter(ax_a, "A")

    # Arrow + polar side by side per condition (Left arrow, Left polar,
    # Right arrow, Right polar) instead of stacked -- a 2x2 stack left a lot
    # of empty vertical space since quiver panels are wide-but-short and
    # polar panels are compact, so GridSpec was giving both rows equal
    # height regardless.
    gs_b = gs_top[0, 1].subgridspec(1, 4, wspace=0.5)
    ax_b_arrow_l = fig.add_subplot(gs_b[0, 0])
    ax_b_polar_l = fig.add_subplot(gs_b[0, 1], polar=True)
    ax_b_arrow_r = fig.add_subplot(gs_b[0, 2])
    ax_b_polar_r = fig.add_subplot(gs_b[0, 3], polar=True)

    _draw_quiver_panel(ax_b_arrow_l, quiver["Left"], "Left")
    _draw_quiver_panel(ax_b_arrow_r, quiver["Right"], "Right")
    _draw_polar_panel(ax_b_polar_l, session_result["left_angle"],
                       session_result["reward_angle"], 180, "tab:green")
    _draw_polar_panel(ax_b_polar_r, session_result["right_angle"],
                       session_result["reward_angle"], 0, "tab:pink")
    _panel_letter(ax_b_arrow_l, "B")

    # --- Row 2: Panel C (single session hist / CDF / accuracy-vs-latency) ---
    # Same width_ratios AND wspace as row D's outer 3-slot layout below, so
    # the hist/accuracy-vs-latency panels end up pixel-identical in width
    # across both rows.
    ROW_CD_WIDTH_RATIOS = [1, 1, 1]
    ROW_CD_WSPACE = 0.5

    gs_c = gs_main[1].subgridspec(1, 3, width_ratios=ROW_CD_WIDTH_RATIOS, wspace=ROW_CD_WSPACE)
    ax_c_hist = fig.add_subplot(gs_c[0, 0])
    ax_c_cdf = fig.add_subplot(gs_c[0, 1])
    ax_c_acc = fig.add_subplot(gs_c[0, 2])

    _draw_latency_hist(ax_c_hist, lo["latencies"], lo["congruent"],
                        session_result["reward_window"], None)
    _draw_latency_cdf(ax_c_cdf, lo["latencies"], lo["congruent"],
                       session_result["reward_window"], None)
    _draw_accuracy_vs_latency(ax_c_acc, lo["latencies"], lo["congruent"],
                               session_result["reward_window"], session_result["congruency_window"])
    _panel_letter(ax_c_hist, "C")

    # --- Row 3: Panel D (population hist / accuracy-vs-latency / validity) ---
    # Outer layout: 3 equal slots, same as row C above (hist, accuracy-vs-
    # latency, and one slot for the whole validity block) -- NOT 5 slots,
    # which would make "1" mean a different width than in row C. The
    # validity block then subdivides its one slot into 3 narrow columns.
    gs_d_outer = gs_main[2].subgridspec(1, 3, width_ratios=ROW_CD_WIDTH_RATIOS, wspace=ROW_CD_WSPACE)
    ax_d_hist = fig.add_subplot(gs_d_outer[0, 0])
    ax_d_acc = fig.add_subplot(gs_d_outer[0, 1])

    gs_d_validity = gs_d_outer[0, 2].subgridspec(1, 3, wspace=0.15)
    ax_d_valid = fig.add_subplot(gs_d_validity[0, 0])
    ax_d_correct = fig.add_subplot(gs_d_validity[0, 1])
    ax_d_window = fig.add_subplot(gs_d_validity[0, 2])

    _draw_latency_hist(ax_d_hist, pooled_lo["latencies"], pooled_lo["congruent"],
                        pooled_all["reward_window"], None)
    _draw_accuracy_vs_latency(ax_d_acc, pooled_lo["latencies"], pooled_lo["congruent"],
                               pooled_all["reward_window"], None)
    _draw_validity_columns((ax_d_valid, ax_d_correct, ax_d_window), session_validity, animal_pooled)
    _panel_letter(ax_d_hist, "D")

    # Single shared legend for the color language used across every panel,
    # instead of repeating it in each one. Delete this block if unwanted.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CORRECT_COLOR, label="correct"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=INCORRECT_COLOR, label="incorrect"),
        plt.Rectangle((0, 0), 1, 1, color=REWARD_COLOR, alpha=0.3, label="reward window"),
        plt.Rectangle((0, 0), 1, 1, color=CONGRUENCY_WINDOW_COLOR, alpha=0.3, label="congruency window"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=FONT_SIZE_LABEL,
               frameon=False, ncol=1, bbox_to_anchor=(0.995, 0.995))

    return fig


################################################### main call to build the figure and save it in the right place/format

fig = build_figure(SESSION_ID, EXPERIMENT_TYPE)

if OUTPUT_STEM:
    out_stem = Path(OUTPUT_STEM)
else:
    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    results_root = Path(manifest.get("results_root") or root_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    out_stem = results_root / f"{EXPERIMENT_TYPE}_summary_figure"

for ext in ("png", "svg"):
    path = out_stem.with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")
plt.show()


