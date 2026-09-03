"""Standalone script: one composite multi-panel summary figure for the
prosaccade task, combining a single session's arrow/polar/latency plots
with the all-animals population summary.

Layout
------
- Panel A: blank placeholder for a hand-drawn task schematic (added
  separately, e.g. in Illustrator/PowerPoint).
- Panel B: one session's Left/Right arrow (quiver) plots on top, Left/Right
  polar saccade-angle histograms below.
- Panel C: that same session's latency-by-outcome histogram and
  saccade-accuracy-vs-latency curve (with pointwise Wilson intervals).
- Panel D: the population (every animal in the manifest, pooled together)
  latency-by-outcome histogram, accuracy-vs-latency curve, and the
  2-column session validity/accuracy summary (trial validity, and
  windowed accuracy vs. pre-cue control), grouped by animal.

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

This script also builds a second, separate SUPPLEMENTARY figure: the
per-animal breakdown. The main figure's population row pools every animal
together, which cannot show whether both animals behave the same way. The
supplement therefore repeats three panels -- latency-by-outcome histogram,
latency CDF by outcome, and accuracy-vs-latency -- once per animal, each
pooling all of that animal's sessions. Both figures are drawn from the same
cache in one run and saved to separate files.

Usage
-----
No CLI arguments -- edit ``SESSION_ID``/``EXPERIMENT_TYPE``/``OUTPUT_STEM``
near the top of this file directly, then run:

    python Python/analysis/Fig3_prosaccade.py
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
SESSION_ID = "Tsh002_2026-08-19T12_49_59"  #Apollo sessions B/C
EXPERIMENT_TYPE = "prosaccade"  # pooled across every animal in the manifest -- panel D
OUTPUT_STEM = None  # None -> <results_root>/<EXPERIMENT_TYPE>_summary_figure; or set an explicit Path/str
SUPPLEMENT_OUTPUT_STEM = None  # None -> <results_root>/<EXPERIMENT_TYPE>_supplement_per_animal; or an explicit Path/str

# ---------------------------------------------------------------------------
# Styling constants — tweak these to fit the final page/journal layout.
# ---------------------------------------------------------------------------
FONT_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans"]
### Type sizes and line weights matched to the other figures in the paper
### (the fixation-task figure), so Figure 3 drops in beside them without
### being restyled by hand in Illustrator. Everything here is one constant,
### so the whole figure rescales together if the journal wants it smaller.
FONT_SIZE_BASE = 7
FONT_SIZE_LABEL = 7
FONT_SIZE_TITLE = 8
FONT_SIZE_TICK = 6
FONT_SIZE_PANEL_LETTER = 12
N_TICKS = 3  # e.g. 0, 25, 50 instead of 0, 10, 20, 30, 40, 50

### Short, thin, INWARD ticks on thin spines -- the look of the other
### figures in the paper, and much lighter than matplotlib's defaults
### (3.5 pt outward ticks on 0.8 pt axes with 3.5 pt padding).
###
### Inward is what keeps the axes corner clean. With outward ticks the
### tick at each axis minimum projects past the spine junction, so the
### bottom-left corner reads as two crossed lines with stubs hanging off
### it rather than a closed L. Pointing them inward puts every tick inside
### the axes, leaving the corner as a single clean join. Tick labels need
### slightly more padding once the tick no longer occupies that space.
SPINE_WIDTH = 0.8
TICK_LENGTH = 2.5
TICK_WIDTH = 0.8
TICK_DIRECTION = "in"
TICK_PAD = 2.5
LABEL_PAD = 2.0

### Fixed post-target latency band that Panel D's third column averages
### accuracy over, and that the latency panels of C and D outline with a
### dashed window. This replaces the manifest's congruency_window, which
### was only meaningful while that window was narrow -- at its current
### [0, 1] it spans essentially every saccade, so "congruency in the
### window" and "overall accuracy" had become the same number. The 0.2 s
### lower edge also excludes the sub-200 ms saccades that sit at chance
### (too fast to be visually driven), which is what the accuracy-vs-latency
### curves in C and D show directly.
ACCURACY_WINDOW = (0.2, 1.0)
ACCURACY_WINDOW_COLOR = "0.35"

### Colorblind-safe palette. The previous green/red pair carried the
### primary correct/incorrect contrast in five panels and is the one
### combination red-green deficient readers (~8% of men) cannot separate;
### blue/orange reads the same to them as to everyone else. Panel B's polar
### histograms were a THIRD color language (Left=green, Right=pink) that
### collided with correct/incorrect, so they are now a neutral gray -- the
### panel titles and the shaded reward zone already say which side is which.
REWARD_COLOR = "#D9D9D9"
CONGRUENCY_WINDOW_COLOR = "tab:purple"
CORRECT_COLOR = "#0173B2"
INCORRECT_COLOR = "#DE8F05"
POLAR_HIST_COLOR = "0.45"
PRECUE_COLOR = "0.45"      # open markers, Panel D's pre-cue control
PRECUE_DX = 0.17           # x-offset of the pre-cue/post-target pair


def _apply_paper_style():
    """Set the rcParams both figures are drawn under.

    Kept in one function rather than repeated at the top of each build
    function, so the main figure and the supplement cannot drift apart
    typographically -- they have to sit side by side in the same paper.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS_SERIF,
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelpad": LABEL_PAD,
        "axes.linewidth": SPINE_WIDTH,
        "legend.fontsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "xtick.direction": TICK_DIRECTION,
        "ytick.direction": TICK_DIRECTION,
        # Square butt-ends on the spines so the left/bottom pair meets in a
        # sharp corner rather than two rounded caps leaving a nick at the
        # junction.
        "lines.solid_capstyle": "projecting",
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.pad": TICK_PAD,
        "ytick.major.pad": TICK_PAD,
        # Matplotlib's SVG default ("path") outlines every glyph into vector
        # paths -- editable in Illustrator, but no longer real text (can't
        # select/retype it, change font, etc). "none" keeps actual <text>
        # elements referencing the font by name instead, so Illustrator
        # imports it as real, editable text -- as long as that font (Arial
        # here) is installed on whatever machine opens the SVG.
        "svg.fonttype": "none",
    })


def _limit_ticks(ax, x=True, y=True, nbins=N_TICKS):
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.tick_params(labelsize=FONT_SIZE_TICK)


def _panel_letter(ax, letter, dx=-22, dy=4):
    """Place the panel letter a fixed distance in *points* from the axes'
    top-left corner rather than in axes-fraction coordinates. An
    axes-fraction offset is a fraction of each panel's own width, so a
    single hardcoded -0.22 put the letter far from a narrow panel and
    almost inside a wide one."""
    ax.annotate(
        letter, xy=(0, 1), xycoords="axes fraction",
        xytext=(dx, dy), textcoords="offset points",
        fontsize=FONT_SIZE_PANEL_LETTER, fontweight="bold",
        va="bottom", ha="left", annotation_clip=False,
    )


def _close_axes_corner(ax):
    """Make the left and bottom spines terminate exactly on each other.

    Matplotlib draws each spine across its own axis independently, so with
    a visible linewidth the two can meet with a small nick at the
    bottom-left. Giving each spine a projecting cap closes that join.
    """
    for side in ("left", "bottom"):
        spine = ax.spines.get(side)
        if spine is not None and spine.get_visible():
            spine.set_capstyle("projecting")


def _strip_box(ax, keep=("left", "bottom")):
    """Remove the axes' bounding-box spines except ``keep``, for a cleaner,
    less boxed-in look. Not applied to polar axes (Panel B's polar plots)
    since their outer circle is the actual radial-extent boundary, not
    decoration."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    _close_axes_corner(ax)


def _annotate_n(ax, text, loc="upper right"):
    """Small unobtrusive n annotation inside a corner of ``ax``. Every panel
    reports its own n rather than leaving the reader to hunt for it in the
    caption."""
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=FONT_SIZE_TICK, color="0.35",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65,
                      boxstyle="square,pad=0.15"))


def _window_is_informative(span, x_hi):
    """True if shading ``span`` would actually delimit part of the panel.

    A window running from <=0 to >=``x_hi`` covers the entire axes: it marks
    nothing, but still tints every bar and line underneath it, and where two
    such windows overlap the whole panel turns brown. That is exactly what
    the current manifest produces (``reward_window`` 1.1 s and
    ``congruency_window`` [0, 1] against latencies that stop around 0.9 s),
    so those windows are simply not drawn. Narrow the windows in
    ``session_manifest.yml`` and the shading comes back on its own."""
    if span is None or x_hi is None or not np.isfinite(x_hi):
        return False
    return not (float(span[0]) <= 0.0 and float(span[1]) >= x_hi)


def _mark_accuracy_window(ax, x_hi):
    """Dashed outline of :data:`ACCURACY_WINDOW` -- the latency band whose
    average accuracy Panel D's third column reports -- so the reader can see
    exactly which saccades that number came from.

    Each edge is drawn only where it actually falls inside the panel. When
    the upper edge runs past the last saccade (as it does whenever the
    reward window is shorter than 1.0 s) the band is left open-ended rather
    than closed at a boundary no data reaches."""
    lo_, hi_ = float(ACCURACY_WINDOW[0]), float(ACCURACY_WINDOW[1])
    lo_v, hi_v = max(lo_, 0.0), min(hi_, float(x_hi))
    if hi_v <= lo_v:
        return
    ax.axvspan(lo_v, hi_v, color=ACCURACY_WINDOW_COLOR, alpha=0.06, lw=0, zorder=0)
    for edge in (lo_, hi_):
        if 0.0 < edge < x_hi:
            ax.axvline(edge, color=ACCURACY_WINDOW_COLOR, ls="--", lw=0.9,
                       alpha=0.85, zorder=0)


def _shade_window(ax, span, color, x_hi):
    """Shade ``span`` when :func:`_window_is_informative`, marking whichever
    edge falls inside the panel so the boundary is readable."""
    if not _window_is_informative(span, x_hi):
        return False
    lo_, hi_ = float(span[0]), float(span[1])
    ax.axvspan(lo_, hi_, color=color, alpha=0.10, lw=0)
    for edge in (lo_, hi_):
        if 0.0 < edge < x_hi:
            ax.axvline(edge, color=color, ls=":", lw=0.8, alpha=0.9)
    return True


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
    ### eyehead.analysis._draw_quiver_arrows hardcodes tab:green/tab:red and
    ### exposes no color argument. Rather than edit that shared helper (used
    ### by sort_saccades throughout the pipeline, so a change there would
    ### repaint every other figure), recolor only this figure's copy of the
    ### arrows in place, so Panel B speaks the same colorblind-safe language
    ### as Panels C/D. If that helper ever gains a color argument, drop this.
    congruent = np.asarray(qd["congruent"], dtype=bool) if qd["congruent"] is not None else None
    if ax.collections and congruent is not None and congruent.size:
        ax.collections[-1].set_color(
            np.where(congruent, CORRECT_COLOR, INCORRECT_COLOR)
        )

    ### _draw_quiver_arrows labels these "X (deg-sign)"; the rest of the
    ### paper writes "(deg)", so relabel here rather than change the shared
    ### helper and restyle every other figure that calls it.
    ax.set_xlabel("X (deg)")
    ax.set_ylabel("Y (deg)")

    ax.set_aspect("equal")
    _annotate_n(ax, f"n = {qd['x'].size}", loc="lower right")
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

    ### The radial axis was entirely unlabelled, so the histogram had no
    ### scale at all. One tick at the peak, as a percentage of trials, is
    ### enough to make the bars quantitative without cluttering the panel.
    if peak > 0:
        ax.set_ylim(0, peak)
        ax.set_yticks([peak])
        ax.set_yticklabels([f"{peak * 100:.0f}%"], fontsize=FONT_SIZE_TICK - 1,
                           color="0.35")
        ax.set_rlabel_position(135)
    else:
        ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"],
                       fontsize=FONT_SIZE_TICK)
    ax.spines["polar"].set_visible(False) #get rid of visible circle outline
    ax.text(0.5, -0.30, f"n = {angle.size}", transform=ax.transAxes,
            ha="center", va="top", fontsize=FONT_SIZE_TICK, color="0.35")

def _draw_latency_hist(ax, latencies, congruent, reward_window, x_hi):
    correct = latencies[congruent]
    incorrect = latencies[~congruent]
    hi = max(reward_window, float(latencies.max())) if latencies.size else reward_window
    bins = np.linspace(0, hi, 21)

    ### A light fill plus a solid step outline, rather than two translucent
    ### fills. At alpha=0.6 the green and red fills alpha-blended into a
    ### third color wherever they overlapped, which read as its own
    ### category; the outlines keep each distribution traceable through the
    ### overlap.
    ### Outlines only, no fills. A filled distribution hides whatever sits
    ### behind it and, where two fills overlap, alpha-blends into a third
    ### color that reads as its own category. Outlines keep both
    ### distributions fully traceable through each other.
    if correct.size:
        ax.hist(correct, bins=bins, color=CORRECT_COLOR, histtype="step", lw=1.4)
    if incorrect.size:
        ax.hist(incorrect, bins=bins, color=INCORRECT_COLOR, histtype="step", lw=1.4)

    _shade_window(ax, (0, reward_window), REWARD_COLOR, x_hi)
    _mark_accuracy_window(ax, x_hi)

    ax.set_xlim(0, x_hi)
    ax.set_xlabel("Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Trial count", fontsize=FONT_SIZE_LABEL)
    _annotate_n(ax, f"n = {latencies.size} trials")
    _limit_ticks(ax)
    ### Trial counts are integers -- MaxNLocator was free to tick at 2.5.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=N_TICKS, integer=True))
    _strip_box(ax)


def _draw_latency_cdf(ax, latencies, congruent, reward_window, x_hi):
    """Cumulative latency distribution, correct and incorrect drawn
    separately.

    Each curve is normalized within its own outcome, so both rise to 1.0 and
    the panel compares the SHAPE of the two latency distributions -- whether
    incorrect saccades happen earlier than correct ones -- not how many of
    each there were. The histogram beside it carries the relative counts.

    This panel was dropped from the main figure (it broke the column
    correspondence between rows C and D) but is the natural third view of
    per-animal latency, so it lives here instead.
    """
    for vals, color in ((latencies[congruent], CORRECT_COLOR),
                        (latencies[~congruent], INCORRECT_COLOR)):
        if vals.size:
            ordered = np.sort(vals)
            ax.step(ordered, np.arange(1, ordered.size + 1) / ordered.size,
                    where="post", color=color, lw=1.4)

    _shade_window(ax, (0, reward_window), REWARD_COLOR, x_hi)
    _mark_accuracy_window(ax, x_hi)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, x_hi)
    ax.set_xlabel("Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Cumulative fraction", fontsize=FONT_SIZE_LABEL)
    _annotate_n(ax, f"n = {latencies.size} trials", loc="lower right")
    _limit_ticks(ax, y=False)
    _strip_box(ax)


def _draw_accuracy_vs_latency(ax, latencies, congruent, reward_window, x_hi):
    centers, frac, n_per_window = prosaccade_session.fraction_toward_target_by_latency(
        latencies, congruent, window_span=(0, reward_window),
    )
    valid = n_per_window > 0

    ### Wilson intervals, the same estimator the Panel D validity columns
    ### use, so the two agree. Caveat worth keeping in the caption:
    ### fraction_toward_target_by_latency slides a 0.3 s window in 0.05 s
    ### steps, so neighbouring points share most of their trials. This is a
    ### pointwise interval on each window, NOT a simultaneous confidence
    ### band, and adjacent points are not independent.
    lo_arr = np.full(centers.shape, np.nan)
    hi_arr = np.full(centers.shape, np.nan)
    for i in np.flatnonzero(valid):
        n_i = int(n_per_window[i])
        k_i = int(round(frac[i] * n_i))
        lo_arr[i], hi_arr[i] = prosaccade_session.wilson_ci(k_i, n_i)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.fill_between(centers[valid], lo_arr[valid], hi_arr[valid],
                    color=CORRECT_COLOR, alpha=0.20, lw=0)
    ax.plot(centers[valid], frac[valid], "-o", color=CORRECT_COLOR, ms=3)

    _shade_window(ax, (0, reward_window), REWARD_COLOR, x_hi)
    _mark_accuracy_window(ax, x_hi)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, x_hi)
    ax.set_xlabel("Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Fraction toward target", fontsize=FONT_SIZE_LABEL)
    _annotate_n(ax, f"n = {latencies.size} trials", loc="lower right")
    _limit_ticks(ax, y=False)
    _strip_box(ax)


def _draw_validity_columns(axes, session_validity, animal_pooled, session_results):
    """Two per-animal summary columns: trial validity, and windowed accuracy
    against its own pre-cue control.

    The all-trials accuracy column that used to sit between them has been
    dropped: it and the windowed column measured the same thing on
    overlapping trials, differing only by whether the sub-200 ms
    anticipatory saccades were included, and the windowed one is the
    measure being reported. Panels C and D already show the full accuracy
    -vs-latency profile, so nothing is lost by not also plotting its
    unwindowed average here.

    The second column reports accuracy averaged over
    :data:`ACCURACY_WINDOW`, not congruency in the manifest's
    ``congruency_window``. That window is currently [0, 1] s, which spans
    essentially every saccade, so "congruency in the window" had collapsed
    onto the plain accuracy already shown in column 2; a fixed 0.2-1.0 s
    band instead excludes the sub-200 ms anticipatory saccades, and so
    measures something column 2 does not. Because the per-session value has
    to be recomputed over that band, this needs ``session_results`` (the
    cached per-session dicts) rather than the pre-summarized ``window_frac``
    already in ``session_validity``.

    It stays a paired comparison against each session's own pre-cue control
    (``precue_congruent``): one thin line per session joining its baseline
    to its windowed accuracy, with the animal's pooled value and Wilson
    interval in black over each. Without the baseline on the panel a reader
    has no way to tell whether 0.8 is a task effect or just what the animal
    does anyway.
    """
    ax_valid, ax_window = axes
    animals_sorted = sorted(animal_pooled.keys())
    rng = np.random.default_rng(0)
    n_sessions_total = 0

    for i, animal in enumerate(animals_sorted):
        sessions = [r for r in session_validity if r["animal_name"] == animal]
        n_sessions_total += len(sessions)
        jitter = rng.uniform(-0.15, 0.15, size=len(sessions))
        xs = np.full(len(sessions), i, dtype=float) + jitter

        ax_valid.scatter(xs, [r["fraction_valid"] for r in sessions],
                          color=CORRECT_COLOR, s=16, alpha=0.6, zorder=2)

        ### Paired pre-cue vs. post-target. No jitter here -- the two x
        ### positions are fixed so each session's connecting line reads as a
        ### slope rather than a random diagonal.
        x_pre, x_post = i - PRECUE_DX, i + PRECUE_DX

        ### Recomputed per session over ACCURACY_WINDOW rather than read from
        ### session_validity["window_frac"], which was computed against the
        ### manifest's congruency_window back at population-run time.
        session_window_fracs = []
        for r in sessions:
            sr = session_results.get(r["session_id"])
            if sr is None:
                session_window_fracs.append(np.nan)
                continue
            s_lo = sr["latency_outcome"]
            s_frac, _, _, _ = prosaccade_session.congruency_in_window(
                s_lo["latencies"], s_lo["congruent"], window=ACCURACY_WINDOW,
            )
            session_window_fracs.append(s_frac)

        for r, s_frac in zip(sessions, session_window_fracs):
            ax_window.plot([x_pre, x_post], [r["precue_frac"], s_frac],
                           color="0.8", lw=0.6, zorder=1)
        ax_window.scatter(np.full(len(sessions), x_pre),
                          [r["precue_frac"] for r in sessions],
                          facecolors="none", edgecolors=PRECUE_COLOR,
                          s=16, lw=0.8, zorder=2)
        ax_window.scatter(np.full(len(sessions), x_post), session_window_fracs,
                          color=CORRECT_COLOR, s=16, alpha=0.6, zorder=2)

        pooled = animal_pooled[animal]
        pooled_lo = pooled["latency_outcome"]
        n_total = pooled_lo["n_total"]
        n_valid = n_total - pooled_lo["n_no_saccade"]
        congruent = pooled_lo["congruent"]
        frac_valid = n_valid / n_total if n_total else np.nan
        valid_lo, valid_hi = prosaccade_session.wilson_ci(n_valid, n_total)
        window_frac, _, window_lo, window_hi = prosaccade_session.congruency_in_window(
            pooled_lo["latencies"], congruent, window=ACCURACY_WINDOW,
        )

        precue_congruent = pooled["precue_congruent"]
        n_precue = len(precue_congruent)
        precue_frac = float(np.mean(precue_congruent)) if n_precue else np.nan
        precue_lo, precue_hi = prosaccade_session.wilson_ci(
            int(np.sum(precue_congruent)), n_precue,
        )

        for ax, x, val, lo_, hi_ in (
            (ax_valid, i, frac_valid, valid_lo, valid_hi),
            (ax_window, x_pre, precue_frac, precue_lo, precue_hi),
            (ax_window, x_post, window_frac, window_lo, window_hi),
        ):
            if not np.isfinite(val):
                continue
            ax.errorbar(
                [x], [val],
                yerr=[[max(0.0, val - lo_)], [max(0.0, hi_ - val)]],
                fmt="o", color="black", ecolor="black", capsize=3, ms=6, zorder=3,
            )

    for ax in axes:
        ax.set_xticks(range(len(animals_sorted)))
        ### No rotation: with a handful of short animal names these fit
        ### horizontally, and rotating them only cost vertical space.
        ax.set_xticklabels(animals_sorted, fontsize=FONT_SIZE_TICK)
        ax.set_xlim(-0.5, len(animals_sorted) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    ### Accuracy and congruency are both chance-0.5 measures, so they get
    ### the same reference line the latency panels already carry. Validity
    ### is a completion rate, not a two-alternative choice, so 0.5 means
    ### nothing there and no line is drawn.
    ax_window.axhline(0.5, color="gray", ls="--", lw=0.8, zorder=0)

    # Squished side by side: only the leftmost keeps a left spine/y-tick
    # labels (the shared 0-1 fraction scale); the other two drop theirs
    # entirely rather than repeat it three times in a cramped space.
    _strip_box(ax_valid, keep=("left", "bottom"))
    _strip_box(ax_window, keep=("bottom",))
    ax_window.tick_params(left=False, labelleft=False)

    ax_valid.set_ylabel("Fraction", fontsize=FONT_SIZE_LABEL)
    ax_valid.set_title("Validity", fontsize=FONT_SIZE_TITLE)
    ax_window.set_title(
        f"Accuracy\n{ACCURACY_WINDOW[0]:.1f}-{ACCURACY_WINDOW[1]:.1f} s",
        fontsize=FONT_SIZE_TITLE)
    _annotate_n(ax_valid,
                f"{n_sessions_total} sessions\n{len(animals_sorted)} animals",
                loc="lower left")


# ---------------------------------------------------------------------------
# This section puts the actual figure together
# ---------------------------------------------------------------------------
def build_figure(session_id: str, experiment_type: str):
    _apply_paper_style()

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


    ### Authored at its FINAL printed size, roughly double-column width.
    ### This matters more than any font constant: point sizes only carry
    ### across figures rendered at the same physical size. At the previous
    ### 11 x 8.5 in, a 6 pt tick label had to be scaled down by ~0.65 to
    ### reach journal width, landing at ~4 pt on the page while the other
    ### figures' 6 pt stayed 6 pt. Sizes here are now literal: what the
    ### constants say is what prints. Rescaling this figure in Illustrator
    ### re-breaks that, so change FIGURE_WIDTH_IN instead.
    FIGURE_WIDTH_IN = 7.2
    FIGURE_HEIGHT_IN = 7.0

    #setup the main figure size and layout
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    ### Three rows: [A + B], C, D. Panel A is a compact box at the top left
    ### with B beside it, matching how the other figures in the paper place
    ### their schematic. It was briefly a full-width banner, which reserved
    ### about a fifth of the figure height as white space for artwork that
    ### is added later anyway.
    gs_main = fig.add_gridspec(3, 1, height_ratios=[0.72, 0.92, 0.92],
                                hspace=0.5)

    # --- Row 1: Panel A (compact schematic box) + Panel B (arrows/polar) ---
    gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[0.6, 3.4], wspace=0.25)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.axis("off")
    ax_a.add_patch(plt.Rectangle(
        (0.0, 0.05), 1.0, 0.90, fill=False, ls="--", lw=1, color="0.6",
        transform=ax_a.transAxes,
    ))
    ax_a.text(0.5, 0.5, "Panel A\n(task\nschematic)", ha="center", va="center",
              fontsize=FONT_SIZE_TICK, color="0.5", transform=ax_a.transAxes)
    _panel_letter(ax_a, "A", dx=-10, dy=2)

    # Arrow + polar side by side per condition (Left arrow, Left polar,
    # Right arrow, Right polar) instead of stacked -- a 2x2 stack left a lot
    # of empty vertical space since quiver panels are wide-but-short and
    # polar panels are compact, so GridSpec was giving both rows equal
    # height regardless.
    gs_b = gs_top[0, 1].subgridspec(1, 4, wspace=0.75)
    ax_b_arrow_l = fig.add_subplot(gs_b[0, 0])
    ax_b_polar_l = fig.add_subplot(gs_b[0, 1], polar=True)
    ax_b_arrow_r = fig.add_subplot(gs_b[0, 2])
    ax_b_polar_r = fig.add_subplot(gs_b[0, 3], polar=True)

    _draw_quiver_panel(ax_b_arrow_l, quiver["Left"], "Left")
    _draw_quiver_panel(ax_b_arrow_r, quiver["Right"], "Right")
    _draw_polar_panel(ax_b_polar_l, session_result["left_angle"],
                       session_result["reward_angle"], 180, POLAR_HIST_COLOR)
    _draw_polar_panel(ax_b_polar_r, session_result["right_angle"],
                       session_result["reward_angle"], 0, POLAR_HIST_COLOR)
    _panel_letter(ax_b_arrow_l, "B")

    # --- Rows 3 & 4: Panel C (one session) and Panel D (population) ---
    ### Both rows use the same width_ratios, the same wspace AND the same
    ### x-limits, so column 1 (latency histogram) and column 2 (accuracy vs.
    ### latency) can be read straight down from the example session to the
    ### population. Previously column 2 was the CDF in row C but accuracy in
    ### row D, and column 3 was accuracy in C but the validity block in D --
    ### the columns invited a vertical comparison they did not support. The
    ### session CDF is dropped and column 3 of row C now holds the legend.
    ROW_CD_WIDTH_RATIOS = [1, 1, 1]
    ROW_CD_WSPACE = 0.45

    x_hi = max(
        float(session_result["reward_window"]),
        float(pooled_all["reward_window"]),
        float(lo["latencies"].max()) if lo["latencies"].size else 0.0,
        float(pooled_lo["latencies"].max()) if pooled_lo["latencies"].size else 0.0,
    )

    gs_c = gs_main[1].subgridspec(1, 3, width_ratios=ROW_CD_WIDTH_RATIOS,
                                  wspace=ROW_CD_WSPACE)
    ax_c_hist = fig.add_subplot(gs_c[0, 0])
    ax_c_acc = fig.add_subplot(gs_c[0, 1])
    ax_legend = fig.add_subplot(gs_c[0, 2])
    ax_legend.axis("off")

    _draw_latency_hist(ax_c_hist, lo["latencies"], lo["congruent"],
                        session_result["reward_window"], x_hi)
    _draw_accuracy_vs_latency(ax_c_acc, lo["latencies"], lo["congruent"],
                               session_result["reward_window"], x_hi)
    _panel_letter(ax_c_hist, "C")

    gs_d_outer = gs_main[2].subgridspec(1, 3, width_ratios=ROW_CD_WIDTH_RATIOS,
                                        wspace=ROW_CD_WSPACE)
    ax_d_hist = fig.add_subplot(gs_d_outer[0, 0])
    ax_d_acc = fig.add_subplot(gs_d_outer[0, 1])

    ### Two sub-columns, not three. With the middle all-trials accuracy
    ### column gone they can breathe a little, so wspace goes up.
    gs_d_validity = gs_d_outer[0, 2].subgridspec(1, 2, wspace=0.25)
    ax_d_valid = fig.add_subplot(gs_d_validity[0, 0])
    ax_d_window = fig.add_subplot(gs_d_validity[0, 1])

    _draw_latency_hist(ax_d_hist, pooled_lo["latencies"], pooled_lo["congruent"],
                        pooled_all["reward_window"], x_hi)
    _draw_accuracy_vs_latency(ax_d_acc, pooled_lo["latencies"], pooled_lo["congruent"],
                               pooled_all["reward_window"], x_hi)
    _draw_validity_columns((ax_d_valid, ax_d_window),
                            session_validity, animal_pooled, session_results)
    _panel_letter(ax_d_hist, "D")

    ### The legend now sits in Panel C's freed third slot, beside the panels
    ### whose colors it explains, instead of floating in the far top-right
    ### corner of the page. Window entries appear only when those windows
    ### were actually drawn, so the legend can never advertise shading that
    ### is not on the figure (see _window_is_informative).
    legend_handles = [
        plt.Line2D([0], [0], marker="o", ls="none", color=CORRECT_COLOR,
                   label="toward target (correct)"),
        plt.Line2D([0], [0], marker="o", ls="none", color=INCORRECT_COLOR,
                   label="away from target (incorrect)"),
        plt.Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor=PRECUE_COLOR, label="pre-cue control (D)"),
    ]
    if _window_is_informative((0, session_result["reward_window"]), x_hi):
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, color=REWARD_COLOR, alpha=0.5, label="reward window"))
    legend_handles.append(plt.Line2D(
        [0], [0], color=ACCURACY_WINDOW_COLOR, ls="--", lw=0.9,
        label=f"accuracy window ({ACCURACY_WINDOW[0]:.1f}-{ACCURACY_WINDOW[1]:.1f} s)"))
    ax_legend.legend(handles=legend_handles, loc="center left",
                     fontsize=FONT_SIZE_LABEL, frameon=False,
                     handletextpad=0.6, borderaxespad=0.0)

    return fig


################################################### main call to build the figure and save it in the right place/format

def build_supplement_figure(experiment_type: str):
    """Per-animal supplementary figure: one row per animal, three columns
    (latency histogram, latency CDF, accuracy vs. latency), each pooling all
    of that animal's sessions.

    Reads the same cache the main figure does, and uses the same drawing
    helpers, so the supplement can never drift out of style or method from
    the panels it supplements. ``animal_pooled[animal]`` is already the
    pooled result across that animal's sessions (built by
    :func:`prosaccade_population.pool_animal_sessions`), so nothing is
    re-pooled here.

    Every panel shares one x-limit across both animals, so the two rows are
    directly comparable -- the whole point of splitting them out.
    """
    _apply_paper_style()

    cache = _load_population_cache(experiment_type)
    animal_pooled = cache["animal_pooled"]
    session_validity = cache["session_validity"]
    if not animal_pooled:
        raise ValueError(
            f"No animals found in the cached population run for "
            f"experiment_type={experiment_type!r}."
        )

    animals = sorted(animal_pooled.keys())

    ### One shared x-limit across every panel and both animals. Letting each
    ### row autoscale would make Apollo's and Paris's latency axes different
    ### widths, which is exactly the comparison this figure exists to make.
    x_hi = 0.0
    for animal in animals:
        pooled = animal_pooled[animal]
        lat = pooled["latency_outcome"]["latencies"]
        x_hi = max(x_hi, float(pooled["reward_window"]),
                   float(lat.max()) if lat.size else 0.0)

    n_rows = len(animals)
    ### Same printed width as the main figure, for the same reason: point
    ### sizes only carry between figures rendered at the same physical size.
    fig = plt.figure(figsize=(7.2, 2.2 * n_rows + 0.5))
    gs = fig.add_gridspec(n_rows, 3, wspace=0.45, hspace=0.7)

    letters = "ABCDEFGHIJKL"
    for row, animal in enumerate(animals):
        pooled = animal_pooled[animal]
        lo = pooled["latency_outcome"]
        reward_window = pooled["reward_window"]
        n_sessions = len([r for r in session_validity
                          if r["animal_name"] == animal])

        ax_hist = fig.add_subplot(gs[row, 0])
        ax_cdf = fig.add_subplot(gs[row, 1])
        ax_acc = fig.add_subplot(gs[row, 2])

        _draw_latency_hist(ax_hist, lo["latencies"], lo["congruent"],
                            reward_window, x_hi)
        _draw_latency_cdf(ax_cdf, lo["latencies"], lo["congruent"],
                           reward_window, x_hi)
        _draw_accuracy_vs_latency(ax_acc, lo["latencies"], lo["congruent"],
                                   reward_window, x_hi)

        for col, ax in enumerate((ax_hist, ax_cdf, ax_acc)):
            _panel_letter(ax, letters[row * 3 + col])

        ### Animal identity goes on the row's left edge rather than in three
        ### repeated per-panel titles.
        ax_hist.text(-0.38, 0.5, f"{animal}\n({n_sessions} sessions)",
                     transform=ax_hist.transAxes, rotation=90,
                     ha="center", va="center", fontsize=FONT_SIZE_TITLE)

    legend_handles = [
        plt.Line2D([0], [0], color=CORRECT_COLOR, lw=1.4,
                   label="toward target (correct)"),
        plt.Line2D([0], [0], color=INCORRECT_COLOR, lw=1.4,
                   label="away from target (incorrect)"),
        plt.Line2D([0], [0], color=ACCURACY_WINDOW_COLOR, ls="--", lw=0.9,
                   label=f"accuracy window ({ACCURACY_WINDOW[0]:.1f}-{ACCURACY_WINDOW[1]:.1f} s)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=FONT_SIZE_LABEL, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    return fig


def _resolve_out_stem(explicit, default_name):
    """Output path stem: ``explicit`` when set, else ``<results_root>``
    from the manifest with ``default_name``."""
    if explicit:
        return Path(explicit)
    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "session_manifest.yml"
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    results_root = Path(manifest.get("results_root") or root_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root / default_name


def _save(fig, out_stem):
    """Save at the figure's declared size, NOT bbox_inches="tight".

    "tight" crops the canvas to its content, so the saved file came out at
    6.29 in wide rather than the 7.2 in the figure declares. Placing that
    at 7.2 in scales everything up by ~1.15x, and 6 pt tick labels print at
    ~6.9 pt -- which defeats the reason the figure is authored at its final
    size in the first place. Saving uncropped means the file is exactly
    FIGURE_WIDTH_IN wide and every point size on it is literal.

    The cost is some outer margin, which is trimmed on the artboard rather
    than by rescaling.
    """
    for ext in ("png", "svg"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


### Guarded so this module can be imported (by tests, or by another
### figure script wanting one of its helpers) without re-running the whole
### analysis and writing files as a side effect of the import. Running the
### script directly behaves exactly as before.
if __name__ == "__main__":
    fig = build_figure(SESSION_ID, EXPERIMENT_TYPE)
    _save(fig, _resolve_out_stem(OUTPUT_STEM,
                                 f"{EXPERIMENT_TYPE}_summary_figure"))

    ### Supplementary figure: the same latency/accuracy panels, per animal.
    fig_supp = build_supplement_figure(EXPERIMENT_TYPE)
    _save(fig_supp, _resolve_out_stem(SUPPLEMENT_OUTPUT_STEM,
                                      f"{EXPERIMENT_TYPE}_supplement_per_animal"))

    plt.show()


