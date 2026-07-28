"""Analyse a single fixation session with randomised left/right targets.

Produces two figures per session:
  1. Four-panel heatmap  – eye positions during all/successful left vs right trials.
  2. Two-panel fixation scatter – every eye sample in each trial's cue→go window,
     coloured green for successful trials and red for failed ones, split by target side.

Usage
-----
Via manifest session ID::

    python Python/analysis/fixation_lr_session.py Tsh001_2025-08-01T15_15_48

Via direct folder path::

    python Python/analysis/fixation_lr_session.py X:\\path\\to\\session_folder
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session, SessionConfig
from eyehead import get_session_date_from_path
from eyehead.io import clean_csv
from eyehead.analysis import _filename_with_animal

_BONSAI_X_RANGE = (-1.7, 1.7)
_BONSAI_Y_RANGE = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading (vstim_cue / vstim_go / endoftrial files)
# ---------------------------------------------------------------------------

def load_lr_trial_data(folder: Path):
    # --- find files ---
    csvs = {f.name.lower(): f for f in folder.glob("*.csv")}

    cue_file = next((v for k, v in csvs.items() if "vstim_cue" in k), None)
    go_file  = next((v for k, v in csvs.items() if "vstim_go"  in k), None)
    eot_file = next((v for k, v in csvs.items() if "endoftrial" in k), None)

    if cue_file is None:
        raise FileNotFoundError(f"No vstim_cue file found in {folder}")
    if go_file is None:
        raise FileNotFoundError(f"No vstim_go file found in {folder}")

    print(f"  cue  : {cue_file.name}")
    print(f"  go   : {go_file.name}")
    print(f"  eot  : {eot_file.name if eot_file else 'not found'}")

    # --- vstim_cue: frame, timestamp, target_x, target_y, ... ---
    cue_arr = np.genfromtxt(clean_csv(str(cue_file)), delimiter=",", skip_header=1)
    if cue_arr.ndim == 1:
        cue_arr = cue_arr.reshape(1, -1)
    _, idx = np.unique(cue_arr[:, 0].astype(int), return_index=True)
    cue_arr   = cue_arr[idx]
    cue_times = cue_arr[:, 1]
    target_x  = cue_arr[:, 2]   # negative = left, positive = right
    target_y        = cue_arr[:, 3]
    target_diameter = cue_arr[:, 4]
    n_trials  = len(cue_arr)
    print(f"  {n_trials} cue events  |  target_x range [{target_x.min():.3f}, {target_x.max():.3f}]")

    # --- vstim_go: frame, timestamp, green_x, green_y, ... ---
    go_arr = np.genfromtxt(clean_csv(str(go_file)), delimiter=",", skip_header=1)
    if go_arr.ndim == 1:
        go_arr = go_arr.reshape(1, -1)
    eye_ts = go_arr[:, 1]
    eye_x  = go_arr[:, 2]
    eye_y  = go_arr[:, 3]

    # --- endoftrial: trial_success == 2 means success ---
    success   = np.zeros(n_trials, dtype=bool)
    eot_times = None
    if eot_file is not None:
        eot_arr = np.genfromtxt(clean_csv(str(eot_file)), delimiter=",", skip_header=1)
        if eot_arr.ndim == 1:
            eot_arr = eot_arr.reshape(1, -1)
        eot_times   = eot_arr[:, 1]
        eot_success = eot_arr[:, 2].astype(int)
        n = min(n_trials, len(eot_success))
        success[:n] = eot_success[:n] == 2
        print(f"  {int(success.sum())}/{n_trials} successful trials")
    else:
        print("  No endoftrial file – success unknown, treating all as failed")

    # --- trial windows: cue_time[i] → eot_time[i] ---
    windows = []
    for i in range(n_trials):
        t_start = cue_times[i]
        if eot_times is not None and i < len(eot_times):
            t_end = eot_times[i]
        elif i + 1 < n_trials:
            t_end = cue_times[i + 1]
        else:
            t_end = eye_ts[-1]
        mask = (eye_ts >= t_start) & (eye_ts <= t_end)
        windows.append({"eye_x": eye_x[mask], "eye_y": eye_y[mask]})

    directions = np.array(["L" if dx < 0 else "R" for dx in target_x])
    return windows, directions, success, target_x, target_y, target_diameter


# ---------------------------------------------------------------------------
# Figure 1 – four-panel heatmap
# ---------------------------------------------------------------------------

def plot_heatmaps_lr(windows, directions, valid_trials, target_xs=None,target_ys=None,target_diameters=None, *, title_prefix=""):
    left    = directions == "L"
    right   = directions == "R"
    success = np.asarray(valid_trials, dtype=bool)

    def _mean_tx(mask):
        if target_xs is None or not np.any(mask):
            return None
        return float(np.mean(target_xs[mask]))

    def _mean_ty(mask):
        if target_ys is None or not np.any(mask):
            return None
        return float(np.mean(target_ys[mask]))

    def _mean_td(mask):
        if target_diameters is None or not np.any(mask):
            return None
        return float(np.mean(target_diameters[mask]))


    panel_specs = [
        (0, 0, left,            "All Left Trials",         _mean_tx(left),  _mean_ty(left),  _mean_td(left)),
        (0, 1, right,           "All Right Trials",        _mean_tx(right), _mean_ty(right), _mean_td(right)),
        (1, 0, left  & success, "Successful Left Trials",  _mean_tx(left),  _mean_ty(left),  _mean_td(left)),
        (1, 1, right & success, "Successful Right Trials", _mean_tx(right), _mean_ty(right), _mean_td(right)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, col, mask, label, tgt_x, tgt_y, tgt_d in panel_specs:
        ax    = axes[row, col]
        idxs  = np.where(mask)[0]
        if len(idxs):
            all_x = np.concatenate([windows[i]["eye_x"] for i in idxs])
            all_y = np.concatenate([windows[i]["eye_y"] for i in idxs])
        else:
            all_x = all_y = np.empty(0)

        n_t = int(mask.sum())
        n_s = len(all_x)

        if n_s >= 4:
            h, xedges, yedges = np.histogram2d(
                all_x, all_y, bins=50,
                range=[list(_BONSAI_X_RANGE), list(_BONSAI_Y_RANGE)],
            )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(h.T, extent=extent, origin="lower",
                           cmap="hot", aspect="equal", interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="Samples")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="0.5")

        if tgt_x is not None and tgt_y is not None and tgt_d is not None:
            from matplotlib.patches import Circle
            circle = Circle((tgt_x, tgt_y), radius=tgt_d / 2,
                            fill=False, edgecolor="cyan", linewidth=2, alpha=0.8)
            ax.add_patch(circle)

        ax.axhline(0, color="0.6", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="0.6", linewidth=0.6, linestyle=":")
        ax.set_xlim(*_BONSAI_X_RANGE)
        ax.set_ylim(*_BONSAI_Y_RANGE)
        ax.set_xlabel("Horizontal (Bonsai units)")
        ax.set_ylabel("Vertical (Bonsai units)")
        ax.set_title(f"{label}\n(n={n_t} trials, {n_s} samples)")

    suptitle = "Eye-position heatmaps – Left vs Right targets"
    if title_prefix:
        suptitle = f"{title_prefix} | {suptitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – fixation scatter coloured by success
# ---------------------------------------------------------------------------

def plot_fixation_scatter_lr(windows, directions, valid_trials,
                              target_xs=None, target_ys=None, target_diameters=None,
                              *, title_prefix=""):
    
    success = np.asarray(valid_trials, dtype=bool)   # ← this line is missing
    fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(5, 8))

    for ax, side, panel_label in [(ax_l, "L", "Left-target trials"),
                                   (ax_r, "R", "Right-target trials")]:
        n_ok = n_fail = 0
        for w, d, ok in zip(windows, directions, success):
            if d != side or len(w["eye_x"]) == 0:
                continue
            color = "green" if ok else "red"
            alpha = 0.35   if ok else 0.25
            ax.scatter(w["eye_x"], w["eye_y"], c=color, s=9, alpha=alpha,
                       linewidths=0, rasterized=True)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        
        # draw target circle for this side
        if target_xs is not None and target_ys is not None and target_diameters is not None:
            side_mask = directions == side
            if np.any(side_mask):
                from matplotlib.patches import Circle
                tx = float(np.mean(target_xs[side_mask]))
                ty = float(np.mean(target_ys[side_mask]))
                td = float(np.mean(target_diameters[side_mask]))
                circle = Circle((tx, ty), radius=td / 2,
                                fill=False, edgecolor="cyan", linewidth=2, alpha=0.8)
                ax.add_patch(circle)

        ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="0.7", linewidth=0.8, linestyle=":")
        ax.set_xlim(*_BONSAI_X_RANGE)
        ax.set_ylim(*_BONSAI_Y_RANGE)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal (Bonsai units)")
        ax.set_ylabel("Vertical (Bonsai units)")
        ax.set_title(f"{panel_label}\n(success={n_ok}, failed={n_fail})")
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                   markersize=8, label="Success"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="Failed"),
        ], loc="upper right", fontsize=9)

    suptitle = "Fixation positions (trial window) – Left vs Right targets"
    if title_prefix:
        suptitle = f"{title_prefix} | {suptitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run(config, *, show_plots=True):
    folder = config.folder_path
    if folder is None:
        raise ValueError("config.folder_path must be set")

    config.results_dir.mkdir(parents=True, exist_ok=True)

    date_str = config.params.get("date", "")
    if not date_str:
        try:
            date_str = get_session_date_from_path(str(folder)).strftime("%Y-%m-%d")
        except Exception:
            date_str = ""

    print(f"\nLoading data from {folder}")
    windows, directions, success, target_xs, target_ys, target_diameters = load_lr_trial_data(folder)

    animal_label = config.animal_name or config.animal_id
    id_part      = str(config.animal_id).strip() if config.animal_id else ""
    eye_part     = (config.eye_name or "Eye").replace(" ", "")
    title_prefix = f"{id_part} {date_str}".strip()

    # Figure 1: heatmaps
    fig_hm = plot_heatmaps_lr(windows, directions, success,
                               target_xs, target_ys, target_diameters,
                               title_prefix=title_prefix)
    stem = "_".join(p for p in (id_part, eye_part, "fixation_lr_heatmap") if p)
    for ext in ("png", "svg"):
        fname = _filename_with_animal(f"{stem}.{ext}", animal_label)
        fig_hm.savefig(config.results_dir / fname, bbox_inches="tight")
        print(f"Saved {config.results_dir / fname}")
    if show_plots:
        plt.show()
    plt.close(fig_hm)

    # Figure 2: fixation scatter
    fig_sc = plot_fixation_scatter_lr(windows, directions, success,
                                       target_xs, target_ys, target_diameters,
                                       title_prefix=title_prefix)

    stem = "_".join(p for p in (id_part, eye_part, "fixation_lr_scatter") if p)
    for ext in ("png", "svg"):
        fname = _filename_with_animal(f"{stem}.{ext}", animal_label)
        fig_sc.savefig(config.results_dir / fname, bbox_inches="tight")
        print(f"Saved {config.results_dir / fname}")
    if show_plots:
        plt.show()
    plt.close(fig_sc)

    n_left  = int(np.sum(directions == "L"))
    n_right = int(np.sum(directions == "R"))
    n_ok    = int(success.sum())
    print(
        f"\nSummary: {len(directions)} trials  "
        f"(left={n_left}, right={n_right})  "
        f"success={n_ok} ({100*n_ok/max(len(directions),1):.1f}%)"
    )


def _config_from_folder(folder: Path):
    sid = folder.name
    m = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}", sid)
    animal_id = m.group(1) if m else sid
    try:
        date_str = get_session_date_from_path(str(folder)).strftime("%Y-%m-%d")
    except Exception:
        date_str = ""
    return SessionConfig(
        session_id=sid,
        folder_path=folder,
        results_dir=folder / "results",
        animal_id=animal_id,
        camera_side="L",
        eye_name="Eye",
        calibration_factor=3.76,
        ttl_freq=60,
        params={"date": date_str},
    )


def main(session_id: str, show_plots: bool = True) -> None:
    config = load_session(session_id)
    _run(config, show_plots=show_plots)


# ---------------------------------------------------------------------------
# CLI  –  pass a session ID or a folder path, it auto-detects
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot left/right fixation heatmaps and scatter for a session."
    )
    parser.add_argument("session", help="Session ID from manifest, or path to session folder")
    parser.add_argument("--no-show", action="store_true", help="Save without displaying plots")
    args = parser.parse_args()

    p = Path(args.session)
    if p.is_dir():
        _run(_config_from_folder(p), show_plots=not args.no_show)
    else:
        main(args.session, show_plots=not args.no_show)