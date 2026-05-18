"""Analyse a single fixation-with-feedback session (psychometric curve).

Usage
-----
Via manifest session ID::

    python Python/analysis/fixationfeedback_session.py Tsh001_2026-01-08T13_07_23

Via direct folder path::

    python Python/analysis/fixationfeedback_session.py --folder /path/to/folder --animal Tsh001
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session
from eyehead.io import clean_csv
from fixation_session import plot_psychometric_central_fixation, bonsai_to_deg


def load_fixation_feedback_data(folder_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load end-of-trial and target CSVs from a fixation feedback session folder.

    Returns
    -------
    (eot_df, target_df)
        eot_df  – one row per trial, always has 'trial_success'; has 'diameter'
                  when recorded there.
        target_df – one row per trial, always has 'diameter'.
    """
    csv_files = list(folder_path.glob("*.csv"))

    endoftrial_file = None
    target_file = None
    for f in csv_files:
        name = f.name.lower()
        if "endoftrial" in name:
            endoftrial_file = f
        elif "vstim_cue" in name or "target" in name:
            target_file = f

    if endoftrial_file is None:
        raise FileNotFoundError(f"Could not find endoftrial CSV in {folder_path}")

    # --- end-of-trial ---
    print(f"\nLoading {endoftrial_file.name}...")
    eot_arr = np.genfromtxt(clean_csv(str(endoftrial_file)), delimiter=",", skip_header=1, dtype=float)
    if eot_arr.ndim == 1:
        eot_arr = eot_arr.reshape(1, -1)
    n = eot_arr.shape[1]
    if n >= 7:
        eot_df = pd.DataFrame(eot_arr, columns=["frame", "timestamp", "trial_success",
                                                  "trial_number", "green_x", "green_y", "diameter"])
    elif n == 6:
        eot_df = pd.DataFrame(eot_arr, columns=["frame", "timestamp", "trial_success",
                                                  "green_x", "green_y", "diameter"])
    elif n >= 3:
        eot_df = pd.DataFrame({"frame": eot_arr[:, 0],
                                "timestamp": eot_arr[:, 1],
                                "trial_success": eot_arr[:, 2]})
    else:
        raise ValueError(f"Unexpected endoftrial column count: {n}")
    eot_df["frame"] = eot_df["frame"].astype(int)
    eot_df["trial_success"] = eot_df["trial_success"].astype(int)
    print(f"  Loaded {len(eot_df)} end-of-trial events")

    # --- target (diameter source) ---
    if target_file is not None:
        print(f"\nLoading {target_file.name}...")
        t_arr = np.genfromtxt(clean_csv(str(target_file)), delimiter=",", skip_header=1, dtype=float)
        if t_arr.ndim == 1:
            t_arr = t_arr.reshape(1, -1)
        nt = t_arr.shape[1]
        col_names = ["frame", "timestamp", "target_x", "target_y", "diameter", "visible"]
        target_df = pd.DataFrame(t_arr, columns=col_names[:nt])
        target_df["diameter"] = target_df["diameter"].astype(float)
        # One row per trial – drop duplicate frames
        if "frame" in target_df.columns:
            target_df = target_df.drop_duplicates(subset=["frame"], keep="first").reset_index(drop=True)
        print(f"  Loaded {len(target_df)} target rows")
    elif "diameter" in eot_df.columns:
        # Diameter recorded in endoftrial file – use it directly
        target_df = eot_df[["diameter"]].copy().reset_index(drop=True)
    else:
        raise FileNotFoundError(
            f"No target CSV and no 'diameter' column in endoftrial – cannot build psychometric curve. "
            f"Folder: {folder_path}"
        )

    return eot_df, target_df



def compute_success_trial_times(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> np.ndarray:
    """Return array of trial durations (s) for successful trials only.

    Returns empty array when required columns are unavailable.
    """
    if (
        "trial_success" not in eot_df.columns
        or "timestamp" not in eot_df.columns
        or "timestamp" not in target_df.columns
    ):
        return np.array([], dtype=float)

    n = min(len(eot_df), len(target_df))
    eot_ts = pd.to_numeric(eot_df["timestamp"], errors="coerce").to_numpy()
    tgt_ts = pd.to_numeric(target_df["timestamp"], errors="coerce").to_numpy()
    durations = eot_ts[:n] - tgt_ts[:n]

    success_mask = eot_df["trial_success"].iloc[:n].values == 2
    return durations[success_mask & ~np.isnan(durations)]


def compute_trial_times_by_diameter(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> dict[float, tuple[float, float, int]]:
    """Return {diameter_deg: (mean_s, sd_s, n)} for successful trials grouped by diameter.

    Returns empty dict when timestamps or diameter data are unavailable.
    """
    if (
        "trial_success" not in eot_df.columns
        or "timestamp" not in eot_df.columns
        or "timestamp" not in target_df.columns
        or "diameter" not in target_df.columns
    ):
        return {}

    n = min(len(eot_df), len(target_df))
    eot_ts = pd.to_numeric(eot_df["timestamp"], errors="coerce").to_numpy()
    tgt_ts = pd.to_numeric(target_df["timestamp"], errors="coerce").to_numpy()
    durations = eot_ts[:n] - tgt_ts[:n]

    combined = pd.DataFrame({
        "trial_success": eot_df["trial_success"].iloc[:n].values,
        "diameter": target_df["diameter"].iloc[:n].values,
        "duration": durations,
    })
    success = combined[combined["trial_success"] == 2]

    result: dict[float, tuple[float, float, int]] = {}
    for diam, group in success.groupby("diameter"):
        diam_key = round(float(bonsai_to_deg(diam)), 3)
        times = group["duration"].dropna().values
        if len(times) == 0:
            continue
        result[diam_key] = (
            float(np.mean(times)),
            float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            len(times),
        )
    return result


def plot_trial_time_session(
    eot_df: pd.DataFrame,
    target_df: pd.DataFrame,
    results_dir: Path,
    animal_id: str = "",
    date_str: str = "",
    show_plots: bool = True,
) -> Optional[plt.Figure]:
    """Bar chart of mean successful trial time per target diameter for one session."""
    times_by_diam = compute_trial_times_by_diameter(eot_df, target_df)
    if not times_by_diam:
        return None

    diameters = sorted(times_by_diam.keys())
    means = [times_by_diam[d][0] for d in diameters]
    sds = [times_by_diam[d][1] for d in diameters]
    ns = [times_by_diam[d][2] for d in diameters]

    fig, ax = plt.subplots(figsize=(max(5, len(diameters) * 1.2), 4))
    ax.bar(range(len(diameters)), means, yerr=sds, color="steelblue",
           edgecolor="white", alpha=0.8, capsize=4)
    ax.set_xticks(range(len(diameters)))
    ax.set_xticklabels([f"{d:.1f}°\n(n={n})" for d, n in zip(diameters, ns)], fontsize=9)
    ax.set_xlabel("Target diameter (°)")
    ax.set_ylabel("Mean trial time (s)")
    title = "Successful trial time by target diameter"
    if animal_id:
        title += f" – {animal_id}"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    fig.tight_layout()

    prefix = f"{animal_id}_" if animal_id else ""
    date_tag = f"_{date_str}" if date_str else ""
    for ext in ("png", "svg"):
        fig.savefig(results_dir / f"{prefix}trial_time_by_diameter{date_tag}.{ext}", bbox_inches="tight")

    if show_plots:
        plt.show()
    plt.close(fig)
    return fig


def analyze_session(
    folder_path: str | Path,
    results_dir: Optional[str | Path] = None,
    animal_id: str = "",
    show_plots: bool = True,
) -> pd.DataFrame:
    """Run fixation feedback analysis for one session folder.

    Returns
    -------
    pd.DataFrame
        One-row summary with trial counts and overall success rate.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if results_dir is None:
        results_dir = folder_path / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {folder_path}")
    print(f"Results:   {results_dir}")

    folder_name = folder_path.name

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", folder_name)
    date_str = date_match.group() if date_match else ""

    session_time = None
    time_match = re.search(r"\d{4}-\d{2}-\d{2}T(\d{2})_(\d{2})_\d{2}", folder_name)
    if time_match:
        session_time = f"{int(time_match.group(1)):02d}:{int(time_match.group(2)):02d}"

    if not animal_id:
        m = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}", folder_name)
        if m:
            animal_id = m.group(1)

    eot_df, target_df = load_fixation_feedback_data(folder_path)

    print("\nGenerating psychometric curve...")
    fig = plot_psychometric_central_fixation(
        eot_df, target_df, results_dir, animal_id, date_str, session_time
    )
    if fig is not None:
        if show_plots:
            plt.show()
        plt.close(fig)

    print("\nGenerating trial time by diameter plot...")
    plot_trial_time_session(
        eot_df, target_df, results_dir, animal_id, date_str, show_plots=show_plots
    )

    return pd.DataFrame({
        "session_id": [folder_name],
        "animal_id": [animal_id],
        "session_date": [date_str],
        "session_time": [session_time],
    })


def main(session_id: str, show_plots: bool = True) -> pd.DataFrame:
    """Run analysis for a manifest session ID."""
    config = load_session(session_id)
    if config.folder_path is None:
        raise ValueError(f"Session '{session_id}' has no folder_path in manifest")
    return analyze_session(
        config.folder_path,
        results_dir=config.results_dir,
        animal_id=config.animal_id or "",
        show_plots=show_plots,
    )


def plot_population(session_dfs: list[pd.DataFrame]) -> plt.Figure:
    """Plot overall success rate across sessions.

    Parameters
    ----------
    session_dfs:
        List of DataFrames returned by :func:`analyze_session` or :func:`main`,
        one per session.

    Returns
    -------
    matplotlib.figure.Figure
    """
    combined = pd.concat(session_dfs, ignore_index=True)
    combined = combined.sort_values(["animal_id", "session_date", "session_time"])

    fig, ax = plt.subplots(figsize=(max(8, len(combined) * 0.8), 5))
    labels = combined.apply(
        lambda r: f"{r['animal_id']}\n{r['session_date']}" if r["session_date"]
        else str(r["session_id"]), axis=1
    )
    ax.bar(range(len(combined)), combined["success_rate"] * 100, color="steelblue", edgecolor="navy")
    ax.set_xticks(range(len(combined)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Overall Success Rate (%)", fontsize=13, fontweight="bold")
    ax.set_title("Fixation Feedback – Population Summary", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse a fixation-with-feedback session (psychometric curve)"
    )
    parser.add_argument("session_id", nargs="?",
                        help="Session identifier from session_manifest.yml")
    parser.add_argument("--folder", type=str,
                        help="Direct path to data folder (alternative to session_id)")
    parser.add_argument("--animal", type=str, default="",
                        help="Animal ID (used in --folder mode if not parseable from folder name)")
    parser.add_argument("--results", type=str,
                        help="Results directory (for --folder mode)")
    parser.add_argument("--no-show", dest="show_plots", action="store_false", default=True,
                        help="Suppress interactive plot display")
    args = parser.parse_args()

    if args.folder:
        analyze_session(args.folder, args.results, args.animal, show_plots=args.show_plots)
    elif args.session_id:
        main(args.session_id, show_plots=args.show_plots)
    else:
        parser.print_help()
