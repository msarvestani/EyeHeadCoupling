"""Analysis script for a single session.

This version accepts a ``session_id`` and derives all required
paths from :func:`load_session` instead of relying on hard coded
file-system locations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from session_loader import load_session

def main(session_id: str) -> None:
    """Run the analysis for a given session identifier."""
    session = load_session(session_id)
    folder_path = Path(session["session_path"])
    results_dir = Path(session["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # The rest of the analysis would operate on ``folder_path`` and
    # save any generated figures into ``results_dir``.  For now we simply
    # report the resolved paths so that the script remains functional
    # even when the full analysis pipeline is not available.
    print(f"Session path: {folder_path}")
    print(f"Results directory: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a recorded session")
    parser.add_argument("session_id", help="Identifier or path for the session")
    args = parser.parse_args()
    main(args.session_id)
