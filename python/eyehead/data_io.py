"""Utility functions for loading session metadata."""
from __future__ import annotations

from pathlib import Path
import csv
import yaml
from typing import Iterable, Dict, Any, List


def load_metadata(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load session metadata from a YAML or CSV file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the metadata file. If ``None`` (default), the function
        searches for ``scripts/metadata.yaml`` relative to this module.

    Returns
    -------
    list of dict
        A list of metadata dictionaries, one per session.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "metadata.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        if isinstance(data, dict):
            sessions = list(data.values())
        elif isinstance(data, list):
            sessions = data
        else:
            raise ValueError("Unexpected YAML structure")
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sessions = [dict(row) for row in reader]
    else:
        raise ValueError(f"Unsupported metadata file extension: {path.suffix}")

    return sessions

