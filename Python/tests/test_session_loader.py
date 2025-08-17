from pathlib import Path
import sys

import pytest

pytest.importorskip("yaml")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session


def test_load_session() -> None:
    cfg = load_session("session_01")
    assert cfg.session_id == "session_01"
    assert cfg.folder_path == Path("/data/session_01")
    assert cfg.results_dir == Path("/data/session_01/results")
