from pathlib import Path
import sys

import pytest

pytest.importorskip("yaml")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.session_loader import load_session, list_sessions_from_manifest


def test_load_session() -> None:
    cfg = load_session("session_01")
    assert cfg.session_id == "session_01"
    assert cfg.folder_path == Path("/data/session_01")
    assert cfg.results_dir == Path("X:/Analysis/EyeHeadCoupling/session_01")
    assert cfg.animal_name == "Paris"
    assert cfg.animal_id == "Tsh001"

    assert cfg.camera_side == "L"


def test_list_sessions_from_manifest_exact_match() -> None:
    sessions = list_sessions_from_manifest("fixation")
    assert "session_01" not in sessions
    assert "session_02" in sessions


def test_list_sessions_from_manifest_prefix_match() -> None:
    sessions = list_sessions_from_manifest("fixation", match_prefix=True)
    assert "session_01" in sessions
    assert "session_02" in sessions
