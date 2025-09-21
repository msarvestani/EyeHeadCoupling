from pathlib import Path
import copy
import sys

import pytest

pytest.importorskip("yaml")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import session_loader as session_loader_mod
from utils.session_loader import load_session, list_sessions_from_manifest


def test_load_session() -> None:
    cfg = load_session("session_01")
    assert cfg.session_id == "session_01"
    assert cfg.folder_path == Path("/data/session_01")
    assert cfg.results_dir == Path("X:/Analysis/EyeHeadCoupling/session_01")
    assert cfg.animal_name == "Paris"
    assert cfg.animal_id == "Tsh001"

    assert cfg.camera_side == "L"


def test_load_session_uses_manifest_default_and_override(monkeypatch) -> None:
    manifest_template = {
        "results_root": "/results",
        "max_interval_s": 2.5,
        "saccade_config": {"saccade_threshold": 3.0},
        "sessions": {
            "custom": {
                "session_path": "/data/custom",
                "params": {},
            }
        },
    }

    def fake_safe_load(_fh):
        return copy.deepcopy(manifest_template)

    monkeypatch.setattr(session_loader_mod.yaml, "safe_load", fake_safe_load)

    cfg = load_session("custom")
    assert cfg.params["max_interval_s"] == 2.5

    manifest_template["sessions"]["custom"]["params"] = {"max_interval_s": 3.0}

    cfg_override = load_session("custom")
    assert cfg_override.params["max_interval_s"] == 3.0


def test_list_sessions_from_manifest_exact_match() -> None:
    sessions = list_sessions_from_manifest("fixation")
    assert "session_01" not in sessions
    assert "session_02" in sessions


def test_list_sessions_from_manifest_prefix_match() -> None:
    sessions = list_sessions_from_manifest("fixation", match_prefix=True)
    assert "session_01" in sessions
    assert "session_02" in sessions


def test_list_sessions_from_manifest_filters_by_animal() -> None:
    sessions = list_sessions_from_manifest(
        "prosaccade", animal_name="Paris", match_prefix=True
    )
    assert "session_08" in sessions
    assert "session_09" in sessions


def test_list_sessions_from_manifest_animal_only_filter() -> None:
    sessions = list_sessions_from_manifest(animal_name="Paris")
    assert "session_01" in sessions
    assert "session_13" in sessions
