from pathlib import Path
import copy
import sys

import pytest

pytest.importorskip("yaml")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import session_loader as session_loader_mod
from utils.session_loader import load_session, list_sessions_from_manifest


def test_load_session() -> None:
    # Key is the folder name (timestamp); animal_id, animal_name, date are derived from session_path.
    session_key = "Tsh001_2025-07-08T15_04_12"
    cfg = load_session(session_key)
    assert cfg.session_id == session_key
    assert cfg.animal_id == "Tsh001"
    assert cfg.animal_name == "Paris"
    assert cfg.camera_side == "L"
    assert cfg.params.get("date") == "2025-07-08"


def test_load_session_uses_manifest_default_and_override(monkeypatch) -> None:
    manifest_template = {
        "results_root": "/results",
        "max_interval_s": 2.5,
        "saccade_config": {"saccade_threshold": 3.0},
        "sessions": {
            "Tsh001_2025-01-01T12_00_00": {
                "session_path": "/data/TSh01_Paris_server/Tsh001_2025-01-01T12_00_00",
                "params": {},
            }
        },
    }

    def fake_safe_load(_fh):
        return copy.deepcopy(manifest_template)

    monkeypatch.setattr(session_loader_mod.yaml, "safe_load", fake_safe_load)

    cfg = load_session("Tsh001_2025-01-01T12_00_00")
    assert cfg.params["max_interval_s"] == 2.5

    manifest_template["sessions"]["Tsh001_2025-01-01T12_00_00"]["params"] = {"max_interval_s": 3.0}

    cfg_override = load_session("Tsh001_2025-01-01T12_00_00")
    assert cfg_override.params["max_interval_s"] == 3.0


def test_load_session_derives_fields_from_path(monkeypatch) -> None:
    manifest = {
        "results_root": "/results",
        "sessions": {
            "Tsh002_2025-06-15T10_30_00": {
                "session_path": "/data/TSh02_Apollo_server/Tsh002_2025-06-15T10_30_00",
                "experiment_type": "fixation",
                "calibration_factor": 3.76,
                "ttl_freq": 60,
                "camera_side": "L",
            }
        },
    }

    monkeypatch.setattr(session_loader_mod.yaml, "safe_load", lambda _fh: copy.deepcopy(manifest))

    cfg = load_session("Tsh002_2025-06-15T10_30_00")
    assert cfg.animal_id == "Tsh002"
    assert cfg.animal_name == "Apollo"
    assert cfg.params.get("date") == "2025-06-15"


def test_list_sessions_from_manifest_exact_match() -> None:
    sessions = list_sessions_from_manifest("prosaccade")
    assert "Tsh001_2025-07-08T15_04_12" in sessions
    # antisaccade sessions should not appear in an exact "prosaccade" match
    assert "Tsh001_2025-08-12T15_04_14" not in sessions


def test_list_sessions_from_manifest_prefix_match(monkeypatch) -> None:
    manifest = {
        "sessions": {
            "Tsh001_2025-07-01T10_00_00": {
                "session_path": "/data/TSh01_Paris_server/Tsh001_2025-07-01T10_00_00",
                "experiment_type": "fixation-control",
            },
            "Tsh001_2025-07-02T10_00_00": {
                "session_path": "/data/TSh01_Paris_server/Tsh001_2025-07-02T10_00_00",
                "experiment_type": "fixation",
            },
        }
    }
    monkeypatch.setattr(session_loader_mod.yaml, "safe_load", lambda _fh: copy.deepcopy(manifest))

    sessions_exact = list_sessions_from_manifest("fixation")
    assert "Tsh001_2025-07-01T10_00_00" not in sessions_exact
    assert "Tsh001_2025-07-02T10_00_00" in sessions_exact

    sessions_prefix = list_sessions_from_manifest("fixation", match_prefix=True)
    assert "Tsh001_2025-07-01T10_00_00" in sessions_prefix
    assert "Tsh001_2025-07-02T10_00_00" in sessions_prefix


def test_list_sessions_from_manifest_filters_by_animal() -> None:
    sessions = list_sessions_from_manifest(
        "prosaccade", animal_name="Paris", match_prefix=True
    )
    assert "Tsh001_2025-07-08T15_04_12" in sessions
    assert "Tsh001_2025-07-02T15_14_29" in sessions


def test_list_sessions_from_manifest_animal_only_filter() -> None:
    sessions = list_sessions_from_manifest(animal_name="Paris")
    assert "Tsh001_2025-08-12T15_04_14" in sessions
    # Apollo sessions should not appear
    assert "Tsh002_2025-09-10T11_28_52" not in sessions
