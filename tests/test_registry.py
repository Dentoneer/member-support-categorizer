import mlflow
import pytest

from support_categorizer.categorizer import CallCategorizerProgram
from support_categorizer.registry import load_program, register_program
from support_categorizer.settings import Settings

FLOWS = ["account_access"]
SUBS = {"reset_2fa": "account_access", "recover_password": "account_access"}


def _boom(*a, **k):
    raise RuntimeError("registry not available on this workspace")


def test_register_falls_back_to_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow.dspy, "log_model", _boom)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    path = str(tmp_path / "program.json")
    with mlflow.start_run():
        out = register_program(CallCategorizerProgram(FLOWS, SUBS), Settings(), {"x": 1},
                               fallback_path=path)
    assert out["path"] == "volume" and out["file"] == path


def test_load_program_from_volume_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow.dspy, "load_model", _boom)
    prog = CallCategorizerProgram(FLOWS, SUBS)
    path = str(tmp_path / "program.json")
    prog.save(path)
    loaded = load_program(Settings(), FLOWS, SUBS, fallback_path=path)
    assert isinstance(loaded, CallCategorizerProgram) and loaded.sub_to_flow == SUBS


def test_load_program_missing_everywhere_is_plain_english(tmp_path, monkeypatch):
    from support_categorizer.errors import UserFacingError

    monkeypatch.setattr(mlflow.dspy, "load_model", _boom)
    with pytest.raises(UserFacingError, match="notebook 03"):
        load_program(Settings(), FLOWS, SUBS, fallback_path=str(tmp_path / "nope.json"))
