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


def test_should_promote():
    from support_categorizer.registry import should_promote

    assert should_promote(0.8, None) and should_promote(0.8, 0.7)
    assert not should_promote(0.7, 0.7) and not should_promote(0.6, 0.7)


def test_model_resources():
    from support_categorizer.registry import model_resources

    res = model_resources("databricks-meta-llama-3-3-70b-instruct", "workspace.support.category_guidelines_index")
    names = [type(r).__name__ for r in res]
    assert names == ["DatabricksServingEndpoint", "DatabricksVectorSearchIndex"]
    assert len(model_resources("ep")) == 1


def test_register_passes_alias_and_resources(tmp_path, monkeypatch):
    calls = {}

    class _Info:
        registered_model_version = 3

    class _Client:
        def set_registered_model_alias(self, name, alias, version):
            calls["alias"] = (alias, version)

        def set_model_version_tag(self, name, version, key, value):
            calls["tag"] = (key, value)

    def fake_log(program, **kw):
        calls["kw"] = kw
        return _Info()

    monkeypatch.setattr(mlflow.dspy, "log_model", fake_log)
    monkeypatch.setattr(mlflow, "MlflowClient", _Client)
    out = register_program(CallCategorizerProgram(FLOWS, SUBS), Settings(), {"x": 1}, alias="challenger",
                           resources=["r"], score=0.81234)
    assert out == {"path": "uc", "version": "3"}
    assert calls["alias"] == ("challenger", "3") and calls["tag"] == ("secondary_accuracy", "0.8123")
    assert calls["kw"]["resources"] == ["r"] and calls["kw"]["code_paths"]


def test_dspy_program_with_retriever_roundtrips_through_mlflow(tmp_path):
    from support_categorizer.retrieval import GuidelineRetriever

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_registry_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    prog = CallCategorizerProgram(FLOWS, SUBS, retriever=GuidelineRetriever("cat.sch.idx", k=5))
    with mlflow.start_run():
        info = mlflow.dspy.log_model(prog, name="model")
    loaded = mlflow.dspy.load_model(info.model_uri)
    assert loaded.retriever.index_name == "cat.sch.idx" and loaded.retriever.k == 5
    assert loaded.sub_to_flow == SUBS


def test_should_promote_replaces_a_champion_on_an_older_category_list():
    from support_categorizer.registry import should_promote

    assert should_promote(0.6, 0.8, new_taxonomy=3, champion_taxonomy=2)  # stale champion
    assert not should_promote(0.6, 0.8, new_taxonomy=3, champion_taxonomy=3)
    assert not should_promote(0.6, 0.8, new_taxonomy=3, champion_taxonomy=None)  # untagged: the score decides
