from support_categorizer.settings import EVAL_N, EVAL_SEED, Settings


class _NoWidgets:
    class widgets:
        @staticmethod
        def get(name):
            raise Exception(f"widget {name} not defined")


class _Widgets:
    class widgets:
        @staticmethod
        def get(name):
            return {"catalog": "cat", "schema": "sch", "llm_endpoint": "ep"}[name]


def test_fqn():
    assert Settings().fqn("raw_calls") == "workspace.support.raw_calls"


def test_volume_path():
    assert Settings().volume_path("a.gz") == "/Volumes/workspace/support/landing/a.gz"


def test_from_widgets_defaults():
    s = Settings.from_widgets(_NoWidgets)
    assert s == Settings()


def test_from_widgets_values():
    s = Settings.from_widgets(_Widgets)
    assert (s.catalog, s.schema, s.llm_endpoint) == ("cat", "sch", "ep")


def test_eval_constants():
    assert (EVAL_N, EVAL_SEED) == (300, 20261028)
