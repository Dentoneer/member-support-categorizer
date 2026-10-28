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


class _WidgetStore:
    def __init__(self):
        self.vals = {}
        self.widgets = self

    def text(self, name, default):
        self.vals.setdefault(name, default)

    def get(self, name):
        return self.vals[name]


def test_notebook_settings_uses_saved_bake_off_winner(tmp_path):
    from support_categorizer.settings import notebook_settings, save_chosen_llm

    path = str(tmp_path / "chosen.txt")
    assert notebook_settings(_WidgetStore(), chosen_path=path).llm_endpoint == Settings().llm_endpoint
    save_chosen_llm(Settings(), "databricks-qwen35-122b-a10b", path=path)
    assert notebook_settings(_WidgetStore(), chosen_path=path).llm_endpoint == "databricks-qwen35-122b-a10b"
    typed = _WidgetStore()
    typed.vals["llm_endpoint"] = "my-endpoint"  # a value typed in the box always wins
    assert notebook_settings(typed, chosen_path=path).llm_endpoint == "my-endpoint"
