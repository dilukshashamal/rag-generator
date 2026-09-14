from pathlib import Path
from typing import Any
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_configuration_state_does_not_expose_values() -> None:
    with patch("src.runtime.get_runtime", side_effect=ValueError("secret-key and internal path")):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        assert not app.exception
        assert "Configuration is incomplete" in app.warning[0].value
        assert "secret-key" not in str(app)


def test_navigation_and_empty_collection_view(harness: Any) -> None:
    from types import SimpleNamespace
    runtime = SimpleNamespace(repository=SimpleNamespace(collections=lambda: []),
                              telemetry=harness.events, settings=harness.settings)
    with patch("src.runtime.get_runtime", return_value=runtime):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        assert not app.exception
        assert "No collections yet" in app.info[0].value
        app.sidebar.radio[0].set_value("Ask Questions").run()
        assert not app.exception and "Select a knowledge base" in app.info[0].value
        app.sidebar.radio[0].set_value("Evaluation").run()
        assert not app.exception and "Select a knowledge base" in app.info[0].value


def test_collection_creation_selects_new_collection(harness: Any) -> None:
    from types import SimpleNamespace
    repo = harness.repo
    repo.collections = lambda: list(repo.cols.values())
    runtime = SimpleNamespace(repository=repo, telemetry=harness.events, settings=harness.settings)
    with patch("src.runtime.get_runtime", return_value=runtime):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.text_input[0].set_value("Another collection")
        next(button for button in app.button if button.label == "Create").click().run()
        assert not app.exception
        selected = app.sidebar.selectbox[0].value
        assert str(selected) == str(next(item.collection_id for item in repo.cols.values() if item.name == "Another collection"))
