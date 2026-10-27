from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "qsr_assistant" / "app" / "streamlit_app.py"


def test_streamlit_app_renders_and_answers(registry):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    at.chat_input[0].set_value("When do refunds arrive?").run()
    assert not at.exception
    assert any("3 to 5 business days" in m.markdown[0].value for m in at.chat_message if m.markdown)
