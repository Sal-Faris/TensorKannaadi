from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from kannaadi.code_execution import CodeExecutionRequest, CodeSession


class FakeEngine:
    architecture = SimpleNamespace(model_id="test")

    def __init__(self) -> None:
        self._runs = []

    def list_runs(self):
        return list(self._runs)

    def get_run(self, _run_id):
        raise KeyError("no runs")


def session() -> CodeSession:
    engine = FakeEngine()
    adapter = SimpleNamespace(model=SimpleNamespace(name="fake-model"))
    return CodeSession(engine, adapter, engine.architecture, "test")


def request(code: str, *, trusted: bool = True) -> CodeExecutionRequest:
    return CodeExecutionRequest(code=code, cellId="cell-1", trusted=trusted, selection=["blocks.0.mlp"])


def test_code_session_requires_explicit_trust() -> None:
    with pytest.raises(PermissionError, match="explicit trust"):
        session().execute(request("1 + 1", trusted=False))


def test_code_session_persists_namespace_and_captures_last_expression() -> None:
    code = session()
    first = code.execute(request("value = 40\nprint('model ready')\nvalue + 2"))
    second = code.execute(request("_ + 1"))

    assert first.status == "complete"
    assert first.stdout == "model ready\n"
    assert first.artifact.kind == "json"
    assert first.artifact.data == 42
    assert second.artifact.data == 43
    assert "value" in second.namespace_keys
    assert second.provenance["selection"] == ["blocks.0.mlp"]


def test_code_session_publishes_structured_tables() -> None:
    result = session().execute(request("kannaadi.table([{'head': 'L0H0', 'effect': 1.25}], title='Head effects')"))

    assert result.artifact.kind == "table"
    assert result.artifact.title == "Head effects"
    assert result.artifact.data == [{"head": "L0H0", "effect": 1.25}]


def test_code_session_interrupts_python_execution_and_can_restart() -> None:
    code = session()
    captured = []
    worker = threading.Thread(target=lambda: captured.append(code.execute(request("while True:\n    pass"))))
    worker.start()
    deadline = time.monotonic() + 2
    while code.status().state == "idle" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert code.status().state == "running"
    code.interrupt()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert captured[0].status == "interrupted"
    assert code.restart().state == "idle"
    assert "value" not in code.status().namespace_keys
