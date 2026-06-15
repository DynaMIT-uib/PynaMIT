"""Package entry-point behavior."""

import runpy

import pynamit.default_run


def test_main_module_import_is_inert(monkeypatch):
    """Importing ``pynamit.__main__`` must not launch a simulation."""
    calls = []
    monkeypatch.setattr(pynamit.default_run, "run_pynamit", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="pynamit.__main_import_test__")

    assert calls == []


def test_main_module_executes_as_script(monkeypatch):
    """Executing ``pynamit.__main__`` preserves script behavior."""
    calls = []
    monkeypatch.setattr(pynamit.default_run, "run_pynamit", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="__main__")

    assert calls == [None]
