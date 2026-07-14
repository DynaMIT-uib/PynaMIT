"""Tests for the installable Panel frontend entry point."""

from types import SimpleNamespace

from pynamit.visualization.gui import build_arg_parser, default_websocket_origins


def test_pynamit_gui_parser_defaults_to_auto_detection():
    """The GUI auto-detects a run when no directory is given."""
    args = build_arg_parser().parse_args([])

    assert args.run_directory is None
    assert args.port == 5006
    assert args.route == "/pynamit"
    assert args.show is True


def test_pynamit_gui_parser_accepts_remote_no_show_options():
    """The GUI command should support remote/headless serving."""
    args = build_arg_parser().parse_args(
        ["run-a", "--address", "0.0.0.0", "--port", "6006", "--no-show"]
    )

    assert args.run_directory == "run-a"
    assert args.address == "0.0.0.0"
    assert args.port == 6006
    assert args.show is False


def test_default_websocket_origins_allow_localhost_and_loopback():
    """Local serving should work for localhost and 127.0.0.1."""
    origins = default_websocket_origins("127.0.0.1", 5006)

    assert origins == ["localhost:5006", "127.0.0.1:5006"]


def test_default_websocket_origins_keep_explicit_origins():
    """Explicit websocket origins should be kept."""
    origins = default_websocket_origins("0.0.0.0", 6006, ["myhost.example:6006", "localhost:6006"])

    assert origins == ["localhost:6006", "127.0.0.1:6006", "0.0.0.0:6006", "myhost.example:6006"]


def test_panel_default_run_directory_finds_workflow_children(tmp_path, monkeypatch):
    """GUI auto-detection should find workflow children."""
    from pynamit.visualization.panel_app import _default_run_directory

    run_dir = tmp_path / "results" / "N50_M50_Ncs50"
    run_dir.mkdir(parents=True)
    (run_dir / "settings.ncdf").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _default_run_directory() == str(run_dir.relative_to(tmp_path))


def test_panel_run_preserves_the_prepared_input_main_field(tmp_path, monkeypatch):
    """The Panel must preserve an input package's main field."""
    from pynamit.simulation.config import INTEGRATORS
    from pynamit.visualization.panel_app import PynamitPanelApp

    captured = {}

    def fake_run_pynamit_from_inputs(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(run_directory=str(tmp_path / "run"))

    monkeypatch.setattr(
        "pynamit.simulation.workflows.prepared_inputs.run_pynamit_from_inputs",
        fake_run_pynamit_from_inputs,
    )
    app = PynamitPanelApp(run_directory=tmp_path)
    monkeypatch.setattr(app, "_load_run", lambda: None)
    app.simulation_input_directory.value = str(tmp_path / "inputs")
    app.simulation_run_directory.value = str(tmp_path / "run")

    assert list(app.sim_integrator.options) == list(INTEGRATORS.values())

    app._run_simulation()

    assert captured["args"] == (tmp_path / "inputs",)
    assert "main_field_kind" not in captured["kwargs"]
