"""Tests for the installable Panel frontend entry point."""

from types import SimpleNamespace

from pynamit.gui.cli import build_arg_parser, default_websocket_origins


def _panel_app_without_loading(tmp_path, monkeypatch):
    """Build Panel controls without requiring a saved simulation."""
    from pynamit.gui.panel_app import PynamitGUI

    monkeypatch.setattr(PynamitGUI, "_load_simulation", lambda self, event=None: None)
    app = PynamitGUI(simulation_directory=tmp_path)
    app._test_layout = app.panel()
    app._test_layout.get_root()
    return app


def test_pynamit_gui_parser_defaults_to_auto_detection():
    """The GUI auto-detects a run when no directory is given."""
    args = build_arg_parser().parse_args([])

    assert args.simulation_directory is None
    assert args.port == 5006
    assert args.route == "/pynamit"
    assert args.show is True


def test_pynamit_gui_parser_accepts_remote_no_show_options():
    """The GUI command should support remote/headless serving."""
    args = build_arg_parser().parse_args(
        ["simulation-a", "--address", "0.0.0.0", "--port", "6006", "--no-show"]
    )

    assert args.simulation_directory == "simulation-a"
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


def test_panel_defaults_to_showing_noninductive_results():
    """New plots include the available non-inductive comparison."""
    from pynamit.plotting.figure_settings import FigureSettings

    settings = FigureSettings()
    assert settings.simulation_directory == "."
    assert settings.time_range == (0, 0)
    assert settings.show_noninductive is True


def test_panel_manual_scales_start_from_field_presets(tmp_path, monkeypatch):
    """Manual controls follow the selected fields' presets."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)

    assert app.color_scale_mode.options == {"Manual": "manual", "Percentile": "percentile"}
    assert app.color_scale_mode.value == "manual"
    assert app.manual_color_min.name == "Color min (nT)"
    assert app.manual_color_max.name == "Color max (nT)"
    assert app.manual_color_min.value == -85.0
    assert app.manual_color_max.value == 85.0

    app.fill.value = "jr"
    app.lines.value = "Phi"

    assert app.manual_color_min.name == "Color min (µA/m²)"
    assert app.manual_color_max.name == "Color max (µA/m²)"
    assert app.manual_color_min.value == -0.85
    assert app.manual_color_max.value == 0.85
    assert app.line_first_abs_level.value == 4.0
    assert app.line_interval.value == 8.0
    assert app.line_levels_per_sign.value == 21

    app.lines.value = "W"

    assert app.line_first_abs_level.value == 4.0
    assert app.line_interval.value == 8.0
    assert app.line_levels_per_sign.value == 5


def test_panel_manual_scale_values_enter_figure_settings(tmp_path, monkeypatch):
    """Editable plot scales remain reproducible in exports."""
    from pynamit.gui.figure_settings_binding import current_figure_settings

    app = _panel_app_without_loading(tmp_path, monkeypatch)
    app.fill.value = "jr"
    app.manual_color_min.value = -0.5
    app.manual_color_max.value = 0.5
    app.line_first_abs_level.value = 4.0
    app.line_interval.value = 4.0
    app.line_levels_per_sign.value = 8

    settings = current_figure_settings(app)

    assert settings.manual_color_min == -5e-7
    assert settings.manual_color_max == 5e-7
    assert settings.line_first_abs_level == 4.0
    assert settings.line_interval == 4.0
    assert settings.line_levels_per_sign == 8


def test_panel_output_controls_display_absolute_paths(tmp_path, monkeypatch):
    """Output fields show exactly where files will be written."""
    monkeypatch.chdir(tmp_path)
    app = _panel_app_without_loading(tmp_path, monkeypatch)

    assert app.output_filename.value == str(tmp_path / "pynamit_figure.png")
    assert app.movie_filename.value == str(tmp_path / "pynamit_movie.gif")


def test_panel_confirms_before_overwriting_figure(tmp_path, monkeypatch):
    """Do not touch a figure until overwrite is confirmed."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)
    output_path = tmp_path / "existing.png"
    output_path.write_text("old", encoding="utf-8")
    writes = []

    class _Figure:
        def savefig(self, path, **kwargs):
            writes.append((path, kwargs))
            path.write_text("new", encoding="utf-8")

    app.figure = _Figure()
    app.output_filename.value = str(output_path)

    app._save_figure()

    assert output_path.read_text(encoding="utf-8") == "old"
    assert not writes
    assert app.overwrite_modal.open is True
    assert str(output_path) in app.overwrite_message.object

    app._confirm_overwrite()

    assert output_path.read_text(encoding="utf-8") == "new"
    assert writes[0][0] == output_path
    assert app.overwrite_modal.open is False
    assert app._pending_overwrite is None


def test_panel_can_cancel_movie_overwrite(tmp_path, monkeypatch):
    """Cancelling preserves the existing movie."""
    from pynamit.gui import panel_app

    app = _panel_app_without_loading(tmp_path, monkeypatch)
    output_path = tmp_path / "existing.gif"
    output_path.write_bytes(b"old")
    writes = []
    monkeypatch.setattr(
        panel_app, "save_movie", lambda *args, **kwargs: writes.append((args, kwargs))
    )
    app.movie_filename.value = str(output_path)

    app._save_movie()
    app._cancel_overwrite()

    assert output_path.read_bytes() == b"old"
    assert not writes
    assert app.overwrite_modal.open is False
    assert app._pending_overwrite is None


def test_panel_default_simulation_directory_finds_workflow_children(tmp_path, monkeypatch):
    """GUI auto-detection should find workflow children."""
    from pynamit.gui.panel_app import _default_simulation_directory

    simulation_directory = tmp_path / "results" / "N50_M50_Ncs50"
    simulation_directory.mkdir(parents=True)
    (simulation_directory / "settings.ncdf").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _default_simulation_directory() == str(simulation_directory.relative_to(tmp_path))


def test_panel_default_simulation_directory_finds_mage_simulation(tmp_path, monkeypatch):
    """GUI auto-detection should follow the MAGE case layout."""
    from pynamit.gui.panel_app import _default_simulation_directory

    simulation_directory = (
        tmp_path
        / "mage_output"
        / "case"
        / "resolutions"
        / "N50_M50_Ncs50"
        / "simulations"
        / "default"
    )
    simulation_directory.mkdir(parents=True)
    (simulation_directory / "settings.ncdf").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _default_simulation_directory() == str(simulation_directory.relative_to(tmp_path))


def test_panel_simulation_preserves_the_prepared_input_main_field(tmp_path, monkeypatch):
    """The Panel must preserve an input package's main field."""
    from pynamit.gui.panel_app import PynamitGUI
    from pynamit.simulation.config import INTEGRATORS

    captured = {}

    def fake_run_from_inputs(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(simulation_directory=str(tmp_path / "simulation"))

    monkeypatch.setattr(
        "pynamit.workflows.prepared_inputs.run_from_inputs",
        fake_run_from_inputs,
    )
    app = PynamitGUI(simulation_directory=tmp_path)
    monkeypatch.setattr(app, "_load_simulation", lambda: None)
    app.simulation_input_directory.value = str(tmp_path / "inputs")
    app.new_simulation_directory.value = str(tmp_path / "simulation")

    assert list(app.sim_integrator.options) == list(INTEGRATORS.values())

    app._run_simulation()

    assert captured["args"] == (tmp_path / "inputs",)
    assert "main_field_kind" not in captured["kwargs"]
