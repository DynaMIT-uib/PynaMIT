"""Tests for the installable Panel frontend entry point."""

import asyncio
import datetime
import json
from types import SimpleNamespace

import pytest

from pynamit.gui.cli import build_arg_parser, default_websocket_origins

# Panel 1.x requires these names while warning that Panel 2 will
# rename them. Keep the exception local so other warnings remain
# visible; remove it when Panel 2 is the minimum supported version.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Widget\\.name is deprecated and will be removed in version "
        "2\\.0\\..*:PendingDeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:button_type is deprecated and will be removed in version "
        "2\\.0\\..*:PendingDeprecationWarning"
    ),
]


def _panel_app_without_loading(tmp_path, monkeypatch):
    """Build Panel controls without requiring a saved simulation."""
    from pynamit.gui.panel_app import PynamitGUI

    monkeypatch.setattr(PynamitGUI, "_load_simulation", lambda self, event=None: None)
    app = PynamitGUI(simulation_directory=tmp_path)
    app._test_layout = app.panel()
    app._test_layout.get_root()
    return app


def test_pynamit_gui_parser_allows_an_unspecified_directory():
    """The constructor resolves an omitted directory."""
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
    assert settings.show_equilibrium is True


def test_panel_manual_scales_start_from_field_presets(tmp_path, monkeypatch):
    """Manual controls follow the selected fields' presets."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)

    assert app.color_scale_mode.options == {"Manual": "manual", "Percentile": "percentile"}
    assert app.color_scale_mode.value == "percentile"
    assert app.manual_color_min.label == "Color min (nT)"
    assert app.manual_color_max.label == "Color max (nT)"
    assert app.manual_color_min.value == -85.0
    assert app.manual_color_max.value == 85.0

    app.fill.value = "jr"
    app.lines.value = "Phi"

    assert app.manual_color_min.label == "Color min (µA/m²)"
    assert app.manual_color_max.label == "Color max (µA/m²)"
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

    # Confirmation must save the originally selected figure.
    app.figure = None

    app._confirm_overwrite()

    assert output_path.read_text(encoding="utf-8") == "new"
    assert writes[0][0] == output_path
    assert app.overwrite_modal.open is False
    assert app._pending_overwrite is None


def test_panel_can_cancel_movie_overwrite(tmp_path, monkeypatch):
    """Cancelling preserves the existing movie."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)
    output_path = tmp_path / "existing.gif"
    output_path.write_bytes(b"old")
    writes = []
    monkeypatch.setattr(app.pn.state, "execute", lambda callback: writes.append(callback))
    app.movie_filename.value = str(output_path)
    monkeypatch.setattr(app, "_require_loaded_directory", lambda: str(tmp_path))

    app._save_movie()
    assert app.overwrite_modal.open is True
    app._cancel_overwrite()

    assert output_path.read_bytes() == b"old"
    assert not writes
    assert app.overwrite_modal.open is False
    assert app._pending_overwrite is None


def test_panel_does_not_choose_an_arbitrary_workflow_child(tmp_path, monkeypatch):
    """A result tree is not an unambiguous simulation selection."""
    from pynamit.gui.panel_app import _default_simulation_directory

    simulation_directory = tmp_path / "results" / "N50_M50_Ncs50"
    simulation_directory.mkdir(parents=True)
    (simulation_directory / "settings.ncdf").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYNAMIT_SIMULATION_DIR", raising=False)

    assert _default_simulation_directory() == ""
    monkeypatch.chdir(simulation_directory)
    assert _default_simulation_directory() == str(simulation_directory)


def test_panel_default_simulation_directory_can_select_mage(tmp_path, monkeypatch):
    """The same explicit preference works for any case layout."""
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
    monkeypatch.setenv("PYNAMIT_SIMULATION_DIR", str(simulation_directory.relative_to(tmp_path)))

    assert _default_simulation_directory() == str(simulation_directory)


def test_panel_simulation_preserves_the_prepared_input_main_field(tmp_path, monkeypatch):
    """The Panel must preserve an input package's main field."""
    from pynamit.simulation.config import INTEGRATORS

    captured = {}

    async def fake_run(workflow, parameters):
        captured["workflow"] = workflow
        captured.update(parameters)
        return True

    monkeypatch.setattr(
        "pynamit.simulation.input_manifest.available_prepared_inputs",
        lambda directory: ("conductance", "boundary_Br"),
    )
    app = _panel_app_without_loading(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_run_workflow", fake_run)
    app.simulation_input_directory.value = str(tmp_path / "inputs")
    app.new_simulation_directory.value = str(tmp_path / "simulation")

    assert list(app.sim_integrator.options) == list(INTEGRATORS.values())

    app.sim_output_interval.value = 17
    asyncio.run(app._run_simulation())

    assert captured["workflow"] == "simulate"
    assert captured["input_directory"] == str(tmp_path / "inputs")
    assert captured["enabled_inputs"] == ("conductance", "boundary_Br")
    assert captured["output_interval"] == 17
    assert "main_field_kind" not in captured
    assert not app.sim_dt.disabled
    assert app.sim_rtol.disabled and app.sim_atol.disabled
    app.sim_run_dynamic.value = False
    assert all(
        widget.disabled for widget in (app.sim_integrator, app.sim_dt, app.sim_rtol, app.sim_atol)
    )
    app.sim_run_dynamic.value = True
    assert not app.sim_integrator.disabled and not app.sim_dt.disabled
    for method in ("exponential", "RK45"):
        app.sim_integrator.value = method
        assert app.sim_dt.disabled
        assert app.sim_rtol.disabled == (method == "exponential")
        assert app.sim_atol.disabled == (method == "exponential")
        asyncio.run(app._run_simulation())
        assert captured["dt"] is None
        assert captured["rtol"] == app.sim_rtol.value
        assert captured["atol"] == app.sim_atol.value == 1e-12


def test_gui_sessions_are_built_independently(tmp_path, monkeypatch):
    """The server must receive a factory, not shared widgets."""
    import panel as pn

    from pynamit.gui import cli, panel_app

    captured = {}
    builds = []
    monkeypatch.setattr(pn, "serve", lambda apps, **kwargs: captured.update(apps))

    def build(**kwargs):
        builds.append(kwargs)
        return pn.Column()

    monkeypatch.setattr(panel_app, "build_gui", build)
    cli.main([str(tmp_path), "--no-show"])

    factory = captured["/pynamit"]
    assert factory() is not factory()
    pn.panel(factory)
    assert builds == [{"simulation_directory": tmp_path}] * 3


def _plot_data(*, streams=("dynamic", "equilibrium"), n_time=4):
    """Only the loaded-data contract used by GUI callbacks."""
    return SimpleNamespace(
        results=SimpleNamespace(datasets=dict.fromkeys(streams)),
        has_model_output=bool(streams),
        available_inputs=("conductance",),
        n_time=n_time,
        timestamp_at_index=lambda index: (
            datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=index)
        ),
    )


def test_gui_load_edit_export_and_directory_switch(tmp_path, monkeypatch):
    """Draft edits cannot relabel displayed or exported results."""
    import matplotlib.pyplot as plt

    from pynamit.gui import panel_app

    data = _plot_data()
    monkeypatch.setattr(panel_app, "get_plot_data", lambda settings: data)
    monkeypatch.setattr(panel_app, "render_figure", lambda *args, **kwargs: plt.figure())
    app = panel_app.PynamitGUI(tmp_path)
    original_figure = app.figure
    assert original_figure is not None
    assert app.panel() is app.panel()
    app.geo_lat_min.value = 80
    app.geo_lat_max.value = 10  # A transient invalid draft is allowed.
    assert app.figure is original_figure
    assert json.load(app._download_settings())["geo_lat_min"] != 80
    assert app._rendered_settings.to_json(indent=4) in app._download_script().getvalue()

    app._redraw()
    assert "Error" in app.status.object
    assert app.figure is original_figure
    app.geo_lat_max.value = 90
    app._redraw()
    assert app.figure is not original_figure
    assert json.load(app._download_settings())["geo_lat_min"] == 80

    app.simulation_directory.value = str(tmp_path / "other")
    assert app.figure is app.plot_data is None
    assert app.script_download.disabled
    assert app.redraw_button.disabled
    with pytest.raises(ValueError, match="Render a figure"):
        app._download_script()


@pytest.mark.parametrize("streams", [("dynamic",), ("equilibrium",), ()])
def test_gui_uses_available_streams_and_reloads_smaller_data(tmp_path, monkeypatch, streams):
    """Plot choices follow data capabilities, not case names."""
    from pynamit.gui import panel_app
    from pynamit.plotting import FigureSettings

    FigureSettings(time_index=9, time_range=(5, 9)).save_defaults(tmp_path)
    data = _plot_data(streams=streams, n_time=10)
    monkeypatch.setattr(panel_app, "get_plot_data", lambda settings: data)
    monkeypatch.setattr(panel_app.PynamitGUI, "_redraw", lambda self: None)
    app = panel_app.PynamitGUI(tmp_path)
    assert app.time_index.value == 9
    assert app.show_dynamic.disabled == ("dynamic" not in streams)
    assert app.show_equilibrium.disabled == ("equilibrium" not in streams)
    assert (app.plot_type.value == "input_summary") == (not streams)
    if "dynamic" not in streams:
        assert "ground_curve_map" not in app.plot_type.options.values()

    data.n_time = 2
    app._load_simulation()
    assert app.time_index.end == app.time_index.value == 1
    assert app.time_range.value == (1, 1)

    def fail(settings):
        raise ValueError("Unreadable artifacts")

    monkeypatch.setattr(panel_app, "get_plot_data", fail)
    app._load_simulation()
    assert app.plot_data is app.figure is None
    assert "Unreadable artifacts" in app.status.object


def test_gui_starts_empty_and_handles_bad_defaults(tmp_path, monkeypatch):
    """Empty launches are normal; malformed defaults are visible."""
    from pynamit.gui.panel_app import PynamitGUI
    from pynamit.plotting.figure_settings import FIGURE_DEFAULTS_FILENAME

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYNAMIT_SIMULATION_DIR", raising=False)
    app = PynamitGUI()
    assert "Error" not in app.status.object
    assert app.simulation_directory.value == ""
    (tmp_path / FIGURE_DEFAULTS_FILENAME).write_text("{bad json")
    app.simulation_directory.value = str(tmp_path)
    app._load_simulation()
    assert "Could not parse plotting defaults" in app.status.object
    assert app.plot_data is None


def test_gui_input_availability_refreshes_without_fallback(tmp_path, monkeypatch):
    """Missing inputs are not silently enabled."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)
    app.simulation_input_directory.value = str(tmp_path / "missing")
    assert app.run_simulation_button.disabled
    assert all(widget.disabled and not widget.value for widget in app.simulation_inputs.values())
    monkeypatch.setattr(
        "pynamit.simulation.input_manifest.available_prepared_inputs",
        lambda directory: ("conductance", "u"),
    )
    app.simulation_input_directory.value = str(tmp_path / "valid")
    assert app._selected_simulation_inputs() == ("conductance", "u")
    app.simulation_inputs["u"].value = False
    app._sync_simulation_input_availability()
    assert app._selected_simulation_inputs() == ("conductance",)


def test_gui_preparation_protects_existing_inputs_and_passes_drivers(tmp_path, monkeypatch):
    """Prepare explicit physical inputs in a fresh directory."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)
    directory = tmp_path / "inputs"
    directory.mkdir()
    existing = directory / "important.txt"
    existing.write_text("keep")
    app.prepared_input_directory.value = str(directory)
    captured = []

    async def run(workflow, parameters):
        captured.append((workflow, parameters))
        return True

    monkeypatch.setattr(app, "_run_workflow", run)
    asyncio.run(app._prepare_example_inputs())
    assert not captured
    assert existing.read_text() == "keep"
    assert "new or empty" in app.status.object

    app.prepared_input_directory.value = str(tmp_path / "new")
    app.example_parameters["kp"].value = 3
    app.example_parameters["event_time"].value = datetime.datetime(2020, 1, 2)
    asyncio.run(app._prepare_example_inputs())
    workflow, parameters = captured[0]
    assert workflow == "prepare"
    assert parameters["kp"] == 3
    assert parameters["event_time"] == "2020-01-02T00:00:00"
    assert parameters["imf_Bz_nT"] == -4


def test_gui_saves_plot_defaults_with_confirmation(tmp_path, monkeypatch):
    """Persist draft settings without changing the rendered snapshot."""
    from pynamit.plotting import FigureSettings

    app = _panel_app_without_loading(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_require_loaded_directory", lambda: str(tmp_path))
    app.sim_time_offset.value = 12
    app._save_plot_defaults()
    assert FigureSettings.from_simulation_directory(tmp_path).simulation_time_offset_seconds == 12
    app.sim_time_offset.value = 24
    app._save_plot_defaults()
    assert app.overwrite_modal.open
    app.sim_time_offset.value = 36
    app._confirm_overwrite()
    assert FigureSettings.from_simulation_directory(tmp_path).simulation_time_offset_seconds == 24


def test_gui_workflow_subprocess_prepares_and_runs(tmp_path, monkeypatch):
    """Exercise the real JSON/process boundary and saved artifacts."""
    from tests.example_scenario import EMPIRICAL_INPUTS

    from pynamit.results import SimulationResults
    from pynamit.simulation.input_manifest import available_prepared_inputs

    app = _panel_app_without_loading(tmp_path, monkeypatch)
    input_directory = tmp_path / "inputs"
    output_directory = tmp_path / "simulation"
    parameters = dict(
        EMPIRICAL_INPUTS,
        input_directory=str(input_directory),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        artifact_storage="netcdf",
    )
    parameters["event_time"] = parameters["event_time"].isoformat()

    async def workflows():
        assert await app._run_workflow("prepare", parameters)
        assert set(available_prepared_inputs(input_directory)) == {"conductance", "boundary_jr"}
        assert await app._run_workflow(
            "simulate",
            dict(
                input_directory=str(input_directory),
                simulation_directory=str(output_directory),
                final_time=0.004,
                dt=0.001,
                output_interval=0.002,
                enable_pfac_coupling=False,
                artifact_storage="netcdf",
            ),
        )

    asyncio.run(asyncio.wait_for(workflows(), timeout=90))
    results = SimulationResults.from_directory(output_directory)
    assert set(results.inputs) == {"conductance", "boundary_jr"}
    assert "dynamic" in results.outputs
    assert results.times[-1] == pytest.approx(0.004)
    assert app._workflow_process is None
    assert app.cancel_workflow_button.disabled


@pytest.mark.parametrize("cancel", [False, True])
def test_gui_workflow_logs_failures_and_stops_processes(tmp_path, monkeypatch, cancel):
    """The UI receives process output and can stop a running job."""
    import sys

    app = _panel_app_without_loading(tmp_path, monkeypatch)
    create_process = asyncio.create_subprocess_exec

    async def start(*args, **kwargs):
        code = (
            "import threading; print('ready', flush=True); threading.Event().wait()"
            if cancel
            else "import sys; print('invalid inputs', flush=True); sys.exit(2)"
        )
        return await create_process(sys.executable, "-u", "-c", code, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    if cancel:
        app.workflow_log.param.watch(lambda event: app._cancel_workflow(), "object")

    async def workflow():
        if cancel:
            assert not await app._run_workflow("simulate", {})
        else:
            with pytest.raises(RuntimeError, match="code 2"):
                await app._run_workflow("simulate", {})

    asyncio.run(asyncio.wait_for(workflow(), timeout=20))
    assert ("ready" if cancel else "invalid inputs") in app.workflow_log.object
    assert app._workflow_process is None
    assert app.cancel_workflow_button.disabled


def test_gui_movie_uses_confirmed_settings_in_background(tmp_path, monkeypatch):
    """Movie confirmation captures controls before scheduling work."""
    app = _panel_app_without_loading(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_require_loaded_directory", lambda: str(tmp_path))
    output = tmp_path / "movie.gif"
    output.write_bytes(b"existing")
    app.movie_filename.value = str(output)
    app.movie_fps.value = 3
    scheduled = []
    calls = []
    monkeypatch.setattr(app.pn.state, "execute", scheduled.append)

    async def workflow(name, parameters):
        calls.append((name, parameters))
        return True

    monkeypatch.setattr(app, "_run_workflow", workflow)
    app._save_movie()
    assert not scheduled
    app.movie_fps.value = 7
    app._confirm_overwrite()
    asyncio.run(scheduled[0]())
    name, parameters = calls[0]
    assert name == "movie"
    assert parameters["settings"]["movie_fps"] == 3
    assert parameters["output_path"] == str(output)
    assert not app._busy


def test_gui_movie_worker_uses_the_regular_renderer(monkeypatch):
    """The worker forwards saved choices, not GUI-specific defaults."""
    from io import StringIO

    from kompe.math import get_backend

    from pynamit.gui import workflow_runner
    from pynamit.plotting import FigureSettings

    settings = FigureSettings(time_range=(2, 5), movie_fps=3, movie_dpi=40)
    parameters = {"settings": settings.to_dict(), "output_path": "movie.gif"}
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(parameters)))
    calls = []
    monkeypatch.setattr("pynamit.plotting.save_movie", lambda *args: calls.append(args))
    workflow_runner.main(["movie", "--backend", get_backend()])
    saved, output_path = calls[0]
    assert saved == settings
    assert output_path == "movie.gif"
