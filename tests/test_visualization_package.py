"""Tests for visualization package exports."""

import importlib

import numpy as np
import pynamit
import pytest
import xarray as xr


def test_magnetic_boundary_operators_are_available():
    """JS operators are available from their owning modules."""
    grid_evaluation = importlib.import_module("pynamit.visualization.grid_evaluation")
    magnetic_boundary = importlib.import_module(
        "pynamit.simulation.electrodynamics.magnetic_boundary"
    )

    assert callable(magnetic_boundary.m_ind_to_gridded_JS)
    assert callable(magnetic_boundary.m_imp_to_gridded_JS)
    assert callable(grid_evaluation.build_JS_operators)


def test_pynameye_is_available_from_visualization():
    """PynamEye is available from the canonical visualization path."""
    visualization = importlib.import_module("pynamit.visualization")
    visualization_pynameye = importlib.import_module("pynamit.visualization.pynameye")

    assert visualization.PynamEye is visualization_pynameye.PynamEye
    assert hasattr(visualization.PynamEye, "style_global_axis")


def test_input_projection_comparison_recipe_is_importable():
    """Input diagnostic recipes are available from their module."""
    diagnostics = importlib.import_module("pynamit.visualization.input_projection_comparison")

    assert callable(diagnostics.plot_input_projection_comparison)


def test_projected_input_inspector_is_visualization_api():
    """Projected-input inspection is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    input_projection = importlib.import_module("pynamit.visualization.input_projection")

    assert visualization.evaluate_projected_input is input_projection.evaluate_projected_input


def test_saved_run_view_is_visualization_api():
    """Saved-run view is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    saved_run = importlib.import_module("pynamit.visualization.saved_run")

    assert visualization.SavedRunView is saved_run.SavedRunView


def test_panel_figure_specs_are_visualization_api():
    """Panel and publication-script primitives are exported lazily."""
    visualization = importlib.import_module("pynamit.visualization")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    figure_builder = importlib.import_module("pynamit.visualization.figure_builder")
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    assert visualization.PynamitFigureSpec is figure_specs.PynamitFigureSpec
    assert visualization.render_pynamit_figure is figure_builder.render_pynamit_figure
    assert visualization.save_pynamit_movie is figure_builder.save_pynamit_movie
    assert visualization.SavedCoefficientFieldView is run_fields.SavedCoefficientFieldView


def test_saved_field_view_loads_projected_input_package_without_state(tmp_path):
    """Projection packages should be inspectable before a run exists."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "resistance"
    ].coefficient_shape
    eta_p = np.ones(resistance_shape)
    eta_h = np.zeros(resistance_shape)
    simulation.set_resistance(etaP_coefficients=eta_p, etaH_coefficients=eta_h, time=0.0)

    jr_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)

    br_shape = simulation.run_data.schema.input_field_spaces["Br"].coefficient_shape
    simulation.set_Br(Br_coefficients=np.zeros(br_shape), time=0.0)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)

    assert view.has_output_state is False
    assert view.n_time == 1
    assert "state" not in view.run_view.datasets
    assert {"Br", "jr", "resistance"}.issubset(view.run_view.datasets)
    assert view.state_evaluator is None
    assert view.run_view.geometry is None
    assert view.state_evaluation_context is None
    assert view.sheet_current_maps is None
    assert view.input_evaluators["jr"] is view.input_evaluators["Br"]
    assert view.input_evaluators["u"] is None
    assert view.input_evaluators["Q_eff"] is None
    assert view.input_evaluators["E_source"] is None


def test_saved_field_view_loads_without_boundary_br(tmp_path):
    """Ordinary runs without RM/Br artifacts can be inspected."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "resistance"
    ].coefficient_shape
    simulation.set_resistance(
        etaP_coefficients=np.ones(resistance_shape),
        etaH_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    jr_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)
    simulation.impose_steady_state(time=0.0, save=True, quiet=True)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    assert view.run_view.geometry is None
    assert view.state_evaluation_context is None
    assert view.sheet_current_maps is None
    with pytest.raises(ValueError, match="Unknown state fields"):
        view.state_comparison_grid_fields(0, field_names={"not-a-field"})
    assert view.run_view.geometry is None

    fields = view.state_comparison_grid_fields(0, field_names={"Br"})
    input_fields = view.input_grid_fields(0)

    assert view.has_output_state
    assert view.run_view.geometry is not None
    assert view.state_evaluation_context is not None
    assert view.sheet_current_maps is None
    assert "Br" not in view.run_view.datasets
    assert set(view.available_inputs) == {"jr", "resistance"}
    assert set(fields) == {"Br_state", "Br_steady"}
    assert fields["Br_state"].shape == view.lat.shape
    assert np.all(np.isnan(input_fields["Br"]))


def test_saved_state_joule_uses_pedersen_dissipation():
    """Saved-state Joule heating follows the Pedersen closure."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    class IdentityEvaluator:
        G = np.eye(2)
        scalar_coeffs_to_grid_operator = np.eye(2)

        @staticmethod
        def synthesize_helmholtz(coeffs):
            return np.asarray(coeffs)[0], np.asarray(coeffs)[1]

    state = xr.Dataset(
        {
            "SH_m_ind": (("time", "coefficient"), [[1.0, 2.0]]),
            "SH_m_imp": (("time", "coefficient"), [[3.0, 4.0]]),
            "SH_Phi": (("time", "coefficient"), [[5.0, 6.0]]),
            "SH_W": (("time", "coefficient"), [[7.0, 8.0]]),
        },
        coords={"time": [0.0]},
    )
    conductance = xr.Dataset(
        {"SH_etaP": (("time", "coefficient"), [[2.0, 3.0]])}, coords={"time": [0.0]}
    )
    zero_map = np.zeros((2, 2))
    state_evaluation_context = {
        "RI": 2.0,
        "m_ind_to_Br": zero_map,
        "m_imp_to_jr": zero_map,
        "m_ind_to_Jeq": zero_map,
        "pedersen_geometry": np.moveaxis(
            np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 0.0], [0.0, 5.0]]]), 0, -1
        ),
    }
    sheet_current_maps = {
        "m_ind_to_JS": np.array([np.eye(2), np.zeros((2, 2))]),
        "m_imp_to_JS": np.array([np.zeros((2, 2)), np.eye(2)]),
        "Br_to_JS": None,
    }

    fields = run_fields.compute_state_comparison_fields_at_index(
        0,
        {"state": state, "resistance": conductance},
        IdentityEvaluator(),
        IdentityEvaluator(),
        state_evaluation_context,
        sheet_current_maps,
    )["state"]

    np.testing.assert_allclose(fields["joule"], [70.0, 288.0])

    steady_fields = run_fields.compute_state_comparison_fields_at_index(
        0,
        {"steady_state": state, "resistance": conductance},
        IdentityEvaluator(),
        IdentityEvaluator(),
        state_evaluation_context,
        sheet_current_maps,
    )["steady"]

    np.testing.assert_allclose(steady_fields["joule"], fields["joule"])


def test_saved_field_view_supports_steady_state_only_output(tmp_path):
    """A valid steady-only run remains visualizable as model output."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "resistance"
    ].coefficient_shape
    simulation.set_resistance(
        etaP_coefficients=np.ones(resistance_shape),
        etaH_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    jr_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)
    simulation.impose_steady_state(time=0.0, save=True, quiet=True)
    simulation.run_data.artifact_store.remove_artifact("state")

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    fields = view.state_comparison_grid_fields(0)

    assert view.has_output_state
    assert "state" not in view.run_view.datasets
    assert "steady_state" in view.run_view.datasets
    assert view.run_view.geometry is not None
    assert "Br_steady" in fields
    assert "Br_state" not in fields


def test_saved_field_view_aligns_inputs_by_time_not_index(tmp_path):
    """Sparse outputs should use dense inputs at matching times."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    br_shape = simulation.run_data.schema.input_field_spaces["Br"].coefficient_shape
    br_coefficients = np.zeros((3, *br_shape))
    br_coefficients[0] = 1.0
    br_coefficients[1] = 2.0
    br_coefficients[2] = 3.0
    simulation.set_Br(Br_coefficients=br_coefficients, time=np.array([0.0, 10.0, 20.0]))
    xr.Dataset(coords={"time": np.array([0.0, 20.0])}).to_netcdf(tmp_path / "state.ncdf")

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    fields = view.input_grid_fields(1)
    expected = view.input_evaluators["Br"].G.dot(br_coefficients[2]).reshape(view.lat.shape)

    assert view.n_time == 2
    assert view.run_view.datasets["Br"].sizes["time"] == 3
    np.testing.assert_allclose(fields["Br"], expected)


def test_saved_field_view_inspects_direct_e_source_input(tmp_path):
    """Projected direct E_source packages should be inspectable."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    coeff_length = simulation.run_data.schema.input_field_spaces["E_source"].index_length
    cf_coeffs = np.zeros(coeff_length)
    df_coeffs = np.zeros(coeff_length)
    cf_coeffs[0] = 1.0e-3
    simulation.set_E_source(E_source_cf=cf_coeffs, E_source_df=df_coeffs, time=0.0)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    fields = view.input_grid_fields(0)

    assert view.available_inputs == ("E_source",)
    assert fields["E_source_theta"].shape == view.wind_lat.shape
    assert fields["E_source_phi"].shape == view.wind_lat.shape
    assert np.any(np.isfinite(fields["E_source_theta"]))


def test_publication_script_export_is_jupyter_friendly():
    """Figure specs can produce editable publication scripts."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    spec = figure_specs.PynamitFigureSpec(run_directory="run", plot_type="global")
    script = figure_specs.publication_script_for_spec(spec, output_path="figures/test.png")

    assert script.startswith("# %%")
    assert "render_pynamit_figure" in script
    assert '"run_directory": "run"' in script
    assert "fig.savefig" in script


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_index": -1},
        {"time_range": (0.5, 2)},
        {"dbdt_window_points": 1.5},
        {"color_scale_percentile": np.nan},
        {"sim_time_offset_seconds": np.inf},
    ],
)
def test_figure_spec_rejects_invalid_renderer_values(kwargs):
    """Figure specs reject values renderers cannot interpret."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    with pytest.raises(ValueError):
        figure_specs.PynamitFigureSpec(**kwargs)


def test_run_plot_defaults_are_applied(tmp_path):
    """Run directories can carry plotting defaults."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    (tmp_path / "pynamit_plot_defaults.json").write_text(
        """
        {
          "data_directory": "/tmp/mag_data",
          "plot_defaults": {
            "plot_type": "ground_timeseries",
            "ground_station": "OTT",
            "time_range": [4, 12],
            "extra_frontend_note": "kept"
          }
        }
        """,
        encoding="utf-8",
    )

    spec = figure_specs.figure_spec_from_run_defaults(tmp_path)

    assert spec.run_directory == str(tmp_path)
    assert spec.data_directory == "/tmp/mag_data"
    assert spec.plot_type == "ground_timeseries"
    assert spec.ground_station == "OTT"
    assert spec.time_range == (4, 12)
    assert spec.extra["extra_frontend_note"] == "kept"


def test_saved_field_view_cache_replaces_stale_run_version(tmp_path, monkeypatch):
    """Live-run updates replace rather than accumulate cached views."""
    figure_context = importlib.import_module("pynamit.visualization.figure_context")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    fingerprint = [("state", 1)]
    views = iter((object(), object()))
    monkeypatch.setattr(
        figure_context, "_artifact_fingerprint", lambda _directory: tuple(fingerprint)
    )
    monkeypatch.setattr(
        figure_context.SavedCoefficientFieldView,
        "from_directory",
        staticmethod(lambda _directory: next(views)),
    )
    figure_context.clear_saved_field_view_cache()
    spec = figure_specs.PynamitFigureSpec(run_directory=str(tmp_path))

    first = figure_context.get_saved_field_view(spec)
    assert figure_context.get_saved_field_view(spec) is first
    fingerprint[0] = ("state", 2)
    second = figure_context.get_saved_field_view(spec)

    assert second is not first
    assert len(figure_context._VIEW_CACHE) == 1


def test_saved_field_view_fingerprint_detects_nested_store_changes(tmp_path):
    """In-place Zarr chunk additions invalidate a saved field view."""
    figure_context = importlib.import_module("pynamit.visualization.figure_context")
    chunk_directory = tmp_path / "state.zarr" / "SH_m_ind"
    chunk_directory.mkdir(parents=True)
    (chunk_directory / "0").write_bytes(b"first")

    before = figure_context._artifact_fingerprint(tmp_path)
    (chunk_directory / "1").write_bytes(b"second")
    after = figure_context._artifact_fingerprint(tmp_path)

    assert after != before


def test_panel_app_module_imports_when_panel_is_installed():
    """Panel app construction API is importable when Panel exists."""
    pytest.importorskip("panel")
    visualization = importlib.import_module("pynamit.visualization")
    panel_app = importlib.import_module("pynamit.visualization.panel_app")

    assert visualization.PynamitPanelApp is panel_app.PynamitPanelApp
    assert callable(panel_app.build_pynamit_panel_app)
