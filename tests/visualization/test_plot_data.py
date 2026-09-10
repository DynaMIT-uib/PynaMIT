"""Tests for plot data contracts."""

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from kompe import GlobalCSBasis, SHBasis, SphericalGrid
from kompe.coefficients import CoefficientSpace

import pynamit


@pytest.mark.parametrize("basis_kind", ["SH", "CS"])
def test_input_transform_grid_follows_coefficient_representation(basis_kind):
    """Sampling grids depend on field spaces, not input spellings."""
    from pynamit.plotting.plot_data import _build_input_transforms

    basis = SHBasis(2, 1, mean_free=True) if basis_kind == "SH" else GlobalCSBasis(4)
    scalar_space = CoefficientSpace(basis, representation="scalar")
    vector_space = CoefficientSpace(basis, representation="helmholtz")
    spaces = {"density": scalar_space, "velocity": vector_space, "flux": vector_space}
    schema = SimpleNamespace(input_field_spaces=spaces)
    scalar_grid = SphericalGrid([30.0, 60.0, 90.0], [0.0, 90.0, 180.0])
    vector_grid = SphericalGrid([45.0, 75.0], [45.0, 135.0])

    transforms = _build_input_transforms(schema, scalar_grid, vector_grid, keys=spaces)

    assert transforms["density"].grid is scalar_grid
    assert transforms["velocity"].grid is vector_grid
    assert transforms["flux"] is transforms["velocity"]


def test_plot_data_loads_projected_input_package_without_output(tmp_path):
    """Projection packages should be inspectable before a run exists."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(resistance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(resistance_shape),
        },
        time=0.0,
    )

    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(boundary_jr_shape), time=0.0)

    boundary_Br_shape = simulation.results.schema.input_field_spaces["boundary_Br"].shape
    simulation.inputs.set_coefficients("boundary_Br", np.zeros(boundary_Br_shape), time=0.0)

    view = plot_data.PlotData.from_directory(tmp_path)

    assert view.has_model_output is False
    assert not hasattr(view, "run_view")
    assert view.n_time == 1
    assert "dynamic" not in view.results.datasets
    assert {"boundary_Br", "boundary_jr", "conductance"}.issubset(view.results.datasets)
    assert "surface_to_poloidal_operator" not in view.results.geometry.__dict__
    assert view.output_evaluation is None
    assert (
        view.output_evaluation is None
        or "sheet_current_operators" not in view.output_evaluation.__dict__
    )
    assert view.input_transforms["boundary_jr"] is view.input_transforms["boundary_Br"]
    assert view.input_transforms["u"] is view.input_transforms["Q_eff"]
    assert view.input_transforms["u"] is view.input_transforms["E_neutral_wind"]


def test_plot_data_loads_without_boundary_br(tmp_path):
    """Ordinary simulations without RM/Br artifacts can be inspected."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(resistance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(resistance_shape),
        },
        time=0.0,
    )
    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(boundary_jr_shape), time=0.0)
    simulation.set_state(
        simulation.equilibrium_coefficients(time=0.0, interpolation=True)["induced_Br"], time=0.0
    )
    simulation.record_state(save=True)

    simulation.sample_equilibria([0.0], quiet=True)

    view = plot_data.PlotData.from_directory(tmp_path)
    assert "surface_to_poloidal_operator" not in view.results.geometry.__dict__
    assert view.output_evaluation is None
    assert (
        view.output_evaluation is None
        or "sheet_current_operators" not in view.output_evaluation.__dict__
    )
    with pytest.raises(ValueError, match="Unknown output fields"):
        view.output_plot_data(0, field_names={"not-a-field"})
    assert "surface_to_poloidal_operator" not in view.results.geometry.__dict__

    fields = view.output_plot_data(0, field_names={"Br"})
    input_fields = view.input_plot_data(0)
    last_fields = view.output_plot_data(-1, field_names={"Br"})
    for name in fields:
        np.testing.assert_array_equal(last_fields[name], fields[name])
    for index in (1, -2, 0.5):
        with pytest.raises(IndexError):
            view.output_plot_data(index, field_names={"Br"})

    assert view.has_model_output
    assert view.results._geometry is not None
    assert view.output_evaluation is not None
    assert (
        view.output_evaluation is None
        or "sheet_current_operators" not in view.output_evaluation.__dict__
    )
    assert "boundary_Br" not in view.results.datasets
    assert set(view.available_inputs) == {"boundary_jr", "conductance"}
    assert set(fields) == {"Br_dynamic", "Br_equilibrium"}
    assert fields["Br_dynamic"].shape == view.lat.shape
    assert np.all(np.isnan(input_fields["Br"]))


@pytest.mark.parametrize("last_time", [3 * 0.1, 1 / 3])
def test_plot_fields_use_exact_model_times_and_never_future_samples(last_time):
    """Display rounding cannot select the preceding coefficient row."""
    from pynamit.plotting.plot_data import PlotData
    from pynamit.results import evaluate_simulation_output

    simulation = pynamit.Simulation(Nmax=2, Mmax=1, Ncs=4, enable_pfac_coupling=False)
    results = simulation.results
    spaces = results.schema.output_field_spaces["dynamic"]
    initial = {name: np.zeros(space.shape) for name, space in spaces.items()}
    final = {**initial, "Phi": np.ones(spaces["Phi"].shape)}
    results.output_series.add_entry("dynamic", initial, 0.0)
    results.output_series.add_entry("dynamic", final, last_time)
    results.output_series.add_entry("equilibrium", {**final, "Phi": 2 * final["Phi"]}, last_time)
    current = np.ones(results.schema.input_field_spaces["boundary_jr"].shape)
    simulation.inputs.set_coefficients(
        "boundary_jr", np.stack([np.zeros_like(current), current]), time=[0.0, last_time]
    )
    view = PlotData.from_results(results, nlat=4, nlon=5)

    for key, fields in view.output_fields(-1, field_names={"Phi"}).items():
        expected = evaluate_simulation_output(
            results, last_time, key=key, transform=view.output_transform, field_names={"Phi"}
        )["Phi"]
        np.testing.assert_allclose(fields["Phi"], np.asarray(expected).ravel() * 1e-3)
        assert np.any(fields["Phi"] != 0)
    earlier = view.output_fields(0, field_names={"Phi"})
    assert np.all(np.isnan(earlier["equilibrium"]["Phi"]))
    np.testing.assert_array_equal(earlier["dynamic"]["Phi"], 0)
    with pytest.raises(ValueError, match="No 'equilibrium' output"):
        evaluate_simulation_output(results, 0.0, key="equilibrium", field_names={"Phi"})

    expected_jr = view.input_transforms["boundary_jr"].synthesize_scalar(current)
    np.testing.assert_allclose(view.input_plot_data(-1)["jr"], expected_jr.reshape(view.lat.shape))
    np.testing.assert_allclose(
        view.input_plot_data_at_time(last_time)["jr"], expected_jr.reshape(view.lat.shape)
    )


def test_plot_and_script_joule_evaluation_share_causal_inputs():
    """Late conductance and Br never leak into earlier output."""
    from pynamit.plotting.plot_data import PlotData
    from pynamit.results import evaluate_projected_input, evaluate_simulation_output

    simulation = pynamit.Simulation(
        Nmax=2, Mmax=1, Ncs=4, RM=4 * 6381e3, enable_pfac_coupling=False
    )
    results = simulation.results
    spaces = results.schema.output_field_spaces["dynamic"]
    coefficients = {name: np.full(space.shape, 1e-8) for name, space in spaces.items()}
    for key in ("dynamic", "equilibrium"):
        results.output_series.add_entries(
            key, {name: np.stack([values] * 3) for name, values in coefficients.items()}, [0, 1, 2]
        )
    space = results.schema.input_field_spaces["conductance"]
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(space.shape),
            "log_hall_to_pedersen_ratio": np.zeros(space.shape),
        },
        time=1.0,
    )
    space = results.schema.input_field_spaces["boundary_Br"]
    simulation.inputs.set_coefficients("boundary_Br", np.full(space.shape, 1e-7), time=2.0)
    view = PlotData.from_results(results, nlat=4, nlon=5)

    with pytest.raises(ValueError, match="No conductance"):
        evaluate_simulation_output(results, 0, field_names={"joule_heating"})
    assert "joule_heating" not in evaluate_simulation_output(results, 0)
    assert np.all(np.isnan(view.output_fields(0, field_names={"joule"})["dynamic"]["joule"]))

    for time in (1, 2):
        fields = view.output_fields(time, field_names={"joule"})
        evaluated = evaluate_simulation_output(results, time, transform=view.output_transform)
        for key in ("dynamic", "equilibrium"):
            np.testing.assert_allclose(
                fields[key]["joule"], np.asarray(evaluated["joule_heating"]).ravel()
            )
        # The dissipative part is etaP * J^T P J, never etaH * |J|^2.
        etaP = evaluate_projected_input(
            results, "conductance", time, transform=view.output_transform
        )["etaP"]
        current = np.stack([evaluated["JS_theta"], evaluated["JS_phi"]]).reshape(2, -1)
        expected = np.asarray(etaP).ravel() * np.einsum(
            "ig,ijg,jg->g", current, view.output_evaluation.pedersen_geometry, current
        )
        np.testing.assert_allclose(fields["dynamic"]["joule"], expected)
    output_maps, current_maps = (
        view.output_evaluation,
        view.output_evaluation.sheet_current_operators,
    )
    view.output_fields(-1, field_names={"joule"})
    assert view.output_evaluation is output_maps
    assert view.output_evaluation.sheet_current_operators is current_maps


def test_plot_data_supports_equilibrium_only_output(tmp_path):
    """An equilibrium-only run remains visualizable."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(resistance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(resistance_shape),
        },
        time=0.0,
    )
    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(boundary_jr_shape), time=0.0)
    simulation.sample_equilibria([0.0], quiet=True)

    view = plot_data.PlotData.from_directory(tmp_path)
    fields = view.output_plot_data(0)

    assert view.has_model_output
    assert "dynamic" not in view.results.datasets
    assert "equilibrium" in view.results.datasets
    assert view.results.geometry is not None
    assert "Br_equilibrium" in fields
    assert "Br_dynamic" not in fields


def test_plot_data_aligns_inputs_by_time_not_index(tmp_path):
    """Sparse outputs should use dense inputs at matching times."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    br_shape = simulation.results.schema.input_field_spaces["boundary_Br"].shape
    br_coefficients = np.zeros((3, *br_shape))
    br_coefficients[0] = 1.0
    br_coefficients[1] = 2.0
    br_coefficients[2] = 3.0
    simulation.inputs.set_coefficients(
        "boundary_Br", br_coefficients, time=np.array([0.0, 10.0, 20.0])
    )
    output_spaces = simulation.results.schema.output_field_spaces["dynamic"]
    empty_output = {
        variable: np.zeros(field_space.shape) for variable, field_space in output_spaces.items()
    }
    for time in (0.0, 20.0):
        simulation.results.output_series.add_entry("dynamic", empty_output, time)
    simulation.results.output_series.save("dynamic", simulation.results.artifact_store)

    view = plot_data.PlotData.from_directory(tmp_path)
    fields = view.input_plot_data(1)
    expected = (
        view.input_transforms["boundary_Br"]
        .scalar_synthesis_array.dot(br_coefficients[2])
        .reshape(view.lat.shape)
    )

    assert view.n_time == 2
    assert view.results.datasets["boundary_Br"].sizes["time"] == 3
    np.testing.assert_allclose(fields["Br"], expected)


def test_plot_data_inspects_neutral_wind_electric_field_input(tmp_path):
    """Projected neutral-wind E packages should be inspectable."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    coeff_length = simulation.results.schema.input_field_spaces["E_neutral_wind"].coefficient_count
    cf_coeffs = np.zeros(coeff_length)
    df_coeffs = np.zeros(coeff_length)
    cf_coeffs[0] = 1.0e-3
    simulation.inputs.set_coefficients(
        "E_neutral_wind", np.stack((cf_coeffs, df_coeffs)), time=0.0
    )

    view = plot_data.PlotData.from_directory(tmp_path)
    fields = view.input_plot_data(0)

    assert view.available_inputs == ("E_neutral_wind",)
    assert fields["E_neutral_wind_theta"].shape == view.wind_lat.shape
    assert fields["E_neutral_wind_phi"].shape == view.wind_lat.shape
    assert np.any(np.isfinite(fields["E_neutral_wind_theta"]))


def test_plot_data_keeps_model_and_geographic_evaluation_grids_separate(tmp_path):
    """Geographic maps must not replace the magnetic hemisphere grid."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(resistance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(resistance_shape),
        },
        time=0.0,
    )
    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(boundary_jr_shape), time=0.0)
    simulation.set_state(
        simulation.equilibrium_coefficients(time=0.0, interpolation=True)["induced_Br"], time=0.0
    )
    simulation.record_state(save=True)

    view = plot_data.PlotData.from_directory(tmp_path, nlat=6, nlon=8)
    geographic = view._get_geographic_evaluation()
    geographic_output_transform = view._geographic_output_transform(geographic)
    expected_lat, expected_lon = view.results.main_field.geo_to_model_coordinates(
        view.lat, view.lon
    )

    np.testing.assert_allclose(view.output_transform.grid.lat, view.lat.reshape(-1))
    np.testing.assert_allclose(view.output_transform.grid.lon, view.lon.reshape(-1))
    np.testing.assert_allclose(geographic_output_transform.grid.lat, expected_lat.reshape(-1))
    np.testing.assert_allclose(geographic_output_transform.grid.lon, expected_lon.reshape(-1))
    assert geographic_output_transform.grid != view.output_transform.grid
    assert view._get_geographic_evaluation() is geographic
    assert view.geographic_map_context() == plot_data.MapCoordinateContext.geographic(
        pd.Timestamp(view.results.config.t0).to_pydatetime()
    )


def test_plot_data_reuses_earth_fixed_geographic_mapping(tmp_path):
    """Kaiju model and display geometry stay fixed in GEO."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="kaiju_dipole",
        main_field_epoch=2011.8,
        t0="2011-10-24T18:00:10",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients(
        "boundary_jr", np.zeros((2, *boundary_jr_shape)), time=np.array([0.0, 3600.0])
    )

    view = plot_data.PlotData.from_directory(tmp_path, nlat=6, nlon=8)
    first_time = view.timestamp_at_index(0)
    last_time = view.timestamp_at_index(1)
    first = view._get_geographic_evaluation()
    last = view._get_geographic_evaluation()

    assert first is last
    np.testing.assert_allclose(first.scalar_grid.lat, last.scalar_grid.lat)
    np.testing.assert_allclose(first.scalar_grid.lon, last.scalar_grid.lon)
    assert view._get_geographic_evaluation() is last
    assert (
        view.model_map_context(first_time).noon_longitude
        != view.model_map_context(last_time).noon_longitude
    )


def test_kaiju_hemisphere_plot_coordinates_are_magnetic(tmp_path):
    """Kaiju polar plots rotate GEO samples into MAG and use MLT."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="kaiju_dipole",
        main_field_epoch=2011.8,
        t0="2011-10-24T18:00:10",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(boundary_jr_shape), time=0.0)

    view = plot_data.PlotData.from_directory(tmp_path, nlat=6, nlon=8)
    magnetic_latitude, magnetic_longitude = view.magnetic_plot_coordinates()
    expected = view.results.main_field.geographic_to_magnetic_coordinates(view.lat, view.lon)
    timestamp = view.timestamp_at_index(0)
    context = view.magnetic_map_context(timestamp)

    np.testing.assert_allclose(magnetic_latitude, expected[0])
    np.testing.assert_allclose(magnetic_longitude, expected[1])
    assert not np.allclose(magnetic_latitude, view.lat)
    assert context.longitude_kind == "magnetic"
    assert context.local_time_kind == "magnetic"
    assert context.noon_longitude == pytest.approx(
        view.results.main_field.magnetic_noon_longitude(pd.Timestamp(timestamp).to_pydatetime())
    )


def test_geographic_input_vectors_are_rotated_to_geographic_components(tmp_path):
    """Global quivers use geographic tangent-vector components."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    wind_shape = simulation.results.schema.input_field_spaces["u"].shape
    u_coefficients = np.stack(
        (np.linspace(0.0, 1.0, wind_shape[1]), np.linspace(1.0, 0.0, wind_shape[1]))
    )
    simulation.inputs.set_coefficients("u", u_coefficients, time=0.0)

    view = plot_data.PlotData.from_directory(tmp_path, wind_nlat=5, wind_nlon=7)
    fields = view.input_plot_data(0, coordinate_system="geographic")
    evaluation = view._get_geographic_evaluation()
    coefficients = view.dataset_values("u", "u")[0]
    model_theta, model_phi = evaluation.input_transforms["u"].synthesize_helmholtz(coefficients)
    _, _, expected_east, expected_north = view.results.main_field.model_to_geo_coordinates(
        evaluation.vector_grid.lat.reshape(view.wind_lat.shape),
        evaluation.vector_grid.lon.reshape(view.wind_lon.shape),
        model_phi.reshape(view.wind_lat.shape),
        -model_theta.reshape(view.wind_lat.shape),
    )

    np.testing.assert_allclose(fields["wind_phi"], expected_east.reshape(view.wind_lat.shape))
    np.testing.assert_allclose(fields["wind_theta"], -expected_north.reshape(view.wind_lat.shape))


def test_plot_data_rejects_unknown_display_coordinate_system(tmp_path):
    """Display-coordinate selection should fail explicitly on typos."""
    plot_data = importlib.import_module("pynamit.plotting.plot_data")

    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    boundary_jr_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(boundary_jr_shape), time=0.0)
    view = plot_data.PlotData.from_directory(tmp_path)

    with pytest.raises(ValueError, match="coordinate_system"):
        view.input_plot_data(0, coordinate_system="geomagnetic-ish")


def test_plot_data_cache_replaces_stale_simulation_version(tmp_path, monkeypatch):
    """Live updates replace rather than accumulate cached views."""
    plot_data_module = importlib.import_module("pynamit.plotting.plot_data")
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")
    fingerprint = [("dynamic", 1)]
    views = iter((object(), object()))
    monkeypatch.setattr(
        plot_data_module, "_artifact_fingerprint", lambda _directory: tuple(fingerprint)
    )
    monkeypatch.setattr(
        plot_data_module.PlotData, "from_directory", staticmethod(lambda _directory: next(views))
    )
    plot_data_module.clear_plot_data_cache()
    settings = figure_settings.FigureSettings(simulation_directory=str(tmp_path))

    first = plot_data_module.get_plot_data(settings)
    assert plot_data_module.get_plot_data(settings) is first
    fingerprint[0] = ("dynamic", 2)
    second = plot_data_module.get_plot_data(settings)

    assert second is not first
    assert len(plot_data_module._PLOT_DATA_CACHE) == 1


def test_plot_data_fingerprint_detects_nested_store_changes(tmp_path):
    """In-place Zarr chunk additions invalidate cached grid fields."""
    plot_data_module = importlib.import_module("pynamit.plotting.plot_data")
    chunk_directory = tmp_path / "dynamic.zarr" / "SH_induced_Br"
    chunk_directory.mkdir(parents=True)
    (chunk_directory / "0").write_bytes(b"first")

    before = plot_data_module._artifact_fingerprint(tmp_path)
    (chunk_directory / "1").write_bytes(b"second")
    after = plot_data_module._artifact_fingerprint(tmp_path)

    assert after != before
