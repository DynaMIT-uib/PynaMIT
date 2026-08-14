"""Tests for prepared input package helpers."""

import json
import shutil

import numpy as np
import pytest
from kompe.constants import EARTH_RADIUS_M

import pynamit
from pynamit.external_input_contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PYNAMIT_CENTERED_DIPOLE_110KM,
)
from pynamit.geomagnetism import MainField, decimal_year
from pynamit.simulation.api import Simulation
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.input_manifest import (
    INPUT_MANIFEST_FILENAME,
    available_prepared_inputs,
    clear_prepared_input_package,
    input_dataset_requirements,
    input_geometry_settings,
    input_projection_settings,
    prepared_input_contract,
    read_input_manifest,
    validate_input_manifest,
    validate_prepared_input_compatibility,
    write_input_manifest,
)
from pynamit.storage import ArtifactStore
from pynamit.workflows import example_inputs as example_inputs_module
from pynamit.workflows.example_inputs import prepare_example_inputs
from pynamit.workflows.prepared_inputs import (
    SIMULATION_MANIFEST_FILENAME,
    load_prepared_inputs_into_simulation,
    run_from_inputs,
)


def test_geographic_wind_is_rotated_into_model_coordinates():
    """Prepared dipole winds transform positions and components."""
    main_field = MainField(kind="dipole", epoch=2020)
    event_time = example_inputs_module._EXAMPLE_EVENT_TIME
    lat = np.array([20.0, 60.0])
    lon = np.array([-30.0, 80.0])
    u_theta = np.array([[10.0, 20.0], [30.0, 40.0]])
    u_phi = np.array([[5.0, 6.0], [7.0, 8.0]])

    theta_model, phi_model, model_lat, model_lon = (
        example_inputs_module._wind_to_model_coordinates(
            main_field, u_theta, u_phi, lat, lon, event_time=event_time
        )
    )
    expected_lat, expected_lon = main_field.geo_to_model_coordinates(
        lat, lon, event_time=event_time
    )
    vector_lat = np.broadcast_to(lat, u_theta.shape)
    vector_lon = np.broadcast_to(lon, u_theta.shape)
    _, _, expected_east, expected_north = main_field.geo_to_model_coordinates(
        vector_lat, vector_lon, east=u_phi, north=-u_theta, event_time=event_time
    )

    np.testing.assert_allclose(model_lat, expected_lat)
    np.testing.assert_allclose(model_lon, expected_lon)
    np.testing.assert_allclose(theta_model, -expected_north)
    np.testing.assert_allclose(phi_model, expected_east)


def test_default_inputs_share_one_provider_request_cache(tmp_path, monkeypatch):
    """Hardy, AMPS, and HWM receive one shared request object."""
    captured = {"requests": []}
    original_set_conductance = example_inputs_module.InputPreparation.set_conductance
    original_set_boundary_jr = example_inputs_module.InputPreparation.set_boundary_jr
    original_set_neutral_wind = example_inputs_module.InputPreparation.set_neutral_wind

    def fake_conductance(_date, lat=None, lon=None, time=None, *, request):
        assert lat is None and lon is None and time is None
        captured["requests"].append(request)
        source = request.source_grid
        values = np.ones(source.size)
        return values, values, source.lat, source.lon

    def fake_boundary_jr(_date, lat=None, lon=None, time=None, *, request):
        assert lat is None and lon is None and time is None
        captured["requests"].append(request)
        source = request.source_grid
        return np.zeros(source.size), source.lat, source.lon

    def fake_wind(_date, use_wind, time, lat=None, lon=None, *, request):
        assert use_wind
        assert time is None
        assert lat is None and lon is None
        captured["requests"].append(request)
        source = request.source_grid
        return (np.zeros(source.size), np.ones(source.size), source.lat, source.lon, None)

    def capture_set_conductance(self, *, lat, lon, **kwargs):
        captured["conductance_storage"] = (np.asarray(lat).copy(), np.asarray(lon).copy())
        return original_set_conductance(self, lat=lat, lon=lon, **kwargs)

    def capture_set_boundary_jr(self, *args, lat, lon, **kwargs):
        captured["boundary_jr_storage"] = (np.asarray(lat).copy(), np.asarray(lon).copy())
        return original_set_boundary_jr(self, *args, lat=lat, lon=lon, **kwargs)

    def capture_set_neutral_wind(self, *args, lat, lon, **kwargs):
        captured["wind_storage"] = (np.asarray(lat).copy(), np.asarray(lon).copy())
        return original_set_neutral_wind(self, *args, lat=lat, lon=lon, **kwargs)

    monkeypatch.setattr(example_inputs_module, "get_conductance_inputs", fake_conductance)
    monkeypatch.setattr(example_inputs_module, "get_jr_inputs", fake_boundary_jr)
    monkeypatch.setattr(example_inputs_module, "get_wind_inputs", fake_wind)
    monkeypatch.setattr(
        example_inputs_module.InputPreparation, "set_conductance", capture_set_conductance
    )
    monkeypatch.setattr(
        example_inputs_module.InputPreparation, "set_boundary_jr", capture_set_boundary_jr
    )
    monkeypatch.setattr(
        example_inputs_module.InputPreparation, "set_neutral_wind", capture_set_neutral_wind
    )

    prepared = prepare_example_inputs(
        tmp_path / "inputs",
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        use_boundary_jr=True,
        use_wind=True,
        artifact_storage="netcdf",
    )
    assert isinstance(prepared, pynamit.InputPreparation)
    assert not hasattr(prepared, "_runner")
    assert not hasattr(prepared, "outputs")
    assert not hasattr(prepared, "response")
    assert len(captured["requests"]) == 3
    assert captured["requests"][0] is captured["requests"][1] is captured["requests"][2]
    request = captured["requests"][0]
    assert (
        request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
        is request.grid_for(BOUNDARY_JR_PROVIDER_SPEC)
        is request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    )

    model_grid = prepared.geometry.model_grid
    expected_geo = prepared.geometry.main_field.model_to_geo_coordinates(
        model_grid.lat, model_grid.lon, event_time=example_inputs_module._EXAMPLE_EVENT_TIME
    )
    np.testing.assert_allclose(request.source_grid.lat, expected_geo[0])
    np.testing.assert_allclose(request.source_grid.lon, expected_geo[1])
    assert request.model_grid.coordinate_contract is PYNAMIT_CENTERED_DIPOLE_110KM
    assert request.model_epoch == pytest.approx(prepared.geometry.main_field.epoch)
    np.testing.assert_allclose(request.model_grid.lat, model_grid.lat)
    np.testing.assert_allclose(request.model_grid.lon, model_grid.lon)

    for name in ("conductance_storage", "boundary_jr_storage", "wind_storage"):
        np.testing.assert_allclose(captured[name][0], model_grid.lat)
        np.testing.assert_allclose(captured[name][1], model_grid.lon)


def test_adapter_cannot_return_another_source_grid(tmp_path, monkeypatch):
    """An adapter cannot silently remap the source grid."""

    def wrong_conductance(_date, lat=None, lon=None, time=None, *, request):
        assert lat is None and lon is None and time is None
        source = request.source_grid
        values = np.ones(source.size)
        return values, values, source.lat + 0.5, source.lon

    monkeypatch.setattr(example_inputs_module, "get_conductance_inputs", wrong_conductance)

    with pytest.raises(ValueError, match="shared geocentric_geographic source grid"):
        prepare_example_inputs(
            tmp_path / "inputs",
            final_time=0.0,
            Nmax=2,
            Mmax=1,
            Ncs=8,
            use_boundary_jr=False,
            artifact_storage="netcdf",
        )


def test_prepared_input_compatibility_ignores_simulation_only_settings():
    """Run-only settings can change without invalidating inputs."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    simulation_config = SimulationConfig(
        Nmax=4,
        Mmax=3,
        Ncs=8,
        RM=8.0e6,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=60,
        magnetic_boundary_shielding=True,
        integrator="exponential",
    )

    validate_prepared_input_compatibility(
        input_config.to_dataset(), simulation_config.to_dataset()
    )
    validate_prepared_input_compatibility(
        input_config.to_dataset(), simulation_config.to_dataset(), input_datasets=("u",)
    )


def test_prepared_input_compatibility_catches_projection_mismatch():
    """Projection settings remain part of the input package boundary."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    simulation_config = SimulationConfig(Nmax=5, Mmax=3, Ncs=8)

    with pytest.raises(ValueError, match="Nmax"):
        validate_prepared_input_compatibility(
            input_config.to_dataset(), simulation_config.to_dataset()
        )


def test_input_manifest_records_projection_settings(tmp_path):
    """The manifest records the projection-space input contract."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, horizontal_basis_kind="CS")

    manifest = write_input_manifest(
        tmp_path, config.to_dataset(), input_datasets=("conductance", "boundary_jr"), source="test"
    )

    loaded = read_input_manifest(tmp_path)
    assert loaded == manifest
    assert loaded["kind"] == "pynamit_prepared_inputs"
    assert loaded["version"] == 7
    assert "input_datasets" not in loaded
    assert "input_projection_settings" not in loaded
    assert "t0" not in loaded["input_contract"]["coefficient_space"]
    assert loaded["input_contract"]["coefficient_space"] == input_projection_settings(config)
    assert loaded["input_contract"]["geometry"] == input_geometry_settings(config)
    assert loaded["input_contract"]["geometry"]["input_time_origin"] == config.t0
    assert (
        loaded["input_contract"]["geometry"]["horizontal_coordinate_system"] == "centered_dipole"
    )
    assert loaded["input_contract"]["input_datasets"] == ["conductance", "boundary_jr"]
    assert loaded["input_contract"]["dataset_requirements"] == {}
    assert (
        validate_input_manifest(
            tmp_path, config.to_dataset(), available_inputs=("conductance", "boundary_jr")
        )
        == loaded
    )


def test_input_manifest_records_geometry_bound_dataset_requirements(tmp_path):
    """Boundary Br declares the extra run geometry it requires."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=7.0e6, main_field_kind="igrf")

    write_input_manifest(
        tmp_path,
        config.to_dataset(),
        input_datasets=("boundary_Br", "Q_eff", "E_neutral_wind"),
        source="test",
    )

    loaded = read_input_manifest(tmp_path)
    assert loaded["input_contract"] == prepared_input_contract(
        config, ["boundary_Br", "Q_eff", "E_neutral_wind"]
    )
    assert loaded["input_contract"]["geometry"]["RM"] == 7.0e6
    assert loaded["input_contract"]["geometry"]["main_field_kind"] == "igrf"
    assert loaded["input_contract"]["dataset_requirements"] == {"boundary_Br": ["RM"]}
    assert input_dataset_requirements(("conductance", "u")) == {}


def test_input_manifest_validation_catches_stale_dataset_lists(tmp_path):
    """Manifest datasets should match the stored artifacts."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    write_input_manifest(
        tmp_path, config.to_dataset(), input_datasets=("boundary_jr",), source="test"
    )

    with pytest.raises(ValueError, match="listed but missing"):
        validate_input_manifest(tmp_path, config.to_dataset(), available_inputs=())

    with pytest.raises(ValueError, match="stored but not listed"):
        validate_input_manifest(
            tmp_path, config.to_dataset(), available_inputs=("boundary_jr", "conductance")
        )
    validate_input_manifest(
        tmp_path,
        config.to_dataset(),
        available_inputs=("boundary_jr", "conductance"),
        allow_unlisted=True,
    )


def test_input_manifest_validation_can_require_manifest(tmp_path):
    """Run paths can reject input directories without a contract."""
    assert validate_input_manifest(tmp_path, available_inputs=()) is None

    with pytest.raises(ValueError, match=INPUT_MANIFEST_FILENAME):
        validate_input_manifest(tmp_path, available_inputs=(), require=True)


def test_clear_prepared_input_package_removes_only_pynamit_artifacts(tmp_path):
    """Repreparing inputs must not retain old PynaMIT artifacts."""
    (tmp_path / "boundary_jr.ncdf").write_text("stale", encoding="utf-8")
    (tmp_path / "dynamic.zarr").mkdir()
    (tmp_path / INPUT_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep", encoding="utf-8")

    removed = clear_prepared_input_package(tmp_path, artifact_storage="netcdf")

    assert removed == ("boundary_jr", "dynamic")
    assert not (tmp_path / "boundary_jr.ncdf").exists()
    assert not (tmp_path / "dynamic.zarr").exists()
    assert not (tmp_path / INPUT_MANIFEST_FILENAME).exists()
    assert notes.read_text(encoding="utf-8") == "keep"


def test_available_prepared_inputs_follows_schema_order(tmp_path):
    """Package inspection reports only stored input streams."""
    preparation = pynamit.InputPreparation(
        input_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    shape = preparation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    preparation.set_boundary_jr(boundary_jr_coefficients=np.zeros(shape), time=0.0)

    assert available_prepared_inputs(tmp_path, artifact_storage="netcdf") == ("boundary_jr",)


def test_input_manifest_validation_catches_contract_mismatch(tmp_path):
    """Manifest contracts should describe the settings artifact."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    changed_config = SimulationConfig(Nmax=5, Mmax=3, Ncs=8)
    write_input_manifest(
        tmp_path, config.to_dataset(), input_datasets=("boundary_jr",), source="test"
    )

    with pytest.raises(ValueError, match="does not match the settings artifact"):
        validate_input_manifest(
            tmp_path, changed_config.to_dataset(), available_inputs=("boundary_jr",)
        )


def test_prepared_inputs_require_matching_main_field():
    """Projected packages cannot be reused with another main field."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, main_field_kind="igrf")
    simulation_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, main_field_kind="dipole")

    with pytest.raises(ValueError, match="main_field_kind"):
        validate_prepared_input_compatibility(
            input_config.to_dataset(), simulation_config.to_dataset()
        )


def test_prepared_inputs_require_matching_input_time_origin():
    """Projected packages retain their physical input time origin."""
    input_config = SimulationConfig(
        Nmax=4, Mmax=3, Ncs=8, main_field_kind="kaiju_dipole", t0="2011-10-24 18:00:10"
    )
    simulation_config = SimulationConfig(
        Nmax=4, Mmax=3, Ncs=8, main_field_kind="kaiju_dipole", t0="2011-10-24 18:10:10"
    )

    with pytest.raises(ValueError, match="input_time_origin"):
        validate_prepared_input_compatibility(
            input_config.to_dataset(), simulation_config.to_dataset()
        )


def test_br_inputs_require_matching_magnetosphere_radius():
    """Boundary Br inputs are tied to their projection radius."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=7.0e6)
    simulation_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=8.0e6)

    validate_prepared_input_compatibility(
        input_config.to_dataset(), simulation_config.to_dataset()
    )

    with pytest.raises(ValueError, match="RM"):
        validate_prepared_input_compatibility(
            input_config.to_dataset(),
            simulation_config.to_dataset(),
            input_datasets=("boundary_Br",),
        )


def test_prepare_and_run_from_inputs_smoke(tmp_path):
    """A tiny prepared package can drive a self-contained run."""
    input_directory = tmp_path / "inputs"
    simulation_directory = tmp_path / "simulation"
    selected_simulation_directory = tmp_path / "selected_simulation"

    prepared = prepare_example_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    assert prepared.data.simulation_directory == str(input_directory.resolve())
    assert prepared.config.t0 == "2001-05-12 21:45:00"
    assert prepared.config.main_field_epoch == pytest.approx(
        decimal_year(example_inputs_module._EXAMPLE_EVENT_TIME)
    )
    assert (input_directory / INPUT_MANIFEST_FILENAME).exists()
    manifest = read_input_manifest(input_directory)
    assert manifest["input_contract"]["coefficient_space"]["Nmax"] == 2
    assert manifest["metadata"]["external_input_source"] in {"auto", "fallback", "native"}

    simulation = run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        final_time=0.0,
        dt=0.01,
        RM=2 * EARTH_RADIUS_M,
        sampling_step_interval=2,
        write_sample_interval=1,
        artifact_storage="netcdf",
    )

    assert simulation.data.simulation_directory == str(simulation_directory.resolve())
    assert simulation.config.fac_integration_radii[-1] == pytest.approx(simulation.config.RM)
    assert "dynamic" in simulation.data.output_series.datasets
    assert "conductance" in simulation.data.input_series.datasets
    assert (simulation_directory / "conductance.ncdf").exists()
    assert (simulation_directory / SIMULATION_MANIFEST_FILENAME).exists()
    simulation_manifest = json.loads(
        (simulation_directory / SIMULATION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert simulation_manifest["version"] == 5
    assert simulation_manifest["input_manifest"] == manifest
    assert simulation_manifest["time_evolution"]["sampling_step_interval"] == 2

    selected_simulation = run_from_inputs(
        input_directory,
        simulation_directory=selected_simulation_directory,
        enabled_inputs=("conductance",),
        final_time=0.0,
        dt=0.01,
        RM=2 * EARTH_RADIUS_M,
        write_sample_interval=1,
        artifact_storage="netcdf",
    )

    assert set(selected_simulation.data.input_series.datasets) == {"conductance"}
    assert not (selected_simulation_directory / "boundary_jr.ncdf").exists()


def test_manual_input_preparation_writes_a_reusable_package(tmp_path):
    """Interactive preparation stores coefficients without a runner."""
    input_directory = tmp_path / "inputs"
    preparation = pynamit.InputPreparation(
        input_directory=input_directory, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    assert not hasattr(preparation, "simulation_directory")
    shape = preparation.data.schema.input_field_spaces["conductance"].coefficient_shape
    preparation.set_conductance(
        log_magnitude_coefficients=np.zeros(shape),
        log_ratio_coefficients=np.zeros(shape),
        time=0.0,
    )

    manifest = preparation.write_manifest(source="test")
    reopened = pynamit.InputPreparation.from_directory(input_directory, artifact_storage="netcdf")

    assert manifest["source"] == "test"
    assert manifest["input_contract"]["input_datasets"] == ["conductance"]
    assert set(reopened.inputs) == {"conductance"}
    assert not hasattr(reopened, "_runner")


def test_run_from_inputs_rejects_changed_input_identity(tmp_path):
    """A trajectory cannot silently change its input selection."""
    input_directory = tmp_path / "inputs"
    simulation_directory = tmp_path / "simulation"
    prepare_example_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
    )

    with pytest.raises(ValueError, match="different trajectory identity"):
        run_from_inputs(
            input_directory,
            simulation_directory=simulation_directory,
            enabled_inputs=("conductance",),
            final_time=0.0,
            RM=2 * EARTH_RADIUS_M,
            artifact_storage="netcdf",
        )

    assert (simulation_directory / "boundary_jr.ncdf").exists()


def test_run_from_inputs_allows_prepared_package_relocation(tmp_path):
    """The input manifest identifies a relocated package."""
    input_directory = tmp_path / "inputs"
    relocated_directory = tmp_path / "relocated-inputs"
    simulation_directory = tmp_path / "simulation"
    prepare_example_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
    )
    shutil.copytree(input_directory, relocated_directory)

    result = run_from_inputs(
        relocated_directory,
        simulation_directory=simulation_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
        skip_completed=True,
    )

    assert result is None


def test_run_from_inputs_skips_completed_simulation_before_geometry(monkeypatch, tmp_path):
    """Batch sweeps do not rebuild completed simulation geometry."""
    input_directory = tmp_path / "inputs"
    simulation_directory = tmp_path / "simulation"
    prepare_example_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
    )

    monkeypatch.setattr(
        Simulation,
        "from_config",
        lambda *_args, **_kwargs: pytest.fail("completed simulation rebuilt geometry"),
    )
    result = run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
        skip_completed=True,
    )

    assert result is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"final_time": -1.0}, "finite, non-negative"),
        ({"skip_completed": "yes"}, "skip_completed"),
        ({"run_dynamic": False, "run_equilibrium": False}, "At least one"),
    ],
)
def test_run_from_inputs_validates_batch_options_before_skipping(tmp_path, kwargs, match):
    """Completion preflight cannot bypass evolution validation."""
    input_directory = tmp_path / "inputs"
    prepare_example_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )

    with pytest.raises(ValueError, match=match):
        run_from_inputs(
            input_directory,
            simulation_directory=tmp_path / "simulation",
            RM=2 * EARTH_RADIUS_M,
            artifact_storage="netcdf",
            **kwargs,
        )


def test_run_from_inputs_requires_a_separate_simulation_directory(tmp_path):
    """A simulation cannot overwrite its prepared-input package."""
    input_directory = tmp_path / "inputs"
    prepare_example_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )

    with pytest.raises(ValueError, match="must differ from input_directory"):
        run_from_inputs(
            input_directory,
            simulation_directory=input_directory,
            final_time=0.0,
            artifact_storage="netcdf",
        )


def test_loading_prepared_inputs_transfers_simulation_ownership(tmp_path):
    """The simulation owns copied inputs and removes stale data."""
    input_directory = tmp_path / "inputs"
    simulation_directory = tmp_path / "simulation"
    prepared = prepare_example_inputs(
        input_directory,
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        artifact_storage="netcdf",
        use_wind=False,
    )
    simulation = Simulation(
        simulation_directory=simulation_directory,
        artifact_storage="netcdf",
        **prepared.config.to_kwargs(),
    )
    wind_length = simulation.data.schema.input_field_spaces["u"].index_length
    simulation.set_neutral_wind(u_cf=np.zeros(wind_length), u_df=np.zeros(wind_length), time=0.0)

    loaded = load_prepared_inputs_into_simulation(
        simulation, input_directory, artifact_storage="netcdf", enabled_inputs=("conductance",)
    )

    assert loaded == ["conductance"]
    assert set(simulation.data.input_series.datasets) == {"conductance"}
    assert (simulation_directory / "conductance.ncdf").exists()
    assert not (simulation_directory / "u.ncdf").exists()
    assert (
        simulation.data.input_series.datasets["conductance"]
        is not prepared.data.input_series.datasets["conductance"]
    )

    ArtifactStore(input_directory).remove_artifact("conductance")
    assert simulation.data.input_series.get_entry("conductance", 0.0) is not None


def test_run_from_inputs_errors_on_requested_missing_dataset(tmp_path):
    """Explicitly enabled inputs must exist in the prepared package."""
    input_directory = tmp_path / "inputs"
    simulation_directory = tmp_path / "simulation"

    prepare_example_inputs(
        input_directory,
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        artifact_storage="netcdf",
        use_wind=False,
    )

    with pytest.raises(ValueError, match="Requested prepared input"):
        run_from_inputs(
            input_directory,
            simulation_directory=simulation_directory,
            enabled_inputs=("u",),
            final_time=0.0,
            dt=0.01,
            write_sample_interval=1,
            artifact_storage="netcdf",
        )

    assert not (simulation_directory / "settings.ncdf").exists()
