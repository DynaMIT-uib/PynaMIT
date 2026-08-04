"""Tests for prepared input package helpers."""

import json
import shutil

import numpy as np
import pytest
from kompe.constants import EARTH_RADIUS_M

from pynamit.geomagnetism import MainField
from pynamit.simulation.api import Simulation
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.workflows import prepared_inputs as prepared_inputs_module
from pynamit.simulation.workflows.prepared_inputs import (
    INPUT_MANIFEST_FILENAME,
    RUN_MANIFEST_FILENAME,
    clear_prepared_input_package,
    input_dataset_requirements,
    input_geometry_settings,
    input_projection_settings,
    load_prepared_inputs_into_simulation,
    prepare_pynamit_inputs,
    prepared_input_contract,
    read_input_manifest,
    run_pynamit_from_inputs,
    validate_input_manifest,
    validate_prepared_input_compatibility,
    write_input_manifest,
)
from pynamit.storage import ArtifactStore


def test_geographic_wind_is_rotated_into_model_coordinates():
    """Prepared dipole winds use magnetic positions and components."""
    main_field = MainField(kind="dipole", epoch=2020)
    event_time = prepared_inputs_module._DEFAULT_INPUT_TIME
    lat = np.array([20.0, 60.0])
    lon = np.array([-30.0, 80.0])
    u_theta = np.array([[10.0, 20.0], [30.0, 40.0]])
    u_phi = np.array([[5.0, 6.0], [7.0, 8.0]])

    theta_model, phi_model, model_lat, model_lon = (
        prepared_inputs_module._wind_to_model_coordinates(main_field, u_theta, u_phi, lat, lon)
    )
    expected_lat, expected_lon = main_field.geo_to_model_coordinates(
        lat, lon, event_time=event_time
    )

    assert theta_model.shape == u_theta.shape
    assert phi_model.shape == u_phi.shape
    np.testing.assert_allclose(model_lat, expected_lat)
    np.testing.assert_allclose(model_lon, expected_lon)


def test_default_scalar_inputs_use_geographic_query_positions(tmp_path, monkeypatch):
    """Providers receive GEO positions while PynaMIT stores model positions."""
    captured = {}
    original_set_conductance = prepared_inputs_module.Simulation.set_conductance
    original_set_boundary_jr = prepared_inputs_module.Simulation.set_boundary_jr

    def fake_conductance(_date, lat, lon, _time):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        captured["conductance_query"] = (lat.copy(), lon.copy())
        values = np.ones(lat.shape)
        return values, values, lat, lon

    def fake_boundary_jr(_date, lat, lon, _time):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        captured["boundary_jr_query"] = (lat.copy(), lon.copy())
        values = np.zeros(lat.shape)
        return values, lat, lon

    def capture_set_conductance(self, *args, lat, lon, **kwargs):
        captured["conductance_storage"] = (
            np.asarray(lat).copy(),
            np.asarray(lon).copy(),
        )
        return original_set_conductance(self, *args, lat=lat, lon=lon, **kwargs)

    def capture_set_boundary_jr(self, *args, lat, lon, **kwargs):
        captured["boundary_jr_storage"] = (
            np.asarray(lat).copy(),
            np.asarray(lon).copy(),
        )
        return original_set_boundary_jr(self, *args, lat=lat, lon=lon, **kwargs)

    monkeypatch.setattr(
        prepared_inputs_module,
        "get_conductance_inputs",
        fake_conductance,
    )
    monkeypatch.setattr(
        prepared_inputs_module,
        "get_jr_inputs",
        fake_boundary_jr,
    )
    monkeypatch.setattr(
        prepared_inputs_module.Simulation,
        "set_conductance",
        capture_set_conductance,
    )
    monkeypatch.setattr(
        prepared_inputs_module.Simulation,
        "set_boundary_jr",
        capture_set_boundary_jr,
    )

    prepared = prepare_pynamit_inputs(
        tmp_path / "inputs",
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        use_boundary_jr=True,
        artifact_storage="netcdf",
    )
    model_grid = prepared.geometry.model_grid
    expected_geo = prepared.geometry.main_field.model_to_geo_coordinates(
        model_grid.lat,
        model_grid.lon,
        event_time=prepared_inputs_module._DEFAULT_INPUT_TIME,
    )

    for name in ("conductance_query", "boundary_jr_query"):
        np.testing.assert_allclose(captured[name][0], expected_geo[0])
        np.testing.assert_allclose(captured[name][1], expected_geo[1])

    for name in ("conductance_storage", "boundary_jr_storage"):
        np.testing.assert_allclose(captured[name][0], model_grid.lat)
        np.testing.assert_allclose(captured[name][1], model_grid.lon)

    assert not np.allclose(captured["conductance_query"][0], model_grid.lat)


def test_prepared_input_compatibility_ignores_run_only_settings():
    """Run-only settings can change without invalidating inputs."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    run_config = SimulationConfig(
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

    validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())
    validate_prepared_input_compatibility(
        input_config.to_dataset(), run_config.to_dataset(), input_datasets=("u",)
    )


def test_prepared_input_compatibility_catches_projection_mismatch():
    """Projection settings remain part of the input package boundary."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    run_config = SimulationConfig(Nmax=5, Mmax=3, Ncs=8)

    with pytest.raises(ValueError, match="Nmax"):
        validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())


def test_input_manifest_records_projection_settings(tmp_path):
    """The manifest records the projection-space input contract."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, horizontal_basis_kind="CS")

    manifest = write_input_manifest(
        tmp_path, config.to_dataset(), input_datasets=("conductance", "boundary_jr"), source="test"
    )

    loaded = read_input_manifest(tmp_path)
    assert loaded == manifest
    assert loaded["kind"] == "pynamit_prepared_inputs"
    assert loaded["version"] == 5
    assert "input_datasets" not in loaded
    assert "input_projection_settings" not in loaded
    assert "t0" not in loaded["input_contract"]["coefficient_space"]
    assert loaded["input_contract"]["coefficient_space"] == input_projection_settings(config)
    assert loaded["input_contract"]["geometry"] == input_geometry_settings(config)
    assert loaded["input_contract"]["geometry"]["input_time_origin"] == config.t0
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
    run_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, main_field_kind="dipole")

    with pytest.raises(ValueError, match="main_field_kind"):
        validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())


def test_prepared_inputs_require_matching_input_time_origin():
    """Projected packages retain their physical input time origin."""
    input_config = SimulationConfig(
        Nmax=4, Mmax=3, Ncs=8, main_field_kind="kaiju_dipole", t0="2011-10-24 18:00:10"
    )
    run_config = SimulationConfig(
        Nmax=4, Mmax=3, Ncs=8, main_field_kind="kaiju_dipole", t0="2011-10-24 18:10:10"
    )

    with pytest.raises(ValueError, match="input_time_origin"):
        validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())


def test_br_inputs_require_matching_magnetosphere_radius():
    """Boundary Br inputs are tied to their projection radius."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=7.0e6)
    run_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=8.0e6)

    validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())

    with pytest.raises(ValueError, match="RM"):
        validate_prepared_input_compatibility(
            input_config.to_dataset(), run_config.to_dataset(), input_datasets=("boundary_Br",)
        )


def test_prepare_and_run_from_inputs_smoke(tmp_path):
    """A tiny prepared package can drive a self-contained run."""
    input_directory = tmp_path / "inputs"
    run_directory = tmp_path / "run"
    selected_run_directory = tmp_path / "selected_run"

    prepared = prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    assert prepared.run_data.run_directory == str(input_directory.resolve())
    assert prepared.config.t0 == "2001-05-12 21:45:00"
    assert prepared.config.main_field_epoch == pytest.approx(2020.0)
    assert (input_directory / INPUT_MANIFEST_FILENAME).exists()
    manifest = read_input_manifest(input_directory)
    assert manifest["input_contract"]["coefficient_space"]["Nmax"] == 2
    assert manifest["metadata"]["external_input_source"] in {"auto", "fallback", "native"}

    run = run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        final_time=0.0,
        dt=0.01,
        RM=2 * EARTH_RADIUS_M,
        sampling_step_interval=2,
        saving_sample_interval=1,
        artifact_storage="netcdf",
    )

    assert run.run_data.run_directory == str(run_directory.resolve())
    assert run.config.fac_integration_radii[-1] == pytest.approx(run.config.RM)
    assert "dynamic" in run.run_data.output_series.datasets
    assert "conductance" in run.run_data.input_series.datasets
    assert (run_directory / "conductance.ncdf").exists()
    assert (run_directory / RUN_MANIFEST_FILENAME).exists()
    run_manifest = json.loads((run_directory / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert run_manifest["version"] == 3
    assert run_manifest["input_manifest"] == manifest
    assert run_manifest["time_evolution"]["sampling_step_interval"] == 2

    selected_run = run_pynamit_from_inputs(
        input_directory,
        run_directory=selected_run_directory,
        enabled_inputs=("conductance",),
        final_time=0.0,
        dt=0.01,
        RM=2 * EARTH_RADIUS_M,
        saving_sample_interval=1,
        artifact_storage="netcdf",
    )

    assert set(selected_run.run_data.input_series.datasets) == {"conductance"}
    assert not (selected_run_directory / "boundary_jr.ncdf").exists()


def test_run_from_inputs_rejects_changed_input_identity(tmp_path):
    """A trajectory cannot silently change its input selection."""
    input_directory = tmp_path / "inputs"
    run_directory = tmp_path / "run"
    prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
    )

    with pytest.raises(ValueError, match="different trajectory identity"):
        run_pynamit_from_inputs(
            input_directory,
            run_directory=run_directory,
            enabled_inputs=("conductance",),
            final_time=0.0,
            RM=2 * EARTH_RADIUS_M,
            artifact_storage="netcdf",
        )

    assert (run_directory / "boundary_jr.ncdf").exists()


def test_run_from_inputs_allows_prepared_package_relocation(tmp_path):
    """The input manifest identifies a relocated package."""
    input_directory = tmp_path / "inputs"
    relocated_directory = tmp_path / "relocated-inputs"
    run_directory = tmp_path / "run"
    prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
    )
    shutil.copytree(input_directory, relocated_directory)

    result = run_pynamit_from_inputs(
        relocated_directory,
        run_directory=run_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
        skip_completed=True,
    )

    assert result is None


def test_run_from_inputs_skips_completed_run_before_geometry(monkeypatch, tmp_path):
    """Batch sweeps do not rebuild completed-run geometry."""
    input_directory = tmp_path / "inputs"
    run_directory = tmp_path / "run"
    prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        final_time=0.0,
        RM=2 * EARTH_RADIUS_M,
        artifact_storage="netcdf",
    )

    monkeypatch.setattr(
        prepared_inputs_module.Simulation,
        "from_config",
        lambda *_args, **_kwargs: pytest.fail("completed run rebuilt geometry"),
    )
    result = run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
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
    prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )

    with pytest.raises(ValueError, match=match):
        run_pynamit_from_inputs(
            input_directory,
            run_directory=tmp_path / "run",
            RM=2 * EARTH_RADIUS_M,
            artifact_storage="netcdf",
            **kwargs,
        )


def test_run_from_inputs_requires_a_separate_run_directory(tmp_path):
    """A run must not overwrite its reusable prepared-input package."""
    input_directory = tmp_path / "inputs"
    prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )

    with pytest.raises(ValueError, match="must differ from input_directory"):
        run_pynamit_from_inputs(
            input_directory,
            run_directory=input_directory,
            final_time=0.0,
            artifact_storage="netcdf",
        )


def test_loading_prepared_inputs_transfers_run_ownership(tmp_path):
    """The consuming run owns copied inputs and removes stale data."""
    input_directory = tmp_path / "inputs"
    run_directory = tmp_path / "run"
    prepared = prepare_pynamit_inputs(
        input_directory,
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        artifact_storage="netcdf",
        use_wind=False,
    )
    simulation = Simulation(
        run_directory=run_directory, artifact_storage="netcdf", **prepared.config.to_kwargs()
    )
    wind_length = simulation.run_data.schema.input_field_spaces["u"].index_length
    simulation.set_u(u_cf=np.zeros(wind_length), u_df=np.zeros(wind_length), time=0.0)

    loaded = load_prepared_inputs_into_simulation(
        simulation, input_directory, artifact_storage="netcdf", enabled_inputs=("conductance",)
    )

    assert loaded == ["conductance"]
    assert set(simulation.run_data.input_series.datasets) == {"conductance"}
    assert (run_directory / "conductance.ncdf").exists()
    assert not (run_directory / "u.ncdf").exists()
    assert (
        simulation.run_data.input_series.datasets["conductance"]
        is not prepared.run_data.input_series.datasets["conductance"]
    )

    ArtifactStore(input_directory).remove_artifact("conductance")
    assert simulation.run_data.input_series.get_entry("conductance", 0.0) is not None


def test_run_from_inputs_errors_on_requested_missing_dataset(tmp_path):
    """Explicitly enabled inputs must exist in the prepared package."""
    input_directory = tmp_path / "inputs"
    run_directory = tmp_path / "run"

    prepare_pynamit_inputs(
        input_directory,
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        artifact_storage="netcdf",
        use_wind=False,
    )

    with pytest.raises(ValueError, match="Requested prepared input"):
        run_pynamit_from_inputs(
            input_directory,
            run_directory=run_directory,
            enabled_inputs=("u",),
            final_time=0.0,
            dt=0.01,
            saving_sample_interval=1,
            artifact_storage="netcdf",
        )

    assert not (run_directory / "settings.ncdf").exists()
