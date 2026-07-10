"""Tests for prepared input package helpers."""

import importlib

import pytest

import pynamit
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.prepared_inputs import (
    INPUT_MANIFEST_FILENAME,
    RUN_MANIFEST_FILENAME,
    clear_prepared_input_package,
    input_dataset_requirements,
    input_geometry_settings,
    input_projection_settings,
    prepare_pynamit_inputs,
    prepared_input_contract,
    read_input_manifest,
    run_pynamit_from_inputs,
    validate_input_manifest,
    validate_prepared_input_compatibility,
    write_input_manifest,
)


def test_prepared_input_workflows_are_public():
    """High-level prepared-input workflows are package exports."""
    prepared_inputs = importlib.import_module("pynamit.simulation.prepared_inputs")

    assert pynamit.prepare_pynamit_inputs is prepared_inputs.prepare_pynamit_inputs
    assert pynamit.run_pynamit_from_inputs is prepared_inputs.run_pynamit_from_inputs


def test_prepared_input_compatibility_ignores_run_only_settings():
    """Run-only settings can change without invalidating inputs."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    run_config = SimulationConfig(
        Nmax=4,
        Mmax=3,
        Ncs=8,
        ignore_PFAC=True,
        connect_hemispheres=True,
        latitude_boundary=60,
        RM_shielding=True,
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
        tmp_path, config.to_dataset(), input_datasets=("conductance", "jr"), source="test"
    )

    loaded = read_input_manifest(tmp_path)
    assert loaded == manifest
    assert loaded["kind"] == "pynamit_prepared_inputs"
    assert loaded["input_datasets"] == ["conductance", "jr"]
    assert loaded["input_projection_settings"]["Nmax"] == 4
    assert loaded["input_projection_settings"]["horizontal_basis_kind"] == "CS"
    assert "mainfield_kind" not in loaded["input_projection_settings"]
    assert "t0" not in loaded["input_contract"]["coefficient_space"]
    assert loaded["input_contract"]["coefficient_space"] == input_projection_settings(config)
    assert loaded["input_contract"]["geometry"] == input_geometry_settings(config)
    assert loaded["input_contract"]["geometry"]["mainfield_coordinate_time"] == config.t0
    assert loaded["input_contract"]["input_datasets"] == ["conductance", "jr"]
    assert loaded["input_contract"]["dataset_requirements"] == {}
    assert (
        validate_input_manifest(
            tmp_path, config.to_dataset(), available_inputs=("conductance", "jr")
        )
        == loaded
    )


def test_input_manifest_records_geometry_bound_dataset_requirements(tmp_path):
    """Boundary Br declares the extra run geometry it requires."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=7.0e6, mainfield_kind="igrf")

    write_input_manifest(
        tmp_path, config.to_dataset(), input_datasets=("Br", "Q_eff", "E_source"), source="test"
    )

    loaded = read_input_manifest(tmp_path)
    assert loaded["input_contract"] == prepared_input_contract(config, ["Br", "Q_eff", "E_source"])
    assert loaded["input_contract"]["geometry"]["RM"] == 7.0e6
    assert loaded["input_contract"]["geometry"]["mainfield_kind"] == "igrf"
    assert loaded["input_contract"]["dataset_requirements"] == {"Br": ["RM"]}
    assert input_dataset_requirements(("conductance", "u")) == {}


def test_input_manifest_validation_catches_stale_dataset_lists(tmp_path):
    """Manifest datasets should match the stored artifacts."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    write_input_manifest(tmp_path, config.to_dataset(), input_datasets=("jr",), source="test")

    with pytest.raises(ValueError, match="listed but missing"):
        validate_input_manifest(tmp_path, config.to_dataset(), available_inputs=())

    with pytest.raises(ValueError, match="stored but not listed"):
        validate_input_manifest(
            tmp_path, config.to_dataset(), available_inputs=("jr", "conductance")
        )
    validate_input_manifest(
        tmp_path, config.to_dataset(), available_inputs=("jr", "conductance"), allow_unlisted=True
    )


def test_input_manifest_validation_can_require_manifest(tmp_path):
    """Run paths can reject input directories without a contract."""
    assert validate_input_manifest(tmp_path, available_inputs=()) is None

    with pytest.raises(ValueError, match=INPUT_MANIFEST_FILENAME):
        validate_input_manifest(tmp_path, available_inputs=(), require=True)


def test_clear_prepared_input_package_removes_only_pynamit_artifacts(tmp_path):
    """Repreparing inputs must not retain old PynaMIT artifacts."""
    (tmp_path / "jr.ncdf").write_text("stale", encoding="utf-8")
    (tmp_path / "state.zarr").mkdir()
    (tmp_path / INPUT_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep", encoding="utf-8")

    removed = clear_prepared_input_package(tmp_path, artifact_storage="netcdf")

    assert removed == ("jr", "state")
    assert not (tmp_path / "jr.ncdf").exists()
    assert not (tmp_path / "state.zarr").exists()
    assert not (tmp_path / INPUT_MANIFEST_FILENAME).exists()
    assert notes.read_text(encoding="utf-8") == "keep"


def test_input_manifest_validation_catches_contract_mismatch(tmp_path):
    """Manifest contracts should describe the settings artifact."""
    config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8)
    changed_config = SimulationConfig(Nmax=5, Mmax=3, Ncs=8)
    write_input_manifest(tmp_path, config.to_dataset(), input_datasets=("jr",), source="test")

    with pytest.raises(ValueError, match="does not match the settings artifact"):
        validate_input_manifest(tmp_path, changed_config.to_dataset(), available_inputs=("jr",))


def test_prepared_inputs_require_matching_mainfield():
    """Projected packages cannot be reused with another main field."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, mainfield_kind="igrf")
    run_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, mainfield_kind="dipole")

    with pytest.raises(ValueError, match="mainfield_kind"):
        validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())


def test_prepared_inputs_require_matching_mainfield_coordinate_time():
    """Kaiju/SM projected packages are tied to coordinate time."""
    input_config = SimulationConfig(
        Nmax=4, Mmax=3, Ncs=8, mainfield_kind="kaiju_dipole", t0="2011-10-24 18:00:10"
    )
    run_config = SimulationConfig(
        Nmax=4, Mmax=3, Ncs=8, mainfield_kind="kaiju_dipole", t0="2011-10-24 18:10:10"
    )

    with pytest.raises(ValueError, match="mainfield_coordinate_time"):
        validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())


def test_br_inputs_require_matching_magnetosphere_radius():
    """Boundary Br inputs are tied to their projection radius."""
    input_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=7.0e6)
    run_config = SimulationConfig(Nmax=4, Mmax=3, Ncs=8, RM=8.0e6)

    validate_prepared_input_compatibility(input_config.to_dataset(), run_config.to_dataset())

    with pytest.raises(ValueError, match="RM"):
        validate_prepared_input_compatibility(
            input_config.to_dataset(), run_config.to_dataset(), input_datasets=("Br",)
        )


def test_prepare_and_run_from_inputs_smoke(tmp_path):
    """A tiny prepared package can drive a self-contained run."""
    input_directory = tmp_path / "inputs"
    run_directory = tmp_path / "run"

    prepared = prepare_pynamit_inputs(
        input_directory, final_time=0.0, Nmax=2, Mmax=1, Ncs=8, artifact_storage="netcdf"
    )
    assert prepared.run_directory == str(input_directory.resolve())
    assert (input_directory / INPUT_MANIFEST_FILENAME).exists()
    manifest = read_input_manifest(input_directory)
    assert manifest["input_contract"]["coefficient_space"]["Nmax"] == 2
    assert manifest["metadata"]["external_input_source"] in {"auto", "fallback", "native"}

    run = run_pynamit_from_inputs(
        input_directory,
        run_directory=run_directory,
        final_time=0.0,
        dt=0.01,
        plotsteps=1,
        artifact_storage="netcdf",
    )

    assert run.run_directory == str(run_directory.resolve())
    assert "state" in run.output_timeseries.datasets
    assert "conductance" in run.input_timeseries.datasets
    assert (run_directory / "conductance.ncdf").exists()
    assert (run_directory / RUN_MANIFEST_FILENAME).exists()


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
            plotsteps=1,
            artifact_storage="netcdf",
        )
