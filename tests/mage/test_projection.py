"""Tests for projecting prepared MAGE forcing into PynaMIT."""

import datetime as dt
import json

import h5py
import matplotlib as mpl
import numpy as np
import pytest
from tests.mage._support import _FakeH5, _write_projection_forcing

from pynamit.simulation.input_manifest import clear_prepared_input_package
from pynamit.workflows.mage.diagnostics import write_input_projection_diagnostics
from pynamit.workflows.mage.gamera import _centered_dipole_alignment_attrs
from pynamit.workflows.mage.preparation import _create_output_datasets
from pynamit.workflows.mage.prepared_forcing import (
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    forcing_times,
    validate_prepared_forcing,
)
from pynamit.workflows.mage.projection import (
    MAGE_MAIN_FIELD_KIND,
    _boundary_radius,
    _dipole_B0,
    _gamera_dipole_metadata,
    _load_weighted_winds,
    _source_file_metadata,
    prepare_inputs,
)
from pynamit.workflows.mage.remix import _upward_fac_to_radial_current


def test_prepared_forcing_schema_contains_only_projection_inputs(tmp_path):
    """Preparation should not allocate unused diagnostic datasets."""
    output_path = tmp_path / "prepared.h5"
    with h5py.File(output_path, "w") as output:
        _create_output_datasets(
            output, n_steps=2, ion_shape=(3, 4), inner_shape=(5, 6), compression="none"
        )

        assert set(output) == {
            "jr",
            "SH",
            "SP",
            "u_p_theta",
            "u_p_phi",
            "u_h_theta",
            "u_h_phi",
            "delta_Br",
        }


def test_mage_projection_uses_kaiju_dipole_by_default():
    """MAGE projection uses Kaiju dipole physics on a GEO model grid."""
    assert MAGE_MAIN_FIELD_KIND == "kaiju_dipole"


def test_mage_projection_replaces_stale_pynamit_input_artifacts(tmp_path):
    """Reprojection must not retain old forcing or source artifacts."""
    stale_artifacts = (tmp_path / "boundary_jr.ncdf", tmp_path / "dynamic.ncdf")
    for path in stale_artifacts:
        path.write_text("stale", encoding="utf-8")
    (tmp_path / "u.zarr").mkdir()
    (tmp_path / "pynamit_input_manifest.json").write_text("{}", encoding="utf-8")
    unrelated_path = tmp_path / "notes.txt"
    unrelated_path.write_text("keep", encoding="utf-8")

    clear_prepared_input_package(tmp_path, artifact_storage="netcdf")

    assert not any(path.exists() for path in stale_artifacts)
    assert not (tmp_path / "u.zarr").exists()
    assert not (tmp_path / "pynamit_input_manifest.json").exists()
    assert unrelated_path.read_text(encoding="utf-8") == "keep"


def test_invalid_forcing_does_not_remove_existing_projection(tmp_path):
    """Validate source before replacing projected inputs."""
    forcing_path = tmp_path / "incomplete.h5"
    input_directory = tmp_path / "projection"
    input_directory.mkdir()
    existing_input = input_directory / "boundary_jr.ncdf"
    existing_input.write_text("existing", encoding="utf-8")
    with h5py.File(forcing_path, "w") as output:
        output.attrs["kind"] = MAGE_FORCING_KIND
        output.attrs["version"] = MAGE_FORCING_VERSION
        output.attrs["complete"] = False

    with pytest.raises(RuntimeError, match="incomplete"):
        prepare_inputs(
            forcing_path=forcing_path,
            input_directory=input_directory,
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=None,
            boundary_Br_lambda=0.1,
            conductance_lambda=0.1,
            boundary_jr_lambda=0.1,
            e_neutral_wind_lambda=0.1,
            artifact_storage="netcdf",
        )

    assert existing_input.read_text(encoding="utf-8") == "existing"


def test_prepared_forcing_rejects_incompatible_lower_dynamo_parameters(tmp_path):
    """Reject conductance prepared with another lower extension."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing.attrs["tiegcm_hall_lower_scale_m"] = 4_000.0

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="lower-dynamo parameters.*hall"):
            validate_prepared_forcing(forcing)


def test_prepared_forcing_rejects_incompatible_conductance_floor(tmp_path):
    """Projection requires the exact floor used by the MAGE case."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing.attrs["pedersen_conductance_floor_S"] = 1.5

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="incompatible conductance floors"):
            validate_prepared_forcing(forcing)


def test_prepared_forcing_rejects_unfloored_global_conductance(tmp_path):
    """Validate the prepared values, not only floor metadata."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing["SH"][:, 0, :] = 0.5

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="Hall conductance violates.*global hard floor"):
            validate_prepared_forcing(forcing)


def test_prepared_forcing_rejects_incompatible_sm_time_convention(tmp_path):
    """Projection must not silently mix SM timestamp conventions."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing.attrs["gamera_sm_transform_time_convention"] = "fractional_source_time"

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="nearest-second SM transform"):
            validate_prepared_forcing(forcing)


def test_prepared_forcing_rejects_inconsistent_source_time_offsets(tmp_path):
    """Source-time offsets must agree with stored timestamps."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing["gamera_time_offset_seconds"][1] = 0.01

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="gamera_time_offset_seconds.*timestamps"):
            validate_prepared_forcing(forcing)


def test_load_weighted_winds_requires_prepared_hall_products():
    """Projection does not reconstruct missing winds from TIEGCM."""
    h5_like = {"u_p_theta": np.zeros((1, 2)), "u_p_phi": np.zeros((1, 2))}

    with pytest.raises(RuntimeError, match="u_h_theta"):
        _load_weighted_winds(h5_like, 0)


def test_load_weighted_winds_reads_all_prepared_products():
    """Projection loads Pedersen and Hall weighted winds from HDF5."""
    h5_like = {
        "u_p_theta": np.array([[1.0, 2.0]]),
        "u_p_phi": np.array([[3.0, 4.0]]),
        "u_h_theta": np.array([[5.0, 6.0]]),
        "u_h_phi": np.array([[7.0, 8.0]]),
    }

    loaded = _load_weighted_winds(h5_like, 0)

    for value, expected in zip(
        loaded, ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]), strict=True
    ):
        np.testing.assert_allclose(value, expected)


def test_upward_remix_fac_is_projected_onto_the_radial_direction():
    """Upward FAC becomes radial current with the dip-angle factor."""
    fac = np.array([2.0, -3.0, 4.0])
    magnetic_latitude = np.array([90.0, -45.0, 0.0])

    radial_current = _upward_fac_to_radial_current(fac, magnetic_latitude)

    np.testing.assert_allclose(radial_current, [2.0, -3.0 * np.sqrt(0.8), 0.0])


def test_projection_geometry_requires_prepared_radius_or_explicit_value():
    """RM must come from the prepared file or the edited settings."""
    with pytest.raises(RuntimeError, match="boundary_radius"):
        _boundary_radius(_FakeH5(), explicit_radius=None)

    assert _boundary_radius(_FakeH5(), explicit_radius=7.0) == 7.0
    with pytest.raises(ValueError, match="positive"):
        _boundary_radius(_FakeH5(), explicit_radius=0.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        _boundary_radius(
            _FakeH5(boundary_radius=np.array([7.0e6, np.nan]), boundary_solid_angle=np.ones(2)),
            explicit_radius=None,
        )

    weighted = _FakeH5(
        boundary_radius=np.array([6.0e6, 8.0e6]), boundary_solid_angle=np.array([1.0, 3.0])
    )
    assert _boundary_radius(weighted, explicit_radius=None) == pytest.approx(7.5e6)


def test_projection_geometry_requires_prepared_dipole_strength_or_explicit_value():
    """B0 must come from prepared metadata or the edited settings."""
    with pytest.raises(RuntimeError, match="dipole strength"):
        _dipole_B0(_FakeH5(), explicit_B0=None)

    assert _dipole_B0(_FakeH5(), explicit_B0=3.0e-5) == 3.0e-5
    assert _dipole_B0(_FakeH5(attrs={"main_field_B0_T": 3.0e-5}), explicit_B0=None) == 3.0e-5
    with pytest.raises(ValueError, match="positive"):
        _dipole_B0(_FakeH5(), explicit_B0=np.nan)


def test_projection_geometry_requires_prepared_dipole_axes():
    """GAMERA dipole axis metadata should not be guessed."""
    with pytest.raises(RuntimeError, match="GAMERA dipole metadata"):
        _gamera_dipole_metadata(_FakeH5(attrs={"gamera_mag_m0_nT": -30_000.0}))

    details = _gamera_dipole_metadata(
        _FakeH5(
            attrs={
                "gamera_mag_m0_nT": -30_000.0,
                "gamera_internal_dipole_moment_axis": [0.0, 0.0, -2.0],
                "gamera_internal_magnetic_north_axis": [0.0, 0.0, 4.0],
            }
        )
    )

    assert details["mag_m0_nT"] == -30_000.0
    np.testing.assert_allclose(details["moment_axis"], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(details["north_axis"], [0.0, 0.0, 1.0])

    inconsistent = _FakeH5(
        attrs={
            "gamera_mag_m0_nT": -30_000.0,
            "gamera_internal_dipole_moment_axis": [0.0, 0.0, -1.0],
            "gamera_internal_magnetic_north_axis": [0.0, 0.0, -1.0],
        }
    )
    with pytest.raises(RuntimeError, match="antiparallel"):
        _gamera_dipole_metadata(inconsistent)

    reversed_axis = _FakeH5(
        attrs={
            "gamera_mag_m0_nT": 30_000.0,
            "gamera_internal_dipole_moment_axis": [0.0, 0.0, 1.0],
            "gamera_internal_magnetic_north_axis": [0.0, 0.0, -1.0],
        }
    )
    with pytest.raises(RuntimeError, match=r"magnetic north along \+Z"):
        _gamera_dipole_metadata(reversed_axis)


def test_mage_projection_times_are_relative_to_first_hdf5_time():
    """Projection should preserve the 18:00:10 event-time origin."""
    times, seconds = forcing_times(
        [b"2011-10-24T18:00:10", b"2011-10-24T18:00:20", b"2011-10-24T18:00:40"]
    )

    assert times[0].isoformat() == "2011-10-24T18:00:10"
    np.testing.assert_allclose(seconds, np.array([0.0, 10.0, 30.0]))


def test_mage_projection_normalizes_timezone_offsets_to_utc():
    """Equivalent timestamp offsets should share one time axis."""
    times, seconds = forcing_times(["2011-10-24T20:00:10+02:00", "2011-10-24T18:00:20+00:00"])

    assert times == [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    np.testing.assert_allclose(seconds, [0.0, 10.0])


def test_mage_source_file_metadata_is_lightweight(tmp_path):
    """Projection provenance records source path, size, and mtime."""
    source = tmp_path / "forcing.h5"
    source.write_bytes(b"abc")

    metadata = _source_file_metadata(source)

    assert metadata["path"] == str(source.resolve())
    assert metadata["size_bytes"] == 3
    assert metadata["mtime_ns"] > 0
    assert "mtime" in metadata


def test_mage_projection_reuses_geometry_for_complete_input_series(tmp_path):
    """Project a MAGE time series through one fixed geometry."""
    forcing_path = tmp_path / "forcing.h5"
    input_directory = tmp_path / "projection"
    _write_projection_forcing(forcing_path)

    result = prepare_inputs(
        forcing_path=forcing_path,
        input_directory=input_directory,
        dipole_B0_override=None,
        boundary_radius_override=None,
        nmax=2,
        mmax=1,
        ncs=4,
        max_steps=None,
        boundary_Br_lambda=0.1,
        conductance_lambda=0.1,
        boundary_jr_lambda=0.1,
        e_neutral_wind_lambda=0.1,
        artifact_storage="netcdf",
    )

    assert result == input_directory
    assert (input_directory / "pynamit_input_manifest.json").is_file()
    manifest = json.loads(
        (input_directory / "pynamit_input_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["projection_regularization"]["boundary_Br_lambda"] == 0.1
    for dataset in ("boundary_Br", "boundary_jr", "conductance", "E_neutral_wind"):
        assert (input_directory / f"{dataset}.ncdf").is_file()


def test_projection_diagnostics_write_figure_and_area_weighted_metrics(monkeypatch, tmp_path):
    """Compare a projection package with its prepared source."""
    forcing_path = tmp_path / "forcing.h5"
    input_directory = tmp_path / "projection"
    _write_projection_forcing(forcing_path)
    prepare_inputs(
        forcing_path=forcing_path,
        input_directory=input_directory,
        dipole_B0_override=None,
        boundary_radius_override=None,
        nmax=2,
        mmax=1,
        ncs=4,
        max_steps=1,
        boundary_Br_lambda=0.1,
        conductance_lambda=0.1,
        boundary_jr_lambda=0.1,
        e_neutral_wind_lambda=0.1,
        artifact_storage="netcdf",
    )
    monkeypatch.setattr(
        "pynamit.workflows.mage.diagnostics.style_global_input_axis", lambda *args, **kwargs: None
    )

    with mpl.rc_context({"text.usetex": False}):
        result = write_input_projection_diagnostics(
            forcing_path, input_directory, timesteps=None, fields=("etaP", "SigmaP")
        )

    assert result["figure"].is_file()
    assert result["metrics"].is_file()
    report = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert report["selected_steps"] == [
        {"index": 0, "timestamp": "2020-01-01T00:00:00", "time_seconds": 0.0}
    ]
    assert set(report["aggregate"]) == {"etaP", "SigmaP"}
    assert report["aggregate"]["etaP"]["weighted_rms_error"] >= 0.0
    assert report["aggregate"]["etaP"]["expected_maximum"] == pytest.approx(0.4)
    assert report["aggregate"]["SigmaP"]["expected_minimum"] == 2.0
    assert "projected_below_minimum_area_fraction" in report["aggregate"]["SigmaP"]


def test_mage_projection_rejects_incompatible_prepared_units(tmp_path):
    """The forcing contract must make unit conversions explicit."""
    forcing_path = tmp_path / "forcing.h5"
    input_directory = tmp_path / "projection"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing["delta_Br"].attrs["units"] = "T"

    with pytest.raises(RuntimeError, match="incompatible dataset units.*delta_Br"):
        prepare_inputs(
            forcing_path=forcing_path,
            input_directory=input_directory,
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=None,
            boundary_Br_lambda=0.1,
            conductance_lambda=0.1,
            boundary_jr_lambda=0.1,
            e_neutral_wind_lambda=0.1,
            artifact_storage="netcdf",
        )


def test_mage_projection_requires_pynamit_dipole_reference_radius(tmp_path):
    """Prepared B0 must declare the radius used by PynaMIT's dipole."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as output:
        output.attrs["main_field_B0_reference_radius_m"] = 6_378_100.0

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="dipole reference radius"):
            validate_prepared_forcing(forcing)


def test_projection_failure_preserves_last_complete_package(tmp_path):
    """A failed projection must not replace reusable inputs."""
    forcing_path = tmp_path / "forcing.h5"
    input_directory = tmp_path / "projection"
    input_directory.mkdir()
    existing_input = input_directory / "boundary_jr.ncdf"
    existing_input.write_text("existing", encoding="utf-8")
    hall = np.ones((2, 4, 4)) * 5.0
    hall[1] = -1.0
    _write_projection_forcing(forcing_path, hall_conductance=hall)

    with pytest.raises(RuntimeError, match="Hall conductance violates.*global hard floor"):
        prepare_inputs(
            forcing_path=forcing_path,
            input_directory=input_directory,
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=None,
            boundary_Br_lambda=0.1,
            conductance_lambda=0.1,
            boundary_jr_lambda=0.1,
            e_neutral_wind_lambda=0.1,
            artifact_storage="netcdf",
        )

    assert existing_input.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*-projecting-*"))


def test_projection_rejects_subfloor_hall_anywhere_on_global_sheet(tmp_path):
    """The global sheet floor is independent of ReMIX FAC coverage."""
    forcing_path = tmp_path / "forcing.h5"
    input_directory = tmp_path / "projection"
    hall = np.full((2, 4, 4), 5.0)
    hall[:, 1:3, :] = 0.0
    _write_projection_forcing(forcing_path, hall_conductance=hall)

    with pytest.raises(RuntimeError, match="Hall conductance violates.*global hard floor"):
        prepare_inputs(
            forcing_path=forcing_path,
            input_directory=input_directory,
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=1,
            boundary_Br_lambda=0.1,
            conductance_lambda=0.1,
            boundary_jr_lambda=0.1,
            e_neutral_wind_lambda=0.1,
            artifact_storage="netcdf",
        )


def test_mage_workflow_contract_is_kaiju_dipole():
    """All MAGE phases share the Kaiju MAG alignment."""
    event_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    attributes = _centered_dipole_alignment_attrs(event_time, -30_000.0)
    assert MAGE_MAIN_FIELD_KIND == "kaiju_dipole"
    assert attributes["main_field_kind"] == MAGE_MAIN_FIELD_KIND
    assert attributes["main_field_horizontal_coordinate_system"] == "geocentric_geographic"
