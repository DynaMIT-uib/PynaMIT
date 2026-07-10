"""Tests for the MAGE forcing preparation helpers."""

from pathlib import Path

import numpy as np
import pytest

from scripts.simulation.mage_forcing_final import SETTINGS as MAGE_RUN_SETTINGS
from scripts.simulation.mage_forcing_final import DEFAULT_MAGE_RUN_ROOT as MAGE_RUN_ROOT
from scripts.simulation.mage_project_inputs import DEFAULT_FORCING_CANDIDATES
from scripts.simulation.mage_project_inputs import DEFAULT_INPUT_DIRECTORY
from scripts.simulation.mage_project_inputs import DEFAULT_MAGE_RUN_ROOT as MAGE_PROJECT_ROOT
from scripts.simulation.mage_project_inputs import DEFAULT_RESULT_DIRECTORY
from scripts.simulation.mage_project_inputs import SETTINGS as MAGE_PROJECT_SETTINGS
from scripts.simulation.mage_project_inputs import _clear_existing_input_package
from scripts.simulation.mage_prepare_forcing import (
    DEFAULT_GAMERA_DIR,
    DEFAULT_OUTPUT_NAME,
    integrate_tiegcm_step,
    wrap_longitude_180_value,
)
from pynamit.simulation.mage_workflow import (
    boundary_radius_from_h5,
    dipole_B0_from_h5,
    file_fingerprint,
    gamera_internal_dipole_details,
    h5_time_vector_seconds,
    load_weighted_winds,
    projection_directory_for_resolution,
    result_directory_for_resolution,
)


class _FakeVariable:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, item):
        return self.values[item]


class _FakeDataset:
    def __init__(self, **variables):
        self.variables = {name: _FakeVariable(values) for name, values in variables.items()}


class _FakeH5(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})


def _two_layer_tiegcm_dataset():
    """Return a tiny TIEGCM-like dataset with cm inputs."""
    return _FakeDataset(
        SIGMA_PED=np.array([[[1.0, 2.0], [3.0, 0.5], [99.0, 99.0]]]),
        SIGMA_HAL=np.array([[[4.0, 1.0], [1.0, 2.0], [99.0, 99.0]]]),
        ZG=np.array([[[10_000.0, 20_000.0], [20_000.0, 35_000.0], [50_000.0, 65_000.0]]]),
        UN=np.array([[[1000.0, -2000.0], [3000.0, 4000.0], [999.0, 999.0]]]),
        VN=np.array([[[500.0, 2000.0], [1.0e31, -1000.0], [999.0, 999.0]]]),
        gzigm1=np.array([[1200.0, 310.0]]),
        gzigm2=np.array([[500.0, 750.0]]),
    )


def test_default_prepared_forcing_artifact_name_is_canonical():
    """Default HDF5 name should match the final script search path."""
    assert DEFAULT_OUTPUT_NAME == "mage_prepared_forcing.h5"
    assert DEFAULT_FORCING_CANDIDATES[0].name == DEFAULT_OUTPUT_NAME


def test_default_gamera_directory_is_cluster_path():
    """Preparation defaults to the intended MAGE machine data path."""
    assert DEFAULT_GAMERA_DIR == Path("/disk/Gamera_Dong")


def test_projected_input_default_matches_run_input_directory():
    """Projection and run scripts should agree on the input package."""
    assert MAGE_PROJECT_SETTINGS.input_directory is None
    assert MAGE_RUN_SETTINGS.input_directory is None
    assert (
        projection_directory_for_resolution(
            MAGE_PROJECT_SETTINGS.nmax,
            MAGE_PROJECT_SETTINGS.mmax,
            MAGE_PROJECT_SETTINGS.ncs,
            MAGE_PROJECT_ROOT,
        )
        == DEFAULT_INPUT_DIRECTORY
    )
    assert (
        projection_directory_for_resolution(
            MAGE_RUN_SETTINGS.nmax, MAGE_RUN_SETTINGS.mmax, MAGE_RUN_SETTINGS.ncs, MAGE_RUN_ROOT
        )
        == DEFAULT_INPUT_DIRECTORY
    )
    assert (
        result_directory_for_resolution(
            MAGE_RUN_SETTINGS.nmax, MAGE_RUN_SETTINGS.mmax, MAGE_RUN_SETTINGS.ncs, MAGE_RUN_ROOT
        )
        == DEFAULT_RESULT_DIRECTORY
    )


def test_mage_projection_uses_kaiju_dipole_by_default():
    """MAGE projection should use the Kaiju/Geopack SM dipole."""
    assert MAGE_PROJECT_SETTINGS.mainfield_kind == "kaiju_dipole"


def test_mage_projection_replaces_stale_pynamit_input_artifacts(tmp_path):
    """Reprojection must not retain old forcing or source artifacts."""
    stale_artifacts = (tmp_path / "jr.ncdf", tmp_path / "state.ncdf")
    for path in stale_artifacts:
        path.write_text("stale", encoding="utf-8")
    (tmp_path / "u.zarr").mkdir()
    (tmp_path / "pynamit_input_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mage_input_metadata.json").write_text("{}", encoding="utf-8")
    unrelated_path = tmp_path / "notes.txt"
    unrelated_path.write_text("keep", encoding="utf-8")

    _clear_existing_input_package(tmp_path, artifact_storage="netcdf")

    assert not any(path.exists() for path in stale_artifacts)
    assert not (tmp_path / "u.zarr").exists()
    assert not (tmp_path / "pynamit_input_manifest.json").exists()
    assert not (tmp_path / "mage_input_metadata.json").exists()
    assert unrelated_path.read_text(encoding="utf-8") == "keep"


def test_load_weighted_winds_requires_prepared_hall_products():
    """Projection does not reconstruct missing winds from TIEGCM."""
    h5_like = {"We": np.zeros((1, 2)), "Wn": np.zeros((1, 2))}

    with pytest.raises(RuntimeError, match="WeH"):
        load_weighted_winds(h5_like, 0)


def test_load_weighted_winds_reads_all_prepared_products():
    """Projection loads Pedersen and Hall weighted winds from HDF5."""
    h5_like = {
        "We": np.array([[1.0, 2.0]]),
        "Wn": np.array([[3.0, 4.0]]),
        "WeH": np.array([[5.0, 6.0]]),
        "WnH": np.array([[7.0, 8.0]]),
    }

    loaded = load_weighted_winds(h5_like, 0)

    for value, expected in zip(loaded, ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0])):
        np.testing.assert_allclose(value, expected)


def test_projection_geometry_requires_prepared_radius_or_explicit_value():
    """RM must come from the prepared file or the edited settings."""
    with pytest.raises(RuntimeError, match="radius dataset 'r'"):
        boundary_radius_from_h5(_FakeH5(), explicit_rm=None)

    assert boundary_radius_from_h5(_FakeH5(), explicit_rm=7.0) == 7.0


def test_projection_geometry_requires_prepared_dipole_strength_or_explicit_value():
    """B0 must come from prepared metadata or the edited settings."""
    with pytest.raises(RuntimeError, match="dipole strength"):
        dipole_B0_from_h5(_FakeH5(), explicit_B0=None)

    assert dipole_B0_from_h5(_FakeH5(), explicit_B0=3.0e-5) == 3.0e-5
    assert (
        dipole_B0_from_h5(_FakeH5(attrs={"gamera_mag_m0_nT": -30_000.0}), explicit_B0=None)
        == 3.0e-5
    )


def test_projection_geometry_requires_prepared_dipole_axes():
    """GAMERA dipole axis metadata should not be guessed."""
    with pytest.raises(RuntimeError, match="GAMERA dipole metadata"):
        gamera_internal_dipole_details(_FakeH5(attrs={"gamera_mag_m0_nT": -30_000.0}))

    details = gamera_internal_dipole_details(
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


def test_mage_projection_times_are_relative_to_first_hdf5_time():
    """Projection should preserve the 18:00:10 event-time origin."""
    times, seconds = h5_time_vector_seconds(
        [b"2011-10-24T18:00:10", b"2011-10-24T18:00:20", b"2011-10-24T18:00:40"]
    )

    assert times[0].isoformat() == "2011-10-24T18:00:10"
    np.testing.assert_allclose(seconds, np.array([0.0, 10.0, 30.0]))


def test_mage_source_file_fingerprint_is_lightweight(tmp_path):
    """Projection provenance records source path, size, and mtime."""
    source = tmp_path / "forcing.h5"
    source.write_bytes(b"abc")

    fingerprint = file_fingerprint(source)

    assert fingerprint["path"] == str(source.resolve())
    assert fingerprint["size_bytes"] == 3
    assert fingerprint["mtime_ns"] > 0
    assert "mtime" in fingerprint
    assert file_fingerprint(None) is None


def test_integrate_tiegcm_step_computed_conductances_and_weighted_winds():
    """Computed outputs should match layer integrals."""
    dataset = _two_layer_tiegcm_dataset()

    integrated = integrate_tiegcm_step(dataset, 0, conductance_source="computed")

    dz = np.array([[100.0, 150.0], [300.0, 300.0]])
    sigma_p = np.array([[1.0, 2.0], [3.0, 0.5]])
    sigma_h = np.array([[4.0, 1.0], [1.0, 2.0]])
    east = np.array([[10.0, -20.0], [30.0, 40.0]])
    north = np.array([[5.0, 20.0], [np.nan, -10.0]])
    sp = np.nansum(sigma_p * dz, axis=0)
    sh = np.nansum(sigma_h * dz, axis=0)

    np.testing.assert_allclose(integrated["SP"], sp)
    np.testing.assert_allclose(integrated["SH"], sh)
    np.testing.assert_allclose(integrated["We"], np.nansum(sigma_p * east * dz, axis=0) / sp)
    np.testing.assert_allclose(integrated["Wn"], np.nansum(sigma_p * north * dz, axis=0) / sp)
    np.testing.assert_allclose(integrated["WeH"], np.nansum(sigma_h * east * dz, axis=0) / sh)
    np.testing.assert_allclose(integrated["WnH"], np.nansum(sigma_h * north * dz, axis=0) / sh)


def test_integrate_tiegcm_step_native_conductances_preserve_wind_current_numerators():
    """Native conductances should preserve source numerators."""
    dataset = _two_layer_tiegcm_dataset()

    integrated = integrate_tiegcm_step(dataset, 0, conductance_source="native")

    dz = np.array([[100.0, 150.0], [300.0, 300.0]])
    sigma_p = np.array([[1.0, 2.0], [3.0, 0.5]])
    sigma_h = np.array([[4.0, 1.0], [1.0, 2.0]])
    east = np.array([[10.0, -20.0], [30.0, 40.0]])
    north = np.array([[5.0, 20.0], [np.nan, -10.0]])

    native_sp = np.array([1200.0, 310.0])
    native_sh = np.array([500.0, 750.0])
    np.testing.assert_allclose(integrated["SP"], native_sp)
    np.testing.assert_allclose(integrated["SH"], native_sh)
    np.testing.assert_allclose(
        integrated["SP"] * integrated["We"], np.nansum(sigma_p * east * dz, axis=0)
    )
    np.testing.assert_allclose(
        integrated["SP"] * integrated["Wn"], np.nansum(sigma_p * north * dz, axis=0)
    )
    np.testing.assert_allclose(
        integrated["SH"] * integrated["WeH"], np.nansum(sigma_h * east * dz, axis=0)
    )
    np.testing.assert_allclose(
        integrated["SH"] * integrated["WnH"], np.nansum(sigma_h * north * dz, axis=0)
    )


def test_integrate_tiegcm_step_zero_conductance_returns_zero_weighted_winds():
    """Zero conductance should produce zero winds instead of NaNs."""
    dataset = _FakeDataset(
        SIGMA_PED=np.zeros((1, 3, 2)),
        SIGMA_HAL=np.zeros((1, 3, 2)),
        ZG=np.array([[[10_000.0, 20_000.0], [20_000.0, 35_000.0], [50_000.0, 65_000.0]]]),
        UN=np.full((1, 3, 2), 1000.0),
        VN=np.full((1, 3, 2), -2000.0),
        gzigm1=np.zeros((1, 2)),
        gzigm2=np.zeros((1, 2)),
    )

    integrated = integrate_tiegcm_step(dataset, 0, conductance_source="computed")

    for key in ("SP", "SH", "We", "Wn", "WeH", "WnH"):
        np.testing.assert_allclose(integrated[key], 0.0)


def test_wrap_longitude_180_value_accepts_arrays():
    """REMIX longitude grids should wrap elementwise."""
    values = np.array([[0.0, 180.0, 181.0], [-181.0, 540.0, -540.0]])

    wrapped = wrap_longitude_180_value(values)

    np.testing.assert_allclose(wrapped, np.array([[0.0, -180.0, -179.0], [179.0, -180.0, -180.0]]))


def test_wrap_longitude_180_value_preserves_scalar_return_type():
    """Scalar callers still get a Python float."""
    wrapped = wrap_longitude_180_value(181.0)

    assert isinstance(wrapped, float)
    assert wrapped == -179.0
