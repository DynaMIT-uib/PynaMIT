"""Tests for MAGE preparation, projection, and run conventions."""

import datetime as dt
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scripts.simulation.mage_prepare import (
    DEFAULT_GAMERA_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_OUTPUT_NAME,
    PreparationSettings,
    _atomic_prepared_output,
    _centered_dipole_alignment_attrs,
    _create_output_datasets,
    _gamera_internal_dipole_axes,
    _integrate_tiegcm_step,
    _interpolate_to_tiegcm_grid,
    _read_tiegcm_step,
    _resolve_tiegcm_path,
    _tiegcm_times,
    _validate_settings,
    _validate_source_times,
    _write_static_datasets,
)
from scripts.simulation.mage_project import CASE_DIRECTORY as MAGE_PROJECT_CASE
from scripts.simulation.mage_project import DEFAULT_FORCING_PATH
from scripts.simulation.mage_project import SETTINGS as MAGE_PROJECT_SETTINGS
from scripts.simulation.mage_run import CASE_DIRECTORY as MAGE_RUN_CASE
from scripts.simulation.mage_run import DEFAULT_PROJECTION_DIRECTORY
from scripts.simulation.mage_run import SETTINGS as MAGE_RUN_SETTINGS

from pynamit.simulation.workflows.mage_projection import (
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    _boundary_radius,
    _clear_existing_input_package,
    _dipole_B0,
    _gamera_dipole_metadata,
    _h5_time_vector_seconds,
    _load_weighted_winds,
    _source_file_metadata,
    project_inputs,
)


class _FakeVariable:
    def __init__(self, values, **attrs):
        self.values = np.asanyarray(values)
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getitem__(self, item):
        return self.values[item]


class _FakeDataset:
    def __init__(self, **variables):
        self.variables = {
            name: values if isinstance(values, _FakeVariable) else _FakeVariable(values)
            for name, values in variables.items()
        }


class _FakeH5(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})


def _write_projection_forcing(path: Path, *, hall_conductance=5.0) -> None:
    """Write a complete, tiny prepared-forcing contract."""
    latitude, longitude = np.meshgrid(
        np.array([-60.0, -20.0, 20.0, 60.0]), np.array([-135.0, -45.0, 45.0, 135.0]), indexing="ij"
    )
    step_shape = (2, *latitude.shape)
    hall = np.broadcast_to(np.asarray(hall_conductance, dtype=float), step_shape)

    with h5py.File(path, "w") as output:
        output.create_dataset(
            "time",
            data=np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:10"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        for name, values in {
            "r": np.full(latitude.shape, 7.0e6),
            "glat": latitude,
            "glon": longitude,
            "Blat": latitude,
            "Blon": longitude,
            "Bu": np.full(step_shape, 10.0),
            "FAC": np.full(step_shape, 0.1),
            "SH": hall,
            "SP": np.full(step_shape, 10.0),
            "We": np.full(step_shape, 50.0),
            "Wn": np.full(step_shape, -20.0),
            "WeH": np.full(step_shape, 30.0),
            "WnH": np.full(step_shape, 10.0),
        }.items():
            output.create_dataset(name, data=values)
        for name, units in {
            "r": "m",
            "glat": "degree",
            "glon": "degree",
            "Blat": "degree",
            "Blon": "degree",
            "Bu": "nT",
            "FAC": "uA m-2",
            "SH": "S",
            "SP": "S",
            "We": "m s-1",
            "Wn": "m s-1",
            "WeH": "m s-1",
            "WnH": "m s-1",
        }.items():
            output[name].attrs["units"] = units
        output.attrs["gamera_mag_m0_nT"] = -30_000.0
        output.attrs["gamera_internal_dipole_moment_axis"] = [0.0, 0.0, -1.0]
        output.attrs["gamera_internal_magnetic_north_axis"] = [0.0, 0.0, 1.0]
        output.attrs["gamera_coordinate_system"] = "SM"
        output.attrs["fac_convention"] = "upward"
        output.attrs["kind"] = MAGE_FORCING_KIND
        output.attrs["version"] = MAGE_FORCING_VERSION
        output.attrs["complete"] = True


def test_gamera_signed_dipole_axes_follow_magnetic_poles():
    """GAMERA moment sign determines magnetic north."""
    earth_like = _gamera_internal_dipole_axes(-30_000.0)
    reversed_field = _gamera_internal_dipole_axes(30_000.0)

    np.testing.assert_array_equal(earth_like["moment_axis"], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(earth_like["north_axis"], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(reversed_field["north_axis"], [0.0, 0.0, -1.0])


@pytest.mark.parametrize("mag_m0", [0.0, np.nan, np.inf])
def test_gamera_dipole_axes_reject_unknown_orientation(mag_m0):
    """Preparation must not guess a missing or invalid dipole sign."""
    with pytest.raises(ValueError, match="finite and nonzero"):
        _gamera_internal_dipole_axes(mag_m0)


def test_centered_dipole_alignment_uses_gamera_axis_convention():
    """Prepared alignment metadata carries the signed GAMERA axes."""
    attrs = _centered_dipole_alignment_attrs(dt.datetime(2020, 1, 1), -30_000.0)

    assert attrs["gamera_coordinate_system"] == "SM"
    np.testing.assert_array_equal(attrs["gamera_internal_dipole_moment_axis"], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(attrs["gamera_internal_magnetic_north_axis"], [0.0, 0.0, 1.0])
    assert "gamera_internal_dipole_axis" not in attrs
    assert "pynamit_run_coordinate_system" not in attrs


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
    """Preparation and projection share one canonical forcing path."""
    assert DEFAULT_OUTPUT_NAME == "mage_prepared_forcing.h5"
    assert DEFAULT_FORCING_PATH == DEFAULT_OUTPUT_DIRECTORY / DEFAULT_OUTPUT_NAME


def test_default_gamera_directory_is_cluster_path():
    """Preparation defaults to the intended MAGE machine data path."""
    assert DEFAULT_GAMERA_DIRECTORY == Path("/disk/Gamera_Dong")


def test_ambiguous_tiegcm_discovery_requires_an_explicit_path(tmp_path):
    """Preparation must reject ambiguous forcing discovery."""
    for name in ("first_sech_tie.nc", "second_sech_tie.nc"):
        (tmp_path / name).touch()

    with pytest.raises(RuntimeError, match="multiple TIEGCM"):
        _resolve_tiegcm_path(tmp_path, explicit_path=None)


@pytest.mark.parametrize("time_dtype", ["S26", "U26", object])
def test_static_time_dataset_is_written_as_utf8(tmp_path, time_dtype):
    """Prepared ISO timestamps should be valid HDF5 UTF-8 strings."""
    time_values = np.array(
        ["2011-10-24T18:00:10.459051", "2011-10-24T18:00:20.459051"], dtype=time_dtype
    )
    output_path = tmp_path / "prepared.h5"
    grid = np.zeros((1, 1))

    with h5py.File(output_path, "w") as output:
        _write_static_datasets(
            output,
            time_values,
            dt.datetime(2011, 10, 24, 18, 0, 10),
            grid,
            grid,
            grid,
            grid,
            grid,
            PreparationSettings(gamera_directory=tmp_path),
            tmp_path,
            6.3781e6,
            -29_617.4,
            tmp_path / "tiegcm.nc",
        )

    with h5py.File(output_path) as output:
        assert output["time"].asstr()[:].tolist() == [
            "2011-10-24T18:00:10.459051",
            "2011-10-24T18:00:20.459051",
        ]
        assert output.attrs["kind"] == MAGE_FORCING_KIND
        assert output.attrs["version"] == MAGE_FORCING_VERSION
        assert not output.attrs["complete"]
        _, relative_seconds = _h5_time_vector_seconds(output["time"][:])

    np.testing.assert_array_equal(relative_seconds, [0.0, 10.0])


def test_prepared_forcing_schema_contains_only_projection_inputs(tmp_path):
    """Preparation should not allocate unused diagnostic datasets."""
    output_path = tmp_path / "prepared.h5"
    with h5py.File(output_path, "w") as output:
        _create_output_datasets(
            output, n_steps=2, ion_shape=(3, 4), inner_shape=(5, 6), compression="none"
        )

        assert set(output) == {"FAC", "SH", "SP", "We", "Wn", "WeH", "WnH", "Bu"}


def test_atomic_prepared_output_preserves_last_complete_file(tmp_path):
    """A failed preparation should preserve the published forcing."""
    output_path = tmp_path / "prepared.h5"
    with h5py.File(output_path, "w") as output:
        output.attrs["generation"] = "previous"

    with pytest.raises(RuntimeError, match="failed step"):
        with _atomic_prepared_output(output_path) as output:
            output.attrs["generation"] = "incomplete"
            raise RuntimeError("failed step")

    with h5py.File(output_path) as output:
        assert output.attrs["generation"] == "previous"
    assert not list(tmp_path.glob(".*.tmp.h5"))


def test_projected_input_default_matches_run_input_directory():
    """Projection and run scripts should agree on the input package."""
    assert MAGE_PROJECT_SETTINGS.projection_directory is None
    project_resolution = (
        f"N{MAGE_PROJECT_SETTINGS.nmax}_M{MAGE_PROJECT_SETTINGS.mmax}_"
        f"Ncs{MAGE_PROJECT_SETTINGS.ncs}"
    )
    assert MAGE_PROJECT_CASE == MAGE_RUN_CASE
    assert DEFAULT_PROJECTION_DIRECTORY == MAGE_PROJECT_CASE / "projections" / project_resolution
    assert MAGE_RUN_SETTINGS.projection_directory == DEFAULT_PROJECTION_DIRECTORY


def test_mage_projection_uses_kaiju_dipole_by_default():
    """MAGE projection should use the Kaiju/Geopack SM dipole."""
    assert MAGE_PROJECT_SETTINGS.main_field_kind == "kaiju_dipole"


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


def test_invalid_forcing_does_not_remove_existing_projection(tmp_path):
    """Validate source before replacing projected inputs."""
    forcing_path = tmp_path / "incomplete.h5"
    projection_directory = tmp_path / "projection"
    projection_directory.mkdir()
    existing_input = projection_directory / "jr.ncdf"
    existing_input.write_text("existing", encoding="utf-8")
    with h5py.File(forcing_path, "w") as output:
        output.attrs["kind"] = MAGE_FORCING_KIND
        output.attrs["version"] = MAGE_FORCING_VERSION
        output.attrs["complete"] = False

    with pytest.raises(RuntimeError, match="incomplete"):
        project_inputs(
            forcing_path=forcing_path,
            projection_directory=projection_directory,
            main_field_kind="dipole",
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=None,
            br_lambda=0.1,
            conductance_lambda=0.1,
            jr_lambda=0.1,
            e_source_lambda=0.1,
            artifact_storage="netcdf",
        )

    assert existing_input.read_text(encoding="utf-8") == "existing"


def test_load_weighted_winds_requires_prepared_hall_products():
    """Projection does not reconstruct missing winds from TIEGCM."""
    h5_like = {"We": np.zeros((1, 2)), "Wn": np.zeros((1, 2))}

    with pytest.raises(RuntimeError, match="WeH"):
        _load_weighted_winds(h5_like, 0)


def test_load_weighted_winds_reads_all_prepared_products():
    """Projection loads Pedersen and Hall weighted winds from HDF5."""
    h5_like = {
        "We": np.array([[1.0, 2.0]]),
        "Wn": np.array([[3.0, 4.0]]),
        "WeH": np.array([[5.0, 6.0]]),
        "WnH": np.array([[7.0, 8.0]]),
    }

    loaded = _load_weighted_winds(h5_like, 0)

    for value, expected in zip(
        loaded, ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]), strict=True
    ):
        np.testing.assert_allclose(value, expected)


def test_projection_geometry_requires_prepared_radius_or_explicit_value():
    """RM must come from the prepared file or the edited settings."""
    with pytest.raises(RuntimeError, match="radius dataset 'r'"):
        _boundary_radius(_FakeH5(), explicit_radius=None)

    assert _boundary_radius(_FakeH5(), explicit_radius=7.0) == 7.0
    with pytest.raises(ValueError, match="positive"):
        _boundary_radius(_FakeH5(), explicit_radius=0.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        _boundary_radius(_FakeH5(r=np.array([7.0e6, np.nan])), explicit_radius=None)


def test_projection_geometry_requires_prepared_dipole_strength_or_explicit_value():
    """B0 must come from prepared metadata or the edited settings."""
    with pytest.raises(RuntimeError, match="dipole strength"):
        _dipole_B0(_FakeH5(), explicit_B0=None)

    assert _dipole_B0(_FakeH5(), explicit_B0=3.0e-5) == 3.0e-5
    assert _dipole_B0(_FakeH5(attrs={"gamera_mag_m0_nT": -30_000.0}), explicit_B0=None) == 3.0e-5
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
    times, seconds = _h5_time_vector_seconds(
        [b"2011-10-24T18:00:10", b"2011-10-24T18:00:20", b"2011-10-24T18:00:40"]
    )

    assert times[0].isoformat() == "2011-10-24T18:00:10"
    np.testing.assert_allclose(seconds, np.array([0.0, 10.0, 30.0]))


def test_mage_projection_normalizes_timezone_offsets_to_utc():
    """Equivalent timestamp offsets should share one time axis."""
    times, seconds = _h5_time_vector_seconds(
        ["2011-10-24T20:00:10+02:00", "2011-10-24T18:00:20+00:00"]
    )

    assert times == [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    np.testing.assert_allclose(seconds, [0.0, 10.0])


def test_mage_preparation_rejects_misaligned_source_times():
    """GAMERA and TIEGCM histories must correspond by time."""
    gamera_times = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    tiegcm_times = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 1, 20)]

    with pytest.raises(RuntimeError, match="not time-aligned"):
        _validate_source_times(gamera_times, tiegcm_times, tolerance_seconds=1.0)


def test_tiegcm_uses_mtime_instead_of_model_relative_time():
    """Canonical mtime defines TIEGCM history timestamps."""
    reference = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    dataset = _FakeDataset(
        time=_FakeVariable(
            [0.0, 10.0], units="seconds since 0000-01-01 00:00:00", calendar="standard"
        ),
        mtime=_FakeVariable([[297, 18, 0, 10], [297, 18, 0, 20]], dimensions=("time", "mtimedim")),
        year=_FakeVariable([2011, 2011], dimensions=("time",)),
    )

    times, tolerance = _tiegcm_times(dataset, reference)

    assert times == reference
    assert tolerance == 1.0


def test_tiegcm_three_component_mtime_uses_minute_precision():
    """Standard mtime triplets use their documented minute precision."""
    reference = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    dataset = _FakeDataset(
        mtime=_FakeVariable([[297, 18, 0], [297, 18, 0]], dimensions=("time", "mtimedim")),
        year=_FakeVariable([2011, 2011], dimensions=("time",)),
    )

    times, tolerance = _tiegcm_times(dataset, reference)

    assert times == [dt.datetime(2011, 10, 24, 18, 0)] * 2
    assert tolerance == 60.0


def test_tiegcm_mtime_uses_named_component_axis():
    """The mtimedim name determines axis order."""
    reference = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 1, 10)]
    dataset = _FakeDataset(
        mtime=_FakeVariable(
            [[297, 297], [18, 18], [0, 1], [10, 10]], dimensions=("mtimedim", "time")
        ),
        year=_FakeVariable([2011, 2011], dimensions=("time",)),
    )

    times, tolerance = _tiegcm_times(dataset, reference)

    assert times == reference
    assert tolerance == 1.0


def test_tiegcm_mtime_rejects_day_366_without_a_nearby_leap_year():
    """Do not roll an invalid day 366 into the next year."""
    dataset = _FakeDataset(
        mtime=_FakeVariable([[366, 0, 0]], dimensions=("time", "mtimedim")),
        year=_FakeVariable([2022], dimensions=("time",)),
    )

    with pytest.raises(RuntimeError, match="invalid value"):
        _tiegcm_times(dataset, [dt.datetime(2022, 12, 31)])


@pytest.mark.parametrize("max_steps", [True, 1.5])
def test_mage_step_limits_require_positive_integers(tmp_path, max_steps):
    """Preparation and projection must not truncate step limits."""
    with pytest.raises(ValueError, match="positive integer"):
        _validate_settings(PreparationSettings(max_steps=max_steps))

    with pytest.raises(ValueError, match="positive integer"):
        project_inputs(
            forcing_path=tmp_path / "missing.h5",
            projection_directory=tmp_path / "projection",
            main_field_kind="dipole",
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=max_steps,
            br_lambda=0.1,
            conductance_lambda=0.1,
            jr_lambda=0.1,
            e_source_lambda=0.1,
            artifact_storage="netcdf",
        )


@pytest.mark.parametrize("inner_index", [True, 0.5, -1])
def test_mage_inner_index_requires_a_nonnegative_integer(inner_index):
    """Reject values that cannot index a GAMERA shell."""
    with pytest.raises(ValueError, match="inner_index"):
        _validate_settings(PreparationSettings(inner_index=inner_index))


def test_tiegcm_reading_normalizes_masked_and_signed_fill_values():
    """NetCDF masks and either fill-value sign should become NaN."""
    values = np.ma.array([1.0, 2.0, 1.0e31, -1.0e31], mask=[False, True, False, False])
    dataset = _FakeDataset(variable=np.ma.array([values]))

    normalized = _read_tiegcm_step(dataset, "variable", 0)

    assert normalized[0] == 1.0
    assert np.isnan(normalized[1:]).all()


def test_remix_interpolation_is_periodic_across_longitude_seam():
    """Equivalent longitudes should receive the same interpolation."""
    source_lat = np.array([[60.0, 60.0], [80.0, 80.0]])
    source_lon = np.array([[-170.0, 170.0], [-170.0, 170.0]])
    values = np.array([[1.0, 1.0], [3.0, 3.0]])
    target_lon = np.array([[-179.0, 181.0, 359.0]])
    target_lat = np.full_like(target_lon, 70.0)

    interpolated = _interpolate_to_tiegcm_grid(
        source_lat, source_lon, values, target_lon, target_lat
    )

    np.testing.assert_allclose(interpolated, 2.0)


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
    projection_directory = tmp_path / "projection"
    _write_projection_forcing(forcing_path)

    result = project_inputs(
        forcing_path=forcing_path,
        projection_directory=projection_directory,
        main_field_kind="dipole",
        dipole_B0_override=None,
        boundary_radius_override=None,
        nmax=2,
        mmax=1,
        ncs=4,
        max_steps=None,
        br_lambda=0.1,
        conductance_lambda=0.1,
        jr_lambda=0.1,
        e_source_lambda=0.1,
        artifact_storage="netcdf",
    )

    assert result == projection_directory
    assert (projection_directory / "pynamit_input_manifest.json").is_file()
    manifest = json.loads(
        (projection_directory / "pynamit_input_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["projection_regularization"]["Br_lambda"] == 0.1
    for dataset in ("Br", "jr", "resistance", "E_source"):
        assert (projection_directory / f"{dataset}.ncdf").is_file()


def test_mage_projection_rejects_incompatible_prepared_units(tmp_path):
    """The forcing contract must make unit conversions explicit."""
    forcing_path = tmp_path / "forcing.h5"
    projection_directory = tmp_path / "projection"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as forcing:
        forcing["Bu"].attrs["units"] = "T"

    with pytest.raises(RuntimeError, match="incompatible dataset units.*Bu"):
        project_inputs(
            forcing_path=forcing_path,
            projection_directory=projection_directory,
            main_field_kind="dipole",
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=None,
            br_lambda=0.1,
            conductance_lambda=0.1,
            jr_lambda=0.1,
            e_source_lambda=0.1,
            artifact_storage="netcdf",
        )


def test_projection_failure_preserves_last_complete_package(tmp_path):
    """A failed projection must not replace reusable inputs."""
    forcing_path = tmp_path / "forcing.h5"
    projection_directory = tmp_path / "projection"
    projection_directory.mkdir()
    existing_input = projection_directory / "jr.ncdf"
    existing_input.write_text("existing", encoding="utf-8")
    hall = np.ones((2, 4, 4)) * 5.0
    hall[1] = -1.0
    _write_projection_forcing(forcing_path, hall_conductance=hall)

    with pytest.raises(ValueError, match="Hall conductance"):
        project_inputs(
            forcing_path=forcing_path,
            projection_directory=projection_directory,
            main_field_kind="dipole",
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=None,
            br_lambda=0.1,
            conductance_lambda=0.1,
            jr_lambda=0.1,
            e_source_lambda=0.1,
            artifact_storage="netcdf",
        )

    assert existing_input.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*-projecting-*"))


def test_projection_accepts_zero_hall_with_positive_pedersen(tmp_path):
    """Zero Hall conductance need not make the tensor singular."""
    forcing_path = tmp_path / "forcing.h5"
    projection_directory = tmp_path / "projection"
    _write_projection_forcing(forcing_path, hall_conductance=0.0)

    project_inputs(
        forcing_path=forcing_path,
        projection_directory=projection_directory,
        main_field_kind="dipole",
        dipole_B0_override=None,
        boundary_radius_override=None,
        nmax=2,
        mmax=1,
        ncs=4,
        max_steps=1,
        br_lambda=0.1,
        conductance_lambda=0.1,
        jr_lambda=0.1,
        e_source_lambda=0.1,
        artifact_storage="netcdf",
    )

    assert (projection_directory / "resistance.ncdf").is_file()


def test_integrate_tiegcm_step_computed_conductances_and_weighted_winds():
    """Computed outputs should match layer integrals."""
    dataset = _two_layer_tiegcm_dataset()

    integrated = _integrate_tiegcm_step(dataset, 0, conductance_source="computed")

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

    integrated = _integrate_tiegcm_step(dataset, 0, conductance_source="native")

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

    integrated = _integrate_tiegcm_step(dataset, 0, conductance_source="computed")

    for key in ("SP", "SH", "We", "Wn", "WeH", "WnH"):
        np.testing.assert_allclose(integrated[key], 0.0)
