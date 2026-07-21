"""Tests for MAGE preparation, projection, and run conventions."""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

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
    _combine_remix_hemispheres,
    _create_output_datasets,
    _gamera_inner_boundary_geometry,
    _gamera_inner_boundary_solid_angle,
    _gamera_internal_dipole_axes,
    _geographic_grid_in_sm,
    _integrate_tiegcm_step,
    _interpolate_periodic_latlon_field,
    _naive_utc_datetime,
    _PeriodicLatLonInterpolator,
    _pynamit_dipole_B0_T,
    _read_tiegcm_step,
    _remix_upward_fac_source,
    _resolve_tiegcm_path,
    _tiegcm_step_in_geographic_coordinates,
    _tiegcm_times,
    _trilinear_hexahedron_volume_centers,
    _upward_fac_to_radial_current,
    _validate_settings,
    _validate_source_times,
    _write_static_datasets,
)
from scripts.simulation.mage_project import CASE_DIRECTORY as MAGE_PROJECT_CASE
from scripts.simulation.mage_project import DEFAULT_FORCING_PATH
from scripts.simulation.mage_project import SETTINGS as MAGE_PROJECT_SETTINGS
from scripts.simulation.mage_run import CASE_DIRECTORY as MAGE_RUN_CASE
from scripts.simulation.mage_run import DEFAULT_PROJECTION_DIRECTORY, _last_projected_input_time
from scripts.simulation.mage_run import SETTINGS as MAGE_RUN_SETTINGS

from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm
from pynamit.simulation.workflows.mage_projection import (
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    MAGE_MAIN_FIELD_KIND,
    _boundary_radius,
    _clear_existing_input_package,
    _dipole_B0,
    _gamera_dipole_metadata,
    _h5_time_vector_seconds,
    _load_weighted_winds,
    _source_file_metadata,
    _validate_prepared_forcing,
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
            "boundary_radius": np.full(latitude.shape, 7.0e6),
            "boundary_solid_angle": np.full(latitude.shape, 4.0 * np.pi / latitude.size),
            "ionosphere_lat": latitude,
            "ionosphere_lon": longitude,
            "boundary_lat": latitude,
            "boundary_lon": longitude,
            "delta_Br": np.full(step_shape, 10.0),
            "jr": np.full(step_shape, 0.1),
            "SH": hall,
            "SP": np.full(step_shape, 10.0),
            "u_p_theta": np.full(step_shape, 20.0),
            "u_p_phi": np.full(step_shape, 50.0),
            "u_h_theta": np.full(step_shape, -10.0),
            "u_h_phi": np.full(step_shape, 30.0),
        }.items():
            output.create_dataset(name, data=values)
        for name, units in {
            "boundary_radius": "m",
            "boundary_solid_angle": "sr",
            "ionosphere_lat": "degree",
            "ionosphere_lon": "degree",
            "boundary_lat": "degree",
            "boundary_lon": "degree",
            "delta_Br": "nT",
            "jr": "uA m-2",
            "SH": "S",
            "SP": "S",
            "u_p_theta": "m s-1",
            "u_p_phi": "m s-1",
            "u_h_theta": "m s-1",
            "u_h_phi": "m s-1",
        }.items():
            output[name].attrs["units"] = units
        output.attrs["gamera_mag_m0_nT"] = -30_000.0
        output.attrs["main_field_B0_T"] = 3.0e-5
        output.attrs["main_field_B0_reference_radius_m"] = 6_371_200.0
        output.attrs["gamera_internal_dipole_moment_axis"] = [0.0, 0.0, -1.0]
        output.attrs["gamera_internal_magnetic_north_axis"] = [0.0, 0.0, 1.0]
        output.attrs["gamera_source_coordinate_system"] = "SM"
        output.attrs["coordinate_system"] = "GEO"
        output.attrs["longitude_convention"] = "east_positive_degrees"
        output.attrs["fac_convention"] = "upward"
        output.attrs["radial_current_convention"] = "outward"
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


def test_gamera_dipole_strength_uses_pynamit_reference_radius():
    """MagM0 must retain its field after changing radius units."""
    gamera_radius = 6_378_100.0
    mag_m0_nT = -29_617.4

    B0 = _pynamit_dipole_B0_T(mag_m0_nT, gamera_radius)

    np.testing.assert_allclose(B0 * (6_371_200.0 / gamera_radius) ** 3, 29_617.4e-9)


def test_centered_dipole_alignment_uses_gamera_axis_convention():
    """Prepared alignment metadata carries the signed GAMERA axes."""
    attrs = _centered_dipole_alignment_attrs(dt.datetime(2020, 1, 1), -30_000.0)

    assert attrs["gamera_source_coordinate_system"] == "SM"
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
            np.ones((1, 1)),
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


def test_mage_run_defaults_to_steady_state_initialization_and_output():
    """MAGE starts from and records the steady-state response."""
    assert MAGE_RUN_SETTINGS.steady_state_initialization is True
    assert MAGE_RUN_SETTINGS.run_steady_state is True
    assert MAGE_RUN_SETTINGS.magnetic_boundary_shielding is False
    assert MAGE_RUN_SETTINGS.final_time is None


def test_mage_run_infers_final_time_from_projected_boundary_input():
    """An unedited run must stop at its last projected forcing."""
    store = SimpleNamespace(
        load_dataset=lambda key: SimpleNamespace(
            time=SimpleNamespace(values=np.array([0.0, 10.25, 20.5]))
        )
    )

    assert _last_projected_input_time(store) == 20.5


def test_mage_projection_uses_kaiju_dipole_by_default():
    """MAGE projection uses Kaiju dipole physics on a GEO model grid."""
    assert MAGE_MAIN_FIELD_KIND == "kaiju_dipole"


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


def test_remix_hemisphere_source_conventions_preserve_zero_longitude():
    """South preserves the zero cell when negating SM longitude."""

    class FakeRemix:
        ion = {
            "Field-aligned current NORTH": np.array([[1.0, 2.0, 3.0, 4.0]]),
            "Field-aligned current SOUTH": np.array([[5.0, 6.0, 7.0, 8.0]]),
        }

    latitude = np.full((1, 4), 70.0)
    longitude = np.array([[0.0, 90.0, 180.0, 270.0]])

    north = _remix_upward_fac_source(FakeRemix(), "NORTH", latitude, longitude)
    south = _remix_upward_fac_source(FakeRemix(), "SOUTH", latitude, longitude)

    np.testing.assert_array_equal(north[0], latitude)
    np.testing.assert_array_equal(north[1], [[0.0, 90.0, -180.0, -90.0]])
    np.testing.assert_array_equal(north[2], [[-1.0, -2.0, -3.0, -4.0]])
    np.testing.assert_array_equal(south[0], -latitude)
    np.testing.assert_array_equal(south[1], [[0.0, -90.0, -180.0, 90.0]])
    np.testing.assert_array_equal(south[2], [[5.0, 6.0, 7.0, 8.0]])


def test_projection_geometry_requires_prepared_radius_or_explicit_value():
    """RM must come from the prepared file or the edited settings."""
    with pytest.raises(RuntimeError, match="boundary_radius"):
        _boundary_radius(_FakeH5(), explicit_radius=None)

    assert _boundary_radius(_FakeH5(), explicit_radius=7.0) == 7.0
    with pytest.raises(ValueError, match="positive"):
        _boundary_radius(_FakeH5(), explicit_radius=0.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        _boundary_radius(_FakeH5(boundary_radius=np.array([7.0e6, np.nan])), explicit_radius=None)


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


def test_mage_preparation_normalizes_timezone_offsets_to_utc():
    """Preparation must convert offsets before discarding them."""
    source_time = dt.datetime(
        2011, 10, 24, 20, 0, 10, tzinfo=dt.timezone(dt.timedelta(hours=2))
    )

    assert _naive_utc_datetime(source_time) == dt.datetime(2011, 10, 24, 18, 0, 10)


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


def test_gamera_boundary_geometry_uses_selected_volume_cell():
    """The B[0] center belongs to the cell between shells 0 and 1."""

    class FakeGameraGrid:
        pass

    grid = FakeGameraGrid()
    grid.X = np.array([[[1.0, 1.0], [1.0, 1.0]], [[3.0, 3.0], [3.0, 3.0]]])
    grid.Y = np.array([[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]])
    grid.Z = np.array([[[-1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, 1.0]]])

    lat, lon, radius, sin_theta, cos_theta, sin_phi, cos_phi = _gamera_inner_boundary_geometry(
        grid, inner_index=0, length_scale_m=10.0
    )

    np.testing.assert_allclose(lat, 0.0, atol=1e-14)
    np.testing.assert_allclose(lon, 0.0, atol=1e-14)
    np.testing.assert_allclose(radius, 20.0)
    np.testing.assert_allclose(
        (sin_theta, cos_theta, sin_phi, cos_phi),
        (np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1))),
        atol=1e-14,
    )


def test_trilinear_volume_center_matches_wedge_centroid():
    """Volume quadrature must reproduce a non-affine cell centroid."""
    alpha = 1.0
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0 + alpha],
            [1.0, 1.0, 1.0 + alpha],
            [0.0, 1.0, 1.0],
        ]
    )
    volume = 1.0 + alpha / 2.0
    expected = np.array(
        [
            (0.5 + alpha / 3.0) / volume,
            0.5,
            0.5 * (1.0 + alpha + alpha**2 / 3.0) / volume,
        ]
    )

    center = _trilinear_hexahedron_volume_centers(vertices)

    np.testing.assert_allclose(center, expected, rtol=0.0, atol=1e-14)


def test_gamera_boundary_solid_angles_follow_true_cell_vertices():
    """Boundary weights use spherical cell areas, not latitude alone."""

    class FakeGameraGrid:
        pass

    latitude_edges = np.deg2rad(np.array([-90.0, -30.0, 30.0, 90.0]))
    longitude_edges = np.deg2rad(np.array([-180.0, -60.0, 60.0, 180.0]))
    longitude, latitude = np.meshgrid(longitude_edges, latitude_edges)
    radius = np.array([1.0, 2.0])[:, None, None]
    cos_latitude = np.cos(latitude)[None, ...]
    grid = FakeGameraGrid()
    grid.X = radius * cos_latitude * np.cos(longitude)[None, ...]
    grid.Y = radius * cos_latitude * np.sin(longitude)[None, ...]
    grid.Z = radius * np.sin(latitude)[None, ...]

    solid_angle = _gamera_inner_boundary_solid_angle(grid, inner_index=0)

    assert solid_angle.shape == (3, 3)
    np.testing.assert_allclose(np.sum(solid_angle), 4.0 * np.pi, rtol=0.0, atol=1e-12)
    assert np.all(solid_angle > 0.0)


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

    interpolated = _interpolate_periodic_latlon_field(
        source_lat, source_lon, values, target_lon, target_lat
    )

    np.testing.assert_allclose(interpolated, 2.0)


def test_remix_hemispheres_leave_zero_current_outside_source_coverage():
    """Uncovered low latitudes have no prescribed REMIX current."""
    south = np.array([1.0, np.nan, np.nan])
    north = np.array([np.nan, 2.0, np.nan])

    np.testing.assert_array_equal(
        _combine_remix_hemispheres(south, north), np.array([1.0, 2.0, 0.0])
    )


def test_periodic_grid_interpolator_reuses_geometry_without_changing_values():
    """Cached boundary interpolation preserves linear interpolation."""
    source_lat, source_lon = np.meshgrid(
        np.linspace(-80.0, 80.0, 9), np.linspace(-180.0, 150.0, 12), indexing="ij"
    )
    values = np.cos(np.deg2rad(source_lat)) * np.cos(np.deg2rad(source_lon))
    target_lon = source_lon + 7.5
    interpolator = _PeriodicLatLonInterpolator(source_lat, source_lon)

    observed = interpolator.interpolate(values, target_lon, source_lat, require_complete=True)
    expected = _interpolate_periodic_latlon_field(
        source_lat, source_lon, values, target_lon, source_lat, require_complete=True
    )

    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_periodic_latlon_interpolator_uses_spherical_nearest_fallback_lazily():
    """Build the polar great-circle fallback only when needed."""
    source_lat = np.array([88.0, 80.0, 80.0, 80.0])
    source_lon = np.array([180.0, 0.0, 120.0, -120.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    interpolator = _PeriodicLatLonInterpolator(source_lat, source_lon)

    interpolator.interpolate(values, np.array([0.0]), np.array([85.0]))
    assert interpolator._nearest_tree is None

    observed = interpolator.interpolate(
        values, np.array([0.0]), np.array([89.0]), require_complete=True
    )

    assert interpolator._nearest_tree is not None
    np.testing.assert_array_equal(observed, [1.0])


def test_tiegcm_history_stays_on_its_native_geographic_grid():
    """Preparation only changes east/north into spherical components."""
    source_lat = np.linspace(-89.0, 89.0, 45)
    source_lon = np.linspace(-180.0, 175.0, 72)
    lon_grid, lat_grid = np.meshgrid(source_lon, source_lat)
    scalar = 5.0 + np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
    integrated = {
        "SP": scalar,
        "SH": 2.0 * scalar,
        "We": np.full_like(scalar, 100.0),
        "Wn": np.zeros_like(scalar),
        "WeH": np.full_like(scalar, -40.0),
        "WnH": np.full_like(scalar, 30.0),
    }
    result = _tiegcm_step_in_geographic_coordinates(integrated)

    np.testing.assert_allclose(result["SP"], integrated["SP"], rtol=1e-7)
    np.testing.assert_allclose(result["SH"], integrated["SH"], rtol=1e-7)
    np.testing.assert_allclose(result["u_p_theta"], -integrated["Wn"])
    np.testing.assert_allclose(result["u_p_phi"], integrated["We"])
    np.testing.assert_allclose(result["u_h_theta"], -integrated["WnH"])
    np.testing.assert_allclose(result["u_h_phi"], integrated["WeH"])


def test_fixed_geo_grid_maps_through_each_timestamped_sm_frame():
    """Source transforms change without moving GEO coordinates."""
    first_time = dt.datetime(2011, 10, 24, 18, 0, 10)
    last_time = dt.datetime(2011, 10, 24, 19, 0, 0)
    geographic_latitude = np.array([65.0, -40.0, 5.0])
    geographic_longitude = np.array([-120.0, 15.0, 90.0])

    first_sm = _geographic_grid_in_sm(geographic_latitude, geographic_longitude, first_time)
    last_sm = _geographic_grid_in_sm(geographic_latitude, geographic_longitude, last_time)

    assert not np.allclose(first_sm[1], last_sm[1])
    for event_time, (sm_lat, sm_lon) in ((first_time, first_sm), (last_time, last_sm)):
        geo_lat, geo_lon = kaiju_geopack_sm(event_time).sm2geo(sm_lat, sm_lon)
        np.testing.assert_allclose(geo_lat, geographic_latitude, atol=1e-12)
        np.testing.assert_allclose(
            ((geo_lon - geographic_longitude + 180.0) % 360.0) - 180.0, 0.0, atol=1e-12
        )


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
        forcing["delta_Br"].attrs["units"] = "T"

    with pytest.raises(RuntimeError, match="incompatible dataset units.*delta_Br"):
        project_inputs(
            forcing_path=forcing_path,
            projection_directory=projection_directory,
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


def test_mage_projection_requires_pynamit_dipole_reference_radius(tmp_path):
    """Prepared B0 must declare the radius used by PynaMIT's dipole."""
    forcing_path = tmp_path / "forcing.h5"
    _write_projection_forcing(forcing_path)
    with h5py.File(forcing_path, "r+") as output:
        output.attrs["main_field_B0_reference_radius_m"] = 6_378_100.0

    with h5py.File(forcing_path) as forcing:
        with pytest.raises(RuntimeError, match="dipole reference radius"):
            _validate_prepared_forcing(forcing)


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
