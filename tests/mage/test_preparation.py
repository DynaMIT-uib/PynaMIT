"""Tests for preparing MAGE forcing data."""

import datetime as dt

import h5py
import numpy as np
import pytest
from tests.mage._support import _FakeDataset, _FakeVariable

from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm
from pynamit.workflows.mage.preparation import (
    CONDUCTANCE_FLOOR_MODEL,
    HALL_CONDUCTANCE_FLOOR_S,
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    PEDERSEN_CONDUCTANCE_FLOOR_S,
    ForcingSettings,
    _apply_conductance_floor,
    _atomic_prepared_output,
    _centered_dipole_alignment_attrs,
    _combine_remix_hemispheres,
    _datetime_from_mjd,
    _gamera_inner_boundary_geometry,
    _gamera_internal_dipole_axes,
    _gamera_native_angles,
    _GameraBoundaryInterpolator,
    _geographic_grid_in_sm,
    _integrate_tiegcm_step,
    _kaiju_sm_transform_time,
    _pynamit_dipole_B0_T,
    _read_tiegcm_step,
    _remix_cell_center_coordinates,
    _remix_upward_fac_source,
    _RemixGridInterpolator,
    _resolve_tiegcm_path,
    _tiegcm_times,
    _trilinear_hexahedron_volume_centers,
    _validate_forcing_time_axis,
    _validate_settings,
    _validate_tiegcm_variables,
    _write_static_datasets,
    _write_time_axis,
)
from pynamit.workflows.mage.projection import _h5_time_vector_seconds, prepare_inputs


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
    attrs = _centered_dipole_alignment_attrs(dt.datetime(2020, 1, 1, 0, 0, 0, 12_674), -30_000.0)

    assert attrs["gamera_source_coordinate_system"] == "SM"
    assert attrs["dipole_sm_transform_time"] == "2020-01-01T00:00:00"
    np.testing.assert_array_equal(attrs["gamera_internal_dipole_moment_axis"], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(attrs["gamera_internal_magnetic_north_axis"], [0.0, 0.0, 1.0])
    assert "gamera_internal_dipole_axis" not in attrs
    assert "pynamit_simulation_coordinate_system" not in attrs


@pytest.mark.parametrize(
    ("source_time", "expected"),
    [
        (dt.datetime(2011, 10, 24, 18, 0, 10, 499_999), dt.datetime(2011, 10, 24, 18, 0, 10)),
        (dt.datetime(2011, 10, 24, 18, 0, 10, 500_000), dt.datetime(2011, 10, 24, 18, 0, 11)),
        (dt.datetime(2011, 10, 24, 18, 0, 59, 900_000), dt.datetime(2011, 10, 24, 18, 1, 0)),
    ],
)
def test_kaiju_sm_transform_time_matches_fortran_nint(source_time, expected):
    """SM transformations reproduce Kaiju's whole-second MJD wrapper."""
    assert _kaiju_sm_transform_time(source_time) == expected


def test_geographic_grid_in_sm_uses_kaiju_nearest_second():
    """Fractional source seconds do not leak into the Kaiju SM frame."""
    source_time = dt.datetime(2011, 10, 24, 18, 0, 10, 33_088)
    latitude = np.array([20.0, 65.0])
    longitude = np.array([-40.0, 130.0])

    observed = _geographic_grid_in_sm(latitude, longitude, source_time)
    expected = kaiju_geopack_sm(dt.datetime(2011, 10, 24, 18, 0, 10)).geo2sm(latitude, longitude)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-13)


def test_hard_conductance_floor_applies_globally():
    """Give the complete PynaMIT sheet its background minimum."""
    pedersen = np.full((4, 4), 0.25)
    hall = np.full((4, 4), 0.1)
    pedersen[0, 0] = 4.0
    hall[-1, -1] = 3.0

    floored_pedersen, floored_hall = _apply_conductance_floor(pedersen, hall)

    np.testing.assert_array_equal(
        floored_pedersen, np.maximum(pedersen, PEDERSEN_CONDUCTANCE_FLOOR_S)
    )
    np.testing.assert_array_equal(floored_hall, np.maximum(hall, HALL_CONDUCTANCE_FLOOR_S))


def _two_layer_tiegcm_dataset():
    """Return a tiny TIEGCM-like dataset with cm inputs."""
    height_cm = np.array(
        [
            [
                [10_000_000.0, 10_200_000.0],
                [11_000_000.0, 11_400_000.0],
                [13_000_000.0, 13_500_000.0],
            ]
        ]
    )
    return _FakeDataset(
        SIGMA_PED=np.array([[[1.0, 2.0], [3.0, 0.5], [99.0, 99.0]]]),
        SIGMA_HAL=np.array([[[4.0, 1.0], [1.0, 2.0], [99.0, 99.0]]]),
        Z=height_cm,
        ZG=height_cm,
        UN=np.array([[[1000.0, -2000.0], [3000.0, 4000.0], [999.0, 999.0]]]),
        VN=np.array([[[500.0, 2000.0], [100.0, -1000.0], [1.0e31, 1.0e31]]]),
        ilev=np.array([-7.0, -6.75, -6.5]),
    )


def _valid_tiegcm_contract():
    """Return a minimal global TIEGCM variable contract."""
    horizontal = (4, 4)
    layers = (1, 3, *horizontal)
    common = np.ones(layers)
    return _FakeDataset(
        lon=_FakeVariable([-135.0, -45.0, 45.0, 135.0], dimensions=("lon",), units="degrees_east"),
        lat=_FakeVariable([-67.5, -22.5, 22.5, 67.5], dimensions=("lat",), units="degrees_north"),
        lev=_FakeVariable([-6.875, -6.625, -6.375], dimensions=("lev",)),
        ilev=_FakeVariable([-7.0, -6.75, -6.5], dimensions=("ilev",)),
        mtime=_FakeVariable([[1, 0, 0, 0]], dimensions=("time", "mtimedim")),
        year=_FakeVariable([2020], dimensions=("time",)),
        SIGMA_PED=_FakeVariable(common, dimensions=("time", "lev", "lat", "lon"), units="S/m"),
        SIGMA_HAL=_FakeVariable(common, dimensions=("time", "lev", "lat", "lon"), units="S/m"),
        Z=_FakeVariable(common, dimensions=("time", "ilev", "lat", "lon"), units="cm"),
        ZG=_FakeVariable(common, dimensions=("time", "ilev", "lat", "lon"), units="cm"),
        UN=_FakeVariable(common, dimensions=("time", "lev", "lat", "lon"), units="cm/s"),
        VN=_FakeVariable(common, dimensions=("time", "lev", "lat", "lon"), units="cm/s"),
    )


def test_tiegcm_contract_requires_geo_units_and_midpoint_layers():
    """Validate the geographic and vertical-grid assumptions."""
    dataset = _valid_tiegcm_contract()

    _validate_tiegcm_variables(dataset, n_steps=1)

    dataset.variables["UN"].units = "m/s"
    with pytest.raises(RuntimeError, match="incompatible units.*UN"):
        _validate_tiegcm_variables(dataset, n_steps=1)

    dataset.variables["UN"].units = "cm/s"
    dataset.variables["lev"].values[1] += 0.1
    with pytest.raises(RuntimeError, match="centered between"):
        _validate_tiegcm_variables(dataset, n_steps=1)


def test_ambiguous_tiegcm_discovery_requires_an_explicit_path(tmp_path):
    """Preparation must reject ambiguous forcing discovery."""
    for name in ("first_sech_tie.nc", "second_sech_tie.nc"):
        (tmp_path / name).touch()

    with pytest.raises(RuntimeError, match="multiple TIEGCM"):
        _resolve_tiegcm_path(tmp_path, explicit_path=None)


def test_prepared_time_axis_is_written_as_utf8_with_source_provenance(tmp_path):
    """Prepared time retains exact source-time provenance."""
    nominal_times = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    gamera_times = [
        dt.datetime(2011, 10, 24, 18, 0, 10, 12_674),
        dt.datetime(2011, 10, 24, 18, 0, 20, 16_929),
    ]
    remix_times = [value + dt.timedelta(microseconds=400) for value in gamera_times]
    output_path = tmp_path / "prepared.h5"
    grid = np.zeros((1, 1))

    with h5py.File(output_path, "w") as output:
        _write_time_axis(output, nominal_times, gamera_times, remix_times)
        _write_static_datasets(
            output,
            gamera_times[0],
            grid,
            grid,
            grid,
            grid,
            grid,
            np.ones((1, 1)),
            ForcingSettings(gamera_directory=tmp_path, output_path=output_path),
            tmp_path,
            6.3781e6,
            -29_617.4,
            tmp_path / "tiegcm.nc",
            35.0,
        )

    with h5py.File(output_path) as output:
        assert output["time"].asstr()[:].tolist() == ["2011-10-24T18:00:10", "2011-10-24T18:00:20"]
        assert output["gamera_source_time"].asstr()[:].tolist() == [
            "2011-10-24T18:00:10.012674",
            "2011-10-24T18:00:20.016929",
        ]
        np.testing.assert_allclose(output["gamera_time_offset_seconds"][:], [0.012674, 0.016929])
        np.testing.assert_allclose(output["remix_time_offset_seconds"][:], [0.013074, 0.017329])
        assert output.attrs["time_axis"] == "tiegcm_mtime_nominal"
        assert output.attrs["source_time_tolerance_seconds"] == 0.1
        assert output.attrs["kind"] == MAGE_FORCING_KIND
        assert output.attrs["version"] == MAGE_FORCING_VERSION
        assert output.attrs["remix_fac_interpolation"] == "kaiju_native_periodic"
        assert (
            output.attrs["gamera_boundary_interpolation"]
            == "gamera_native_periodic_bilinear_with_polar_mean"
        )
        assert (
            output.attrs["gamera_sm_transform_time_convention"] == "kaiju_mjdrecalc_nearest_second"
        )
        assert (
            output.attrs["tiegcm_conductance_integration"]
            == "radial_geometric_height_with_lower_dynamo_extension"
        )
        assert output.attrs["tiegcm_dynamo_bottom_ilev"] == -8.5
        assert output.attrs["tiegcm_dynamo_reference_height_m"] == 90_000.0
        assert output.attrs["conductance_floor_model"] == CONDUCTANCE_FLOOR_MODEL
        assert output.attrs["pedersen_conductance_floor_S"] == PEDERSEN_CONDUCTANCE_FLOOR_S
        assert output.attrs["hall_conductance_floor_S"] == HALL_CONDUCTANCE_FLOOR_S
        assert output.attrs["remix_grid_equatorward_sm_latitude_deg"] == 35.0
        assert not output.attrs["complete"]
        _, relative_seconds = _h5_time_vector_seconds(output["time"][:])

    np.testing.assert_array_equal(relative_seconds, [0.0, 10.0])


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


def test_remix_hemisphere_source_conventions_preserve_zero_longitude():
    """South preserves the zero cell when negating SM longitude."""

    class FakeRemix:
        ion = {
            "Field-aligned current NORTH": np.array([[1.0, 2.0, 3.0, 4.0]]),
            "Field-aligned current SOUTH": np.array([[5.0, 6.0, 7.0, 8.0]]),
        }

    latitude = np.full((1, 4), 70.0)
    longitude = np.array([[0.0, 90.0, 180.0, 270.0]])

    north = _remix_upward_fac_source(
        "NORTH", FakeRemix.ion["Field-aligned current NORTH"], latitude, longitude
    )
    south = _remix_upward_fac_source(
        "SOUTH", FakeRemix.ion["Field-aligned current SOUTH"], latitude, longitude
    )

    np.testing.assert_array_equal(north[0], latitude)
    np.testing.assert_array_equal(north[1], [[0.0, 90.0, -180.0, -90.0]])
    np.testing.assert_array_equal(north[2], [[-1.0, -2.0, -3.0, -4.0]])
    np.testing.assert_array_equal(south[0], -latitude)
    np.testing.assert_array_equal(south[1], [[0.0, -90.0, -180.0, 90.0]])
    np.testing.assert_array_equal(south[2], [[5.0, 6.0, 7.0, 8.0]])


def test_remix_cell_centers_match_kaipy_cartesian_average_convention():
    """Saved ReMIX fields live at Cartesian averages of X/Y corners."""
    x = np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])
    y = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

    latitude, longitude = _remix_cell_center_coordinates(x, y)

    x_center = np.array([[0.05, 0.15], [0.05, 0.15]])
    y_center = np.array([[0.05, 0.05], [0.15, 0.15]])
    np.testing.assert_allclose(
        latitude, 90.0 - np.degrees(np.arcsin(np.hypot(x_center, y_center)))
    )
    np.testing.assert_allclose(longitude, np.degrees(np.arctan2(y_center, x_center)))


def test_mage_preparation_preserves_mjd_subsecond_precision():
    """Source MJD conversion should retain microsecond-scale timing."""
    expected = dt.datetime(2011, 10, 24, 18, 0, 10, 12_674)
    mjd = (expected - dt.datetime(1858, 11, 17)).total_seconds() / 86_400.0

    actual = _datetime_from_mjd(mjd)

    assert abs((actual - expected).total_seconds()) <= 1e-6


def test_mage_preparation_canonicalizes_realistic_cfl_output_jitter():
    """Small MAGE output overshoots keep exact provenance."""
    nominal_times = [dt.datetime(2011, 10, 24, 18, 0, second) for second in (10, 20, 30, 40)]
    gamera_offsets = (0.012674, 0.016929, 0.015364, 0.023720)
    gamera_times = [
        value + dt.timedelta(seconds=offset)
        for value, offset in zip(nominal_times, gamera_offsets, strict=True)
    ]
    remix_times = [value + dt.timedelta(microseconds=400) for value in gamera_times]

    stored_gamera_offsets, stored_remix_offsets = _validate_forcing_time_axis(
        nominal_times, gamera_times, remix_times
    )

    np.testing.assert_allclose(stored_gamera_offsets, gamera_offsets)
    np.testing.assert_allclose(stored_remix_offsets, np.asarray(gamera_offsets) + 0.0004)
    nominal_seconds = np.array(
        [(value - nominal_times[0]).total_seconds() for value in nominal_times]
    )
    np.testing.assert_array_equal(nominal_seconds, [0.0, 10.0, 20.0, 30.0])


def test_mage_preparation_rejects_misaligned_source_times():
    """Every source history must match one nominal step."""
    nominal_times = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    gamera_times = nominal_times.copy()
    remix_times = [nominal_times[0], nominal_times[1] + dt.timedelta(seconds=1.0)]

    with pytest.raises(RuntimeError, match="ReMIX is not aligned with the nominal"):
        _validate_forcing_time_axis(nominal_times, gamera_times, remix_times)


def test_mage_preparation_rejects_nonuniform_nominal_time_axis():
    """Fixed-step simulations require a uniform nominal schedule."""
    nominal_times = [
        dt.datetime(2011, 10, 24, 18, 0, 10),
        dt.datetime(2011, 10, 24, 18, 0, 20),
        dt.datetime(2011, 10, 24, 18, 0, 31),
    ]

    with pytest.raises(RuntimeError, match="nominal TIEGCM.*uniform cadence"):
        _validate_forcing_time_axis(nominal_times, nominal_times, nominal_times)


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

    times = _tiegcm_times(dataset, reference)

    assert times == reference


def test_tiegcm_three_component_mtime_uses_minute_precision():
    """Standard mtime triplets use their documented minute precision."""
    reference = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 0, 20)]
    dataset = _FakeDataset(
        mtime=_FakeVariable([[297, 18, 0], [297, 18, 0]], dimensions=("time", "mtimedim")),
        year=_FakeVariable([2011, 2011], dimensions=("time",)),
    )

    times = _tiegcm_times(dataset, reference)

    assert times == [dt.datetime(2011, 10, 24, 18, 0)] * 2


def test_tiegcm_mtime_uses_named_component_axis():
    """The mtimedim name determines axis order."""
    reference = [dt.datetime(2011, 10, 24, 18, 0, 10), dt.datetime(2011, 10, 24, 18, 1, 10)]
    dataset = _FakeDataset(
        mtime=_FakeVariable(
            [[297, 297], [18, 18], [0, 1], [10, 10]], dimensions=("mtimedim", "time")
        ),
        year=_FakeVariable([2011, 2011], dimensions=("time",)),
    )

    times = _tiegcm_times(dataset, reference)

    assert times == reference


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
        _validate_settings(
            ForcingSettings(
                gamera_directory=tmp_path, output_path=tmp_path / "forcing.h5", max_steps=max_steps
            )
        )

    with pytest.raises(ValueError, match="positive integer"):
        prepare_inputs(
            forcing_path=tmp_path / "missing.h5",
            input_directory=tmp_path / "projection",
            dipole_B0_override=None,
            boundary_radius_override=None,
            nmax=2,
            mmax=1,
            ncs=4,
            max_steps=max_steps,
            boundary_Br_lambda=0.1,
            conductance_lambda=0.1,
            boundary_jr_lambda=0.1,
            e_neutral_wind_lambda=0.1,
            artifact_storage="netcdf",
        )


@pytest.mark.parametrize("inner_index", [True, 0.5, -1])
def test_mage_inner_index_requires_a_nonnegative_integer(tmp_path, inner_index):
    """Reject values that cannot index a GAMERA shell."""
    with pytest.raises(ValueError, match="inner_index"):
        _validate_settings(
            ForcingSettings(
                gamera_directory=tmp_path,
                output_path=tmp_path / "forcing.h5",
                inner_index=inner_index,
            )
        )


def test_gamera_boundary_geometry_uses_selected_volume_cell():
    """The B[0] center belongs to the cell between shells 0 and 1."""

    class FakeGameraGrid:
        pass

    grid = FakeGameraGrid()
    grid.X = np.array([[[1.0, 1.0], [1.0, 1.0]], [[3.0, 3.0], [3.0, 3.0]]])
    grid.Y = np.array([[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]])
    grid.Z = np.array([[[-1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, 1.0]]])

    geometry = _gamera_inner_boundary_geometry(grid, inner_index=0, length_scale_m=10.0)

    np.testing.assert_allclose(geometry.sm_latitude, 0.0, atol=1e-14)
    np.testing.assert_allclose(geometry.sm_longitude, 0.0, atol=1e-14)
    np.testing.assert_allclose(geometry.radius_m, 20.0)
    np.testing.assert_allclose(
        (geometry.radial_unit_x, geometry.radial_unit_y, geometry.radial_unit_z),
        (np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        geometry.radial_component(
            np.full((1, 1), 2.0), np.full((1, 1), 3.0), np.full((1, 1), 4.0)
        ),
        2.0,
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
        [(0.5 + alpha / 3.0) / volume, 0.5, 0.5 * (1.0 + alpha + alpha**2 / 3.0) / volume]
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

    solid_angle = _gamera_inner_boundary_geometry(
        grid, inner_index=0, length_scale_m=1.0
    ).solid_angle

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


def _sm_from_gamera_angles(colatitude, azimuth):
    """Return SM latitude/longitude for GAMERA-native angles."""
    colatitude, azimuth = np.broadcast_arrays(
        np.asarray(colatitude, dtype=float), np.asarray(azimuth, dtype=float)
    )
    x = np.cos(colatitude)
    y = np.sin(colatitude) * np.cos(azimuth)
    z = np.sin(colatitude) * np.sin(azimuth)
    return np.rad2deg(np.arcsin(z)), np.rad2deg(np.arctan2(y, x))


def test_gamera_native_angles_follow_kaiju_axis_convention():
    """GAMERA uses +x as its pole and measures azimuth from +y."""
    sm_latitude = np.array([0.0, 0.0, 90.0])
    sm_longitude = np.array([0.0, 90.0, 0.0])

    colatitude, azimuth = _gamera_native_angles(sm_latitude, sm_longitude)

    np.testing.assert_allclose(colatitude, np.deg2rad([0.0, 90.0, 90.0]), atol=1e-15)
    np.testing.assert_allclose(azimuth, np.deg2rad([0.0, 0.0, 90.0]), atol=1e-15)


def test_remix_hemispheres_leave_zero_current_outside_source_coverage():
    """Uncovered low latitudes have no prescribed REMIX current."""
    south = np.array([1.0, np.nan, np.nan])
    north = np.array([np.nan, 2.0, np.nan])

    np.testing.assert_array_equal(
        _combine_remix_hemispheres(south, north), np.array([1.0, 2.0, 0.0])
    )


def test_gamera_boundary_interpolator_preserves_source_nodes():
    """Kaiju-style four-point weights preserve GAMERA cell values."""
    colatitude, azimuth = np.meshgrid(
        np.deg2rad([30.0, 90.0, 150.0]), np.deg2rad([45.0, 135.0, 225.0, 315.0]), indexing="ij"
    )
    source_latitude, source_longitude = _sm_from_gamera_angles(colatitude, azimuth)
    values = np.arange(colatitude.size, dtype=float).reshape(colatitude.shape)
    interpolator = _GameraBoundaryInterpolator(source_latitude, source_longitude)

    observed = interpolator.interpolate(
        values, target_sm_lat=source_latitude, target_sm_lon=source_longitude
    )

    np.testing.assert_allclose(observed, values, atol=1e-12)


def test_gamera_boundary_interpolator_is_periodic_and_bilinear():
    """Interpolate GAMERA cells periodically with four weights."""
    colatitude, azimuth = np.meshgrid(
        np.deg2rad([30.0, 90.0, 150.0]), np.deg2rad([45.0, 135.0, 225.0, 315.0]), indexing="ij"
    )
    source_latitude, source_longitude = _sm_from_gamera_angles(colatitude, azimuth)
    values = np.rad2deg(colatitude) + 2.0 * np.rad2deg(azimuth)
    interpolator = _GameraBoundaryInterpolator(source_latitude, source_longitude)
    target_colatitude = np.deg2rad([60.0, 60.0, 60.0])
    target_azimuth = np.deg2rad([90.0, 450.0, 0.0])
    target_latitude, target_longitude = _sm_from_gamera_angles(target_colatitude, target_azimuth)

    observed = interpolator.interpolate(
        values, target_sm_lat=target_latitude, target_sm_lon=target_longitude
    )

    np.testing.assert_allclose(observed, [240.0, 240.0, 420.0], atol=1e-12)


def test_gamera_boundary_interpolator_reconstructs_native_poles():
    """Use adjacent ring means at the two omitted GAMERA axes."""
    colatitude, azimuth = np.meshgrid(
        np.deg2rad([30.0, 90.0, 150.0]), np.deg2rad([45.0, 135.0, 225.0, 315.0]), indexing="ij"
    )
    source_latitude, source_longitude = _sm_from_gamera_angles(colatitude, azimuth)
    values = np.array([[1.0, 3.0, 5.0, 7.0], [9.0, 9.0, 9.0, 9.0], [2.0, 6.0, 10.0, 14.0]])
    interpolator = _GameraBoundaryInterpolator(source_latitude, source_longitude)
    target_latitude, target_longitude = _sm_from_gamera_angles(
        np.array([0.0, np.pi]), np.deg2rad([20.0, 200.0])
    )

    observed = interpolator.interpolate(
        values, target_sm_lat=target_latitude, target_sm_lon=target_longitude
    )

    np.testing.assert_allclose(observed, [4.0, 8.0], atol=1e-12)


def test_remix_grid_interpolator_is_periodic_and_bilinear():
    """ReMIX interpolation follows its native four-point tensor grid."""
    latitude = np.array([60.0, 80.0])
    longitude = np.array([0.0, 90.0, 180.0, 270.0])
    source_lat, source_lon = np.meshgrid(latitude, longitude, indexing="ij")
    values = np.array([[0.0, 10.0, 20.0, 30.0], [100.0, 110.0, 120.0, 130.0]])
    interpolator = _RemixGridInterpolator(source_lat, source_lon)

    observed = interpolator.interpolate(
        values, target_lon=np.array([45.0, 405.0, 315.0]), target_lat=np.array([70.0, 70.0, 70.0])
    )

    np.testing.assert_allclose(observed, [55.0, 55.0, 65.0])


def test_remix_grid_interpolator_handles_pole_and_coverage():
    """Apply Kaiju's polar triangle and equatorward coverage."""
    latitude = np.broadcast_to(np.array([[60.0], [80.0]]), (2, 4))
    longitude = np.broadcast_to(np.array([[0.0, 90.0, 180.0, 270.0]]), (2, 4))
    values = np.array([[1.0, 3.0, 5.0, 7.0], [10.0, 14.0, 22.0, 30.0]])
    interpolator = _RemixGridInterpolator(latitude, longitude)

    observed = interpolator.interpolate(
        values, target_lon=np.array([20.0, 110.0, 0.0]), target_lat=np.array([89.0, 90.0, 50.0])
    )

    pole_value = np.mean(values[-1])
    np.testing.assert_allclose(
        observed[:2], [(pole_value + 10.0 + 14.0) / 3.0, (pole_value + 14.0 + 22.0) / 3.0]
    )
    assert np.isnan(observed[2])


def test_remix_grid_interpolator_accepts_southern_orientation():
    """Preserve values on Kaiju's reversed southern coordinates."""
    latitude = np.array([[-80.0, -80.0, -80.0], [-60.0, -60.0, -60.0]])
    longitude = np.array([[0.0, -120.0, -240.0], [0.0, -120.0, -240.0]])
    values = np.array([[8.0, 4.0, 6.0], [18.0, 14.0, 16.0]])
    interpolator = _RemixGridInterpolator(latitude, longitude)

    np.testing.assert_allclose(
        interpolator.interpolate(values, target_lon=120.0, target_lat=-70.0), 11.0
    )


def test_remix_grid_interpolator_rejects_nonrectilinear_coordinates():
    """Reject malformed coordinates instead of using tensor weights."""
    latitude = np.array([[60.0, 61.0], [80.0, 80.0]])
    longitude = np.array([[0.0, 180.0], [0.0, 180.0]])

    with pytest.raises(ValueError, match="rectilinear"):
        _RemixGridInterpolator(latitude, longitude)


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


def test_integrate_tiegcm_step_computed_conductances_and_weighted_winds():
    """Computed outputs include TIEGCM's lower dynamo extension."""
    dataset = _two_layer_tiegcm_dataset()

    integrated = _integrate_tiegcm_step(dataset, 0)

    interface_height = np.array(
        [[100_000.0, 102_000.0], [110_000.0, 114_000.0], [130_000.0, 135_000.0]]
    )
    dz = np.diff(interface_height, axis=0)
    sigma_p = np.array([[1.0, 2.0], [3.0, 0.5]])
    sigma_h = np.array([[4.0, 1.0], [1.0, 2.0]])
    east = np.array([[10.0, -20.0], [30.0, 40.0]])
    north = np.array([[5.0, 20.0], [1.0, -10.0]])
    lower_interfaces = 90_000.0 + np.linspace(0.0, 1.0, 7)[:, None] * (
        interface_height[0] - 90_000.0
    )
    lower_midpoints = 0.5 * (lower_interfaces[:-1] + lower_interfaces[1:])
    first_saved_midpoint = 0.5 * (interface_height[0] + interface_height[1])
    lower_dz = np.diff(lower_interfaces, axis=0)
    lower_p = np.sum(
        sigma_p[0] * np.exp((lower_midpoints - first_saved_midpoint) / 5_000.0) * lower_dz, axis=0
    )
    lower_h = np.sum(
        sigma_h[0] * np.exp((lower_midpoints - first_saved_midpoint) / 3_000.0) * lower_dz, axis=0
    )
    sp = np.sum(sigma_p * dz, axis=0) + lower_p
    sh = np.sum(sigma_h * dz, axis=0) + lower_h

    np.testing.assert_allclose(integrated["SP"], sp)
    np.testing.assert_allclose(integrated["SH"], sh)
    np.testing.assert_allclose(
        integrated["u_p_phi"], (np.sum(sigma_p * east * dz, axis=0) + lower_p * east[0]) / sp
    )
    np.testing.assert_allclose(
        integrated["u_p_theta"], -(np.sum(sigma_p * north * dz, axis=0) + lower_p * north[0]) / sp
    )
    np.testing.assert_allclose(
        integrated["u_h_phi"], (np.sum(sigma_h * east * dz, axis=0) + lower_h * east[0]) / sh
    )
    np.testing.assert_allclose(
        integrated["u_h_theta"], -(np.sum(sigma_h * north * dz, axis=0) + lower_h * north[0]) / sh
    )


def test_integrate_tiegcm_step_rejects_missing_values_inside_integrated_layers():
    """Preparation must reject missing integrated wind values."""
    dataset = _two_layer_tiegcm_dataset()
    dataset.variables["VN"].values[0, 1, 0] = 1.0e31

    with pytest.raises(RuntimeError, match="missing or non-finite.*VN"):
        _integrate_tiegcm_step(dataset, 0)


def test_integrate_tiegcm_step_zero_conductance_returns_zero_weighted_winds():
    """Zero conductance should produce zero winds instead of NaNs."""
    dataset = _FakeDataset(
        SIGMA_PED=np.zeros((1, 3, 2)),
        SIGMA_HAL=np.zeros((1, 3, 2)),
        Z=np.array(
            [
                [
                    [10_000_000.0, 10_200_000.0],
                    [11_000_000.0, 11_400_000.0],
                    [13_000_000.0, 13_500_000.0],
                ]
            ]
        ),
        ZG=np.array(
            [
                [
                    [10_000_000.0, 10_200_000.0],
                    [11_000_000.0, 11_400_000.0],
                    [13_000_000.0, 13_500_000.0],
                ]
            ]
        ),
        UN=np.full((1, 3, 2), 1000.0),
        VN=np.full((1, 3, 2), -2000.0),
        ilev=np.array([-7.0, -6.75, -6.5]),
    )

    integrated = _integrate_tiegcm_step(dataset, 0)

    for key in ("SP", "SH", "u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi"):
        np.testing.assert_allclose(integrated[key], 0.0)
