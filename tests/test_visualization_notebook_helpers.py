"""Tests for helpers extracted from the plotting notebook."""

import datetime as dt

import numpy as np

import pynamit.coordinates as coordinates
import pynamit.visualization as visualization
from pynamit.math.constants import mu0
from pynamit.sphere import SHBasis
from pynamit.sphere import SolidHarmonics
from pynamit.coordinates import (
    datetime_to_utc_hours,
    local_noon_longitude,
    local_time_hours_to_longitude,
    local_time_longitude_to_geographic,
    longitude_to_local_time_from_noon_longitude,
    longitude_to_local_time_hours,
    wrap_longitude_180,
)
from pynamit.visualization.artifacts import (
    artifact_path,
    resolve_xarray_artifact_path,
    xarray_artifact_exists,
)
from pynamit.visualization.grid_evaluation import build_evaluator
from pynamit.visualization.grid_evaluation import build_plot_grid
from pynamit.visualization.grid_evaluation import build_sheet_current_operators
from pynamit.visualization.grid_evaluation import resistance_to_conductance
from pynamit.visualization.local_time import local_time_grid_longitudes
from pynamit.visualization.plot_helpers import (
    contour_kwargs_for_display,
    format_contour_interval,
    get_ticks_from_levels,
    symmetric_contour_levels_without_zero,
)


def test_local_time_longitude_helpers_are_vectorized():
    """Longitude helpers support scalar and vector inputs."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)

    assert wrap_longitude_180(190.0) == -170.0
    np.testing.assert_allclose(
        wrap_longitude_180(np.array([-190.0, 180.0, 540.0])),
        np.array([170.0, -180.0, -180.0]),
    )
    assert datetime_to_utc_hours(reference_time) == 18.5
    assert local_noon_longitude(reference_time) == -97.5
    np.testing.assert_allclose(
        longitude_to_local_time_hours(np.array([-97.5, -7.5]), reference_time),
        np.array([12.0, 18.0]),
    )
    np.testing.assert_allclose(
        local_time_hours_to_longitude(np.array([12.0, 18.0]), reference_time),
        np.array([-97.5, -7.5]),
    )
    np.testing.assert_allclose(
        longitude_to_local_time_from_noon_longitude(
            np.array([-97.5, -7.5]),
            local_noon_longitude(reference_time),
        ),
        longitude_to_local_time_hours(np.array([-97.5, -7.5]), reference_time),
    )


def test_noon_meridian_local_time_helper_preserves_plot_coordinate():
    """The shared helper covers globalplot's polar formula."""
    lon = np.array([-180.0, -100.0, 80.0, 180.0])
    noon_longitude = -100.0

    np.testing.assert_allclose(
        longitude_to_local_time_from_noon_longitude(
            lon,
            noon_longitude,
            wrap=False,
        ),
        (lon - noon_longitude + 180.0) / 15.0,
    )
    np.testing.assert_allclose(
        longitude_to_local_time_from_noon_longitude(lon, noon_longitude),
        ((lon - noon_longitude + 180.0) / 15.0) % 24.0,
    )


def test_local_time_grid_and_source_longitude_conversion():
    """Local-time grid helpers match the notebook convention."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)

    np.testing.assert_allclose(
        local_time_grid_longitudes(reference_time),
        np.array([127.5, -142.5, -52.5, 37.5]),
    )
    np.testing.assert_allclose(
        local_time_longitude_to_geographic(
            np.array([-180.0, 0.0, 90.0]),
            noon_longitude=-100.0,
            local_noon_longitude=0.0,
        ),
        np.array([80.0, -100.0, -10.0]),
    )


def test_artifact_path_resolution_prefers_zarr_then_netcdf(tmp_path):
    """Artifact resolver handles extension and base-path inputs."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "state.zarr").mkdir()
    (run_directory / "settings.ncdf").write_bytes(b"")

    assert artifact_path(run_directory, "state") == str(run_directory / "state")
    assert resolve_xarray_artifact_path(run_directory / "state") == str(
        run_directory / "state.zarr"
    )
    assert resolve_xarray_artifact_path(run_directory / "state.ncdf") == str(
        run_directory / "state.zarr"
    )
    assert resolve_xarray_artifact_path(run_directory / "settings") == str(
        run_directory / "settings.ncdf"
    )
    assert xarray_artifact_exists(run_directory / "settings")


def test_plot_helper_functions_match_notebook_behaviour():
    """Plot helper outputs are stable and metadata is filtered."""
    levels = symmetric_contour_levels_without_zero(10.0, 2.0)
    np.testing.assert_allclose(
        levels,
        np.array([-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0]),
    )
    np.testing.assert_allclose(
        get_ticks_from_levels({"levels": np.array([0.0, 2.0, 4.0])}),
        np.array([1.0, 3.0]),
    )
    assert format_contour_interval(0.001) == "1.00e-03"
    assert contour_kwargs_for_display(
        {"levels": levels, "symbol": "x", "units": "T", "scale": 1.0}
    ) == {"levels": levels}


def test_grid_and_conductance_helpers_are_importable_from_visualization():
    """Grid/evaluation helpers are available through the package."""
    lat, lon, grid = build_plot_grid(nlat=3, nlon=4)
    assert lat.shape == (3, 4)
    assert lon.shape == (3, 4)
    assert grid.size == 12

    sigmaP, sigmaH = resistance_to_conductance(
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(sigmaP[0], 0.4)
    np.testing.assert_allclose(sigmaH[0], 0.2)
    assert np.isnan(sigmaP[1])
    assert np.isnan(sigmaH[1])

    assert visualization.wrap_longitude_180 is wrap_longitude_180
    assert visualization.wrap_longitude_180 is coordinates.wrap_longitude_180
    assert visualization.build_plot_grid is build_plot_grid
    assert visualization.resistance_to_conductance is resistance_to_conductance
    assert (
        visualization.longitude_to_local_time_from_noon_longitude
        is longitude_to_local_time_from_noon_longitude
    )


def test_sheet_current_operator_bundle_matches_core_formulas():
    """Shared sheet-current helper follows geometry formulas."""

    class Settings:
        RI = 1.0
        RM = 2.0

    sh_basis = SHBasis(3, 2, mean_free=True)
    solid_harmonics = SolidHarmonics(sh_basis)
    _, _, grid = build_plot_grid(nlat=4, nlon=5)
    transform = build_evaluator(sh_basis, grid)
    t_to_ve = np.eye(sh_basis.index_length)

    operators = build_sheet_current_operators(
        Settings,
        sh_basis,
        transform,
        T_to_Ve=t_to_ve,
    )

    poloidal_to_sheet = (
        -transform.scalar_coeffs_to_gridded_rhat_cross_gradient
        * solid_harmonics.poloidal_to_boundary_potential_jump_factor.reshape(1, 1, -1)
        / mu0
    )
    toroidal_to_sheet = -transform.scalar_coeffs_to_gridded_gradient / mu0
    regular_shift = solid_harmonics.regular_reference_shift(Settings.RM, Settings.RI)
    irregular_shift = solid_harmonics.irregular_reference_shift(Settings.RI, Settings.RM)
    denominator = 1.0 - regular_shift * irregular_shift
    m_ind_to_br = -(Settings.RI**2) * sh_basis.laplacian(Settings.RI)

    np.testing.assert_allclose(
        operators["G_m_imp_to_JS"],
        toroidal_to_sheet + np.tensordot(poloidal_to_sheet, t_to_ve, axes=([2], [0])),
    )
    np.testing.assert_allclose(
        operators["G_m_ind_to_JS"],
        poloidal_to_sheet * (1.0 + regular_shift * irregular_shift / denominator),
    )
    np.testing.assert_allclose(
        operators["G_Br_to_JS"],
        poloidal_to_sheet * (-regular_shift / (denominator * m_ind_to_br)),
    )
