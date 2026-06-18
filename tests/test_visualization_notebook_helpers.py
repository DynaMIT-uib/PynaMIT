"""Tests for helpers extracted from the plotting notebook."""

import datetime as dt

import numpy as np

import pynamit.coordinates as coordinates
import pynamit.visualization as visualization
from pynamit.math.constants import RE, mu0
from pynamit.simulation.dynamics import Dynamics
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
from pynamit.visualization.grid_evaluation import build_JS_operators
from pynamit.visualization.grid_evaluation import resistance_to_conductance
from pynamit.visualization.hemisphere import (
    coerce_hemisphere_min_abs_latitude,
    hemisphere_masks_for_latitude,
)
from pynamit.visualization.local_time import local_time_grid_longitudes
from pynamit.visualization.map_curves import (
    build_even_global_sites,
    build_timeseries_curve_layers,
    geographic_local_time_mask,
    local_time_window_extent,
    split_wrapped_curve,
    wrap_longitudes,
)
from pynamit.visualization.plot_helpers import (
    build_percentile_color_scale,
    contour_kwargs_for_display,
    format_contour_interval,
    get_ticks_from_levels,
    style_global_axis,
    symmetric_contour_levels_without_zero,
)


def test_local_time_longitude_helpers_are_vectorized():
    """Longitude helpers support scalar and vector inputs."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)

    assert wrap_longitude_180(190.0) == -170.0
    np.testing.assert_allclose(
        wrap_longitude_180(np.array([-190.0, 180.0, 540.0])), np.array([170.0, -180.0, -180.0])
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
            np.array([-97.5, -7.5]), local_noon_longitude(reference_time)
        ),
        longitude_to_local_time_hours(np.array([-97.5, -7.5]), reference_time),
    )


def test_noon_meridian_local_time_helper_preserves_plot_coordinate():
    """The shared helper covers the polar local-time formula."""
    lon = np.array([-180.0, -100.0, 80.0, 180.0])
    noon_longitude = -100.0

    np.testing.assert_allclose(
        longitude_to_local_time_from_noon_longitude(lon, noon_longitude, wrap=False),
        (lon - noon_longitude + 180.0) / 15.0,
    )
    np.testing.assert_allclose(
        longitude_to_local_time_from_noon_longitude(lon, noon_longitude),
        ((lon - noon_longitude + 180.0) / 15.0) % 24.0,
    )


def test_hemisphere_helpers_are_parameterized_notebook_primitives():
    """Hemisphere masks should use an explicit configurable cutoff."""
    latitudes = np.array([-80.0, -40.0, 0.0, 45.0, 70.0])

    assert coerce_hemisphere_min_abs_latitude("bad", default=40.0) == 40.0
    assert coerce_hemisphere_min_abs_latitude(95.0) == 89.9
    north, south = hemisphere_masks_for_latitude(latitudes, min_abs_latitude=40.0)

    np.testing.assert_array_equal(north, np.array([False, False, False, True, True]))
    np.testing.assert_array_equal(south, np.array([True, False, False, False, False]))


def test_local_time_grid_and_source_longitude_conversion():
    """Local-time grid helpers match the notebook convention."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)

    np.testing.assert_allclose(
        local_time_grid_longitudes(reference_time), np.array([127.5, -142.5, -52.5, 37.5])
    )
    np.testing.assert_allclose(
        local_time_longitude_to_geographic(
            np.array([-180.0, 0.0, 90.0]), noon_longitude=-100.0, local_noon_longitude=0.0
        ),
        np.array([80.0, -100.0, -10.0]),
    )


def test_map_curve_sampling_matches_notebook_distribution():
    """Global-site sampling keeps the notebook row-count rules."""
    lon, lat = build_even_global_sites(
        min_lat=-60.0, max_lat=60.0, lat_step=60.0, equatorial_count=12, min_sites_per_row=3
    )

    assert lat.size == 24
    np.testing.assert_array_equal(np.unique(lat), np.array([-60.0, 0.0, 60.0]))
    assert np.count_nonzero(lat == 0.0) == 12
    assert np.count_nonzero(lat == -60.0) == 6
    assert np.count_nonzero(lat == 60.0) == 6
    assert np.all(lon >= -180.0)
    assert np.all(lon < 180.0)


def test_map_curve_sampling_can_use_local_time_rows():
    """Local-time sampling should align rows around local noon."""
    reference_time = dt.datetime(2011, 10, 24, 18, 0)

    lon, lat = build_even_global_sites(
        min_lat=0.0,
        max_lat=0.0,
        lat_count=1,
        equatorial_count=4,
        min_sites_per_row=1,
        reference_time=reference_time,
    )

    np.testing.assert_array_equal(lat, np.zeros(4))
    np.testing.assert_allclose(lon, np.array([90.0, -180.0, -90.0, 0.0]))


def test_map_curve_wrapping_and_segmentation_are_reusable():
    """Wrapped curves should split across dateline jumps and NaNs."""
    np.testing.assert_allclose(
        wrap_longitudes(np.array([-200.0, -170.0, 190.0]), central_longitude=0.0),
        np.array([160.0, -170.0, -170.0]),
    )

    segments = split_wrapped_curve(
        np.array([160.0, 170.0, -175.0, -160.0, np.nan, 10.0, 20.0]),
        np.array([0.0, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0]),
    )

    assert len(segments) == 3
    np.testing.assert_allclose(segments[0][0], np.array([160.0, 170.0]))
    np.testing.assert_allclose(segments[1][0], np.array([-175.0, -160.0]))
    np.testing.assert_allclose(segments[2][0], np.array([10.0, 20.0]))


def test_map_curve_layer_builder_filters_visible_series():
    """Curve layer specs should be reusable outside the notebook."""
    layers = build_timeseries_curve_layers(
        [
            {"series_key": "measured", "label": "Measured", "values": [[1.0, 2.0]]},
            {"series_key": "inductive", "label": "Inductive", "values": [[3.0, 4.0]]},
        ],
        visible_series={"inductive"},
    )

    assert [layer["series_key"] for layer in layers] == ["inductive"]
    np.testing.assert_allclose(layers[0]["values"], np.array([[3.0, 4.0]]))


def test_geographic_local_time_window_helpers_are_parameterized():
    """Site masks and extents should not depend on notebook widgets."""
    reference_time = dt.datetime(2011, 10, 24, 18, 0)
    lat = np.array([-30.0, 0.0, 20.0, 70.0])
    lon = np.array([-90.0, 0.0, 90.0, 180.0])

    mask = geographic_local_time_mask(
        lat,
        lon,
        lat_window=(-45.0, 45.0),
        local_time_window=(6.0, 18.0),
        reference_time=reference_time,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))
    assert (
        local_time_window_extent(
            lat_window=(-90.0, 90.0), local_time_window=(0.0, 24.0), reference_time=reference_time
        )
        is None
    )
    np.testing.assert_allclose(
        local_time_window_extent(
            lat_window=(-30.0, 60.0),
            local_time_window=(6.0, 18.0),
            reference_time=reference_time,
            central_longitude=0.0,
        ),
        np.array([-180.0, 0.0, -30.0, 60.0]),
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
        levels, np.array([-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0])
    )
    np.testing.assert_allclose(
        get_ticks_from_levels({"levels": np.array([0.0, 2.0, 4.0])}), np.array([1.0, 3.0])
    )
    assert format_contour_interval(0.001) == "1.00e-03"
    assert contour_kwargs_for_display(
        {"levels": levels, "symbol": "x", "units": "T", "scale": 1.0}
    ) == {"levels": levels}


def test_percentile_color_scale_handles_diverging_data_symmetrically():
    """Diverging data use a symmetric percentile limit."""
    scale = build_percentile_color_scale(
        [np.array([-1.0, 0.0, 2.0, 100.0, np.nan])], strictly_positive=False, vmax_percentile=75.0
    )

    assert scale["strictly_positive"] is False
    assert scale["scale_type"] == "linear"
    np.testing.assert_allclose(scale["vmin"], -26.5)
    np.testing.assert_allclose(scale["vmax"], 26.5)
    assert scale["norm"].vmin == scale["vmin"]
    assert scale["norm"].vmax == scale["vmax"]


def test_percentile_color_scale_handles_positive_linear_and_log_data():
    """Positive fields can use linear or log percentile scales."""
    values = np.array([0.0, 1.0, 10.0, 100.0, np.nan])

    linear = build_percentile_color_scale(
        [values], strictly_positive=True, vmin_percentile=25.0, vmax_percentile=75.0
    )
    assert linear["vmin"] == 0.0
    np.testing.assert_allclose(linear["vmax"], 32.5)

    log = build_percentile_color_scale(
        [values],
        strictly_positive=True,
        vmin_percentile=0.0,
        vmax_percentile=100.0,
        scale_type="log",
    )
    np.testing.assert_allclose(log["vmin"], 1.0)
    np.testing.assert_allclose(log["vmax"], 100.0)
    assert log["norm"].vmin == log["vmin"]
    assert log["norm"].vmax == log["vmax"]


def test_style_global_axis_centralizes_map_setup():
    """Global-axis styling controls map decorations."""

    class FakeGridliner:
        pass

    class FakeAxis:
        def __init__(self):
            self.global_called = False
            self.coastline_kwargs = None
            self.gridline_kwargs = None

        def set_global(self):
            self.global_called = True

        def coastlines(self, **kwargs):
            self.coastline_kwargs = kwargs

        def gridlines(self, **kwargs):
            self.gridline_kwargs = kwargs
            return FakeGridliner()

    ax = FakeAxis()
    gridliner = style_global_axis(ax, draw_labels=False, draw_coastlines=False, set_global=False)

    assert not ax.global_called
    assert ax.coastline_kwargs is None
    assert ax.gridline_kwargs["draw_labels"] is False
    assert gridliner.left_labels is False
    assert gridliner.bottom_labels is False
    assert gridliner.top_labels is False
    assert gridliner.right_labels is False


def test_grid_and_conductance_helpers_are_importable_from_visualization():
    """Grid/evaluation helpers are available through the package."""
    lat, lon, grid = build_plot_grid(nlat=3, nlon=4)
    assert lat.shape == (3, 4)
    assert lon.shape == (3, 4)
    assert grid.size == 12

    sigmaP, sigmaH = resistance_to_conductance(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
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


def test_JS_operator_bundle_matches_core_formulas():
    """Shared JS helper follows geometry formulas."""

    class Settings:
        RI = 1.0
        RM = 2.0
        RM_shielding = True

    sh_basis = SHBasis(3, 2, mean_free=True)
    solid_harmonics = SolidHarmonics(sh_basis)
    _, _, grid = build_plot_grid(nlat=4, nlon=5)
    transform = build_evaluator(sh_basis, grid)
    t_to_ve = np.eye(sh_basis.index_length)

    operators = build_JS_operators(Settings, sh_basis, transform, T_to_Ve=t_to_ve)

    poloidal_to_JS = (
        -transform.scalar_coeffs_to_gridded_rhat_cross_gradient
        * solid_harmonics.poloidal_to_boundary_potential_jump_factor.reshape(1, 1, -1)
        / mu0
    )
    toroidal_to_JS = -transform.scalar_coeffs_to_gridded_gradient / mu0
    regular_shift = solid_harmonics.regular_reference_shift(Settings.RM, Settings.RI)
    irregular_shift = solid_harmonics.irregular_reference_shift(Settings.RI, Settings.RM)
    denominator = 1.0 - regular_shift * irregular_shift
    m_ind_to_br = -(Settings.RI**2) * sh_basis.laplacian(Settings.RI)

    np.testing.assert_allclose(
        operators["G_m_imp_to_JS"],
        toroidal_to_JS + np.tensordot(poloidal_to_JS, t_to_ve, axes=([2], [0])),
    )
    np.testing.assert_allclose(
        operators["G_m_ind_to_JS"],
        poloidal_to_JS * (1.0 + regular_shift * irregular_shift / denominator),
    )
    np.testing.assert_allclose(
        operators["G_Br_to_JS"], poloidal_to_JS * (-regular_shift / (denominator * m_ind_to_br))
    )


def test_JS_operator_bundle_defaults_to_unshielded_rm():
    """RM does not impose shielding unless requested."""

    class Settings:
        RI = 1.0
        RM = 2.0

    sh_basis = SHBasis(3, 2, mean_free=True)
    solid_harmonics = SolidHarmonics(sh_basis)
    _, _, grid = build_plot_grid(nlat=4, nlon=5)
    transform = build_evaluator(sh_basis, grid)

    operators = build_JS_operators(Settings, sh_basis, transform)
    poloidal_to_JS = (
        -transform.scalar_coeffs_to_gridded_rhat_cross_gradient
        * solid_harmonics.poloidal_to_boundary_potential_jump_factor.reshape(1, 1, -1)
        / mu0
    )

    np.testing.assert_allclose(operators["G_m_ind_to_JS"], poloidal_to_JS)


def test_JS_operator_bundle_matches_geometry(tmp_path):
    """Notebook helper matches Geometry JS conventions."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * RE,
        ignore_PFAC=True,
        artifact_storage="netcdf",
    )
    geometry = dynamics.state.geometry
    _, _, grid = build_plot_grid(nlat=4, nlon=5)
    transform = build_evaluator(dynamics.horizontal_basis, grid)
    operators = build_JS_operators(
        dynamics.settings, dynamics.horizontal_basis, transform, T_to_Ve=geometry.T_to_Ve.values
    )

    np.testing.assert_allclose(operators["G_m_ind_to_JS"], geometry.m_ind_to_gridded_JS(transform))
    np.testing.assert_allclose(operators["G_m_imp_to_JS"], geometry.m_imp_to_gridded_JS(transform))
    np.testing.assert_allclose(operators["G_Br_to_JS"], geometry.Br_to_gridded_JS(transform))
