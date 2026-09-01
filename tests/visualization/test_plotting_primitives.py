"""Tests for reusable map, contour, and plotting primitives."""

import datetime as dt

import cartopy.crs as ccrs
import numpy as np
import pytest

from pynamit.coordinates import (
    datetime_to_utc_hours,
    local_noon_longitude,
    local_time_hours_to_longitude,
    local_time_longitude_to_geographic,
    longitude_to_local_time_from_noon_longitude,
    longitude_to_local_time_hours,
    wrap_longitude_180,
)
from pynamit.plotting.contours import (
    build_percentile_color_scale,
    contour_kwargs_for_display,
    format_contour_interval,
    get_ticks_from_levels,
    symmetric_contour_levels,
    symmetric_contour_levels_without_zero,
)
from pynamit.plotting.hemisphere import (
    coerce_hemisphere_min_abs_latitude,
    hemisphere_masks_for_latitude,
)
from pynamit.plotting.map_axes import style_global_axis
from pynamit.plotting.map_coordinates import MapCoordinateContext, regular_geographic_grid
from pynamit.plotting.map_curves import (
    build_even_global_sites,
    build_timeseries_curve_layers,
    geographic_local_time_mask,
    local_time_window_extent,
    split_wrapped_curve,
    wrap_longitudes,
)
from pynamit.storage import ArtifactStore


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

    with pytest.raises(ValueError):
        coerce_hemisphere_min_abs_latitude("bad")
    with pytest.raises(ValueError, match="hemisphere_min_abs_latitude"):
        coerce_hemisphere_min_abs_latitude(95.0)
    north, south = hemisphere_masks_for_latitude(latitudes, min_abs_latitude=40.0)

    np.testing.assert_array_equal(north, np.array([False, False, False, True, True]))
    np.testing.assert_array_equal(south, np.array([True, False, False, False, False]))


def test_local_time_grid_and_source_longitude_conversion():
    """Local-time grid helpers match the notebook convention."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)

    np.testing.assert_allclose(
        MapCoordinateContext.geographic(reference_time).local_time_grid_longitudes(),
        np.array([127.5, -142.5, -52.5, 37.5]),
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


def test_map_curve_sampling_rejects_invalid_geometry_settings():
    """Invalid sampling geometry is not silently repaired."""
    with pytest.raises(ValueError, match="ordered"):
        build_even_global_sites(min_lat=60.0, max_lat=-60.0)
    with pytest.raises(ValueError, match="lat_step"):
        build_even_global_sites(lat_step=0.0)
    with pytest.raises(ValueError, match="lat_count"):
        build_even_global_sites(lat_count=0)
    with pytest.raises(ValueError, match="equatorial_count"):
        build_even_global_sites(equatorial_count=1.5)


def test_percentile_contours_reject_invalid_percentiles():
    """Invalid percentile requests are not silently clipped."""
    from pynamit.plotting.contours import percentile_contour_levels

    with pytest.raises(ValueError, match="percentile"):
        percentile_contour_levels([np.arange(3.0)], np.arange(3.0), percentile=101.0)


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
            {"series_key": "dynamic", "label": "Dynamic", "values": [[3.0, 4.0]]},
        ],
        visible_series={"dynamic"},
    )

    assert [layer["series_key"] for layer in layers] == ["dynamic"]
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


def test_geographic_window_helpers_reject_invalid_limits():
    """Coordinate windows are not silently sorted or clipped."""
    reference_time = dt.datetime(2011, 10, 24, 18, 30)
    with pytest.raises(ValueError, match="lat_window"):
        geographic_local_time_mask([0.0], [0.0], lat_window=(30.0, -30.0))
    with pytest.raises(ValueError, match="local_time_window"):
        local_time_window_extent(local_time_window=(-1.0, 12.0), reference_time=reference_time)


def test_artifact_store_reports_existing_visualization_artifact_path(tmp_path):
    """Visualization uses the canonical artifact store paths."""
    simulation_directory = tmp_path / "run"
    simulation_directory.mkdir()
    (simulation_directory / "sample.zarr").mkdir()
    (simulation_directory / "settings.ncdf").write_bytes(b"")

    store = ArtifactStore(simulation_directory)
    assert store.existing_artifact_path("sample") == simulation_directory / "sample.zarr"
    assert store.existing_artifact_path("settings") == simulation_directory / "settings.ncdf"
    assert store.existing_artifact_path("jr") is None


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


def test_symmetric_contour_levels_use_explicit_start_spacing_and_count():
    """Manual controls create a zero-free sequence on both signs."""
    levels = symmetric_contour_levels(4.0, 4.0, 3)

    np.testing.assert_allclose(levels, [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0])


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


def test_style_global_axis_does_not_mix_coastlines_with_magnetic_coordinates():
    """Omit geographic coastlines from magnetic-coordinate axes."""

    class FakeGridliner:
        pass

    class FakeAxis:
        def __init__(self):
            self.coastline_kwargs = None

        def set_global(self):
            pass

        def coastlines(self, **kwargs):
            self.coastline_kwargs = kwargs

        @staticmethod
        def gridlines(**kwargs):
            del kwargs
            return FakeGridliner()

    context = MapCoordinateContext.from_noon_longitude(
        0.0, longitude_kind="magnetic", local_time_kind="magnetic"
    )
    axis = FakeAxis()

    style_global_axis(axis, coordinate_context=context)

    assert axis.coastline_kwargs is None


def test_style_global_axis_keeps_coastlines_and_gridlines_geographic():
    """Noon-centered GEO axes retain geographic decorations."""

    class FakeGridliner:
        pass

    class FakeAxis:
        def __init__(self):
            self.coastline_kwargs = None
            self.gridline_kwargs = None

        def set_global(self):
            pass

        def coastlines(self, **kwargs):
            self.coastline_kwargs = kwargs

        def gridlines(self, **kwargs):
            self.gridline_kwargs = kwargs
            return FakeGridliner()

    reference_time = dt.datetime(2011, 10, 24, 18, 30)
    context = MapCoordinateContext.geographic(reference_time)
    axis = FakeAxis()

    style_global_axis(axis, coordinate_context=context)

    assert axis.coastline_kwargs is not None
    assert axis.gridline_kwargs["crs"].equals(ccrs.PlateCarree())
    np.testing.assert_allclose(
        axis.gridline_kwargs["crs"].transform_points(
            ccrs.PlateCarree(), np.array([context.noon_longitude]), np.array([0.0])
        )[:, :2],
        [[context.noon_longitude, 0.0]],
        atol=1e-10,
    )


def test_grid_helpers_are_importable_from_visualization():
    """Expose spherical-grid helpers through the package."""
    lat, lon, grid = regular_geographic_grid(nlat=3, nlon=4)
    assert lat.shape == (3, 4)
    assert lon.shape == (3, 4)
    assert grid.size == 12
