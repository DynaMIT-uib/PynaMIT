"""Tests for figure renderers contracts."""

import importlib

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("plot_type", "expected_coordinate_system"),
    [("global", "geographic"), ("hemispheres", "model")],
)
def test_field_renderer_selects_coordinates_for_map_type(
    monkeypatch, plot_type, expected_coordinate_system
):
    """Global maps are geographic while hemispheres stay magnetic."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.plotting.field_comparison_figures")
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")
    map_coordinates = importlib.import_module("pynamit.plotting.map_coordinates")

    class CapturingView:
        lat, lon = np.meshgrid(np.linspace(-80.0, 80.0, 4), np.linspace(-180.0, 180.0, 5))
        results = type("Results", (), {"datasets": {"dynamic": object()}})()
        coordinate_system = None
        map_context_calls = 0

        def output_plot_data(self, index, *, field_names, coordinate_system):
            del index, field_names
            self.coordinate_system = coordinate_system
            return {"Br_dynamic": np.zeros(self.lat.shape)}

        @staticmethod
        def timestamp_at_index(index):
            return pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=index)

        def geographic_map_context(self, reference_time=None):
            self.map_context_calls += 1
            return map_coordinates.MapCoordinateContext.from_noon_longitude(
                30.0,
                longitude_kind="geographic",
                local_time_kind="solar",
                reference_time=reference_time,
            )

        @classmethod
        def magnetic_plot_coordinates(cls):
            return cls.lat, cls.lon

        magnetic_map_context = geographic_map_context
        model_map_context = geographic_map_context

    monkeypatch.setattr(
        figures, "_draw_field_comparison_artists", lambda *args, **kwargs: (None, None)
    )
    view = CapturingView()
    settings = figure_settings.FigureSettings(
        plot_type=plot_type, show_dynamic=True, show_equilibrium=False, show_difference=False
    )

    figure = figures.FieldComparisonRenderer(settings, plot_data=view).render()
    try:
        assert view.coordinate_system == expected_coordinate_system
        assert view.map_context_calls == 1
        if plot_type == "global":
            geo_axes = [axis for axis in figure.axes if hasattr(axis, "projection")]
            assert len(geo_axes) == 1
            assert geo_axes[0].projection.equals(ccrs.PlateCarree(central_longitude=30.0))
    finally:
        plt.close(figure)


@pytest.mark.parametrize("hemisphere", ["global", "north", "south"])
@pytest.mark.parametrize("filled_key", [None, "Br"])
def test_comparison_contours_remain_on_axes(hemisphere, filled_key):
    """Axes retain contours without a separate artist collection."""
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourSet

    from pynamit.plotting.field_comparison_figures import _draw_field_comparison_artists
    from pynamit.plotting.figure_styles import FIELD_DIFF_KWARGS, FIELD_PLOT_KWARGS
    from pynamit.plotting.hemisphere import make_hemisphere_polarplot
    from pynamit.plotting.map_coordinates import MapCoordinateContext

    lat, lon = np.meshgrid(np.linspace(-85, 85, 18), np.linspace(-180, 180, 25), indexing="ij")
    pattern = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    fields = {
        "Br_dynamic": 8e-8 * pattern,
        "Br_equilibrium": 6e-8 * pattern,
        "Phi_dynamic": 100 * pattern,
        "Phi_equilibrium": 80 * pattern,
    }
    subplot_kw = {"projection": ccrs.PlateCarree()} if hemisphere == "global" else {}
    figure, axes = plt.subplots(1, 3, subplot_kw=subplot_kw)
    try:
        plot_axes = (
            axes if hemisphere == "global" else [make_hemisphere_polarplot(axis) for axis in axes]
        )
        main_mappable, diff_mappable = _draw_field_comparison_artists(
            [{"hemisphere": hemisphere, "axes": plot_axes}],
            filled_key,
            ["Phi"],
            fields,
            lat,
            lon,
            MapCoordinateContext.from_noon_longitude(0.0),
            plot_kwargs=FIELD_PLOT_KWARGS,
            diff_kwargs=FIELD_DIFF_KWARGS,
        )
        for axis in axes:
            contours = [artist for artist in axis.collections if isinstance(artist, ContourSet)]
            assert len(contours) == (1 if filled_key is None else 2)
            assert sum(artist.filled for artist in contours) == (filled_key is not None)
        if filled_key is None:
            assert main_mappable is None and diff_mappable is None
        else:
            assert main_mappable.axes is axes[1]
            assert diff_mappable.axes is axes[2]
    finally:
        plt.close(figure)


def test_field_renderer_applies_manual_fill_and_line_levels(monkeypatch):
    """Manual controls replace selected main-field presets."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.plotting.field_comparison_figures")
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")
    map_coordinates = importlib.import_module("pynamit.plotting.map_coordinates")

    class View:
        lat, lon = np.meshgrid(np.linspace(-80.0, 80.0, 4), np.linspace(-180.0, 180.0, 5))
        results = type("Results", (), {"datasets": {"equilibrium": object()}})()

        @classmethod
        def output_plot_data(cls, index, *, field_names, coordinate_system):
            del index, field_names, coordinate_system
            return {
                "jr_equilibrium": np.zeros(cls.lat.shape),
                "Phi_equilibrium": np.zeros(cls.lat.shape),
            }

        @staticmethod
        def timestamp_at_index(index):
            return pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=index)

        @staticmethod
        def geographic_map_context(reference_time=None):
            return map_coordinates.MapCoordinateContext.geographic(reference_time)

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(figures, "_draw_field_comparison_artists", capture)
    settings = figure_settings.FigureSettings(
        plot_type="global",
        fill="jr",
        lines="Phi",
        show_dynamic=False,
        show_equilibrium=True,
        show_difference=False,
        manual_color_min=-5e-7,
        manual_color_max=5e-7,
        color_scale_mode="manual",
        line_first_abs_level=4.0,
        line_interval=4.0,
        line_levels_per_sign=3,
    )

    figure = figures.FieldComparisonRenderer(settings, plot_data=View()).render()
    try:
        np.testing.assert_allclose(
            captured["plot_kwargs"]["jr"]["levels"], np.linspace(-5e-7, 5e-7, 18)
        )
        np.testing.assert_allclose(
            captured["plot_kwargs"]["Phi"]["levels"], [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0]
        )
    finally:
        plt.close(figure)


def test_hemisphere_renderer_uses_cutoff_and_writes_coordinate_labels(monkeypatch):
    """Polar axes use the requested edge and show MLT orientation."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.plotting.field_comparison_figures")
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")
    captured = []

    class FakePolar:
        def __init__(self, axis, min_abs_latitude):
            self.ax = axis
            self.min_abs_latitude = min_abs_latitude
            self.lat_labels = []
            self.lt_labels = 0

        def writeLATlabels(self, **kwargs):
            self.lat_labels.append(kwargs)

        def writeLTlabels(self):
            self.lt_labels += 1

    def fake_polar(axis, min_abs_latitude):
        polar = FakePolar(axis, min_abs_latitude)
        captured.append(polar)
        return polar

    monkeypatch.setattr(figures, "make_hemisphere_polarplot", fake_polar)
    renderer = object.__new__(figures.FieldComparisonRenderer)
    renderer.settings = figure_settings.FigureSettings(
        plot_type="hemispheres",
        show_north=True,
        show_south=False,
        hemisphere_min_abs_latitude=42.0,
    )

    figure, _, _ = renderer._create_hemisphere_axes([("dynamic", "Dynamic")])
    try:
        assert len(captured) == 1
        assert captured[0].min_abs_latitude == 42.0
        assert captured[0].lat_labels == [
            {"color": "black", "backgroundcolor": (0, 0, 0, 0), "north": True}
        ]
        assert captured[0].lt_labels == 1
    finally:
        plt.close(figure)


def test_line_legend_omits_unused_difference_interval():
    """A single-field plot should not advertise difference contours."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.plotting.field_comparison_figures")
    styles = importlib.import_module("pynamit.plotting.figure_styles")
    figure, axis = plt.subplots()
    try:
        figures.FieldComparisonRenderer._draw_map_line_legend(
            axis,
            ["Phi"],
            styles.FIELD_PLOT_KWARGS,
            styles.FIELD_DIFF_KWARGS,
            include_difference=False,
        )
        labels = [text.get_text() for text in axis.texts]
        assert len(labels) == 1
        assert "8 kV" in labels[0]
        assert "diff" not in labels[0]
    finally:
        plt.close(figure)


def test_input_summary_keeps_polar_jr_model_aligned(monkeypatch):
    """Mixed input figures request both model and geographic fields."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.plotting.input_driver_figures")
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")
    map_coordinates = importlib.import_module("pynamit.plotting.map_coordinates")

    class CapturingView:
        lat, lon = np.meshgrid(np.linspace(-80.0, 80.0, 4), np.linspace(-180.0, 180.0, 5))
        wind_lat, wind_lon = lat, lon

        def __init__(self):
            self.coordinate_systems = []

        def input_plot_data(self, index, *, coordinate_system):
            del index
            self.coordinate_systems.append(coordinate_system)
            shape = self.lat.shape
            return {
                "jr": np.zeros(shape),
                "Br": np.zeros(shape),
                "SigmaP": np.ones(shape),
                "SigmaH": np.ones(shape),
                "wind_theta": np.zeros(shape),
                "wind_phi": np.zeros(shape),
                "Q_eff_theta": np.full(shape, np.nan),
                "Q_eff_phi": np.full(shape, np.nan),
                "E_neutral_wind_theta": np.full(shape, np.nan),
                "E_neutral_wind_phi": np.full(shape, np.nan),
            }

        @staticmethod
        def timestamp_at_index(index):
            return np.datetime64("2020-01-01") + np.timedelta64(index, "s")

        @staticmethod
        def geographic_map_context(reference_time=None):
            return map_coordinates.MapCoordinateContext.from_noon_longitude(
                30.0,
                longitude_kind="geographic",
                local_time_kind="solar",
                reference_time=reference_time,
            )

        @classmethod
        def magnetic_plot_coordinates(cls):
            return cls.lat, cls.lon

        magnetic_map_context = geographic_map_context
        model_map_context = geographic_map_context

    monkeypatch.setattr(figures.InputDriverRenderer, "_draw_jr_hemispheres", lambda *args: None)
    monkeypatch.setattr(
        figures.InputDriverRenderer, "_draw_global_scalars", lambda *args: (None, None)
    )
    monkeypatch.setattr(figures.InputDriverRenderer, "_draw_tangential_source", lambda *args: None)
    monkeypatch.setattr(figures.InputDriverRenderer, "_draw_colorbars", lambda *args: None)
    view = CapturingView()
    settings = figure_settings.FigureSettings(plot_type="input_summary")

    figure = figures.InputDriverRenderer(settings, plot_data=view).render()
    try:
        assert view.coordinate_systems == ["model", "geographic"]
        geo_axes = [axis for axis in figure.axes if hasattr(axis, "projection")]
        assert len(geo_axes) == 4
        assert all(
            axis.projection.equals(ccrs.PlateCarree(central_longitude=30.0)) for axis in geo_axes
        )
    finally:
        plt.close(figure)
