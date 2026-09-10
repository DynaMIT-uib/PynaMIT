"""Tests for package contracts."""

import importlib

import pynamit


def test_package_entry_points_are_available():
    """Each maintained namespace exposes its main interactive tools."""
    results = importlib.import_module("pynamit.results")
    plotting = importlib.import_module("pynamit.plotting")
    plot_data = importlib.import_module("pynamit.plotting.plot_data")
    gui = importlib.import_module("pynamit.gui")
    mage = importlib.import_module("pynamit.workflows.mage")

    assert results.SimulationResults is pynamit.SimulationResults
    assert callable(results.evaluate_projected_input)
    assert plotting.FigureSettings.__name__ == "FigureSettings"
    assert plotting.PlotData.__name__ == "PlotData"
    assert callable(plot_data.evaluate_output_fields_at_time)
    assert callable(plotting.render_figure)
    assert gui.PynamitGUI.__name__ == "PynamitGUI"
    assert callable(gui.build_gui)
    assert mage.ForcingSettings.__name__ == "ForcingSettings"
    assert callable(mage.prepare_forcing)
    assert callable(mage.prepare_inputs)


def test_optional_namespaces_advertise_only_the_common_api():
    """Optional namespaces keep a small explicit export set."""
    plotting = importlib.import_module("pynamit.plotting")
    gui = importlib.import_module("pynamit.gui")
    mage = importlib.import_module("pynamit.workflows.mage")

    assert plotting.__all__ == ["FigureSettings", "render_figure", "save_movie"]
    assert gui.__all__ == ["PynamitGUI", "build_gui"]
    assert mage.__all__ == ["ForcingSettings", "prepare_forcing", "prepare_inputs"]
    assert {"FigureSettings", "PlotData", "plot_output_quicklook"}.issubset(dir(plotting))
    assert {"PynamitGUI", "build_gui", "main"}.issubset(dir(gui))
    assert {
        "ForcingSettings",
        "prepare_forcing",
        "prepare_inputs",
        "write_input_projection_diagnostics",
    }.issubset(dir(mage))


def test_input_projection_comparison_recipe_is_importable():
    """Input diagnostic recipes are available from their module."""
    diagnostics = importlib.import_module("pynamit.workflows.mage.diagnostics")

    assert callable(diagnostics.plot_input_projection_comparison)
    assert callable(diagnostics.write_input_projection_diagnostics)
