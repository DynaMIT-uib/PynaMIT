"""Tests for cached main-field values on spherical grids."""

import numpy as np
from kompe import SphericalGrid
from kompe.constants import EARTH_RADIUS_M
from kompe.math import JAX_AVAILABLE, backend_context, get_backend

import pynamit
import pynamit.geomagnetism as geomagnetism
from pynamit.geomagnetism import MainField


def test_bound_field_evaluation_object_is_not_public():
    """Grid evaluation belongs directly to MainField."""
    assert not hasattr(pynamit, "MagneticFieldEvaluation")
    assert not hasattr(geomagnetism, "MagneticFieldEvaluation")


def test_main_field_grid_values_use_selected_array_backend():
    """Provider output crosses to the selected backend only once."""
    grid = SphericalGrid(lat=[-60.0, 20.0, 70.0], lon=[0.0, 90.0, -120.0])
    main_field = MainField(kind="radial")
    components = main_field.evaluate(grid, EARTH_RADIUS_M)

    assert components.shape == (3, grid.size)
    assert main_field.unit_vector(grid, EARTH_RADIUS_M).shape == (3, grid.size)
    assert main_field.horizontal_to_apex_array(grid, EARTH_RADIUS_M).shape == (2, 2, grid.size)
    assert main_field.radial_to_apex_scale(grid, EARTH_RADIUS_M).shape == (grid.size,)

    array_module = type(components).__module__.split(".", maxsplit=1)[0]
    expected_modules = {"jax", "jaxlib"} if get_backend() == "jax" else {"numpy"}
    assert array_module in expected_modules

    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(main_field.unit_vector(grid, EARTH_RADIUS_M)), axis=0), 1.0
    )


def test_main_field_reuses_values_for_equivalent_grid_coordinates():
    """Equivalent coordinate grids share numerical field values."""
    main_field = MainField(kind="radial")
    grid = SphericalGrid(
        lat=np.array([[-60.0], [20.0]]), lon=np.array([[0.0], [90.0]]), area_weights=[1.0, 1.0]
    )
    equivalent_grid = SphericalGrid(lat=[-60.0, 20.0], lon=[0.0, 90.0], area_weights=[2.0, 2.0])

    components = main_field.evaluate(grid, EARTH_RADIUS_M)
    repeated = main_field.evaluate(grid, EARTH_RADIUS_M)
    equivalent = main_field.evaluate(equivalent_grid, EARTH_RADIUS_M)
    another_radius = main_field.evaluate(grid, EARTH_RADIUS_M + 1.0)

    assert repeated is components
    assert equivalent is components
    assert main_field.unit_vector(grid, EARTH_RADIUS_M) is main_field.unit_vector(
        equivalent_grid, EARTH_RADIUS_M
    )
    assert another_radius is not components


def test_main_field_cache_lifecycle():
    """Field models expose bounded cache occupancy and clearing."""
    main_field = MainField(kind="radial")
    grid = SphericalGrid(lat=[-60.0, 20.0], lon=[0.0, 90.0])

    assert main_field.cache_info() == {"grids": 0, "max_size": 8}
    components = main_field.evaluate(grid, EARTH_RADIUS_M)
    _ = main_field.horizontal_to_apex_array(grid, EARTH_RADIUS_M)
    assert main_field.cache_info() == {"grids": 1, "max_size": 8}

    main_field.clear_cache()
    assert main_field.cache_info() == {"grids": 0, "max_size": 8}
    assert main_field.evaluate(grid, EARTH_RADIUS_M) is not components

    for longitude in range(1, 9):
        another_grid = SphericalGrid(lat=[20.0], lon=[float(longitude)])
        main_field.evaluate(another_grid, EARTH_RADIUS_M)
    assert main_field.cache_info() == {"grids": 8, "max_size": 8}


def test_main_field_grid_cache_is_backend_specific():
    """NumPy and JAX values occupy distinct cache entries."""
    main_field = MainField(kind="radial")
    grid = SphericalGrid(lat=[-60.0, 20.0], lon=[0.0, 90.0])

    with backend_context("numpy"):
        numpy_components = main_field.evaluate(grid, EARTH_RADIUS_M)

    switched_backend = "jax" if JAX_AVAILABLE else "numpy"
    with backend_context(switched_backend):
        switched_components = main_field.evaluate(grid, EARTH_RADIUS_M)

    assert type(numpy_components).__module__.split(".", maxsplit=1)[0] == "numpy"
    if JAX_AVAILABLE:
        switched_module = type(switched_components).__module__.split(".", maxsplit=1)[0]
        assert switched_module in {"jax", "jaxlib"}
        assert switched_components is not numpy_components
        assert main_field.cache_info()["grids"] == 2
