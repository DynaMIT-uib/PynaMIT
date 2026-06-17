"""Tests for the MAGE forcing preparation helpers."""

import numpy as np

from scripts.simulation.mage_forcing_final import DEFAULT_FORCING_CANDIDATES
from scripts.simulation.mage_prepare_forcing import (
    DEFAULT_OUTPUT_NAME,
    integrate_tiegcm_step,
    wrap_longitude_180_value,
)


class _FakeVariable:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, item):
        return self.values[item]


class _FakeDataset:
    def __init__(self, **variables):
        self.variables = {name: _FakeVariable(values) for name, values in variables.items()}


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
