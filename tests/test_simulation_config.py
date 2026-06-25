"""Tests for simulation configuration normalization."""

import numpy as np
import pytest
import xarray as xr

from pynamit.simulation.config import SimulationConfig, setting_value


def test_simulation_config_normalizes_projection_defaults():
    """Projection settings inherit from basis and wind route."""
    config = SimulationConfig(u_projection_basis="CS")

    assert config.horizontal_basis_kind == "SH"
    assert config.jr_projection_basis == "SH"
    assert config.Br_projection_basis == "SH"
    assert config.conductance_projection_basis == "SH"
    assert config.u_projection_basis == "CS"
    assert config.Q_eff_projection_basis == "CS"


def test_simulation_config_cs_mode_requires_cs_projection_routes():
    """CS horizontal mode cannot mix in SH input routes."""
    config = SimulationConfig(horizontal_basis_kind="cs")

    assert config.horizontal_basis_kind == "CS"
    assert config.jr_projection_basis == "CS"
    assert config.Br_projection_basis == "CS"
    assert config.conductance_projection_basis == "CS"
    assert config.u_projection_basis == "CS"
    assert config.Q_eff_projection_basis == "CS"

    with pytest.raises(ValueError, match="jr_projection_basis"):
        SimulationConfig(horizontal_basis_kind="CS", jr_projection_basis="SH")


def test_simulation_config_dataset_roundtrip_preserves_stored_sentinels():
    """The settings dataset is the canonical serialized config."""
    config = SimulationConfig(
        Nmax=3,
        Mmax=2,
        Ncs=4,
        RM=None,
        RM_shielding=True,
        mainfield_B0=None,
        ignore_PFAC=True,
        area_weighted_least_squares=True,
        static_preconditioner=True,
        m_imp_regularization_lambda=1e-3,
    )

    settings = config.to_dataset()
    restored = SimulationConfig.from_settings(settings)

    assert settings.attrs["RM"] == 0
    assert settings.attrs["mainfield_B0"] == 0
    assert setting_value(settings, "ignore_PFAC") == 1
    assert restored.RM is None
    assert restored.RM_shielding
    assert restored.mainfield_B0 is None
    assert restored.ignore_PFAC
    assert restored.area_weighted_least_squares
    assert restored.static_preconditioner
    assert restored.m_imp_regularization_lambda == pytest.approx(1e-3)
    np.testing.assert_allclose(restored.FAC_integration_steps, config.FAC_integration_steps)


def test_simulation_config_from_minimal_settings_accepts_missing_defaults():
    """Minimal settings are expanded through the same default path."""
    settings = xr.Dataset(attrs={"Nmax": 3, "Mmax": 2, "Ncs": 4})

    config = SimulationConfig.from_settings(settings)

    assert config.Nmax == 3
    assert config.Mmax == 2
    assert config.Ncs == 4
    assert config.horizontal_basis_kind == "SH"
    assert config.conductance_projection_basis == "SH"


def test_simulation_config_preserves_decimal_mainfield_epoch():
    """Main-field epochs can be decimal years."""
    config = SimulationConfig(mainfield_epoch=2011.813014015728)
    restored = SimulationConfig.from_settings(config.to_dataset())

    assert config.mainfield_epoch == pytest.approx(2011.813014015728)
    assert restored.mainfield_epoch == pytest.approx(config.mainfield_epoch)


def test_simulation_config_from_settings_rejects_conflicting_override():
    """Explicit overrides must agree with stored settings."""
    settings = xr.Dataset(attrs={"Nmax": 3, "Mmax": 2, "Ncs": 4, "horizontal_basis_kind": "CS"})

    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        SimulationConfig.from_settings(settings, horizontal_basis_kind="SH")


def test_simulation_config_from_settings_uses_override_when_missing():
    """Explicit overrides fill absent settings."""
    settings = xr.Dataset(attrs={"Nmax": 3, "Mmax": 2, "Ncs": 4})

    config = SimulationConfig.from_settings(settings, horizontal_basis_kind="CS")

    assert config.horizontal_basis_kind == "CS"
    assert config.jr_projection_basis == "CS"
