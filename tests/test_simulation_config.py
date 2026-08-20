"""Tests for simulation configuration normalization."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr
from kompe.constants import EARTH_RADIUS_M

from pynamit.geomagnetism import decimal_year
from pynamit.simulation import Simulation
from pynamit.simulation.config import SimulationConfig, dipole_fac_integration_radii, setting_value


def test_simulation_constructs_from_normalized_config(tmp_path):
    """A public config can construct a simulation directly."""
    config = SimulationConfig(
        Nmax=2, Mmax=1, Ncs=4, main_field_kind="radial", enable_pfac_coupling=False
    )

    simulation = Simulation.from_config(
        config, simulation_directory=tmp_path, artifact_storage="netcdf", backend="numpy"
    )

    assert simulation.config.to_dataset().identical(config.to_dataset())
    assert simulation.data.simulation_directory == str(tmp_path)


def test_simulation_from_config_requires_normalized_config():
    """Reject settings-like objects at the explicit config boundary."""
    with pytest.raises(TypeError, match="requires a SimulationConfig"):
        Simulation.from_config({"Nmax": 2})


def test_operator_cache_is_a_nonphysical_runtime_preference(tmp_path):
    """Restart can change or omit the disposable operator-cache path."""
    simulation_directory = tmp_path / "run"
    first_cache = tmp_path / "first-cache"
    config = SimulationConfig(
        Nmax=2, Mmax=1, Ncs=4, main_field_kind="radial", enable_pfac_coupling=False
    )
    original = Simulation.from_config(
        config,
        simulation_directory=simulation_directory,
        artifact_storage="netcdf",
        operator_cache_directory=first_cache,
        backend="numpy",
    )
    assert original.operator_cache.directory == first_cache.resolve()
    assert list(first_cache.rglob("*.npy"))

    reloaded = Simulation.from_directory(
        simulation_directory,
        artifact_storage="netcdf",
        operator_cache_directory=tmp_path / "second-cache",
        backend="numpy",
    )

    assert reloaded.config.to_dataset().identical(original.config.to_dataset())
    assert "operator_cache_directory" not in reloaded.config.to_kwargs()


def test_simulation_config_normalizes_projection_defaults():
    """Projection settings inherit from basis and wind route."""
    config = SimulationConfig(u_projection_basis="CS")

    assert config.horizontal_basis_kind == "SH"
    assert config.boundary_jr_projection_basis == "SH"
    assert config.boundary_Br_projection_basis == "SH"
    assert config.conductance_projection_basis == "SH"
    assert config.u_projection_basis == "CS"
    assert config.Q_eff_projection_basis == "CS"
    assert config.E_neutral_wind_projection_basis == "SH"


def test_simulation_config_cs_mode_requires_cs_projection_routes():
    """CS horizontal mode cannot mix in SH input routes."""
    config = SimulationConfig(horizontal_basis_kind="cs")

    assert config.horizontal_basis_kind == "CS"
    assert config.boundary_jr_projection_basis == "CS"
    assert config.boundary_Br_projection_basis == "CS"
    assert config.conductance_projection_basis == "CS"
    assert config.u_projection_basis == "CS"
    assert config.Q_eff_projection_basis == "CS"
    assert config.E_neutral_wind_projection_basis == "CS"

    with pytest.raises(ValueError, match="boundary_jr_projection_basis"):
        SimulationConfig(horizontal_basis_kind="CS", boundary_jr_projection_basis="SH")


def test_simulation_config_dataset_roundtrip_preserves_stored_sentinels():
    """The settings dataset is the canonical serialized config."""
    config = SimulationConfig(
        Nmax=3,
        Mmax=2,
        Ncs=4,
        RM=None,
        magnetic_boundary_shielding=False,
        main_field_B0=None,
        enable_pfac_coupling=False,
        area_weighted_least_squares=True,
        least_squares_solver="lsmr",
        least_squares_preconditioner="pinv",
        reuse_preconditioner=True,
        toroidal_potential_regularization_lambda=1e-3,
    )

    settings = config.to_dataset()
    restored = SimulationConfig.from_settings(settings)

    assert settings.attrs["RM"] == 0
    assert settings.attrs["main_field_B0"] == 0
    assert setting_value(settings, "enable_pfac_coupling") == 0
    assert restored.RM is None
    assert not restored.magnetic_boundary_shielding
    assert restored.main_field_B0 is None
    assert not restored.enable_pfac_coupling
    assert restored.area_weighted_least_squares
    assert restored.least_squares_solver == "lsmr"
    assert restored.least_squares_preconditioner == "pinv"
    assert restored.reuse_preconditioner
    assert restored.toroidal_potential_regularization_lambda == pytest.approx(1e-3)
    np.testing.assert_allclose(restored.fac_integration_radii, config.fac_integration_radii)


def test_simulation_config_from_minimal_settings_accepts_missing_defaults():
    """Minimal settings are expanded through the same default path."""
    settings = xr.Dataset(attrs={"Nmax": 3, "Mmax": 2, "Ncs": 4})

    config = SimulationConfig.from_settings(settings)

    assert config.Nmax == 3
    assert config.Mmax == 2
    assert config.least_squares_preconditioner is None
    assert config.Ncs == 4
    assert config.horizontal_basis_kind == "SH"
    assert config.conductance_projection_basis == "SH"


def test_simulation_config_preserves_decimal_main_field_epoch():
    """Main-field epochs can be decimal years."""
    config = SimulationConfig(main_field_epoch=2011.813014015728)
    restored = SimulationConfig.from_settings(config.to_dataset())

    assert config.main_field_epoch == pytest.approx(2011.813014015728)
    assert restored.main_field_epoch == pytest.approx(config.main_field_epoch)


def test_simulation_config_derives_main_field_epoch_from_start_time():
    """The physical event time supplies the default field epoch."""
    config = SimulationConfig(t0="2001-05-12 21:45:00")

    assert config.main_field_epoch == pytest.approx(
        decimal_year(dt.datetime.fromisoformat(config.t0))
    )
    assert config.to_dataset().attrs["main_field_epoch"] == pytest.approx(config.main_field_epoch)

    explicit = SimulationConfig(t0=config.t0, main_field_epoch=2000.0)
    assert explicit.main_field_epoch == pytest.approx(2000.0)


def test_simulation_config_normalizes_start_time_to_utc():
    """Equivalent start-time spellings share one persisted identity."""
    config = SimulationConfig(t0="2020-01-01T01:30:00+01:00")

    assert config.t0 == "2020-01-01 00:30:00"
    with pytest.raises(ValueError, match="t0"):
        SimulationConfig(t0="not a time")


def test_simulation_config_normalizes_and_validates_main_field_kind():
    """Main-field model names are canonical persisted configuration."""
    config = SimulationConfig(main_field_kind=" IGRF ")
    assert config.main_field_kind == "igrf"
    assert config.horizontal_coordinate_system == "geocentric_geographic"
    assert config.to_dataset().attrs["horizontal_coordinate_system"] == "geocentric_geographic"

    dipole_config = SimulationConfig(main_field_kind="dipole")
    assert dipole_config.horizontal_coordinate_system == "centered_dipole"

    with pytest.raises(ValueError, match="main_field_kind"):
        SimulationConfig(main_field_kind="unknown")


def test_simulation_config_rejects_stale_persisted_coordinate_frame():
    """The persisted frame cannot disagree with its main-field kind."""
    settings = SimulationConfig(main_field_kind="dipole").to_dataset()
    settings.attrs["horizontal_coordinate_system"] = "geocentric_geographic"

    with pytest.raises(ValueError, match="does not match main_field_kind"):
        SimulationConfig.from_settings(settings)


def test_simulation_config_normalizes_executable_policies():
    """Canonicalize algorithms before building runtime objects."""
    config = SimulationConfig(
        integrator=" dop853 ", least_squares_solver=" LSMR ", least_squares_preconditioner=" NONE "
    )

    assert config.integrator == "DOP853"
    assert config.least_squares_solver == "lsmr"
    assert config.least_squares_preconditioner is None
    assert config.to_dataset().attrs["least_squares_preconditioner"] == "none"
    assert SimulationConfig.from_settings(config.to_dataset()).least_squares_preconditioner is None


def test_simulation_config_parses_persisted_boolean_values_explicitly():
    """String booleans do not inherit Python's truthiness semantics."""
    config = SimulationConfig(enable_pfac_coupling="false", enable_interhemispheric_coupling="yes")

    assert not config.enable_pfac_coupling
    assert config.enable_interhemispheric_coupling

    with pytest.raises(ValueError, match="save_equilibria"):
        SimulationConfig(save_equilibria="sometimes")


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
    assert config.boundary_jr_projection_basis == "CS"


def test_simulation_config_enforces_radial_boundary_invariants():
    """Outer-boundary options and FAC samples share a radial domain."""
    with pytest.raises(ValueError, match="magnetic_boundary_shielding requires"):
        SimulationConfig(magnetic_boundary_shielding=True)
    with pytest.raises(ValueError, match="greater than RI"):
        SimulationConfig(RI=7.0e6, RM=6.9e6)
    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationConfig(fac_integration_radii=[7.0e6, 7.1e6, 7.05e6])


def test_simulation_config_owns_immutable_fac_integration_radii():
    """Frozen configuration owns its FAC radial grid."""
    source = np.array([7.0e6, 7.1e6, 7.2e6])
    config = SimulationConfig(fac_integration_radii=source)
    source[0] = 8.0e6

    assert config.fac_integration_radii[0] == pytest.approx(7.0e6)
    with pytest.raises(ValueError, match="read-only"):
        config.fac_integration_radii[0] = 8.0e6


def test_simulation_config_defaults_fac_grid_to_outer_boundary():
    """An RM run receives a compatible default FAC integration grid."""
    config = SimulationConfig(RI=7.0e6, RM=8.0e6)

    assert config.fac_integration_radii[0] == pytest.approx(config.RI)
    assert config.fac_integration_radii[-1] == pytest.approx(config.RM)


def test_dipole_fac_integration_radii_use_uniform_magnetic_latitude():
    """Dipole sampling should include both radial endpoints."""
    inner_radius = 7.0e6
    outer_radius = 9.0e6

    radii = dipole_fac_integration_radii(inner_radius, outer_radius, n_points=5)

    np.testing.assert_allclose(radii[[0, -1]], [inner_radius, outer_radius])
    magnetic_latitude = np.arccos(np.sqrt(inner_radius / radii))
    np.testing.assert_allclose(np.diff(magnetic_latitude), np.diff(magnetic_latitude)[0])


@pytest.mark.parametrize("n_points", [1, 2.5, np.nan])
def test_dipole_fac_integration_radii_require_an_integer_point_count(n_points):
    """FAC integration requires an integral number of points."""
    with pytest.raises(ValueError, match="integer points"):
        dipole_fac_integration_radii(7.0e6, 9.0e6, n_points)


def test_simulation_config_derives_missing_fac_grid_from_loaded_radii():
    """Legacy settings derive radial samples from their own domain."""
    settings = xr.Dataset(attrs={"Nmax": 3, "Mmax": 2, "Ncs": 4, "RI": 7.0e6, "RM": 8.0e6})

    config = SimulationConfig.from_settings(settings)

    assert config.fac_integration_radii[0] == pytest.approx(7.0e6)
    assert config.fac_integration_radii[-1] == pytest.approx(8.0e6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"Nmax": 0}, "Nmax"),
        ({"Mmax": -1}, "Mmax"),
        ({"Nmax": 2, "Mmax": 3}, "Mmax"),
        ({"Ncs": 3}, "Ncs must be even"),
        ({"RI": EARTH_RADIUS_M}, "reference radius"),
        ({"interhemispheric_coupling_latitude": 91}, "interhemispheric_coupling_latitude"),
        (
            {"interhemispheric_electric_field_weight": np.inf},
            "interhemispheric_electric_field_weight",
        ),
        ({"main_field_epoch": np.nan}, "main_field_epoch"),
        ({"main_field_B0": 0.0}, "main_field_B0"),
        ({"main_field_kind": "igrf", "main_field_B0": 3e-5}, "main_field_B0"),
        ({"integrator": "leapfrog"}, "integrator"),
        ({"least_squares_solver": "inverse"}, "least_squares_solver"),
        ({"least_squares_preconditioner": "ilu"}, "least_squares_preconditioner"),
        (
            {"toroidal_potential_regularization_lambda": np.nan},
            "toroidal_potential_regularization_lambda",
        ),
    ],
)
def test_simulation_config_rejects_invalid_space_and_physical_settings(kwargs, message):
    """Reject values that cannot define the numerical domain."""
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**kwargs)
