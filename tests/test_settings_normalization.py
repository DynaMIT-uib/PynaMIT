from dataclasses import dataclass
from pathlib import Path

from pynamit.math.constants import RE
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.settings import DynamicsSettings, SimulationMode


def test_dynamics_settings_default_conductance_mode_is_string() -> None:
    settings = DynamicsSettings()

    assert settings.conductance_interpolation_mode == "legacy_eta_linear"
    assert isinstance(settings.conductance_interpolation_mode, str)
    assert settings.apply_m_imp_gauge is True
    assert settings.run_directory is None


@dataclass
class PartialSettings:
    Nmax: int = 5
    Mmax: int = 5
    Ncs: int = 10
    RI: float = RE + 110.0e3
    RM: float = 0.0
    connect_hemispheres: bool = False
    latitude_boundary: float = 50.0
    ignore_PFAC: bool = True
    FAC_integration_steps: object = None
    least_squares_solver: str = "lsmr"
    least_squares_preconditioner: str = "pinv"
    integrator: str = "euler"
    m_imp_regularization_lambda: float = 0.0
    ih_constraint_scaling: float = 1e-5
    simulation_mode: SimulationMode = SimulationMode.CS_DOMINANT
    dynamics_mode: str = "full_induction"
    apply_m_imp_gauge: bool = False


def test_dynamics_settings_coerce_applies_derived_defaults() -> None:
    settings = DynamicsSettings.coerce(PartialSettings())

    assert settings.simulation_mode == SimulationMode.CS_DOMINANT
    assert settings.solution_basis_kind == "CS"
    assert settings.RM is None
    assert settings.m_imp_regularization_lambda == 1e-4
    assert settings.toroidal_weighting == "quadratic"
    assert settings.poloidal_weighting == "quadratic"
    assert settings.toroidal_regularization_lambda == 1e-10
    assert settings.apply_m_imp_gauge is False


def test_dynamics_accepts_normalized_settings_object(tmp_path) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "settings_ctor"),
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    dynamics = Dynamics(settings, benchmark_mode=True)

    assert dynamics.settings.Nmax == settings.Nmax
    assert dynamics.run_directory == settings.run_directory


def test_dynamics_uses_temporary_run_directory_when_no_run_directory() -> None:
    settings = DynamicsSettings(
        run_directory=None,
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    dynamics = Dynamics(settings, benchmark_mode=False)

    assert dynamics.uses_temporary_run_directory is True
    assert dynamics.run_directory is not None
    assert Path(dynamics.run_directory).name.startswith("pynamit-run-")
    assert (Path(dynamics.run_directory) / "settings.ncdf").exists()
    assert (Path(dynamics.run_directory) / "PFAC_matrix.ncdf").exists()
