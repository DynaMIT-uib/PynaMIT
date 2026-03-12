from dataclasses import dataclass
from pathlib import Path

import pytest

from pynamit.math.constants import RE
from pynamit.primitives.io import IO
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.settings import (
    ArtifactStorageKind,
    ConductanceInterpolationMode,
    DynamicsMode,
    DynamicsSettings,
    IntegratorKind,
    LLConstraintMode,
    MainfieldKind,
    SimulationMode,
    SolutionBasisKind,
    WeightingMode,
)


def _artifact_path(run_directory: str, name: str, storage_kind: str) -> Path:
    suffix = ".zarr" if storage_kind == "zarr" else ".ncdf"
    return Path(run_directory) / f"{name}{suffix}"


def test_dynamics_settings_default_conductance_mode_is_string() -> None:
    settings = DynamicsSettings()

    assert (
        settings.conductance_interpolation_mode == ConductanceInterpolationMode.LEGACY_ETA_LINEAR
    )
    assert isinstance(settings.conductance_interpolation_mode, str)
    assert settings.mainfield_kind == MainfieldKind.DIPOLE
    assert settings.integrator == IntegratorKind.EULER
    assert settings.dynamics_mode == DynamicsMode.LEGACY
    assert settings.apply_m_imp_gauge is True
    assert settings.ll_constraint_mode == LLConstraintMode.AUTO
    assert settings.run_directory is None
    assert settings.artifact_storage == ArtifactStorageKind.AUTO


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
    integrator: IntegratorKind = IntegratorKind.EULER
    m_imp_regularization_lambda: float = 0.0
    ih_constraint_scaling: float = 1e-5
    ll_constraint_mode: LLConstraintMode = LLConstraintMode.AUTO
    simulation_mode: SimulationMode = SimulationMode.CS_DOMINANT
    dynamics_mode: DynamicsMode = DynamicsMode.FULL_INDUCTION
    apply_m_imp_gauge: bool = False


def test_dynamics_settings_coerce_applies_derived_defaults() -> None:
    settings = DynamicsSettings.coerce(PartialSettings())

    assert settings.simulation_mode == SimulationMode.CS_DOMINANT
    assert settings.solution_basis_kind == SolutionBasisKind.CS
    assert settings.RM is None
    assert settings.m_imp_regularization_lambda == 1e-4
    assert settings.toroidal_weighting == WeightingMode.QUADRATIC
    assert settings.poloidal_weighting == WeightingMode.QUADRATIC
    assert settings.toroidal_regularization_lambda == 1e-10
    assert settings.apply_m_imp_gauge is False


def test_dynamics_accepts_normalized_settings_object(tmp_path) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "settings_ctor"), Nmax=4, Mmax=3, Ncs=8
    )

    dynamics = Dynamics(settings, benchmark_mode=True)

    assert dynamics.settings.Nmax == settings.Nmax
    assert dynamics.run_directory == settings.run_directory


@pytest.mark.parametrize(
    "field_name,value,expected_fragment",
    [
        ("dynamics_mode", "ful_induction", "Did you mean 'full_induction'?"),
        ("integrator", "expotential", "Did you mean 'exponential'?"),
        ("mainfield_kind", "IGRFf", "Did you mean 'igrf'?"),
        ("conductance_interpolation_mode", "sigma_lgo", "Did you mean 'sigma_log'?"),
        ("artifact_storage", "zra", "Valid options: \\['auto', 'netcdf', 'zarr'\\]"),
        ("ll_constraint_mode", "sof", "Did you mean 'soft'?"),
    ],
)
def test_dynamics_settings_reject_invalid_string_choices(
    field_name: str, value: str, expected_fragment: str
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ValueError, match=expected_fragment):
        DynamicsSettings(**kwargs)


def test_dynamics_settings_normalizes_backend_bool_to_canonical_string() -> None:
    settings = DynamicsSettings(backend=True)

    assert settings.backend == "jax"


def test_dynamics_settings_accepts_scipy_integrator_name() -> None:
    settings = DynamicsSettings(integrator="DOP853")

    assert settings.integrator == "DOP853"


def test_dynamics_settings_roundtrips_induction_null_diagnostics() -> None:
    settings = DynamicsSettings(
        induction_null_diagnostics=True,
        induction_null_svd_rtol=1e-6,
        induction_null_warn_ratio=0.25,
    )

    restored = DynamicsSettings.from_dataset(settings.to_dataset(), defaults=DynamicsSettings())

    assert restored.induction_null_diagnostics is True
    assert restored.induction_null_svd_rtol == pytest.approx(1e-6)
    assert restored.induction_null_warn_ratio == pytest.approx(0.25)


def test_dynamics_settings_roundtrips_ll_constraint_mode() -> None:
    settings = DynamicsSettings(ll_constraint_mode="soft")

    restored = DynamicsSettings.from_dataset(settings.to_dataset(), defaults=DynamicsSettings())

    assert restored.ll_constraint_mode == LLConstraintMode.SOFT


def test_from_dataset_rejects_invalid_simulation_mode() -> None:
    ds = DynamicsSettings().to_dataset()
    ds.attrs["simulation_mode"] = "cs_domnant"

    with pytest.raises(ValueError, match="Did you mean 'cs_dominant'\\?"):
        DynamicsSettings.from_dataset(ds, defaults=DynamicsSettings())


def test_dynamics_uses_temporary_run_directory_when_no_run_directory() -> None:
    settings = DynamicsSettings(run_directory=None, Nmax=4, Mmax=3, Ncs=8)

    dynamics = Dynamics(settings, benchmark_mode=False)

    assert dynamics.uses_temporary_run_directory is True
    assert dynamics.run_directory is not None
    assert Path(dynamics.run_directory).name.startswith("pynamit-run-")
    settings_storage = dynamics.io.get_dataset_storage_kind("settings")
    pfac_storage = dynamics.io.get_dataset_storage_kind("PFAC_matrix")
    assert settings_storage == ("zarr" if IO.zarr_available() else "netcdf")
    assert pfac_storage == ("zarr" if IO.zarr_available() else "netcdf")
    assert _artifact_path(dynamics.run_directory, "settings", settings_storage).exists()
    assert _artifact_path(dynamics.run_directory, "PFAC_matrix", pfac_storage).exists()


def test_dynamics_respects_explicit_netcdf_artifact_storage(tmp_path) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "explicit_netcdf"),
        artifact_storage="netcdf",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    dynamics = Dynamics(settings, benchmark_mode=False)

    assert dynamics.io.get_dataset_storage_kind("settings") == "netcdf"
    assert dynamics.io.get_dataset_storage_kind("PFAC_matrix") == "netcdf"
    assert _artifact_path(dynamics.run_directory, "settings", "netcdf").exists()
    assert _artifact_path(dynamics.run_directory, "PFAC_matrix", "netcdf").exists()
