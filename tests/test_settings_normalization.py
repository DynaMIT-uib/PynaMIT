from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pynamit.math.constants import RE
from pynamit.primitives.io import IO
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.induction import (
    CurrentContinuityExteriorToroidalUpdateModel,
    EquivalentNonlocalRadialShellResponseModel,
    FrozenConductanceIncrementalKnownElectricRadialResponseModel,
    HarmonicShellElectricTraceModel,
    HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel,
    NonlocalShellElectricRadialResponseModel,
    PFACNonlocalRadialShellResponseModel,
    QTraceKnownSourceRadialResponseModel,
    ShellCurrentContinuityKnownElectricRadialResponseModel,
)
from pynamit.simulation.settings import (
    ArtifactStorageKind,
    ConductanceInterpolationMode,
    DynamicsMode,
    DynamicsSettings,
    ExponentialStepForm,
    IntegratorKind,
    LLConstraintMode,
    MainfieldKind,
    SimulationMode,
    SolutionBasisKind,
    StabilizationPolicy,
    WeightingMode,
)


def _artifact_path(run_directory: str, name: str, storage_kind: str) -> Path:
    suffix = ".zarr" if storage_kind == "zarr" else ".ncdf"
    return Path(run_directory) / f"{name}{suffix}"


def _set_unit_conductance(dynamics: Dynamics) -> None:
    grid = dynamics.state.geometry.grid
    hall = np.zeros(grid.size)
    pedersen = np.ones(grid.size)
    dynamics.set_conductance(hall, pedersen, lat=grid.lat, lon=grid.lon, time=None)
    dynamics.state.update(dynamics.input_manager, dynamics.current_time, interpolation=True)


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
    assert settings.stabilization_policy == StabilizationPolicy.AUTO
    assert settings.steady_state_regularization_lambda == pytest.approx(1e-10)
    assert settings.exponential_step_form == ExponentialStepForm.AFFINE


def test_legacy_connected_defaults_to_small_runtime_feedback_regularization() -> None:
    settings = DynamicsSettings(dynamics_mode="legacy", connect_hemispheres=True)

    assert settings.m_imp_regularization_lambda == pytest.approx(1e-10)


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


def test_dynamics_settings_roundtrips_stabilization_policy() -> None:
    settings = DynamicsSettings(
        stabilization_policy="regularized",
        steady_state_regularization_lambda=3e-9,
    )

    restored = DynamicsSettings.from_dataset(settings.to_dataset(), defaults=DynamicsSettings())

    assert restored.stabilization_policy == StabilizationPolicy.REGULARIZED
    assert restored.steady_state_regularization_lambda == pytest.approx(3e-9)


def test_dynamics_settings_does_not_serialize_removed_full_induction_mode_labels() -> None:
    attrs = DynamicsSettings().to_dataset().attrs

    assert "toroidal_closure_mode" not in attrs
    assert "radial_shell_forcing_mode" not in attrs


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("toroidal_closure_mode", "radial_shell"),
        ("radial_shell_forcing_mode", "frozen_conductance_incremental"),
    ],
)
def test_dynamics_settings_coerce_rejects_removed_full_induction_mode_overrides(
    field_name: str, value: str
) -> None:
    with pytest.raises(TypeError, match="Unknown DynamicsSettings override"):
        DynamicsSettings.coerce({}, **{field_name: value})


def test_dynamics_settings_roundtrips_exponential_step_form() -> None:
    settings = DynamicsSettings(exponential_step_form="centered")

    restored = DynamicsSettings.from_dataset(settings.to_dataset(), defaults=DynamicsSettings())

    assert restored.exponential_step_form == ExponentialStepForm.CENTERED


def test_dynamics_settings_normalizes_legacy_auto_exponential_step_form_to_affine() -> None:
    settings = DynamicsSettings(exponential_step_form="auto")

    assert settings.exponential_step_form == ExponentialStepForm.AFFINE

    ds = settings.to_dataset()
    ds.attrs["exponential_step_form"] = "auto"
    restored = DynamicsSettings.from_dataset(ds, defaults=DynamicsSettings())

    assert restored.exponential_step_form == ExponentialStepForm.AFFINE


def test_from_dataset_ignores_legacy_toroidal_closure_attrs() -> None:
    ds = DynamicsSettings().to_dataset()
    defaults = DynamicsSettings()

    for legacy_mode in ("tangential_full", "tangential_projected", "radial_shell_local"):
        ds.attrs["toroidal_closure_mode"] = legacy_mode
        restored = DynamicsSettings.from_dataset(ds, defaults=defaults)
        assert not hasattr(restored, "toroidal_closure_mode")


def test_from_dataset_ignores_legacy_radial_shell_forcing_attrs() -> None:
    ds = DynamicsSettings().to_dataset()
    defaults = DynamicsSettings()

    for legacy_mode in ("condensed_inductive", "pragmatic_homogeneous_rm_connector"):
        ds.attrs["radial_shell_forcing_mode"] = legacy_mode
        restored = DynamicsSettings.from_dataset(ds, defaults=defaults)
        assert not hasattr(restored, "radial_shell_forcing_mode")


@pytest.mark.parametrize(
    "forcing_mode",
    ["condensed_inductive", "pragmatic_homogeneous_rm_connector"],
)
def test_builtin_pfac_response_model_rejects_removed_forcing_modes(
    tmp_path, forcing_mode: str
) -> None:
    response_model = PFACNonlocalRadialShellResponseModel(forcing_mode=forcing_mode)
    settings = DynamicsSettings(
        run_directory=str(tmp_path / f"reject_removed_forcing_{forcing_mode}"),
        dynamics_mode="full_induction",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    with pytest.raises(ValueError, match="forcing_mode must be one of"):
        Dynamics(settings, benchmark_mode=True, radial_shell_response_model=response_model)


def test_full_induction_rejects_explicit_response_override_outside_benchmark(tmp_path) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "reject_response_override"),
        dynamics_mode="full_induction",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    with pytest.raises(
        ValueError, match="built-in canonical shell-gap response model"
    ):
        Dynamics(
            settings,
            benchmark_mode=False,
            radial_shell_response_model=EquivalentNonlocalRadialShellResponseModel(),
        )


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


def test_full_induction_radial_shell_closure_uses_builtin_pfac_response_model(tmp_path) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "radial_shell_builtin"),
        dynamics_mode="full_induction",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    dynamics = Dynamics(settings, benchmark_mode=True)

    assert isinstance(
        dynamics.state.radial_shell_response_model, PFACNonlocalRadialShellResponseModel
    )
    assert (
        "forcing_mode=frozen_conductance_incremental"
        in dynamics.state.radial_shell_response_model.description
    )
    _set_unit_conductance(dynamics)
    builtin_rhs = np.asarray(
        dynamics.state.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float
    )
    explicit_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    explicit_model.bind_state(dynamics.state)
    explicit_rhs = np.asarray(
        explicit_model.build_rhs_operator(dynamics.state.toroidal_matrices), dtype=float
    )
    np.testing.assert_allclose(builtin_rhs, explicit_rhs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "simulation_mode",
    ["pure_spectral", "spectral_transform_gl", "spectral_transform_cs"],
)
def test_canonical_radial_shell_forcing_mode_is_available_across_solution_modes(
    tmp_path, simulation_mode: str
) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / f"radial_shell_canonical_{simulation_mode}"),
        dynamics_mode="full_induction",
        simulation_mode=simulation_mode,
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )

    dynamics = Dynamics(settings, benchmark_mode=True)
    _set_unit_conductance(dynamics)

    rhs_op = np.asarray(dynamics.state.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float)
    feedback_op = np.asarray(
        dynamics.state.toroidal_matrices.full_radial_shell_feedback_dtalpha_operator, dtype=float
    )
    n = int(dynamics.state.solution_space.index_length)

    assert rhs_op.shape == (n, 2 * n)
    assert feedback_op.shape == (n, n)
    assert np.all(np.isfinite(rhs_op))
    assert np.all(np.isfinite(feedback_op))

def test_full_induction_radial_shell_closure_is_available_with_trace_based_response_model(
    tmp_path,
) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "radial_shell_trace_available"),
        dynamics_mode="full_induction",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )
    response_model = NonlocalShellElectricRadialResponseModel(
        shell_trace_model=HarmonicShellElectricTraceModel()
    )

    dynamics = Dynamics(
        settings,
        benchmark_mode=True,
        radial_shell_response_model=response_model,
    )

    assert str(dynamics.state.toroidal_closure_mode) == "radial_shell"
    assert (
        dynamics.state.radial_shell_response_model.shell_trace_model.__class__
        is HarmonicShellElectricTraceModel
    )
    rhs_op = np.asarray(dynamics.state.toroidal_matrices.full_radial_shell_rhs_from_E_operator)
    n = int(dynamics.state.solution_space.index_length)
    assert rhs_op.shape == (n, 2 * n)
    assert np.all(np.isfinite(rhs_op))


def test_full_induction_radial_shell_closure_accepts_equivalent_nonlocal_model(tmp_path) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "radial_shell_equivalent"),
        dynamics_mode="full_induction",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )
    response_model = EquivalentNonlocalRadialShellResponseModel()

    dynamics = Dynamics(
        settings,
        benchmark_mode=True,
        radial_shell_response_model=response_model,
    )

    assert dynamics.state.radial_shell_response_model is response_model
    assert isinstance(response_model.shell_response_model, EquivalentNonlocalRadialShellResponseModel)
    rhs_op = np.asarray(dynamics.state.toroidal_matrices.toroidal_rhs_from_E_operator, dtype=float)
    assert rhs_op.shape == (
        dynamics.state.solution_space.index_length,
        2 * dynamics.state.solution_space.index_length,
    )
    assert np.all(np.isfinite(rhs_op))


def test_full_induction_radial_shell_closure_accepts_nonlocal_shell_electric_model(
    tmp_path,
) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "radial_shell_nonlocal_shell_electric"),
        dynamics_mode="full_induction",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )
    response_model = NonlocalShellElectricRadialResponseModel()

    dynamics = Dynamics(
        settings,
        benchmark_mode=True,
        radial_shell_response_model=response_model,
    )

    assert dynamics.state.radial_shell_response_model is response_model


def test_full_induction_radial_shell_closure_accepts_explicit_shell_current_continuity_model(
    tmp_path,
) -> None:
    settings = DynamicsSettings(
        run_directory=str(tmp_path / "radial_shell_shell_current_continuity"),
        dynamics_mode="full_induction",
        simulation_mode="pure_spectral",
        Nmax=4,
        Mmax=3,
        Ncs=8,
    )
    response_model = NonlocalShellElectricRadialResponseModel(
        shell_response_model=FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    )

    dynamics = Dynamics(
        settings,
        benchmark_mode=True,
        radial_shell_response_model=response_model,
    )
    grid = dynamics.state.geometry.grid
    hall = np.zeros(grid.size)
    pedersen = np.ones(grid.size)
    dynamics.set_conductance(hall, pedersen, lat=grid.lat, lon=grid.lon, time=None)
    dynamics.state.update(dynamics.input_manager, dynamics.current_time, interpolation=True)

    assert dynamics.state.radial_shell_response_model is response_model
    rhs_op = np.asarray(dynamics.state.toroidal_matrices.full_radial_shell_rhs_from_E_operator)
    n = int(dynamics.state.solution_space.index_length)
    assert rhs_op.shape == (n, 2 * n)
    assert np.all(np.isfinite(rhs_op))


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
