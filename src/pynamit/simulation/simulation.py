"""Configure and evolve a coupled PynaMIT simulation."""

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.evolution import (
    DEFAULT_DT_SECONDS,
    DEFAULT_SAMPLES_PER_WRITE,
    DEFAULT_STEPS_PER_SAMPLE,
    _TimeEvolution,
)
from pynamit.simulation.input_preparation import InputPreparation
from pynamit.storage import ArtifactStore


class Simulation(InputPreparation):
    """Configure, drive, evolve, and persist one coupled MIT simulation.

    A simulation supports the same input setters as
    :class:`InputPreparation`, then adds the electrodynamic response and
    time evolution.
    """

    def __init__(
        self,
        simulation_directory=None,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=EARTH_RADIUS_M + 110.0e3,
        RM=None,
        magnetic_boundary_shielding=False,
        main_field_kind="dipole",
        main_field_epoch=None,
        main_field_B0=None,
        fac_integration_radii=None,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=False,
        interhemispheric_coupling_latitude=50,
        interhemispheric_electric_field_weight=1e-5,
        boundary_jr_projection_basis=None,
        boundary_Br_projection_basis=None,
        conductance_projection_basis=None,
        u_projection_basis=None,
        Q_eff_projection_basis=None,
        E_neutral_wind_projection_basis=None,
        t0="2020-01-01 00:00:00",
        save_equilibria=True,
        integrator="euler",
        least_squares_solver=None,
        least_squares_preconditioner=None,
        reuse_preconditioner=False,
        toroidal_potential_regularization_lambda=0.0,
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
    ):
        """Initialize a coupled MIT simulation.

        Parameters
        ----------
        simulation_directory : path-like, optional
            Directory for persisted settings, inputs, and outputs.
        Nmax, Mmax, Ncs : int, optional
            Spherical-harmonic truncation and cubed-sphere resolution.
        RI, RM : float, optional
            Ionospheric and magnetospheric radii in meters.
        magnetic_boundary_shielding : bool, optional
            Impose a shielding condition at ``RM``.
        main_field_kind, main_field_epoch, main_field_B0 : optional
            Main-field model, decimal-year epoch, and optional field
            magnitude override.
        fac_integration_radii : array-like, optional
            Radii used to integrate the FAC poloidal field.
        enable_pfac_coupling : bool, optional
            Include the FAC poloidal field in the coupled response.
        enable_interhemispheric_coupling : bool, optional
            Couple conjugate current and electric-field solutions.
        interhemispheric_coupling_latitude : float, optional
            Absolute latitude bounding the coupled low-latitude region.
        interhemispheric_electric_field_weight : float, optional
            Relative least-squares weight of the conjugate constraint.
        boundary_jr_projection_basis, boundary_Br_projection_basis,
        conductance_projection_basis, u_projection_basis,
        Q_eff_projection_basis, E_neutral_wind_projection_basis :
            Input storage bases. Each defaults to the corresponding
            choice derived by :class:`SimulationConfig`.
        t0 : str, optional
            Physical start time.
        save_equilibria : bool, optional
            Save instantaneous induction equilibria by default.
        integrator : str, optional
            Integrator used for ``induced_Br`` evolution.
        least_squares_solver, least_squares_preconditioner : optional
            Toroidal-potential solver and preconditioner.
        reuse_preconditioner : bool, optional
            Reuse a compatible iterative-solver preconditioner.
        toroidal_potential_regularization_lambda : float, optional
            Regularization strength for the toroidal-potential solve.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred storage backend for newly saved artifacts.
        operator_cache_directory : path-like, optional
            Shared cache for deterministic numerical operators.
        backend : {'auto', 'numpy', 'jax', bool}, optional
            Array backend.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Horizontal surface basis.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for projections without explicit
            weights.
        """
        config = SimulationConfig(
            Nmax=Nmax,
            Mmax=Mmax,
            Ncs=Ncs,
            RI=RI,
            RM=RM,
            magnetic_boundary_shielding=magnetic_boundary_shielding,
            interhemispheric_coupling_latitude=interhemispheric_coupling_latitude,
            enable_pfac_coupling=enable_pfac_coupling,
            enable_interhemispheric_coupling=enable_interhemispheric_coupling,
            fac_integration_radii=fac_integration_radii,
            interhemispheric_electric_field_weight=interhemispheric_electric_field_weight,
            main_field_kind=main_field_kind,
            main_field_epoch=main_field_epoch,
            main_field_B0=main_field_B0,
            boundary_jr_projection_basis=boundary_jr_projection_basis,
            boundary_Br_projection_basis=boundary_Br_projection_basis,
            conductance_projection_basis=conductance_projection_basis,
            u_projection_basis=u_projection_basis,
            Q_eff_projection_basis=Q_eff_projection_basis,
            E_neutral_wind_projection_basis=E_neutral_wind_projection_basis,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=area_weighted_least_squares,
            t0=t0,
            save_equilibria=save_equilibria,
            integrator=integrator,
            least_squares_solver=least_squares_solver,
            least_squares_preconditioner=least_squares_preconditioner,
            reuse_preconditioner=reuse_preconditioner,
            toroidal_potential_regularization_lambda=toroidal_potential_regularization_lambda,
        )
        self._open_input_preparation(
            config,
            directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
        )
        self._open_simulation_runtime()

    def _open_simulation_runtime(self):
        """Initialize output and evolution state."""
        self.simulation_directory = self.data.simulation_directory
        self.outputs = self.data.output_series.datasets
        current_output = self.outputs.get("dynamic", self.outputs.get("equilibrium"))
        self.current_time = (
            np.max(current_output.time.values) if current_output is not None else np.float64(0)
        )
        self._time_evolution = _TimeEvolution(self)

    def __repr__(self):
        """Summarize the live simulation for interactive sessions."""
        inputs = ", ".join(sorted(self.inputs)) or "none"
        outputs = ", ".join(sorted(self.outputs)) or "none"
        return (
            f"Simulation(Nmax={self.config.Nmax}, Mmax={self.config.Mmax}, "
            f"Ncs={self.config.Ncs}, current_time={float(self.current_time):g}, "
            f"inputs=[{inputs}], outputs=[{outputs}], "
            f"simulation_directory={self.simulation_directory!r})"
        )

    @property
    def response(self):
        """Return the lazily constructed electrodynamic response."""
        return self._require_response()

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        simulation_directory=None,
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
    ):
        """Construct a simulation from a normalized configuration."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("Simulation.from_config requires a SimulationConfig.")
        simulation = cls.__new__(cls)
        simulation._open_input_preparation(
            config,
            directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
        )
        simulation._open_simulation_runtime()
        return simulation

    @classmethod
    def from_directory(cls, simulation_directory, **kwargs):
        """Construct a simulation from one simulation directory."""
        simulation_directory = ArtifactStore.require_artifact_directory(
            simulation_directory, ("settings",)
        )
        artifact_storage = kwargs.get("artifact_storage", "auto")
        settings = ArtifactStore(
            simulation_directory, preferred_dataset_storage=artifact_storage
        ).load_dataset("settings")

        stored_config = SimulationConfig.from_settings(settings)
        config_values = stored_config.to_kwargs()
        config_overrides = {
            name: value
            for name, value in kwargs.items()
            if name in config_values and value is not None
        }
        config = SimulationConfig(**{**config_values, **config_overrides})
        runtime_kwargs = {
            name: value for name, value in kwargs.items() if name not in config_values
        }
        return cls.from_config(config, simulation_directory=simulation_directory, **runtime_kwargs)

    def evolve_to_time(
        self,
        t,
        dt=DEFAULT_DT_SECONDS,
        steps_per_sample=DEFAULT_STEPS_PER_SAMPLE,
        samples_per_write=DEFAULT_SAMPLES_PER_WRITE,
        quiet=False,
        initialize_from_equilibrium=True,
        run_dynamic=True,
        run_equilibrium=None,
    ):
        """Evolve the inductive solution to ``t`` seconds after ``t0``.

        ``steps_per_sample`` controls how often output is
        retained; ``samples_per_write`` controls how many retained
        samples are accumulated between persistence writes.
        """
        return self._time_evolution.evolve_to_time(
            t,
            dt=dt,
            steps_per_sample=steps_per_sample,
            samples_per_write=samples_per_write,
            quiet=quiet,
            initialize_from_equilibrium=initialize_from_equilibrium,
            run_dynamic=run_dynamic,
            run_equilibrium=run_equilibrium,
        )

    def impose_equilibrium(self, time=None, interpolation=True, save=True, quiet=False):
        """Solve the instantaneous induction equilibrium."""
        return self._time_evolution.impose_equilibrium(
            time=time, interpolation=interpolation, save=save, quiet=quiet
        )


__all__ = ["Simulation"]
