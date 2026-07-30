"""User-facing assembly of a PynaMIT simulation."""

import numpy as np

from pynamit.math.backend import set_backend
from pynamit.math.constants import RE
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.geometry import SimulationGeometry, build_main_field
from pynamit.simulation.inputs import InputPipeline
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.simulation.run_data import RunData
from pynamit.simulation.runner import SimulationRunner
from pynamit.storage import ArrayCache, ArtifactStore


class Simulation:
    """Configure, drive, evolve, and persist one coupled MIT simulation.

    Evolves the inductive magnetic response to field-aligned currents,
    neutral winds, and magnetic-boundary forcing in a coupled
    magnetosphere-ionosphere-thermosphere (MIT) model. Saves and loads
    simulation data to and from persisted xarray artifacts.

    Attributes
    ----------
    current_time : float
        Current simulation time in seconds.
    config : SimulationConfig
        Normalized immutable simulation configuration.
    run_data : RunData
        Persisted settings, schema, and input/output time series.
    geometry : SimulationGeometry
        Run-invariant spatial realization of the model equations.
    response : ElectrodynamicResponse
        Instantaneous forcing and electrodynamic response model.
    operator_cache : ArrayCache, optional
        Shared cache for deterministic materialized operators.
    """

    def __init__(
        self,
        run_directory=None,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=RE + 110.0e3,
        RM=None,
        magnetic_boundary_shielding=False,
        main_field_kind="dipole",
        main_field_epoch=2020,
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
        least_squares_preconditioner="pinv",
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
        run_directory : str, optional
            Preferred directory for this persisted run. Artifacts are
            written as fixed names like ``settings.zarr`` inside this
            directory.
        Nmax : int, optional
            Maximum spherical harmonic degree.
        Mmax : int, optional
            Maximum spherical harmonic order.
        Ncs : int, optional
            Number of cubed sphere grid points per edge.
        RI : float, optional
            Ionospheric radius in meters.
        RM : float, optional
            Magnetospheric boundary radius in meters.
        magnetic_boundary_shielding : bool, optional
            Whether induced fields are solved with a shielding condition
            at the magnetospheric boundary ``RM``.
        main_field_kind : {'dipole', 'kaiju_dipole', 'igrf', 'radial'}
            Type of main magnetic field model.
        main_field_epoch : float, optional
            Decimal year for main field model.
        main_field_B0 : float, optional
            Main field strength.
        fac_integration_radii : array-like, optional
            Integration radii for FAC poloidal field calculation.
        enable_pfac_coupling : bool, optional
            Whether field-aligned currents contribute their poloidal
            magnetic field to the coupled response.
        enable_interhemispheric_coupling : bool, optional
            Whether to impose conjugate current and electric-field
            constraints in the low-latitude coupling region.
        interhemispheric_coupling_latitude : float, optional
            Absolute magnetic latitude bounding that coupling region,
            in degrees.
        interhemispheric_electric_field_weight : float, optional
            Relative least-squares weight of the conjugate
            electric-field constraint.
        boundary_jr_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting radial-current inputs.
            Defaults to ``horizontal_basis_kind``.
        boundary_Br_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting radial magnetic-field
            inputs. Defaults to ``horizontal_basis_kind``.
        conductance_projection_basis : {'SH', 'CS'}, optional
            Basis used to store the dimensionless log conductance
            magnitude and log Hall/Pedersen ratio. ``'CS'`` makes
            matching model-grid inputs a no-op. Defaults to
            ``horizontal_basis_kind``.
        u_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting neutral-wind inputs.
            Defaults to ``horizontal_basis_kind``.
        Q_eff_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting effective wind-current
            inputs. Defaults to ``u_projection_basis``.
        E_neutral_wind_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting equivalent neutral-wind
            electric fields. Defaults to ``horizontal_basis_kind``.
        t0 : str, optional
            Start time in UTC format.
        save_equilibria : bool, optional
            Default for whether ``evolve_to_time`` calculates and saves
            instantaneous induction equilibria.
        integrator : {'euler', 'exponential', 'RK23', 'RK45', 'DOP853',
                      'Radau', 'BDF', 'LSODA'}, optional
            Integrator used for ``induced_Br`` evolution. SciPy method
            names are case-insensitive and stored in canonical form.
        least_squares_solver : str, optional
            Solver used for the toroidal-potential problem.
        least_squares_preconditioner : {'jacobi', 'pinv', None},
            optional
            Preconditioner used by iterative toroidal-potential solves.
        reuse_preconditioner : bool, optional
            Keep a reusable iterative-solver preconditioner when valid.
        toroidal_potential_regularization_lambda : float, optional
            Optional Tikhonov regularization strength for the private
            toroidal-potential solve. The default is unregularized;
            coefficient gauges are constrained separately.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred backend for new saved xarray artifacts. Existing
            artifacts keep their format on restart.
        operator_cache_directory : path-like, optional
            Shared content-addressed cache for expensive deterministic
            numerical arrays, including spherical-harmonic evaluation
            matrices. This runtime optimization is not part of the
            persisted physical configuration.
        backend : {'auto', 'numpy', 'jax', bool}, optional
            Array backend to use. ``"auto"`` respects the current global
            setting (environment variable or previous choice).
            ``"numpy"``/``False`` enforce NumPy arrays, while
            ``"jax"``/``True`` enables JAX.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Basis requested for horizontal surface potentials and
            operators. ``'SH'`` is the default. ``'CS'`` uses
            cubed-sphere nodal coefficients and finite-difference
            derivatives. ``induced_Br`` and radial continuation
            remain in the configured mean-free poloidal SH space.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for least-squares projections when
            no explicit ``sqrt_weights`` are supplied. Cubed-sphere
            grids use their native cell areas; ordinary spherical grids
            use ``sin(theta)``. Disabled by default to preserve the
            established projection norm.
        """
        set_backend(backend)
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
        self.operator_cache = (
            None if operator_cache_directory is None else ArrayCache(operator_cache_directory)
        )
        self.run_data = RunData.open(
            config,
            run_directory=run_directory,
            artifact_storage=artifact_storage,
            operator_cache=self.operator_cache,
            print_info=True,
        )
        self.config = self.run_data.config
        schema = self.run_data.schema
        main_field = build_main_field(self.config)

        self.geometry = SimulationGeometry(
            schema.horizontal_basis,
            schema.cs_basis,
            main_field,
            self.config,
            gap_Br_response_matrix=self.run_data.gap_Br_response,
            solid_harmonics=schema.solid_harmonics,
            operator_cache=self.operator_cache,
        )
        self.response = ElectrodynamicResponse(self.geometry, self.config)
        self._input_pipeline = InputPipeline(self)
        self._runner = SimulationRunner(self)

        if "dynamic" in self.run_data.output_series.datasets:
            current_output = self.run_data.output_series.datasets["dynamic"]
        else:
            current_output = self.run_data.output_series.datasets.get("equilibrium")
        self.current_time = (
            np.max(current_output.time.values) if current_output is not None else np.float64(0)
        )

        self.run_data.save_settings_if_missing(print_info=True)

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        run_directory=None,
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
    ):
        """Construct a simulation from a normalized configuration.

        The run directory, artifact storage, operator-cache directory,
        and backend are runtime preferences rather than persisted model
        settings.
        """
        if not isinstance(config, SimulationConfig):
            raise TypeError("Simulation.from_config requires a SimulationConfig.")
        return cls(
            run_directory=run_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
            **config.to_kwargs(),
        )

    @classmethod
    def from_directory(cls, run_directory, **kwargs):
        """Construct a simulation from one run directory."""
        run_directory = ArtifactStore.require_artifact_directory(run_directory, ("settings",))
        artifact_storage = kwargs.get("artifact_storage", "auto")
        settings = ArtifactStore(
            run_directory, preferred_dataset_storage=artifact_storage
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
        return cls.from_config(config, run_directory=run_directory, **runtime_kwargs)

    def evolve_to_time(
        self,
        t,
        dt=5e-4,
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
        equilibrium_initialization=True,
        run_dynamic=True,
        run_equilibrium=None,
    ):
        """Evolve the inductive solution to a specified time.

        Parameters
        ----------
        t : float
            Target time to evolve to in seconds.
        dt : float, optional
            Time step size in seconds.
        sampling_step_interval : int, optional
            Number of steps between samples.
        saving_sample_interval : int, optional
            Number of samples between saves.
        quiet : bool, optional
            Whether to suppress progress output.
        equilibrium_initialization : bool, optional
            Whether to initialize a new dynamic run from equilibrium.
        run_dynamic : bool, optional
            Whether to run and save the time-dependent inductive
            solution.
        run_equilibrium : bool, optional
            Whether to calculate and save the instantaneous equilibrium
            solution. Defaults to ``self.config.save_equilibria``.
        """
        return self._runner.evolve_to_time(
            t,
            dt=dt,
            sampling_step_interval=sampling_step_interval,
            saving_sample_interval=saving_sample_interval,
            quiet=quiet,
            equilibrium_initialization=equilibrium_initialization,
            run_dynamic=run_dynamic,
            run_equilibrium=run_equilibrium,
        )

    def impose_equilibrium(self, time=None, interpolation=True, save=True, quiet=False):
        """Solve the instantaneous induction equilibrium."""
        return self._runner.impose_equilibrium(
            time=time, interpolation=interpolation, save=save, quiet=quiet
        )

    def set_FAC(
        self,
        FAC,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set field-aligned current (FAC) input.

        Projects FAC onto the radial direction with the background-field
        direction cosine, then stores that radial current density.
        Positive FAC follows the background magnetic-field vector; it is
        not defined as upward in both hemispheres.

        Parameters
        ----------
        FAC : array-like
            Signed field-parallel current density in A/m². Positive
            values follow the background magnetic-field vector.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the FAC data.
        sqrt_weights : array-like, optional
            sqrt_weights for the FAC data points.
        reg_lambda : float, optional
            Regularization parameter for the least squares solver.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        radial_current = self._input_pipeline.radial_current_from_FAC(
            FAC, lat=lat, lon=lon, theta=theta, phi=phi
        )

        self.set_boundary_jr(
            radial_current,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_boundary_jr(
        self,
        boundary_jr=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        boundary_jr_coefficients=None,
    ):
        """Set radial current at the upper ionospheric boundary.

        Parameters
        ----------
        boundary_jr : array-like
            Radial current density in A/m².
        boundary_jr_coefficients : array-like, optional
            Radial-current coefficients in the input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the current data.
        sqrt_weights : array-like, optional
            sqrt_weights for the current data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_scalar_input(
            "boundary_jr",
            samples={"boundary_jr": boundary_jr},
            coefficients={"boundary_jr": boundary_jr_coefficients},
            sample_label="boundary_jr samples",
            coefficient_label="boundary_jr_coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_boundary_Br(
        self,
        boundary_Br=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        boundary_Br_coefficients=None,
    ):
        """Set radial magnetic field at the outer boundary.

        Parameters
        ----------
        boundary_Br : array-like
            Radial component of magnetic field.
        boundary_Br_coefficients : array-like, optional
            Radial magnetic-field coefficients in the input storage
            basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the magnetic-field data.
        sqrt_weights : array-like, optional
            sqrt_weights for the magnetic-field data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        if self.config.RM is None:
            raise ValueError("boundary_Br can only be set if magnetospheric radius (RM) is set.")

        self._input_pipeline.set_scalar_input(
            "boundary_Br",
            samples={"boundary_Br": boundary_Br},
            coefficients={"boundary_Br": boundary_Br_coefficients},
            sample_label="boundary_Br samples",
            coefficient_label="boundary_Br_coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_resistance(
        self,
        etaP=None,
        etaH=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set positive resistance through canonical coordinates.

        Parameters
        ----------
        etaP : array-like
            Strictly positive Pedersen resistance values.
        etaH : array-like
            Strictly positive Hall resistance values.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the resistance data.
        sqrt_weights : array-like, optional
            sqrt_weights for the resistance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.require_complete_values("resistance samples", etaP=etaP, etaH=etaH)
        log_magnitude, log_ratio = ionospheric_closure.resistance_to_log_conductance_coordinates(
            etaP, etaH
        )
        self._store_conductance_coordinates(
            log_magnitude,
            log_ratio,
            sample_label="resistance samples",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_conductance(
        self,
        hall=None,
        pedersen=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        log_magnitude_coefficients=None,
        log_ratio_coefficients=None,
    ):
        """Set positive conductance through canonical log coordinates.

        Sampled Pedersen/Hall values are converted to
        ``log(hypot(SigmaP, SigmaH) / 1 S)`` and
        ``log(SigmaH / SigmaP)`` before projection. This guarantees
        positive reconstructed conductance and treats the reciprocal
        resistance tensor symmetrically.

        Parameters
        ----------
        hall : array-like
            Hall conductance.
        pedersen : array-like
            Pedersen conductance.
        log_magnitude_coefficients, log_ratio_coefficients
            : array-like, optional
            Canonical conductance coordinates already represented in
            the configured input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the conductance data.
        sqrt_weights : array-like, optional
            sqrt_weights for the conductance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        coefficients_supplied = (
            log_magnitude_coefficients is not None or log_ratio_coefficients is not None
        )
        if coefficients_supplied:
            log_magnitude = None
            log_ratio = None
        else:
            self._input_pipeline.require_complete_values(
                "conductance samples", pedersen=pedersen, hall=hall
            )
            log_magnitude, log_ratio = ionospheric_closure.conductance_to_log_coordinates(
                pedersen, hall
            )

        self._store_conductance_coordinates(
            log_magnitude,
            log_ratio,
            log_magnitude_coefficients=log_magnitude_coefficients,
            log_ratio_coefficients=log_ratio_coefficients,
            sample_label="conductance samples",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def _store_conductance_coordinates(
        self,
        log_magnitude,
        log_ratio,
        *,
        log_magnitude_coefficients=None,
        log_ratio_coefficients=None,
        sample_label,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Store canonical conductance samples or coefficients."""
        self._input_pipeline.set_scalar_input(
            "conductance",
            samples={
                "log_conductance_magnitude": log_magnitude,
                "log_hall_to_pedersen_ratio": log_ratio,
            },
            coefficients={
                "log_conductance_magnitude": log_magnitude_coefficients,
                "log_hall_to_pedersen_ratio": log_ratio_coefficients,
            },
            sample_label=sample_label,
            coefficient_label="log-conductance coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_neutral_wind(
        self,
        u_theta=None,
        u_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        u_cf=None,
        u_df=None,
    ):
        """Set neutral wind velocities.

        Parameters
        ----------
        u_theta : array-like
            Meridional (south) wind velocity in m/s.
        u_phi : array-like
            Zonal (east) wind velocity in m/s.
        u_cf, u_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the wind data.
        sqrt_weights : array-like, optional
            sqrt_weights for the wind data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_tangential_input(
            "u",
            theta_component=u_theta,
            phi_component=u_phi,
            cf_coefficients=u_cf,
            df_coefficients=u_df,
            sample_label="wind samples",
            coefficient_label="wind coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_u(self, *args, **kwargs):
        """Set neutral wind velocities using the historical API name."""
        return self.set_neutral_wind(*args, **kwargs)

    def set_Q_eff(
        self,
        Q_eff_theta=None,
        Q_eff_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        Q_eff_cf=None,
        Q_eff_df=None,
    ):
        """Set effective wind-current input.

        ``Q_eff`` is a tangential JS proxy for the neutral
        wind ``u x B`` term. It is added to the JS before
        the conductance/resistance tensor maps currents to the electric
        field.

        Parameters
        ----------
        Q_eff_theta : array-like
            Southward effective-current component in A/m.
        Q_eff_phi : array-like
            Eastward effective-current component in A/m.
        Q_eff_cf, Q_eff_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the Q_eff data.
        sqrt_weights : array-like, optional
            sqrt_weights for the Q_eff data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_tangential_input(
            "Q_eff",
            theta_component=Q_eff_theta,
            phi_component=Q_eff_phi,
            cf_coefficients=Q_eff_cf,
            df_coefficients=Q_eff_df,
            sample_label="Q_eff samples",
            coefficient_label="Q_eff coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_E_neutral_wind(
        self,
        E_neutral_wind_theta=None,
        E_neutral_wind_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        E_neutral_wind_cf=None,
        E_neutral_wind_df=None,
    ):
        """Set an equivalent neutral-wind electric-field input.

        ``E_neutral_wind`` is added to the non-induced electric field
        before the imposed-current coupling is solved. It is an electric
        field in V/m, not an effective current. Use this route for
        externally prepared neutral-wind electrodynamics, such as
        separate Pedersen- and Hall-weighted winds. It is an alternative
        to ``set_neutral_wind`` and ``set_Q_eff``; use only one of the
        three representations.

        Parameters
        ----------
        E_neutral_wind_theta : array-like
            Southward neutral-wind electric-field component in V/m.
        E_neutral_wind_phi : array-like
            Eastward neutral-wind electric-field component in V/m.
        E_neutral_wind_cf, E_neutral_wind_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the neutral-wind electric-field data.
        sqrt_weights : array-like, optional
            Square-root weights for the neutral-wind electric-field
            samples.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_tangential_input(
            "E_neutral_wind",
            theta_component=E_neutral_wind_theta,
            phi_component=E_neutral_wind_phi,
            cf_coefficients=E_neutral_wind_cf,
            df_coefficients=E_neutral_wind_df,
            sample_label="neutral-wind electric-field samples",
            coefficient_label="neutral-wind electric-field coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_Q_eff_from_neutral_wind(
        self,
        u_theta,
        u_phi,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        wind_reg_lambda=None,
        Q_eff_reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Fit and store Q_eff coefficients from wind and conductance.

        The wind is projected once, then ``Q_eff`` is solved in its
        storage basis so that the resistance-weighted electric response
        matches the wind forcing. Use
        ``calculate_Q_eff_from_neutral_wind`` to inspect the equivalent
        model-grid field without storing it.
        """
        self._input_pipeline.require_no_exclusive_conflict("Q_eff")
        input_time, wind_coeff_rows = self._input_pipeline.project_tangential_samples(
            "u",
            u_theta,
            u_phi,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=wind_reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        q_coeff_rows = self._input_pipeline.fit_Q_eff_from_neutral_wind(
            input_time, wind_coeff_rows, reg_lambda=Q_eff_reg_lambda, pinv_rtol=pinv_rtol
        )
        self._input_pipeline.add_input_coefficients("Q_eff", {"Q_eff": q_coeff_rows}, input_time)

    def calculate_Q_eff_from_neutral_wind(
        self,
        u_theta,
        u_phi,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Return model-grid Q_eff equivalent to wind forcing."""
        if "conductance" not in self.run_data.input_series.datasets:
            raise RuntimeError(
                "Ionospheric resistance or conductance must be set before "
                "calculating Q_eff from wind."
            )

        input_time, wind_coeff_rows = self._input_pipeline.project_tangential_samples(
            "u",
            u_theta,
            u_phi,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        return self._input_pipeline.evaluate_Q_eff_from_neutral_wind(input_time, wind_coeff_rows)
