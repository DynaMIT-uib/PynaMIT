"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

import numpy as np
from pynamit.math.constants import RE
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.simulation.config import SimulationConfig
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.sphere import Grid
from pynamit.primitives.io import IO
from pynamit.simulation.data import SimulationData
from pynamit.simulation.evolution import EvolutionRunner
from pynamit.simulation.inputs import InputProjector
from pynamit.simulation import induction, ionospheric_closure
from pynamit.simulation.mainfield import mainfield_from_config
from pynamit.simulation.state import State
from pynamit.math.backend import set_backend, to_numpy


class Dynamics:
    """Class for simulating dynamic MIT coupling.

    Manages the temporal evolution of the state of the ionosphere in
    response to field-aligned currents and neutral winds, giving rise to
    dynamic magnetosphere-ionosphere-thermosphere (MIT) coupling. Saves
    and loads simulation data to and from persisted xarray artifacts.

    Attributes
    ----------
    current_time : float
        Current simulation time in seconds.
    state : State
        Current state of the system.
    RI : float
        Radius of the ionosphere in meters.
    mainfield : Mainfield
        Main magnetic field model.
    """

    def __init__(
        self,
        run_directory=None,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=RE + 110.0e3,
        RM=None,
        RM_shielding=False,
        mainfield_kind="dipole",
        mainfield_epoch=2020,
        mainfield_B0=None,
        FAC_integration_steps=None,
        ignore_PFAC=False,
        connect_hemispheres=False,
        latitude_boundary=50,
        ih_constraint_scaling=1e-5,
        jr_projection_basis=None,
        Br_projection_basis=None,
        conductance_projection_basis=None,
        u_projection_basis=None,
        Q_eff_projection_basis=None,
        t0="2020-01-01 00:00:00",
        save_steady_states=True,
        integrator="euler",
        least_squares_solver=None,
        least_squares_preconditioner="pinv",
        static_preconditioner=False,
        m_imp_regularization_lambda=0.0,
        artifact_storage="auto",
        backend="auto",
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
    ):
        """Initialize the Dynamics class.

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
        RM_shielding : bool, optional
            Whether induced fields are solved with a shielding condition
            at the magnetospheric boundary ``RM``.
        mainfield_kind : {'dipole', 'kaiju_dipole', 'igrf', 'radial'}
            Type of main magnetic field model.
        mainfield_epoch : float, optional
            Decimal year for main field model.
        mainfield_B0 : float, optional
            Main field strength.
        FAC_integration_steps : array-like, optional
            Integration radii for FAC poloidal field calculation.
        ignore_PFAC : bool, optional
            Whether to ignore FAC poloidal fields.
        connect_hemispheres : bool, optional
            Whether hemispheres are electrically connected.
        latitude_boundary : float, optional
            Simulation boundary latitude in degrees.
        ih_constraint_scaling : float, optional
            Scaling for interhemispheric coupling constraint.
        jr_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting radial-current inputs.
            Defaults to ``horizontal_basis_kind``.
        Br_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting radial magnetic-field
            inputs. Defaults to ``horizontal_basis_kind``.
        conductance_projection_basis : {'SH', 'CS'}, optional
            Conductance storage/projection basis. ``'SH'`` stores fitted
            resistance coefficients. ``'CS'`` stores grid-basis
            resistance values; matching CS-grid inputs are a no-op.
            Defaults to ``horizontal_basis_kind``.
        u_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting neutral-wind inputs.
            Defaults to ``horizontal_basis_kind``.
        Q_eff_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting effective wind-current
            inputs. Defaults to ``u_projection_basis``.
        t0 : str, optional
            Start time in UTC format.
        save_steady_states : bool, optional
            Default for whether ``evolve_to_time`` calculates and saves
            steady states.
        integrator : {'euler', 'exponential'}, optional
            Integrator type for time evolution.
        least_squares_solver : str, optional
            Least-squares solver used by state feedback solves.
        least_squares_preconditioner : {'jacobi', 'pinv', None},
            optional
            Preconditioner used by iterative least-squares state solves.
        static_preconditioner : bool, optional
            Keep a reusable iterative-solver preconditioner when valid.
        m_imp_regularization_lambda : float, optional
            Regularization strength for imposed-potential solves.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred backend for new saved xarray artifacts. Existing
            artifacts keep their format on restart.
        backend : {'auto', 'numpy', 'jax', bool}, optional
            Array backend to use. ``"auto"`` respects the current global
            setting (environment variable or previous choice).
            ``"numpy"``/``False`` enforce NumPy arrays, while
            ``"jax"``/``True`` enables JAX.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Basis requested for horizontal state coefficients and
            surface operators. ``'SH'`` is the default. ``'CS'`` uses
            cubed-sphere nodal coefficients and finite-difference
            derivatives for horizontal surface operators. Radial
            Laplace-continuation terms use the SH radial-continuation
            basis.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for least-squares projections when
            no explicit ``sqrt_weights`` are supplied. Cubed-sphere
            grids use their native cell areas; ordinary spherical grids
            use ``sin(theta)``.
        """
        self.backend = set_backend(backend)
        config = SimulationConfig(
            Nmax=Nmax,
            Mmax=Mmax,
            Ncs=Ncs,
            RI=RI,
            RM=RM,
            RM_shielding=RM_shielding,
            latitude_boundary=latitude_boundary,
            ignore_PFAC=ignore_PFAC,
            connect_hemispheres=connect_hemispheres,
            FAC_integration_steps=FAC_integration_steps,
            ih_constraint_scaling=ih_constraint_scaling,
            mainfield_kind=mainfield_kind,
            mainfield_epoch=mainfield_epoch,
            mainfield_B0=mainfield_B0,
            jr_projection_basis=jr_projection_basis,
            Br_projection_basis=Br_projection_basis,
            conductance_projection_basis=conductance_projection_basis,
            u_projection_basis=u_projection_basis,
            Q_eff_projection_basis=Q_eff_projection_basis,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=area_weighted_least_squares,
            t0=t0,
            save_steady_states=save_steady_states,
            integrator=integrator,
            least_squares_solver=least_squares_solver,
            least_squares_preconditioner=least_squares_preconditioner,
            static_preconditioner=static_preconditioner,
            m_imp_regularization_lambda=m_imp_regularization_lambda,
        )
        self.data = SimulationData.create(
            config,
            run_directory=run_directory,
            artifact_storage=artifact_storage,
            print_info=True,
        )
        self.config = self.data.config
        self.settings = self.data.settings
        self.io = self.data.io
        self.run_directory = self.data.run_directory

        self.schema = self.data.schema
        self.cs_basis = self.schema.cs_basis
        self.sh_basis = self.schema.sh_basis
        self.sh_basis_mean_free = self.schema.sh_basis_mean_free
        self.horizontal_basis = self.schema.horizontal_basis
        self.solid_harmonics = self.schema.solid_harmonics

        self.input_vars = self.schema.input_vars
        self.input_field_spaces = self.schema.input_field_spaces
        self.input_timeseries = self.data.input_timeseries

        self.output_vars = self.schema.output_vars
        self.output_field_spaces = self.schema.output_field_spaces
        self.output_timeseries = self.data.output_timeseries

        self.input_projection_bases = self.schema.input_projection_bases

        input_grid = Grid(
            theta=self.cs_basis.arr_theta,
            phi=self.cs_basis.arr_phi,
            area_weights=self.cs_basis.unit_area,
        )
        input_transform_cache = {}
        self.input_transforms = {}
        for key in self.input_vars:
            representation = self.input_field_spaces[key].representation
            cache_key = getattr(
                representation,
                "signature",
                getattr(representation, "coefficient_space_signature", id(representation)),
            )
            if cache_key not in input_transform_cache:
                input_transform_cache[cache_key] = SphericalTransform(
                    representation,
                    input_grid,
                    grid_remap_basis=self.cs_basis,
                    area_weighted=self.config.area_weighted_least_squares,
                )
            self.input_transforms[key] = input_transform_cache[cache_key]
        self.input_projector = InputProjector(self)

        self.mainfield = mainfield_from_config(self.config)

        # Initialize the state of the ionosphere, restarting from the
        # last state checkpoint if available.
        self.state = State(
            self.horizontal_basis,
            self.mainfield,
            self.cs_basis,
            self.settings,
            PFAC_matrix=self.data.pfac_matrix,
            solid_harmonics=self.solid_harmonics,
        )
        self.horizontal_spherical_transform = self.state.geometry.spherical_transform
        self.evolution_runner = EvolutionRunner(self)

        if "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
        else:
            self.current_time = np.float64(0)

        self.data.save_settings_if_missing(print_info=True)
        self.data.save_pfac_matrix_if_missing(self.state.geometry.T_to_Ve, print_info=True)

    @classmethod
    def from_directory(cls, run_directory, **kwargs):
        """Construct a simulation from one run directory."""
        run_directory = IO.discover_run_directory(run_directory)
        artifact_storage = kwargs.get("artifact_storage", "auto")
        settings = IO(run_directory, preferred_dataset_storage=artifact_storage).load_dataset(
            "settings"
        )
        if settings is None:
            return cls(run_directory=run_directory, **kwargs)

        config_kwargs = SimulationConfig.from_settings(settings).to_kwargs()
        config_kwargs.update(
            {
                name: value
                for name, value in kwargs.items()
                if name in config_kwargs and value is not None
            }
        )
        extra_kwargs = {name: value for name, value in kwargs.items() if name not in config_kwargs}
        return cls(run_directory=run_directory, **config_kwargs, **extra_kwargs)

    def evolve_to_time(
        self,
        t,
        dt=np.float64(5e-4),
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
        steady_state_initialization=True,
        run_inductive=True,
        run_steady_state=None,
    ):
        """Evolve the system state to a specified time.

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
        steady_state_initialization : bool, optional
            Whether to initialize a new inductive run from steady state.
        run_inductive : bool, optional
            Whether to run and save the inductive time-dependent state.
        run_steady_state : bool, optional
            Whether to calculate and save the algebraic steady-state
            solution. Defaults to ``self.settings.save_steady_states``.
        """
        return self.evolution_runner.evolve_to_time(
            t,
            dt=dt,
            sampling_step_interval=sampling_step_interval,
            saving_sample_interval=saving_sample_interval,
            quiet=quiet,
            steady_state_initialization=steady_state_initialization,
            run_inductive=run_inductive,
            run_steady_state=run_steady_state,
        )

    def impose_steady_state(self, time=None, interpolation=True, save=True, quiet=False):
        """Replace the current model state with the steady state."""
        if time is not None:
            self.current_time = np.float64(time)

        self.state.update(self.input_timeseries, self.current_time, interpolation=interpolation)
        E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
        steady_state_m_ind = induction.steady_state_m_ind(self.state, E_coeffs_noind)

        if save:
            self.add_state_to_timeseries("state", steady_state_m_ind, E_coeffs_noind, m_imp_noind)
            if self.config.save_steady_states:
                self.add_state_to_timeseries(
                    "steady_state", steady_state_m_ind, E_coeffs_noind, m_imp_noind
                )

            self.data.save_output_dataset("state")
            if self.config.save_steady_states:
                self.data.save_output_dataset("steady_state")

            if not quiet:
                print(f"Imposed steady state at t = {float(self.current_time):.2f} s")

        return steady_state_m_ind

    def add_state_to_timeseries(self, key, m_ind, E_coeffs_noind, m_imp_noind):
        """Add the current state to the time series.

        Parameters
        ----------
        key : str
            Key for the time series entry.
        m_ind : array-like
            Inductive magnetic field coefficients.
        E_coeffs_noind : tuple
            Electric field coefficients without induced effects.
        m_imp_noind : array-like
            Imposed magnetic field coefficients without induced effects.
        """
        m_ind = self.state.project_scalar_mean_free(m_ind)
        E_coeffs_ind, m_imp_ind = self.state.calculate_ind_coeffs(m_ind)

        E_coeffs = self.state.project_helmholtz_mean_free(E_coeffs_noind + E_coeffs_ind)
        m_imp = self.state.project_scalar_mean_free(m_imp_noind + m_imp_ind)

        # Append current state to time series.
        state_data = {
            "m_ind": to_numpy(m_ind),
            "m_imp": to_numpy(m_imp),
            "Phi": to_numpy(
                self.state.geometry.helmholtz_curl_free_potential_operator.matvec(E_coeffs)
            ),
            "W": to_numpy(
                self.state.geometry.helmholtz_divergence_free_potential_operator.matvec(E_coeffs)
            ),
        }

        self.data.add_output_entry(key, state_data, time=self.current_time)

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

        Converts FAC to radial current density by multiplying with the
        radial component of the main field, and sets the radial current
        density as input.

        Parameters
        ----------
        FAC : array-like
            Field-aligned current density in A/m².
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
        FAC_b_evaluator = FieldEvaluator(
            self.mainfield, Grid(lat=lat, lon=lon, theta=theta, phi=phi), self.config.RI
        )

        self.set_jr(
            FAC * FAC_b_evaluator.br,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_jr(
        self,
        jr=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        jr_coefficients=None,
    ):
        """Set radial current density input.

        Parameters
        ----------
        jr : array-like
            Radial current density in A/m².
        jr_coefficients : array-like, optional
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
        self.input_projector.set_scalar_input(
            "jr",
            samples={"jr": jr},
            coefficients={"jr": jr_coefficients},
            sample_label="jr samples",
            coefficient_label="jr_coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_Br(
        self,
        Br=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        Br_coefficients=None,
    ):
        """Set radial component of magnetic field input.

        Parameters
        ----------
        Br : array-like
            Radial component of magnetic field.
        Br_coefficients : array-like, optional
            Radial magnetic-field coefficients in the input storage
            basis.
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
        if self.config.RM is None:
            raise ValueError("Br can only be set if magnetospheric radius (RM) is set.")

        self.input_projector.set_scalar_input(
            "Br",
            samples={"Br": Br},
            coefficients={"Br": Br_coefficients},
            sample_label="Br samples",
            coefficient_label="Br_coefficients",
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
        *,
        etaP_coefficients=None,
        etaH_coefficients=None,
    ):
        """Set etaP and etaH resistance inputs.

        Parameters
        ----------
        etaP : array-like
            Pedersen resistance values, inverse-conductance tensor
            component.
        etaH : array-like
            Hall resistance values, inverse-conductance tensor
            component.
        etaP_coefficients, etaH_coefficients : array-like, optional
            Resistance coefficients in the input storage basis.
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
        self.input_projector.set_scalar_input(
            "conductance",
            samples={"etaP": etaP, "etaH": etaH},
            coefficients={"etaP": etaP_coefficients, "etaH": etaH_coefficients},
            sample_label="resistance samples",
            coefficient_label="resistance coefficients",
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
        Hall,
        Pedersen,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set Hall and Pedersen conductance values.

        Direct coefficient storage uses ``set_resistance`` with
        ``etaP_coefficients`` and ``etaH_coefficients``.
        Conductance values are converted pointwise to the stored
        resistance variables before projection.

        Parameters
        ----------
        Hall : array-like
            Hall conductance.
        Pedersen : array-like
            Pedersen conductance.
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
        etaP, etaH = ionospheric_closure.conductance_to_resistance(Hall, Pedersen)

        self.set_resistance(
            etaP,
            etaH,
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
        self.input_projector.set_tangential_input(
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
        self.input_projector.set_tangential_input(
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

    def set_E_source(
        self,
        E_source_theta=None,
        E_source_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        E_source_cf=None,
        E_source_df=None,
    ):
        """Set a direct electric-field source input.

        ``E_source`` is added to the non-induced electric field before
        the imposed-current coupling is solved. It is an electric field
        source in V/m, not an effective current, so it must not be
        passed through ``set_Q_eff``.

        Parameters
        ----------
        E_source_theta : array-like
            Southward electric-field source component in V/m.
        E_source_phi : array-like
            Eastward electric-field source component in V/m.
        E_source_cf, E_source_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the E-source data.
        sqrt_weights : array-like, optional
            sqrt_weights for the E-source data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self.input_projector.set_tangential_input(
            "E_source",
            theta_component=E_source_theta,
            phi_component=E_source_phi,
            cf_coefficients=E_source_cf,
            df_coefficients=E_source_df,
            sample_label="E_source samples",
            coefficient_label="E_source coefficients",
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
        *,
        fit_coefficients=True,
    ):
        """Compute and store Q_eff from neutral wind and conductance."""
        self.input_projector.require_no_exclusive_conflict("Q_eff")
        if fit_coefficients:
            input_time, wind_coeff_rows = self.input_projector.project_tangential_samples(
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
            q_coeff_rows = ionospheric_closure.fit_Q_eff_coefficients(
                self.state,
                self.input_timeseries,
                self.input_field_spaces["Q_eff"],
                input_time,
                wind_coeff_rows,
                reg_lambda=Q_eff_reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            self.input_projector.add_input_coefficients(
                "Q_eff", {"Q_eff": q_coeff_rows}, input_time
            )
            return

        Q_eff_theta, Q_eff_phi, Q_lat, Q_lon = self.calculate_Q_eff_from_neutral_wind(
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
        self.set_Q_eff(
            Q_eff_theta=Q_eff_theta,
            Q_eff_phi=Q_eff_phi,
            lat=Q_lat,
            lon=Q_lon,
            time=time,
            reg_lambda=Q_eff_reg_lambda,
            pinv_rtol=pinv_rtol,
        )

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
        if "conductance" not in self.input_timeseries.datasets:
            raise RuntimeError("Conductance must be set before calculating Q_eff from wind.")

        input_time, wind_coeff_rows = self.input_projector.project_tangential_samples(
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
        return ionospheric_closure.Q_eff_from_neutral_wind(
            self.state,
            self.input_timeseries,
            self.input_field_spaces["u"].representation,
            input_time,
            wind_coeff_rows,
        )
