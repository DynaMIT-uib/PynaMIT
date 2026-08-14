"""End-to-end workflow for PynaMIT's bundled example.

This module contains ``run_example``, which prepares the bundled example
inputs and runs a simulation. It is primarily used for testing and as a
starting point for simulation scripts.
"""

from pathlib import Path

from pynamit.simulation.runner import DEFAULT_DT_SECONDS
from pynamit.storage import ArtifactStore
from pynamit.workflows.example_inputs import prepare_example_inputs
from pynamit.workflows.prepared_inputs import run_from_inputs


def run_example(
    final_time=100,
    sampling_step_interval=1,
    write_sample_interval=200,
    dt=DEFAULT_DT_SECONDS,
    Nmax=20,
    Mmax=20,
    Ncs=30,
    RM=None,
    main_field_kind="dipole",
    main_field_epoch=None,
    main_field_B0=None,
    enable_pfac_coupling=False,
    enable_interhemispheric_coupling=False,
    interhemispheric_coupling_latitude=50,
    use_wind=False,
    use_Q_eff=False,
    use_boundary_jr=True,
    initialize_from_equilibrium=True,
    run_dynamic=True,
    run_equilibrium=True,
    boundary_jr_projection_basis=None,
    boundary_Br_projection_basis=None,
    conductance_projection_basis=None,
    u_projection_basis=None,
    Q_eff_projection_basis=None,
    integrator="euler",
    boundary_jr_lambda=None,
    conductance_lambda=None,
    u_lambda=None,
    Q_eff_lambda=None,
    multi_data=False,
    least_squares_solver=None,
    least_squares_preconditioner="pinv",
    reuse_preconditioner=False,
    toroidal_potential_regularization_lambda=0.0,
    simulation_directory=None,
    input_directory=None,
    artifact_storage="auto",
    horizontal_basis_kind="SH",
    area_weighted_least_squares=False,
    magnetic_boundary_shielding=False,
):
    """Prepare the bundled example inputs and run a PynaMIT simulation.

    Parameters
    ----------
    final_time : float, optional
        The final time of the simulation in seconds.
    sampling_step_interval : int, optional
        Number of integration steps between retained output samples.
    write_sample_interval : int, optional
        Number of output samples between persistence writes.
    dt : float, optional
        The time step for the simulation.
    Nmax : int, optional
        The maximum degree of the spherical harmonics.
    Mmax : int, optional
        The maximum order of the spherical harmonics.
    Ncs : int, optional
        The number of grid points in the cubed sphere grid.
    main_field_kind : str, optional
        The type of main field model.
    main_field_epoch : float, optional
        Decimal year used for the background-field coefficients.
        Defaults to the input event time.
    main_field_B0 : float, optional
        Optional background-field magnitude override in tesla.
    enable_pfac_coupling : bool, optional
        Whether field-aligned currents contribute their poloidal
        magnetic field to the coupled response.
    enable_interhemispheric_coupling : bool, optional
        Whether to impose conjugate current and electric-field
        constraints.
    interhemispheric_coupling_latitude : float, optional
        Absolute magnetic latitude bounding the low-latitude coupling
        region.
    use_wind : bool, optional
        Whether to include neutral-wind driving in the simulation.
    use_Q_eff : bool, optional
        Whether to represent neutral-wind driving through the effective
        current input Q_eff instead of direct wind forcing.
    use_boundary_jr : bool, optional
        Whether to include radial-current driving in the simulation.
    initialize_from_equilibrium : bool, optional
        Whether to initialize a new dynamic solution from equilibrium.
    run_dynamic : bool, optional
        Whether to run and save the time-dependent inductive solution.
    run_equilibrium : bool, optional
        Whether to calculate and save the instantaneous equilibrium
        solution.
    boundary_jr_projection_basis : {'SH', 'CS'}, optional
        Basis route used when projecting radial-current inputs. Defaults
        to ``horizontal_basis_kind``.
    boundary_Br_projection_basis : {'SH', 'CS'}, optional
        Basis route used when projecting radial magnetic-field inputs.
        Defaults to ``horizontal_basis_kind``.
    conductance_projection_basis : {'SH', 'CS'}, optional
        Basis used to store the dimensionless log conductance magnitude
        and log Hall/Pedersen ratio. ``'CS'`` makes matching model-grid
        inputs a no-op. Defaults to ``horizontal_basis_kind``.
    u_projection_basis : {'SH', 'CS'}, optional
        Basis route used when projecting neutral-wind inputs. Defaults
        to ``horizontal_basis_kind``.
    Q_eff_projection_basis : {'SH', 'CS'}, optional
        Basis route used when projecting effective wind-current inputs.
        Defaults to ``u_projection_basis``.
    integrator : {'euler', 'exponential', 'RK23', 'RK45', 'DOP853',
                  'Radau', 'BDF', 'LSODA'}, optional
        Integrator used for ``induced_Br`` evolution. SciPy method names
        are accepted case-insensitively and stored canonically.
    boundary_jr_lambda : float, optional
        Regularization parameter for the boundary radial current.
    conductance_lambda : float, optional
        Regularization parameter for the conductance.
    u_lambda : float, optional
        Regularization parameter for the wind.
    Q_eff_lambda : float, optional
        Regularization parameter for the effective wind current.
    least_squares_solver : str, optional
        Solver used for the toroidal-potential least-squares problem.
    least_squares_preconditioner : {'jacobi', 'pinv', None}, optional
        Preconditioner used by iterative toroidal-potential solves.
    reuse_preconditioner : bool, optional
        Keep a reusable iterative-solver preconditioner when valid.
    toroidal_potential_regularization_lambda : float, optional
        Regularization strength for toroidal-potential solves.
    simulation_directory : str, optional
        Directory for one persisted simulation. If omitted, a unique
        timestamped directory is created under ``simulation/``.
    input_directory : str, optional
        Directory for the prepared input package. Defaults to a
        ``prepared_inputs`` subdirectory in ``simulation_directory``.
    artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
        Preferred storage backend for new saved xarray artifacts.
    horizontal_basis_kind : {'SH', 'CS'}, optional
        Basis requested for horizontal surface coefficients and
        operators. ``'SH'`` is the default; ``'CS'`` uses cubed-sphere
        nodal coefficients and finite differences for horizontal
        surface operators. Radial Laplace-continuation terms use the SH
        radial-continuation basis.
    area_weighted_least_squares : bool, optional
        Use surface-area weights for least-squares projections when no
        explicit ``sqrt_weights`` are supplied.

    Returns
    -------
    simulation : Simulation
        The simulation object for performing the simulation and handling
        the simulation results.
    """
    if simulation_directory is None:
        simulation_directory = ArtifactStore.create_temporary_directory("simulation")
    else:
        simulation_directory = str(Path(simulation_directory).resolve())

    if input_directory is None:
        input_directory = Path(simulation_directory) / "prepared_inputs"

    prepare_example_inputs(
        input_directory=input_directory,
        final_time=final_time,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        main_field_kind=main_field_kind,
        main_field_epoch=main_field_epoch,
        main_field_B0=main_field_B0,
        boundary_jr_projection_basis=boundary_jr_projection_basis,
        boundary_Br_projection_basis=boundary_Br_projection_basis,
        conductance_projection_basis=conductance_projection_basis,
        u_projection_basis=u_projection_basis,
        Q_eff_projection_basis=Q_eff_projection_basis,
        boundary_jr_lambda=boundary_jr_lambda,
        conductance_lambda=conductance_lambda,
        u_lambda=u_lambda,
        Q_eff_lambda=Q_eff_lambda,
        multi_data=multi_data,
        artifact_storage=artifact_storage,
        horizontal_basis_kind=horizontal_basis_kind,
        area_weighted_least_squares=area_weighted_least_squares,
        use_wind=use_wind,
        use_Q_eff=use_Q_eff,
        use_boundary_jr=use_boundary_jr,
    )

    return run_from_inputs(
        input_directory,
        simulation_directory=simulation_directory,
        final_time=final_time,
        sampling_step_interval=sampling_step_interval,
        write_sample_interval=write_sample_interval,
        dt=dt,
        RM=RM,
        main_field_kind=main_field_kind,
        enable_pfac_coupling=enable_pfac_coupling,
        enable_interhemispheric_coupling=enable_interhemispheric_coupling,
        interhemispheric_coupling_latitude=interhemispheric_coupling_latitude,
        initialize_from_equilibrium=initialize_from_equilibrium,
        run_dynamic=run_dynamic,
        run_equilibrium=run_equilibrium,
        integrator=integrator,
        least_squares_solver=least_squares_solver,
        least_squares_preconditioner=least_squares_preconditioner,
        reuse_preconditioner=reuse_preconditioner,
        toroidal_potential_regularization_lambda=toroidal_potential_regularization_lambda,
        artifact_storage=artifact_storage,
        magnetic_boundary_shielding=magnetic_boundary_shielding,
    )


__all__ = ["run_example"]
