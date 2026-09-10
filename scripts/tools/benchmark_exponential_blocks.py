"""Compare directly forced and equilibrium-centred exponentials.

Run with JAX_ENABLE_X64=1 for scientific double precision. Construction,
first use, and warm applications are separate. Measurements exclude
storage and output synthesis; benchmark_evolution.py includes those.
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from kompe.math import affine_exponential, block_until_ready, get_array_module, set_backend

from pynamit import Simulation
from pynamit.simulation.electrodynamics import induction


def timed(function, *args):
    """Time one synchronized numerical evaluation."""
    start = perf_counter()
    value = block_until_ready(function(*args))
    return value, perf_counter() - start


def benchmark(args, degree):
    """Compare the same fixed equation in potential coordinates."""
    set_backend(args.backend)
    xp = get_array_module()
    if args.backend == "jax":
        from jax.scipy.linalg import expm
    else:
        from scipy.linalg import expm

    simulation = Simulation(
        Nmax=degree,
        Mmax=degree,
        Ncs=args.ncs,
        horizontal_basis_kind=args.horizontal,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        integrator="exponential",
        least_squares_solver="normal_pinv",
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=5 + np.cos(np.deg2rad(grid.theta)),
        hall=3 + np.sin(np.deg2rad(grid.theta)) * np.cos(np.deg2rad(grid.phi)),
        grid=grid,
    )
    response = simulation.response
    geometry = simulation.geometry
    current = xp.linspace(-1e-6, 1e-6, geometry.horizontal_basis.coefficient_count)
    forcing, _ = response.solve_noninductive_response(boundary_jr=current)
    rate = float(geometry.induced_poloidal_potential_faraday_rate_scale)
    A = rate * response.induced_poloidal_potential_feedback_operator
    matrix = A.to_matrix()
    b = rate * geometry.surface_to_poloidal_operator(
        geometry.helmholtz_divergence_free_potential_operator(forcing)
    )
    equilibrium_Br, equilibrium_setup = timed(induction.equilibrium_induced_Br, response, forcing)
    equilibrium = geometry.induced_Br_to_poloidal_potential_operator(equilibrium_Br)
    initial = xp.linspace(-2e-9, 1e-9, A.shape[0])
    potential = geometry.induced_Br_to_poloidal_potential_operator(initial)
    advance = induction.build_induction_stepper(response, forcing)
    # Compile both exponential shapes before warm measurements.
    timed(expm, matrix * args.durations[0])
    first_affine, first_affine_setup = timed(affine_exponential, A, b, args.durations[0])
    block_until_ready(first_affine[0].materialized_matrix)

    # Benchmark only: cache the response to any forcing vector.
    # This costs a second n-by-n matrix and a 2n exponential.
    n = A.shape[0]

    def full_forcing_response(duration):
        augmented = xp.concatenate(
            [xp.concatenate([matrix, xp.eye(n)], axis=1), xp.zeros((n, 2 * n))], axis=0
        )
        exponential = expm(duration * augmented)
        return exponential[:n, :n].copy(), exponential[:n, n:].copy()

    def apply_forcing_response(P, Q, changed_b):
        return geometry.induced_poloidal_potential_to_Br_operator(P @ potential + Q @ changed_b)

    for duration in args.durations:
        legacy_setup, affine_setup, warm_apply, changing_forcing = [], [], [], []
        (P, Q), first_full_setup = timed(full_forcing_response, duration)
        full_setup, full_apply = [], []
        expected = geometry.induced_poloidal_potential_to_Br_operator(
            equilibrium + expm(duration * matrix) @ (potential - equilibrium)
        )
        actual, first_step = timed(lambda duration=duration: next(advance(initial, [duration])))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-19)
        for index in range(args.repeats):
            _, elapsed = timed(expm, duration * matrix)
            legacy_setup.append(elapsed)
            pair, elapsed = timed(affine_exponential, A, b, duration)
            block_until_ready(pair[0].materialized_matrix)
            affine_setup.append(elapsed)
            actual, elapsed = timed(lambda duration=duration: next(advance(initial, [duration])))
            warm_apply.append(elapsed)
            changed = (1 + 0.01 * (index + 1)) * forcing
            changed_result, elapsed = timed(
                induction.evolve_induced_Br, response, initial, duration, changed
            )
            changing_forcing.append(elapsed)
            _, elapsed = timed(full_forcing_response, duration)
            full_setup.append(elapsed)
            changed_b = (1 + 0.01 * (index + 1)) * b
            candidate, elapsed = timed(apply_forcing_response, P, Q, changed_b)
            full_apply.append(elapsed)
            np.testing.assert_allclose(candidate, changed_result, rtol=1e-9, atol=1e-19)
        print(
            json.dumps(
                dict(
                    backend=args.backend,
                    degree=degree,
                    horizontal=args.horizontal,
                    coefficients=A.shape[0],
                    duration=duration,
                    equilibrium_setup_s=equilibrium_setup,
                    first_affine_setup_s=first_affine_setup,
                    first_production_step_s=first_step,
                    homogeneous_expm_s=median(legacy_setup),
                    affine_expm_s=median(affine_setup),
                    cached_production_step_s=median(warm_apply),
                    changed_forcing_step_s=median(changing_forcing),
                    first_full_forcing_setup_s=first_full_setup,
                    full_forcing_setup_s=median(full_setup),
                    full_forcing_apply_s=median(full_apply),
                    extra_forcing_response_bytes=(n * n - n) * matrix.dtype.itemsize,
                    full_forcing_break_even_changes=(
                        median(full_setup) / (median(changing_forcing) - median(full_apply))
                        if median(changing_forcing) > median(full_apply)
                        else None
                    ),
                    relative_error=float(
                        xp.linalg.norm(actual - expected) / xp.linalg.norm(expected)
                    ),
                )
            ),
            flush=True,
        )


def main():
    """Report synchronized CPU or accelerator measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--degrees", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--ncs", type=int, default=12)
    parser.add_argument("--horizontal", choices=("SH", "CS"), default="SH")
    parser.add_argument("--durations", type=float, nargs="+", default=[0.002, 0.2, 2.0])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or not all(np.isfinite(t) and t > 0 for t in args.durations):
        parser.error("repeats and durations must be positive")
    environment = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )
    if args.backend == "jax":
        import jax

        if not jax.config.x64_enabled:
            parser.error("Set JAX_ENABLE_X64=1 for this numerical comparison")
        environment.update(jax=jax.__version__, devices=[str(device) for device in jax.devices()])
    print(json.dumps(environment), flush=True)
    for degree in args.degrees:
        benchmark(args, degree)


if __name__ == "__main__":
    main()
