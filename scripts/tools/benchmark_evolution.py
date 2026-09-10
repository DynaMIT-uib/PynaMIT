"""Measure evolution including field sampling and in-memory output.

Construction, first evolution (including JAX compilation), and warm
continuation are reported separately. Warm times include xarray output;
all samples have reached the CPU when evolve_to_time returns. Disk I/O
and native input providers are deliberately excluded.

Examples::

    python scripts/tools/benchmark_evolution.py --steps 1000 --sample 10
    JAX_ENABLE_X64=1 python scripts/tools/benchmark_evolution.py \
        --backend jax --degree 20 --integrator exponential \
        --steps 10000 --sample 1000

Use --profile to profile one additional warm continuation. Times are
measurements, not CI pass/fail thresholds.
"""

import argparse
import cProfile
import json
import platform
import pstats
from statistics import median
from time import perf_counter

import numpy as np
from kompe.math import set_backend

from pynamit import Simulation


def benchmark(args):
    """Evolve a dipole with nonuniform conductance and fixed current."""
    set_backend(args.backend)
    start = perf_counter()
    simulation = Simulation(
        Nmax=args.degree,
        Mmax=args.degree,
        Ncs=args.ncs,
        horizontal_basis_kind=args.horizontal,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        integrator=args.integrator,
        least_squares_solver="normal_pinv",
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=5 + np.cos(np.deg2rad(grid.theta)),
        hall=3 + np.sin(np.deg2rad(grid.theta)) * np.cos(np.deg2rad(grid.phi)),
        grid=grid,
    )
    simulation.inputs.set_coefficients(
        "boundary_jr",
        np.linspace(-1e-6, 1e-6, simulation.geometry.horizontal_basis.coefficient_count),
    )
    setup_seconds = perf_counter() - start

    def run():
        simulation.evolve_to_time(
            simulation.current_time + args.dt * args.steps,
            dt=args.dt if args.integrator == "euler" else None,
            output_interval=args.dt * args.sample,
            samples_per_write=args.write,
            initialize_from_equilibrium=False,
            sample_equilibrium=False,
            quiet=True,
        )

    start = perf_counter()
    run()
    first_seconds = perf_counter() - start
    times = []
    for _ in range(args.repeats):
        start = perf_counter()
        run()
        times.append(perf_counter() - start)
    print(
        json.dumps(
            dict(
                **vars(args),
                python=platform.python_version(),
                platform=platform.platform(),
                setup_s=setup_seconds,
                first_evolution_s=first_seconds,
                warm_evolution_s=median(times),
                sample_count=simulation.outputs["dynamic"].sizes["time"],
                final_Br_norm=float(
                    np.linalg.norm(simulation.outputs["dynamic"]["SH_induced_Br"].values[-1])
                ),
            )
        ),
        flush=True,
    )
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(run)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)


def main():
    """Run a reproducible warm-continuation benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--ncs", type=int, default=12)
    parser.add_argument("--horizontal", choices=("SH", "CS"), default="SH")
    parser.add_argument("--integrator", choices=("euler", "exponential"), default="euler")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--write", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.0002)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.sample, args.write, args.repeats) < 1 or args.dt <= 0:
        parser.error("steps, sample/write intervals, repeats, and dt must be positive")
    if args.backend == "jax":
        import jax

        if not jax.config.x64_enabled:
            parser.error("Set JAX_ENABLE_X64=1 for a scientific double-precision comparison")
    benchmark(args)


if __name__ == "__main__":
    main()
