"""Compare warm single-time and batched physical result evaluation.

Uses synthetic coefficient histories, not a time-integration benchmark.
Both paths reuse the same geometry and maps. First-use JAX compilation
is excluded; all returned arrays are synchronized. This also works on
CPU-only machines and does not imply a GPU speedup.
"""

import argparse
import statistics
from time import perf_counter

import numpy as np
from kompe import SphericalGrid
from kompe.math import block_until_ready, set_backend

from pynamit import Simulation
from pynamit.results import OutputEvaluation, evaluate_simulation_output


def main():
    """Measure identical physical fields with shared evaluation maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 1 or args.repeat < 1:
        parser.error("samples and repeat must be positive")
    set_backend(args.backend)
    simulation = Simulation(Nmax=6, Mmax=6, Ncs=6, RM=4 * 6381e3, enable_pfac_coupling=False)
    results = simulation.results
    times = np.arange(float(args.samples))
    rng = np.random.default_rng(4)
    coefficients = {
        name: rng.normal(size=(args.samples,) + space.shape) * 1e-8
        for name, space in results.schema.output_field_spaces["dynamic"].items()
    }
    results.output_series.add_entries("dynamic", coefficients, times)
    space = results.schema.input_field_spaces["conductance"]
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(space.shape),
            "log_hall_to_pedersen_ratio": np.zeros(space.shape),
        },
        time=0.0,
    )
    grid = SphericalGrid(
        lat=np.linspace(30, 80, 20)[:, None], lon=np.linspace(-180, 180, 30)[None, :]
    )
    evaluation = OutputEvaluation(results.geometry, grid)

    def singles():
        return [evaluate_simulation_output(results, time, evaluation=evaluation) for time in times]

    def batch():
        return evaluate_simulation_output(results, times, evaluation=evaluation)

    expected, actual = singles(), batch()
    for name, values in actual.items():
        np.testing.assert_allclose(
            values, np.stack([row[name] for row in expected], axis=-1), rtol=1e-5, atol=1e-12
        )
    print(f"{args.backend}, {args.samples} samples, {grid.size} points, warm evaluation")
    for name, evaluate in (("single-time calls", singles), ("time block", batch)):
        block_until_ready(evaluate())
        durations = []
        for _ in range(args.repeat):
            start = perf_counter()
            block_until_ready(evaluate())
            durations.append(perf_counter() - start)
        print(f"{name}: {statistics.median(durations):.6f} s (median of {args.repeat})")


if __name__ == "__main__":
    main()
