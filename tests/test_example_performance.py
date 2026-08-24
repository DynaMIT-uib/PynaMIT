"""Performance regression for the public example workflow."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_native_inputs,
    pytest.mark.parametrize(
        ("backend", "data_source"), [("numpy", "native")], ids=["numpy-native"]
    ),
]


@pytest.mark.skipif(
    os.environ.get("PYNAMIT_SKIP_PERFORMANCE_TESTS") == "1",
    reason="Set PYNAMIT_SKIP_PERFORMANCE_TESTS=1 to skip performance regressions.",
)
def test_two_second_example_runtime(tmp_path):
    """Keep the public coupled example practical for scripts."""
    repo_root = Path(__file__).resolve().parents[1]
    resolution = int(os.environ.get("PYNAMIT_BENCH_RESOLUTION", "20"))
    max_seconds = float(os.environ.get("PYNAMIT_EXAMPLE_MAX_SECONDS", "120"))
    timeout_seconds = float(os.environ.get("PYNAMIT_EXAMPLE_TIMEOUT_SECONDS", "180"))

    benchmark = r"""
import os
import pathlib
import time

from tests.example_scenario import run_example

output = pathlib.Path(os.environ["PYNAMIT_BENCH_OUT"])
resolution = int(os.environ["PYNAMIT_BENCH_RESOLUTION"])

start = time.perf_counter()
run_example(
    simulation_directory=output,
    final_time=2.0,
    dt=5e-4,
    steps_per_sample=4000,
    samples_per_write=1,
    Nmax=resolution,
    Mmax=resolution,
    Ncs=resolution,
    main_field_kind="igrf",
    enable_pfac_coupling=True,
    enable_interhemispheric_coupling=True,
    use_wind=True,
    initialize_from_equilibrium=False,
    run_equilibrium=False,
    artifact_storage="netcdf",
)
print(f"PYNAMIT_EXAMPLE_SECONDS={time.perf_counter() - start:.6f}", flush=True)
"""

    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env["PYNAMIT_BENCH_OUT"] = str(tmp_path / "example")
    env["PYNAMIT_BENCH_RESOLUTION"] = str(resolution)
    env["KOMPE_USE_JAX"] = "0"
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, "-c", benchmark],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        pytest.fail(
            f"Example workflow timed out after {timeout_seconds:.1f}s.\n"
            f"Last output:\n{output[-4000:]}"
        )

    assert result.returncode == 0, result.stdout[-4000:]
    match = re.search(r"PYNAMIT_EXAMPLE_SECONDS=([0-9.]+)", result.stdout)
    assert match is not None, result.stdout[-4000:]

    elapsed = float(match.group(1))
    print(f"Two-second example workflow: {elapsed:.2f}s")
    assert elapsed <= max_seconds, (
        f"Two-second example workflow took {elapsed:.2f}s; "
        f"threshold is {max_seconds:.2f}s.\nLast output:\n{result.stdout[-4000:]}"
    )
