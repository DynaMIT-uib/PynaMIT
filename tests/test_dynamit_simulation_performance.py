"""Opt-in performance regression test for the DynaMIT example script."""

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
        "backend,data_source", [("numpy", "auto")], ids=["backend=numpy,data=auto"]
    ),
]


@pytest.mark.skipif(
    os.environ.get("PYNAMIT_SKIP_PERFORMANCE_TESTS") == "1",
    reason="Set PYNAMIT_SKIP_PERFORMANCE_TESTS=1 to skip this performance regression test.",
)
def test_dynamit_simulation_first_step_runtime(tmp_path):
    """The first 2-second DynaMIT update/evolve step should stay fast."""
    pytest.importorskip("apexpy")
    pytest.importorskip("dipole")
    pytest.importorskip("lompe")
    pytest.importorskip("pyamps")
    pytest.importorskip("pyhwm2014")

    repo_root = Path(__file__).resolve().parents[1]
    bench_resolution = int(os.environ.get("PYNAMIT_DYNAMIT_BENCH_RESOLUTION", "20"))
    max_step_seconds = float(os.environ.get("PYNAMIT_DYNAMIT_STEP_MAX_SECONDS", "100"))
    timeout_seconds = float(os.environ.get("PYNAMIT_DYNAMIT_STEP_TIMEOUT_SECONDS", "200"))

    bench_code = r"""
import os
import pathlib
import time

path = pathlib.Path("scripts/simulation/dynamit_simulation.py")
src = path.read_text()
out = pathlib.Path(os.environ["PYNAMIT_BENCH_OUT"])
out.mkdir(parents=True, exist_ok=True)
resolution = int(os.environ["PYNAMIT_DYNAMIT_BENCH_RESOLUTION"])

src = src.replace(
    'filename_prefix = "aurora2"',
    "filename_prefix = " + repr(str(out / "aurora2")),
    1,
)
src = src.replace(
    "Nmax, Mmax, Ncs = 30, 30, 30",
    f"Nmax, Mmax, Ncs = {resolution}, {resolution}, {resolution}",
    1,
)
src = src.replace(
    "while True:",
    "for __bench_step in range(1):\n    __bench_step_t0 = time.perf_counter()",
    1,
)
src = src.replace(
    "    dynamics.evolve_to_time(next_time)",
    '    dynamics.evolve_to_time(next_time)\n'
    '    print(f"PYNAMIT_STEP_SECONDS={time.perf_counter()-__bench_step_t0:.6f}", flush=True)',
    1,
)

exec(compile(src, str(path), "exec"), {"__name__": "__main__", "__file__": str(path), "time": time})
"""

    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env["PYNAMIT_BENCH_OUT"] = str(tmp_path / "dynamit-simulation")
    env["PYNAMIT_DYNAMIT_BENCH_RESOLUTION"] = str(bench_resolution)
    env["PYNAMIT_USE_JAX"] = "0"
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, "-c", bench_code],
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
            f"dynamit_simulation.py first step timed out after {timeout_seconds:.1f}s.\n"
            f"Last output:\n{output[-4000:]}"
        )

    assert result.returncode == 0, result.stdout[-4000:]

    match = re.search(r"PYNAMIT_STEP_SECONDS=([0-9.]+)", result.stdout)
    assert match is not None, result.stdout[-4000:]

    step_seconds = float(match.group(1))
    print(f"dynamit_simulation.py first 2-second step: {step_seconds:.2f}s")
    assert step_seconds <= max_step_seconds, (
        f"dynamit_simulation.py first 2-second step took {step_seconds:.2f}s; "
        f"threshold is {max_step_seconds:.2f}s.\n"
        f"Last output:\n{result.stdout[-4000:]}"
    )
