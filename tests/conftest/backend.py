"""Pytest parametrisation for array and input backends."""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest

from pynamit.external_inputs import get_input_source, native_inputs_available, set_input_source
from pynamit.math.least_squares_solver import LEAST_SQUARES_SOLVER_ENV, LeastSquaresSolver
from pynamit.utils import JAX_AVAILABLE, set_backend, use_jax

BACKEND_OPTION_NAME = "--backend"
DATA_OPTION_NAME = "--data-source"
SOLVER_OPTION_NAME = "--least-squares-solver"


def _available_backends(requested: List[str] | None) -> List[str]:
    selectable = ["numpy"]
    if JAX_AVAILABLE:
        selectable.append("jax")
    if not requested:
        return selectable

    invalid = sorted(set(requested) - set(selectable))
    if invalid:
        raise pytest.UsageError(
            f"Unsupported backend(s): {', '.join(invalid)}. "
            f"Available options: {', '.join(selectable)}."
        )
    return requested


def _available_sources(requested: List[str] | None) -> List[str]:
    selectable = ["fallback"]
    if native_inputs_available():
        selectable.append("native")
    if not requested:
        return selectable

    invalid = sorted(set(requested) - set(selectable))
    if invalid:
        raise pytest.UsageError(
            f"Unsupported data source(s): {', '.join(invalid)}. "
            f"Available options: {', '.join(selectable)}."
        )
    return requested


def _available_least_squares_solvers(requested: List[str] | None) -> List[str]:
    selectable = list(LeastSquaresSolver.VALID_SOLVERS)
    if not requested:
        configured = os.environ.get(LEAST_SQUARES_SOLVER_ENV, "normal_pinv")
        requested = [configured]

    expanded: List[str] = []
    for solver in requested:
        expanded.extend(selectable if solver == "all" else [solver])

    invalid = sorted(set(expanded) - set(selectable))
    if invalid:
        raise pytest.UsageError(
            f"Unsupported least-squares solver(s): {', '.join(invalid)}. "
            f"Available options: {', '.join(selectable)} or all."
        )

    unique: List[str] = []
    for solver in expanded:
        if solver not in unique:
            unique.append(solver)
    return unique


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI options for selecting backends and data sources."""
    parser.addoption(
        BACKEND_OPTION_NAME,
        action="append",
        dest="pynamit_backends",
        choices=("numpy", "jax"),
        help=(
            "Array backends to test against. By default the suite runs once with "
            "NumPy and once with JAX (if JAX is installed). Provide this option "
            "multiple times to limit the selection."
        ),
    )
    parser.addoption(
        DATA_OPTION_NAME,
        action="append",
        dest="pynamit_data_sources",
        choices=("fallback", "native"),
        help=(
            "Input-data sources to exercise. The default runs against the bundled "
            "fallback dataset and, when available, against native lompe/pyamps/pyhwm2014 "
            "inputs. Provide this option multiple times to limit the selection."
        ),
    )
    parser.addoption(
        SOLVER_OPTION_NAME,
        action="append",
        dest="pynamit_least_squares_solvers",
        choices=tuple(LeastSquaresSolver.VALID_SOLVERS) + ("all",),
        help=(
            "Least-squares solver defaults to exercise. The default is normal_pinv "
            "unless PYNAMIT_LEAST_SQUARES_SOLVER is set. Use 'all' or provide this "
            "option multiple times to run the suite across multiple solvers."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Store available backends and data sources in pytest config."""
    config._pynamit_backend_list = _available_backends(config.getoption("pynamit_backends"))  # type: ignore[attr-defined]
    config._pynamit_data_sources = _available_sources(config.getoption("pynamit_data_sources"))  # type: ignore[attr-defined]
    config._pynamit_least_squares_solvers = _available_least_squares_solvers(  # type: ignore[attr-defined]
        config.getoption("pynamit_least_squares_solvers")
    )


def _build_combinations(backends: List[str], sources: List[str]) -> List[Tuple[str, str]]:
    combos: List[Tuple[str, str]] = []
    if "numpy" in backends:
        combos.append(("numpy", "fallback"))
        if "native" in sources:
            combos.append(("numpy", "native"))
    if "jax" in backends and "fallback" in sources:
        combos.append(("jax", "fallback"))
    return combos


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise tests from available backends and data sources."""
    backends: List[str] = getattr(metafunc.config, "_pynamit_backend_list", ["numpy"])
    sources: List[str] = getattr(metafunc.config, "_pynamit_data_sources", ["fallback"])
    solvers: List[str] = getattr(
        metafunc.config, "_pynamit_least_squares_solvers", ["normal_pinv"]
    )

    def _is_parametrized(arg: str) -> bool:
        for marker in metafunc.definition.iter_markers("parametrize"):
            params = [name.strip() for name in marker.args[0].split(",")]
            if arg in params:
                return True
        return False

    if {"backend", "data_source"}.issubset(metafunc.fixturenames):
        if not (_is_parametrized("backend") or _is_parametrized("data_source")):
            combos = _build_combinations(backends, sources)
            ids = [f"backend={b},data={s}" for b, s in combos]
            metafunc.parametrize(("backend", "data_source"), combos, ids=ids)
    else:
        if "backend" in metafunc.fixturenames and not _is_parametrized("backend"):
            metafunc.parametrize("backend", backends, ids=[f"backend={name}" for name in backends])
        if "data_source" in metafunc.fixturenames and not _is_parametrized("data_source"):
            metafunc.parametrize("data_source", sources, ids=[f"data={name}" for name in sources])
    if "least_squares_solver" in metafunc.fixturenames and not _is_parametrized(
        "least_squares_solver"
    ):
        metafunc.parametrize(
            "least_squares_solver", solvers, ids=[f"ls={name}" for name in solvers]
        )


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> str:
    """Fixture to provide the backend parameter."""
    return request.param  # type: ignore[attr-defined]


@pytest.fixture
def data_source(request: pytest.FixtureRequest) -> str:
    """Fixture to provide the data source parameter."""
    return request.param  # type: ignore[attr-defined]


@pytest.fixture
def least_squares_solver(request: pytest.FixtureRequest) -> str:
    """Fixture to provide the least-squares solver parameter."""
    return request.param  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def configure_runtime(backend: str, data_source: str, least_squares_solver: str):
    """Fixture to configure backend and data source for each test."""
    previous_backend = use_jax()
    previous_backend_env = os.environ.get("PYNAMIT_USE_JAX")
    previous_source = get_input_source()
    previous_source_env = os.environ.get("PYNAMIT_INPUT_SOURCE")
    previous_solver_env = os.environ.get(LEAST_SQUARES_SOLVER_ENV)

    try:
        set_backend(backend)
        set_input_source(data_source)
        os.environ[LEAST_SQUARES_SOLVER_ENV] = least_squares_solver
        yield
    finally:
        set_backend(previous_backend)
        set_input_source(previous_source)
        if previous_backend_env is None:
            os.environ.pop("PYNAMIT_USE_JAX", None)
        else:
            os.environ["PYNAMIT_USE_JAX"] = previous_backend_env
        if previous_source_env is None:
            os.environ.pop("PYNAMIT_INPUT_SOURCE", None)
        else:
            os.environ["PYNAMIT_INPUT_SOURCE"] = previous_source_env
        if previous_solver_env is None:
            os.environ.pop(LEAST_SQUARES_SOLVER_ENV, None)
        else:
            os.environ[LEAST_SQUARES_SOLVER_ENV] = previous_solver_env
