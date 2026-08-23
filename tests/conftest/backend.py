"""Pytest parametrisation for array and input backends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from kompe.math import (
    JAX_AVAILABLE,
    LEAST_SQUARES_SOLVER_ENV,
    LeastSquaresSolver,
    set_backend,
    use_jax,
)

from pynamit.external_inputs import get_input_source, native_inputs_available, set_input_source
from pynamit.storage import ArtifactStore
from tests import DETERMINISTIC_REGRESSION_RTOL, SINGLE_PRECISION_REGRESSION_RTOL

BACKEND_OPTION_NAME = "--backend"
DATA_OPTION_NAME = "--data-source"
SOLVER_OPTION_NAME = "--least-squares-solver"


def _available_backends(requested: list[str] | None) -> list[str]:
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


def _available_sources(requested: list[str] | None) -> list[str]:
    selectable = ["fallback"]
    if requested is None or "native" in requested:
        native_available = native_inputs_available()
    else:
        native_available = False
    if native_available:
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


def _available_least_squares_solvers(requested: list[str] | None) -> list[str]:
    selectable = list(LeastSquaresSolver.VALID_SOLVERS)
    if not requested:
        configured = os.environ.get(LEAST_SQUARES_SOLVER_ENV, "normal_pinv")
        requested = [configured]

    expanded: list[str] = []
    for solver in requested:
        expanded.extend(selectable if solver == "all" else [solver])

    invalid = sorted(set(expanded) - set(selectable))
    if invalid:
        raise pytest.UsageError(
            f"Unsupported least-squares solver(s): {', '.join(invalid)}. "
            f"Available options: {', '.join(selectable)} or all."
        )

    unique: list[str] = []
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
            "Input-data sources to exercise. By default, the complete suite uses the "
            "bundled fallback and tests marked native_input_validation also use the "
            "native Lompe, PyAMPS, and HWM models when installed. Providing this option "
            "applies the selected source to the complete suite."
        ),
    )
    parser.addoption(
        SOLVER_OPTION_NAME,
        action="append",
        dest="pynamit_least_squares_solvers",
        choices=tuple(LeastSquaresSolver.VALID_SOLVERS) + ("all",),
        help=(
            "Least-squares solver defaults to exercise. The default is normal_pinv "
            "unless KOMPE_LEAST_SQUARES_SOLVER is set. Use 'all' or provide this "
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


def _build_combinations(
    backends: list[str], sources: list[str], *, include_native: bool
) -> list[tuple[str, str]]:
    return [
        (backend, source)
        for backend in backends
        for source in sources
        if source == "fallback" or include_native
    ]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise tests from available backends and data sources."""
    backends: list[str] = getattr(metafunc.config, "_pynamit_backend_list", ["numpy"])
    sources: list[str] = getattr(metafunc.config, "_pynamit_data_sources", ["fallback"])
    solvers: list[str] = getattr(
        metafunc.config, "_pynamit_least_squares_solvers", ["normal_pinv"]
    )

    def _is_parametrized(arg: str) -> bool:
        for marker in metafunc.definition.iter_markers("parametrize"):
            marker_arg = marker.args[0]
            if isinstance(marker_arg, str):
                params = [name.strip() for name in marker_arg.split(",")]
            else:
                params = list(marker_arg)
            if arg in params:
                return True
        return False

    if {"backend", "data_source"}.issubset(metafunc.fixturenames):
        if not (_is_parametrized("backend") or _is_parametrized("data_source")):
            data_sources_were_requested = (
                metafunc.config.getoption("pynamit_data_sources") is not None
            )
            validates_native_inputs = (
                metafunc.definition.get_closest_marker("native_input_validation") is not None
            )
            combos = _build_combinations(
                backends,
                sources,
                include_native=data_sources_were_requested or validates_native_inputs,
            )
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
    previous_backend_env = os.environ.get("KOMPE_USE_JAX")
    previous_source = get_input_source()
    previous_source_env = os.environ.get("PYNAMIT_INPUT_SOURCE")
    previous_solver_env = os.environ.get(LEAST_SQUARES_SOLVER_ENV)
    previous_mplconfig = os.environ.get("MPLCONFIGDIR")
    previous_xdg_cache = os.environ.get("XDG_CACHE_HOME")

    with tempfile.TemporaryDirectory(prefix="pynamit-test-cache-") as cache_root:
        mplconfig_dir = os.path.join(cache_root, "matplotlib")
        xdg_cache_dir = os.path.join(cache_root, "xdg-cache")
        os.makedirs(mplconfig_dir, exist_ok=True)
        os.makedirs(xdg_cache_dir, exist_ok=True)

        try:
            if JAX_AVAILABLE:
                from jax import config

                config.update("jax_enable_x64", True)
            os.environ["MPLCONFIGDIR"] = mplconfig_dir
            os.environ["XDG_CACHE_HOME"] = xdg_cache_dir
            set_backend(backend)
            set_input_source(data_source)
            os.environ[LEAST_SQUARES_SOLVER_ENV] = least_squares_solver
            yield
        finally:
            set_backend(previous_backend)
            set_input_source(previous_source)
            if previous_backend_env is None:
                os.environ.pop("KOMPE_USE_JAX", None)
            else:
                os.environ["KOMPE_USE_JAX"] = previous_backend_env
            if previous_source_env is None:
                os.environ.pop("PYNAMIT_INPUT_SOURCE", None)
            else:
                os.environ["PYNAMIT_INPUT_SOURCE"] = previous_source_env
            if previous_solver_env is None:
                os.environ.pop(LEAST_SQUARES_SOLVER_ENV, None)
            else:
                os.environ[LEAST_SQUARES_SOLVER_ENV] = previous_solver_env
            if previous_mplconfig is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mplconfig
            if previous_xdg_cache is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous_xdg_cache


@pytest.fixture(autouse=True)
def isolate_default_simulation_directories(tmp_path, monkeypatch):
    """Route implicit artifacts into per-test temporary space."""
    original_creator = ArtifactStore.create_temporary_directory

    def _create_temporary_directory(parent: str | os.PathLike[str] | None = None) -> str:
        target = None if parent is None else Path(parent)
        if target is not None and not target.is_absolute() and str(target) == "simulation":
            return original_creator(tmp_path / target)
        return original_creator(parent)

    monkeypatch.setattr(
        ArtifactStore, "create_temporary_directory", staticmethod(_create_temporary_directory)
    )


@pytest.fixture
def regression_rtol(request: pytest.FixtureRequest, data_source: str) -> float:
    """Return the tolerance implied by the test's numerical path."""
    uses_apexpy = request.node.get_closest_marker("apexpy_precision") is not None
    uses_native_hwm = (
        data_source == "native"
        and request.node.get_closest_marker("native_hwm_precision") is not None
    )
    if uses_apexpy or uses_native_hwm:
        return SINGLE_PRECISION_REGRESSION_RTOL
    return DETERMINISTIC_REGRESSION_RTOL


@pytest.fixture
def regression_approx(regression_rtol: float):
    """Return ``pytest.approx`` configured for a stored regression."""
    from functools import partial

    return partial(pytest.approx, rel=regression_rtol, abs=0.0)
