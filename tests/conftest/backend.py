"""Pytest parametrisation for array and input backends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

from pynamit.data import get_input_source, set_input_source
from pynamit.data.loaders import native_inputs_available
from pynamit.primitives.io import IO
from pynamit.utils import JAX_AVAILABLE, set_backend, use_jax

BACKEND_OPTION_NAME = "--backend"
DATA_OPTION_NAME = "--data-source"


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


def pytest_configure(config: pytest.Config) -> None:
    """Store available backends and data sources in pytest config."""
    config._pynamit_backend_list = _available_backends(config.getoption("pynamit_backends"))  # type: ignore[attr-defined]
    config._pynamit_data_sources = _available_sources(config.getoption("pynamit_data_sources"))  # type: ignore[attr-defined]


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


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> str:
    """Fixture to provide the backend parameter."""
    return request.param  # type: ignore[attr-defined]


@pytest.fixture
def data_source(request: pytest.FixtureRequest) -> str:
    """Fixture to provide the data source parameter."""
    return request.param  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def configure_runtime(backend: str, data_source: str):
    """Fixture to configure backend and data source for each test."""
    previous_backend = use_jax()
    previous_backend_env = os.environ.get("PYNAMIT_USE_JAX")
    previous_source = get_input_source()
    previous_source_env = os.environ.get("PYNAMIT_INPUT_SOURCE")
    previous_mplconfig = os.environ.get("MPLCONFIGDIR")
    previous_xdg_cache = os.environ.get("XDG_CACHE_HOME")
    with tempfile.TemporaryDirectory(prefix="pynamit-test-cache-") as cache_root:
        mplconfig_dir = os.path.join(cache_root, "matplotlib")
        xdg_cache_dir = os.path.join(cache_root, "xdg-cache")
        os.makedirs(mplconfig_dir, exist_ok=True)
        os.makedirs(xdg_cache_dir, exist_ok=True)

        try:
            # Enforce JAX 64-bit precision to match Numpy results for regression testing
            if JAX_AVAILABLE:
                from jax import config

                config.update("jax_enable_x64", True)

            os.environ["MPLCONFIGDIR"] = mplconfig_dir
            os.environ["XDG_CACHE_HOME"] = xdg_cache_dir
            set_backend(backend)
            set_input_source(data_source)
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
            if previous_mplconfig is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mplconfig
            if previous_xdg_cache is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous_xdg_cache


@pytest.fixture(autouse=True)
def isolate_default_run_directories(tmp_path, monkeypatch):
    """Route implicit ``run_pynamit`` output directories into per-test temp space."""
    original_builder = IO.build_temporary_run_directory_in_directory

    def _build_temporary_run_directory_in_directory(directory: str | os.PathLike[str]) -> str:
        target = Path(directory)
        if not target.is_absolute() and str(target) == "simulation":
            return original_builder(tmp_path / target)
        return original_builder(directory)

    monkeypatch.setattr(
        IO,
        "build_temporary_run_directory_in_directory",
        staticmethod(_build_temporary_run_directory_in_directory),
    )


@pytest.fixture
def rel_tol(request: pytest.FixtureRequest, data_source: str) -> float:
    """Fixture to provide the relative tolerance for comparisons.

    Returns 1e-5 for native tests (pyHWM uses single precision).
    Returns 1e-9 for fallback tests (allows for minor platform/version differences).
    """
    if data_source == "native":
        return 1e-5
    return 1e-9


@pytest.fixture
def pynamit_approx(rel_tol: float):
    """Fixture providing a configured pytest.approx with appropriate rel threshold."""
    from functools import partial

    return partial(pytest.approx, rel=rel_tol, abs=0.0)
