"""Pytest parametrisation for array and input backends."""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest

from pynamit.external_inputs import get_input_source, native_inputs_available, set_input_source
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
    backends: List[str] = getattr(metafunc.config, "_pynamit_backend_list", ["numpy"])
    sources: List[str] = getattr(metafunc.config, "_pynamit_data_sources", ["fallback"])

    if {"backend", "data_source"}.issubset(metafunc.fixturenames):
        combos = _build_combinations(backends, sources)
        ids = [f"backend={b},data={s}" for b, s in combos]
        metafunc.parametrize(("backend", "data_source"), combos, ids=ids)
    else:
        if "backend" in metafunc.fixturenames:
            metafunc.parametrize("backend", backends, ids=[f"backend={name}" for name in backends])
        if "data_source" in metafunc.fixturenames:
            metafunc.parametrize("data_source", sources, ids=[f"data={name}" for name in sources])


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[attr-defined]


@pytest.fixture
def data_source(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def configure_runtime(backend: str, data_source: str):
    previous_backend = use_jax()
    previous_backend_env = os.environ.get("PYNAMIT_USE_JAX")
    previous_source = get_input_source()
    previous_source_env = os.environ.get("PYNAMIT_INPUT_SOURCE")

    try:
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
