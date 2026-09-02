"""Pytest configuration for PynaMIT tests."""

from __future__ import annotations

import pytest
from kompe.math import JAX_AVAILABLE

from pynamit.external_inputs import native_inputs_available

pytest_plugins = ("tests.conftest.backend",)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify to skip based on environment capabilities."""
    skip_jax_marker = None if JAX_AVAILABLE else pytest.mark.skip(reason="Requires JAX runtime.")
    requested_sources = config.getoption("pynamit_data_sources")
    native_selected = requested_sources is None or "native" in requested_sources
    if not native_selected:
        skip_native_marker = pytest.mark.skip(
            reason="Native-input tests are excluded by --data-source fallback."
        )
    elif not native_inputs_available():
        skip_native_marker = pytest.mark.skip(reason="Requires native input datasets.")
    else:
        skip_native_marker = None

    for item in items:
        if skip_jax_marker is not None and item.get_closest_marker("requires_jax"):
            item.add_marker(skip_jax_marker)
        if skip_native_marker is not None and item.get_closest_marker("requires_native_inputs"):
            item.add_marker(skip_native_marker)
