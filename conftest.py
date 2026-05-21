"""Pytest configuration for PynaMIT tests."""

from __future__ import annotations

import pytest

from pynamit.external_inputs import native_inputs_available
from pynamit.math import JAX_AVAILABLE

pytest_plugins = ("tests.conftest.backend",)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify to skip based on environment capabilities."""
    skip_jax_marker = None if JAX_AVAILABLE else pytest.mark.skip(reason="Requires JAX runtime.")
    skip_native_marker = (
        None
        if native_inputs_available()
        else pytest.mark.skip(reason="Requires native input datasets.")
    )

    for item in items:
        if skip_jax_marker is not None and item.get_closest_marker("requires_jax"):
            item.add_marker(skip_jax_marker)
        if skip_native_marker is not None and item.get_closest_marker("requires_native_inputs"):
            item.add_marker(skip_native_marker)
