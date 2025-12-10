"""Tests for JAX backend functionality."""

import numpy as np
import pytest

from pynamit.utils import use_jax, xp


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_backend_toggle_round_trip(backend: str, data_source: str):
    """Verify that `use_jax` faithfully toggles the active backend."""
    previous = use_jax()
    try:
        use_jax(True)
        assert use_jax() is True
        use_jax(False)
        assert use_jax() is False
    finally:
        use_jax(previous)
