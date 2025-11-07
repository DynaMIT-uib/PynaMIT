"""Tests for JAX backend functionality."""

import numpy as np
import pytest

from pynamit.math.tensor_operations import tensor_product
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


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_tensor_product_backend_parity(backend: str, data_source: str):
    """Ensure tensor_product produces identical results with JAX."""
    rng = np.random.default_rng(0)
    A = rng.random((3, 4, 5))
    B = rng.random((5, 6, 2))

    numpy_result = tensor_product(A, B, n_contracted=1)
    backend_A = xp.asarray(A) if use_jax() else A
    backend_B = xp.asarray(B) if use_jax() else B
    backend_result = tensor_product(backend_A, backend_B, n_contracted=1)

    np.testing.assert_allclose(np.asarray(backend_result), numpy_result)
