import numpy as np
import pytest

from pynamit.math.tensor_operations import tensor_product
from pynamit.utils import to_jax, to_numpy, use_jax


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
    """Ensure tensor_product produces identical results when JAX is active."""
    rng = np.random.default_rng(0)
    A = rng.random((3, 4, 5))
    B = rng.random((5, 6, 2))

    numpy_result = tensor_product(A, B, n_contracted=1)
    jax_result = tensor_product(to_jax(A), to_jax(B), n_contracted=1)

    np.testing.assert_allclose(to_numpy(jax_result), numpy_result)
