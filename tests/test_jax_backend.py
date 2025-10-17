import numpy as np
import pytest

from pynamit.math.tensor_operations import tensor_product
from pynamit.utils import JAX_AVAILABLE, to_jax, to_numpy, use_jax


def test_backend_toggle_round_trip():
    """Verify that `use_jax` faithfully toggles the active backend."""
    previous = use_jax()
    try:
        use_jax(True)
        assert use_jax() is True
        use_jax(False)
        assert use_jax() is False
    finally:
        use_jax(previous)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Requires JAX runtime.")
def test_tensor_product_backend_parity(backend: str):
    """Ensure tensor_product produces identical results when JAX is active."""
    if backend != "jax":
        pytest.skip("Only meaningful when executed under the JAX backend.")

    rng = np.random.default_rng(0)
    A = rng.random((3, 4, 5))
    B = rng.random((5, 6, 2))

    numpy_result = tensor_product(A, B, n_contracted=1)
    jax_result = tensor_product(to_jax(A), to_jax(B), n_contracted=1)

    np.testing.assert_allclose(to_numpy(jax_result), numpy_result)
