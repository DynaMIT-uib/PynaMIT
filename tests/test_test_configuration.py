"""Tests for optional test-environment selection."""

from tests.conftest import backend as test_backend


def test_missing_jax_selects_only_numpy(monkeypatch):
    """The default suite must not request JAX when it is unavailable."""
    monkeypatch.setattr(test_backend, "JAX_AVAILABLE", False)

    assert test_backend._available_backends(None) == ["numpy"]


def test_missing_native_providers_selects_only_fallback(monkeypatch):
    """Do not request unavailable input providers by default."""
    monkeypatch.setattr(test_backend, "native_inputs_available", lambda: False)

    assert test_backend._available_sources(None) == ["fallback"]


def test_backend_and_input_selection_is_a_full_matrix():
    """Explicit selection checks each backend with each input source."""
    assert test_backend._build_combinations(
        ["numpy", "jax"], ["fallback", "native"], include_native=True
    ) == [("numpy", "fallback"), ("numpy", "native"), ("jax", "fallback"), ("jax", "native")]


def test_ordinary_default_selection_uses_fallback_inputs():
    """Native inputs are reserved for their focused validation tests."""
    assert test_backend._build_combinations(
        ["numpy", "jax"], ["fallback", "native"], include_native=False
    ) == [("numpy", "fallback"), ("jax", "fallback")]
