"""Tests for JAX backend functionality."""

import os

import pytest
from kompe.math import backend_context, set_backend, use_jax


def test_backend_context_restores_backend_and_environment(monkeypatch):
    """Scoped backend changes restore the previous process state."""
    previous_backend = use_jax()
    previous_env = os.environ.get("KOMPE_USE_JAX")
    try:
        set_backend("numpy")
        monkeypatch.setenv("KOMPE_USE_JAX", "preserved")

        with backend_context("numpy") as active_backend:
            assert active_backend == "numpy"
            assert use_jax() is False
            assert os.environ["KOMPE_USE_JAX"] == "0"

        assert use_jax() is False
        assert os.environ["KOMPE_USE_JAX"] == "preserved"
    finally:
        set_backend(previous_backend)
        if previous_env is None:
            os.environ.pop("KOMPE_USE_JAX", None)
        else:
            os.environ["KOMPE_USE_JAX"] = previous_env


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
