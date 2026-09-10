"""Tests for JAX backend functionality."""

import os

import pytest
from kompe.math import backend_context, get_backend, set_backend


def test_backend_context_restores_backend_and_environment(monkeypatch):
    """Scoped backend changes restore the previous process state."""
    previous_backend = get_backend()
    previous_env = os.environ.get("KOMPE_USE_JAX")
    try:
        set_backend("numpy")
        monkeypatch.setenv("KOMPE_USE_JAX", "preserved")

        with backend_context("numpy") as active_backend:
            assert active_backend == "numpy"
            assert get_backend() == "numpy"
            assert os.environ["KOMPE_USE_JAX"] == "0"

        assert get_backend() == "numpy"
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
    """Verify that the backend can be changed and restored."""
    previous = get_backend()
    try:
        set_backend("jax")
        assert get_backend() == "jax"
        set_backend("numpy")
        assert get_backend() == "numpy"
    finally:
        set_backend(previous)
