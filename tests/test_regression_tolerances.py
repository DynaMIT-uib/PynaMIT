"""Tests for numerical-regression tolerance selection."""

import pytest

from tests import DETERMINISTIC_REGRESSION_RTOL, SINGLE_PRECISION_REGRESSION_RTOL


def test_deterministic_regression_uses_tight_tolerance(regression_rtol):
    """Deterministic paths use the tight stored-baseline tolerance."""
    assert regression_rtol == DETERMINISTIC_REGRESSION_RTOL


@pytest.mark.apexpy_precision
def test_apexpy_regression_uses_single_precision_tolerance(regression_rtol):
    """ApexPy paths use the relaxed cross-platform tolerance."""
    assert regression_rtol == SINGLE_PRECISION_REGRESSION_RTOL


@pytest.mark.native_hwm_precision
def test_hwm_regression_relaxes_only_native_inputs(regression_rtol, data_source):
    """Fallback HWM data stay deterministic while native HWM relaxes."""
    expected = (
        SINGLE_PRECISION_REGRESSION_RTOL
        if data_source == "native"
        else DETERMINISTIC_REGRESSION_RTOL
    )
    assert regression_rtol == expected
