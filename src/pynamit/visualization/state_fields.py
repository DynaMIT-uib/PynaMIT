"""Evaluate simulation output fields on visualization grids."""

import numpy as np

from pynamit.math.constants import mu0
from pynamit.visualization.field_maps import evaluate_JS_from_maps
from pynamit.visualization.grid_evaluation import transform_for_source


def current_output_key(dynamics, preferred=None):
    """Return the available output key to visualize."""
    datasets = dynamics.output_timeseries.datasets
    if preferred is not None:
        if preferred not in datasets:
            raise ValueError(f"No output dataset named {preferred!r} is available.")
        return preferred
    if "state" in datasets:
        return "state"
    if "steady_state" in datasets:
        return "steady_state"
    raise RuntimeError("No state or steady_state output is available to visualize.")


def current_output_entry(dynamics, key=None):
    """Return current output coefficients from a ``Dynamics`` object."""
    key = current_output_key(dynamics, preferred=key)
    entry = dynamics.output_timeseries.get_entry(key, dynamics.current_time)
    if entry is None:
        raise RuntimeError(
            f"No {key!r} output is available at t={float(dynamics.current_time):.3f}."
        )
    return entry


def evaluate_Br_coefficients(geometry, m_ind, transform):
    """Evaluate radial magnetic perturbation from ``m_ind``."""
    coeffs = geometry.m_ind_to_Br_operator.matvec(m_ind)
    return transform_for_source(geometry.basis, transform).synthesize_scalar(coeffs)


def evaluate_Br(dynamics, transform, *, key=None):
    """Evaluate radial magnetic perturbation on ``transform.target``."""
    entry = current_output_entry(dynamics, key=key)
    return evaluate_Br_coefficients(dynamics.state.geometry, entry["m_ind"], transform)


def evaluate_jr_coefficients(geometry, m_imp, transform):
    """Evaluate radial current density from ``m_imp`` coefficients."""
    coeffs = geometry.m_imp_to_jr_operator.matvec(m_imp)
    return transform_for_source(geometry.basis, transform).synthesize_scalar(coeffs)


def evaluate_jr(dynamics, transform, *, key=None):
    """Evaluate radial current density on ``transform.target``."""
    entry = current_output_entry(dynamics, key=key)
    return evaluate_jr_coefficients(dynamics.state.geometry, entry["m_imp"], transform)


def evaluate_equivalent_current_coefficients(geometry, m_ind, transform):
    """Evaluate equivalent-current stream function from ``m_ind``."""
    coeffs = (
        -geometry.RI
        / mu0
        * geometry.horizontal_to_boundary_potential_jump_factor_operator.matvec(m_ind)
    )
    return geometry.solid_transform_for(transform).synthesize_scalar(coeffs)


def evaluate_equivalent_current_function(dynamics, transform, *, key=None):
    """Evaluate the equivalent-current stream function."""
    entry = current_output_entry(dynamics, key=key)
    return evaluate_equivalent_current_coefficients(
        dynamics.state.geometry, entry["m_ind"], transform
    )


def evaluate_JS_coefficients(geometry, m_imp, m_ind, transform):
    """Evaluate total horizontal JS from coefficients."""
    m_imp = np.asarray(m_imp)
    m_ind = np.asarray(m_ind)

    horizontal_transform = transform_for_source(geometry.basis, transform)
    m_imp_to_JS = geometry.m_imp_to_gridded_JS(horizontal_transform)
    m_ind_to_JS = geometry.m_ind_to_gridded_JS(horizontal_transform)
    return evaluate_JS_from_maps(m_imp, m_ind, m_imp_to_JS=m_imp_to_JS, m_ind_to_JS=m_ind_to_JS)


def evaluate_JS(dynamics, transform, *, key=None):
    """Evaluate total horizontal JS."""
    entry = current_output_entry(dynamics, key=key)
    return evaluate_JS_coefficients(
        dynamics.state.geometry, entry["m_imp"], entry["m_ind"], transform
    )

def evaluate_Phi_coefficients(geometry, Phi, transform):
    """Evaluate saved curl-free E coefficients as potential in volts."""
    return geometry.RI * transform_for_source(geometry.basis, transform).synthesize_scalar(Phi)


def evaluate_Phi(dynamics, transform, *, key=None):
    """Evaluate electric curl-free potential in volts."""
    entry = current_output_entry(dynamics, key=key)
    return evaluate_Phi_coefficients(dynamics.state.geometry, entry["Phi"], transform)


def evaluate_W_coefficients(geometry, W, transform):
    """Evaluate divergence-free E coefficients as potential in volts."""
    return geometry.RI * transform_for_source(geometry.basis, transform).synthesize_scalar(W)


def evaluate_W(dynamics, transform, *, key=None):
    """Evaluate electric divergence-free potential in volts."""
    entry = current_output_entry(dynamics, key=key)
    return evaluate_W_coefficients(dynamics.state.geometry, entry["W"], transform)


__all__ = [
    "current_output_entry",
    "current_output_key",
    "evaluate_Br",
    "evaluate_Br_coefficients",
    "evaluate_Phi",
    "evaluate_Phi_coefficients",
    "evaluate_W",
    "evaluate_W_coefficients",
    "evaluate_equivalent_current_coefficients",
    "evaluate_equivalent_current_function",
    "evaluate_jr",
    "evaluate_jr_coefficients",
    "evaluate_JS",
    "evaluate_JS_coefficients",
]
