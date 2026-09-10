"""Independent vector-calculus checks of the thin-sheet model.

Expected values come from 3-D Ohm's law, scalar magnetic potentials,
and analytic spherical-harmonic modes, not the operators under test.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from kompe import SHBasis, SolidHarmonicOperators, SphericalGrid, SphericalTransform
from kompe.constants import MU0

from pynamit import Simulation
from pynamit.geomagnetism import MainField
from pynamit.results.output_fields import build_ground_magnetic_field_operators
from pynamit.simulation.electrodynamics.induction import (
    equilibrium_induced_Br,
    evolve_induced_Br,
    induced_Br_time_derivative,
)
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_to_resistance,
    electric_field_from_weighted_winds,
    electric_field_on_grid,
    hall_geometry_tensor,
    joule_heating_from_current,
    pedersen_geometry_tensor,
    resistance_tensor_on_grid,
)
from pynamit.simulation.electrodynamics.magnetic_boundary import (
    boundary_jr_to_gap_Br_matrix,
    poloidal_potential_to_gridded_JS_operator,
)


@pytest.mark.parametrize("varying_wind", [False, True])
def test_thin_sheet_closure_is_the_limit_of_three_dimensional_ohms_law(varying_wind):
    """Invert 3-D conductivity with Jr=0 and height-dependent winds."""
    rng = np.random.default_rng(301)
    b = rng.normal(size=(7, 3))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    b[-1] = [0.0, 0.6, 0.8]  # The resistive formulation also exists at the dip equator.
    B = 4e-5 * b
    sigmaP = rng.uniform(0.2, 2.0, size=(3, 7))
    sigmaH = rng.uniform(0.2, 3.0, size=(3, 7))
    dz = np.array([1.0, 2.0, 1.5])[:, None]
    winds = rng.uniform(-100, 100, size=(3, 7, 3))
    winds[..., 0] = 0.0
    if not varying_wind:
        winds[:] = winds[0]
    SigmaP = np.sum(sigmaP * dz, axis=0)
    SigmaH = np.sum(sigmaH * dz, axis=0)
    uP = np.sum((sigmaP * dz)[..., None] * winds, axis=0) / SigmaP[:, None]
    uH = np.sum((sigmaH * dz)[..., None] * winds, axis=0) / SigmaH[:, None]
    motional = np.cross(winds, B)
    q = np.sum(
        dz[..., None] * (sigmaP[..., None] * motional + sigmaH[..., None] * np.cross(b, motional)),
        axis=0,
    )
    J = rng.uniform(-0.1, 0.1, size=(7, 3))
    J[:, 0] = 0.0

    etaP, etaH = conductance_to_resistance(SigmaP, SigmaH)
    P = pedersen_geometry_tensor(b[:, 1], b[:, 2], b[:, 0])
    resistance = resistance_tensor_on_grid(etaP, etaH, P, hall_geometry_tensor(b[:, 0]))
    Ewind = electric_field_from_weighted_winds(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
        u_p_theta=uP[:, 1],
        u_p_phi=uP[:, 2],
        u_h_theta=uH[:, 1],
        u_h_phi=uH[:, 2],
        magnetic_field=B.T,
        magnetic_unit_vector=b.T,
        etaP=etaP,
        etaH=etaH,
    )
    actual = np.asarray(electric_field_on_grid(J[:, 1:].T, resistance)) + Ewind

    # Columns of b-cross are independently computed with numpy.cross.
    bb = b[..., None] * b[:, None, :]
    b_cross = np.cross(b[:, None, :], np.eye(3)[None, :, :]).transpose(0, 2, 1)
    conductivity = (
        SigmaP[:, None, None] * (np.eye(3) - bb) + SigmaH[:, None, None] * b_cross + 1e7 * bb
    )
    expected = np.linalg.solve(conductivity, (J - q)[..., None])[..., 0]
    np.testing.assert_allclose(actual.T, expected[:, 1:], rtol=2e-6, atol=3e-8)
    if not varying_wind:
        np.testing.assert_allclose(
            np.asarray(Ewind).T, -np.cross(winds[0], B)[:, 1:], rtol=1e-13, atol=1e-17
        )

    # For the single-wind sheet, friction is J_perp dot E'_perp;
    # the Hall term cannot contribute to heating.
    Jperp = J - np.sum(J * b, axis=1)[:, None] * b
    expected_heating = np.asarray(etaP) * np.sum(Jperp**2, axis=1)
    actual_heating = joule_heating_from_current(J[:, 1:].T, etaP, P)
    np.testing.assert_allclose(actual_heating, expected_heating, rtol=1e-13, atol=1e-19)


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_poloidal_current_satisfies_ampere_jump_from_scalar_potentials(degree):
    """Differentiate V below/above and apply r-hat cross delta B."""
    grid = SphericalGrid(theta=[17.0, 54.0, 101.0, 155.0], phi=[0.0, 30.0, 150.0, 270.0])
    basis = SHBasis(max_degree=degree, max_order=0)
    potential = np.zeros(basis.coefficient_count)
    potential[-1] = 2e-9  # PynaMIT's private poloidal coefficient, in tesla.
    theta = np.deg2rad(grid.theta)
    harmonic = np.polynomial.Legendre.basis(degree)
    dY = -np.sin(theta) * harmonic.deriv()(np.cos(theta))
    # V-=-R(n+1)kY and V+=Rn kY at R; Btheta=-(1/R)dV/dtheta.
    Btheta_below = (degree + 1) * potential[-1] * dY
    Btheta_above = -degree * potential[-1] * dY
    expected = np.stack((np.zeros(grid.size), (Btheta_above - Btheta_below) / MU0))
    operator = poloidal_potential_to_gridded_JS_operator(
        SolidHarmonicOperators(basis), SphericalTransform(basis, grid)
    )
    np.testing.assert_allclose(operator(potential), expected, rtol=1e-13, atol=1e-18)


@pytest.mark.parametrize("radial_sign", [-1, 1])
def test_radial_uniform_sheet_matches_analytic_faraday_decay_and_hall_forcing(
    backend, monkeypatch, radial_sign
):
    """Pedersen damps each mode; reversing b reverses Hall forcing."""
    original_field = MainField.field_components

    def signed_radial_field(self, *args, **kwargs):
        return tuple(radial_sign * value for value in original_field(self, *args, **kwargs))

    # The public radial model points outward. Inject the inward case
    # to check Hall reversal through the complete response.
    monkeypatch.setattr(MainField, "field_components", signed_radial_field)
    simulation = Simulation(
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=False,
        integrator="exponential",
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=np.full(grid.size, 5.0), hall=np.full(grid.size, 8.0), grid=grid
    )
    geometry, response = simulation.geometry, simulation.response
    n = geometry.poloidal_basis.n
    Br0 = np.linspace(-3e-9, 4e-9, n.size)
    jr = np.linspace(1e-8, -2e-8, n.size)
    Eforcing, solved_jr = response.solve_noninductive_response(boundary_jr=jr)
    etaP, etaH = 5.0 / 89.0, 8.0 / 89.0
    rate = etaP * (2 * n + 1) / (MU0 * geometry.RI)
    expected_derivative = -rate * Br0 - radial_sign * etaH * jr
    np.testing.assert_allclose(solved_jr, jr, rtol=1e-12, atol=1e-21)
    np.testing.assert_allclose(
        induced_Br_time_derivative(response, Br0, Eforcing),
        expected_derivative,
        rtol=1e-11,
        atol=1e-22,
    )
    equilibrium = -radial_sign * etaH * jr / rate
    np.testing.assert_allclose(
        equilibrium_induced_Br(response, Eforcing), equilibrium, rtol=1e-11, atol=1e-20
    )
    duration = 0.7
    expected = equilibrium + (Br0 - equilibrium) * np.exp(-rate * duration)
    np.testing.assert_allclose(
        evolve_induced_Br(response, Br0, duration, Eforcing), expected, rtol=1e-11, atol=1e-21
    )

    # Continuing a degree-n interior potential gives r**(n-1), not r**n.
    observation_grid = SphericalGrid(theta=[20.0, 65.0, 110.0], phi=[0.0, 10.0, 20.0])
    ground = build_ground_magnetic_field_operators(
        geometry, observation_grid, ground_radius=0.8 * geometry.RI
    )
    for degree in (1, 2, 4):
        mode = np.zeros(n.size)
        mode[np.flatnonzero((n == degree) & (geometry.poloidal_basis.m == 0))[0]] = 1e-9
        theta = np.deg2rad(observation_grid.theta)
        harmonic = np.polynomial.Legendre.basis(degree)
        radial = 1e-9 * 0.8 ** (degree - 1) * harmonic(np.cos(theta))
        south = (
            -1e-9 * 0.8 ** (degree - 1) / degree * np.sin(theta) * harmonic.deriv()(np.cos(theta))
        )
        np.testing.assert_allclose(ground["radial"](mode), radial, rtol=1e-13, atol=1e-23)
        np.testing.assert_allclose(
            ground["tangential"](mode),
            np.stack((south, np.zeros_like(south))),
            rtol=1e-13,
            atol=1e-23,
        )


@pytest.mark.parametrize("outer_boundary", [None, 3.0])
def test_gap_field_integral_matches_an_analytic_twisted_radial_field(outer_boundary):
    """A twisted current gives an analytic degree-two magnetic field."""
    # Set RI=1. B_r=1/r², B_phi=c sin(theta)/r is divergence free;
    # theta is constant on its field lines and dphi/dr=c.
    # jr(RI)=cos(theta) continues to j_phi=c sin(theta)cos(theta)/r.
    # Each shell dr has poloidal k_2=mu0*c*dr/(15r), from Ampere's jump.
    c = 0.3

    def evaluate(grid, radius):
        return np.array(
            (
                np.full(grid.size, radius**-2),
                np.zeros(grid.size),
                c * np.sin(np.deg2rad(grid.theta)) / radius,
            )
        )

    def footpoint(*, r_dest, r, theta, phi):
        return theta, phi - np.rad2deg(c * (r - r_dest))

    main_field = SimpleNamespace(evaluate=evaluate, map_along_field_lines=footpoint)
    source_basis = SHBasis(max_degree=1, max_order=0)
    poloidal_basis = SHBasis(max_degree=2, max_order=0)
    grid = SphericalGrid(theta=np.linspace(5, 175, 30), phi=np.zeros(30))
    operator = boundary_jr_to_gap_Br_matrix(
        main_field,
        source_basis,
        SphericalTransform(poloidal_basis, grid),
        SolidHarmonicOperators(poloidal_basis),
        ionosphere_radius=1.0,
        integration_radii=np.linspace(1, 3, 161),
        boundary_radius=outer_boundary,
    )
    # Without an outer image: Br_2(RI)=2 mu0*c/5 integral_1^3 dr/r².
    integral = 1 - 1 / 3
    if outer_boundary is not None:
        # The image adds -r³/RM^5 inside the integral, followed by
        # the multiple-reflection factor 1/(1-(RI/RM)^5).
        integral = (integral - (3**4 - 1) / (4 * 3**5)) / (1 - 3**-5)
    np.testing.assert_allclose(
        operator[:, 0], [0, 2 * MU0 * c / 5 * integral], rtol=3e-5, atol=1e-21
    )
