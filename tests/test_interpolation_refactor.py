import numpy as np
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.grid_basis import GridBasis
from pynamit.primitives.grid import Grid
from pynamit.primitives.interpolation import CSInterpolator, UnstructuredInterpolator


def test_interpolation_identity():
    """Verify CSInterpolator matches/approximates UnstructuredInterpolator."""
    N = 20
    basis = CSBasis(N)

    # Define a smooth scalar function: Y_1^0 = cos(theta)
    # Note: CSBasis uses degrees for theta/phi arrays
    theta_src = basis.theta
    phi_src = basis.phi

    # Source values
    val_src = np.cos(np.deg2rad(theta_src))

    # Target grid: higher resolution or shifted
    # Let's use points offset by half a cell to maximize interpolation error
    # For simulation, we interact with mapped grids.
    # Let's just create a random target grid on the sphere
    target_theta = np.linspace(10, 170, 50)
    target_phi = np.linspace(0, 360, 50)
    tt, tp = np.meshgrid(target_theta, target_phi)
    tt = tt.flatten()
    tp = tp.flatten()

    # 1. OPTIMIZED (CSInterpolator)
    # The basis should automatically use it
    cs_interp = CSInterpolator(N)
    val_opt = cs_interp.interpolate_scalar(val_src, tt, tp)

    # 2. LEGACY (UnstructuredInterpolator)
    legacy_interp = UnstructuredInterpolator(theta_src, phi_src)
    val_leg = legacy_interp.interpolate_scalar(val_src, tt, tp)

    # Compare
    diff = np.abs(val_opt - val_leg)
    max_diff = np.nanmax(diff)
    print(f"Max Difference optimized vs legacy: {max_diff:.3e}")

    # Expectation: They won't be identical (Bilinear Quad vs Linear Tri)
    # But they should be close (order of O(h^2) or similar)
    # With N=20, error should be decent.
    # Allow some tolerance.
    assert max_diff < 1e-2, "Optimization deviated significantly from legacy!"


def test_vector_interpolation_identity():
    """Verify Vector Interpolation matches/approximates legacy."""
    N = 20
    basis = CSBasis(N)

    # Source Vector: Radial dipole B-field
    # B = -2 g10 cos(theta) r-hat - g10 sin(theta) theta-hat
    # Let g10 = -30000nT.
    # B_r = -2 * (-30000) * cos(th) = 60000 cos(th)
    # B_th = -(-30000) * sin(th) = 30000 sin(th)
    # B_phi = 0
    # PynaMIT conventions: u_r, u_north=-B_th, u_east=B_phi

    th_rad = np.deg2rad(basis.theta)
    u_r = 60000 * np.cos(th_rad)
    u_north = -30000 * np.sin(th_rad)
    u_east = np.zeros_like(u_r)

    # Target grid
    theta_tgt = np.array([45.0, 90.0, 135.0])
    phi_tgt = np.array([0.0, 45.0, 90.0])

    # 1. OPTIMIZED
    cs_interp = CSInterpolator(N)
    v1_opt, v2_opt, v3_opt = cs_interp.interpolate_vector(u_east, u_north, u_r, theta_tgt, phi_tgt)

    # 2. LEGACY
    legacy_interp = UnstructuredInterpolator(basis.theta, basis.phi)
    v1_leg, v2_leg, v3_leg = legacy_interp.interpolate_vector(
        u_east, u_north, u_r, theta_tgt, phi_tgt
    )

    print("\nVector Comparison:")
    print("Opt u_r:", v3_opt)
    print("Leg u_r:", v3_leg)

    np.testing.assert_allclose(v1_opt, v1_leg, rtol=0.1, atol=1000.0)  # 1000nT tolerance
    np.testing.assert_allclose(v2_opt, v2_leg, rtol=0.1, atol=1000.0)
    np.testing.assert_allclose(v3_opt, v3_leg, rtol=0.1, atol=1000.0)


if __name__ == "__main__":
    test_interpolation_identity()
    test_vector_interpolation_identity()
