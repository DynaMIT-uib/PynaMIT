import numpy as np

from pynamit.math.constants import RE
from pynamit.postprocess import build_ground_magnetic_response_operators
from pynamit.primitives.field_spec import FieldSpec
from pynamit.primitives.grid import Grid
from pynamit.spherical_harmonics.sh_basis import SHBasis


def test_ground_response_matches_manual_sh_formula():
    basis = SHBasis(6, 4)
    state_spec = FieldSpec(basis=basis, field_type="scalar", mean_free=True)
    grid = Grid(lat=np.array([60.0, 70.0]), lon=np.array([10.0, 30.0]))
    ri = RE + 110e3

    ops = build_ground_magnetic_response_operators(
        state_spec=state_spec,
        ground_grid=grid,
        ionosphere_radius=ri,
        ground_radius=RE,
    )

    degrees = state_spec.n
    ve_to_ground = state_spec.radial_shift_Ve(ri, RE)
    manual_radial = (
        state_spec.get_evaluation_matrix(grid)
        * (ve_to_ground * (-(ri**2) * state_spec.laplacian(ri)))[None, :]
    )
    manual_horizontal = (
        state_spec.get_gradient_matrix(grid)
        * (((degrees + 1.0) * ve_to_ground)[None, None, :])
    )

    rng = np.random.default_rng(0)
    m_ind = rng.standard_normal(state_spec.index_length)

    np.testing.assert_allclose(ops.radial_matrix, manual_radial)
    np.testing.assert_allclose(ops.horizontal_matrix, manual_horizontal)
    np.testing.assert_allclose(ops.evaluate_radial(m_ind), manual_radial @ m_ind)
    np.testing.assert_allclose(
        ops.evaluate_horizontal(m_ind),
        np.tensordot(manual_horizontal, m_ind, axes=([2], [0])),
    )
