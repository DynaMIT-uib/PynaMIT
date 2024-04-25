""" 

Testing differentiation on internal grid points (not including ghost cells)

"""

import matplotlib.pyplot as plt
from cupyx.scipy.sparse import coo_matrix
import cupy as np
from ppigrf.ppigrf import igrf_gc, igrf_V
import datetime
from pynamit.cubedsphere import cubedsphere, diffutils
from functools import reduce

def test_differentiation():
    # set up projection and make a grid (not using the grid class)
    R = np.array(6371.2e3)
    p = cubedsphere.CSprojection()
    N = 40 # number of grid points in each direction per block
    dxi = np.pi / 2 / (N - 1)
    #deta = np.pi / 2 / (N - 1)
    block, xi, eta = np.meshgrid(np.arange(6), np.linspace(-np.pi/4, np.pi/4, N), np.linspace(-np.pi/4, np.pi/4, N), indexing = 'ij')
    r, theta, phi = p.cube2spherical(xi, eta, r = R, block = block, deg = True)

    # calculate IGRF potential and spherical vector components at the grid points
    V                = np.array(igrf_V (np.asnumpy(r), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1))).squeeze()
    Br, Btheta, Bphi = np.array(igrf_gc(np.asnumpy(r), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1)))


    # - sanity check differentiate V wrt to R and get back Br
    rs  = np.array([-2, -1, 0, 1, 2]) * 1e3 + R
    Vs  = np.vstack([np.array(igrf_V( np.asnumpy(rr), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1))) for rr in rs])
    r_stencil = diffutils.stencil(rs - R)
    Br_num = -np.sum(Vs * r_stencil.reshape((-1, 1, 1, 1)), axis = 0)

    Br_num_matches_Br = np.allclose(Br_num.flatten() - Br.flatten(), 0)

    print('Testing differentiation on grid points that are internal to the blocks:')
    print('Numerically calculated Br     matches Br     calculated with spherical harmonics: {}'.format(Br_num_matches_Br))

    assert Br_num_matches_Br

    # differentiate wrt to xi and eta
    stencil_points = [-2, -1, 0, 1, 2]
    stencil = diffutils.stencil(np.array(stencil_points) * dxi)
    #dxi_dxi = reduce(lambda x, y: x+ y, [stencil[i] * xi [:, 2 + stencil_points[i] :  (N - 2) + stencil_points[i], 2     :                            -2] for i in range(len(stencil_points))])
    dV_dxi  = reduce(lambda x, y: x+ y, [stencil[i] * V  [:, 2 + stencil_points[i] :  (N - 2) + stencil_points[i], 2     :                            -2] for i in range(len(stencil_points))])

    #det_det = reduce(lambda x, y: x+ y, [stencil[i] * eta[:, 2                     :            -2, 2 + stencil_points[i] :  (N - 2) + stencil_points[i]] for i in range(len(stencil_points))])
    dV_det  = reduce(lambda x, y: x+ y, [stencil[i] * V  [:, 2                     :            -2, 2 + stencil_points[i] :  (N - 2) + stencil_points[i]] for i in range(len(stencil_points))])

    # calcualte the contravariant components of the gradient:
    gc = p.get_metric_tensor(xi[0, 2:-2, 2:-2], eta[0, 2:-2, 2:-2], r = R, covariant = False)
    u1 = gc[:, 0, 0].reshape((1, N - 4, N - 4)) * dV_dxi + gc[:, 0, 1].reshape((1, N - 4, N - 4)) * dV_det
    u2 = gc[:, 0, 1].reshape((1, N - 4, N - 4)) * dV_dxi + gc[:, 1, 1].reshape((1, N - 4, N - 4)) * dV_det
    u3 = gc[:, 2, 2].reshape((1, N - 4, N - 4)) * Br_num[:, 2:-2, 2:-2]
    u  = np.vstack((u1.flatten(), u2.flatten(), u3.flatten()))


    Ps_inv = p.get_Ps(xi[:, 2:-2, 2:-2], eta[:, 2:-2, 2:-2], r = R, block = block[:, 2:-2, 2:-2], inverse = True)
    Q      = p.get_Q(90 - theta[:, 2:-2, 2:-2], R)
    Ps_normalized = np.einsum('nij, njk -> nik', Q, Ps_inv)
    u_east, u_north, u_r = -np.einsum('nij, nj -> ni', Ps_normalized, u.T).T

    Btheta_num_matches_Btheta = np.allclose(-u_north.flatten() - Btheta.squeeze()[:, 2:-2, 2:-2].flatten(), 0)
    Bphi_num_matches_Bphi = np.allclose( u_east.flatten()  - Bphi.squeeze()[:, 2:-2, 2:-2].flatten(), 0)

    print('Numerically calculated Btheta matches Btheta calculated with spherical harmonics: {}'.format(Btheta_num_matches_Btheta))
    print('Numerically calculated Bphi   matches Bphi   calculated with spherical harmonics: {}'.format(Bphi_num_matches_Bphi))

    assert Btheta_num_matches_Btheta
    assert Bphi_num_matches_Bphi

    # Now to calculate the Laplacian of V:
    stencil2 = diffutils.stencil(np.array(stencil_points) * dxi, order = 2)

    dV2_dxi2 = reduce(lambda x, y: x+ y, [stencil2[i] * V[:, 2 + stencil_points[i] :  (N - 2) + stencil_points[i], 2                     :                           -2] for i in range(len(stencil_points))])
    dV2_det2 = reduce(lambda x, y: x+ y, [stencil2[i] * V[:, 2                     :                           -2, 2 + stencil_points[i] :  (N - 2) + stencil_points[i]] for i in range(len(stencil_points))])

    # second-order derivative of the radial component:
    r_stencil2 = diffutils.stencil(rs - R, order = 2)
    dV2_dr2 = np.sum(Vs * r_stencil2.reshape((-1, 1, 1, 1)), axis = 0)

    # construct a 2D stencil for calculating cross-derivative
    stencil_cross = diffutils.get_2D_stencil_coefficients(np.array(stencil_points) * dxi, np.array(stencil_points) * dxi, derivative = 'xy')
    dV2_dxideta = reduce(lambda x, y: x+ y, [stencil_cross[i] * V[:, 2 + stencil_points[i] :  (N - 2) + stencil_points[i], 2 + stencil_points[i] :  (N - 2) + stencil_points[i]] for i in range(len(stencil_points))])

    # stitch it together:
    del2V = gc[:, 0, 0].reshape((1, N - 4, N - 4)) * dV2_dxi2 + gc[:, 1, 1].reshape((1, N - 4, N - 4)) * dV2_det2 + 2 * gc[:, 0, 1].reshape((1, N - 4, N - 4)) * dV2_dxideta + dV2_dr2[:, 2:-2, 2:-2] - 2 * Br_num[:, 2:-2, 2:-2] / R
    #radial = dV2_dr2[:, 2:-2, 2:-2] - 2 * Br_num[:, 2:-2, 2:-2] / R
    #horizontal = gc[:, 0, 0].reshape((1, N - 4, N - 4)) * dV2_dxi2 + gc[:, 1, 1].reshape((1, N - 4, N - 4)) * dV2_det2 + 2 * gc[:, 0, 1].reshape((1, N - 4, N - 4)) * dV2_dxideta

    del2V_is_small = np.allclose(del2V, 0)
    print('You have discovered magnetic monopoles: {} (which is {})'.format(not del2V_is_small, 'good' if del2V_is_small else 'bad'))
    print('Is del2V really small though? It doesnt seem to get smaller when increasing the resolution...')

    assert del2V_is_small


    ####
    # Trying to do the same, only with differentiation matrices

    Ns = 2
    shape = (6, N, N)
    size = np.product(np.array(shape)).item()

    p = cubedsphere.CSprojection()
    h = p.xi(1, N) - p.xi(0, N) # step size between each grid cell

    # with ij indexing, (i, xi) vary along the first (numpy vertical) axis, and (j, eta) along the second (numpy horizontal) axis. 
    k, i, j = np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing = 'ij')

    # indices for inner grid cells, for which no interpolation is required
    k_inner, i_inner, j_inner = k[:, Ns:-Ns, Ns:-Ns], i[:, Ns:-Ns, Ns:-Ns], j[:, Ns:-Ns, Ns:-Ns]

    # set up differentiation stencil
    stencil_points  = np.hstack((np.arange(-Ns, 0), np.arange(1, Ns + 1)))
    Nsp = len(stencil_points)
    stencil_weight1 = diffutils.stencil(stencil_points, order = 1, h = h) # 1st order differentiation
    stencil_weight2 = diffutils.stencil(stencil_points, order = 2, h = h) # 2nd order differentiation

    i_diff           = np.hstack([i_inner.flatten() + _ for _ in stencil_points])
    j_diff           = np.hstack([j_inner.flatten() + _ for _ in stencil_points])
    k_const, i_const, j_const = np.tile(k_inner.flatten(), Nsp), np.tile(i_inner.flatten(), Nsp), np.tile(j_inner.flatten(), Nsp)
    weights1  = np.repeat(stencil_weight1, k_inner.size)
    weights2  = np.repeat(stencil_weight2, k_inner.size)
    cols_xi     = np.ravel_multi_index((k_const, i_diff , j_const), shape)
    cols_eta    = np.ravel_multi_index((k_const, i_const, j_diff ), shape)
    #cols_xi_2d  = np.ravel_multi_index((k_const, i_diff , j_diff ), shape)
    #cols_eta_2d = np.ravel_multi_index((k_const, i_diff , j_diff ), shape)

    # cols = ravel_multi_index_cs(k, i, j, shape) -- some will have -1's, and those should be interpolated...?
    # interpolation_points, weights = get_interpolation_kernel(k[cols == -1], i[cols == -1], j[cols == -1])
    # cols.insert() <- this could be done by expnadning cols to 2D - with one dimension the same size as the interpolation kernel. It would create lots of duplicate values, that would be summed in the sparse matrix in the end. 
    # -- so if I want to get a function value at 3, -3, -5, I can call get_interpoaltion_kernel and from the output construct a matrix that interpolates to these points...


    rows = np.tile(np.ravel_multi_index((k_inner.flatten(), i_inner.flatten(), j_inner.flatten()), shape), Nsp)


    Dxi1  = coo_matrix((weights1, (rows, cols_xi )), shape = (size, size))
    Deta1 = coo_matrix((weights1, (rows, cols_eta)), shape = (size, size))
    Dxi2  = coo_matrix((weights2, (rows, cols_xi )), shape = (size, size))
    Deta2 = coo_matrix((weights2, (rows, cols_eta)), shape = (size, size))


    # to test it, recalculate IGRF values in the grid chosen for the differentiation:
    xi, eta = p.xi(i, N), p.eta(j, N)
    r, theta, phi = p.cube2spherical(xi, eta, r = R, block = k, deg = True)
    V                = np.array(igrf_V (np.asnumpy(r), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1))).flatten()
    Br, Btheta, Bphi = map(np.squeeze, np.array(igrf_gc(np.asnumpy(r), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1))))

    dVdxi    = Dxi1 .dot(V)
    dVdeta   = Deta1.dot(V)
    #dV2dxi2  = Dxi2 .dot(V)
    #dV2deta2 = Deta2.dot(V)


    gc = p.get_metric_tensor(xi, eta, r = R, covariant = False)
    u1 = gc[:, 0, 0].flatten() * dVdxi + gc[:, 0, 1].flatten() * dVdeta
    u2 = gc[:, 0, 1].flatten() * dVdxi + gc[:, 1, 1].flatten() * dVdeta
    u3 = np.ones(size)
    u  = np.vstack((u1.flatten(), u2.flatten(), u3.flatten()))


    Ps_inv = p.get_Ps(xi, eta, r = R, block = k, inverse = True)
    Q      = p.get_Q(90 - theta, R)
    Ps_normalized = np.einsum('nij, njk -> nik', Q, Ps_inv)
    u_east, u_north, u_r = -np.einsum('nij, nj -> ni', Ps_normalized, u.T).T

    sparse_Btheta_matches_Btheta = np.allclose( (-u_north.reshape(shape) - Btheta.squeeze()).reshape(shape)[k_inner, i_inner, j_inner], 0)
    sparse_Bphi_matches_Bphi = np.allclose( ( u_east.reshape(shape)  - Bphi.squeeze()  ).reshape(shape)[k_inner, i_inner, j_inner], 0)

    # compare the values on the inner grid cells (the numerical derivatives are all zero on the cells that would require ghost cells)
    print('Using sparse differentiation matrix: Numerically calculated Btheta matches Btheta calculated with spherical harmonics: {}'.format(sparse_Btheta_matches_Btheta))
    print('Using sparse differentiation matrix: Numerically calculated Bphi   matches Bphi   calculated with spherical harmonics: {}'.format(sparse_Bphi_matches_Bphi))

    assert sparse_Btheta_matches_Btheta
    assert sparse_Bphi_matches_Bphi

    print('\nTesting differention on FULL grid, using interpolation matrix')
    Ni = 4 # number of interpolation points
    shape = (6, N, N)
    size = 6 * N * N


    k, i, j = p.get_gridpoints(N)
    kk, ii, jj = p.get_gridpoints(N, flat = True)

    # get differentiation matrices:
    Dxi, Deta = p.get_Diff(N, coordinate = 'both', Ns = Ns, Ni = Ni, order = 1)


    xi, eta = p.xi(ii, N), p.eta(jj, N)
    r, theta, phi = p.cube2spherical(xi, eta, r = R, block = kk, deg = True)
    V                = np.array(igrf_V (np.asnumpy(r), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1))).squeeze()
    Br, Btheta, Bphi = map(np.squeeze, np.array(igrf_gc(np.asnumpy(r), np.asnumpy(theta), np.asnumpy(phi), datetime.datetime(2020, 1, 1))))

    dVdxi    = Dxi .dot(V)
    dVdeta   = Deta.dot(V)

    gc = p.get_metric_tensor(xi, eta, r = R, covariant = False)
    u1 = gc[:, 0, 0] * dVdxi + gc[:, 0, 1] * dVdeta
    u2 = gc[:, 0, 1] * dVdxi + gc[:, 1, 1] * dVdeta
    u3 = np.ones(size)
    u  = np.vstack((u1, u2, u3))

    Ps_inv = p.get_Ps(xi, eta, r = R, block = kk, inverse = True)
    Q      = p.get_Q(90 - theta, R)
    Ps_normalized = np.einsum('nij, njk -> nik', Q, Ps_inv)
    u_east, u_north, u_r = -np.einsum('nij, nj -> ni', Ps_normalized, u.T).T

    east_matches_Bphi = np.all(np.isclose(u_east - Bphi, 0))
    north_matches_Btheta = np.all(np.isclose(u_north + Btheta, 0))

    print('Numerically calculated eastward  components consistent with SH: {}'.format(east_matches_Bphi))
    print('Numerically calculated northward components consistent with SH: {}'.format(north_matches_Btheta))

    assert east_matches_Bphi
    assert north_matches_Btheta


    print('\nTesting 2D differentiation (Laplacian) on full sphere')
    Ns = 2
    stencil_points  = np.hstack((np.arange(-Ns, 0), np.arange(1, Ns + 1)))
    Nsp = len(stencil_points)

    h = p.xi(1, N) - p.xi(0, N) # step size between each grid cell

    # make a stencil that has a cross + first diagonal points (no clue what is a good idea here)
    stencil_i = np.hstack( ( stencil_points, np.zeros(2*Ns), np.array([-1, -1,  1, 1]) ) )
    stencil_j = np.hstack( ( np.zeros(2*Ns), stencil_points, np.array([-1,  1, -1, 1]) ) )

    stencil_cross = diffutils.get_2D_stencil_coefficients(stencil_i/h, stencil_j/h, derivative='xy')

    # get matrices for second order differentation along single dimension:
    Dxi2, Deta2 = p.get_Diff(N, coordinate = 'both', Ns = Ns, Ni = Ni, order = 2)

    # construct cross-term derivative
    i_diff           = np.hstack([i + _ for _ in stencil_i])
    j_diff           = np.hstack([j + _ for _ in stencil_j])
    k_const = np.tile(k, len(stencil_i))
    weights  = np.repeat(stencil_cross, size)

    rows = np.tile(np.ravel_multi_index((k, i, j), shape), len(stencil_i))
    Ddxideta = p.get_interpolation_matrix(k_const, i_diff , j_diff, N, Ni, rows = rows, weights = weights)

    del2v_1 = gc[:, 0, 0] * Dxi2.dot(V) + gc[:, 1, 1] * Deta2.dot(V)
    del2v_2 = 2 * gc[:, 0, 1] * Ddxideta.dot(V)
    ddr = dV2_dr2.flatten() - 2 * Br_num.flatten() / R
    del2v = del2v_1 + del2v_2 + ddr



    mismatch = del2v*1e12
    mismatches = np.split(mismatch, 6)

    ii, jj = np.meshgrid(np.arange(N + 10), np.arange(N + 10), indexing = 'ij')
    fig, axes = plt.subplots(ncols = 3, nrows = 2)
    cc = 0
    for ax, mm in zip(axes.flatten(), mismatches):
        ax.scatter(np.asnumpy(i[0]), np.asnumpy(j[0]), c = np.asnumpy(mm), cmap = plt.cm.bwr, vmin = -10, vmax = 10)
        ax.set_aspect('equal')
        ax.set_title(str(cc))
        cc+=1
    #plt.show()

