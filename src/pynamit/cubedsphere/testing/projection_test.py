""" 
Testing of the cubedsphere CSprojection class

1) Conversions to / from cubed sphere, Cartesian, and spherical coordinates
2) Conversions to / from cubed sphere, Cartesians, and spherical components
3) Plot cubed sphere grid and vector components in Cartesian 3D and on Cartopy projeciton
4) Plot eastward and northward vector fields on cubed sphere blocks 

"""

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import numpy as np
from igrf import igrf_gc, igrf_V
import datetime
from cubedsphere import cubedsphere, diffutils, spherical
from testutils import Geocentric_to_PlateCarree_vector_components
from importlib import reload
import cartopy.crs as ccrs
reload(cubedsphere)


p = cubedsphere.CSprojection()


### 1) Test conversions to / from cubed sphere, Cartesians, and spherical coordinates
#####################################################################################
N = 5000 # number of test points
xx, yy, zz = 2 * np.random.random(N) - 1, 2 * np.random.random(N) - 1, 2 * np.random.random(N) - 1
rr = np.sqrt(xx**2 + yy**2 + zz**2)
iii = rr <= 1
rr = rr[iii]
xx = xx[iii]
yy = yy[iii]
zz = zz[iii]

lon = np.rad2deg( np.arctan2(yy, xx) )
lat = np.rad2deg( np.arcsin(zz / rr ) ) 

block = p.block(lon, lat)
xi, eta, block = p.geo2cube(lon, lat, block)

r, theta, phi = p.cube2spherical(xi, eta, r = rr, block = block)
print('Conversion from (lon, lat, block) to (xi, eta) and back works: {}'.format(np.allclose(90 - np.rad2deg(theta) - lat, 0) & np.allclose(np.rad2deg(phi) - lon, 0)))

x, y, z = p.cube2cartesian(xi, eta, r = rr, block = block)
print('Conversion from (x, y, z, block ) to (xi, eta) and back works: {}'.format(np.allclose(x - xx, 0) & np.allclose(y - yy, 0) & np.allclose(z - zz, 0)))

### 2) Test conversions to / from cubed sphere, Cartesians, and spherical components
####################################################################################
N = xx.size
Axyz = 2 * np.random.random((3, N)) - 1 # (3, N) random vector components

Pc    = p.get_Pc(xi, eta, r = rr, block = block)
Pcinv = p.get_Pc(xi, eta, r = rr, block = block, inverse = True)
Ps    = p.get_Ps(xi, eta, r = rr, block = block)
Psinv = p.get_Ps(xi, eta, r = rr, block = block, inverse = True)
Q     = p.get_Q(lat, rr)

A = np.einsum('nij, nj -> ni', Pc, Axyz.T).T
Axyz_ = np.einsum('nij, nj -> ni', Pcinv, A.T).T
Asph  = np.einsum('nij, nj -> ni', Psinv, A.T).T
Asph_normed = np.einsum('nij, nj -> ni', Q, Asph.T).T
A_ = np.einsum('nij, nj -> ni', Ps, Asph.T).T

print('Converting vector components between cubed sphere and Cartesian give consistent results: {}'.format(np.allclose(Axyz - Axyz_, 0)))
print('Converting vector components between cubed sphere and spherical give consistent results: {}'.format(np.allclose(A_ - A, 0)))
print('Cartesian and normalizd spherical vectors have the same norm: {}'.format(np.allclose(np.linalg.norm(Asph_normed, axis = 0) - np.linalg.norm(Axyz, axis = 0), 0)))



### 3) Plot cubed sphere grid and vector components in Cartesian 3D and on Cartopy projeciton
#############################################################################################
print('Plotting cubed sphere grid and vector components in Cartesian 3D and on Cartopy projection')

phi0, lat0 = 0, 0
N = 16 # number of grid points in each direction per block (should be even)

fig = plt.figure(figsize = (12, 8))
axxyz1 = fig.add_subplot(233, projection = '3d')
axxyz2 = fig.add_subplot(236, projection = '3d')

cartopyprojection1 = ccrs.Orthographic(phi0, lat0 + 20)
cartopyprojection2 = ccrs.Orthographic(phi0 + 180, lat0 + 70)
axg1 = fig.add_subplot(231, projection = cartopyprojection1)
axg2 = fig.add_subplot(232, projection = cartopyprojection1)
axg3 = fig.add_subplot(234, projection = cartopyprojection2)
axg4 = fig.add_subplot(235, projection = cartopyprojection2)
for ax in [axg1, axg2, axg3, axg4]:
    ax.coastlines(zorder = 3)



xi, eta = np.meshgrid(np.linspace(-np.pi/4, np.pi/4, N), np.linspace(-np.pi/4, np.pi/4, N), indexing = 'ij')
ones  = np.ones_like(xi).flatten()
zeros = np.zeros_like(eta).flatten()
rs    = np.zeros_like(eta).flatten()
Axis  = np.vstack((ones , zeros, rs)).T
Aetas = np.vstack((zeros, ones, rs)).T

print('--- some Cartopy / matplotlib warnings:')
for i in range(6):
    C = 'C' + str(i)

    # Spherical / Cartopy:
    # --------------------
    r, theta, phi = p.cube2spherical(xi, eta, block = i)
    lo, la = np.rad2deg(phi), 90 - np.rad2deg(theta)
    lon, lat = np.rad2deg(phi).flatten(), 90 - np.rad2deg(theta).flatten()
    Ps_inv = p.get_Ps(xi, eta, r = 1, block = i, inverse = True)
    Q      = p.get_Q(lat, r.flatten())
    Ps_normalized = np.einsum('nij, njk -> nik', Q, Ps_inv) # multiply Ps_inv by Q to get normalized vector components 

    # xi-direction:
    Aeast, Anorth, Ar = np.einsum('nij, nj -> ni', Ps_normalized, Axis).T
    assert np.all(np.isclose(Ar, 0))
    norms = np.sqrt(Aeast**2 + Anorth**2)

    Ae_pc, An_pc = Geocentric_to_PlateCarree_vector_components(Aeast.flatten(), Anorth.flatten(), lat)
    axg1.quiver(lon, lat, Ae_pc, An_pc, transform = ccrs.PlateCarree(), color = C)
    axg2.quiver(lon, lat, Ae_pc, An_pc, transform = ccrs.PlateCarree(), color = C)

    # eta-direction:
    Aeast, Anorth, Ar = np.einsum('nij, nj -> ni', Ps_normalized, Aetas).T
    assert np.all(np.isclose(Ar, 0))

    Ae_pc, An_pc = Geocentric_to_PlateCarree_vector_components(Aeast.flatten(), Anorth.flatten(), lat)
    axg3.quiver(lon % 360, lat, Ae_pc, An_pc, transform = ccrs.PlateCarree(), color = C)
    axg4.quiver(lon % 360, lat, Ae_pc, An_pc, transform = ccrs.PlateCarree(), color = C)

    for ax in [axg1, axg2, axg3, axg4]:
        ax.scatter(lon, lat, color = C, transform = ccrs.PlateCarree(), s = 5, zorder = 60)

        for k in range(N):
            ax.plot(lo[k, :].flatten(), la[k, :].flatten(), color = C, linewidth = .5, linestyle = '--', transform = ccrs.Geodetic())
            ax.plot(lo[:, k].flatten(), la[:, k].flatten(), color = C, linewidth = .5, linestyle = '--', transform = ccrs.Geodetic())


    # Cartesian 3D:
    # -------------
    x, y, z = p.cube2cartesian(xi, eta, block = i)
    axxyz1.scatter(x, y, z, c = C, s = 5)
    axxyz2.scatter(x, y, z, c = C, s = 5)
    Pc = p.get_Pc(xi, eta, r = 1, block = i, inverse = True)

    Ax, Ay, Az = np.einsum('nij, nj -> ni', Pc, Axis).T
    axxyz1.quiver(x.flatten(), y.flatten(), z.flatten(), Ax, Ay, Az, length = 1e-1, color = C)

    Ax, Ay, Az = np.einsum('nij, nj -> ni', Pc, Aetas).T
    axxyz2.quiver(x.flatten(), y.flatten(), z.flatten(), Ax, Ay, Az, length = 1e-1, color = C)


# make Cartesian plots prettier:
for ax in [axxyz1, axxyz2]: 
    ax.set_axis_off()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    a = .95

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = a * np.outer(np.cos(u), np.sin(v))
    y = a * np.outer(np.sin(u), np.sin(v))
    z = a * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color = 'white')

axxyz1.set_title(r'$\xi$ direction')
axxyz2.set_title(r'$\eta$ direction')

# link the two 3D axes so that if I rotate one, the other will adjust in the same way:
def on_move(event):
    if event.inaxes == axxyz2:
        axxyz1.view_init(elev=axxyz2.elev, azim=axxyz2.azim)
    elif event.inaxes == axxyz1:
        axxyz2.view_init(elev=axxyz1.elev, azim=axxyz1.azim)
    else:
        return
    fig.canvas.draw_idle()

c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.tight_layout()


### 4) Plot eastward and northward vector fields on cubed sphere blocks 
########################################################################
print('Plotting eastward and northward vector fields on cubed sphere blocks')
fig, axes = plt.subplots(nrows = 2, ncols = 6, figsize = (18, 6))

xihd, etahd = np.meshgrid(np.linspace(-np.pi/4, np.pi/4, 400), np.linspace(-np.pi/4, np.pi/4, 400))


Aeast  = np.vstack((ones , zeros, zeros))
Anorth = np.vstack((zeros, ones , zeros))

for block in range(6):
    r_hd, theta_hd, phi_hd = p.cube2spherical(xihd, etahd, r = 1, block = block)
    la = 90 - np.rad2deg(theta_hd)
    lo = np.rad2deg(phi_hd)

    r, theta, phi = p.cube2spherical(xi, eta, r = 1, block = block)
    Ps = p.get_Ps(xi, eta, r = 1, block = block)
    Q  = p.get_Q(90 - np.rad2deg(theta), r, inverse = True)
    Ps_normalized = np.einsum('nij, njk -> nik', Ps, Q)

    Ae = np.einsum('nij, nj -> ni', Ps_normalized, Aeast.T).T
    An = np.einsum('nij, nj -> ni', Ps_normalized, Anorth.T).T
    assert np.allclose(np.hstack((Ae[2], An[2])), 0)

    axes[0, block].scatter(xi, eta, c = 'grey', zorder = 1, s = 5)
    axes[1, block].scatter(xi, eta, c = 'grey', zorder = 1, s = 5)

    axes[0, block].quiver(xi.flatten(), eta.flatten(), Ae[0], Ae[1], scale = 15)
    axes[1, block].quiver(xi.flatten(), eta.flatten(), An[0], An[1], scale = 15)

    axes[0, block].set_title('block ' + str(block) + ', eastward')
    axes[1, block].set_title('block ' + str(block) + ', northward')

    for ax in axes.T[block]:
        cs = ax.contour(xihd, etahd, la, levels = np.r_[-80:90:10], colors = 'lightgrey', linewidths = 1, zorder = 0)
        ax.clabel(cs, cs.levels, inline = True, fmt = lambda x: '{:.0f}$^\circ$N'.format(x), zorder = 0)
        for lo_ in np.r_[-180:180:30]:
            la_ = np.linspace(-90, 90, 181)
            f = p.block(lo_, la_)
            la_ = la_[f == block]

            la_[np.abs(la_) > 80] = np.nan
            xi_, eta_, _ = p.geo2cube(lo_, la_, block)
            ax.plot(xi_, eta_, zorder = 0, linewidth = 1, color = 'lightgrey')


for ax in axes.flatten():
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim(-np.pi/4, np.pi/4)
    ax.set_ylim(-np.pi/4, np.pi/4)




# 5) Plot grid with ghost cells on a map
fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(projection = cartopyprojection1)
ax.coastlines(zorder = 3)

xi, eta = np.meshgrid(np.linspace(-np.pi/4, np.pi/4, N), np.linspace(-np.pi/4, np.pi/4, N))
dxi = np.diff(xi [0   ])[0]
det = np.diff(eta[: ,0])[0]

N_extra = 5
extra_xi  = np.arange(1, N_extra + 1) * dxi
extra_eta = np.arange(1, N_extra + 1) * det
xi_larger  = np.hstack((xi [0, 0] - extra_xi[ ::-1], xi[ 0, :], xi[  0, -1] + extra_xi))
eta_larger = np.hstack((eta[0, 0] - extra_eta[::-1], eta[:, 0], eta[-1,  0] + extra_eta))
xi_, eta_ = np.meshgrid(xi_larger, eta_larger)

for i in range(6): # plot the main grid on each block:
    C = 'C' + str(i)

    # Spherical / Cartopy:
    # --------------------
    r, theta, phi = p.cube2spherical(xi, eta, block = i)
    lo, la = np.rad2deg(phi), 90 - np.rad2deg(theta)

    for k in range(N):
        ax.plot(lo[k, :].flatten(), la[k, :].flatten(), color = C, linewidth = .5, linestyle = '--', transform = ccrs.Geodetic())
        ax.plot(lo[:, k].flatten(), la[:, k].flatten(), color = C, linewidth = .5, linestyle = '--', transform = ccrs.Geodetic())

r, theta, phi = p.cube2spherical(xi_, eta_, block = 0)
lo, la = np.rad2deg(phi), 90 - np.rad2deg(theta)
for k in range(0, N + 2*N_extra):
        ax.plot(lo[k, :].flatten(), la[k, :].flatten(), color = 'C0', linewidth = 1, linestyle = '-', transform = ccrs.Geodetic())
        ax.plot(lo[:, k].flatten(), la[:, k].flatten(), color = 'C0', linewidth = 1, linestyle = '-', transform = ccrs.Geodetic())



plt.show()



