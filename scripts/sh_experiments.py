""" script for testing various things with spherical harmonics """

import numpy as np
import pynamit
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import ppigrf

Nmax, Mmax = 20, 20
RI = (6371.2 + 110) * 1e3

ckeys = pynamit.sha.helpers.SHKeys(Nmax, Mmax).MleN()
skeys = pynamit.sha.helpers.SHKeys(Nmax, Mmax).MleN().Mge(1)
ubasis = pynamit.sha.sh_basis.SHBasis(Nmax, Mmax)


# fit a wind pattern using a Helmholtz representation expanded in spherical harmonics:
date = datetime.datetime(2000, 5, 12, 21, 45)
assert date.year % 5 == 0  # since I'm grabbing IGRF coefficients without interpolation... (fix this if used by something different)
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour, day = date.timetuple().tm_yday)
u_phi   =  hwm14Obj.Uwind
u_theta = -hwm14Obj.Vwind
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

ugrid = pynamit.grid.Grid(u_lat.flatten(), u_lon.flatten())

Gphi, Gtheta = ubasis.get_G(ugrid, derivative = 'phi'), ubasis.get_G(ugrid, derivative = 'theta')
G_df = np.vstack((-Gphi, Gtheta)) # u_df = r x grad()
G_cf = np.vstack((Gtheta, Gphi))  # u_cf = grad()

d = np.hstack((u_theta.flatten(), u_phi.flatten()))
u_coeffs = np.linalg.lstsq(np.hstack((G_df, G_cf)), d, rcond = 0)[0]
u_coeff_df, u_coeff_cf = np.split(u_coeffs, 2)

misfit = G_df.dot(u_coeff_df) + G_cf.dot(u_coeff_cf) - d
print('rms misfit for fitted wind field is {:.5f} m/s'.format(np.sqrt(np.mean(misfit**2))))

# get IGRF gauss coefficients:
igrf_date = datetime.datetime(date.year, 1, 1)
g , h  = ppigrf.ppigrf.read_shc()
_n, _m = np.array([k for k in g.columns]).T # n and m
g , h  = g.loc[igrf_date, :].values, h.loc[igrf_date, :].values # gauss coefficients

#igrf_basis = pynamit.sha.sh_basis.SHBasis(_n.max(), _m.max())

igrf_keys = pynamit.sha.helpers.SHKeys(_n.max(), _m.max()).setNmin(1).MleN()

# Calculate u x B numerically on grid (we just evaluate on the ground...):
ph = np.deg2rad(u_lon).reshape((-1, 1))
P, dP  = np.split(pynamit.sha.helpers.legendre(_n.max(), _m.max(), 90 - u_lat, keys = igrf_keys), 2, axis = 1)
GBr = np.hstack(((igrf_keys.n + 1) * P * np.cos(igrf_keys.m * ph), (igrf_keys.n + 1) * P * np.sin(igrf_keys.m * ph)))
Br = GBr.dot(np.hstack((g, h))) * 1e-9

uxB_theta =  Br * u_phi
uxB_phi   = -Br * u_theta


# now we calculate u x B directly from spherical harmonic coefficients without going through grid evaluation:








