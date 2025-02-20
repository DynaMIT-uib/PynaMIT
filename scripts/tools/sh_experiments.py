"""Script for testing various things with spherical harmonics."""

import numpy as np
import pynamit
import pyhwm2014  # https://github.com/rilma/pyHWM14
from lompe import conductance
import datetime
import ppigrf

Nmax, Mmax = 20, 20
Ncs = 30
RI = (6371.2 + 110) * 1e3

Kp = 3
ckeys = pynamit.sha.helpers.SHKeys(Nmax, Mmax).MleN()
skeys = pynamit.sha.helpers.SHKeys(Nmax, Mmax).MleN().Mge(1)
ubasis = pynamit.sha.sh_basis.SHBasis(Nmax, Mmax)


# Fit a wind pattern using a spherical harmonic Helmholtz representation
date = datetime.datetime(2000, 5, 12, 21, 45)
# Since IGRF coefficients are used without interpolation...
# (fix if used by something different)
assert date.year % 5 == 0

### CONDUCTANCE EXPERIMENT
cbasis = pynamit.sha.sh_basis.SHBasis(Nmax, Mmax, Nmin=0)

cs_basis = pynamit.CSBasis(Ncs)
conductance_lat = 90 - cs_basis.arr_theta
conductance_lon = cs_basis.arr_phi
hall, pedersen = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=False
)

etaH, etaP = hall / (hall**2 + pedersen**2), pedersen / (hall**2 + pedersen**2)

G = cbasis.get_G(pynamit.Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi))
d = etaH

m_plain = np.linalg.lstsq(G, d, rcond=0)[0]

GTG = G.T.dot(G)
R = np.eye(cbasis.index_length) * cbasis.n * (cbasis.n + 1)
alpha = 1
GTd = G.T.dot(d)
m_regularized = np.linalg.lstsq(GTG + alpha * R, GTd, rcond=0)[0]

# alpha = np.logspace(-5, 5, 11)
# ms = []
# for a in alpha:
#    ms.append(np.linalg.lstsq(GTG + a * R, GTd, rcond = 0)[0])
#
# misfits = [np.sqrt(np.sum((G.dot(m) - d)**2)) for m in ms]
# norms = [np.linalg.norm(m) for m in ms]


if False:
    hwm14Obj = pyhwm2014.HWM142D(
        alt=110.0,
        ap=[35, 35],
        glatlim=[-89.0, 88.0],
        glatstp=3.0,
        glonlim=[-180.0, 180.0],
        glonstp=8.0,
        option=6,
        verbose=False,
        ut=date.hour,
        day=date.timetuple().tm_yday,
    )
    u_phi = hwm14Obj.Uwind
    u_theta = -hwm14Obj.Vwind
    u_lat, u_lon = np.meshgrid(
        hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij"
    )

    ugrid = pynamit.Grid(lat=u_lat.flatten(), lon=u_lon.flatten())

    Gphi, Gtheta = ubasis.get_G(ugrid, derivative="phi"), ubasis.get_G(
        ugrid, derivative="theta"
    )
    G_df = np.vstack((-Gphi, Gtheta))  # u_df = r x grad()
    G_cf = np.vstack((Gtheta, Gphi))  # u_cf = grad()

    d = np.hstack((u_theta.flatten(), u_phi.flatten()))
    u_coeffs = np.linalg.lstsq(np.hstack((G_df, G_cf)), d, rcond=0)[0]
    u_coeff_df, u_coeff_cf = np.split(u_coeffs, 2)

    misfit = G_df.dot(u_coeff_df) + G_cf.dot(u_coeff_cf) - d
    print(
        "rms misfit for fitted wind field is {:.5f} m/s".format(
            np.sqrt(np.mean(misfit**2))
        )
    )

    # get IGRF gauss coefficients:
    igrf_date = datetime.datetime(date.year, 1, 1)
    g, h = ppigrf.ppigrf.read_shc()
    _n, _m = np.array([k for k in g.columns]).T  # n and m
    g, h = (
        g.loc[igrf_date, :].values,
        h.loc[igrf_date, :].values,
    )  # gauss coefficients

    igrf_basis = pynamit.sha.sh_basis.SHBasis(_n.max(), _m.max())

    igrf_keys = (
        pynamit.sha.helpers.SHKeys(_n.max(), _m.max()).setNmin(1).MleN()
    )

    # Calculate u x B numerically on grid (we evaluate on the ground):
    ph = np.deg2rad(u_lon).reshape((-1, 1))
    P, dP = np.split(
        igrf_basis.legendre(_n.max(), _m.max(), 90 - u_lat, keys=igrf_keys),
        2,
        axis=1,
    )
    G_Br = np.hstack(
        (
            (igrf_keys.n + 1) * P * np.cos(igrf_keys.m * ph),
            (igrf_keys.n + 1) * P * np.sin(igrf_keys.m * ph),
        )
    )
    Br = G_Br.dot(np.hstack((g, h))) * 1e-9

    uxB_theta = Br * u_phi.flatten()
    uxB_phi = -Br * u_theta.flatten()


# Now u x B is calculated from SH coefficients without grid evaluation
